from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import mlflow
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json

try:
    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NHITS, PatchTST

    NEURAL_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - surfaced by /ready and the job
    torch = None
    NeuralForecast = None
    MAE = None
    NHITS = None
    PatchTST = None
    NEURAL_IMPORT_ERROR = str(exc)


MODEL_VERSION = "b58b1-native-wave-sequence-challengers-v1"
SOURCE_NAME = "b58b1_wave_sequence_challengers"
DATASET_NAME = "maritime_wave_native_sequence_challengers"
SOURCE_BUCKET = "gold-maritime"
SOURCE_KEY = (
    "datasets/b58a/version=1/"
    "maritime_weather_hourly_past_only_v1.parquet"
)
B58B_PREDICTIONS_KEY = (
    "predictions/b58b/version=1/selected_point_predictions.parquet"
)
B58B_DECISION_KEY = "configs/b58b/version=1/14_b58b_decision.json"

HORIZONS_H = (6, 12, 24, 48, 72)
LATENCY_H = 3
MAX_LEAD_H = max(HORIZONS_H) + LATENCY_H
PURGE_H = 72
STEP_SIZE_H = 24
INTERNAL_VALIDATION_H = 24 * 180
MIN_REPLACEMENT_GAIN_PCT = 5.0
RANDOM_SEED = 20260724
ALIASES = ("NHITS_SEQ", "PATCHTST_SEQ")
TARGETS = ("wave_height_m", "wave_period_s", "wave_direction_deg")
SERIES_IDS = (
    "HEIGHT",
    "PERIOD",
    "DIRECTION_SIN",
    "DIRECTION_COS",
)


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _query_frame(
    query: str, parameters: tuple[Any, ...] | None = None
) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _read_s3_bytes(client, key: str) -> bytes:
    return client.get_object(Bucket=SOURCE_BUCKET, Key=key)["Body"].read()


def _load_inputs(client) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source = pd.read_parquet(io.BytesIO(_read_s3_bytes(client, SOURCE_KEY)))
    champion = pd.read_parquet(
        io.BytesIO(_read_s3_bytes(client, B58B_PREDICTIONS_KEY))
    )
    decision = json.loads(
        _read_s3_bytes(client, B58B_DECISION_KEY).decode("utf-8")
    )
    if decision.get("decision") != "READY_FOR_IBI_HYBRID_ENRICHMENT":
        raise RuntimeError(
            f"B58B champion contract is not ready: {decision.get('decision')}"
        )
    if int(decision.get("critical_leakage_violations", -1)) != 0:
        raise RuntimeError("B58B champion contains leakage violations")

    source["observed_at"] = pd.to_datetime(
        source["observed_at"], errors="coerce", utc=True
    )
    source = source.sort_values("observed_at").reset_index(drop=True)
    required = {
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise RuntimeError(f"B58A source columns missing: {missing}")
    if source["observed_at"].isna().any():
        raise RuntimeError("Invalid source timestamps")
    if source["observed_at"].duplicated().any():
        raise RuntimeError("Source is not one row per hour")
    expected = pd.date_range(
        source["observed_at"].min(),
        source["observed_at"].max(),
        freq="h",
        tz="UTC",
    )
    if len(expected) != len(source):
        raise RuntimeError("Source hourly grid is incomplete")
    if source[list(required)].isna().any().any():
        raise RuntimeError("Wave source contains missing target values")

    for column in ("issue_at", "target_at"):
        champion[column] = pd.to_datetime(
            champion[column], errors="coerce", utc=True
        )
    return source, champion, decision


def _source_signature(
    source: pd.DataFrame, b58b_decision: dict[str, Any]
) -> str:
    digest = hashlib.sha256(MODEL_VERSION.encode("ascii"))
    digest.update(
        json.dumps(
            b58b_decision.get("selected_models", {}),
            sort_keys=True,
        ).encode("utf-8")
    )
    hashed = pd.util.hash_pandas_object(
        source[
            [
                "observed_at",
                "wave_height_m",
                "wave_period_s",
                "wave_direction_deg",
            ]
        ],
        index=False,
    )
    digest.update(hashed.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _start_run(checksum: str) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "training_executed": False,
        "selection_used_test": False,
        "b58b_modified": False,
        "production_promotion_allowed": False,
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit.ingestion_run
                    (source_name, dataset_name, object_uri, checksum, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (
                    SOURCE_NAME,
                    DATASET_NAME,
                    f"s3://{SOURCE_BUCKET}/{SOURCE_KEY}",
                    checksum,
                    Json(metadata),
                ),
            )
            return str(cursor.fetchone()[0])


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    frame = _query_frame(
        """
        SELECT run_id::text, metadata
        FROM audit.ingestion_run
        WHERE source_name=%s
          AND dataset_name=%s
          AND checksum=%s
          AND status='SUCCESS'
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (SOURCE_NAME, DATASET_NAME, checksum),
    )
    if frame.empty:
        return None
    metadata = frame.iloc[0]["metadata"]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return str(frame.iloc[0]["run_id"]), metadata


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at=now(), status=%s, row_count=%s,
                    metadata=metadata || %s, error_message=%s
                WHERE run_id=%s
                """,
                (
                    status,
                    row_count,
                    Json(
                        _clean_json(metadata),
                        dumps=lambda value: json.dumps(
                            value, default=_json_default, allow_nan=False
                        ),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    payload = {
        "progress": _clean_json(
            {
                "stage": stage,
                "updated_at": pd.Timestamp.now(tz="UTC"),
                **details,
            }
        )
    }
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET metadata=metadata || %s
                WHERE run_id=%s
                """,
                (
                    Json(
                        payload,
                        dumps=lambda value: json.dumps(
                            value, allow_nan=False
                        ),
                    ),
                    run_id,
                ),
            )


def _upload_file(client, path: Path, bucket: str, key: str) -> str:
    client.upload_file(str(path), bucket, key)
    return f"s3://{bucket}/{key}"


def _calendar(times: pd.Series) -> pd.DataFrame:
    hour = times.dt.hour.to_numpy(dtype="float64")
    day = times.dt.dayofyear.to_numpy(dtype="float64")
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2.0 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * hour / 24.0),
            "day_sin": np.sin(2.0 * np.pi * day / 366.0),
            "day_cos": np.cos(2.0 * np.pi * day / 366.0),
        }
    )


def _build_panel(source: pd.DataFrame) -> pd.DataFrame:
    times = source["observed_at"].dt.tz_localize(None)
    calendar = _calendar(times)
    radians = np.deg2rad(
        pd.to_numeric(source["wave_direction_deg"], errors="coerce")
    )
    series = {
        "HEIGHT": pd.to_numeric(
            source["wave_height_m"], errors="coerce"
        ).to_numpy(),
        "PERIOD": pd.to_numeric(
            source["wave_period_s"], errors="coerce"
        ).to_numpy(),
        "DIRECTION_SIN": np.sin(radians),
        "DIRECTION_COS": np.cos(radians),
    }
    blocks = []
    for unique_id, values in series.items():
        block = pd.DataFrame(
            {
                "unique_id": unique_id,
                "ds": times,
                "y": values,
            }
        )
        for column in calendar.columns:
            block[column] = calendar[column].to_numpy()
        blocks.append(block)
    panel = pd.concat(blocks, ignore_index=True)
    if panel["y"].isna().any():
        raise RuntimeError("Sequence panel contains missing targets")
    return panel


def _build_models() -> list[Any]:
    if NeuralForecast is None:
        raise RuntimeError(
            "NeuralForecast is unavailable: "
            f"{NEURAL_IMPORT_ERROR or 'unknown import error'}"
        )
    common = {
        "h": MAX_LEAD_H,
        "loss": MAE(),
        "valid_loss": MAE(),
        "max_steps": 350,
        "learning_rate": 0.001,
        "early_stop_patience_steps": 5,
        "val_check_steps": 50,
        "batch_size": 4,
        "valid_batch_size": 4,
        "windows_batch_size": 256,
        "inference_windows_batch_size": 128,
        "scaler_type": "robust",
        "random_seed": RANDOM_SEED,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "logger": False,
        "accelerator": "cpu",
        "devices": 1,
    }
    return [
        NHITS(
            input_size=24 * 14,
            n_blocks=[1, 1, 1],
            mlp_units=[[256, 256], [256, 256], [256, 256]],
            n_pool_kernel_size=[24, 6, 1],
            n_freq_downsample=[24, 6, 1],
            dropout_prob_theta=0.10,
            futr_exog_list=[
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
            ],
            alias="NHITS_SEQ",
            **common,
        ),
        PatchTST(
            input_size=24 * 30,
            encoder_layers=3,
            n_heads=4,
            hidden_size=64,
            linear_hidden_size=128,
            dropout=0.10,
            fc_dropout=0.10,
            attn_dropout=0.05,
            patch_len=24,
            stride=12,
            revin=True,
            alias="PATCHTST_SEQ",
            **common,
        ),
    ]


def _split_boundaries(
    source: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp, int, int]:
    train_index = int(len(source) * 0.70)
    valid_index = int(len(source) * 0.85)
    return (
        source.loc[train_index, "observed_at"],
        source.loc[valid_index, "observed_at"],
        train_index,
        valid_index,
    )


def _train_and_cross_validate(
    panel: pd.DataFrame,
    source_rows: int,
    train_index: int,
) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    models = _build_models()
    raw_backtest_size = source_rows - (train_index - PURGE_H)
    backtest_windows = math.ceil(
        (raw_backtest_size - MAX_LEAD_H) / STEP_SIZE_H
    )
    backtest_size = MAX_LEAD_H + backtest_windows * STEP_SIZE_H
    fit_history_size = source_rows - backtest_size
    optimization_train_size = fit_history_size - INTERNAL_VALIDATION_H
    if optimization_train_size < 24 * 365 * 3:
        raise RuntimeError("Insufficient sequence training history")
    if (backtest_size - MAX_LEAD_H) % STEP_SIZE_H:
        raise RuntimeError("Sequence backtest is not aligned to step_size")
    started = time.perf_counter()
    forecast = NeuralForecast(models=models, freq="h")
    cv = forecast.cross_validation(
        df=panel,
        val_size=INTERNAL_VALIDATION_H,
        test_size=backtest_size,
        n_windows=None,
        step_size=STEP_SIZE_H,
        refit=False,
        verbose=0,
    )
    elapsed = time.perf_counter() - started
    inventory = []
    for model in forecast.models:
        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        inventory.append(
            {
                "model": str(model.alias or type(model).__name__),
                "trainable_parameters": int(parameters),
                "input_size_h": int(model.input_size),
                "forecast_lead_h": MAX_LEAD_H,
                "max_steps": int(model.max_steps),
                "future_exogenous_variables": bool(
                    getattr(model, "futr_exog_list", [])
                ),
                "device": "CPU",
            }
        )
    resources = {
        "training_elapsed_seconds": elapsed,
        "torch_version": getattr(torch, "__version__", None),
        "neuralforecast_version": __import__(
            "neuralforecast"
        ).__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "optimization_train_hours": optimization_train_size,
        "internal_validation_hours": INTERNAL_VALIDATION_H,
        "fit_history_hours": fit_history_size,
        "backtest_hours": backtest_size,
        "backtest_windows": backtest_windows,
        "cv_step_size_h": STEP_SIZE_H,
        "models": inventory,
    }
    return forecast, cv, resources


def _direction_from_components(
    sine: np.ndarray, cosine: np.ndarray
) -> np.ndarray:
    norm = np.sqrt(np.square(sine) + np.square(cosine))
    norm = np.where(norm < 1e-8, 1.0, norm)
    return np.mod(
        np.rad2deg(np.arctan2(sine / norm, cosine / norm)), 360.0
    )


def _prepare_sequence_predictions(
    cv: pd.DataFrame,
    train_boundary: pd.Timestamp,
    valid_boundary: pd.Timestamp,
) -> pd.DataFrame:
    cv = cv.copy()
    cv["ds"] = pd.to_datetime(cv["ds"], errors="coerce")
    cv["cutoff"] = pd.to_datetime(cv["cutoff"], errors="coerce")
    values = ["y", *ALIASES]
    pivot = cv.pivot_table(
        index=["cutoff", "ds"],
        columns="unique_id",
        values=values,
        aggfunc="first",
    )
    pivot.columns = [f"{metric}__{series}" for metric, series in pivot.columns]
    pivot = pivot.reset_index()
    pivot["lead_h"] = (
        (pivot["ds"] - pivot["cutoff"]).dt.total_seconds() / 3600.0
    ).round().astype("int16")
    wanted_leads = {horizon + LATENCY_H: horizon for horizon in HORIZONS_H}
    pivot = pivot.loc[pivot["lead_h"].isin(wanted_leads)].copy()
    pivot["horizon_h"] = pivot["lead_h"].map(wanted_leads).astype("int16")
    pivot["issue_at"] = (
        pivot["cutoff"].dt.tz_localize("UTC")
        + pd.Timedelta(hours=LATENCY_H)
    )
    pivot["target_at"] = pivot["ds"].dt.tz_localize("UTC")
    pivot["split"] = "EXCLUDED_PURGE"
    valid_mask = (
        pivot["issue_at"]
        >= train_boundary + pd.Timedelta(hours=PURGE_H)
    ) & (pivot["target_at"] < valid_boundary)
    test_mask = (
        pivot["issue_at"]
        >= valid_boundary + pd.Timedelta(hours=PURGE_H)
    )
    pivot.loc[valid_mask, "split"] = "VALID"
    pivot.loc[test_mask, "split"] = "TEST"
    pivot = pivot.loc[pivot["split"].isin(["VALID", "TEST"])].copy()

    pivot["target_wave_height_m"] = pivot["y__HEIGHT"]
    pivot["target_wave_period_s"] = pivot["y__PERIOD"]
    pivot["target_wave_direction_deg"] = _direction_from_components(
        pivot["y__DIRECTION_SIN"].to_numpy(dtype="float64"),
        pivot["y__DIRECTION_COS"].to_numpy(dtype="float64"),
    )
    for alias in ALIASES:
        pivot[f"pred__{alias}__wave_height_m"] = pivot[f"{alias}__HEIGHT"]
        pivot[f"pred__{alias}__wave_period_s"] = pivot[f"{alias}__PERIOD"]
        pivot[f"pred__{alias}__wave_direction_deg"] = (
            _direction_from_components(
                pivot[f"{alias}__DIRECTION_SIN"].to_numpy(dtype="float64"),
                pivot[f"{alias}__DIRECTION_COS"].to_numpy(dtype="float64"),
            )
        )
    keep = [
        "issue_at",
        "target_at",
        "horizon_h",
        "split",
        *[f"target_{target}" for target in TARGETS],
        *[
            f"pred__{alias}__{target}"
            for alias in ALIASES
            for target in TARGETS
        ],
    ]
    return pivot[keep].sort_values(
        ["issue_at", "horizon_h"]
    ).reset_index(drop=True)


def _join_b58b(
    sequence: pd.DataFrame, champion: pd.DataFrame
) -> pd.DataFrame:
    champion_columns = [
        "issue_at",
        "target_at",
        "horizon_h",
        *[f"target_{target}" for target in TARGETS],
        *[f"selected_pred_{target}" for target in TARGETS],
        *[f"selected_model_{target}" for target in TARGETS],
    ]
    renamed = champion[champion_columns].rename(
        columns={
            f"selected_pred_{target}": f"pred__B58B_CHAMPION__{target}"
            for target in TARGETS
        }
    )
    joined = sequence.merge(
        renamed,
        on=["issue_at", "target_at", "horizon_h"],
        how="inner",
        suffixes=("", "__b58b"),
        validate="one_to_one",
    )
    if len(joined) < int(len(sequence) * 0.98):
        raise RuntimeError(
            "Sequence/B58B comparison join lost more than 2% of rows"
        )
    for target in TARGETS:
        difference = np.abs(
            joined[f"target_{target}"]
            - joined[f"target_{target}__b58b"]
        )
        if target == "wave_direction_deg":
            difference = np.abs(
                (
                    joined[f"target_{target}"]
                    - joined[f"target_{target}__b58b"]
                    + 180.0
                )
                % 360.0
                - 180.0
            )
        if float(difference.max()) > 1e-5:
            raise RuntimeError(f"Target mismatch after B58B join: {target}")
        joined = joined.drop(columns=[f"target_{target}__b58b"])
    return joined


def _angular_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return (predicted - actual + 180.0) % 360.0 - 180.0


def _metric(
    actual: np.ndarray, predicted: np.ndarray, target: str
) -> dict[str, Any]:
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    if not len(actual):
        return {"n": 0, "MAE": None, "RMSE": None, "BIAS": None}
    error = (
        _angular_error(actual, predicted)
        if target == "wave_direction_deg"
        else predicted - actual
    )
    return {
        "n": len(actual),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "BIAS": float(np.mean(error)),
    }


def _metrics(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    rows = []
    subset = frame.loc[frame["split"].eq(split)]
    for horizon_h in HORIZONS_H:
        horizon = subset.loc[subset["horizon_h"].eq(horizon_h)]
        for target in TARGETS:
            actual = horizon[f"target_{target}"].to_numpy(dtype="float64")
            for model in ("B58B_CHAMPION", *ALIASES):
                predicted = horizon[
                    f"pred__{model}__{target}"
                ].to_numpy(dtype="float64")
                rows.append(
                    {
                        "split": split,
                        "horizon_h": horizon_h,
                        "target": target,
                        "model": model,
                        **_metric(actual, predicted, target),
                    }
                )
    return pd.DataFrame(rows)


def _select(validation: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (horizon_h, target), group in validation.groupby(
        ["horizon_h", "target"], sort=True
    ):
        champion = group.loc[group["model"].eq("B58B_CHAMPION")].iloc[0]
        challengers = group.loc[group["model"].isin(ALIASES)].sort_values(
            ["MAE", "model"]
        )
        best = challengers.iloc[0]
        champion_mae = float(champion["MAE"])
        challenger_mae = float(best["MAE"])
        gain = 100.0 * (champion_mae - challenger_mae) / champion_mae
        accepted = bool(gain >= MIN_REPLACEMENT_GAIN_PCT)
        rows.append(
            {
                "horizon_h": int(horizon_h),
                "target": target,
                "b58b_validation_mae": champion_mae,
                "best_sequence_model": best["model"],
                "best_sequence_validation_mae": challenger_mae,
                "sequence_gain_pct": gain,
                "replacement_threshold_pct": MIN_REPLACEMENT_GAIN_PCT,
                "sequence_accepted": accepted,
                "selected_model": (
                    best["model"] if accepted else "B58B_CHAMPION"
                ),
                "selection_split": "VALID",
                "test_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def _apply_selection(
    frame: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    result = frame.copy()
    for target in TARGETS:
        result[f"selected_pred_{target}"] = np.nan
        result[f"selected_model_{target}"] = ""
    for row in selection.itertuples(index=False):
        mask = result["horizon_h"].eq(row.horizon_h)
        result.loc[mask, f"selected_pred_{row.target}"] = result.loc[
            mask, f"pred__{row.selected_model}__{row.target}"
        ]
        result.loc[mask, f"selected_model_{row.target}"] = row.selected_model
    return result


def _selected_test_metrics(
    frame: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    test = frame.loc[frame["split"].eq("TEST")]
    for choice in selection.itertuples(index=False):
        subset = test.loc[test["horizon_h"].eq(choice.horizon_h)]
        rows.append(
            {
                "split": "TEST",
                "horizon_h": choice.horizon_h,
                "target": choice.target,
                "selected_model": choice.selected_model,
                **_metric(
                    subset[f"target_{choice.target}"].to_numpy(
                        dtype="float64"
                    ),
                    subset[f"selected_pred_{choice.target}"].to_numpy(
                        dtype="float64"
                    ),
                    choice.target,
                ),
                "test_role": "FINAL_DIAGNOSTIC_ONLY",
            }
        )
    return pd.DataFrame(rows)


def _quarterly_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    working = frame.copy()
    working["quarter"] = working["issue_at"].dt.to_period("Q").astype(str)
    for (split, quarter, horizon_h), group in working.groupby(
        ["split", "quarter", "horizon_h"], sort=True
    ):
        for target in TARGETS:
            actual = group[f"target_{target}"].to_numpy(dtype="float64")
            for model in ("B58B_CHAMPION", *ALIASES):
                rows.append(
                    {
                        "split": split,
                        "quarter": quarter,
                        "horizon_h": horizon_h,
                        "target": target,
                        "model": model,
                        **_metric(
                            actual,
                            group[f"pred__{model}__{target}"].to_numpy(
                                dtype="float64"
                            ),
                            target,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _calibration(frame: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = frame.loc[frame["split"].eq("VALID")]
    test = frame.loc[frame["split"].eq("TEST")]
    for choice in selection.itertuples(index=False):
        valid_h = valid.loc[valid["horizon_h"].eq(choice.horizon_h)]
        test_h = test.loc[test["horizon_h"].eq(choice.horizon_h)]
        valid_actual = valid_h[f"target_{choice.target}"].to_numpy(
            dtype="float64"
        )
        valid_pred = valid_h[f"selected_pred_{choice.target}"].to_numpy(
            dtype="float64"
        )
        test_actual = test_h[f"target_{choice.target}"].to_numpy(
            dtype="float64"
        )
        test_pred = test_h[f"selected_pred_{choice.target}"].to_numpy(
            dtype="float64"
        )
        if choice.target == "wave_direction_deg":
            half_width = float(
                np.quantile(
                    np.abs(_angular_error(valid_actual, valid_pred)), 0.80
                )
            )
            covered = (
                np.abs(_angular_error(test_actual, test_pred)) <= half_width
            )
            width = 2.0 * half_width
        else:
            residual = valid_actual - valid_pred
            low_offset = float(np.quantile(residual, 0.10))
            high_offset = float(np.quantile(residual, 0.90))
            low = np.maximum(0.0, test_pred + low_offset)
            high = np.maximum(low, test_pred + high_offset)
            covered = (test_actual >= low) & (test_actual <= high)
            width = float(np.mean(high - low))
        rows.append(
            {
                "horizon_h": choice.horizon_h,
                "target": choice.target,
                "selected_model": choice.selected_model,
                "nominal_coverage_pct": 80.0,
                "test_coverage_pct": float(np.mean(covered) * 100.0),
                "mean_interval_width": width,
                "calibration_source": "VALID_ONLY",
            }
        )
    return pd.DataFrame(rows)


def _anti_leakage(
    frame: pd.DataFrame,
    selection: pd.DataFrame,
    train_boundary: pd.Timestamp,
    valid_boundary: pd.Timestamp,
) -> pd.DataFrame:
    valid = frame.loc[frame["split"].eq("VALID")]
    test = frame.loc[frame["split"].eq("TEST")]
    checks = [
        (
            "SOURCE_CUTOFF_PRECEDES_ISSUE_BY_3H",
            True,
            "CRITICAL",
            "Neural forecast lead = requested horizon + 3h latency",
        ),
        (
            "VALID_START_AFTER_TRAIN_PURGE",
            bool(
                valid["issue_at"].min()
                >= train_boundary + pd.Timedelta(hours=PURGE_H)
            ),
            "CRITICAL",
            str(valid["issue_at"].min()),
        ),
        (
            "TEST_START_AFTER_VALID_PURGE",
            bool(
                test["issue_at"].min()
                >= valid_boundary + pd.Timedelta(hours=PURGE_H)
            ),
            "CRITICAL",
            str(test["issue_at"].min()),
        ),
        (
            "VALID_TARGET_BEFORE_TEST_ISSUE",
            bool(valid["target_at"].max() < test["issue_at"].min()),
            "CRITICAL",
            (
                f"{valid['target_at'].max()} < {test['issue_at'].min()}"
            ),
        ),
        (
            "SELECTION_USES_VALID_ONLY",
            bool(
                selection["selection_split"].eq("VALID").all()
                and ~selection["test_used_for_selection"].any()
            ),
            "CRITICAL",
            "TEST is final diagnostic only",
        ),
        (
            "HISTORICAL_AVAILABLE_AT_PRESENT",
            False,
            "WARNING",
            "Formal production replay remains blocked",
        ),
    ]
    return pd.DataFrame(
        [
            {
                "check": name,
                "passed": passed,
                "severity": severity,
                "details": details,
                "violations": 0 if passed or severity == "WARNING" else 1,
            }
            for name, passed, severity, details in checks
        ]
    )


def _log_mlflow(
    output_dir: Path,
    decision: dict[str, Any],
    selected_test: pd.DataFrame,
) -> str:
    try:
        mlflow.set_tracking_uri(
            os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        )
        mlflow.set_experiment("smart-port-wave-sequence-challengers")
        with mlflow.start_run(run_name=MODEL_VERSION):
            mlflow.log_params(
                {
                    "model_version": MODEL_VERSION,
                    "models": ",".join(ALIASES),
                    "max_lead_h": MAX_LEAD_H,
                    "latency_h": LATENCY_H,
                    "purge_h": PURGE_H,
                    "cv_step_h": STEP_SIZE_H,
                    "replacement_gain_pct": MIN_REPLACEMENT_GAIN_PCT,
                    "selection_split": "VALID",
                }
            )
            for row in selected_test.itertuples(index=False):
                target = row.target.replace("wave_", "").replace("_", "-")
                mlflow.log_metric(
                    f"test_mae_{target}_{int(row.horizon_h)}h",
                    float(row.MAE),
                )
            mlflow.set_tag("decision", decision["status"])
            mlflow.log_artifacts(str(output_dir), artifact_path="b58b1")
        return "LOGGED"
    except Exception as exc:
        return f"FAILED: {exc}"


def _write_readme(path: Path, decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# B58B.1 Native Wave Sequence Challengers",
                "",
                f"Decision: **{decision['status']}**",
                "",
                "This block compares official NeuralForecast implementations of "
                "N-HiTS and PatchTST against the frozen B58B HGB/CatBoost champion.",
                "",
                "## Fairness protocol",
                "",
                "- Same B58A hourly source and B58B temporal boundaries.",
                "- 72-hour purge and a simulated 3-hour source latency.",
                "- Predictions are evaluated every 24 hours to bound CPU cost.",
                "- Selection uses VALID only; TEST is final diagnostic only.",
                "- A neural challenger replaces B58B only with at least 5% gain.",
                "- Direction is represented as sine/cosine and scored circularly.",
                "",
                "## Guardrails",
                "",
                "- B58B artifacts are read-only and remain the default champion.",
                "- Bronze and Core are not modified.",
                "- No production promotion is allowed without real available_at.",
                "",
                "## Next block",
                "",
                decision["next_block"],
            ]
        ),
        encoding="utf-8",
    )


def run_b58b1_wave_sequence_challengers(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    source, champion, b58b_decision = _load_inputs(client)
    checksum = _source_signature(source, b58b_decision)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "results": previous[1],
        }

    run_id = _start_run(checksum)
    try:
        _update_progress(run_id, "BUILDING_SEQUENCE_PANEL")
        panel = _build_panel(source)
        train_boundary, valid_boundary, train_index, valid_index = (
            _split_boundaries(source)
        )
        _update_progress(
            run_id,
            "TRAINING_NHITS_AND_PATCHTST",
            panel_rows=len(panel),
            max_steps=350,
            device="CPU",
        )
        forecast, cv, resources = _train_and_cross_validate(
            panel, len(source), train_index
        )
        _update_progress(
            run_id,
            "COMPARING_WITH_FROZEN_B58B",
            cross_validation_rows=len(cv),
        )
        sequence = _prepare_sequence_predictions(
            cv, train_boundary, valid_boundary
        )
        compared = _join_b58b(sequence, champion)
        validation_metrics = _metrics(compared, "VALID")
        test_metrics = _metrics(compared, "TEST")
        selection = _select(validation_metrics)
        selected = _apply_selection(compared, selection)
        selected_test = _selected_test_metrics(selected, selection)
        stability = _quarterly_stability(compared)
        calibration = _calibration(selected, selection)
        leakage = _anti_leakage(
            compared, selection, train_boundary, valid_boundary
        )
        critical_violations = int(
            leakage.loc[
                leakage["severity"].eq("CRITICAL") & ~leakage["passed"],
                "violations",
            ].sum()
        )
        accepted = int(selection["sequence_accepted"].sum())
        if critical_violations:
            status = "NEED_SEQUENCE_CHALLENGER_REPAIR"
            next_block = "B58B1_SEQUENCE_PROTOCOL_REPAIR"
        elif accepted:
            status = "SEQUENCE_CHALLENGER_ACCEPTED"
            next_block = "B58C_IBI_PLUS_LOCAL_CHAMPION_HYBRID"
        else:
            status = "KEEP_B58B_CHAMPION"
            next_block = "B58C_IBI_PLUS_B58B_HYBRID"

        decision = {
            "status": status,
            "decision": status,
            "model_version": MODEL_VERSION,
            "source_rows": len(source),
            "panel_rows": len(panel),
            "comparison_rows": len(compared),
            "validation_rows": int(compared["split"].eq("VALID").sum()),
            "test_rows": int(compared["split"].eq("TEST").sum()),
            "horizons_h": list(HORIZONS_H),
            "latency_h": LATENCY_H,
            "purge_h": PURGE_H,
            "cv_step_size_h": STEP_SIZE_H,
            "sequence_acceptances": accepted,
            "selected_models": {
                f"{int(row.horizon_h)}h:{row.target}": row.selected_model
                for row in selection.itertuples(index=False)
            },
            "critical_leakage_violations": critical_violations,
            "training_executed": True,
            "selection_split": "VALID",
            "test_role": "FINAL_DIAGNOSTIC_ONLY",
            "selection_used_test": False,
            "b58b_modified": False,
            "bronze_modified": False,
            "core_modified": False,
            "historical_replay_allowed": False,
            "production_promotion_allowed": False,
            "resources": resources,
            "next_block": next_block,
        }

        reports = {
            "01_source_and_champion_contract.csv": pd.DataFrame(
                [
                    {
                        "source_key": SOURCE_KEY,
                        "b58b_predictions_key": B58B_PREDICTIONS_KEY,
                        "source_rows": len(source),
                        "first_observed_at": source["observed_at"].min(),
                        "last_observed_at": source["observed_at"].max(),
                        "b58b_decision": b58b_decision.get("decision"),
                        "b58b_read_only": True,
                    }
                ]
            ),
            "02_temporal_protocol.csv": pd.DataFrame(
                [
                    {
                        "train_boundary": train_boundary,
                        "valid_boundary": valid_boundary,
                        "purge_h": PURGE_H,
                        "latency_h": LATENCY_H,
                        "maximum_model_lead_h": MAX_LEAD_H,
                        "cv_step_size_h": STEP_SIZE_H,
                        "selection_split": "VALID",
                        "test_role": "FINAL_DIAGNOSTIC_ONLY",
                    }
                ]
            ),
            "03_training_inventory.csv": pd.DataFrame(resources["models"]),
            "04_validation_comparison.csv": validation_metrics,
            "05_model_selection.csv": selection,
            "06_test_comparison.csv": test_metrics,
            "07_selected_test_metrics.csv": selected_test,
            "08_quarterly_stability.csv": stability,
            "09_probabilistic_calibration.csv": calibration,
            "10_resource_usage.csv": pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in resources.items()
                        if key != "models"
                    }
                ]
            ),
            "11_anti_leakage.csv": leakage,
        }

        with tempfile.TemporaryDirectory(prefix="b58b1-") as temporary:
            output_dir = Path(temporary)
            for name, report in reports.items():
                report.to_csv(output_dir / name, index=False)
            selected.to_parquet(
                output_dir / "sequence_comparison_predictions.parquet",
                index=False,
            )
            decision_path = output_dir / "12_b58b1_decision.json"
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            _write_readme(output_dir / "README_B58B1.md", decision)
            checkpoint_dir = output_dir / "neuralforecast_checkpoints"
            forecast.save(
                path=str(checkpoint_dir),
                overwrite=True,
                save_dataset=True,
            )
            mlflow_status = _log_mlflow(
                output_dir, decision, selected_test
            )
            decision["mlflow_status"] = mlflow_status
            decision_path.write_text(
                json.dumps(
                    _clean_json(decision),
                    indent=2,
                    ensure_ascii=True,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )

            uploaded = {}
            for path in sorted(output_dir.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(output_dir).as_posix()
                if relative.startswith("neuralforecast_checkpoints/"):
                    key = f"models/b58b1/{output_prefix}/{relative}"
                elif path.suffix == ".parquet":
                    key = f"predictions/b58b1/{output_prefix}/{path.name}"
                elif path == decision_path:
                    key = f"configs/b58b1/{output_prefix}/{path.name}"
                else:
                    key = f"reports/b58b1/{output_prefix}/{path.name}"
                uploaded[relative] = _upload_file(
                    client, path, output_bucket, key
                )

        metadata = {
            **decision,
            "checksum": checksum,
            "outputs": uploaded,
            "output_prefix": (
                f"s3://{output_bucket}/reports/b58b1/{output_prefix}/"
            ),
        }
        _update_progress(run_id, "COMPLETED", decision=status)
        _finish_run(run_id, "SUCCESS", len(compared), metadata)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "results": metadata,
            "outputs": uploaded,
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "model_version": MODEL_VERSION,
                "neural_import_error": NEURAL_IMPORT_ERROR,
            },
            error_message=str(exc),
        )
        raise
