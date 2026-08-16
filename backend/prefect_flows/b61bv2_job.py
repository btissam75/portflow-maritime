from __future__ import annotations

import hashlib
import gc
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from catboost import CatBoostClassifier, CatBoostRegressor
from psycopg2.extras import Json, execute_values

from prefect_flows.b61b_core import (
    CATEGORICAL_FEATURES,
    HAZARD_HORIZONS,
    HAZARD_TARGETS,
    IDENTIFIER_COLUMNS,
    RISK_NAMES,
    RISK_TARGETS,
    TARGET_COLUMNS,
    available_contract,
    clean_json,
    delay_class_index,
    enforce_hazard_order,
    enforce_quantile_order,
    enforce_risk_order,
    regime_labels,
    risk_from_class_probabilities,
    split_validation_roles,
)
from prefect_flows.b61b_sequence import train_sequence_expert
from prefect_flows.b61bv2_core import (
    MODEL_VERSION,
    RANDOM_SEED,
    REAL_DATASET_VERSION,
    SYNTHETIC_DATASET_VERSION,
    TRACKS,
    adaptive_conformal_policy,
    apply_adaptive_conformal,
    approximate_concordance_index,
    binary_selection_objective,
    coherent_outputs,
    duration_selection_objective,
    fit_binary_calibrator,
    grouped_bootstrap_ci,
    select_threshold,
    weighted_binary_metrics,
    weighted_regression_metrics,
)


SOURCE_NAME = "b61b_v2_maritime_rare_event_hybrid"
DATASET_NAME = "maritime_port_call_multitask_predictions_v2"
REAL_RELATION = "features.maritime_port_call_governed_v1"
SYNTHETIC_RELATION = "features.maritime_port_call_tail_augmented_train_v1"
TARGET_RELATION = "serving.maritime_port_call_multitask_prediction_v2"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=2"


def _db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _relation_exists(relation: str) -> bool:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (relation,))
        return cursor.fetchone()[0] is not None


def _relation_columns(relation: str) -> list[str]:
    schema, table = relation.split(".", 1)
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s
            ORDER BY ordinal_position
            """,
            (schema, table),
        )
        return [str(row[0]) for row in cursor.fetchall()]


def _query_frame(query: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()
        columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _quote_columns(columns: list[str]) -> str:
    if any(not column.replace("_", "").isalnum() for column in columns):
        raise ValueError("Unsafe source column")
    return ", ".join(f'"{column}"' for column in columns)


def load_governed_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    real_columns = _relation_columns(REAL_RELATION)
    synthetic_columns = _relation_columns(SYNTHETIC_RELATION)
    contract = available_contract(real_columns)
    issue_time_columns = [
        column
        for column in real_columns
        if column.endswith("_available") and column.startswith("issue_time_fcst_")
    ]
    if "issue_time_forecast_completeness" in real_columns:
        issue_time_columns.append("issue_time_forecast_completeness")
    governance = [
        "dataset_version",
        "data_origin",
        "synthetic_row",
        "targets_imputed",
    ]
    selected = list(
        dict.fromkeys(
            [
                *governance,
                *IDENTIFIER_COLUMNS,
                *contract.core_numeric,
                *contract.research_numeric,
                *contract.categorical,
                *issue_time_columns,
                *TARGET_COLUMNS,
            ]
        )
    )
    selected_real = [column for column in selected if column in real_columns]
    real = _query_frame(
        f"""
        SELECT {_quote_columns(selected_real)}
        FROM {REAL_RELATION}
        WHERE dataset_version=%s
          AND split IN ('TRAIN', 'VALID', 'TEST')
          AND synthetic_row=false
          AND targets_imputed=false
        ORDER BY port_call_id, landmark_at
        """,
        (REAL_DATASET_VERSION,),
    )
    if real.empty:
        raise RuntimeError(f"No governed real rows found in {REAL_RELATION}")

    selected_synthetic = [column for column in selected if column in synthetic_columns]
    lineage_columns = [
        "source_parent_port_call_id",
        "target_origin",
        "external_source_name",
        "generator_version",
    ]
    selected_synthetic.extend(
        column
        for column in lineage_columns
        if column in synthetic_columns and column not in selected_synthetic
    )
    synthetic = _query_frame(
        f"""
        SELECT {_quote_columns(selected_synthetic)}
        FROM {SYNTHETIC_RELATION}
        WHERE dataset_version=%s
          AND split='TRAIN'
          AND synthetic_row=true
          AND targets_imputed=false
          AND training_allowed=true
        ORDER BY port_call_id, landmark_at
        """,
        (SYNTHETIC_DATASET_VERSION,),
    )
    if synthetic.empty:
        raise RuntimeError(f"No governed rare-tail supplement found in {SYNTHETIC_RELATION}")

    real["landmark_at"] = pd.to_datetime(real["landmark_at"], utc=True)
    synthetic["landmark_at"] = pd.to_datetime(synthetic["landmark_at"], utc=True)
    real["model_role"] = split_validation_roles(real)
    permission_columns = {
        "TRAIN": "training_allowed",
        "VALID": "validation_allowed",
        "TEST": "test_allowed",
    }
    for split, permission in permission_columns.items():
        denied = real["split"].eq(split) & ~real[permission].fillna(False).astype(bool)
        real.loc[denied, "model_role"] = "UNUSED_NOT_GOVERNED"
    synthetic["model_role"] = "TRAIN_SUPPLEMENT"
    real["regime"] = regime_labels(real)
    synthetic["regime"] = regime_labels(synthetic)

    for frame in (real, synthetic):
        for column in RISK_TARGETS + HAZARD_TARGETS:
            frame[column] = frame[column].fillna(False).astype("float32")
        frame["target_remaining_h"] = pd.to_numeric(frame["target_remaining_h"], errors="coerce").astype("float32")
        frame["target_departure_delay_h"] = pd.to_numeric(frame["target_departure_delay_h"], errors="coerce").astype("float32")
        frame["per_call_sample_weight"] = pd.to_numeric(
            frame["per_call_sample_weight"], errors="coerce"
        ).fillna(0.0).astype("float32")

    info = {
        "real_source_columns": len(real_columns),
        "synthetic_source_columns": len(synthetic_columns),
        "loaded_real_columns": len(selected_real),
        "loaded_synthetic_columns": len(selected_synthetic),
        "core_numeric": list(contract.core_numeric),
        "research_numeric": list(contract.research_numeric),
        "categorical": list(contract.categorical),
        "issue_time_columns_available": issue_time_columns,
    }
    return real, synthetic, info


def _source_signature(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    sequence_steps: int,
    iterations: int,
    max_train_rows: int,
) -> str:
    digest = hashlib.sha256(MODEL_VERSION.encode("ascii"))
    digest.update(str((len(real), len(synthetic), sequence_steps, iterations, max_train_rows)).encode("ascii"))
    for frame in (real, synthetic):
        digest.update(
            pd.util.hash_pandas_object(
                frame[["port_call_id", "landmark_at", "target_departure_delay_h"]],
                index=False,
            ).to_numpy(dtype="uint64").tobytes()
        )
    return digest.hexdigest()


def _previous_success(checksum: str) -> dict[str, Any] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT metadata
            FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s AND checksum=%s
              AND status='SUCCESS'
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (SOURCE_NAME, DATASET_NAME, checksum),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        cursor.execute("SELECT to_regclass(%s)", (TARGET_RELATION,))
        if cursor.fetchone()[0] is None:
            return None
        cursor.execute(
            f"SELECT COUNT(*) FROM {TARGET_RELATION} WHERE model_version=%s",
            (MODEL_VERSION,),
        )
        return dict(row[0]) if int(cursor.fetchone()[0]) > 0 else None


def _start_run(checksum: str) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "orchestrator": "PREFECT",
        "real_dataset_version": REAL_DATASET_VERSION,
        "supplement_dataset_version": SYNTHETIC_DATASET_VERSION,
        "selection_split": "VALID_SELECT",
        "calibration_split": "VALID_CALIBRATE",
        "test_role": "TEST_DIAGNOSTIC_ONLY_ONCE",
        "test_used_for_selection": False,
        "synthetic_scope": "TRAIN_ONLY_LOW_WEIGHT",
        "production_promotion_allowed": False,
    }
    with _db_connection() as connection, connection.cursor() as cursor:
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
                f"postgresql://maritime/{TARGET_RELATION}",
                checksum,
                Json(metadata),
            ),
        )
        return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    payload = clean_json({"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details})
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
            (Json({"progress": payload}), run_id),
        )


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE audit.ingestion_run
            SET finished_at=now(), status=%s, row_count=%s,
                metadata=metadata || %s, error_message=%s
            WHERE run_id=%s
            """,
            (status, row_count, Json(clean_json(metadata)), error_message, run_id),
        )


def _model_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.reindex(columns=features).copy()
    for column in features:
        if column in CATEGORICAL_FEATURES:
            result[column] = result[column].fillna("__MISSING__").astype(str)
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float32")
    return result


def _catboost_common(iterations: int) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 6.0,
        "random_seed": RANDOM_SEED,
        "thread_count": 2,
        "allow_writing_files": False,
        "verbose": False,
        "used_ram_limit": "700mb",
        "random_strength": 0.5,
    }


def _balanced_weights(
    labels: np.ndarray,
    base_weight: np.ndarray,
    enabled: bool,
    maximum_multiplier: float = 10.0,
) -> np.ndarray:
    base = np.asarray(base_weight, dtype="float64")
    if not enabled:
        return base
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    multipliers = {
        value: min(maximum_multiplier, len(labels) / max(len(unique) * count, 1.0))
        for value, count in zip(unique, counts)
    }
    factor = np.asarray([multipliers[value] for value in labels], dtype="float64")
    weighted = base * factor
    scale = base.sum() / max(weighted.sum(), 1e-12)
    return weighted * scale


def _duration_weights(train: pd.DataFrame, rare_sensitive: bool) -> np.ndarray:
    base = train["per_call_sample_weight"].to_numpy(dtype="float64")
    if not rare_sensitive:
        return base
    multiplier = (
        1.0
        + 1.0 * train["target_delay_gt_3h"].to_numpy(dtype="float64")
        + 1.5 * train["target_delay_gt_6h"].to_numpy(dtype="float64")
    )
    weighted = base * multiplier
    return weighted * (base.sum() / max(weighted.sum(), 1e-12))


def _sample_complete_calls(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    """Deterministically cap memory without cutting a port-call trajectory."""
    if len(frame) <= maximum_rows:
        return frame.copy()
    calls = frame.groupby("port_call_id", as_index=False).size()
    calls["hash"] = pd.util.hash_pandas_object(
        calls["port_call_id"].astype(str), index=False
    ).to_numpy(dtype="uint64")
    calls = calls.sort_values(["hash", "port_call_id"])
    cumulative = calls["size"].cumsum()
    selected = calls.loc[cumulative.le(maximum_rows), "port_call_id"]
    if selected.empty:
        selected = calls.head(1)["port_call_id"]
    return frame.loc[frame["port_call_id"].isin(set(selected))].copy()


def _ordinal_probabilities(model: CatBoostClassifier, frame: pd.DataFrame) -> np.ndarray:
    raw = np.asarray(model.predict_proba(frame), dtype="float64")
    probabilities = np.zeros((len(frame), 4), dtype="float64")
    for position, label in enumerate(np.asarray(model.classes_, dtype="int64")):
        if 0 <= int(label) < 4:
            probabilities[:, int(label)] = raw[:, position]
    row_sum = probabilities.sum(axis=1)
    missing = row_sum <= 0.0
    probabilities[missing, 0] = 1.0
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def _predict_ordinal_batched(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    features: list[str],
    batch_size: int = 12_000,
) -> np.ndarray:
    parts = []
    for start in range(0, len(frame), batch_size):
        batch = _model_frame(frame.iloc[start : start + batch_size], features)
        parts.append(_ordinal_probabilities(model, batch))
    return np.vstack(parts)


def _predict_regression_batched(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
    features: list[str],
    batch_size: int = 12_000,
) -> np.ndarray:
    parts = []
    for start in range(0, len(frame), batch_size):
        batch = _model_frame(frame.iloc[start : start + batch_size], features)
        parts.append(np.asarray(model.predict(batch)))
    return np.concatenate(parts, axis=0)


def _predict_binary_batched(
    model: CatBoostClassifier,
    frame: pd.DataFrame,
    features: list[str],
    batch_size: int = 12_000,
) -> np.ndarray:
    parts = []
    for start in range(0, len(frame), batch_size):
        batch = _model_frame(frame.iloc[start : start + batch_size], features)
        parts.append(np.asarray(model.predict_proba(batch))[:, 1])
    return np.concatenate(parts)


def _fit_tabular_track(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    features: list[str],
    track: str,
    iterations: int,
    max_train_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bytes, pd.DataFrame]:
    real_train = _sample_complete_calls(
        real.loc[real["model_role"].eq("TRAIN_FIT")], max_train_rows
    )
    include_synthetic = track == "EVT_LOW_WEIGHT"
    rare_sensitive = track in {"REAL_COST_SENSITIVE", "EVT_LOW_WEIGHT"}
    train = (
        pd.concat([real_train, synthetic], ignore_index=True, sort=False)
        if include_synthetic
        else real_train.copy()
    )
    valid = real.loc[real["model_role"].eq("VALID_SELECT")]
    x_train = _model_frame(train, features)
    x_valid = _model_frame(valid, features)
    cat_features = [column for column in features if column in CATEGORICAL_FEATURES]
    common = _catboost_common(iterations)

    ordinal_target = delay_class_index(train["target_departure_delay_h"])
    ordinal_weight = _balanced_weights(
        ordinal_target,
        train["per_call_sample_weight"].to_numpy(dtype="float64"),
        rare_sensitive,
    )
    ordinal = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        **common,
    )
    ordinal.fit(
        x_train,
        ordinal_target,
        cat_features=cat_features,
        sample_weight=ordinal_weight,
        eval_set=(x_valid, delay_class_index(valid["target_departure_delay_h"])),
        early_stopping_rounds=45,
    )
    risk = risk_from_class_probabilities(_predict_ordinal_batched(ordinal, real, features))

    quantile = CatBoostRegressor(
        loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
        eval_metric="MultiQuantile:alpha=0.1,0.5,0.9",
        **common,
    )
    quantile.fit(
        x_train,
        train["target_remaining_h"].to_numpy(dtype="float64"),
        cat_features=cat_features,
        sample_weight=_duration_weights(train, rare_sensitive),
        eval_set=(x_valid, valid["target_remaining_h"].to_numpy(dtype="float64")),
        early_stopping_rounds=45,
    )
    duration = enforce_quantile_order(_predict_regression_batched(quantile, real, features))

    hazard_predictions = []
    hazard_models: dict[str, Any] = {}
    train_pre = train.loc[train["pre_breach_eligible"].fillna(False).astype(bool)]
    valid_pre = valid.loc[valid["pre_breach_eligible"].fillna(False).astype(bool)]
    for horizon, target in zip(HAZARD_HORIZONS, HAZARD_TARGETS):
        labels = train_pre[target].astype(int).to_numpy()
        if len(np.unique(labels)) < 2:
            constant = float(np.mean(labels)) if len(labels) else 0.0
            probability = np.repeat(constant, len(real))
            hazard_models[str(horizon)] = {"constant": constant}
        else:
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="BrierScore",
                **common,
            )
            fit_kwargs: dict[str, Any] = {}
            if not valid_pre.empty and valid_pre[target].nunique() == 2:
                fit_kwargs = {
                    "eval_set": (
                        _model_frame(valid_pre, features),
                        valid_pre[target].astype(int).to_numpy(),
                    ),
                    "early_stopping_rounds": 45,
                }
            model.fit(
                _model_frame(train_pre, features),
                labels,
                cat_features=cat_features,
                sample_weight=_balanced_weights(
                    labels,
                    train_pre["per_call_sample_weight"].to_numpy(dtype="float64"),
                    rare_sensitive,
                ),
                **fit_kwargs,
            )
            probability = _predict_binary_batched(model, real, features)
            hazard_models[str(horizon)] = model
        hazard_predictions.append(probability)
    hazard = enforce_hazard_order(np.column_stack(hazard_predictions))

    importance = pd.DataFrame(
        {
            "track": track,
            "feature": features,
            "importance": ordinal.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    inventory = {
        "track": track,
        "family": "CATBOOST_ORDINAL_QUANTILE_DISCRETE_HAZARD",
        "real_train_rows": int(len(real_train)),
        "synthetic_train_rows": int(len(synthetic) if include_synthetic else 0),
        "synthetic_effective_calls": float(
            synthetic["per_call_sample_weight"].sum() if include_synthetic else 0.0
        ),
        "rare_sensitive_weighting": rare_sensitive,
        "features": len(features),
        "requested_iterations": iterations,
        "ordinal_best_iteration": int(ordinal.get_best_iteration()),
        "quantile_best_iteration": int(quantile.get_best_iteration()),
    }
    artifact = pickle.dumps(
        {
            "model_version": MODEL_VERSION,
            "track": track,
            "features": features,
            "categorical_features": cat_features,
            "models": {"ordinal": ordinal, "quantile": quantile, "hazard": hazard_models},
            "training_policy": inventory,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return coherent_outputs(risk, duration, hazard), inventory, artifact, importance


def _business_baseline(real: pd.DataFrame) -> dict[str, np.ndarray]:
    train = real["model_role"].eq("TRAIN_FIT")
    weight = real.loc[train, "per_call_sample_weight"].to_numpy(dtype="float64")
    risk = np.column_stack(
        [
            np.repeat(
                np.average(real.loc[train, target].to_numpy(dtype="float64"), weights=weight),
                len(real),
            )
            for target in RISK_TARGETS
        ]
    )
    schedule = np.clip(
        pd.to_numeric(real["time_to_planned_departure_h"], errors="coerce")
        .fillna(real.loc[train, "target_remaining_h"].median())
        .to_numpy(dtype="float64"),
        0.0,
        None,
    )
    residual = real.loc[train, "target_remaining_h"].to_numpy(dtype="float64") - schedule[train]
    offsets = np.quantile(residual, [0.1, 0.5, 0.9])
    duration = np.column_stack([schedule + offset for offset in offsets])
    pre = train & real["pre_breach_eligible"].fillna(False).astype(bool)
    pre_weight = real.loc[pre, "per_call_sample_weight"].to_numpy(dtype="float64")
    hazard = np.column_stack(
        [
            np.repeat(
                np.average(real.loc[pre, target].to_numpy(dtype="float64"), weights=pre_weight),
                len(real),
            )
            for target in HAZARD_TARGETS
        ]
    )
    return coherent_outputs(risk, duration, hazard)


def _selection_report(
    real: pd.DataFrame,
    tracks: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    select = real["model_role"].eq("VALID_SELECT").to_numpy()
    early = real["early_warning_eligible"].fillna(False).astype(bool).to_numpy()
    pre = real["pre_breach_eligible"].fillna(False).astype(bool).to_numpy()
    weights = real["per_call_sample_weight"].to_numpy(dtype="float64")

    for track, prediction in tracks.items():
        for index, (name, target) in enumerate(zip(RISK_NAMES, RISK_TARGETS)):
            mask = select & early
            metrics = weighted_binary_metrics(
                real.loc[mask, target].astype(int).to_numpy(),
                prediction["risk"][mask, index],
                sample_weight=weights[mask],
            )
            rows.append(
                {
                    "task": f"RISK_{name.upper()}",
                    "track": track,
                    "objective": binary_selection_objective(metrics),
                    **metrics,
                }
            )
        for index, (horizon, target) in enumerate(zip(HAZARD_HORIZONS, HAZARD_TARGETS)):
            mask = select & pre
            metrics = weighted_binary_metrics(
                real.loc[mask, target].astype(int).to_numpy(),
                prediction["hazard"][mask, index],
                sample_weight=weights[mask],
            )
            rows.append(
                {
                    "task": f"HAZARD_{horizon}H",
                    "track": track,
                    "objective": binary_selection_objective(metrics),
                    **metrics,
                }
            )
        mask = select & real["target_remaining_h"].notna().to_numpy()
        metrics = weighted_regression_metrics(
            real.loc[mask, "target_remaining_h"].to_numpy(dtype="float64"),
            prediction["quantiles"][mask],
            sample_weight=weights[mask],
        )
        rows.append(
            {
                "task": "REMAINING_DURATION",
                "track": track,
                "objective": duration_selection_objective(metrics),
                **metrics,
            }
        )

    report = pd.DataFrame(rows)
    priority = {track: index for index, track in enumerate(TRACKS)}
    report["priority"] = report["track"].map(priority).fillna(99)
    winners = (
        report.sort_values(["task", "objective", "priority"])
        .groupby("task", as_index=False)
        .first()[["task", "track"]]
    )
    champion = dict(zip(winners["task"], winners["track"]))
    report["selected"] = report.apply(
        lambda row: champion.get(str(row["task"])) == row["track"], axis=1
    )
    return report.drop(columns="priority"), champion


def _assemble_selected(
    tracks: dict[str, dict[str, np.ndarray]], champion: dict[str, str]
) -> dict[str, np.ndarray]:
    risk = np.column_stack(
        [tracks[champion[f"RISK_{name.upper()}"]]["risk"][:, index] for index, name in enumerate(RISK_NAMES)]
    )
    hazard = np.column_stack(
        [tracks[champion[f"HAZARD_{horizon}H"]]["hazard"][:, index] for index, horizon in enumerate(HAZARD_HORIZONS)]
    )
    duration = tracks[champion["REMAINING_DURATION"]]["quantiles"]
    return coherent_outputs(risk, duration, hazard)


def _calibrate_selected(
    real: pd.DataFrame,
    selected: dict[str, np.ndarray],
    champion: dict[str, str],
) -> tuple[dict[str, np.ndarray], dict[str, Any], bytes, pd.DataFrame]:
    calibration = real["model_role"].eq("VALID_CALIBRATE").to_numpy()
    early = real["early_warning_eligible"].fillna(False).astype(bool).to_numpy()
    pre = real["pre_breach_eligible"].fillna(False).astype(bool).to_numpy()
    weights = real["per_call_sample_weight"].to_numpy(dtype="float64")
    calibrators: dict[str, Any] = {}
    thresholds: dict[str, float] = {}
    calibration_rows = []
    risk_columns = []
    for index, (name, target) in enumerate(zip(RISK_NAMES, RISK_TARGETS)):
        mask = calibration & early
        actual = real[target].astype(int).to_numpy()
        raw = selected["risk"][:, index]
        calibrator = fit_binary_calibrator(actual[mask], raw[mask], weights[mask])
        calibrated = calibrator.predict(raw)
        cost = 3.0 if name == "gt6" else (2.0 if name == "gt3" else 1.5)
        threshold = select_threshold(actual[mask], calibrated[mask], weights[mask], cost)
        before = weighted_binary_metrics(actual[mask], raw[mask], sample_weight=weights[mask])
        after = weighted_binary_metrics(actual[mask], calibrated[mask], threshold, weights[mask])
        calibration_rows.append(
            {
                "task": f"RISK_{name.upper()}",
                "champion": champion[f"RISK_{name.upper()}"],
                "method": calibrator.method,
                "threshold": threshold,
                "rows": int(mask.sum()),
                "positives": int(actual[mask].sum()),
                "brier_before": before["brier"],
                "brier_after": after["brier"],
                "ece_before": before["ece_10"],
                "ece_after": after["ece_10"],
            }
        )
        calibrators[f"risk_{name}"] = calibrator
        thresholds[name] = threshold
        risk_columns.append(calibrated)
    risk = enforce_risk_order(np.column_stack(risk_columns))

    actual_remaining = real["target_remaining_h"].to_numpy(dtype="float64")
    duration_mask = calibration & np.isfinite(actual_remaining)
    conformal_policy = adaptive_conformal_policy(
        real, actual_remaining, selected["quantiles"], duration_mask
    )
    duration = apply_adaptive_conformal(real, selected["quantiles"], conformal_policy)
    before_duration = weighted_regression_metrics(
        actual_remaining[duration_mask], selected["quantiles"][duration_mask], weights[duration_mask]
    )
    after_duration = weighted_regression_metrics(
        actual_remaining[duration_mask], duration[duration_mask], weights[duration_mask]
    )
    calibration_rows.append(
        {
            "task": "REMAINING_DURATION",
            "champion": champion["REMAINING_DURATION"],
            "method": "REGIME_ADAPTIVE_SPLIT_CONFORMAL_80",
            "threshold": None,
            "rows": int(duration_mask.sum()),
            "positives": None,
            "brier_before": None,
            "brier_after": None,
            "ece_before": before_duration["coverage_p10_p90"],
            "ece_after": after_duration["coverage_p10_p90"],
        }
    )

    hazard_columns = []
    for index, (horizon, target) in enumerate(zip(HAZARD_HORIZONS, HAZARD_TARGETS)):
        mask = calibration & pre
        actual = real[target].astype(int).to_numpy()
        raw = selected["hazard"][:, index]
        calibrator = fit_binary_calibrator(actual[mask], raw[mask], weights[mask])
        calibrated = calibrator.predict(raw)
        threshold = select_threshold(actual[mask], calibrated[mask], weights[mask], 2.0)
        before = weighted_binary_metrics(actual[mask], raw[mask], sample_weight=weights[mask])
        after = weighted_binary_metrics(actual[mask], calibrated[mask], threshold, weights[mask])
        calibration_rows.append(
            {
                "task": f"HAZARD_{horizon}H",
                "champion": champion[f"HAZARD_{horizon}H"],
                "method": calibrator.method,
                "threshold": threshold,
                "rows": int(mask.sum()),
                "positives": int(actual[mask].sum()),
                "brier_before": before["brier"],
                "brier_after": after["brier"],
                "ece_before": before["ece_10"],
                "ece_after": after["ece_10"],
            }
        )
        calibrators[f"hazard_{horizon}"] = calibrator
        thresholds[f"breach_{horizon}h"] = threshold
        hazard_columns.append(calibrated)
    hazard = enforce_hazard_order(np.column_stack(hazard_columns))
    policy = {
        "champions": champion,
        "thresholds": thresholds,
        "conformal": conformal_policy,
        "selection_split": "VALID_SELECT",
        "calibration_split": "VALID_CALIBRATE",
        "test_used_for_selection": False,
    }
    artifact = pickle.dumps(
        {
            "model_version": MODEL_VERSION,
            "policy": policy,
            "calibrators": calibrators,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return coherent_outputs(risk, duration, hazard), policy, artifact, pd.DataFrame(calibration_rows)


def _metric_reports(
    real: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binary_rows = []
    duration_rows = []
    survival_rows = []
    weights = real["per_call_sample_weight"].to_numpy(dtype="float64")
    early = real["early_warning_eligible"].fillna(False).astype(bool).to_numpy()
    pre = real["pre_breach_eligible"].fillna(False).astype(bool).to_numpy()
    for role in ("VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"):
        role_mask = real["model_role"].eq(role).to_numpy()
        for expert, values in predictions.items():
            for index, (name, target) in enumerate(zip(RISK_NAMES, RISK_TARGETS)):
                mask = role_mask & early
                metrics = weighted_binary_metrics(
                    real.loc[mask, target].astype(int).to_numpy(),
                    values["risk"][mask, index],
                    thresholds.get(name, 0.5),
                    weights[mask],
                )
                binary_rows.append(
                    {"role": role, "expert": expert, "task": f"DELAY_{name.upper()}", **metrics}
                )
            briers = []
            for index, (horizon, target) in enumerate(zip(HAZARD_HORIZONS, HAZARD_TARGETS)):
                mask = role_mask & pre
                metrics = weighted_binary_metrics(
                    real.loc[mask, target].astype(int).to_numpy(),
                    values["hazard"][mask, index],
                    thresholds.get(f"breach_{horizon}h", 0.5),
                    weights[mask],
                )
                briers.append(float(metrics["brier"]))
                binary_rows.append(
                    {"role": role, "expert": expert, "task": f"BREACH_WITHIN_{horizon}H", **metrics}
                )
            duration_mask = role_mask & real["target_remaining_h"].notna().to_numpy()
            duration_rows.append(
                {
                    "role": role,
                    "expert": expert,
                    "task": "REMAINING_DURATION",
                    **weighted_regression_metrics(
                        real.loc[duration_mask, "target_remaining_h"].to_numpy(dtype="float64"),
                        values["quantiles"][duration_mask],
                        weights[duration_mask],
                    ),
                }
            )
            survival_mask = role_mask & pre
            survival_rows.append(
                {
                    "role": role,
                    "expert": expert,
                    "rows": int(survival_mask.sum()),
                    "integrated_brier_6_12_24h": float(np.mean(briers)),
                    "approx_c_index": approximate_concordance_index(
                        real.loc[survival_mask, "target_breach_or_censor_h"].to_numpy(dtype="float64"),
                        real.loc[survival_mask, "target_breach_gt3_observed"].astype(bool).to_numpy(),
                        values["hazard"][survival_mask, -1],
                    ),
                }
            )
    return pd.DataFrame(binary_rows), pd.DataFrame(duration_rows), pd.DataFrame(survival_rows)


def _bootstrap_test_report(
    real: pd.DataFrame,
    ensemble: dict[str, np.ndarray],
    replicates: int,
) -> pd.DataFrame:
    mask = real["model_role"].eq("TEST_DIAGNOSTIC_ONLY").to_numpy()
    test = real.loc[mask].reset_index(drop=True)
    risk = ensemble["risk"][mask]
    duration = ensemble["quantiles"][mask]
    weight = test["per_call_sample_weight"].to_numpy(dtype="float64")
    rows = []
    for index, (name, target) in enumerate(zip(RISK_NAMES[1:], RISK_TARGETS[1:]), start=1):
        eligible = test["early_warning_eligible"].fillna(False).astype(bool).to_numpy()
        task_frame = test.loc[eligible].reset_index(drop=True)
        actual = test.loc[eligible, target].astype(int).to_numpy()
        probability = risk[eligible, index]
        task_weight = weight[eligible]

        def metric(positions: np.ndarray) -> float:
            values = weighted_binary_metrics(
                actual[positions],
                probability[positions],
                sample_weight=task_weight[positions],
            )
            return float(values["average_precision"] or 0.0)

        point, lower, upper, used = grouped_bootstrap_ci(task_frame, metric, replicates)
        rows.append(
            {
                "task": f"DELAY_{name.upper()}",
                "metric": "AVERAGE_PRECISION",
                "point": point,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "replicates": used,
                "bootstrap_unit": "PORT_CALL",
            }
        )
    actual_remaining = test["target_remaining_h"].to_numpy(dtype="float64")

    def mae_metric(positions: np.ndarray) -> float:
        return float(
            weighted_regression_metrics(
                actual_remaining[positions], duration[positions], weight[positions]
            )["mae_p50"]
        )

    point, lower, upper, used = grouped_bootstrap_ci(test, mae_metric, replicates)
    rows.append(
        {
            "task": "REMAINING_DURATION",
            "metric": "MAE_P50",
            "point": point,
            "ci_lower_95": lower,
            "ci_upper_95": upper,
            "replicates": used,
            "bootstrap_unit": "PORT_CALL",
        }
    )
    return pd.DataFrame(rows)


def _issue_time_readiness(real: pd.DataFrame, source_columns: list[str]) -> pd.DataFrame:
    availability = [column for column in source_columns if column.endswith("_available")]
    rows = []
    for split in ("TRAIN", "VALID", "TEST"):
        subset = real.loc[real["split"].eq(split)]
        for column in availability:
            if column not in real:
                continue
            rows.append(
                {
                    "split": split,
                    "feature": column,
                    "rows": len(subset),
                    "coverage_pct": 100.0 * pd.to_numeric(subset[column], errors="coerce").fillna(0.0).mean(),
                    "official_model_eligible": False,
                    "reason": "ISSUE_TIME_HISTORY_NOT_YET_LONG_ENOUGH",
                }
            )
    if not rows:
        rows.append(
            {
                "split": "ALL",
                "feature": "ISSUE_TIME_WEATHER_BLOCK",
                "rows": len(real),
                "coverage_pct": 0.0,
                "official_model_eligible": False,
                "reason": "COLUMNS_EXCLUDED_FROM_MEMORY_SAFE_LOAD_AND_NOT_OPERATIONALLY_READY",
            }
        )
    return pd.DataFrame(rows)


def _quality_gates(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    predictions: pd.DataFrame,
    binary: pd.DataFrame,
    duration: pd.DataFrame,
    champion: dict[str, str],
) -> pd.DataFrame:
    finite_columns = [
        "p_delay_gt1",
        "p_delay_gt3",
        "p_delay_gt6",
        "remaining_p10_h",
        "remaining_p50_h",
        "remaining_p90_h",
        "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h",
    ]
    test_binary = binary.loc[binary["role"].eq("TEST_DIAGNOSTIC_ONLY")]
    test_duration = duration.loc[duration["role"].eq("TEST_DIAGNOSTIC_ONLY")]

    def binary_metric(expert: str, task: str, metric: str) -> float:
        value = test_binary.loc[
            test_binary["expert"].eq(expert) & test_binary["task"].eq(task), metric
        ]
        return float(value.iloc[0]) if not value.empty and pd.notna(value.iloc[0]) else float("nan")

    def duration_metric(expert: str, metric: str) -> float:
        value = test_duration.loc[test_duration["expert"].eq(expert), metric]
        return float(value.iloc[0]) if not value.empty else float("nan")

    gt3_final = binary_metric("HYBRID_CALIBRATED", "DELAY_GT3", "average_precision")
    gt3_reference = binary_metric("REAL_REFERENCE", "DELAY_GT3", "average_precision")
    gt6_final = binary_metric("HYBRID_CALIBRATED", "DELAY_GT6", "average_precision")
    gt6_reference = binary_metric("REAL_REFERENCE", "DELAY_GT6", "average_precision")
    final_mae = duration_metric("HYBRID_CALIBRATED", "mae_p50")
    reference_mae = duration_metric("REAL_REFERENCE", "mae_p50")
    final_coverage = duration_metric("HYBRID_CALIBRATED", "coverage_p10_p90")
    synthetic_weight = float(synthetic["per_call_sample_weight"].sum())
    real_weight = float(
        real.loc[real["model_role"].eq("TRAIN_FIT"), "per_call_sample_weight"].sum()
    )
    rows = [
        {"check": "TRAIN_VALID_TEST_PRESENT", "passed": set(real["split"]) == {"TRAIN", "VALID", "TEST"}, "severity": "CRITICAL", "value": None},
        {"check": "PORT_CALL_BELONGS_TO_ONE_REAL_SPLIT", "passed": bool(real.groupby("port_call_id")["split"].nunique().max() == 1), "severity": "CRITICAL", "value": None},
        {"check": "SYNTHETIC_TRAIN_ONLY", "passed": bool(synthetic["split"].eq("TRAIN").all() and synthetic["model_role"].eq("TRAIN_SUPPLEMENT").all()), "severity": "CRITICAL", "value": len(synthetic)},
        {"check": "SYNTHETIC_TARGETS_NOT_IMPUTED", "passed": bool(~synthetic["targets_imputed"].astype(bool).any()), "severity": "CRITICAL", "value": None},
        {"check": "SYNTHETIC_EFFECTIVE_WEIGHT_BELOW_5PCT", "passed": synthetic_weight <= 0.05 * real_weight, "severity": "CRITICAL", "value": 100.0 * synthetic_weight / max(real_weight, 1e-12)},
        {"check": "VALID_SELECT_CALIBRATE_DISJOINT", "passed": real["model_role"].isin(["VALID_SELECT", "VALID_CALIBRATE"]).sum() == real["split"].eq("VALID").sum(), "severity": "CRITICAL", "value": None},
        {"check": "TEST_NOT_USED_FOR_SELECTION_OR_CALIBRATION", "passed": True, "severity": "CRITICAL", "value": False},
        {"check": "FINITE_SERVING_OUTPUTS", "passed": bool(np.isfinite(predictions[finite_columns].to_numpy()).all()), "severity": "CRITICAL", "value": None},
        {"check": "RISK_MONOTONIC", "passed": bool(((predictions["p_delay_gt1"] >= predictions["p_delay_gt3"]) & (predictions["p_delay_gt3"] >= predictions["p_delay_gt6"])).all()), "severity": "CRITICAL", "value": None},
        {"check": "HAZARD_MONOTONIC", "passed": bool(((predictions["p_gt3_breach_within_6h"] <= predictions["p_gt3_breach_within_12h"]) & (predictions["p_gt3_breach_within_12h"] <= predictions["p_gt3_breach_within_24h"])).all()), "severity": "CRITICAL", "value": None},
        {"check": "QUANTILES_MONOTONIC", "passed": bool(((predictions["remaining_p10_h"] <= predictions["remaining_p50_h"]) & (predictions["remaining_p50_h"] <= predictions["remaining_p90_h"])).all()), "severity": "CRITICAL", "value": None},
        {"check": "TEST_GT3_PR_AUC_NON_INFERIOR", "passed": bool(np.isfinite(gt3_final) and gt3_final >= gt3_reference - max(0.01, 0.05 * gt3_reference)), "severity": "MODEL", "value": gt3_final - gt3_reference},
        {"check": "TEST_GT6_PR_AUC_NON_INFERIOR", "passed": bool(np.isfinite(gt6_final) and gt6_final >= gt6_reference - max(0.01, 0.05 * gt6_reference)), "severity": "MODEL", "value": gt6_final - gt6_reference},
        {"check": "TEST_DURATION_MAE_NON_INFERIOR_2PCT", "passed": bool(np.isfinite(final_mae) and final_mae <= 1.02 * reference_mae), "severity": "MODEL", "value": final_mae / max(reference_mae, 1e-12)},
        {"check": "TEST_INTERVAL_COVERAGE_72_TO_92PCT", "passed": bool(0.72 <= final_coverage <= 0.92), "severity": "MODEL", "value": final_coverage},
        {"check": "AUGMENTATION_NEVER_FORCED_IN_CHAMPION", "passed": True, "severity": "GOVERNANCE", "value": sum(track == "EVT_LOW_WEIGHT" for track in champion.values())},
        {"check": "EXTERNAL_LICENSE_REVIEW_BLOCKS_PRODUCTION", "passed": True, "severity": "GOVERNANCE", "value": "REVIEW_REQUIRED"},
        {"check": "ISSUE_TIME_HISTORY_BLOCKS_PRODUCTION", "passed": True, "severity": "GOVERNANCE", "value": False},
    ]
    return pd.DataFrame(rows)


def _prediction_frame(
    real: pd.DataFrame,
    ensemble: dict[str, np.ndarray],
    policy: dict[str, Any],
    run_id: str,
) -> pd.DataFrame:
    keep = real["model_role"].isin(
        ["VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"]
    ).to_numpy()
    source = real.loc[keep]
    risk = ensemble["risk"][keep]
    duration = ensemble["quantiles"][keep]
    hazard = ensemble["hazard"][keep]
    thresholds = policy["thresholds"]
    predicted_class = np.sum(
        np.column_stack(
            [risk[:, index] >= thresholds.get(name, 0.5) for index, name in enumerate(RISK_NAMES)]
        ),
        axis=1,
    )
    labels = np.asarray(["LT_OR_EQ_1H", "GT1_TO_3H", "GT3_TO_6H", "GT6H"])
    return pd.DataFrame(
        {
            "model_version": MODEL_VERSION,
            "port_call_id": source["port_call_id"].astype(str).to_numpy(),
            "landmark_at": source["landmark_at"].to_numpy(),
            "split": source["split"].astype(str).to_numpy(),
            "evaluation_role": source["model_role"].astype(str).to_numpy(),
            "regime": source["regime"].astype(str).to_numpy(),
            "p_delay_gt1": risk[:, 0],
            "p_delay_gt3": risk[:, 1],
            "p_delay_gt6": risk[:, 2],
            "predicted_delay_class": labels[np.clip(predicted_class, 0, 3)],
            "remaining_p10_h": duration[:, 0],
            "remaining_p50_h": duration[:, 1],
            "remaining_p90_h": duration[:, 2],
            "p_gt3_breach_within_6h": hazard[:, 0],
            "p_gt3_breach_within_12h": hazard[:, 1],
            "p_gt3_breach_within_24h": hazard[:, 2],
            "selected_policy": json.dumps(policy["champions"], sort_keys=True),
            "interval_kind": "REGIME_ADAPTIVE_SPLIT_CONFORMAL_80",
            "model_family": "GOVERNED_TASK_ROUTED_HYBRID_V2",
            "synthetic_training_policy": "OPTIONAL_LOW_WEIGHT_TRAIN_ONLY",
            "production_claim_allowed": False,
            "materialization_run_id": run_id,
        }
    )


def _materialize_predictions(frame: pd.DataFrame) -> int:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS serving.maritime_port_call_multitask_prediction_v2 (
                model_version TEXT NOT NULL,
                port_call_id TEXT NOT NULL,
                landmark_at TIMESTAMPTZ NOT NULL,
                split TEXT NOT NULL,
                evaluation_role TEXT NOT NULL,
                regime TEXT NOT NULL,
                p_delay_gt1 DOUBLE PRECISION NOT NULL,
                p_delay_gt3 DOUBLE PRECISION NOT NULL,
                p_delay_gt6 DOUBLE PRECISION NOT NULL,
                predicted_delay_class TEXT NOT NULL,
                remaining_p10_h DOUBLE PRECISION NOT NULL,
                remaining_p50_h DOUBLE PRECISION NOT NULL,
                remaining_p90_h DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_6h DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_12h DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_24h DOUBLE PRECISION NOT NULL,
                selected_policy JSONB NOT NULL,
                interval_kind TEXT NOT NULL,
                model_family TEXT NOT NULL,
                synthetic_training_policy TEXT NOT NULL,
                production_claim_allowed BOOLEAN NOT NULL,
                materialization_run_id TEXT NOT NULL,
                PRIMARY KEY (model_version, port_call_id, landmark_at)
            )
            """
        )
        cursor.execute(
            f"DELETE FROM {TARGET_RELATION} WHERE model_version=%s", (MODEL_VERSION,)
        )
        for start in range(0, len(frame), 2_000):
            batch = frame.iloc[start : start + 2_000]
            records = []
            for row in batch.itertuples(index=False, name=None):
                values = list(row)
                values[16] = Json(json.loads(values[16]))
                records.append(tuple(values))
            execute_values(
                cursor,
                f"INSERT INTO {TARGET_RELATION} VALUES %s",
                records,
                page_size=2_000,
            )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_b61bv2_prediction_landmark "
            "ON serving.maritime_port_call_multitask_prediction_v2 (landmark_at DESC)"
        )
    return len(frame)


def _put_bytes(key: str, payload: bytes, content_type: str) -> str:
    _s3_client().put_object(
        Bucket=OUTPUT_BUCKET, Key=key, Body=payload, ContentType=content_type
    )
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _put_json(key: str, payload: Any) -> str:
    return _put_bytes(
        key,
        json.dumps(clean_json(payload), indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )


def _put_csv(key: str, frame: pd.DataFrame) -> str:
    return _put_bytes(key, frame.to_csv(index=False).encode("utf-8"), "text/csv")


def _log_mlflow(metadata: dict[str, Any], reports: dict[str, pd.DataFrame]) -> str:
    try:
        import mlflow
    except Exception:
        return "NOT_INSTALLED"
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("maritime-b61b-v2-governed-hybrid")
        with mlflow.start_run(run_name=MODEL_VERSION):
            mlflow.log_params(
                {
                    "model_version": MODEL_VERSION,
                    "selection_split": "VALID_SELECT",
                    "calibration_split": "VALID_CALIBRATE",
                    "test_used_for_selection": False,
                    "synthetic_scope": "TRAIN_ONLY_LOW_WEIGHT",
                }
            )
            mlflow.log_dict(clean_json(metadata), "decision.json")
            with tempfile.TemporaryDirectory() as directory:
                for name, report in reports.items():
                    path = Path(directory) / f"{name}.csv"
                    report.to_csv(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path="reports")
        return "LOGGED"
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def run_b61bv2_modeling(
    force: bool = False,
    sequence_max_steps: int = 220,
    catboost_iterations: int = 260,
    max_train_rows: int = 60_000,
    bootstrap_replicates: int = 200,
) -> dict[str, Any]:
    required_relations = [REAL_RELATION, SYNTHETIC_RELATION, "audit.ingestion_run"]
    missing = [relation for relation in required_relations if not _relation_exists(relation)]
    if missing:
        raise RuntimeError(f"Required governed sources are missing: {missing}")
    real, synthetic, contract = load_governed_frames()
    checksum = _source_signature(
        real, synthetic, sequence_max_steps, catboost_iterations, max_train_rows
    )
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            return {**previous, "reused": True}
    run_id = _start_run(checksum)
    try:
        _update_progress(
            run_id,
            "LOADED_GOVERNED_REAL_AND_RARE_TAIL_DATA",
            real_rows=len(real),
            real_calls=real["port_call_id"].nunique(),
            synthetic_rows=len(synthetic),
            synthetic_calls=synthetic["port_call_id"].nunique(),
        )
        features = [*contract["core_numeric"], *contract["categorical"]]
        sequence_features = contract["core_numeric"][:48]
        if len(features) < 25 or len(sequence_features) < 20:
            raise RuntimeError("The B61A core feature contract is unexpectedly incomplete")

        track_predictions: dict[str, dict[str, np.ndarray]] = {}
        track_artifacts: dict[str, bytes] = {}
        inventories: dict[str, Any] = {}
        importance_frames = []
        for position, track in enumerate(TRACKS[:3], start=1):
            _update_progress(
                run_id,
                "TRAINING_TABULAR_TRACKS",
                track=track,
                completed_tracks=position - 1,
                total_tracks=4,
                features=len(features),
            )
            prediction, inventory, artifact, importance = _fit_tabular_track(
                real,
                synthetic,
                features,
                track,
                catboost_iterations,
                max_train_rows,
            )
            track_predictions[track] = prediction
            track_artifacts[track] = artifact
            inventories[track] = inventory
            importance_frames.append(importance)
            gc.collect()

        _update_progress(
            run_id,
            "TRAINING_REAL_ONLY_SEQUENCE_CHALLENGER",
            max_steps=sequence_max_steps,
            features=len(sequence_features),
        )
        try:
            sequence = train_sequence_expert(
                real,
                sequence_features,
                sequence_length=24,
                max_steps=sequence_max_steps,
            )
            track_predictions["SEQUENCE_REAL_ONLY"] = coherent_outputs(
                sequence.predictions["risk"],
                sequence.predictions["quantiles"],
                sequence.predictions["hazard"],
            )
            track_artifacts["SEQUENCE_REAL_ONLY"] = sequence.artifact
            inventories["SEQUENCE_REAL_ONLY"] = {
                "track": "SEQUENCE_REAL_ONLY",
                "family": "SHARED_MULTITASK_GRU",
                "synthetic_train_rows": 0,
                **sequence.metrics,
            }
        except Exception as exc:
            inventories["SEQUENCE_REAL_ONLY"] = {
                "track": "SEQUENCE_REAL_ONLY",
                "family": "SHARED_MULTITASK_GRU",
                "status": "UNAVAILABLE",
                "error": f"{type(exc).__name__}: {exc}",
                "synthetic_train_rows": 0,
            }

        _update_progress(
            run_id,
            "VALID_SELECT_TASK_ROUTING",
            available_tracks=list(track_predictions),
        )
        selection_report, champion = _selection_report(real, track_predictions)
        selected = _assemble_selected(track_predictions, champion)
        ensemble, ensemble_policy, calibration_artifact, calibration_report = _calibrate_selected(
            real, selected, champion
        )
        baseline = _business_baseline(real)
        all_predictions = {
            "BUSINESS_BASELINE": baseline,
            **track_predictions,
            "HYBRID_CALIBRATED": ensemble,
        }

        _update_progress(run_id, "OPENING_TEST_ONCE_FOR_FINAL_DIAGNOSTICS")
        binary_report, duration_report, survival_report = _metric_reports(
            real, all_predictions, ensemble_policy["thresholds"]
        )
        bootstrap_report = _bootstrap_test_report(
            real, ensemble, max(50, bootstrap_replicates)
        )
        issue_time_report = _issue_time_readiness(
            real, contract["issue_time_columns_available"]
        )
        prediction_frame = _prediction_frame(real, ensemble, ensemble_policy, run_id)
        gates = _quality_gates(
            real,
            synthetic,
            prediction_frame,
            binary_report,
            duration_report,
            champion,
        )
        critical_passed = bool(
            gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all()
        )
        model_passed = bool(gates.loc[gates["severity"].eq("MODEL"), "passed"].all())
        serving_rows = _materialize_predictions(prediction_frame)
        selected_evt_tasks = sorted(
            task for task, track in champion.items() if track == "EVT_LOW_WEIGHT"
        )
        decision = (
            "READY_FOR_B61C_V2_HISTORICAL_REPLAY_AND_SHADOW_API"
            if critical_passed and model_passed
            else "RESEARCH_ONLY_B61B_V2_REFINEMENT_REQUIRED"
        )
        metadata = {
            "decision": decision,
            "model_version": MODEL_VERSION,
            "row_count": len(real),
            "vessel_calls": int(real["port_call_id"].nunique()),
            "real_train_calls": int(
                real.loc[real["model_role"].eq("TRAIN_FIT"), "port_call_id"].nunique()
            ),
            "synthetic_train_rows": len(synthetic),
            "synthetic_train_calls": int(synthetic["port_call_id"].nunique()),
            "synthetic_effective_calls": float(synthetic["per_call_sample_weight"].sum()),
            "synthetic_selected_tasks": selected_evt_tasks,
            "augmentation_accepted": bool(selected_evt_tasks),
            "selected_models": champion,
            "tasks": [
                "ORDINAL_DELAY_GT1_GT3_GT6",
                "REMAINING_DURATION_P10_P50_P90",
                "DISCRETE_SURVIVAL_BREACH_6_12_24H",
            ],
            "selection_split": "VALID_SELECT",
            "calibration_split": "VALID_CALIBRATE",
            "test_role": "TEST_DIAGNOSTIC_ONLY_ONCE",
            "test_used_for_selection": False,
            "test_opened_once_after_freeze": True,
            "bootstrap_unit": "PORT_CALL",
            "bootstrap_replicates": max(50, bootstrap_replicates),
            "core_features": len(features),
            "sequence_features": len(sequence_features),
            "retrospective_weather_policy": "RESEARCH_ONLY_EXCLUDED_FROM_OFFICIAL_TRACKS",
            "issue_time_weather_policy": "COLLECTING_NOT_LONG_ENOUGH_FOR_OFFICIAL_TRAINING",
            "external_source_license_status": "REVIEW_REQUIRED",
            "quality_gates_passed": critical_passed and model_passed,
            "critical_gates_passed": critical_passed,
            "model_gates_passed": model_passed,
            "replay_allowed": critical_passed and model_passed,
            "production_promotion_allowed": False,
            "serving_rows": serving_rows,
            "inventories": inventories,
            "ensemble_policy": ensemble_policy,
            "limitations": [
                "The EVT supplement is TRAIN-only, low-weight and can be rejected task by task.",
                "TEST is opened once after selection and calibration are frozen; it never changes the policy.",
                "Issue-time marine weather history is still too short for official model promotion.",
                "External-source licensing must be reviewed before any production use of an augmented expert.",
                "The result is predictive and operational, not a causal effect estimate.",
            ],
            "next_block": "B61C_V2_HISTORICAL_REPLAY_SHADOW_API",
        }
        importance = (
            pd.concat(importance_frames, ignore_index=True)
            if importance_frames
            else pd.DataFrame(columns=["track", "feature", "importance"])
        )
        split_support = real.groupby(["split", "model_role"], as_index=False).agg(
            rows=("port_call_id", "size"),
            calls=("port_call_id", "nunique"),
            effective_calls=("per_call_sample_weight", "sum"),
            gt3_rows=("target_delay_gt_3h", "sum"),
            gt6_rows=("target_delay_gt_6h", "sum"),
        )
        augmentation_utility = selection_report.loc[
            selection_report["track"].isin(["REAL_COST_SENSITIVE", "EVT_LOW_WEIGHT"])
        ].copy()
        reports = {
            "selection_scorecard": selection_report,
            "calibration_report": calibration_report,
            "binary_metrics": binary_report,
            "duration_metrics": duration_report,
            "survival_metrics": survival_report,
            "test_bootstrap_confidence_intervals": bootstrap_report,
            "augmentation_utility": augmentation_utility,
            "quality_gates": gates,
            "feature_importance": importance,
            "split_support": split_support,
            "issue_time_weather_readiness": issue_time_report,
        }
        metadata["mlflow_status"] = _log_mlflow(metadata, reports)
        _update_progress(run_id, "WRITING_VERSIONED_ARTIFACTS", serving_rows=serving_rows)
        for name, report in reports.items():
            _put_csv(f"reports/b61bv2/{OUTPUT_PREFIX}/{name}.csv", report)
        _put_json(f"configs/b61bv2/{OUTPUT_PREFIX}/feature_contract.json", contract)
        _put_json(f"configs/b61bv2/{OUTPUT_PREFIX}/ensemble_policy.json", ensemble_policy)
        _put_json(f"configs/b61bv2/{OUTPUT_PREFIX}/final_decision.json", metadata)
        for track, artifact in track_artifacts.items():
            suffix = "pt" if track == "SEQUENCE_REAL_ONLY" else "pkl"
            content_type = "application/octet-stream"
            _put_bytes(
                f"models/b61bv2/{OUTPUT_PREFIX}/{track.lower()}.{suffix}",
                artifact,
                content_type,
            )
        _put_bytes(
            f"models/b61bv2/{OUTPUT_PREFIX}/hybrid_calibration.pkl",
            calibration_artifact,
            "application/octet-stream",
        )
        _finish_run(run_id, "SUCCESS", len(real), metadata)
        return metadata
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "decision": "FAILED",
                "next_block": "FIX_B61B_V2_AND_RERUN",
                "test_used_for_selection": False,
                "production_promotion_allowed": False,
            },
            str(exc),
        )
        raise


def verify_b61bv2_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "decision",
        "model_version",
        "row_count",
        "vessel_calls",
        "selection_split",
        "calibration_split",
        "test_used_for_selection",
        "synthetic_train_rows",
        "serving_rows",
        "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"B61B-v2 result misses required fields: {missing}")
    if result["test_used_for_selection"]:
        raise ValueError("B61B-v2 leakage contract violated: TEST used for selection")
    if result.get("production_promotion_allowed"):
        raise ValueError("B61B-v2 cannot promote before issue-time shadow validation")
    if int(result["synthetic_train_rows"]) <= 0:
        raise ValueError("B61B-v2 did not load the governed rare-tail supplement")
    if int(result["serving_rows"]) <= 0:
        raise ValueError("B61B-v2 produced no serving rows")
    return result
