from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from scipy.stats import kruskal
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


AUDIT_VERSION = "b54ea-dependency-weather-v1"
SOURCE_NAME = "b54d_temporal_model_artifacts"
DATASET_NAME = "b54ea_dependency_weather_audit"
TARGET = "target_arrival_delay_h"
RANDOM_SEED = int(os.getenv("B54EA_RANDOM_SEED", "42"))
MI_MAX_ROWS = int(os.getenv("B54EA_MI_MAX_ROWS", "25000"))
CORR_MAX_ROWS = int(os.getenv("B54EA_CORR_MAX_ROWS", "60000"))
MIN_CORR_ROWS = int(os.getenv("B54EA_MIN_CORR_ROWS", "100"))
MIN_CATEGORY_ROWS = int(os.getenv("B54EA_MIN_CATEGORY_ROWS", "30"))

SEA_LABELS = [
    "CALM_LT0P5",
    "SLIGHT_0P5_1P25",
    "MODERATE_1P25_2P5",
    "ROUGH_2P5_4",
    "VERY_ROUGH_GE4",
]
SEA_BINS = [-np.inf, 0.5, 1.25, 2.5, 4.0, np.inf]

BANNED_EXACT = {
    "port_call_id",
    "source_record_id",
    "prediction_time",
    "planned_eta",
    "actual_ata",
    "actual_atd",
    "target_arrival_delay_h",
    "target_departure_delay_h",
    "split",
    "model_ready_flag",
}
BANNED_TOKENS = (
    "actual_",
    "target_",
    "future_",
    "label",
    "quarantine",
    "outlier",
)


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


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


def _etag(client, bucket: str, key: str) -> str:
    return str(client.head_object(Bucket=bucket, Key=key)["ETag"]).strip('"')


def _download(client, bucket: str, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target))


def _upload(client, source: Path, bucket: str, key: str, content_type: str) -> str:
    client.upload_file(
        str(source), bucket, key, ExtraArgs={"ContentType": content_type}
    )
    return f"s3://{bucket}/{key}"


def _signature(client, objects: list[tuple[str, str]]) -> str:
    payload = {
        "audit_version": AUDIT_VERSION,
        "objects": [
            {"bucket": bucket, "key": key, "etag": _etag(client, bucket, key)}
            for bucket, key in objects
        ],
        "mi_max_rows": MI_MAX_ROWS,
        "corr_max_rows": CORR_MAX_ROWS,
        "random_seed": RANDOM_SEED,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _start_run(source_uri: str, checksum: str, metadata: dict[str, Any]) -> str:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO audit.ingestion_run
                    (source_name, dataset_name, object_uri, checksum, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING run_id
                """,
                (SOURCE_NAME, DATASET_NAME, source_uri, checksum, Json(metadata)),
            )
            return str(cursor.fetchone()[0])


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at = now(), status = %s, row_count = %s,
                    metadata = metadata || %s, error_message = %s
                WHERE run_id = %s
                """,
                (
                    status,
                    row_count,
                    Json(
                        metadata or {},
                        dumps=lambda obj: json.dumps(obj, default=_json_default),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, metadata
                FROM audit.ingestion_run
                WHERE source_name = %s
                  AND dataset_name = %s
                  AND checksum = %s
                  AND status = 'SUCCESS'
                ORDER BY finished_at DESC
                LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    if row is None:
        return None
    return str(row[0]), dict(row[1] or {})


def _safe_feature_name(column: str) -> bool:
    if column in BANNED_EXACT:
        return False
    lowered = column.lower()
    if any(token in lowered for token in BANNED_TOKENS):
        return False
    if "delay" in lowered and not lowered.startswith(
        ("vessel_hist_", "global_hist_")
    ):
        return False
    return True


def _sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=RANDOM_SEED)


def _numeric_values(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _segments(frame: pd.DataFrame):
    yield "ALL", frame
    for horizon in (24, 12, 6, 3):
        subset = frame.loc[frame["horizon_h"] == horizon]
        if not subset.empty:
            yield f"HORIZON_{horizon}H", subset


def _load_inputs(
    source_path: Path,
    split_path: Path,
    config_path: Path,
    decision_path: Path,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
    list[str],
    list[str],
    list[str],
    list[str],
]:
    frame = pd.read_parquet(source_path)
    assignment = pd.read_parquet(split_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    required = {"port_call_id", "horizon_h", "planned_eta", TARGET}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"B54C model-ready dataset is missing: {missing}")
    if not {"port_call_id", "split"}.issubset(assignment.columns):
        raise RuntimeError("B54D split assignment has no port_call_id/split")

    assignment = assignment[["port_call_id", "split"]].drop_duplicates("port_call_id")
    if "split" in frame.columns:
        frame = frame.drop(columns=["split"])
    frame = frame.merge(assignment, on="port_call_id", how="left", validate="many_to_one")
    if frame["split"].isna().any():
        raise RuntimeError("Some B54C rows are absent from the B54D split assignment")

    frame["planned_eta"] = pd.to_datetime(frame["planned_eta"], utc=True, errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame["horizon_h"] = pd.to_numeric(frame["horizon_h"], errors="coerce")
    frame = frame.dropna(subset=["planned_eta", TARGET, "horizon_h"]).copy()
    frame["horizon_h"] = frame["horizon_h"].astype("int16")

    configured = [
        column
        for column in config.get("all_features", [])
        if column in frame.columns and _safe_feature_name(column)
    ]
    if not configured:
        raise RuntimeError("No safe B54D feature is available for dependency audit")

    categorical = [
        column
        for column in config.get("with_wave_categorical", [])
        if column in configured
    ]
    for column in configured:
        if (
            pd.api.types.is_object_dtype(frame[column].dtype)
            or pd.api.types.is_string_dtype(frame[column].dtype)
            or isinstance(frame[column].dtype, pd.CategoricalDtype)
        ) and column not in categorical:
            categorical.append(column)
    numeric = [
        column
        for column in configured
        if column not in categorical
        and not pd.api.types.is_datetime64_any_dtype(frame[column].dtype)
    ]
    wave = [column for column in config.get("wave_features", []) if column in configured]
    return frame, config, decision, configured, numeric, categorical, wave


def _leakage_audit(
    frame: pd.DataFrame,
    configured: list[str],
    numeric: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    duplicate_rows = int(frame.duplicated(["port_call_id", "horizon_h"]).sum())
    split_counts = frame.groupby("port_call_id")["split"].nunique()
    split_overlap_calls = int((split_counts > 1).sum())
    unsafe_features = sorted(column for column in configured if not _safe_feature_name(column))

    split_dates = {}
    for split in ("TRAIN", "VALID", "TEST"):
        values = frame.loc[frame["split"] == split, "planned_eta"]
        split_dates[split] = {
            "min": values.min(),
            "max": values.max(),
            "calls": int(frame.loc[frame["split"] == split, "port_call_id"].nunique()),
            "rows": int((frame["split"] == split).sum()),
        }
    chronology_ok = bool(
        split_dates["TRAIN"]["max"] <= split_dates["VALID"]["min"]
        and split_dates["VALID"]["max"] <= split_dates["TEST"]["min"]
    )

    train = _sample(frame.loc[frame["split"] == "TRAIN"], CORR_MAX_ROWS)
    near_rows = []
    y = train[TARGET]
    for column in numeric:
        x = _numeric_values(train[column])
        mask = x.notna() & y.notna()
        if int(mask.sum()) < MIN_CORR_ROWS or x.loc[mask].nunique() <= 1:
            continue
        value = x.loc[mask].corr(y.loc[mask], method="spearman")
        if pd.notna(value) and abs(float(value)) >= 0.98:
            near_rows.append(
                {"feature": column, "spearman_target_train": float(value)}
            )
    near = pd.DataFrame(near_rows)
    passed = bool(
        duplicate_rows == 0
        and split_overlap_calls == 0
        and not unsafe_features
        and chronology_ok
        and near.empty
    )
    report = {
        "passed": passed,
        "duplicate_port_call_horizon_rows": duplicate_rows,
        "port_calls_in_multiple_splits": split_overlap_calls,
        "unsafe_features": unsafe_features,
        "near_perfect_target_features": near_rows,
        "chronology_ok": chronology_ok,
        "split_dates": split_dates,
    }
    return report, near


def _numeric_dependency(
    frame: pd.DataFrame, numeric: list[str]
) -> pd.DataFrame:
    rows = []
    for split in ("TRAIN", "VALID", "TEST"):
        split_frame = frame.loc[frame["split"] == split]
        for segment, subset_raw in _segments(split_frame):
            subset = _sample(subset_raw, CORR_MAX_ROWS)
            y = subset[TARGET]
            for column in numeric:
                x = _numeric_values(subset[column])
                mask = x.notna() & y.notna()
                n = int(mask.sum())
                if n < MIN_CORR_ROWS or x.loc[mask].nunique() <= 1:
                    continue
                rows.append(
                    {
                        "split": split,
                        "segment": segment,
                        "feature": column,
                        "n": n,
                        "n_unique": int(x.loc[mask].nunique()),
                        "missing_pct": float(100.0 * (1.0 - n / len(subset))),
                        "pearson": float(x.loc[mask].corr(y.loc[mask], method="pearson")),
                        "spearman": float(x.loc[mask].corr(y.loc[mask], method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def _encoded_matrix(
    subset: pd.DataFrame,
    configured: list[str],
    categorical: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[dict[str, Any]]]:
    matrix = pd.DataFrame(index=subset.index)
    discrete = []
    metadata = []
    categorical_set = set(categorical)
    for column in configured:
        if column in categorical_set:
            values = subset[column].astype("string").fillna("MISSING")
            codes, uniques = pd.factorize(values, sort=True)
            matrix[column] = codes.astype("int32")
            discrete.append(True)
            metadata.append(
                {
                    "feature": column,
                    "feature_type": "categorical",
                    "n_unique": int(len(uniques)),
                    "high_cardinality_flag": int(len(uniques) > max(50, len(values) * 0.05)),
                }
            )
        else:
            values = _numeric_values(subset[column])
            median = values.median()
            matrix[column] = values.fillna(0.0 if pd.isna(median) else median).astype("float64")
            discrete.append(bool(pd.api.types.is_bool_dtype(subset[column].dtype)))
            metadata.append(
                {
                    "feature": column,
                    "feature_type": "numeric",
                    "n_unique": int(values.nunique(dropna=True)),
                    "high_cardinality_flag": 0,
                }
            )
    return matrix, np.asarray(discrete, dtype=bool), metadata


def _mutual_information(
    frame: pd.DataFrame,
    configured: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "TRAIN"]
    rows = []
    for segment, subset_raw in _segments(train):
        subset = _sample(subset_raw, MI_MAX_ROWS).copy()
        matrix, discrete, metadata = _encoded_matrix(subset, configured, categorical)
        usable_mask = matrix.nunique(dropna=False).to_numpy() > 1
        usable_columns = matrix.columns[usable_mask].tolist()
        if not usable_columns:
            continue
        usable_matrix = matrix[usable_columns]
        usable_discrete = discrete[usable_mask]
        y = subset[TARGET].to_numpy(dtype=float)
        mi_reg = mutual_info_regression(
            usable_matrix,
            y,
            discrete_features=usable_discrete,
            random_state=RANDOM_SEED,
            n_neighbors=3,
        )
        class_scores = {}
        for threshold in (1, 3, 6):
            binary = (y > threshold).astype("int8")
            if np.unique(binary).size == 2:
                class_scores[threshold] = mutual_info_classif(
                    usable_matrix,
                    binary,
                    discrete_features=usable_discrete,
                    random_state=RANDOM_SEED,
                    n_neighbors=3,
                )
            else:
                class_scores[threshold] = np.full(len(usable_columns), np.nan)
        metadata_by_feature = {item["feature"]: item for item in metadata}
        for index, column in enumerate(usable_columns):
            item = metadata_by_feature[column]
            rows.append(
                {
                    "split": "TRAIN",
                    "segment": segment,
                    "feature": column,
                    "feature_type": item["feature_type"],
                    "n_rows": int(len(subset)),
                    "n_unique": item["n_unique"],
                    "high_cardinality_flag": item["high_cardinality_flag"],
                    "mi_regression": float(mi_reg[index]),
                    "mi_late_gt1h": float(class_scores[1][index]),
                    "mi_late_gt3h": float(class_scores[3][index]),
                    "mi_late_gt6h": float(class_scores[6][index]),
                }
            )
    return pd.DataFrame(rows)


def _correlation_matrix_and_redundancy(
    frame: pd.DataFrame, numeric: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = _sample(frame.loc[frame["split"] == "TRAIN"], CORR_MAX_ROWS)
    matrix = pd.DataFrame(index=train.index)
    for column in numeric:
        values = _numeric_values(train[column])
        if values.nunique(dropna=True) > 1:
            matrix[column] = values
    matrix[TARGET] = train[TARGET]
    corr = matrix.corr(method="spearman", min_periods=MIN_CORR_ROWS)
    rows = []
    feature_columns = [column for column in corr.columns if column != TARGET]
    for left_index, left in enumerate(feature_columns):
        for right in feature_columns[left_index + 1 :]:
            value = corr.at[left, right]
            if pd.notna(value) and abs(float(value)) >= 0.90:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "spearman": float(value),
                        "abs_spearman": abs(float(value)),
                        "left_target_abs": abs(float(corr.at[left, TARGET])),
                        "right_target_abs": abs(float(corr.at[right, TARGET])),
                        "drop_candidate": (
                            left
                            if abs(float(corr.at[left, TARGET]))
                            < abs(float(corr.at[right, TARGET]))
                            else right
                        ),
                    }
                )
    redundancy = pd.DataFrame(rows)
    if not redundancy.empty:
        redundancy = redundancy.sort_values("abs_spearman", ascending=False)
    return corr, redundancy


def _category_effect(
    frame: pd.DataFrame, categorical: list[str]
) -> pd.DataFrame:
    rows = []
    for split in ("TRAIN", "VALID", "TEST"):
        split_frame = frame.loc[frame["split"] == split]
        for segment, subset in _segments(split_frame):
            y = subset[TARGET]
            total_variance = float(((y - y.mean()) ** 2).sum())
            for column in categorical:
                values = subset[column].astype("string").fillna("MISSING")
                counts = values.value_counts()
                kept = counts.loc[counts >= MIN_CATEGORY_ROWS].head(30).index
                reduced = values.where(values.isin(kept), "OTHER")
                work = pd.DataFrame({"category": reduced, "target": y}).dropna()
                if len(work) < MIN_CORR_ROWS or work["category"].nunique() <= 1:
                    continue
                group_stats = work.groupby("category", observed=True)["target"].agg(
                    ["count", "mean"]
                )
                between = float(
                    (
                        group_stats["count"]
                        * (group_stats["mean"] - work["target"].mean()) ** 2
                    ).sum()
                )
                eta_squared = between / total_variance if total_variance > 0 else np.nan
                groups = [
                    group["target"].to_numpy(dtype=float)
                    for _, group in work.groupby("category", observed=True)
                    if len(group) >= MIN_CATEGORY_ROWS
                ]
                statistic = np.nan
                pvalue = np.nan
                if len(groups) >= 2:
                    result = kruskal(*groups, nan_policy="omit")
                    statistic = float(result.statistic)
                    pvalue = float(result.pvalue)
                rows.append(
                    {
                        "split": split,
                        "segment": segment,
                        "feature": column,
                        "n_rows": int(len(work)),
                        "original_levels": int(values.nunique()),
                        "tested_levels": int(work["category"].nunique()),
                        "eta_squared": float(eta_squared),
                        "kruskal_stat": statistic,
                        "kruskal_pvalue": pvalue,
                    }
                )
    return pd.DataFrame(rows)


def _sea_state_effect(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    candidates = [
        "wave_height_m",
        "wave_height_now_m",
        "wave_height_mean_3h",
        "wave_height_mean_6h",
    ]
    height_column = next((column for column in candidates if column in frame.columns), None)
    if height_column is None:
        return pd.DataFrame(), None
    work = frame.copy()
    height = _numeric_values(work[height_column])
    work["sea_state"] = pd.cut(
        height,
        bins=SEA_BINS,
        labels=SEA_LABELS,
        right=False,
        ordered=True,
    )
    rows = []
    for split in ("TRAIN", "VALID", "TEST"):
        split_frame = work.loc[work["split"] == split]
        for segment, subset in _segments(split_frame):
            for regime, group in subset.groupby("sea_state", observed=True):
                y = group[TARGET]
                rows.append(
                    {
                        "split": split,
                        "segment": segment,
                        "sea_state": str(regime),
                        "height_column": height_column,
                        "n_rows": int(len(group)),
                        "n_calls": int(group["port_call_id"].nunique()),
                        "wave_height_mean_m": float(_numeric_values(group[height_column]).mean()),
                        "delay_mean_h": float(y.mean()),
                        "delay_median_h": float(y.median()),
                        "delay_p95_h": float(y.quantile(0.95)),
                        "late_gt1h_pct": float(100.0 * (y > 1).mean()),
                        "late_gt3h_pct": float(100.0 * (y > 3).mean()),
                        "late_gt6h_pct": float(100.0 * (y > 6).mean()),
                    }
                )
    return pd.DataFrame(rows), height_column


def _within_vessel_weather(
    frame: pd.DataFrame, wave: list[str], numeric: list[str]
) -> pd.DataFrame:
    vessel_column = next(
        (column for column in ("imo", "vessel_name") if column in frame.columns), None
    )
    if vessel_column is None:
        return pd.DataFrame()
    preferred_tokens = ("height", "energy", "period", "trend", "max", "mean")
    weather_features = [
        column
        for column in wave
        if column in numeric and any(token in column.lower() for token in preferred_tokens)
    ][:24]
    rows = []
    for split in ("TRAIN", "VALID", "TEST"):
        subset = frame.loc[frame["split"] == split].copy()
        groups = [vessel_column, "horizon_h"]
        y = subset[TARGET]
        y_within = y - subset.groupby(groups, dropna=False)[TARGET].transform("mean")
        for column in weather_features:
            x = _numeric_values(subset[column])
            subset["_x_temp"] = x
            x_within = x - subset.groupby(groups, dropna=False)["_x_temp"].transform("mean")
            mask = x_within.notna() & y_within.notna()
            if int(mask.sum()) < MIN_CORR_ROWS or x_within.loc[mask].nunique() <= 1:
                continue
            rows.append(
                {
                    "split": split,
                    "vessel_key": vessel_column,
                    "feature": column,
                    "n_rows": int(mask.sum()),
                    "n_vessels": int(subset.loc[mask, vessel_column].nunique()),
                    "within_vessel_pearson": float(
                        x_within.loc[mask].corr(y_within.loc[mask], method="pearson")
                    ),
                    "within_vessel_spearman": float(
                        x_within.loc[mask].corr(y_within.loc[mask], method="spearman")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _stability(numeric_dependency: pd.DataFrame) -> pd.DataFrame:
    if numeric_dependency.empty:
        return pd.DataFrame()
    pivot = numeric_dependency.pivot_table(
        index=["segment", "feature"],
        columns="split",
        values="spearman",
        aggfunc="first",
    ).reset_index()
    for split in ("TRAIN", "VALID", "TEST"):
        if split not in pivot.columns:
            pivot[split] = np.nan
    values = pivot[["TRAIN", "VALID", "TEST"]]
    pivot["mean_abs_spearman"] = values.abs().mean(axis=1)
    pivot["std_spearman"] = values.std(axis=1)
    pivot["range_spearman"] = values.max(axis=1) - values.min(axis=1)
    pivot["train_valid_delta"] = pivot["VALID"] - pivot["TRAIN"]
    pivot["train_test_delta_diagnostic_only"] = pivot["TEST"] - pivot["TRAIN"]
    signs = np.sign(values.fillna(0.0))
    pivot["sign_consistent"] = (
        ((signs >= 0).all(axis=1)) | ((signs <= 0).all(axis=1))
    ).astype("int8")
    pivot["stability_score_train_valid"] = (
        (pivot["TRAIN"].abs() + pivot["VALID"].abs()) / 2.0
        - pivot["train_valid_delta"].abs()
    )
    return pivot.sort_values(
        ["segment", "stability_score_train_valid"], ascending=[True, False]
    )


def _residual_dependency(
    frame: pd.DataFrame,
    numeric: list[str],
    decision: dict[str, Any],
    valid_predictions_path: Path,
    test_predictions_path: Path,
) -> pd.DataFrame:
    official = str(decision.get("official_model", "CATBOOST_WITH_WAVE"))
    prediction_column = f"PRED_{official}"
    outputs = []
    for split, path in (("VALID", valid_predictions_path), ("TEST", test_predictions_path)):
        predictions = pd.read_parquet(path)
        if prediction_column not in predictions.columns:
            continue
        keys = ["port_call_id", "horizon_h"]
        selected = predictions[keys + [prediction_column]].drop_duplicates(keys)
        subset = frame.loc[frame["split"] == split].merge(
            selected, on=keys, how="inner", validate="one_to_one"
        )
        subset["residual_actual_minus_prediction_h"] = (
            subset[TARGET] - pd.to_numeric(subset[prediction_column], errors="coerce")
        )
        for segment, part_raw in _segments(subset):
            part = _sample(part_raw, CORR_MAX_ROWS)
            residual = part["residual_actual_minus_prediction_h"]
            for column in numeric:
                x = _numeric_values(part[column])
                mask = x.notna() & residual.notna()
                if int(mask.sum()) < MIN_CORR_ROWS or x.loc[mask].nunique() <= 1:
                    continue
                outputs.append(
                    {
                        "split": split,
                        "segment": segment,
                        "official_model": official,
                        "feature": column,
                        "n": int(mask.sum()),
                        "pearson_residual": float(
                            x.loc[mask].corr(residual.loc[mask], method="pearson")
                        ),
                        "spearman_residual": float(
                            x.loc[mask].corr(residual.loc[mask], method="spearman")
                        ),
                    }
                )
    return pd.DataFrame(outputs)


def _summary(
    frame: pd.DataFrame,
    numeric_dependency: pd.DataFrame,
    mutual_information: pd.DataFrame,
    stability: pd.DataFrame,
    sea_state: pd.DataFrame,
    residual: pd.DataFrame,
    leakage: dict[str, Any],
    wave: list[str],
) -> dict[str, Any]:
    train_all = numeric_dependency.loc[
        (numeric_dependency["split"] == "TRAIN")
        & (numeric_dependency["segment"] == "ALL")
    ].copy()
    train_all["abs_spearman"] = train_all["spearman"].abs()
    top_numeric = train_all.nlargest(15, "abs_spearman")[
        ["feature", "pearson", "spearman"]
    ].to_dict("records")

    mi_all = mutual_information.loc[mutual_information["segment"] == "ALL"].copy()
    top_mi = mi_all.nlargest(15, "mi_regression")[
        [
            "feature",
            "feature_type",
            "mi_regression",
            "mi_late_gt1h",
            "mi_late_gt3h",
            "mi_late_gt6h",
        ]
    ].to_dict("records")

    stable_all = stability.loc[stability["segment"] == "ALL"].copy()
    stable_all = stable_all.loc[stable_all["sign_consistent"] == 1]
    top_stable = stable_all.nlargest(15, "stability_score_train_valid")[
        [
            "feature",
            "TRAIN",
            "VALID",
            "TEST",
            "stability_score_train_valid",
        ]
    ].to_dict("records")

    wave_set = set(wave)
    wave_corr = train_all.loc[train_all["feature"].isin(wave_set)]
    wave_mi = mi_all.loc[mi_all["feature"].isin(wave_set)]
    max_wave_corr = float(wave_corr["abs_spearman"].max()) if not wave_corr.empty else None
    max_wave_mi = float(wave_mi["mi_regression"].max()) if not wave_mi.empty else None

    residual_top = []
    if not residual.empty:
        diagnostic = residual.loc[
            (residual["split"] == "TEST") & (residual["segment"] == "ALL")
        ].copy()
        diagnostic["abs_spearman_residual"] = diagnostic["spearman_residual"].abs()
        residual_top = diagnostic.nlargest(15, "abs_spearman_residual")[
            ["feature", "spearman_residual", "pearson_residual"]
        ].to_dict("records")

    sea_counts = []
    if not sea_state.empty:
        sea_counts = sea_state.loc[
            (sea_state["split"] == "TRAIN") & (sea_state["segment"] == "ALL")
        ][
            ["sea_state", "n_calls", "wave_height_mean_m", "delay_mean_h", "late_gt1h_pct"]
        ].to_dict("records")

    return {
        "audit_version": AUDIT_VERSION,
        "status": "READY_FOR_B54E" if leakage["passed"] else "BLOCKED_BY_LEAKAGE_GATE",
        "selection_scope": "TRAIN_WITH_VALID_STABILITY_TEST_DIAGNOSTIC_ONLY",
        "rows": int(len(frame)),
        "port_calls": int(frame["port_call_id"].nunique()),
        "features_audited": int(len(set(numeric_dependency["feature"]))),
        "wave_features_audited": int(len(wave)),
        "leakage_gate": leakage,
        "top_numeric_train": top_numeric,
        "top_mutual_information_train": top_mi,
        "top_stable_train_valid": top_stable,
        "top_test_residual_dependencies_diagnostic_only": residual_top,
        "weather_signal": {
            "max_abs_spearman_train": max_wave_corr,
            "max_mutual_information_train": max_wave_mi,
            "sea_state_train": sea_counts,
            "interpretation": (
                "WEATHER_SIGNAL_PRESENT_BUT_CONDITIONAL"
                if (max_wave_corr or 0.0) >= 0.03 or (max_wave_mi or 0.0) > 0.0
                else "WEATHER_SIGNAL_WEAK_IN_CURRENT_POINT_DATA"
            ),
        },
        "recommended_next_block": (
            "B54E_PROBABILISTIC_HORIZON_EXPERTS_WITH_SEA_STATE"
            if leakage["passed"]
            else "FIX_B54EA_LEAKAGE_OR_SPLIT"
        ),
        "generated_at_utc": datetime.now(timezone.utc),
    }


def run_b54ea_dependency_audit(
    source_bucket: str,
    source_key: str,
    artifacts_bucket: str,
    split_key: str,
    feature_config_key: str,
    decision_key: str,
    valid_predictions_key: str,
    test_predictions_key: str,
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    objects = [
        (source_bucket, source_key),
        (artifacts_bucket, split_key),
        (artifacts_bucket, feature_config_key),
        (artifacts_bucket, decision_key),
        (artifacts_bucket, valid_predictions_key),
        (artifacts_bucket, test_predictions_key),
    ]
    checksum = _signature(client, objects)
    source_uri = f"s3://{source_bucket}/{source_key}"
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            run_id, metadata = previous
            return {
                "status": "SKIPPED_ALREADY_PROCESSED",
                "run_id": run_id,
                "checksum": checksum,
                "results": metadata,
                "outputs": metadata.get("output_uris", {}),
            }

    run_id = _start_run(
        source_uri,
        checksum,
        {
            "audit_version": AUDIT_VERSION,
            "selection_scope": "TRAIN_WITH_VALID_STABILITY_TEST_DIAGNOSTIC_ONLY",
            "input_objects": [f"s3://{bucket}/{key}" for bucket, key in objects],
        },
    )
    try:
        with tempfile.TemporaryDirectory(prefix="b54ea-") as temp_dir:
            work = Path(temp_dir)
            paths = {
                "source": work / "source.parquet",
                "split": work / "split.parquet",
                "config": work / "feature_config.json",
                "decision": work / "decision.json",
                "valid_predictions": work / "valid_predictions.parquet",
                "test_predictions": work / "test_predictions.parquet",
            }
            for (bucket, key), path in zip(objects, paths.values()):
                _download(client, bucket, key, path)

            frame, config, decision, configured, numeric, categorical, wave = _load_inputs(
                paths["source"], paths["split"], paths["config"], paths["decision"]
            )
            leakage, near_perfect = _leakage_audit(frame, configured, numeric)
            numeric_dependency = _numeric_dependency(frame, numeric)
            mutual_information = _mutual_information(frame, configured, categorical)
            correlation_matrix, redundancy = _correlation_matrix_and_redundancy(
                frame, numeric
            )
            category_effect = _category_effect(frame, categorical)
            sea_state, sea_height_column = _sea_state_effect(frame)
            within_vessel = _within_vessel_weather(frame, wave, numeric)
            stability = _stability(numeric_dependency)
            residual = _residual_dependency(
                frame,
                numeric,
                decision,
                paths["valid_predictions"],
                paths["test_predictions"],
            )
            summary = _summary(
                frame,
                numeric_dependency,
                mutual_information,
                stability,
                sea_state,
                residual,
                leakage,
                wave,
            )
            summary["sea_height_column"] = sea_height_column
            summary["categorical_features"] = categorical
            summary["numeric_features"] = numeric
            summary["b54d_official_model"] = decision.get("official_model")

            reports: dict[str, tuple[Path, str, str]] = {}

            def save_csv(name: str, frame_to_save: pd.DataFrame, key_name: str) -> None:
                path = work / name
                frame_to_save.to_csv(path, index=False)
                reports[key_name] = (
                    path,
                    f"reports/b54ea/{output_prefix.strip('/') or 'version=1'}/{name}",
                    "text/csv",
                )

            save_csv("01_numeric_target_dependency.csv", numeric_dependency, "numeric_dependency")
            save_csv("02_mutual_information.csv", mutual_information, "mutual_information")
            correlation_path = work / "03_spearman_correlation_matrix.csv"
            correlation_matrix.to_csv(correlation_path, index=True)
            reports["correlation_matrix"] = (
                correlation_path,
                f"reports/b54ea/{output_prefix.strip('/') or 'version=1'}/03_spearman_correlation_matrix.csv",
                "text/csv",
            )
            save_csv("04_redundancy_pairs_ge_0p90.csv", redundancy, "redundancy")
            save_csv("05_categorical_effects.csv", category_effect, "categorical_effects")
            save_csv("06_sea_state_effects.csv", sea_state, "sea_state_effects")
            save_csv("07_within_vessel_weather_effects.csv", within_vessel, "within_vessel")
            save_csv("08_temporal_stability.csv", stability, "temporal_stability")
            save_csv("09_official_model_residual_dependency.csv", residual, "residual_dependency")
            save_csv("10_near_perfect_target_features.csv", near_perfect, "near_perfect")

            summary_path = work / "b54ea_dependency_audit_summary.json"
            summary_path.write_text(
                json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
            )
            reports["summary"] = (
                summary_path,
                f"configs/b54ea/{output_prefix.strip('/') or 'version=1'}/b54ea_dependency_audit_summary.json",
                "application/json",
            )
            output_uris = {
                name: _upload(client, path, output_bucket, key, content_type)
                for name, (path, key, content_type) in reports.items()
            }

        results = {
            "audit_version": AUDIT_VERSION,
            "source_rows": int(len(frame)),
            "source_calls": int(frame["port_call_id"].nunique()),
            "feature_count": int(len(configured)),
            "numeric_feature_count": int(len(numeric)),
            "categorical_feature_count": int(len(categorical)),
            "wave_feature_count": int(len(wave)),
            "leakage_gate": leakage,
            "decision": summary,
            "output_uris": output_uris,
            "next_block": summary["recommended_next_block"],
        }
        _finish_run(run_id, "SUCCESS", row_count=len(frame), metadata=results)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "checksum": checksum,
            "results": results,
            "outputs": output_uris,
        }
    except Exception as exc:
        _finish_run(run_id, "FAILED", error_message=str(exc))
        raise
