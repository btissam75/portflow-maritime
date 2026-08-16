from __future__ import annotations

import hashlib
import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from psycopg2.extras import Json
from sklearn.feature_selection import mutual_info_regression


AUDIT_VERSION = "b54fb-one-row-audit-v1"
SOURCE_NAME = "b54fa_one_row_gold"
DATASET_NAME = "b54fb_structure_dependency_audit"
TARGET_COLUMN = "target_arrival_delay_h"
MIN_DEPENDENCY_ROWS = 200
REDUNDANCY_THRESHOLD = 0.90
FORBIDDEN_FEATURE_TOKENS = (
    "actual_",
    "target_",
    "arrival_delay",
    "departure_delay",
    "label",
    "arrived_before",
    "exclusion_reason",
    "model_ready_flag",
    "split",
    "snapshot",
    "horizon_h",
)


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os_environ("SMART_PORT_S3_ENDPOINT"),
        aws_access_key_id=os_environ("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os_environ("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1",
    )


def os_environ(name: str) -> str:
    import os

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _db_connection():
    import os

    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _download(client, bucket: str, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))


def _upload(
    client,
    source: Path,
    bucket: str,
    key: str,
    content_type: str,
) -> str:
    client.upload_file(
        str(source),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{bucket}/{key}"


def _signature(client, objects: list[tuple[str, str]]) -> str:
    payload = []
    for bucket, key in objects:
        metadata = client.head_object(Bucket=bucket, Key=key)
        payload.append(
            {
                "bucket": bucket,
                "key": key,
                "etag": metadata["ETag"].strip('"'),
                "size": int(metadata["ContentLength"]),
            }
        )
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
                WHERE source_name = %s AND dataset_name = %s
                  AND checksum = %s AND status = 'SUCCESS'
                ORDER BY finished_at DESC LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    if row is None:
        return None
    return str(row[0]), dict(row[1] or {})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric(series: pd.Series) -> pd.Series:
    if is_bool_dtype(series.dtype):
        return series.astype("float64")
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _schema_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = max(1, len(frame))
    for column in frame.columns:
        series = frame[column]
        numeric = _numeric(series) if is_numeric_dtype(series.dtype) or is_bool_dtype(series.dtype) else None
        row = {
            "column": column,
            "dtype": str(series.dtype),
            "rows": int(len(series)),
            "missing_count": int(series.isna().sum()),
            "missing_pct": 100.0 * float(series.isna().sum()) / n,
            "n_unique": int(series.nunique(dropna=True)),
            "constant_flag": bool(series.nunique(dropna=True) <= 1),
            "infinite_count": 0,
            "min": None,
            "p01": None,
            "p50": None,
            "p99": None,
            "max": None,
        }
        if numeric is not None:
            finite = numeric.replace([np.inf, -np.inf], np.nan).dropna()
            row["infinite_count"] = int(np.isinf(numeric.dropna()).sum())
            if not finite.empty:
                row.update(
                    {
                        "min": float(finite.min()),
                        "p01": float(finite.quantile(0.01)),
                        "p50": float(finite.median()),
                        "p99": float(finite.quantile(0.99)),
                        "max": float(finite.max()),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _target_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    y = _numeric(frame[TARGET_COLUMN])
    prediction_at = pd.to_datetime(frame["prediction_at"], errors="coerce", utc=True)
    working = pd.DataFrame({"target": y, "year": prediction_at.dt.year})
    groups: list[tuple[str, pd.Series]] = [("ALL", working["target"])]
    for year, group in working.groupby("year", dropna=True):
        groups.append((f"YEAR_{int(year)}", group["target"]))
    rows = []
    for segment, values in groups:
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        rows.append(
            {
                "segment": segment,
                "n": int(len(values)),
                "mean_h": float(values.mean()),
                "std_h": float(values.std()),
                "min_h": float(values.min()),
                "p01_h": float(values.quantile(0.01)),
                "p05_h": float(values.quantile(0.05)),
                "p50_h": float(values.median()),
                "p95_h": float(values.quantile(0.95)),
                "p99_h": float(values.quantile(0.99)),
                "max_h": float(values.max()),
                "early_gt1h_pct": 100.0 * float((values < -1).mean()),
                "within_1h_pct": 100.0 * float((values.abs() <= 1).mean()),
                "late_gt1h_pct": 100.0 * float((values > 1).mean()),
                "late_gt3h_pct": 100.0 * float((values > 3).mean()),
                "late_gt6h_pct": 100.0 * float((values > 6).mean()),
            }
        )
    return pd.DataFrame(rows)


def _numeric_dependency(
    frame: pd.DataFrame,
    numeric_features: list[str],
) -> pd.DataFrame:
    y = _numeric(frame[TARGET_COLUMN])
    prediction_at = pd.to_datetime(frame["prediction_at"], errors="coerce", utc=True)
    segments: list[tuple[str, pd.Series]] = [("ALL", pd.Series(True, index=frame.index))]
    for year in sorted(prediction_at.dt.year.dropna().unique()):
        segments.append((f"YEAR_{int(year)}", prediction_at.dt.year == year))
    rows = []
    for segment, segment_mask in segments:
        for column in numeric_features:
            x = _numeric(frame[column])
            mask = segment_mask & x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
            n = int(mask.sum())
            if n < MIN_DEPENDENCY_ROWS or x.loc[mask].nunique() <= 1:
                continue
            rows.append(
                {
                    "segment": segment,
                    "feature": column,
                    "n": n,
                    "pearson": float(x.loc[mask].corr(y.loc[mask], method="pearson")),
                    "spearman": float(x.loc[mask].corr(y.loc[mask], method="spearman")),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["abs_pearson"] = result["pearson"].abs()
        result["abs_spearman"] = result["spearman"].abs()
        result = result.sort_values(["segment", "abs_spearman"], ascending=[True, False])
    return result


def _encoded_matrix(
    frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    matrix = pd.DataFrame(index=frame.index)
    discrete = []
    categorical_set = set(categorical_features)
    for column in features:
        if column in categorical_set:
            values = frame[column].astype("string").fillna("MISSING")
            matrix[column] = pd.factorize(values, sort=True)[0].astype("int32")
            discrete.append(True)
        else:
            values = _numeric(frame[column]).replace([np.inf, -np.inf], np.nan)
            median = values.median()
            matrix[column] = values.fillna(0.0 if pd.isna(median) else median)
            discrete.append(False)
    return matrix, np.asarray(discrete, dtype=bool)


def _mutual_information(
    frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    subset = frame.loc[frame[TARGET_COLUMN].notna(), features + [TARGET_COLUMN]].copy()
    if len(subset) > 50000:
        subset = subset.sample(50000, random_state=54)
    matrix, discrete = _encoded_matrix(subset, features, categorical_features)
    y = _numeric(subset[TARGET_COLUMN]).to_numpy()
    variable_mask = matrix.nunique(dropna=False).to_numpy() > 1
    scores = np.zeros(len(features), dtype=float)
    if variable_mask.any():
        scores[variable_mask] = mutual_info_regression(
            matrix.loc[:, variable_mask].to_numpy(),
            y,
            discrete_features=discrete[variable_mask],
            random_state=54,
            n_neighbors=5,
        )
    result = pd.DataFrame(
        {
            "feature": features,
            "feature_type": [
                "categorical" if item in set(categorical_features) else "numeric"
                for item in features
            ],
            "mutual_information": scores,
            "sample_rows": int(len(subset)),
        }
    )
    return result.sort_values("mutual_information", ascending=False)


def _correlation_matrices(
    frame: pd.DataFrame,
    numeric_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = pd.DataFrame(
        {column: _numeric(frame[column]) for column in numeric_features}
    )
    matrix[TARGET_COLUMN] = _numeric(frame[TARGET_COLUMN])
    pearson = matrix.corr(method="pearson", min_periods=MIN_DEPENDENCY_ROWS)
    spearman = matrix.corr(method="spearman", min_periods=MIN_DEPENDENCY_ROWS)
    rows = []
    columns = list(spearman.columns)
    for index, left in enumerate(columns):
        if left == TARGET_COLUMN:
            continue
        for right in columns[index + 1 :]:
            if right == TARGET_COLUMN:
                continue
            value = spearman.loc[left, right]
            if pd.notna(value) and abs(float(value)) >= REDUNDANCY_THRESHOLD:
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "spearman": float(value),
                        "abs_spearman": abs(float(value)),
                    }
                )
    redundancy = pd.DataFrame(rows)
    if not redundancy.empty:
        redundancy = redundancy.sort_values("abs_spearman", ascending=False)
    return pearson, spearman, redundancy


def _eta_squared(categories: pd.Series, target: pd.Series) -> float:
    work = pd.DataFrame(
        {"category": categories.astype("string").fillna("MISSING"), "target": target}
    ).dropna()
    if work.empty or work["category"].nunique() <= 1:
        return 0.0
    overall = float(work["target"].mean())
    total = float(((work["target"] - overall) ** 2).sum())
    if total <= 0:
        return 0.0
    between = 0.0
    for _, group in work.groupby("category", observed=True):
        between += len(group) * (float(group["target"].mean()) - overall) ** 2
    return float(between / total)


def _categorical_target_association(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    y = _numeric(frame[TARGET_COLUMN])
    rows = []
    for column in categorical_features:
        values = frame[column].astype("string").fillna("MISSING")
        work = pd.DataFrame({"category": values, "target": y}).dropna()
        if len(work) < MIN_DEPENDENCY_ROWS:
            continue
        group_stats = work.groupby("category", observed=True)["target"].agg(
            ["size", "mean", "median", "std"]
        )
        rows.append(
            {
                "feature": column,
                "n": int(len(work)),
                "levels": int(work["category"].nunique()),
                "eta_squared": _eta_squared(work["category"], work["target"]),
                "largest_level_rows": int(group_stats["size"].max()),
                "smallest_level_rows": int(group_stats["size"].min()),
                "between_level_mean_range_h": float(
                    group_stats["mean"].max() - group_stats["mean"].min()
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("eta_squared", ascending=False)
    return result


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(
        left.astype("string").fillna("MISSING"),
        right.astype("string").fillna("MISSING"),
    ).to_numpy(dtype=float)
    n = table.sum()
    if n <= 0 or min(table.shape) <= 1:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    valid = expected > 0
    chi2 = float((((table - expected) ** 2) / np.where(valid, expected, 1.0))[valid].sum())
    phi2 = chi2 / n
    rows, columns = table.shape
    corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / max(1.0, n - 1))
    denom = min(
        columns - 1 - ((columns - 1) ** 2) / max(1.0, n - 1),
        rows - 1 - ((rows - 1) ** 2) / max(1.0, n - 1),
    )
    return math.sqrt(corrected / denom) if denom > 0 else 0.0


def _categorical_redundancy(
    frame: pd.DataFrame,
    categorical_features: list[str],
) -> pd.DataFrame:
    rows = []
    for index, left in enumerate(categorical_features):
        for right in categorical_features[index + 1 :]:
            rows.append(
                {
                    "feature_left": left,
                    "feature_right": right,
                    "cramers_v": _cramers_v(frame[left], frame[right]),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("cramers_v", ascending=False)
    return result


def _temporal_stability(dependency: pd.DataFrame) -> pd.DataFrame:
    if dependency.empty:
        return pd.DataFrame()
    yearly = dependency.loc[dependency["segment"].str.startswith("YEAR_")].copy()
    if yearly.empty:
        return pd.DataFrame()
    rows = []
    for feature, group in yearly.groupby("feature"):
        values = group["spearman"].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "feature": feature,
                "years": int(len(values)),
                "mean_spearman": float(values.mean()),
                "mean_abs_spearman": float(values.abs().mean()),
                "std_spearman": float(values.std()) if len(values) > 1 else 0.0,
                "min_spearman": float(values.min()),
                "max_spearman": float(values.max()),
                "sign_changes": int(
                    ((np.sign(values.to_numpy()[1:]) != np.sign(values.to_numpy()[:-1])).sum())
                ),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("mean_abs_spearman", ascending=False)
    return result


def _leakage_audit(
    full: pd.DataFrame,
    model_ready: pd.DataFrame,
    features: list[str],
    build_report: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, violations: int, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "violations": int(violations),
                "passed": int(violations) == 0,
                "detail": detail,
            }
        )

    add(
        "ONE_ROW_PER_PORT_CALL",
        int(model_ready["port_call_id"].duplicated().sum()),
        "port_call_id must be unique in model-ready data",
    )
    forbidden = [
        column
        for column in features
        if any(token in column.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    add("FORBIDDEN_FEATURE_NAMES", len(forbidden), ",".join(forbidden))
    missing_features = [column for column in features if column not in model_ready.columns]
    add("CONFIGURED_FEATURES_EXIST", len(missing_features), ",".join(missing_features))
    split_columns = [
        column for column in model_ready.columns if "split" in column.lower()
    ]
    add("NO_SPLIT_COLUMNS", len(split_columns), ",".join(split_columns))
    snapshot_columns = [
        column
        for column in model_ready.columns
        if "snapshot" in column.lower() or column.lower() == "horizon_h"
    ]
    add("NO_SNAPSHOT_OR_HORIZON_COLUMNS", len(snapshot_columns), ",".join(snapshot_columns))
    add(
        "TARGET_COMPLETE",
        int(model_ready[TARGET_COLUMN].isna().sum()),
        "target must be present in model-ready data",
    )

    full_prediction = pd.to_datetime(full.get("prediction_at"), errors="coerce", utc=True)
    full_observed = pd.to_datetime(full.get("observed_at"), errors="coerce", utc=True)
    future_sea = int((full_observed.notna() & (full_observed > full_prediction)).sum())
    add("SEA_AVAILABLE_AT_CUTOFF", future_sea, "observed_at must be <= prediction_at")

    for column in ("vessel_history_event_time", "global_history_event_time"):
        event = pd.to_datetime(full.get(column), errors="coerce", utc=True)
        violations = int((event.notna() & (event >= full_prediction)).sum())
        add(
            f"{column.upper()}_STRICTLY_PAST",
            violations,
            f"{column} must be < prediction_at",
        )

    arrived_in_ready = 0
    if "arrived_before_cutoff_flag" in full.columns and "model_ready_flag" in full.columns:
        arrived_in_ready = int(
            (
                full["arrived_before_cutoff_flag"].fillna(False)
                & full["model_ready_flag"].fillna(False)
            ).sum()
        )
    add(
        "ARRIVED_BEFORE_CUTOFF_QUARANTINED",
        arrived_in_ready,
        "already-arrived calls cannot enter the 24h model-ready dataset",
    )
    add(
        "BUILDER_REPORTED_ZERO_LEAKAGE",
        int(build_report.get("temporal_leakage_violations", -1)),
        "B54F-A report must contain zero leakage violations",
    )

    audit = pd.DataFrame(checks)
    summary = {
        "passed": bool(audit["passed"].all()),
        "checks": int(len(audit)),
        "violations": int(audit["violations"].sum()),
        "failed_checks": audit.loc[~audit["passed"], "check"].tolist(),
        "forbidden_features": forbidden,
    }
    return audit, summary


def _decision(
    frame: pd.DataFrame,
    leakage: dict[str, Any],
    dependency: pd.DataFrame,
    mutual_information: pd.DataFrame,
    schema: pd.DataFrame,
) -> dict[str, Any]:
    target = _numeric(frame[TARGET_COLUMN]).dropna()
    all_dependency = dependency.loc[dependency["segment"] == "ALL"] if not dependency.empty else dependency
    max_spearman = (
        float(all_dependency["spearman"].abs().max())
        if not all_dependency.empty
        else 0.0
    )
    max_mi = (
        float(mutual_information["mutual_information"].max())
        if not mutual_information.empty
        else 0.0
    )
    infinite_total = int(schema["infinite_count"].sum())
    if not leakage["passed"] or infinite_total > 0 or len(frame) < 5000:
        status = "NEED_DATA_REPAIR"
    elif target.nunique() < 10 or float(target.std()) < 0.05:
        status = "TARGET_NOT_PREDICTABLE"
    else:
        status = "READY_FOR_TEMPORAL_SPLIT"
    return {
        "status": status,
        "target_rows": int(len(target)),
        "target_unique": int(target.nunique()),
        "target_std_h": float(target.std()),
        "max_abs_spearman": max_spearman,
        "max_mutual_information": max_mi,
        "signal_assessment": (
            "WEAK_MARGINAL_SIGNAL"
            if max_spearman < 0.15 and max_mi < 0.20
            else "MEASURABLE_SIGNAL"
        ),
        "split_created": False,
        "training_executed": False,
        "next_block": (
            "B54F_C_TEMPORAL_SPLIT_AND_BASELINES"
            if status == "READY_FOR_TEMPORAL_SPLIT"
            else "B54F_DATA_REPAIR"
        ),
    }


def run_b54fb_one_row_audit(
    source_bucket: str,
    full_key: str,
    model_ready_key: str,
    feature_config_key: str,
    build_report_key: str,
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    objects = [
        (source_bucket, full_key),
        (source_bucket, model_ready_key),
        (source_bucket, feature_config_key),
        (source_bucket, build_report_key),
    ]
    checksum = _signature(client, objects)
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

    source_uri = f"s3://{source_bucket}/{model_ready_key}"
    run_id = _start_run(
        source_uri,
        checksum,
        {"audit_version": AUDIT_VERSION, "source_objects": objects},
    )
    try:
        with tempfile.TemporaryDirectory(prefix="b54f-b-") as temporary:
            work = Path(temporary)
            paths = {
                "full": work / "full.parquet",
                "model_ready": work / "model_ready.parquet",
                "config": work / "config.json",
                "build_report": work / "build_report.json",
            }
            _download(client, source_bucket, full_key, paths["full"])
            _download(client, source_bucket, model_ready_key, paths["model_ready"])
            _download(client, source_bucket, feature_config_key, paths["config"])
            _download(client, source_bucket, build_report_key, paths["build_report"])
            full = pd.read_parquet(paths["full"])
            frame = pd.read_parquet(paths["model_ready"])
            config = _read_json(paths["config"])
            build_report = _read_json(paths["build_report"])

            features = [column for column in config["feature_columns"] if column in frame.columns]
            categorical = [
                column for column in config.get("categorical_features", []) if column in features
            ]
            numeric = [column for column in features if column not in categorical]

            schema = _schema_audit(frame)
            target_distribution = _target_distribution(frame)
            dependency = _numeric_dependency(frame, numeric)
            mutual_information = _mutual_information(frame, features, categorical)
            pearson_matrix, spearman_matrix, redundancy = _correlation_matrices(frame, numeric)
            categorical_effects = _categorical_target_association(frame, categorical)
            categorical_redundancy = _categorical_redundancy(frame, categorical)
            stability = _temporal_stability(dependency)
            leakage_report, leakage_summary = _leakage_audit(
                full, frame, features, build_report
            )
            decision = _decision(
                frame, leakage_summary, dependency, mutual_information, schema
            )

            prefix = output_prefix.strip("/") or "version=1"
            report_prefix = f"reports/b54f/{prefix}"
            outputs: dict[str, str] = {}

            def save_csv(name: str, data: pd.DataFrame, label: str) -> None:
                path = work / name
                data.to_csv(path, index=True if "matrix" in label else False)
                outputs[label] = _upload(
                    client,
                    path,
                    output_bucket,
                    f"{report_prefix}/{name}",
                    "text/csv",
                )

            save_csv("01_schema_audit.csv", schema, "schema_audit")
            save_csv("02_target_distribution.csv", target_distribution, "target_distribution")
            save_csv("03_numeric_target_dependencies.csv", dependency, "numeric_dependencies")
            save_csv("04_mutual_information.csv", mutual_information, "mutual_information")
            save_csv("05_pearson_matrix.csv", pearson_matrix, "pearson_matrix")
            save_csv("06_spearman_matrix.csv", spearman_matrix, "spearman_matrix")
            save_csv("07_numeric_redundancy.csv", redundancy, "numeric_redundancy")
            save_csv("08_categorical_target_associations.csv", categorical_effects, "categorical_associations")
            save_csv("09_categorical_cramers_v.csv", categorical_redundancy, "categorical_cramers_v")
            save_csv("10_temporal_stability.csv", stability, "temporal_stability")
            save_csv("11_leakage_audit.csv", leakage_report, "leakage_audit")

            summary = {
                "audit_version": AUDIT_VERSION,
                "analysis_policy": (
                    "FULL_NO_SPLIT_DESCRIPTIVE_AUDIT_ONLY. Correlations and mutual "
                    "information must not be used to select final model features "
                    "before the later temporal split."
                ),
                "source_rows": int(len(frame)),
                "source_calls": int(frame["port_call_id"].nunique()),
                "full_rows": int(len(full)),
                "feature_count": int(len(features)),
                "numeric_feature_count": int(len(numeric)),
                "categorical_feature_count": int(len(categorical)),
                "wave_feature_count": int(len(config.get("wave_features", []))),
                "leakage_gate": leakage_summary,
                "decision": decision,
                "build_decision": build_report.get("decision", {}),
                "split_created": False,
                "training_executed": False,
                "output_uris": outputs,
                "generated_at_utc": datetime.now(timezone.utc),
            }
            summary_path = work / "12_b54fb_decision.json"
            summary_path.write_text(
                json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
            )
            outputs["decision"] = _upload(
                client,
                summary_path,
                output_bucket,
                f"configs/b54f/{prefix}/b54fb_decision_v1.json",
                "application/json",
            )
            summary["output_uris"] = outputs

        _finish_run(run_id, "SUCCESS", len(frame), summary)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "checksum": checksum,
            "results": summary,
            "outputs": outputs,
        }
    except Exception as exc:
        _finish_run(run_id, "FAILED", error_message=str(exc)[:4000])
        raise
