from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import boto3
import matplotlib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from scipy import stats
from sklearn.feature_selection import mutual_info_regression


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STUDY_VERSION = "b54g-maritime-influence-v1"
SOURCE_NAME = "b54g_maritime_influence"
DATASET_NAME = "delay_weather_vessel_influence"
TARGET = "target_arrival_delay_h"
MIN_ANALYSIS_ROWS = 200
MIN_GROUP_ROWS = 200
MI_SAMPLE_SIZE = 20_000
RANDOM_SEED = 42
PHYSICAL_HIGH_WAVE_M = 1.5
WEATHER_TOLERANCE = pd.Timedelta(minutes=90)

PREDICTIVE_FORBIDDEN = (
    "actual_",
    "target_",
    "arrival_delay",
    "departure_delay",
    "oracle_",
    "negative_control",
)

WEATHER_TRACKS = {
    "SAFE_T24_POINT": "wave_height_m",
    "SAFE_T24_MAX_12H": "wave_height_max_12h",
    "SAFE_T24_MAX_24H": "wave_height_max_24h",
    "SAFE_T24_MAX_72H": "wave_height_max_72h",
    "ORACLE_PLANNED_ETA": "oracle_planned_wave_height_m",
    "ORACLE_ACTUAL_ATA": "oracle_actual_wave_height_m",
    "NEGATIVE_CONTROL_ETA_PLUS_72H": "negative_control_wave_height_m",
}


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
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


def _query_frame(query: str) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _signature(client, bucket: str, key: str) -> str:
    metadata = client.head_object(Bucket=bucket, Key=key)
    payload = {
        "bucket": bucket,
        "key": key,
        "etag": metadata["ETag"].strip('"'),
        "size": int(metadata["ContentLength"]),
        "study_version": STUDY_VERSION,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _start_run(source_uri: str, checksum: str) -> str:
    metadata = {
        "study_version": STUDY_VERSION,
        "claim_policy": "ASSOCIATION_NOT_CAUSATION",
        "training_executed": False,
        "predictive_oracle_contamination": False,
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
                (SOURCE_NAME, DATASET_NAME, source_uri, checksum, Json(metadata)),
            )
            return str(cursor.fetchone()[0])


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    payload = _clean_json(metadata)
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
                    Json(payload, dumps=lambda x: json.dumps(x, default=_json_default)),
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
                WHERE source_name=%s AND dataset_name=%s
                  AND checksum=%s AND status='SUCCESS'
                ORDER BY finished_at DESC LIMIT 1
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _upload_file(client, source: Path, bucket: str, key: str) -> str:
    suffix = source.suffix.lower()
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".png": "image/png",
        ".parquet": "application/octet-stream",
    }.get(suffix, "application/octet-stream")
    client.upload_file(
        str(source), bucket, key, ExtraArgs={"ContentType": content_type}
    )
    return f"s3://{bucket}/{key}"


def _load_gold(client, bucket: str, key: str) -> pd.DataFrame:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    frame = pd.read_parquet(io.BytesIO(body))
    required = {
        "port_call_id",
        "prediction_at",
        "planned_eta",
        "imo",
        "vessel_name",
        "wave_height_m",
        "wave_height_max_12h",
        "wave_height_max_24h",
        "wave_height_max_72h",
        TARGET,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"B54G missing required Gold columns: {missing}")
    if frame["port_call_id"].duplicated().any():
        raise RuntimeError("B54G requires exactly one Gold row per port call")
    frame["port_call_id"] = frame["port_call_id"].astype(str)
    frame["prediction_at"] = pd.to_datetime(frame["prediction_at"], utc=True)
    frame["planned_eta"] = pd.to_datetime(frame["planned_eta"], utc=True)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    return frame


def _load_port_call_attributes() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT DISTINCT ON (port_call_id)
            port_call_id::text AS port_call_id,
            cargo_type,
            vessel_type,
            terminal_code,
            mmsi,
            voyage_id,
            actual_ata,
            actual_atd,
            planned_etd,
            departure_delay_h,
            source
        FROM features.port_call_model_ready_v1
        ORDER BY port_call_id, updated_at DESC
        """
    )
    for column in ("actual_ata", "actual_atd", "planned_etd"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
    return frame


def _load_hourly_weather() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT
            observed_at,
            AVG(wave_height_m)::double precision AS wave_height_m,
            AVG(wave_period_s)::double precision AS wave_period_s,
            AVG(wave_direction_deg)::double precision AS wave_direction_deg,
            AVG(wind_speed_ms)::double precision AS wind_speed_ms,
            AVG(surface_current_ms)::double precision AS surface_current_ms,
            AVG(visibility_m)::double precision AS visibility_m,
            AVG(pressure_hpa)::double precision AS pressure_hpa
        FROM core.maritime_observation
        WHERE quality_flag=0
        GROUP BY observed_at
        ORDER BY observed_at
        """
    )
    frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    return frame.sort_values("observed_at").reset_index(drop=True)


def _nearest_weather(
    calls: pd.DataFrame,
    weather: pd.DataFrame,
    event_time: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    left = pd.DataFrame(
        {
            "port_call_id": calls["port_call_id"].astype(str),
            "event_time": pd.to_datetime(event_time, errors="coerce", utc=True),
        }
    ).dropna(subset=["event_time"])
    left = left.sort_values("event_time")
    matched = pd.merge_asof(
        left,
        weather,
        left_on="event_time",
        right_on="observed_at",
        direction="nearest",
        tolerance=WEATHER_TOLERANCE,
    )
    rename = {
        column: f"{prefix}_{column}"
        for column in weather.columns
        if column != "observed_at"
    }
    matched = matched.rename(columns=rename)
    matched[f"{prefix}_observation_at"] = matched["observed_at"]
    matched[f"{prefix}_age_abs_h"] = (
        (matched["observed_at"] - matched["event_time"]).abs().dt.total_seconds()
        / 3600.0
    )
    return matched.drop(columns=["event_time", "observed_at"])


def _build_research_frame(gold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    port_calls = _load_port_call_attributes()
    weather = _load_hourly_weather()
    frame = gold.merge(port_calls, on="port_call_id", how="left", validate="one_to_one")

    planned = _nearest_weather(
        frame, weather, frame["planned_eta"], "oracle_planned"
    )
    actual = _nearest_weather(frame, weather, frame["actual_ata"], "oracle_actual")
    negative = _nearest_weather(
        frame,
        weather,
        frame["planned_eta"] + pd.Timedelta(hours=72),
        "negative_control",
    )
    frame = frame.merge(planned, on="port_call_id", how="left", validate="one_to_one")
    frame = frame.merge(actual, on="port_call_id", how="left", validate="one_to_one")
    frame = frame.merge(negative, on="port_call_id", how="left", validate="one_to_one")

    frame["cargo_type"] = (
        frame["cargo_type"].astype("string").str.strip().replace("", pd.NA)
    )
    frame["vessel_type"] = (
        frame["vessel_type"].astype("string").str.strip().replace("", pd.NA)
    )
    frame["year"] = frame["planned_eta"].dt.year.astype("Int64")
    frame["year_month"] = frame["planned_eta"].dt.strftime("%Y-%m")
    month = frame["planned_eta"].dt.month
    frame["season"] = pd.cut(
        month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=["WINTER", "SPRING", "SUMMER", "AUTUMN", "WINTER_2"],
        include_lowest=True,
    ).astype("string").replace("WINTER_2", "WINTER")
    hour = frame["planned_eta"].dt.hour
    frame["eta_daypart"] = pd.cut(
        hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["NIGHT", "MORNING", "AFTERNOON", "EVENING"],
    ).astype("string")
    frame["delay_band"] = pd.cut(
        frame[TARGET],
        bins=[-np.inf, -1, 1, 3, 6, np.inf],
        labels=[
            "EARLY_GT_1H",
            "WITHIN_PLUS_MINUS_1H",
            "LATE_1_TO_3H",
            "LATE_3_TO_6H",
            "LATE_GT_6H",
        ],
        right=True,
    ).astype("string")
    return frame, weather


def _schema_and_semantic_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = max(1, len(frame))
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        row = {
            "column": column,
            "dtype": str(series.dtype),
            "rows": len(series),
            "missing_count": missing,
            "missing_pct": 100.0 * missing / n,
            "n_unique": int(series.nunique(dropna=True)),
            "constant_flag": bool(series.nunique(dropna=True) <= 1),
            "predictive_forbidden_flag": any(
                token in column.lower() for token in PREDICTIVE_FORBIDDEN
            ),
            "semantic_status": "OK",
        }
        if column == "vessel_type":
            top_share = float(series.value_counts(normalize=True, dropna=True).max() or 0)
            if top_share > 0.95:
                row["semantic_status"] = "INVALID_AS_TRUE_VESSEL_TYPE"
        if column in {"terminal_code", "mmsi"} and missing / n > 0.95:
            row["semantic_status"] = "UNAVAILABLE"
        if column in {
            "wind_speed_ms",
            "surface_current_ms",
            "visibility_m",
            "pressure_hpa",
        } and missing / n > 0.95:
            row["semantic_status"] = "UNAVAILABLE"
        rows.append(row)
    return pd.DataFrame(rows)


def _target_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    groups: list[tuple[str, pd.DataFrame]] = [("ALL", frame)]
    groups.extend((f"YEAR_{year}", group) for year, group in frame.groupby("year"))
    rows = []
    for segment, group in groups:
        y = pd.to_numeric(group[TARGET], errors="coerce").dropna()
        if y.empty:
            continue
        rows.append(
            {
                "segment": segment,
                "n": len(y),
                "mean_h": y.mean(),
                "std_h": y.std(),
                "p01_h": y.quantile(0.01),
                "p05_h": y.quantile(0.05),
                "p50_h": y.quantile(0.50),
                "p90_h": y.quantile(0.90),
                "p95_h": y.quantile(0.95),
                "p99_h": y.quantile(0.99),
                "late_gt_1h_pct": 100.0 * (y > 1).mean(),
                "late_gt_3h_pct": 100.0 * (y > 3).mean(),
                "late_gt_6h_pct": 100.0 * (y > 6).mean(),
            }
        )
    return pd.DataFrame(rows)


def _bh_adjust(pvalues: pd.Series) -> pd.Series:
    values = pd.to_numeric(pvalues, errors="coerce").to_numpy(float)
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return pd.Series(result, index=pvalues.index)
    selected = values[valid]
    order = np.argsort(selected)
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0, 1)
    result[np.where(valid)[0]] = restored
    return pd.Series(result, index=pvalues.index)


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float, int]:
    pair = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < MIN_ANALYSIS_ROWS or pair["x"].nunique() < 2:
        return np.nan, np.nan, len(pair)
    if method == "pearson":
        value, pvalue = stats.pearsonr(pair["x"], pair["y"])
    else:
        value, pvalue = stats.spearmanr(pair["x"], pair["y"])
    return float(value), float(pvalue), len(pair)


def _numeric_associations(frame: pd.DataFrame) -> pd.DataFrame:
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    candidates = []
    for column in frame.columns:
        if column == TARGET or not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        x = pd.to_numeric(frame[column], errors="coerce")
        if x.notna().sum() >= MIN_ANALYSIS_ROWS and x.nunique(dropna=True) > 1:
            candidates.append(column)

    sample = frame.loc[y.notna(), candidates + [TARGET]].copy()
    if len(sample) > MI_SAMPLE_SIZE:
        sample = sample.sample(MI_SAMPLE_SIZE, random_state=RANDOM_SEED)
    matrix = sample[candidates].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0.0)
    mi_values = mutual_info_regression(
        matrix.to_numpy(float),
        sample[TARGET].to_numpy(float),
        random_state=RANDOM_SEED,
    )
    mi_map = dict(zip(candidates, mi_values))

    rows = []
    for column in candidates:
        pearson, pearson_p, n = _safe_corr(frame[column], y, "pearson")
        spearman, spearman_p, _ = _safe_corr(frame[column], y, "spearman")
        rows.append(
            {
                "feature": column,
                "n": n,
                "missing_pct": 100.0 * frame[column].isna().mean(),
                "pearson": pearson,
                "pearson_pvalue": pearson_p,
                "spearman": spearman,
                "spearman_pvalue": spearman_p,
                "mutual_information": float(mi_map[column]),
                "predictive_safe": not any(
                    token in column.lower() for token in PREDICTIVE_FORBIDDEN
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["spearman_fdr"] = _bh_adjust(result["spearman_pvalue"])
    return result.sort_values(
        ["mutual_information", "spearman"], ascending=[False, False]
    ).reset_index(drop=True)


def _cramers_v(table: pd.DataFrame) -> float:
    if min(table.shape) < 2 or table.to_numpy().sum() == 0:
        return np.nan
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    rows, cols = table.shape
    phi2_corrected = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(1, n - 1))
    rows_corrected = rows - ((rows - 1) ** 2) / max(1, n - 1)
    cols_corrected = cols - ((cols - 1) ** 2) / max(1, n - 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)
    return float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else np.nan


def _categorical_associations(frame: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "imo",
        "vessel_name",
        "cargo_type",
        "vessel_type",
        "year",
        "season",
        "eta_daypart",
        "eta_dayofweek",
        "eta_month",
    ]
    rows = []
    y = pd.to_numeric(frame[TARGET], errors="coerce")
    total_variance = float(((y - y.mean()) ** 2).sum())
    for variable in variables:
        if variable not in frame:
            continue
        working = pd.DataFrame({"category": frame[variable], "target": y}).dropna()
        if len(working) < MIN_ANALYSIS_ROWS or working["category"].nunique() < 2:
            continue
        grouped = working.groupby("category", observed=True)["target"]
        summary = grouped.agg(["count", "mean", "median"])
        between = float(
            (summary["count"] * (summary["mean"] - working["target"].mean()) ** 2).sum()
        )
        table = pd.crosstab(working["category"], pd.cut(
            working["target"], [-np.inf, 1, 3, 6, np.inf],
            labels=["LE_1H", "1_TO_3H", "3_TO_6H", "GT_6H"]
        ))
        rows.append(
            {
                "feature": variable,
                "n": len(working),
                "levels": int(summary.shape[0]),
                "eta_squared": between / total_variance if total_variance > 0 else np.nan,
                "cramers_v_delay_band": _cramers_v(table),
                "min_group_mean_h": summary["mean"].min(),
                "max_group_mean_h": summary["mean"].max(),
                "group_mean_range_h": summary["mean"].max() - summary["mean"].min(),
                "largest_level_share_pct": 100.0 * summary["count"].max() / len(working),
            }
        )
    return pd.DataFrame(rows).sort_values("eta_squared", ascending=False)


def _control_matrix(frame: pd.DataFrame) -> np.ndarray:
    cargo = frame["cargo_type"].astype("string").fillna("MISSING")
    frequent = set(cargo.value_counts().head(12).index)
    cargo = cargo.where(cargo.isin(frequent), "OTHER")
    categories = pd.DataFrame(
        {
            "imo": frame["imo"].astype("string").fillna("MISSING"),
            "cargo": cargo,
            "year": frame["year"].astype("string").fillna("MISSING"),
            "month": frame["eta_month"].astype("string").fillna("MISSING"),
            "dow": frame["eta_dayofweek"].astype("string").fillna("MISSING"),
            "daypart": frame["eta_daypart"].astype("string").fillna("MISSING"),
        }
    )
    dummies = pd.get_dummies(categories, drop_first=True, dtype="float64")
    numeric_columns = [
        "vessel_hist_count",
        "vessel_hist_mean_delay_h",
        "vessel_hist_std_delay_h",
        "vessel_hist_late_gt_1h_rate",
        "global_hist_mean_delay_h",
        "global_hist_std_delay_h",
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    numeric["vessel_hist_count"] = np.log1p(numeric["vessel_hist_count"].clip(lower=0))
    for column in numeric:
        missing = numeric[column].isna().astype(float)
        if missing.any():
            dummies[f"{column}_missing"] = missing
        numeric[column] = numeric[column].fillna(numeric[column].median())
        scale = numeric[column].std()
        numeric[column] = (
            (numeric[column] - numeric[column].mean()) / scale if scale > 0 else 0.0
        )
    return np.column_stack(
        [np.ones(len(frame), dtype=float), dummies.to_numpy(float), numeric.to_numpy(float)]
    )


def _residualize(frame: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    controls = _control_matrix(frame)
    targets = []
    masks = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        mask = np.isfinite(values)
        masks[column] = mask
        fill = float(np.nanmedian(values)) if mask.any() else 0.0
        targets.append(np.where(mask, values, fill))
    target_matrix = np.column_stack(targets)
    coefficients = np.linalg.lstsq(controls, target_matrix, rcond=1e-10)[0]
    residuals = target_matrix - controls @ coefficients
    return {
        column: np.where(masks[column], residuals[:, index], np.nan)
        for index, column in enumerate(columns)
    }


def _cluster_effect(
    x: np.ndarray,
    y: np.ndarray,
    clusters: pd.Series,
) -> dict[str, float]:
    valid = np.isfinite(x) & np.isfinite(y) & clusters.notna().to_numpy()
    x = x[valid]
    y = y[valid]
    group = clusters.astype(str).to_numpy()[valid]
    if len(x) < MIN_ANALYSIS_ROWS or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {"n": len(x), "effect": np.nan, "se": np.nan, "pvalue": np.nan}
    denominator = float(np.dot(x, x))
    effect = float(np.dot(x, y) / denominator)
    error = y - effect * x
    scores = pd.Series(x * error).groupby(group).sum().to_numpy(float)
    cluster_count = len(scores)
    correction = cluster_count / max(1, cluster_count - 1)
    variance = correction * float(np.dot(scores, scores)) / (denominator**2)
    se = math.sqrt(max(0.0, variance))
    z = effect / se if se > 0 else np.nan
    pvalue = 2.0 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
    return {
        "n": len(x),
        "clusters": cluster_count,
        "effect": effect,
        "se": se,
        "ci95_low": effect - 1.96 * se,
        "ci95_high": effect + 1.96 * se,
        "pvalue": pvalue,
        "partial_r2": float(np.corrcoef(x, y)[0, 1] ** 2),
    }


def _adjusted_weather_associations(frame: pd.DataFrame) -> pd.DataFrame:
    exposure_columns = [column for column in WEATHER_TRACKS.values() if column in frame]
    columns = [TARGET] + exposure_columns
    residuals = _residualize(frame, columns)
    y_residual = residuals[TARGET]
    rows = []
    reverse = {value: key for key, value in WEATHER_TRACKS.items()}
    for column in exposure_columns:
        effect = _cluster_effect(
            residuals[column], y_residual, frame["imo"].astype("string")
        )
        raw = pd.to_numeric(frame[column], errors="coerce")
        effect.update(
            {
                "track": reverse[column],
                "feature": column,
                "unit": "HOURS_DELAY_PER_METER_WAVE",
                "exposure_std_m": raw.std(),
                "standardized_effect_h": effect.get("effect", np.nan) * raw.std(),
                "predictive_safe": reverse[column].startswith("SAFE_"),
                "causal_claim_allowed": False,
            }
        )
        rows.append(effect)
    result = pd.DataFrame(rows)
    result["pvalue_fdr"] = _bh_adjust(result["pvalue"])
    return result.sort_values("pvalue_fdr").reset_index(drop=True)


def _adjusted_binary_effects(frame: pd.DataFrame) -> pd.DataFrame:
    working = frame.copy()
    outcomes = []
    for threshold in (1, 3, 6):
        column = f"late_gt_{threshold}h"
        working[column] = (working[TARGET] > threshold).astype(float)
        outcomes.append(column)
    high_columns = []
    reverse = {}
    for track, exposure in WEATHER_TRACKS.items():
        if exposure not in working:
            continue
        column = f"high_{track.lower()}"
        working[column] = np.where(
            working[exposure].notna(),
            (working[exposure] >= PHYSICAL_HIGH_WAVE_M).astype(float),
            np.nan,
        )
        high_columns.append(column)
        reverse[column] = track
    residuals = _residualize(working, outcomes + high_columns)
    rows = []
    for high_column in high_columns:
        for outcome in outcomes:
            effect = _cluster_effect(
                residuals[high_column],
                residuals[outcome],
                working["imo"].astype("string"),
            )
            effect.update(
                {
                    "track": reverse[high_column],
                    "outcome": outcome,
                    "high_wave_threshold_m": PHYSICAL_HIGH_WAVE_M,
                    "adjusted_risk_difference_pp": 100.0 * effect.get("effect", np.nan),
                    "ci95_low_pp": 100.0 * effect.get("ci95_low", np.nan),
                    "ci95_high_pp": 100.0 * effect.get("ci95_high", np.nan),
                    "predictive_safe": reverse[high_column].startswith("SAFE_"),
                }
            )
            rows.append(effect)
    result = pd.DataFrame(rows)
    result["pvalue_fdr"] = _bh_adjust(result["pvalue"])
    return result.sort_values(["outcome", "pvalue_fdr"]).reset_index(drop=True)


def _dose_response(frame: pd.DataFrame) -> pd.DataFrame:
    bins = [-np.inf, 0.5, 1.0, 1.5, 2.0, 2.5, np.inf]
    labels = ["LT_0P5", "0P5_TO_1", "1_TO_1P5", "1P5_TO_2", "2_TO_2P5", "GE_2P5"]
    rows = []
    for track, column in WEATHER_TRACKS.items():
        if column not in frame:
            continue
        working = pd.DataFrame(
            {
                "wave": pd.to_numeric(frame[column], errors="coerce"),
                "target": pd.to_numeric(frame[TARGET], errors="coerce"),
            }
        ).dropna()
        working["wave_bin"] = pd.cut(working["wave"], bins=bins, labels=labels)
        for wave_bin, group in working.groupby("wave_bin", observed=True):
            y = group["target"]
            rows.append(
                {
                    "track": track,
                    "wave_bin": str(wave_bin),
                    "n": len(group),
                    "wave_mean_m": group["wave"].mean(),
                    "delay_mean_h": y.mean(),
                    "delay_median_h": y.median(),
                    "delay_p90_h": y.quantile(0.9),
                    "late_gt_1h_pct": 100.0 * (y > 1).mean(),
                    "late_gt_3h_pct": 100.0 * (y > 3).mean(),
                    "late_gt_6h_pct": 100.0 * (y > 6).mean(),
                }
            )
    return pd.DataFrame(rows)


def _within_vessel_matched(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    for track, column in WEATHER_TRACKS.items():
        if column not in frame or track == "NEGATIVE_CONTROL_ETA_PLUS_72H":
            continue
        working = frame[
            ["imo", "year_month", "eta_daypart", "cargo_type", TARGET, column]
        ].copy()
        working[column] = pd.to_numeric(working[column], errors="coerce")
        working = working.dropna(subset=["imo", "year_month", TARGET, column])
        cargo = working["cargo_type"].astype("string").fillna("MISSING")
        frequent = set(cargo.value_counts().head(8).index)
        working["cargo_group"] = cargo.where(cargo.isin(frequent), "OTHER")
        working["high_wave"] = working[column] >= PHYSICAL_HIGH_WAVE_M
        strata_columns = ["imo", "year_month", "eta_daypart", "cargo_group"]
        effects = []
        weights = []
        for key, group in working.groupby(strata_columns, dropna=False):
            high = group.loc[group["high_wave"], TARGET]
            normal = group.loc[~group["high_wave"], TARGET]
            if len(high) < 1 or len(normal) < 1 or len(group) < 4:
                continue
            effect = float(high.mean() - normal.mean())
            weight = float(len(high) * len(normal) / len(group))
            effects.append(effect)
            weights.append(weight)
            detail_rows.append(
                {
                    "track": track,
                    "imo": str(key[0]),
                    "year_month": key[1],
                    "eta_daypart": key[2],
                    "cargo_group": key[3],
                    "n_high": len(high),
                    "n_normal": len(normal),
                    "effect_h": effect,
                    "weight": weight,
                }
            )
        if not effects:
            summary_rows.append(
                {"track": track, "matched_strata": 0, "effect_h": np.nan}
            )
            continue
        effects_array = np.asarray(effects)
        weights_array = np.asarray(weights)
        estimate = float(np.average(effects_array, weights=weights_array))
        bootstrap = []
        for _ in range(500):
            index = rng.integers(0, len(effects_array), len(effects_array))
            bootstrap.append(
                np.average(effects_array[index], weights=weights_array[index])
            )
        summary_rows.append(
            {
                "track": track,
                "matched_strata": len(effects),
                "effective_weight": weights_array.sum(),
                "effect_h": estimate,
                "ci95_low_h": float(np.quantile(bootstrap, 0.025)),
                "ci95_high_h": float(np.quantile(bootstrap, 0.975)),
                "high_wave_threshold_m": PHYSICAL_HIGH_WAVE_M,
                "causal_claim_allowed": False,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(detail_rows)


def _heterogeneity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track in ("SAFE_T24_POINT", "ORACLE_PLANNED_ETA"):
        column = WEATHER_TRACKS[track]
        for group_feature in ("imo", "cargo_type", "season", "eta_daypart", "year"):
            for group_value, group in frame.groupby(group_feature, dropna=False):
                working = group[[column, TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
                if len(working) < MIN_GROUP_ROWS or working[column].nunique() < 5:
                    continue
                high = working.loc[working[column] >= PHYSICAL_HIGH_WAVE_M, TARGET]
                normal = working.loc[working[column] < PHYSICAL_HIGH_WAVE_M, TARGET]
                spearman, pvalue, _ = _safe_corr(
                    working[column], working[TARGET], "spearman"
                )
                rows.append(
                    {
                        "track": track,
                        "group_feature": group_feature,
                        "group_value": str(group_value),
                        "n": len(working),
                        "n_high_wave": len(high),
                        "spearman": spearman,
                        "spearman_pvalue": pvalue,
                        "high_vs_normal_delay_diff_h": (
                            high.mean() - normal.mean()
                            if len(high) >= 10 and len(normal) >= 10
                            else np.nan
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["spearman_fdr"] = _bh_adjust(result["spearman_pvalue"])
    return result


def _temporal_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track, column in WEATHER_TRACKS.items():
        for year, group in frame.groupby("year"):
            spearman, pvalue, n = _safe_corr(group[column], group[TARGET], "spearman")
            rows.append(
                {
                    "track": track,
                    "year": year,
                    "n": n,
                    "spearman": spearman,
                    "pvalue": pvalue,
                    "mean_delay_h": group[TARGET].mean(),
                    "mean_wave_m": pd.to_numeric(group[column], errors="coerce").mean(),
                }
            )
    return pd.DataFrame(rows)


def _data_requirements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, "ETA_REVISION_HISTORY", "PMIS/VTS", "Separate initial schedule from last revised ETA", "CRITICAL"),
            (2, "AIS_T24_T12_T6_T3", "AIS/VTS", "Position, SOG, COG, distance and navigation status", "CRITICAL"),
            (3, "PREVIOUS_PORT_ATD_ROUTE", "AIS/Port calls", "Measure delay already acquired upstream", "CRITICAL"),
            (4, "VTS_ANCHORAGE_BERTH_TIMES", "VTS/PMIS", "Separate navigation delay from port waiting", "CRITICAL"),
            (5, "BERTH_OCCUPANCY_QUEUE", "Terminal operating system", "Measure congestion and resource constraints", "CRITICAL"),
            (6, "DELAY_REASON_CODE", "Operations", "Validate causal interpretation", "CRITICAL"),
            (7, "FORECAST_VINTAGES_T24", "ECMWF/Copernicus", "Use forecasts truly available at prediction time", "HIGH"),
            (8, "WIND_VISIBILITY_CURRENT_PRESSURE", "ECMWF/Copernicus/local station", "Complete environmental exposure", "HIGH"),
            (9, "TRUE_VESSEL_DIMENSIONS", "Vessel registry", "Type, LOA, beam, draught, DWT and service speed", "HIGH"),
            (10, "PILOT_TUG_AVAILABILITY", "Port operations", "Measure nautical service constraints", "MEDIUM"),
        ],
        columns=["priority", "data_item", "source", "scientific_use", "criticality"],
    )


def _make_figures(
    frame: pd.DataFrame,
    dose: pd.DataFrame,
    adjusted: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    figures = []
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(frame[TARGET], bins=80, color="#176B87", alpha=0.85)
    ax.axvline(frame[TARGET].median(), color="#C43D3D", linewidth=2, label="Median")
    ax.set(xlabel="Arrival delay (hours)", ylabel="Port calls", title="Arrival delay distribution")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "01_target_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)

    plot_data = dose[dose["track"].isin(["SAFE_T24_POINT", "ORACLE_PLANNED_ETA"])]
    fig, ax = plt.subplots(figsize=(9, 5))
    for track, group in plot_data.groupby("track"):
        group = group.sort_values("wave_mean_m")
        ax.plot(group["wave_mean_m"], group["delay_mean_h"], marker="o", label=track)
    ax.set(xlabel="Mean significant wave height (m)", ylabel="Mean delay (hours)", title="Wave dose-response")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "02_wave_dose_response.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)

    plot_adjusted = adjusted.dropna(subset=["effect", "ci95_low", "ci95_high"])
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(plot_adjusted))))
    y = np.arange(len(plot_adjusted))
    ax.errorbar(
        plot_adjusted["effect"],
        y,
        xerr=[
            plot_adjusted["effect"] - plot_adjusted["ci95_low"],
            plot_adjusted["ci95_high"] - plot_adjusted["effect"],
        ],
        fmt="o",
        color="#176B87",
        ecolor="#6B7280",
    )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(y, plot_adjusted["track"])
    ax.set(xlabel="Adjusted hours of delay per +1 m wave", title="Adjusted associations, clustered by IMO")
    fig.tight_layout()
    path = output_dir / "03_adjusted_wave_effects.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append(path)
    return figures


def _decision(
    frame: pd.DataFrame,
    adjusted: pd.DataFrame,
    binary: pd.DataFrame,
    matched: pd.DataFrame,
    stability: pd.DataFrame,
    semantic: pd.DataFrame,
) -> dict[str, Any]:
    safe = adjusted[adjusted["track"] == "SAFE_T24_POINT"]
    planned = adjusted[adjusted["track"] == "ORACLE_PLANNED_ETA"]
    negative = adjusted[adjusted["track"] == "NEGATIVE_CONTROL_ETA_PLUS_72H"]
    matched_planned = matched[matched["track"] == "ORACLE_PLANNED_ETA"]

    def first(table: pd.DataFrame, column: str, default=np.nan):
        return default if table.empty else table.iloc[0].get(column, default)

    planned_effect = float(first(planned, "standardized_effect_h", np.nan))
    safe_effect = float(first(safe, "standardized_effect_h", np.nan))
    negative_effect = float(first(negative, "standardized_effect_h", np.nan))
    matched_low = float(first(matched_planned, "ci95_low_h", np.nan))
    matched_high = float(first(matched_planned, "ci95_high_h", np.nan))
    planned_significant = bool(first(planned, "pvalue_fdr", 1.0) < 0.05)
    negative_smaller = bool(
        np.isfinite(planned_effect)
        and np.isfinite(negative_effect)
        and abs(negative_effect) < 0.5 * max(abs(planned_effect), 1e-9)
    )
    matched_consistent = bool(
        np.isfinite(matched_low)
        and np.isfinite(matched_high)
        and (matched_low > 0 or matched_high < 0)
    )
    yearly = stability[stability["track"] == "ORACLE_PLANNED_ETA"].dropna(subset=["spearman"])
    if yearly.empty or not np.isfinite(planned_effect) or planned_effect == 0:
        sign_stability = 0.0
    else:
        sign_stability = float((np.sign(yearly["spearman"]) == np.sign(planned_effect)).mean())

    robust = (
        planned_significant
        and abs(planned_effect) >= 0.05
        and negative_smaller
        and matched_consistent
        and sign_stability >= 0.60
    )
    min_binary_p = (
        float(pd.to_numeric(binary["pvalue_fdr"], errors="coerce").min())
        if not binary.empty
        else 1.0
    )
    any_signal = bool(
        planned_significant or min_binary_p < 0.05 or abs(safe_effect) >= 0.05
    )
    if robust:
        status = "ROBUST_ASSOCIATION_NOT_CAUSAL"
    elif any_signal:
        status = "CONDITIONAL_ASSOCIATION_NEEDS_OPERATIONAL_DATA"
    else:
        status = "NO_STABLE_WEATHER_ASSOCIATION"

    unavailable = semantic.loc[
        semantic["semantic_status"].isin(["UNAVAILABLE", "INVALID_AS_TRUE_VESSEL_TYPE"]),
        "column",
    ].tolist()
    return {
        "status": status,
        "causal_identification": "NOT_IDENTIFIED",
        "claim_allowed": "ADJUSTED_ASSOCIATION_ONLY",
        "source_rows": len(frame),
        "unique_port_calls": int(frame["port_call_id"].nunique()),
        "predictive_oracle_contamination": False,
        "training_executed": False,
        "safe_t24_standardized_effect_h": safe_effect,
        "oracle_planned_standardized_effect_h": planned_effect,
        "negative_control_standardized_effect_h": negative_effect,
        "negative_control_smaller": negative_smaller,
        "matched_effect_ci_excludes_zero": matched_consistent,
        "oracle_yearly_sign_stability": sign_stability,
        "unavailable_or_semantically_invalid": unavailable,
        "predictive_recommendation": (
            "DO_NOT_PROMOTE_ORACLE_FEATURES; KEEP_SAFE_T24_AS_RESEARCH_ONLY"
        ),
        "operational_recommendation": (
            "ACQUIRE_ETA_REVISIONS_AIS_VTS_BERTH_AND_DELAY_REASON_DATA"
        ),
        "next_block": "B54G_B_OPERATIONAL_DATA_CONTRACT_AND_ACQUISITION",
    }


def _write_markdown(path: Path, decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# B54G Maritime Delay Influence Study",
                "",
                f"Decision: **{decision['status']}**",
                "",
                "This study estimates statistical associations. It does not identify causality.",
                "Oracle weather at planned ETA and actual ATA is explanatory only and is forbidden from predictive training.",
                "",
                "## Main safeguards",
                "",
                "- One row per port call.",
                "- SAFE_T24 and Oracle tracks are kept separate.",
                "- Effects are adjusted for IMO, cargo, calendar and strict-past vessel history.",
                "- Standard errors are clustered by IMO.",
                "- ETA +72 h weather is used as a negative control.",
                "- No predictive training or split is created.",
                "",
                "## Operational conclusion",
                "",
                decision["operational_recommendation"],
            ]
        ),
        encoding="utf-8",
    )


def run_b54g_influence_study(
    source_bucket: str,
    model_ready_key: str,
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    client = _s3_client()
    checksum = _signature(client, source_bucket, model_ready_key)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {
            "status": "SUCCESS",
            "run_id": previous[0],
            "reused": True,
            "results": previous[1],
        }

    source_uri = f"s3://{source_bucket}/{model_ready_key}"
    run_id = _start_run(source_uri, checksum)
    try:
        with tempfile.TemporaryDirectory(prefix="b54g-") as temporary:
            output_dir = Path(temporary)
            gold = _load_gold(client, source_bucket, model_ready_key)
            frame, weather = _build_research_frame(gold)

            semantic = _schema_and_semantic_audit(frame)
            target = _target_distribution(frame)
            numeric = _numeric_associations(frame)
            categorical = _categorical_associations(frame)
            adjusted = _adjusted_weather_associations(frame)
            binary = _adjusted_binary_effects(frame)
            dose = _dose_response(frame)
            matched, matched_detail = _within_vessel_matched(frame)
            heterogeneity = _heterogeneity(frame)
            stability = _temporal_stability(frame)
            requirements = _data_requirements()
            decision = _decision(
                frame, adjusted, binary, matched, stability, semantic
            )

            reports = {
                "00_schema_semantic_audit.csv": semantic,
                "01_target_distribution_and_drift.csv": target,
                "02_all_numeric_target_associations.csv": numeric,
                "03_categorical_associations.csv": categorical,
                "04_adjusted_weather_associations.csv": adjusted,
                "05_adjusted_delay_risk_differences.csv": binary,
                "06_wave_threshold_dose_response.csv": dose,
                "07_within_vessel_matched_summary.csv": matched,
                "08_within_vessel_matched_strata.csv": matched_detail,
                "09_effect_heterogeneity.csv": heterogeneity,
                "10_temporal_stability.csv": stability,
                "11_required_operational_data.csv": requirements,
            }
            for name, report in reports.items():
                report.to_csv(output_dir / name, index=False)

            research_columns = [
                "port_call_id",
                "prediction_at",
                "planned_eta",
                "actual_ata",
                "imo",
                "vessel_name",
                "cargo_type",
                TARGET,
                "wave_height_m",
                "wave_height_max_12h",
                "wave_height_max_24h",
                "wave_height_max_72h",
                "oracle_planned_wave_height_m",
                "oracle_planned_wave_period_s",
                "oracle_actual_wave_height_m",
                "oracle_actual_wave_period_s",
                "negative_control_wave_height_m",
            ]
            research_path = output_dir / "b54g_explanatory_weather_tracks_v1.parquet"
            frame[research_columns].to_parquet(research_path, index=False)
            figures = _make_figures(frame, dose, adjusted, output_dir)

            decision_path = output_dir / "b54g_influence_decision_v1.json"
            decision_path.write_text(
                json.dumps(_clean_json(decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            markdown_path = output_dir / "B54G_EXECUTIVE_REPORT.md"
            _write_markdown(markdown_path, decision)

            uploaded = {}
            for path in sorted(output_dir.iterdir()):
                if path == research_path:
                    key = f"research/b54g/{output_prefix}/{path.name}"
                elif path.suffix == ".json":
                    key = f"configs/b54g/{output_prefix}/{path.name}"
                elif path.suffix == ".png":
                    key = f"reports/b54g/{output_prefix}/figures/{path.name}"
                else:
                    key = f"reports/b54g/{output_prefix}/{path.name}"
                uploaded[path.name] = _upload_file(client, path, output_bucket, key)

            metadata = {
                **decision,
                "study_version": STUDY_VERSION,
                "weather_observations": len(weather),
                "reports": uploaded,
                "adjusted_associations": _clean_json(adjusted.to_dict("records")),
                "matched_associations": _clean_json(matched.to_dict("records")),
                "output_prefix": f"s3://{output_bucket}/reports/b54g/{output_prefix}/",
            }
            _finish_run(run_id, "SUCCESS", len(frame), metadata)
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
            {"study_version": STUDY_VERSION},
            error_message=str(exc),
        )
        raise
