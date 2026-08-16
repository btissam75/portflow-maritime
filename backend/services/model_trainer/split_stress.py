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
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.model_selection import GroupShuffleSplit


SPLIT_VERSION = "b54fc-random-temporal-stress-v1"
SOURCE_NAME = "b54fa_one_row_gold"
DATASET_NAME = "b54fc_random_temporal_split_audit"
TARGET_COLUMN = "target_arrival_delay_h"
TIME_COLUMN = "prediction_at"
CALL_COLUMN = "port_call_id"
GROUP_CANDIDATES = ("imo", "vessel_name")
OFFICIAL_PROTOCOL = "TEMPORAL_PURGED"
DIAGNOSTIC_PROTOCOLS = ("RANDOM_IID", "RANDOM_BY_IMO", "ROLLING_TEMPORAL_CV")


def _json_safe(value: Any) -> Any:
    """Return a strict-JSON-compatible copy of nested pandas/numpy values."""
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        return None if pd.isna(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, (np.bool_, bool)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, pd.Series, np.ndarray)):
        return [_json_safe(item) for item in value]
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, (str, bytes)):
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    return value


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
        endpoint_url=os.getenv("SMART_PORT_S3_ENDPOINT", "http://minio:9000"),
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _db_connection():
    return psycopg2.connect(
        host=os.getenv("SMART_PORT_DB_HOST", "timescaledb"),
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


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
                (
                    SOURCE_NAME,
                    DATASET_NAME,
                    source_uri,
                    checksum,
                    Json(
                        _json_safe(metadata),
                        dumps=lambda obj: json.dumps(
                            obj, default=_json_default, allow_nan=False
                        ),
                    ),
                ),
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
                        _json_safe(metadata or {}),
                        dumps=lambda obj: json.dumps(
                            obj, default=_json_default, allow_nan=False
                        ),
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


def _download(client, bucket: str, key: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))
    return destination


def _upload(
    client,
    path: Path,
    bucket: str,
    key: str,
    content_type: str,
) -> str:
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{bucket}/{key}"


def _source_checksum(
    client,
    bucket: str,
    keys: list[str],
    parameters: dict[str, Any],
) -> str:
    payload = []
    for key in keys:
        head = client.head_object(Bucket=bucket, Key=key)
        payload.append(
            {
                "bucket": bucket,
                "key": key,
                "etag": head["ETag"].strip('"'),
                "size": int(head["ContentLength"]),
            }
        )
    payload.append({"parameters": parameters, "split_version": SPLIT_VERSION})
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {CALL_COLUMN, TIME_COLUMN, "planned_eta", TARGET_COLUMN}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"B54F-C source columns missing: {missing}")

    result = frame.copy().reset_index(drop=True)
    result[TIME_COLUMN] = pd.to_datetime(result[TIME_COLUMN], utc=True, errors="coerce")
    result["planned_eta"] = pd.to_datetime(
        result["planned_eta"], utc=True, errors="coerce"
    )
    result[TARGET_COLUMN] = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")

    if result[CALL_COLUMN].isna().any() or result[CALL_COLUMN].duplicated().any():
        raise RuntimeError("B54F-C source is not exactly one row per port_call_id")
    if result[[TIME_COLUMN, "planned_eta", TARGET_COLUMN]].isna().any().any():
        raise RuntimeError("B54F-C model-ready source contains missing time/target values")
    if not np.isfinite(result[TARGET_COLUMN].to_numpy(dtype="float64")).all():
        raise RuntimeError("B54F-C target contains non-finite values")

    result["label_available_at"] = result["planned_eta"] + pd.to_timedelta(
        result[TARGET_COLUMN], unit="h"
    )
    return result


def _assignment(
    frame: pd.DataFrame,
    labels: pd.Series,
    protocol: str,
    fold: int | None = None,
    diagnostic_only: bool = True,
) -> pd.DataFrame:
    output = frame[[CALL_COLUMN, TIME_COLUMN]].copy()
    output["protocol"] = protocol
    output["fold"] = fold
    output["split"] = labels.astype(str).to_numpy()
    output["diagnostic_only"] = bool(diagnostic_only)
    return output.sort_values([TIME_COLUMN, CALL_COLUMN]).reset_index(drop=True)


def build_random_iid(
    frame: pd.DataFrame,
    train_fraction: float,
    valid_fraction: float,
    random_seed: int,
) -> pd.DataFrame:
    n_rows = len(frame)
    n_train = int(np.floor(n_rows * train_fraction))
    n_valid = int(np.floor(n_rows * valid_fraction))
    rng = np.random.default_rng(random_seed)
    order = rng.permutation(n_rows)
    labels = pd.Series("TEST", index=frame.index, dtype="object")
    labels.iloc[order[:n_train]] = "TRAIN"
    labels.iloc[order[n_train : n_train + n_valid]] = "VALID"
    return _assignment(frame, labels, "RANDOM_IID", diagnostic_only=True)


def _group_key(frame: pd.DataFrame) -> pd.Series:
    key = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in GROUP_CANDIDATES:
        if column in frame.columns:
            candidate = frame[column].astype("string").str.strip()
            candidate = candidate.mask(candidate.isin(["", "nan", "None", "<NA>"]))
            key = key.fillna(candidate)
    fallback = "CALL::" + frame[CALL_COLUMN].astype("string")
    return key.fillna(fallback)


def build_random_by_group(
    frame: pd.DataFrame,
    train_fraction: float,
    valid_fraction: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    groups = _group_key(frame)
    if groups.nunique() < 6:
        raise RuntimeError("B54F-C needs at least six vessel groups")

    hold_fraction = 1.0 - train_fraction
    first = GroupShuffleSplit(
        n_splits=1,
        test_size=hold_fraction,
        random_state=random_seed,
    )
    train_pos, hold_pos = next(first.split(frame, groups=groups))
    hold_groups = groups.iloc[hold_pos]
    test_share_of_hold = (1.0 - train_fraction - valid_fraction) / hold_fraction
    second = GroupShuffleSplit(
        n_splits=1,
        test_size=test_share_of_hold,
        random_state=random_seed + 1,
    )
    valid_rel, test_rel = next(
        second.split(frame.iloc[hold_pos], groups=hold_groups)
    )
    valid_pos = hold_pos[valid_rel]
    test_pos = hold_pos[test_rel]

    labels = pd.Series("", index=frame.index, dtype="object")
    labels.iloc[train_pos] = "TRAIN"
    labels.iloc[valid_pos] = "VALID"
    labels.iloc[test_pos] = "TEST"
    if (labels == "").any():
        raise RuntimeError("B54F-C group split left unassigned rows")
    return (
        _assignment(frame, labels, "RANDOM_BY_IMO", diagnostic_only=True),
        groups,
    )


def _time_boundary(frame: pd.DataFrame, fraction: float) -> pd.Timestamp:
    ordered = frame[TIME_COLUMN].sort_values(kind="mergesort").reset_index(drop=True)
    position = min(len(ordered) - 1, max(1, int(np.floor(len(ordered) * fraction))))
    return pd.Timestamp(ordered.iloc[position])


def build_temporal_purged(
    frame: pd.DataFrame,
    train_fraction: float,
    valid_fraction: float,
    purge_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_boundary = _time_boundary(frame, train_fraction)
    valid_boundary = _time_boundary(frame, train_fraction + valid_fraction)
    embargo = pd.Timedelta(hours=purge_hours)
    time = frame[TIME_COLUMN]

    labels = pd.Series("PURGED", index=frame.index, dtype="object")
    labels.loc[time < train_boundary] = "TRAIN"
    labels.loc[(time >= train_boundary + embargo) & (time < valid_boundary)] = "VALID"
    labels.loc[time >= valid_boundary + embargo] = "TEST"

    boundaries = {
        "protocol": "TEMPORAL_PURGED",
        "fold": None,
        "train_boundary": train_boundary,
        "valid_boundary": valid_boundary,
        "valid_start_after_purge": train_boundary + embargo,
        "test_start_after_purge": valid_boundary + embargo,
        "purge_hours": purge_hours,
    }
    return (
        _assignment(
            frame,
            labels,
            "TEMPORAL_PURGED",
            diagnostic_only=False,
        ),
        boundaries,
    )


def build_rolling_temporal(
    frame: pd.DataFrame,
    purge_hours: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    fold_fractions = (
        (1, 0.40, 0.55, 0.70),
        (2, 0.55, 0.70, 0.85),
        (3, 0.70, 0.85, 1.00),
    )
    embargo = pd.Timedelta(hours=purge_hours)
    time = frame[TIME_COLUMN]
    outputs = []
    boundaries = []

    for fold, train_end, valid_end, test_end in fold_fractions:
        train_boundary = _time_boundary(frame, train_end)
        valid_boundary = _time_boundary(frame, valid_end)
        test_boundary = (
            frame[TIME_COLUMN].max() + pd.Timedelta(nanoseconds=1)
            if test_end >= 1.0
            else _time_boundary(frame, test_end)
        )
        participating = time < test_boundary
        labels = pd.Series("OUT_OF_FOLD", index=frame.index, dtype="object")
        labels.loc[participating] = "PURGED"
        labels.loc[time < train_boundary] = "TRAIN"
        labels.loc[
            (time >= train_boundary + embargo) & (time < valid_boundary)
        ] = "VALID"
        labels.loc[
            (time >= valid_boundary + embargo) & (time < test_boundary)
        ] = "TEST"
        outputs.append(
            _assignment(
                frame.loc[participating].copy(),
                labels.loc[participating],
                "ROLLING_TEMPORAL_CV",
                fold=fold,
                diagnostic_only=True,
            )
        )
        boundaries.append(
            {
                "protocol": "ROLLING_TEMPORAL_CV",
                "fold": fold,
                "train_boundary": train_boundary,
                "valid_boundary": valid_boundary,
                "test_boundary": test_boundary,
                "valid_start_after_purge": train_boundary + embargo,
                "test_start_after_purge": valid_boundary + embargo,
                "purge_hours": purge_hours,
            }
        )
    return pd.concat(outputs, ignore_index=True), boundaries


def _finite(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return values[np.isfinite(values)]


def _psi(train: np.ndarray, evaluation: np.ndarray, bins: int = 10) -> float:
    if len(train) < 20 or len(evaluation) < 20:
        return float("nan")
    edges = np.unique(np.quantile(train, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_count = np.histogram(train, bins=edges)[0].astype("float64")
    eval_count = np.histogram(evaluation, bins=edges)[0].astype("float64")
    train_pct = np.clip(train_count / max(1.0, train_count.sum()), 1e-6, None)
    eval_pct = np.clip(eval_count / max(1.0, eval_count.sum()), 1e-6, None)
    return float(np.sum((eval_pct - train_pct) * np.log(eval_pct / train_pct)))


def _numeric_shift(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    protocol: str,
    fold: int | None,
    comparison: str,
) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        train_values = _finite(train[column])
        eval_values = _finite(evaluation[column])
        row = {
            "protocol": protocol,
            "fold": fold,
            "comparison": comparison,
            "feature": column,
            "train_n": len(train_values),
            "evaluation_n": len(eval_values),
            "train_missing_pct": 100.0 * train[column].isna().mean(),
            "evaluation_missing_pct": 100.0 * evaluation[column].isna().mean(),
            "train_mean": float(np.mean(train_values)) if len(train_values) else np.nan,
            "evaluation_mean": (
                float(np.mean(eval_values)) if len(eval_values) else np.nan
            ),
            "psi": _psi(train_values, eval_values),
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "wasserstein": np.nan,
        }
        if len(train_values) >= 20 and len(eval_values) >= 20:
            ks = ks_2samp(train_values, eval_values, method="auto")
            row["ks_statistic"] = float(ks.statistic)
            row["ks_pvalue"] = float(ks.pvalue)
            row["wasserstein"] = float(
                wasserstein_distance(train_values, eval_values)
            )
        rows.append(row)
    return rows


def _js_divergence(train: pd.Series, evaluation: pd.Series) -> float:
    train_text = train.astype("string").fillna("<MISSING>")
    eval_text = evaluation.astype("string").fillna("<MISSING>")
    categories = train_text.unique().tolist()
    categories.extend(
        value for value in eval_text.unique().tolist() if value not in categories
    )
    p = train_text.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    q = eval_text.value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    midpoint = 0.5 * (p + q)
    p_mask = p > 0
    q_mask = q > 0
    p_kl = np.sum(p[p_mask] * np.log(p[p_mask] / midpoint[p_mask]))
    q_kl = np.sum(q[q_mask] * np.log(q[q_mask] / midpoint[q_mask]))
    return float(0.5 * (p_kl + q_kl))


def _categorical_shift(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    protocol: str,
    fold: int | None,
    comparison: str,
) -> list[dict[str, Any]]:
    rows = []
    for column in columns:
        train_values = set(train[column].astype("string").dropna().unique())
        evaluation_text = evaluation[column].astype("string")
        unseen = ~evaluation_text.isin(train_values) & evaluation_text.notna()
        rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "comparison": comparison,
                "feature": column,
                "train_unique": train[column].nunique(dropna=True),
                "evaluation_unique": evaluation[column].nunique(dropna=True),
                "unseen_evaluation_pct": 100.0 * unseen.mean(),
                "train_missing_pct": 100.0 * train[column].isna().mean(),
                "evaluation_missing_pct": 100.0 * evaluation[column].isna().mean(),
                "jensen_shannon_divergence": _js_divergence(
                    train[column], evaluation[column]
                ),
            }
        )
    return rows


def _protocol_groups(assignments: pd.DataFrame):
    group_columns = ["protocol", "fold"]
    for keys, part in assignments.groupby(group_columns, dropna=False, sort=False):
        protocol, fold = keys
        fold_value = None if pd.isna(fold) else int(fold)
        yield str(protocol), fold_value, part


def build_distribution_reports(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    protocol_rows = []
    target_rows = []
    numeric_rows = []
    categorical_rows = []
    columns = list(
        dict.fromkeys(
            [CALL_COLUMN, TARGET_COLUMN] + numeric_features + categorical_features
        )
    )
    source = frame[columns]

    for protocol, fold, assignment in _protocol_groups(assignments):
        joined = assignment.merge(source, on=CALL_COLUMN, how="left", validate="one_to_one")
        active = joined[joined["split"].isin(["TRAIN", "VALID", "TEST"])]
        protocol_rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "source_rows": len(frame),
                "assignment_rows": len(assignment),
                "active_rows": len(active),
                "purged_rows": int((joined["split"] == "PURGED").sum()),
                "train_rows": int((joined["split"] == "TRAIN").sum()),
                "valid_rows": int((joined["split"] == "VALID").sum()),
                "test_rows": int((joined["split"] == "TEST").sum()),
                "diagnostic_only": bool(joined["diagnostic_only"].iloc[0]),
            }
        )
        for split_name in ("TRAIN", "VALID", "TEST"):
            values = _finite(joined.loc[joined["split"] == split_name, TARGET_COLUMN])
            target_rows.append(
                {
                    "protocol": protocol,
                    "fold": fold,
                    "split": split_name,
                    "n": len(values),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "std": float(np.std(values)) if len(values) else np.nan,
                    "min": float(np.min(values)) if len(values) else np.nan,
                    "p05": float(np.quantile(values, 0.05)) if len(values) else np.nan,
                    "p50": float(np.quantile(values, 0.50)) if len(values) else np.nan,
                    "p95": float(np.quantile(values, 0.95)) if len(values) else np.nan,
                    "max": float(np.max(values)) if len(values) else np.nan,
                }
            )

        train = joined[joined["split"] == "TRAIN"]
        for split_name in ("VALID", "TEST"):
            evaluation = joined[joined["split"] == split_name]
            if train.empty or evaluation.empty:
                continue
            comparison = f"TRAIN_VS_{split_name}"
            numeric_rows.extend(
                _numeric_shift(
                    train,
                    evaluation,
                    [TARGET_COLUMN] + numeric_features,
                    protocol,
                    fold,
                    comparison,
                )
            )
            categorical_rows.extend(
                _categorical_shift(
                    train,
                    evaluation,
                    categorical_features,
                    protocol,
                    fold,
                    comparison,
                )
            )

    return (
        pd.DataFrame(protocol_rows),
        pd.DataFrame(target_rows),
        pd.DataFrame(numeric_rows),
        pd.DataFrame(categorical_rows),
    )


def build_overlap_audit(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    groups: pd.Series,
    purge_hours: int,
) -> pd.DataFrame:
    lookup = frame.set_index(CALL_COLUMN)
    group_lookup = pd.Series(groups.to_numpy(), index=frame[CALL_COLUMN].to_numpy())
    rows = []

    for protocol, fold, assignment in _protocol_groups(assignments):
        active = assignment[assignment["split"].isin(["TRAIN", "VALID", "TEST"])]
        sets = {
            split_name: set(active.loc[active["split"] == split_name, CALL_COLUMN])
            for split_name in ("TRAIN", "VALID", "TEST")
        }
        duplicate_rows = int(assignment[CALL_COLUMN].duplicated().sum())
        id_overlap = sum(
            len(sets[left].intersection(sets[right]))
            for left, right in (("TRAIN", "VALID"), ("TRAIN", "TEST"), ("VALID", "TEST"))
        )
        group_overlap = 0
        if protocol == "RANDOM_BY_IMO":
            group_sets = {
                name: set(group_lookup.reindex(list(ids)).dropna())
                for name, ids in sets.items()
            }
            group_overlap = sum(
                len(group_sets[left].intersection(group_sets[right]))
                for left, right in (
                    ("TRAIN", "VALID"),
                    ("TRAIN", "TEST"),
                    ("VALID", "TEST"),
                )
            )

        split_frames = {
            name: lookup.reindex(list(ids)) for name, ids in sets.items()
        }
        chronological = True
        weather_gap_ok = True
        labels_available = True
        train_valid_gap_h = np.nan
        valid_test_gap_h = np.nan
        if protocol in {"TEMPORAL_PURGED", "ROLLING_TEMPORAL_CV"}:
            train = split_frames["TRAIN"]
            valid = split_frames["VALID"]
            test = split_frames["TEST"]
            if train.empty or valid.empty or test.empty:
                chronological = weather_gap_ok = labels_available = False
            else:
                train_valid_gap_h = (
                    valid[TIME_COLUMN].min() - train[TIME_COLUMN].max()
                ).total_seconds() / 3600.0
                valid_test_gap_h = (
                    test[TIME_COLUMN].min() - valid[TIME_COLUMN].max()
                ).total_seconds() / 3600.0
                chronological = bool(
                    train[TIME_COLUMN].max() < valid[TIME_COLUMN].min()
                    and valid[TIME_COLUMN].max() < test[TIME_COLUMN].min()
                )
                weather_gap_ok = bool(
                    train_valid_gap_h >= purge_hours
                    and valid_test_gap_h >= purge_hours
                )
                labels_available = bool(
                    train["label_available_at"].max() <= valid[TIME_COLUMN].min()
                    and valid["label_available_at"].max() <= test[TIME_COLUMN].min()
                )

        rows.append(
            {
                "protocol": protocol,
                "fold": fold,
                "assignment_rows": len(assignment),
                "duplicate_assignment_rows": duplicate_rows,
                "port_call_overlap": id_overlap,
                "vessel_group_overlap": group_overlap,
                "chronological_order_passed": chronological,
                "weather_purge_passed": weather_gap_ok,
                "label_availability_passed": labels_available,
                "train_valid_gap_h": train_valid_gap_h,
                "valid_test_gap_h": valid_test_gap_h,
            }
        )
    return pd.DataFrame(rows)


def _decision(
    frame: pd.DataFrame,
    protocol_summary: pd.DataFrame,
    overlap_audit: pd.DataFrame,
    numeric_shift: pd.DataFrame,
    upstream_decision: str | None,
) -> dict[str, Any]:
    violations = []
    if upstream_decision != "READY_FOR_TEMPORAL_SPLIT":
        violations.append(f"UPSTREAM_DECISION={upstream_decision}")
    if frame[CALL_COLUMN].duplicated().any():
        violations.append("SOURCE_DUPLICATE_PORT_CALLS")
    if (overlap_audit["duplicate_assignment_rows"] > 0).any():
        violations.append("DUPLICATE_SPLIT_ASSIGNMENTS")
    if (overlap_audit["port_call_overlap"] > 0).any():
        violations.append("PORT_CALL_SPLIT_OVERLAP")
    group_row = overlap_audit[overlap_audit["protocol"] == "RANDOM_BY_IMO"]
    if group_row.empty or (group_row["vessel_group_overlap"] > 0).any():
        violations.append("VESSEL_GROUP_SPLIT_OVERLAP")
    temporal = overlap_audit[
        overlap_audit["protocol"].isin(["TEMPORAL_PURGED", "ROLLING_TEMPORAL_CV"])
    ]
    for column in (
        "chronological_order_passed",
        "weather_purge_passed",
        "label_availability_passed",
    ):
        if temporal.empty or not temporal[column].all():
            violations.append(f"TEMPORAL_GATE_FAILED:{column}")
    active_min = protocol_summary[["train_rows", "valid_rows", "test_rows"]].min().min()
    if int(active_min) < 500:
        violations.append(f"SPLIT_TOO_SMALL:{int(active_min)}")

    temporal_target = numeric_shift[
        (numeric_shift["protocol"] == "TEMPORAL_PURGED")
        & (numeric_shift["feature"] == TARGET_COLUMN)
    ]
    max_target_psi = (
        float(temporal_target["psi"].max()) if not temporal_target.empty else np.nan
    )
    if np.isfinite(max_target_psi) and max_target_psi >= 0.25:
        shift_assessment = "SEVERE_TEMPORAL_TARGET_SHIFT"
    elif np.isfinite(max_target_psi) and max_target_psi >= 0.10:
        shift_assessment = "MODERATE_TEMPORAL_TARGET_SHIFT"
    else:
        shift_assessment = "LOW_TEMPORAL_TARGET_SHIFT"

    return {
        "status": "READY_FOR_SPLIT_STRESS_MODELS" if not violations else "NEED_SPLIT_REPAIR",
        "violations": violations,
        "official_protocol": OFFICIAL_PROTOCOL,
        "diagnostic_protocols": list(DIAGNOSTIC_PROTOCOLS),
        "random_results_are_not_official": True,
        "shift_assessment": shift_assessment,
        "temporal_target_max_psi": max_target_psi,
        "next_block": "B54F_D_RANDOM_VS_TEMPORAL_MODEL_STRESS_TEST",
    }


def run_b54fc_split_stress_audit(
    source_bucket: str,
    model_ready_key: str,
    feature_config_key: str,
    upstream_decision_key: str,
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    train_fraction: float = 0.70,
    valid_fraction: float = 0.15,
    purge_hours: int = 72,
    random_seed: int = 42,
    force: bool = False,
) -> dict[str, Any]:
    if train_fraction + valid_fraction >= 0.95:
        raise ValueError("B54F-C requires at least 5% for TEST")
    if purge_hours < 72:
        raise ValueError("B54F-C purge must cover the 72h weather window")

    client = _s3_client()
    source_keys = [model_ready_key, feature_config_key, upstream_decision_key]
    parameters = {
        "train_fraction": train_fraction,
        "valid_fraction": valid_fraction,
        "test_fraction": 1.0 - train_fraction - valid_fraction,
        "purge_hours": purge_hours,
        "random_seed": random_seed,
    }
    checksum = _source_checksum(client, source_bucket, source_keys, parameters)
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            previous_run_id, metadata = previous
            return {
                "status": "SUCCESS",
                "cached": True,
                "run_id": previous_run_id,
                "checksum": checksum,
                "results": metadata,
                "outputs": metadata.get("output_uris", {}),
            }

    source_uri = f"s3://{source_bucket}/{model_ready_key}"
    run_id = _start_run(
        source_uri,
        checksum,
        {
            "split_version": SPLIT_VERSION,
            "parameters": parameters,
            "split_created": False,
            "training_executed": False,
        },
    )
    outputs: dict[str, str] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="b54fc-") as temp_dir:
            work = Path(temp_dir)
            source_path = _download(
                client, source_bucket, model_ready_key, work / "model_ready.parquet"
            )
            config_path = _download(
                client, source_bucket, feature_config_key, work / "feature_config.json"
            )
            decision_path = _download(
                client,
                source_bucket,
                upstream_decision_key,
                work / "upstream_decision.json",
            )
            frame = _prepare_frame(pd.read_parquet(source_path))
            config = json.loads(config_path.read_text(encoding="utf-8"))
            upstream = json.loads(decision_path.read_text(encoding="utf-8"))
            upstream_status = upstream.get("decision", {}).get("status")

            numeric_features = [
                column
                for column in config.get("numeric_features", [])
                if column in frame.columns
            ]
            categorical_features = [
                column
                for column in config.get("categorical_features", [])
                if column in frame.columns
            ]

            random_iid = build_random_iid(
                frame, train_fraction, valid_fraction, random_seed
            )
            random_group, groups = build_random_by_group(
                frame, train_fraction, valid_fraction, random_seed
            )
            temporal, temporal_boundary = build_temporal_purged(
                frame, train_fraction, valid_fraction, purge_hours
            )
            rolling, rolling_boundaries = build_rolling_temporal(frame, purge_hours)
            assignments = pd.concat(
                [random_iid, random_group, temporal, rolling], ignore_index=True
            )
            boundaries = pd.DataFrame([temporal_boundary] + rolling_boundaries)

            (
                protocol_summary,
                target_distribution,
                numeric_shift,
                categorical_shift,
            ) = build_distribution_reports(
                frame,
                assignments,
                numeric_features,
                categorical_features,
            )
            overlap_audit = build_overlap_audit(
                frame, assignments, groups, purge_hours
            )
            decision = _decision(
                frame,
                protocol_summary,
                overlap_audit,
                numeric_shift,
                upstream_status,
            )

            prefix = output_prefix.strip("/")

            def save_parquet(name: str, data: pd.DataFrame, label: str):
                path = work / name
                data.to_parquet(path, index=False)
                outputs[label] = _upload(
                    client,
                    path,
                    output_bucket,
                    f"splits/b54fc/{prefix}/{name}",
                    "application/vnd.apache.parquet",
                )

            def save_csv(name: str, data: pd.DataFrame, label: str):
                path = work / name
                data.to_csv(path, index=False)
                outputs[label] = _upload(
                    client,
                    path,
                    output_bucket,
                    f"reports/b54fc/{prefix}/{name}",
                    "text/csv",
                )

            save_parquet("all_protocol_assignments_v1.parquet", assignments, "all_assignments")
            save_parquet("random_iid_assignments_v1.parquet", random_iid, "random_iid")
            save_parquet("random_by_imo_assignments_v1.parquet", random_group, "random_by_imo")
            save_parquet("temporal_purged_assignments_v1.parquet", temporal, "temporal_purged")
            save_parquet("rolling_temporal_folds_v1.parquet", rolling, "rolling_temporal")
            save_csv("01_protocol_summary.csv", protocol_summary, "protocol_summary")
            save_csv("02_target_distribution.csv", target_distribution, "target_distribution")
            save_csv("03_numeric_distribution_shift.csv", numeric_shift, "numeric_shift")
            save_csv("04_categorical_distribution_shift.csv", categorical_shift, "categorical_shift")
            save_csv("05_temporal_boundaries.csv", boundaries, "temporal_boundaries")
            save_csv("06_overlap_and_purge_audit.csv", overlap_audit, "overlap_audit")

            summary = _json_safe({
                "split_version": SPLIT_VERSION,
                "source_rows": int(len(frame)),
                "source_calls": int(frame[CALL_COLUMN].nunique()),
                "feature_count": int(len(config.get("feature_columns", []))),
                "numeric_feature_count": int(len(numeric_features)),
                "categorical_feature_count": int(len(categorical_features)),
                "wave_feature_count": int(len(config.get("wave_features", []))),
                "parameters": parameters,
                "protocol_summary": protocol_summary.to_dict(orient="records"),
                "overlap_gate": {
                    "passed": bool(decision["status"] == "READY_FOR_SPLIT_STRESS_MODELS"),
                    "violations": decision["violations"],
                },
                "decision": decision,
                "split_created": True,
                "training_executed": False,
                "output_uris": outputs,
                "generated_at_utc": datetime.now(timezone.utc),
            })
            summary_path = work / "b54fc_split_decision_v1.json"
            summary_path.write_text(
                json.dumps(
                    summary,
                    indent=2,
                    default=_json_default,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            outputs["decision"] = _upload(
                client,
                summary_path,
                output_bucket,
                f"configs/b54fc/{prefix}/b54fc_split_decision_v1.json",
                "application/json",
            )
            summary["output_uris"] = outputs

        _finish_run(run_id, "SUCCESS", len(frame), summary)
        return {
            "status": "SUCCESS",
            "cached": False,
            "run_id": run_id,
            "checksum": checksum,
            "results": summary,
            "outputs": outputs,
        }
    except Exception as exc:
        _finish_run(run_id, "FAILED", error_message=str(exc)[:4000])
        raise
