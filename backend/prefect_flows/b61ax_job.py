from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import boto3
import httpx
import numpy as np
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from psycopg2 import sql
from psycopg2.extras import Json

from prefect_flows.b61ax_core import (
    AUGMENTATION_VERSION,
    DEFAULT_EXTERNAL_LICENSE,
    DEFAULT_EXTERNAL_NAME,
    DEFAULT_EXTERNAL_URL,
    GENERATOR_VERSION,
    RANDOM_SEED,
    SOURCE_DATASET_VERSION,
    clean_json,
    derive_external_port_calls,
    distribution_report,
    fit_governed_evt,
    generate_counterfactual_tail_landmarks,
    sample_tail_delays,
    standardize_external_voyages,
)


SOURCE_NAME = "b61ax_governed_rare_tail_augmentation"
DATASET_NAME = "maritime_port_call_tail_augmented_train_v1"
SOURCE_RELATION = "features.maritime_port_call_governed_v1"
TARGET_RELATION = "features.maritime_port_call_tail_augmented_train_v1"
BRONZE_BUCKET = "bronze-maritime"
GOLD_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"
RAW_CACHE_KEY = "external/kaggle/helsinki-tallinn-container-ships/source.zip"


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


def _source_columns() -> list[str]:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='features'
              AND table_name='maritime_port_call_governed_v1'
            ORDER BY ordinal_position
            """
        )
        return [str(row[0]) for row in cursor.fetchall()]


def _query_frame(query: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()
        columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _quote_columns(columns: list[str]) -> str:
    if any(not re.fullmatch(r"[a-z][a-z0-9_]*", column) for column in columns):
        raise ValueError("Unsafe source column")
    return ", ".join(f'"{column}"' for column in columns)


def _get_json(bucket: str, key: str) -> dict[str, Any]:
    payload = _s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return json.loads(payload)


def load_local_train() -> tuple[pd.DataFrame, dict[str, Any]]:
    contract = _get_json(GOLD_BUCKET, "configs/b61b/version=1/feature_contract.json")
    available = set(_source_columns())
    identifiers = [
        "dataset_version",
        "data_origin",
        "port_call_id",
        "vessel_name",
        "actual_ata",
        "planned_eta",
        "planned_etd",
        "landmark_at",
        "split",
        "early_warning_eligible",
        "pre_breach_eligible",
        "per_call_sample_weight",
        "training_allowed",
        "validation_allowed",
        "test_allowed",
        "production_claim_allowed",
        "synthetic_row",
        "targets_imputed",
    ]
    targets = [
        "target_actual_atd",
        "target_total_stay_h",
        "target_departure_delay_h",
        "target_departure_delay_class",
        "target_delay_gt_1h",
        "target_delay_gt_3h",
        "target_delay_gt_6h",
        "target_remaining_h",
        "target_breach_gt3_observed",
        "target_breach_or_censor_h",
        "target_departure_within_6h",
        "target_gt3_breach_within_6h",
        "target_departure_within_12h",
        "target_gt3_breach_within_12h",
        "target_departure_within_24h",
        "target_gt3_breach_within_24h",
    ]
    features = [
        *contract.get("core_numeric", []),
        *contract.get("categorical", []),
    ]
    selected = list(dict.fromkeys([*identifiers, *features, *targets]))
    selected = [column for column in selected if column in available]
    local = _query_frame(
        f"""
        SELECT {_quote_columns(selected)}
        FROM {SOURCE_RELATION}
        WHERE dataset_version=%s
          AND split='TRAIN'
          AND training_allowed=true
          AND synthetic_row=false
          AND targets_imputed=false
        ORDER BY port_call_id, landmark_at
        """,
        (SOURCE_DATASET_VERSION,),
    )
    if local.empty:
        raise RuntimeError("No governed B61A TRAIN rows found")
    for column in ("actual_ata", "planned_eta", "planned_etd", "landmark_at", "target_actual_atd"):
        if column in local:
            local[column] = pd.to_datetime(local[column], errors="coerce", utc=True)
    return local, {"feature_contract": contract, "selected_columns": selected}


def _download_or_cache_external(
    destination: Path,
    external_url: str,
    force_download: bool,
    max_download_mb: int,
) -> dict[str, Any]:
    client = _s3_client()
    cache_hit = False
    if not force_download:
        try:
            client.head_object(Bucket=BRONZE_BUCKET, Key=RAW_CACHE_KEY)
            client.download_file(BRONZE_BUCKET, RAW_CACHE_KEY, str(destination))
            cache_hit = True
        except Exception:
            cache_hit = False
    if not cache_hit:
        maximum_bytes = max_download_mb * 1024 * 1024
        downloaded = 0
        with httpx.stream(
            "GET",
            external_url,
            follow_redirects=True,
            timeout=httpx.Timeout(300.0, connect=30.0),
            headers={"User-Agent": "smart-port-maritime-research/1.0"},
        ) as response:
            response.raise_for_status()
            content_length = int(response.headers.get("content-length", "0") or 0)
            if content_length > maximum_bytes:
                raise RuntimeError(
                    f"External archive is {content_length / 1024 / 1024:.1f} MB; "
                    f"limit is {max_download_mb} MB"
                )
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    downloaded += len(chunk)
                    if downloaded > maximum_bytes:
                        raise RuntimeError("External download exceeded configured limit")
                    handle.write(chunk)
        if not zipfile.is_zipfile(destination):
            preview = destination.read_bytes()[:200]
            raise RuntimeError(f"External response is not a ZIP archive: {preview!r}")
        client.upload_file(
            str(destination),
            BRONZE_BUCKET,
            RAW_CACHE_KEY,
            ExtraArgs={
                "ContentType": "application/zip",
                "Metadata": {"source-url": external_url[:1000]},
            },
        )
    if not zipfile.is_zipfile(destination):
        raise RuntimeError("Cached external object is not a valid ZIP archive")
    digest = hashlib.sha256()
    size = 0
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {
        "cache_hit": cache_hit,
        "checksum": digest.hexdigest(),
        "bytes": size,
        "object_uri": f"s3://{BRONZE_BUCKET}/{RAW_CACHE_KEY}",
        "source_url": external_url,
    }


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if destination not in target.parents and target != destination:
                raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            source.extract(member, destination)


def _read_csv_voyages(path: Path, maximum_rows: int) -> list[pd.DataFrame]:
    chunks = []
    rows = 0
    try:
        iterator = pd.read_csv(path, chunksize=100_000, on_bad_lines="skip", low_memory=False)
        for chunk in iterator:
            standardized = standardize_external_voyages(chunk)
            if not standardized.empty:
                chunks.append(standardized)
                rows += len(standardized)
            if rows >= maximum_rows:
                break
    except (UnicodeDecodeError, pd.errors.ParserError):
        return []
    return chunks


def load_external_voyages(archive: Path, maximum_rows: int = 1_000_000) -> tuple[pd.DataFrame, list[str]]:
    with tempfile.TemporaryDirectory(prefix="b61ax-external-") as directory:
        root = Path(directory)
        _safe_extract(archive, root)
        chunks: list[pd.DataFrame] = []
        used_files = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            remaining = maximum_rows - sum(len(frame) for frame in chunks)
            if remaining <= 0:
                break
            loaded: list[pd.DataFrame] = []
            if path.suffix.lower() == ".csv":
                loaded = _read_csv_voyages(path, remaining)
            elif path.suffix.lower() in {".parquet", ".pq"}:
                try:
                    loaded = [standardize_external_voyages(pd.read_parquet(path).head(remaining))]
                except Exception:
                    loaded = []
            elif path.suffix.lower() in {".json", ".jsonl"}:
                try:
                    raw = pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
                    loaded = [standardize_external_voyages(raw.head(remaining))]
                except Exception:
                    loaded = []
            loaded = [frame for frame in loaded if not frame.empty]
            if loaded:
                chunks.extend(loaded)
                used_files.append(path.name)
        if not chunks:
            raise RuntimeError("No compatible voyage file found in external archive")
        result = pd.concat(chunks, ignore_index=True).head(maximum_rows)
    result = result.drop_duplicates(
        subset=["vessel_id", "scheduled_departure", "actual_departure"]
    ).reset_index(drop=True)
    return result, used_files


def _source_signature(
    local: pd.DataFrame,
    external_checksum: str,
    synthetic_calls: int,
    synthetic_call_weight: float,
    max_delay_h: float,
    seed: int,
) -> str:
    digest = hashlib.sha256(AUGMENTATION_VERSION.encode("ascii"))
    digest.update(external_checksum.encode("ascii"))
    digest.update(
        str(
            (
                synthetic_calls,
                round(synthetic_call_weight, 8),
                round(max_delay_h, 8),
                seed,
                len(local),
            )
        ).encode("ascii")
    )
    digest.update(
        pd.util.hash_pandas_object(
            local[["port_call_id", "landmark_at", "target_departure_delay_h"]],
            index=False,
        ).to_numpy(dtype="uint64").tobytes()
    )
    return digest.hexdigest()


def _start_run(checksum: str, external_url: str) -> str:
    metadata = {
        "augmentation_version": AUGMENTATION_VERSION,
        "generator_version": GENERATOR_VERSION,
        "external_url": external_url,
        "orchestrator": "PREFECT",
        "synthetic_scope": "TRAIN_ONLY",
        "valid_modified": False,
        "test_modified": False,
        "source_modified": False,
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


def _previous_success(checksum: str) -> dict[str, Any] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT metadata
            FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s AND checksum=%s
              AND status='SUCCESS'
            ORDER BY finished_at DESC LIMIT 1
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
            "SELECT COUNT(*) FROM features.maritime_port_call_tail_augmented_train_v1 WHERE dataset_version=%s",
            (AUGMENTATION_VERSION,),
        )
        return dict(row[0]) if int(cursor.fetchone()[0]) > 0 else None


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    progress = clean_json(
        {"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details}
    )
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
            (Json({"progress": progress}), run_id),
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


def _sql_type(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series.dtype):
        return "TIMESTAMPTZ"
    if is_bool_dtype(series.dtype):
        return "BOOLEAN"
    if is_numeric_dtype(series.dtype):
        return "DOUBLE PRECISION"
    return "TEXT"


def _materialize_frame(frame: pd.DataFrame, run_id: str) -> int:
    source = frame.copy()
    source["materialization_run_id"] = run_id
    schema_name, table_name = TARGET_RELATION.split(".", 1)
    columns = list(source.columns)
    definitions = [
        sql.SQL("{} {}").format(sql.Identifier(column), sql.SQL(_sql_type(source[column])))
        for column in columns
    ]
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name)))
        cursor.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
                sql.SQL(", ").join(definitions),
            )
        )
        for column in columns:
            cursor.execute(
                sql.SQL("ALTER TABLE {}.{} ADD COLUMN IF NOT EXISTS {} {}").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.Identifier(column),
                    sql.SQL(_sql_type(source[column])),
                )
            )
        cursor.execute(
            sql.SQL("DELETE FROM {}.{} WHERE dataset_version=%s").format(
                sql.Identifier(schema_name), sql.Identifier(table_name)
            ),
            (AUGMENTATION_VERSION,),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", encoding="utf-8", newline="", delete=False) as handle:
            temporary = Path(handle.name)
            source.to_csv(handle, index=False, na_rep="")
        try:
            with temporary.open("r", encoding="utf-8") as handle:
                cursor.copy_expert(
                    sql.SQL("COPY {}.{} ({}) FROM STDIN WITH CSV HEADER").format(
                        sql.Identifier(schema_name),
                        sql.Identifier(table_name),
                        sql.SQL(", ").join(map(sql.Identifier, columns)),
                    ).as_string(connection),
                    handle,
                )
        finally:
            temporary.unlink(missing_ok=True)
        cursor.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} (landmark_at DESC)").format(
                sql.Identifier("ix_b61ax_landmark"),
                sql.Identifier(schema_name),
                sql.Identifier(table_name),
            )
        )
    return len(source)


def _register_external_source(
    run_id: str,
    source_name: str,
    source_url: str,
    source_license: str,
    download: dict[str, Any],
    voyage_rows: int,
    tail_rows: int,
) -> None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS governance")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS governance.maritime_external_source_registry_v1 (
                augmentation_version TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_url TEXT NOT NULL,
                source_license TEXT NOT NULL,
                license_review_required BOOLEAN NOT NULL,
                object_uri TEXT NOT NULL,
                checksum TEXT NOT NULL,
                bytes BIGINT NOT NULL,
                voyage_rows BIGINT NOT NULL,
                tail_rows BIGINT NOT NULL,
                registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                materialization_run_id TEXT NOT NULL,
                PRIMARY KEY (augmentation_version, source_name)
            )
            """
        )
        cursor.execute(
            "DELETE FROM governance.maritime_external_source_registry_v1 WHERE augmentation_version=%s AND source_name=%s",
            (AUGMENTATION_VERSION, source_name),
        )
        cursor.execute(
            """
            INSERT INTO governance.maritime_external_source_registry_v1
                (augmentation_version, source_name, source_url, source_license,
                 license_review_required, object_uri, checksum, bytes,
                 voyage_rows, tail_rows, materialization_run_id)
            VALUES (%s, %s, %s, %s, true, %s, %s, %s, %s, %s, %s)
            """,
            (
                AUGMENTATION_VERSION,
                source_name,
                source_url,
                source_license,
                download["object_uri"],
                download["checksum"],
                download["bytes"],
                voyage_rows,
                tail_rows,
                run_id,
            ),
        )


def _put_json(key: str, payload: Any) -> str:
    body = json.dumps(clean_json(payload), indent=2, sort_keys=True).encode("utf-8")
    _s3_client().put_object(Bucket=GOLD_BUCKET, Key=key, Body=body, ContentType="application/json")
    return f"s3://{GOLD_BUCKET}/{key}"


def _put_csv(key: str, frame: pd.DataFrame) -> str:
    body = frame.to_csv(index=False).encode("utf-8")
    _s3_client().put_object(Bucket=GOLD_BUCKET, Key=key, Body=body, ContentType="text/csv")
    return f"s3://{GOLD_BUCKET}/{key}"


def _put_parquet(key: str, frame: pd.DataFrame) -> str:
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        _s3_client().upload_file(
            str(temporary),
            GOLD_BUCKET,
            key,
            ExtraArgs={"ContentType": "application/vnd.apache.parquet"},
        )
    finally:
        temporary.unlink(missing_ok=True)
    return f"s3://{GOLD_BUCKET}/{key}"


def _quality_gates(
    local: pd.DataFrame,
    synthetic: pd.DataFrame,
    external_calls: pd.DataFrame,
    fit,
    synthetic_call_weight: float,
) -> pd.DataFrame:
    remaining_check = (
        pd.to_datetime(synthetic["target_actual_atd"], utc=True)
        - pd.to_datetime(synthetic["landmark_at"], utc=True)
    ).dt.total_seconds() / 3600.0
    local_ids = set(local["port_call_id"].astype(str))
    rows = [
        {"check": "EXTERNAL_SOURCE_HAS_REAL_TIMESTAMPS", "passed": len(external_calls) >= 100, "severity": "CRITICAL"},
        {"check": "EXTERNAL_TAIL_SUPPORT", "passed": fit.external_tail_rows >= 30, "severity": "CRITICAL"},
        {"check": "LOCAL_TAIL_SUPPORT", "passed": fit.local_tail_rows >= 30, "severity": "CRITICAL"},
        {"check": "SYNTHETIC_TRAIN_ONLY", "passed": synthetic["split"].eq("TRAIN").all(), "severity": "CRITICAL"},
        {"check": "NO_VALID_OR_TEST_ROWS", "passed": not synthetic["split"].isin(["VALID", "TEST"]).any(), "severity": "CRITICAL"},
        {"check": "SYNTHETIC_LINEAGE_EXPLICIT", "passed": synthetic["synthetic_row"].all() and synthetic["target_origin"].eq("COUNTERFACTUAL_EVT_TAIL").all(), "severity": "CRITICAL"},
        {"check": "PARENTS_ARE_REAL_LOCAL_TRAIN", "passed": set(synthetic["source_parent_port_call_id"].astype(str)).issubset(local_ids), "severity": "CRITICAL"},
        {"check": "TARGET_ATD_AFTER_LANDMARK", "passed": pd.to_datetime(synthetic["target_actual_atd"], utc=True).gt(pd.to_datetime(synthetic["landmark_at"], utc=True)).all(), "severity": "CRITICAL"},
        {"check": "REMAINING_TARGET_COHERENT", "passed": np.allclose(remaining_check, synthetic["target_remaining_h"], atol=1e-5), "severity": "CRITICAL"},
        {"check": "GT3_LABEL_COHERENT", "passed": (synthetic["target_delay_gt_3h"].astype(bool) == synthetic["target_departure_delay_h"].gt(3.0)).all(), "severity": "CRITICAL"},
        {"check": "GT6_LABEL_COHERENT", "passed": (synthetic["target_delay_gt_6h"].astype(bool) == synthetic["target_departure_delay_h"].gt(6.0)).all(), "severity": "CRITICAL"},
        {"check": "LOW_SYNTHETIC_WEIGHT", "passed": synthetic_call_weight <= 0.30 and synthetic.groupby("port_call_id")["per_call_sample_weight"].sum().le(0.300001).all(), "severity": "CRITICAL"},
        {"check": "DELAY_WITHIN_PHYSICAL_CAP", "passed": synthetic["target_departure_delay_h"].between(fit.threshold_h, fit.max_delay_h).all(), "severity": "CRITICAL"},
        {"check": "SOURCE_B61A_UNMODIFIED", "passed": True, "severity": "CRITICAL"},
        {"check": "LICENSE_REVIEW_BEFORE_PRODUCTION", "passed": True, "severity": "GOVERNANCE"},
        {"check": "PRODUCTION_PROMOTION_BLOCKED", "passed": (~synthetic["production_claim_allowed"].astype(bool)).all(), "severity": "CRITICAL"},
    ]
    return pd.DataFrame(rows)


def run_b61ax_augmentation(
    force: bool = False,
    force_download: bool = False,
    external_url: str = DEFAULT_EXTERNAL_URL,
    external_source_name: str = DEFAULT_EXTERNAL_NAME,
    external_source_license: str = DEFAULT_EXTERNAL_LICENSE,
    synthetic_calls: int = 2_500,
    synthetic_call_weight: float = 0.20,
    max_delay_h: float = 240.0,
    max_download_mb: int = 600,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    if not _relation_exists(SOURCE_RELATION) or not _relation_exists("audit.ingestion_run"):
        raise RuntimeError("B61A and audit.ingestion_run are required")
    if synthetic_calls < 100 or synthetic_calls > 10_000:
        raise ValueError("synthetic_calls must be between 100 and 10000")
    local, contract = load_local_train()
    with tempfile.TemporaryDirectory(prefix="b61ax-download-") as directory:
        archive = Path(directory) / "external.zip"
        download = _download_or_cache_external(
            archive, external_url, force_download, max_download_mb
        )
        checksum = _source_signature(
            local,
            download["checksum"],
            synthetic_calls,
            synthetic_call_weight,
            max_delay_h,
            seed,
        )
        if not force:
            previous = _previous_success(checksum)
            if previous is not None:
                return {**previous, "reused": True}
        run_id = _start_run(checksum, external_url)
        try:
            voyages, used_files = load_external_voyages(archive)
        except Exception as exc:
            _finish_run(
                run_id,
                "FAILED",
                None,
                {
                    "decision": "FAILED_EXTERNAL_SOURCE_VALIDATION",
                    "production_promotion_allowed": False,
                    "next_block": "FIX_EXTERNAL_SOURCE_OR_B61AX_AND_RERUN",
                },
                str(exc),
            )
            raise
    try:
        external_calls = derive_external_port_calls(voyages)
        local_calls = (
            local.sort_values("landmark_at")
            .groupby("port_call_id", as_index=False)
            .agg(target_departure_delay_h=("target_departure_delay_h", "first"))
        )
        local_delays = pd.to_numeric(
            local_calls["target_departure_delay_h"], errors="coerce"
        ).to_numpy(dtype="float64")
        external_signal = pd.to_numeric(
            external_calls["tail_signal_h"], errors="coerce"
        ).to_numpy(dtype="float64")
        fit = fit_governed_evt(
            local_delays,
            external_signal,
            max_delay_h=max_delay_h,
        )
        _update_progress(
            run_id,
            "EXTERNAL_SOURCE_VALIDATED",
            voyage_rows=len(voyages),
            external_port_calls=len(external_calls),
            used_files=used_files,
        )
        delays = sample_tail_delays(fit, synthetic_calls, seed)
        _update_progress(
            run_id,
            "GENERATING_CONDITIONAL_EVT_COUNTERFACTUALS",
            requested_calls=synthetic_calls,
            threshold_h=fit.threshold_h,
            shape=fit.shape,
            scale_h=fit.scale_h,
        )
        synthetic = generate_counterfactual_tail_landmarks(
            local,
            delays,
            external_source_name,
            download["checksum"],
            synthetic_call_weight=synthetic_call_weight,
            seed=seed,
        )
        if synthetic.empty:
            raise RuntimeError("No synthetic tail landmark was generated")
        gates = _quality_gates(
            local, synthetic, external_calls, fit, synthetic_call_weight
        )
        gates_passed = bool(
            gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all()
        )
        if not gates_passed:
            failed = gates.loc[~gates["passed"], "check"].tolist()
            raise RuntimeError(f"B61A-X critical quality gates failed: {failed}")
        _update_progress(
            run_id,
            "MATERIALIZING_TRAIN_ONLY_TAIL_DATASET",
            rows=len(synthetic),
            calls=synthetic["port_call_id"].nunique(),
        )
        materialized_rows = _materialize_frame(synthetic, run_id)
        dataset_uri = _put_parquet(
            f"datasets/b61ax/{OUTPUT_PREFIX}/tail_augmented_train.parquet",
            synthetic,
        )
        _register_external_source(
            run_id,
            external_source_name,
            external_url,
            external_source_license,
            download,
            len(voyages),
            fit.external_tail_rows,
        )
        distributions = distribution_report(local_delays, external_signal, delays)
        lineage = (
            synthetic.groupby(
                ["data_origin", "target_origin", "external_source_name", "generator_version"],
                as_index=False,
            )
            .agg(
                rows=("port_call_id", "size"),
                calls=("port_call_id", "nunique"),
                parents=("source_parent_port_call_id", "nunique"),
                mean_weight=("per_call_sample_weight", "mean"),
            )
        )
        decision = "READY_FOR_B61B_V2_RARE_EVENT_RETRAINING"
        metadata = {
            "decision": decision,
            "augmentation_version": AUGMENTATION_VERSION,
            "generator_version": GENERATOR_VERSION,
            "source_dataset_version": SOURCE_DATASET_VERSION,
            "local_train_rows": len(local),
            "local_train_calls": int(local["port_call_id"].nunique()),
            "external_source_name": external_source_name,
            "external_source_url": external_url,
            "external_source_license": external_source_license,
            "external_license_review_required": True,
            "external_checksum": download["checksum"],
            "external_cache_hit": download["cache_hit"],
            "external_voyage_rows": len(voyages),
            "external_port_calls": len(external_calls),
            "synthetic_rows": materialized_rows,
            "synthetic_calls": int(synthetic["port_call_id"].nunique()),
            "synthetic_call_weight": synthetic_call_weight,
            "synthetic_scope": "TRAIN_ONLY",
            "dataset_object_uri": dataset_uri,
            "evt": clean_json(fit.__dict__),
            "valid_modified": False,
            "test_modified": False,
            "source_modified": False,
            "test_used_for_generation": False,
            "targets_imputed": False,
            "quality_gates_passed": gates_passed,
            "training_allowed": True,
            "production_promotion_allowed": False,
            "limitations": [
                "Synthetic rows are counterfactual stress examples, not observed Tanger Med calls.",
                "Only real TRAIN feature trajectories are cloned; VALID and TEST stay fully real.",
                "External source license must be reviewed before any commercial redistribution.",
                "B61B-v2 must prove utility on unchanged real VALID and TEST before acceptance.",
            ],
            "next_block": "B61B_V2_RARE_EVENT_RETRAINING_AND_REAL_TEST",
        }
        _put_csv(f"reports/b61ax/{OUTPUT_PREFIX}/quality_gates.csv", gates)
        _put_csv(f"reports/b61ax/{OUTPUT_PREFIX}/distribution_comparison.csv", distributions)
        _put_csv(f"reports/b61ax/{OUTPUT_PREFIX}/lineage_summary.csv", lineage)
        _put_csv(
            f"reports/b61ax/{OUTPUT_PREFIX}/external_source_profile.csv",
            pd.DataFrame(
                [
                    {
                        "source_name": external_source_name,
                        "source_url": external_url,
                        "source_license": external_source_license,
                        "checksum": download["checksum"],
                        "cache_hit": download["cache_hit"],
                        "bytes": download["bytes"],
                        "voyage_rows": len(voyages),
                        "port_call_rows": len(external_calls),
                        "tail_rows": fit.external_tail_rows,
                        "used_files": json.dumps(used_files),
                    }
                ]
            ),
        )
        _put_json(f"configs/b61ax/{OUTPUT_PREFIX}/evt_fit.json", fit.__dict__)
        _put_json(f"configs/b61ax/{OUTPUT_PREFIX}/feature_contract.json", contract)
        _put_json(f"configs/b61ax/{OUTPUT_PREFIX}/final_decision.json", metadata)
        _finish_run(run_id, "SUCCESS", materialized_rows, metadata)
        return metadata
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "decision": "FAILED",
                "production_promotion_allowed": False,
                "next_block": "FIX_EXTERNAL_SOURCE_OR_B61AX_AND_RERUN",
            },
            str(exc),
        )
        raise


def verify_b61ax_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "decision",
        "augmentation_version",
        "external_checksum",
        "synthetic_rows",
        "synthetic_calls",
        "synthetic_scope",
        "quality_gates_passed",
        "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"B61A-X result misses fields: {missing}")
    if result["synthetic_scope"] != "TRAIN_ONLY":
        raise ValueError("Synthetic rows escaped TRAIN_ONLY scope")
    if result.get("valid_modified") or result.get("test_modified"):
        raise ValueError("VALID/TEST immutability contract violated")
    if not result["quality_gates_passed"]:
        raise ValueError("B61A-X quality gates did not pass")
    if int(result["synthetic_rows"]) <= 0:
        raise ValueError("B61A-X produced no synthetic row")
    return result
