from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import duckdb
import pandas as pd
import psycopg2
from psycopg2.extras import Json


COLLECTOR_VERSION = "b57g-tir-incremental-ingestion-v1"
SOURCE_NAME = "b57g_tir_incremental_collector"
DATASET_NAME = "tir_incremental_canonical"
DEFAULT_BUCKET = "bronze-maritime"
INCOMING_PREFIX = "tir/incoming/"
LANDING_PREFIX = "tir/landing/version=1/"
CANONICAL_PREFIX = "tir/canonical/version=2/snapshots/"
LEGACY_KEY = "tir/source/version=1/data1_maritime_minimal_v1.parquet"
MAX_FUTURE_DAYS = 1

REQUIRED_COLUMNS = (
    "SOURCE_ROW_INDEX",
    "UNITE",
    "DECLARANT",
    "VIDE_PLEIN",
    "NATURE_MARCHANDISE",
    "MATIERE_DANGER",
    "POIDS",
    "GROUPAGE",
    "DATE_ZRE",
    "DATE_EMBARQUEMENT",
)

ALIASES = {
    "SOURCE_ROW_ID": "SOURCE_ROW_INDEX",
    "SOURCE_INDEX": "SOURCE_ROW_INDEX",
    "UNIT": "UNITE",
    "NUM_UNITE": "UNITE",
    "IS_GROUPAGE": "GROUPAGE",
    "DATE_ENTREE_ZRE": "DATE_ZRE",
    "ZRE_DATE": "DATE_ZRE",
    "DATE_EMB": "DATE_EMBARQUEMENT",
}

METADATA_COLUMNS = (
    "B57G_AVAILABLE_AT",
    "B57G_BATCH_ID",
    "B57G_SOURCE_CHECKSUM",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default, allow_nan=False)


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
        dbname=os.getenv("SMART_PORT_DB_NAME", "maritime"),
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _start_audit(
    source_bucket: str,
    source_key: str,
    checksum: str,
    metadata: dict[str, Any],
) -> str:
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
                f"s3://{source_bucket}/{source_key}",
                checksum,
                Json(metadata, dumps=_json_dumps),
            ),
        )
        return str(cursor.fetchone()[0])


def _finish_audit(
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
            SET status=%s, row_count=%s, metadata=metadata || %s,
                error_message=%s, finished_at=now()
            WHERE run_id=%s
            """,
            (
                status,
                row_count,
                Json(metadata, dumps=_json_dumps),
                error_message,
                run_id,
            ),
        )


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT run_id, metadata
            FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s
              AND checksum=%s AND status='SUCCESS'
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (SOURCE_NAME, DATASET_NAME, checksum),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return str(row[0]), dict(row[1] or {})


def _safe_name(value: str) -> str:
    name = Path(value).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)[:180] or "batch"


def _timestamp_token(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _download_and_hash(client, bucket: str, key: str, destination: Path) -> str:
    digest = hashlib.sha256()
    response = client.get_object(Bucket=bucket, Key=key)
    with destination.open("wb") as handle:
        while True:
            chunk = response["Body"].read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            handle.write(chunk)
    return digest.hexdigest()


def _read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, sep=None, engine="python")
    raise ValueError("B57G accepts only CSV, TXT, Parquet or PQ files")


def _canonical_column_name(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_").upper()
    return ALIASES.get(normalized, normalized)


def _normalize_input(
    frame: pd.DataFrame,
    received_at: datetime,
    checksum: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    original_columns = [str(column) for column in frame.columns]
    normalized_columns = [_canonical_column_name(column) for column in frame.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("Column normalization creates duplicate column names")
    frame = frame.copy()
    frame.columns = normalized_columns

    missing = sorted(set(REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required TIR columns: {missing}")

    frame["SOURCE_ROW_INDEX"] = frame["SOURCE_ROW_INDEX"].astype("string").str.strip()
    frame.loc[
        frame["SOURCE_ROW_INDEX"].isin(["", "nan", "None", "<NA>"]),
        "SOURCE_ROW_INDEX",
    ] = pd.NA
    frame["DATE_ZRE"] = pd.to_datetime(
        frame["DATE_ZRE"], errors="coerce", utc=True, format="mixed"
    )
    frame["DATE_EMBARQUEMENT"] = pd.to_datetime(
        frame["DATE_EMBARQUEMENT"], errors="coerce", utc=True, format="mixed"
    )

    now = pd.Timestamp(received_at)
    invalid_key = frame["SOURCE_ROW_INDEX"].isna()
    invalid_date = frame["DATE_ZRE"].isna()
    future_date = frame["DATE_ZRE"].gt(now + pd.Timedelta(days=MAX_FUTURE_DAYS))
    invalid_mask = invalid_key | invalid_date | future_date
    quarantine = frame.loc[invalid_mask].copy()
    valid = frame.loc[~invalid_mask].copy()
    validated_rows_before_dedup = int(len(valid))

    duplicate_rows = int(valid["SOURCE_ROW_INDEX"].duplicated(keep="last").sum())
    valid = valid.drop_duplicates("SOURCE_ROW_INDEX", keep="last")
    valid["B57G_AVAILABLE_AT"] = pd.Timestamp(received_at)
    valid["B57G_BATCH_ID"] = checksum[:16]
    valid["B57G_SOURCE_CHECKSUM"] = checksum

    valid_ratio = validated_rows_before_dedup / max(len(frame), 1)
    if valid_ratio < 0.95:
        raise ValueError(
            f"Only {valid_ratio:.2%} of incoming rows pass key/date validation"
        )

    metadata = {
        "input_rows": int(len(frame)),
        "valid_rows": int(len(valid)),
        "quarantine_rows": int(len(quarantine)),
        "duplicate_source_keys_in_batch": duplicate_rows,
        "original_columns": original_columns,
        "normalized_columns": list(frame.columns),
        "minimum_valid_ratio": 0.95,
    }
    return valid, quarantine, metadata


def _list_objects(client, bucket: str, prefix: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            request["ContinuationToken"] = token
        page = client.list_objects_v2(**request)
        objects.extend(page.get("Contents", []))
        if not page.get("IsTruncated"):
            break
        token = page["NextContinuationToken"]
    return objects


def resolve_canonical_source(client, bucket: str = DEFAULT_BUCKET) -> dict[str, Any]:
    snapshots = [
        item
        for item in _list_objects(client, bucket, CANONICAL_PREFIX)
        if str(item["Key"]).endswith(".parquet")
    ]
    if snapshots:
        selected = max(snapshots, key=lambda item: str(item["Key"]))
        return {
            "bucket": bucket,
            "key": str(selected["Key"]),
            "kind": "B57G_IMMUTABLE_SNAPSHOT",
        }
    client.head_object(Bucket=bucket, Key=LEGACY_KEY)
    return {"bucket": bucket, "key": LEGACY_KEY, "kind": "LEGACY_SNAPSHOT"}


def _duckdb_type(value: str) -> str:
    allowed = re.sub(r"[^A-Za-z0-9_(), ]+", "", value)
    if not allowed:
        raise ValueError(f"Unsafe DuckDB type: {value}")
    return allowed


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _merge_canonical(
    base_path: Path,
    incoming: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    connection = duckdb.connect()
    try:
        base_literal = str(base_path).replace("'", "''")
        output_literal = str(output_path).replace("'", "''")
        connection.execute(
            f"CREATE TEMP VIEW base_raw AS "
            f"SELECT * FROM read_parquet('{base_literal}')"
        )
        duplicate_base_keys = int(
            connection.execute(
                """
                SELECT COUNT(*) - COUNT(DISTINCT CAST(SOURCE_ROW_INDEX AS VARCHAR))
                FROM base_raw
                WHERE SOURCE_ROW_INDEX IS NOT NULL
                """
            ).fetchone()[0]
        )
        if duplicate_base_keys:
            raise RuntimeError(
                "The current canonical source contains "
                f"{duplicate_base_keys} duplicate SOURCE_ROW_INDEX values"
            )
        incoming = incoming.copy()
        connection.register("incoming_frame", incoming)

        schema_rows = connection.execute("DESCRIBE base_raw").fetchall()
        base_types = {str(row[0]): _duckdb_type(str(row[1])) for row in schema_rows}
        incoming_schema = {
            str(row[0]): _duckdb_type(str(row[1]))
            for row in connection.execute("DESCRIBE incoming_frame").fetchall()
        }
        base_non_null_counts = connection.execute(
            "SELECT "
            + ", ".join(
                f"COUNT({_quote_identifier(str(row[0]))})" for row in schema_rows
            )
            + " FROM base_raw"
        ).fetchone()
        for index, row in enumerate(schema_rows):
            column = str(row[0])
            if (
                int(base_non_null_counts[index]) == 0
                and column in incoming_schema
                and bool(incoming[column].notna().any())
            ):
                base_types[column] = incoming_schema[column]

        base_columns = list(base_types)
        for column in METADATA_COLUMNS:
            if column not in base_types:
                base_types[column] = (
                    "TIMESTAMP WITH TIME ZONE"
                    if column == "B57G_AVAILABLE_AT"
                    else "VARCHAR"
                )
                base_columns.append(column)

        incoming_columns = set(incoming.columns)

        base_select = []
        incoming_select = []
        for column in base_columns:
            quoted = _quote_identifier(column)
            data_type = base_types[column]
            if column in {row[0] for row in schema_rows}:
                base_select.append(f"TRY_CAST({quoted} AS {data_type}) AS {quoted}")
            else:
                base_select.append(f"NULL::{data_type} AS {quoted}")
            if column in incoming_columns:
                incoming_select.append(
                    f"TRY_CAST({quoted} AS {data_type}) AS {quoted}"
                )
            else:
                incoming_select.append(f"NULL::{data_type} AS {quoted}")

        connection.execute(
            f"CREATE TEMP VIEW base_norm AS SELECT {', '.join(base_select)} FROM base_raw"
        )
        connection.execute(
            "CREATE TEMP VIEW incoming_norm AS SELECT "
            + ", ".join(incoming_select)
            + " FROM incoming_frame"
        )

        stats = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM base_norm) AS base_rows,
                (SELECT COUNT(*) FROM incoming_norm) AS incoming_rows,
                COUNT(*) FILTER (WHERE b.SOURCE_ROW_INDEX IS NULL) AS inserted_rows,
                COUNT(*) FILTER (WHERE b.SOURCE_ROW_INDEX IS NOT NULL) AS matched_rows
            FROM incoming_norm i
            LEFT JOIN base_norm b
              ON CAST(i.SOURCE_ROW_INDEX AS VARCHAR)
               = CAST(b.SOURCE_ROW_INDEX AS VARCHAR)
            """
        ).fetchone()

        select_columns = []
        for column in base_columns:
            quoted = _quote_identifier(column)
            select_columns.append(
                f"COALESCE(i.{quoted}, b.{quoted}) AS {quoted}"
            )
        key = _quote_identifier("SOURCE_ROW_INDEX")
        query = f"""
            COPY (
                SELECT {", ".join(select_columns)}
                FROM base_norm b
                FULL OUTER JOIN incoming_norm i
                  ON CAST(i.{key} AS VARCHAR) = CAST(b.{key} AS VARCHAR)
            ) TO '{output_literal}'
              (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
        """
        connection.execute(query)
        output_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM read_parquet(?)", [str(output_path)]
            ).fetchone()[0]
        )
        expected_rows = int(stats[0]) + int(stats[2])
        if output_rows != expected_rows:
            raise RuntimeError(
                "Canonical merge changed the expected grain: "
                f"expected {expected_rows} rows, wrote {output_rows}. "
                "Check duplicate SOURCE_ROW_INDEX values in the current canonical source."
            )
        return {
            "base_rows": int(stats[0]),
            "incoming_rows": int(stats[1]),
            "inserted_rows": int(stats[2]),
            "matched_rows": int(stats[3]),
            "output_rows": output_rows,
            "canonical_columns": base_columns,
        }
    finally:
        connection.close()


def _upload_file(
    client,
    path: Path,
    bucket: str,
    key: str,
    metadata: dict[str, str] | None = None,
) -> None:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    extra: dict[str, Any] = {"ContentType": content_type}
    if metadata:
        extra["Metadata"] = metadata
    client.upload_file(str(path), bucket, key, ExtraArgs=extra)


def ingest_tir_increment(
    source_bucket: str = DEFAULT_BUCKET,
    source_key: str = "",
    delete_source: bool = True,
    allowed_lateness_days: int = 3,
) -> dict[str, Any]:
    if not source_key.startswith(INCOMING_PREFIX):
        raise ValueError(f"Incoming objects must be under s3://{source_bucket}/{INCOMING_PREFIX}")
    if allowed_lateness_days < 0 or allowed_lateness_days > 31:
        raise ValueError("allowed_lateness_days must be between 0 and 31")

    client = _s3_client()
    source_head = client.head_object(Bucket=source_bucket, Key=source_key)
    received_at = source_head["LastModified"].astimezone(timezone.utc)
    suffix = Path(source_key).suffix.lower()
    if suffix not in {".csv", ".txt", ".parquet", ".pq"}:
        raise ValueError("Unsupported TIR input extension")

    with tempfile.TemporaryDirectory(prefix="b57g-") as directory:
        work = Path(directory)
        source_path = work / f"incoming{suffix}"
        checksum = _download_and_hash(
            client, source_bucket, source_key, source_path
        )
        previous = _previous_success(checksum)
        if previous is not None:
            if delete_source:
                client.delete_object(Bucket=source_bucket, Key=source_key)
            return {
                "status": "SUCCESS",
                "run_id": previous[0],
                "reused": True,
                "result": previous[1],
            }

        base = resolve_canonical_source(client, DEFAULT_BUCKET)
        audit_metadata = {
            "collector_version": COLLECTOR_VERSION,
            "received_at": received_at,
            "source_key": source_key,
            "canonical_before": base,
            "historical_backfill_used": False,
            "training_executed": False,
        }
        run_id = _start_audit(
            source_bucket, source_key, checksum, audit_metadata
        )
        try:
            raw = _read_input(source_path)
            valid, quarantine, validation = _normalize_input(
                raw, received_at, checksum
            )
            base_path = work / "base.parquet"
            client.download_file(base["bucket"], base["key"], str(base_path))

            with duckdb.connect() as connection:
                base_dates = connection.execute(
                    """
                    SELECT MIN(TRY_CAST(DATE_ZRE AS TIMESTAMPTZ)),
                           MAX(TRY_CAST(DATE_ZRE AS TIMESTAMPTZ))
                    FROM read_parquet(?)
                    """,
                    [str(base_path)],
                ).fetchone()
            base_max = pd.Timestamp(base_dates[1])
            if base_max.tzinfo is None:
                base_max = base_max.tz_localize("UTC")
            eligible_start = base_max.floor("D") - pd.Timedelta(
                days=allowed_lateness_days
            )
            eligible_mask = valid["DATE_ZRE"].ge(eligible_start)
            too_old = valid.loc[~eligible_mask].copy()
            eligible = valid.loc[eligible_mask].copy()
            if not too_old.empty:
                too_old["B57G_QUARANTINE_REASON"] = "OUTSIDE_LATENESS_WINDOW"
                quarantine = pd.concat([quarantine, too_old], ignore_index=True)

            raw_key = (
                f"{LANDING_PREFIX}received_date={received_at:%Y-%m-%d}/"
                f"sha256={checksum}/{_safe_name(source_key)}"
            )
            client.copy_object(
                Bucket=DEFAULT_BUCKET,
                Key=raw_key,
                CopySource={"Bucket": source_bucket, "Key": source_key},
                MetadataDirective="REPLACE",
                Metadata={
                    "sha256": checksum,
                    "received-at": received_at.isoformat(),
                    "collector-version": COLLECTOR_VERSION,
                },
            )

            quarantine_key = None
            if not quarantine.empty:
                quarantine_path = work / "quarantine.parquet"
                quarantine.to_parquet(quarantine_path, index=False)
                quarantine_key = (
                    "tir/quarantine/version=1/"
                    f"received_date={received_at:%Y-%m-%d}/"
                    f"sha256={checksum}/invalid_rows.parquet"
                )
                _upload_file(
                    client,
                    quarantine_path,
                    DEFAULT_BUCKET,
                    quarantine_key,
                    {"sha256": checksum},
                )

            if eligible.empty:
                decision = "ARCHIVED_NO_ELIGIBLE_ROWS"
                result = {
                    **validation,
                    "decision": decision,
                    "raw_archive_key": raw_key,
                    "quarantine_key": quarantine_key,
                    "base_max_date_zre": base_max,
                    "eligible_rows": 0,
                    "canonical_updated": False,
                    "next_block": "B57G_WAIT_FOR_FRESH_TIR_BATCH",
                }
                _finish_audit(run_id, "SUCCESS", 0, result)
                if delete_source:
                    client.delete_object(Bucket=source_bucket, Key=source_key)
                return {
                    "status": "SUCCESS",
                    "run_id": run_id,
                    "reused": False,
                    "result": result,
                }

            incoming_min = eligible["DATE_ZRE"].min()
            incoming_max = eligible["DATE_ZRE"].max()
            fresh_rows = int(eligible["DATE_ZRE"].gt(base_max).sum())
            gap_days = max(
                0, int((incoming_max.floor("D") - base_max.floor("D")).days) - 1
            )

            canonical_path = work / "tir_canonical.parquet"
            merge = _merge_canonical(base_path, eligible, canonical_path)
            token = _timestamp_token(received_at)
            canonical_key = (
                f"{CANONICAL_PREFIX}available_at={token}/"
                "tir_canonical.parquet"
            )
            _upload_file(
                client,
                canonical_path,
                DEFAULT_BUCKET,
                canonical_key,
                {
                    "sha256-source": checksum,
                    "available-at": received_at.isoformat(),
                    "collector-version": COLLECTOR_VERSION,
                },
            )

            manifest = {
                "collector_version": COLLECTOR_VERSION,
                "available_at": received_at.isoformat(),
                "source": f"s3://{source_bucket}/{source_key}",
                "source_checksum": checksum,
                "raw_archive": f"s3://{DEFAULT_BUCKET}/{raw_key}",
                "canonical_snapshot": f"s3://{DEFAULT_BUCKET}/{canonical_key}",
                "canonical_before": base,
                "validation": validation,
                "merge": merge,
                "incoming_min_date_zre": incoming_min.isoformat(),
                "incoming_max_date_zre": incoming_max.isoformat(),
                "base_max_date_zre": base_max.isoformat(),
                "fresh_rows": fresh_rows,
                "source_gap_days": gap_days,
                "missing_days_policy": "KEEP_TARGETS_NULL_NEVER_ZERO_FILL",
            }
            manifest_path = work / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, default=_json_default),
                encoding="utf-8",
            )
            manifest_key = (
                f"{CANONICAL_PREFIX}available_at={token}/manifest.json"
            )
            _upload_file(
                client, manifest_path, DEFAULT_BUCKET, manifest_key
            )

            if fresh_rows > 0 and gap_days > 0:
                decision = "CANONICAL_UPDATED_WITH_SOURCE_GAP"
            elif fresh_rows > 0:
                decision = "CANONICAL_UPDATED_FRESH"
            else:
                decision = "CANONICAL_UPDATED_CORRECTION_ONLY"
            result = {
                **validation,
                **merge,
                "decision": decision,
                "raw_archive_key": raw_key,
                "quarantine_key": quarantine_key,
                "canonical_key": canonical_key,
                "manifest_key": manifest_key,
                "eligible_rows": int(len(eligible)),
                "fresh_rows": fresh_rows,
                "source_gap_days": gap_days,
                "base_max_date_zre": base_max,
                "incoming_min_date_zre": incoming_min,
                "incoming_max_date_zre": incoming_max,
                "canonical_updated": True,
                "missing_days_zero_filled": False,
                "next_block": "B57B_REFRESH_THEN_B57F_OPERATIONAL_CYCLE",
            }
            _finish_audit(run_id, "SUCCESS", merge["output_rows"], result)
            if delete_source:
                client.delete_object(Bucket=source_bucket, Key=source_key)
            return {
                "status": "SUCCESS",
                "run_id": run_id,
                "reused": False,
                "result": result,
            }
        except Exception as exc:
            _finish_audit(
                run_id,
                "FAILED",
                None,
                {"decision": "NEED_TIR_SOURCE_REPAIR"},
                str(exc),
            )
            raise


def list_incoming_tir_objects(
    bucket: str = DEFAULT_BUCKET,
    limit: int = 20,
) -> list[dict[str, Any]]:
    client = _s3_client()
    objects = _list_objects(client, bucket, INCOMING_PREFIX)
    accepted = [
        item
        for item in objects
        if Path(str(item["Key"])).suffix.lower()
        in {".csv", ".txt", ".parquet", ".pq"}
    ]
    accepted.sort(key=lambda item: item["LastModified"])
    return [
        {
            "bucket": bucket,
            "key": str(item["Key"]),
            "size": int(item["Size"]),
            "last_modified": item["LastModified"].isoformat(),
        }
        for item in accepted[: max(1, min(limit, 100))]
    ]
