from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import duckdb
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values


SOURCE_NAME = "tir_data1"
DATASET_NAME = "data1_maritime_minimal"
SOURCE_TIMEZONE = os.getenv("SOURCE_TIMEZONE", "Africa/Casablanca")
DUCKDB_MEMORY_LIMIT = os.getenv("PORT_CALL_DUCKDB_MEMORY_LIMIT", "1400MB")

REQUIRED_COLUMNS = {
    "SOURCE_ROW_INDEX",
    "NO_AMP",
    "UNITE",
    "IMO_NAVIRE",
    "NOM_NAVIRE",
    "ESCALE",
    "ETA",
    "ETD",
    "RTA",
    "RTD",
    "DATE_ZRE",
    "DATE_EMBARQUEMENT",
}


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def _download(client, bucket: str, key: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(destination))


def _upload(client, path: Path, bucket: str, key: str, content_type: str) -> str:
    client.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"ContentType": content_type},
    )
    return f"s3://{bucket}/{key}"


def _start_ingestion_run(source_uri: str, checksum: str) -> str:
    with db_connection() as connection:
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
                    Json({"pipeline": "port-calls-silver-v1"}),
                ),
            )
            return str(cursor.fetchone()[0])


def _already_processed(checksum: str) -> bool:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM audit.ingestion_run
                    WHERE source_name = %s
                      AND dataset_name = %s
                      AND checksum = %s
                      AND status = 'SUCCESS'
                )
                """,
                (SOURCE_NAME, DATASET_NAME, checksum),
            )
            return bool(cursor.fetchone()[0])


def _finish_ingestion_run(
    run_id: str,
    status: str,
    row_count: int | None = None,
    metadata: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> None:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE audit.ingestion_run
                SET finished_at = now(),
                    status = %s,
                    row_count = %s,
                    metadata = metadata || %s,
                    error_message = %s
                WHERE run_id = %s
                """,
                (
                    status,
                    row_count,
                    Json(metadata or {}),
                    error_message,
                    run_id,
                ),
            )


def _validate_manifest(manifest: dict[str, Any], actual_checksum: str) -> None:
    expected = str(manifest.get("sha256", "")).strip().lower()
    if not expected:
        raise RuntimeError("Manifest does not contain a sha256 checksum")
    if expected != actual_checksum.lower():
        raise RuntimeError(
            f"SHA256 mismatch: manifest={expected}, actual={actual_checksum}"
        )


def _configure_duckdb(connection: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    connection.execute(f"SET temp_directory='{_sql_path(temp_dir)}'")
    connection.execute("SET threads=2")
    connection.execute(f"SET TimeZone='{SOURCE_TIMEZONE}'")


def _create_transformed_tables(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
) -> None:
    source_sql = _sql_path(source_path)
    connection.execute(
        f"CREATE VIEW raw_source AS SELECT * FROM read_parquet('{source_sql}')"
    )
    available = {
        row[0] for row in connection.execute("DESCRIBE raw_source").fetchall()
    }
    missing = sorted(REQUIRED_COLUMNS.difference(available))
    if missing:
        raise RuntimeError(f"Required Bronze columns are missing: {missing}")

    connection.execute(
        """
        CREATE OR REPLACE MACRO clean_text(value) AS
        CASE
            WHEN value IS NULL THEN NULL
            WHEN upper(trim(CAST(value AS VARCHAR))) IN
                ('', 'NULL', 'NAN', 'NONE', 'N/A', 'NA', '<NA>') THEN NULL
            ELSE trim(CAST(value AS VARCHAR))
        END
        """
    )

    connection.execute(
        """
        CREATE TABLE normalized AS
        WITH typed AS (
            SELECT
                CAST(SOURCE_ROW_INDEX AS BIGINT) AS source_row_index,
                clean_text(NO_AMP) AS no_amp,
                clean_text(UNITE) AS unite,
                clean_text(NO_ZRE) AS no_zre,
                clean_text(IMO_NAVIRE) AS imo_raw,
                upper(regexp_replace(clean_text(NOM_NAVIRE), '\\s+', ' ', 'g'))
                    AS vessel_name,
                upper(regexp_replace(clean_text(ESCALE), '\\s+', '', 'g')) AS escale,
                try_cast(clean_text(ETA) AS TIMESTAMPTZ) AS planned_eta,
                try_cast(clean_text(ETD) AS TIMESTAMPTZ) AS planned_etd,
                try_cast(clean_text(RTA) AS TIMESTAMPTZ) AS actual_ata,
                try_cast(clean_text(RTD) AS TIMESTAMPTZ) AS actual_atd,
                try_cast(clean_text(DATE_ZRE) AS TIMESTAMPTZ) AS date_zre,
                try_cast(clean_text(DATE_EMBARQUEMENT) AS TIMESTAMPTZ)
                    AS date_embarquement,
                clean_text(TYPE_UNITE) AS type_unite,
                clean_text(SS_TYPE_UNITE) AS ss_type_unite,
                clean_text(DECLARANT) AS declarant,
                clean_text(TRANSPORTEUR_UNITE) AS transporteur_unite,
                clean_text(VIDE_PLEIN) AS vide_plein,
                clean_text(NATURE_MARCHANDISE) AS nature_marchandise,
                try_cast(POIDS AS DOUBLE) AS poids_raw,
                CAST(MATIERE_DANGER AS BOOLEAN) AS matiere_danger,
                clean_text(COULOIR) AS couloir,
                clean_text(ETAT_DECHARGEMENT) AS etat_dechargement,
                clean_text(GROUPAGE) AS groupage_raw
            FROM raw_source
        ), normalized_ids AS (
            SELECT
                *,
                CASE
                    WHEN try_cast(imo_raw AS DOUBLE) BETWEEN 1000000 AND 9999999
                    THEN printf('%.0f', try_cast(imo_raw AS DOUBLE))
                    ELSE regexp_replace(imo_raw, '\\.0$', '')
                END AS imo_candidate
            FROM typed
        )
        SELECT
            * EXCLUDE (imo_candidate),
            CASE
                WHEN regexp_matches(imo_candidate, '^[0-9]{7}$')
                THEN imo_candidate
                ELSE NULL
            END AS imo,
            CASE
                WHEN poids_raw BETWEEN 0 AND 100000 THEN poids_raw
                ELSE NULL
            END AS poids_kg,
            poids_raw IS NOT NULL AND NOT (poids_raw BETWEEN 0 AND 100000)
                AS invalid_weight_flag,
            groupage_raw IS NOT NULL AS is_groupage
        FROM normalized_ids
        """
    )

    connection.execute(
        """
        CREATE TABLE enriched AS
        WITH identities AS (
            SELECT
                *,
                CASE
                    WHEN imo IS NOT NULL THEN imo
                    WHEN vessel_name IS NOT NULL
                        THEN 'NAME_' || substr(md5(vessel_name), 1, 16)
                    ELSE NULL
                END AS vessel_id,
                imo IS NOT NULL AS has_imo,
                vessel_name IS NOT NULL AS has_vessel_name,
                escale IS NOT NULL AS has_escale,
                planned_eta IS NOT NULL AS has_eta,
                actual_ata IS NOT NULL AS has_rta,
                planned_etd IS NOT NULL AS has_etd,
                actual_atd IS NOT NULL AS has_rtd,
                poids_kg IS NOT NULL AS has_weight
            FROM normalized
        )
        SELECT
            *,
            CASE
                WHEN vessel_id IS NOT NULL AND escale IS NOT NULL
                THEN vessel_id || ':' || escale
                ELSE NULL
            END AS port_call_key,
            has_eta AND has_rta AS has_arrival_label,
            has_etd AND has_rtd AS has_departure_label,
            CASE
                WHEN has_eta AND has_rta
                THEN date_diff('second', planned_eta, actual_ata) / 3600.0
                ELSE NULL
            END AS arrival_delay_h,
            CASE
                WHEN has_etd AND has_rtd
                THEN date_diff('second', planned_etd, actual_atd) / 3600.0
                ELSE NULL
            END AS departure_delay_h,
            concat_ws('|',
                CASE WHEN vessel_id IS NULL THEN 'MISSING_VESSEL_ID' END,
                CASE WHEN escale IS NULL THEN 'MISSING_ESCALE' END,
                CASE WHEN imo_raw IS NOT NULL AND imo IS NULL THEN 'INVALID_IMO' END,
                CASE WHEN invalid_weight_flag THEN 'INVALID_WEIGHT' END
            ) AS quality_reasons
        FROM identities
        """
    )

    connection.execute(
        """
        CREATE TABLE port_calls AS
        WITH aggregated AS (
            SELECT
                port_call_key,
                mode(escale) AS escale,
                mode(vessel_id) AS vessel_id,
                mode(imo) AS imo,
                mode(vessel_name) AS vessel_name,
                min(planned_eta) AS planned_eta,
                min(planned_etd) AS planned_etd,
                min(actual_ata) AS actual_ata,
                min(actual_atd) AS actual_atd,
                mode(nature_marchandise) AS cargo_type,
                mode(type_unite) AS vessel_type,
                count(*) AS n_tir_units,
                count(poids_kg) AS n_weight_known,
                sum(poids_kg) AS total_weight_kg,
                avg(poids_kg) AS mean_weight_kg,
                100.0 * count(poids_kg) / count(*) AS weight_coverage_pct,
                100.0 * count(*) FILTER (WHERE upper(vide_plein) = 'PLEIN')
                    / count(*) AS pct_full,
                100.0 * count(*) FILTER (WHERE matiere_danger)
                    / count(*) AS pct_dangerous,
                count(DISTINCT planned_eta) AS n_distinct_eta,
                count(DISTINCT actual_ata) AS n_distinct_rta,
                min(source_row_index) AS first_source_row_index,
                min(date_zre) AS first_date_zre,
                max(date_embarquement) AS last_date_embarquement
            FROM enriched
            WHERE port_call_key IS NOT NULL
            GROUP BY port_call_key
        )
        SELECT
            *,
            planned_eta IS NOT NULL AND actual_ata IS NOT NULL AS has_arrival_label,
            planned_etd IS NOT NULL AND actual_atd IS NOT NULL AS has_departure_label,
            CASE
                WHEN planned_eta IS NOT NULL AND actual_ata IS NOT NULL
                THEN date_diff('second', planned_eta, actual_ata) / 3600.0
            END AS arrival_delay_h,
            CASE
                WHEN planned_etd IS NOT NULL AND actual_atd IS NOT NULL
                THEN date_diff('second', planned_etd, actual_atd) / 3600.0
            END AS departure_delay_h
        FROM aggregated
        """
    )


def _write_outputs(
    connection: duckdb.DuckDBPyConnection,
    work_dir: Path,
) -> dict[str, Path]:
    paths = {
        "units": work_dir / "tir_unit_events_clean_v1.parquet",
        "port_calls": work_dir / "port_calls_clean_v1.parquet",
        "quarantine": work_dir / "invalid_or_unlinked_rows_v1.parquet",
    }
    connection.execute(
        f"COPY (SELECT * FROM enriched WHERE port_call_key IS NOT NULL) "
        f"TO '{_sql_path(paths['units'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    connection.execute(
        f"COPY (SELECT * FROM port_calls) TO '{_sql_path(paths['port_calls'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY (SELECT * FROM enriched WHERE port_call_key IS NULL) "
        f"TO '{_sql_path(paths['quarantine'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)"
    )
    return paths


def _quality_report(connection: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    metrics = connection.execute(
        """
        SELECT
            count(*) AS source_rows,
            count(*) FILTER (WHERE port_call_key IS NOT NULL) AS linked_rows,
            count(*) FILTER (WHERE port_call_key IS NULL) AS quarantine_rows,
            count(DISTINCT port_call_key) AS port_calls,
            count(*) FILTER (WHERE has_arrival_label) AS rows_with_arrival_label,
            count(*) FILTER (WHERE has_departure_label) AS rows_with_departure_label,
            count(*) FILTER (WHERE has_imo) AS rows_with_valid_imo,
            count(*) FILTER (WHERE invalid_weight_flag) AS rows_invalid_weight
        FROM enriched
        """
    ).fetchone()
    columns = [item[0] for item in connection.description]
    report = dict(zip(columns, [int(value or 0) for value in metrics]))
    source_rows = max(1, report["source_rows"])
    report["linked_pct"] = 100.0 * report["linked_rows"] / source_rows
    report["arrival_label_pct"] = (
        100.0 * report["rows_with_arrival_label"] / source_rows
    )
    report["departure_label_pct"] = (
        100.0 * report["rows_with_departure_label"] / source_rows
    )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["source_timezone_assumption"] = SOURCE_TIMEZONE
    return report


def _none_if_missing(value):
    return None if pd.isna(value) else value


def _upsert_port_calls(path: Path) -> int:
    frame = pd.read_parquet(path)
    if frame.empty:
        return 0

    rows = []
    for row in frame.itertuples(index=False):
        imo_value = _none_if_missing(row.imo)
        rows.append(
            (
                "MAPTM",
                int(imo_value) if imo_value is not None else None,
                _none_if_missing(row.vessel_name),
                _none_if_missing(row.escale),
                _none_if_missing(row.planned_eta),
                _none_if_missing(row.planned_etd),
                _none_if_missing(row.actual_ata),
                _none_if_missing(row.actual_atd),
                _none_if_missing(row.cargo_type),
                _none_if_missing(row.vessel_type),
                SOURCE_NAME,
                row.port_call_key,
            )
        )

    query = """
        INSERT INTO core.port_call (
            port_code, imo, vessel_name, voyage_id,
            planned_eta, planned_etd, actual_ata, actual_atd,
            cargo_type, vessel_type, source, source_record_id
        ) VALUES %s
        ON CONFLICT (source, source_record_id) DO UPDATE SET
            imo = EXCLUDED.imo,
            vessel_name = EXCLUDED.vessel_name,
            voyage_id = EXCLUDED.voyage_id,
            planned_eta = EXCLUDED.planned_eta,
            planned_etd = EXCLUDED.planned_etd,
            actual_ata = EXCLUDED.actual_ata,
            actual_atd = EXCLUDED.actual_atd,
            cargo_type = EXCLUDED.cargo_type,
            vessel_type = EXCLUDED.vessel_type,
            updated_at = now()
    """
    with db_connection() as connection:
        with connection.cursor() as cursor:
            execute_values(cursor, query, rows, page_size=5000)
    return len(rows)


def process_port_call_object(
    source_bucket: str,
    source_key: str,
    manifest_key: str,
    output_bucket: str = "silver-maritime",
    output_prefix: str = "",
    force: bool = False,
) -> dict[str, Any]:
    client = s3_client()
    source_uri = f"s3://{source_bucket}/{source_key}"
    run_id: str | None = None

    with tempfile.TemporaryDirectory(prefix="smart-port-calls-") as temp_name:
        temp_dir = Path(temp_name)
        source_path = temp_dir / "source.parquet"
        manifest_path = temp_dir / "manifest.json"
        _download(client, source_bucket, source_key, source_path)
        _download(client, source_bucket, manifest_key, manifest_path)

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checksum = _sha256(source_path)
        _validate_manifest(manifest, checksum)
        if not force and _already_processed(checksum):
            return {
                "status": "SKIPPED",
                "reason": "checksum_already_processed",
                "source_uri": source_uri,
                "checksum": checksum,
            }

        run_id = _start_ingestion_run(source_uri, checksum)
        try:
            database_path = temp_dir / "port_calls.duckdb"
            connection = duckdb.connect(str(database_path))
            try:
                _configure_duckdb(connection, temp_dir / "duckdb-tmp")
                _create_transformed_tables(connection, source_path)
                output_paths = _write_outputs(connection, temp_dir)
                report = _quality_report(connection)
            finally:
                connection.close()

            prefix = output_prefix.strip("/") or "version=1"
            object_keys = {
                "units": f"tir/{prefix}/tir_unit_events_clean_v1.parquet",
                "port_calls": f"port_calls/{prefix}/port_calls_clean_v1.parquet",
                "quarantine": (
                    f"quarantine/{prefix}/invalid_or_unlinked_rows_v1.parquet"
                ),
                "report": f"audits/{prefix}/port_call_quality_report_v1.json",
            }
            report.update(
                {
                    "run_id": run_id,
                    "source_uri": source_uri,
                    "source_checksum": checksum,
                    "output_bucket": output_bucket,
                }
            )
            report_path = temp_dir / "port_call_quality_report_v1.json"
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )

            output_uris = {
                "units": _upload(
                    client,
                    output_paths["units"],
                    output_bucket,
                    object_keys["units"],
                    "application/x-parquet",
                ),
                "port_calls": _upload(
                    client,
                    output_paths["port_calls"],
                    output_bucket,
                    object_keys["port_calls"],
                    "application/x-parquet",
                ),
                "quarantine": _upload(
                    client,
                    output_paths["quarantine"],
                    output_bucket,
                    object_keys["quarantine"],
                    "application/x-parquet",
                ),
                "report": _upload(
                    client,
                    report_path,
                    output_bucket,
                    object_keys["report"],
                    "application/json",
                ),
            }
            database_rows = _upsert_port_calls(output_paths["port_calls"])
            report["database_port_calls_upserted"] = database_rows
            report["output_uris"] = output_uris
            _finish_ingestion_run(
                run_id,
                "SUCCESS",
                row_count=report["source_rows"],
                metadata=report,
            )
            return {
                "status": "SUCCESS",
                "run_id": run_id,
                "source_uri": source_uri,
                "checksum": checksum,
                "quality": report,
                "outputs": output_uris,
            }
        except Exception as exc:
            if run_id is not None:
                _finish_ingestion_run(run_id, "FAILED", error_message=str(exc))
            raise
