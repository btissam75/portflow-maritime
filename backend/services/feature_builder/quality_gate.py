from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import boto3
import duckdb
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values


SOURCE_NAME = "tir_data1_silver"
DATASET_NAME = "port_calls_revision_quality_gate"
CANONICAL_SOURCE_NAME = "tir_data1"
DUCKDB_MEMORY_LIMIT = os.getenv("PORT_CALL_DUCKDB_MEMORY_LIMIT", "1400MB")
EXTREME_DELAY_HOURS = float(os.getenv("PORT_CALL_EXTREME_DELAY_HOURS", "720"))

REQUIRED_COLUMNS = {
    "source_row_index",
    "port_call_key",
    "escale",
    "vessel_id",
    "imo",
    "vessel_name",
    "planned_eta",
    "planned_etd",
    "actual_ata",
    "actual_atd",
    "date_zre",
    "date_embarquement",
    "nature_marchandise",
    "type_unite",
    "poids_kg",
    "vide_plein",
    "matiere_danger",
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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(client, bucket: str, key: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(target))


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


def _json_default(value: Any):
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _start_run(source_uri: str, checksum: str) -> str:
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
                    Json({"pipeline": "b54b-revision-aware-quality-gate-v1"}),
                ),
            )
            return str(cursor.fetchone()[0])


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with db_connection() as connection:
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


def _finish_run(
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
                    Json(
                        metadata or {},
                        dumps=lambda obj: json.dumps(obj, default=_json_default),
                    ),
                    error_message,
                    run_id,
                ),
            )


def _wave_bounds() -> tuple[datetime | None, datetime | None]:
    with db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT min(observed_at), max(observed_at)
                FROM core.maritime_observation
                """
            )
            lower, upper = cursor.fetchone()
            return lower, upper


def _configure_duckdb(connection: duckdb.DuckDBPyConnection, temp_dir: Path) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}'")
    connection.execute("SET threads = 2")
    connection.execute("SET preserve_insertion_order = false")
    escaped = str(temp_dir).replace("'", "''")
    connection.execute(f"SET temp_directory = '{escaped}'")


def _sql_path(path: Path) -> str:
    return str(path).replace("'", "''")


def _build_quality_tables(
    connection: duckdb.DuckDBPyConnection,
    source_path: Path,
    wave_min: datetime | None,
    wave_max: datetime | None,
) -> None:
    source_sql = _sql_path(source_path)
    connection.execute(
        f"CREATE VIEW unit_events AS SELECT * FROM read_parquet('{source_sql}')"
    )
    available = {
        row[0] for row in connection.execute("DESCRIBE unit_events").fetchall()
    }
    missing = sorted(REQUIRED_COLUMNS.difference(available))
    if missing:
        raise RuntimeError(f"Required Silver columns are missing: {missing}")

    connection.execute(
        "CREATE TABLE wave_bounds(wave_min TIMESTAMPTZ, wave_max TIMESTAMPTZ)"
    )
    connection.execute(
        "INSERT INTO wave_bounds VALUES (?, ?)",
        [wave_min, wave_max],
    )

    connection.execute(
        f"""
        CREATE TABLE canonical_port_calls AS
        WITH aggregated AS (
            SELECT
                port_call_key,
                mode(escale) AS escale,
                mode(vessel_id) AS vessel_id,
                mode(imo) AS imo,
                mode(vessel_name) AS vessel_name,

                mode(planned_eta) AS planned_eta,
                min(planned_eta) AS planned_eta_min,
                max(planned_eta) AS planned_eta_max,
                count(DISTINCT planned_eta) AS n_distinct_eta,

                mode(planned_etd) AS planned_etd,
                min(planned_etd) AS planned_etd_min,
                max(planned_etd) AS planned_etd_max,
                count(DISTINCT planned_etd) AS n_distinct_etd,

                mode(actual_ata) AS actual_ata,
                min(actual_ata) AS actual_ata_min,
                max(actual_ata) AS actual_ata_max,
                count(DISTINCT actual_ata) AS n_distinct_rta,

                mode(actual_atd) AS actual_atd,
                min(actual_atd) AS actual_atd_min,
                max(actual_atd) AS actual_atd_max,
                count(DISTINCT actual_atd) AS n_distinct_rtd,

                mode(nature_marchandise) AS cargo_type,
                mode(type_unite) AS vessel_type,
                count(*) AS n_tir_units,
                count(poids_kg) AS n_weight_known,
                sum(poids_kg) AS total_weight_kg,
                avg(poids_kg) AS mean_weight_kg,
                100.0 * count(poids_kg) / nullif(count(*), 0)
                    AS weight_coverage_pct,
                100.0 * count(*) FILTER (WHERE upper(vide_plein) = 'PLEIN')
                    / nullif(count(*), 0) AS pct_full,
                100.0 * count(*) FILTER (WHERE matiere_danger)
                    / nullif(count(*), 0) AS pct_dangerous,
                min(source_row_index) AS first_source_row_index,
                min(date_zre) AS first_date_zre,
                max(date_zre) AS last_date_zre,
                max(date_embarquement) AS last_date_embarquement
            FROM unit_events
            WHERE port_call_key IS NOT NULL
            GROUP BY port_call_key
        ), labeled AS (
            SELECT
                *,
                planned_eta IS NOT NULL AND actual_ata IS NOT NULL
                    AS has_arrival_label,
                planned_etd IS NOT NULL AND actual_atd IS NOT NULL
                    AS has_departure_label,
                CASE
                    WHEN planned_eta IS NOT NULL AND actual_ata IS NOT NULL
                    THEN date_diff('second', planned_eta, actual_ata) / 3600.0
                END AS arrival_delay_h,
                CASE
                    WHEN planned_etd IS NOT NULL AND actual_atd IS NOT NULL
                    THEN date_diff('second', planned_etd, actual_atd) / 3600.0
                END AS departure_delay_h,
                CASE
                    WHEN planned_eta_min IS NOT NULL AND planned_eta_max IS NOT NULL
                    THEN date_diff('second', planned_eta_min, planned_eta_max) / 3600.0
                END AS eta_value_spread_h,
                CASE
                    WHEN planned_etd_min IS NOT NULL AND planned_etd_max IS NOT NULL
                    THEN date_diff('second', planned_etd_min, planned_etd_max) / 3600.0
                END AS etd_value_spread_h,
                CASE
                    WHEN actual_ata_min IS NOT NULL AND actual_ata_max IS NOT NULL
                    THEN date_diff('second', actual_ata_min, actual_ata_max) / 3600.0
                END AS rta_value_spread_h,
                CASE
                    WHEN actual_atd_min IS NOT NULL AND actual_atd_max IS NOT NULL
                    THEN date_diff('second', actual_atd_min, actual_atd_max) / 3600.0
                END AS rtd_value_spread_h
            FROM aggregated
        )
        SELECT
            labeled.*,
            n_distinct_eta > 1 AS multiple_eta_flag,
            n_distinct_etd > 1 AS multiple_etd_flag,
            n_distinct_rta > 1 AS multiple_rta_flag,
            n_distinct_rtd > 1 AS multiple_rtd_flag,
            planned_eta IS NOT NULL AND planned_etd IS NOT NULL
                AND planned_etd < planned_eta AS invalid_planned_sequence,
            actual_ata IS NOT NULL AND actual_atd IS NOT NULL
                AND actual_atd < actual_ata AS invalid_actual_sequence,
            arrival_delay_h IS NOT NULL
                AND abs(arrival_delay_h) > {EXTREME_DELAY_HOURS}
                AS arrival_delay_abs_gt_30d_flag,
            departure_delay_h IS NOT NULL
                AND abs(departure_delay_h) > {EXTREME_DELAY_HOURS}
                AS departure_delay_abs_gt_30d_flag,
            planned_eta IS NOT NULL
                AND wave_min IS NOT NULL
                AND planned_eta BETWEEN wave_min AND wave_max
                AS has_arrival_wave_coverage,
            planned_etd IS NOT NULL
                AND wave_min IS NOT NULL
                AND planned_etd BETWEEN wave_min AND wave_max
                AS has_departure_wave_coverage,
            'MODE_ACROSS_TIR_UNITS' AS canonicalization_strategy,
            concat_ws('|',
                CASE WHEN n_distinct_eta > 1 THEN 'MULTIPLE_ETA' END,
                CASE WHEN n_distinct_etd > 1 THEN 'MULTIPLE_ETD' END,
                CASE WHEN n_distinct_rta > 1 THEN 'MULTIPLE_RTA' END,
                CASE WHEN n_distinct_rtd > 1 THEN 'MULTIPLE_RTD' END,
                CASE
                    WHEN planned_eta IS NOT NULL AND planned_etd IS NOT NULL
                         AND planned_etd < planned_eta
                    THEN 'INVALID_PLANNED_SEQUENCE'
                END,
                CASE
                    WHEN actual_ata IS NOT NULL AND actual_atd IS NOT NULL
                         AND actual_atd < actual_ata
                    THEN 'INVALID_ACTUAL_SEQUENCE'
                END,
                CASE
                    WHEN arrival_delay_h IS NOT NULL
                         AND abs(arrival_delay_h) > {EXTREME_DELAY_HOURS}
                    THEN 'ARRIVAL_DELAY_ABS_GT_30D'
                END,
                CASE
                    WHEN departure_delay_h IS NOT NULL
                         AND abs(departure_delay_h) > {EXTREME_DELAY_HOURS}
                    THEN 'DEPARTURE_DELAY_ABS_GT_30D'
                END,
                CASE
                    WHEN planned_eta IS NOT NULL
                         AND NOT (
                             wave_min IS NOT NULL
                             AND planned_eta BETWEEN wave_min AND wave_max
                         )
                    THEN 'NO_ARRIVAL_WAVE_COVERAGE'
                END
            ) AS quality_reasons
        FROM labeled
        CROSS JOIN wave_bounds
        """
    )

    connection.execute(
        """
        CREATE TABLE chronology_review AS
        SELECT *
        FROM canonical_port_calls
        WHERE invalid_planned_sequence
           OR invalid_actual_sequence
           OR arrival_delay_abs_gt_30d_flag
           OR departure_delay_abs_gt_30d_flag
        """
    )
    connection.execute(
        """
        CREATE TABLE revision_review AS
        SELECT *
        FROM canonical_port_calls
        WHERE multiple_eta_flag
           OR multiple_etd_flag
           OR multiple_rta_flag
           OR multiple_rtd_flag
        """
    )


def _fetch_dict(connection: duckdb.DuckDBPyConnection, query: str) -> dict[str, Any]:
    row = connection.execute(query).fetchone()
    columns = [item[0] for item in connection.description]
    return dict(zip(columns, row))


def _delay_stats(
    connection: duckdb.DuckDBPyConnection,
    column: str,
) -> dict[str, Any]:
    return _fetch_dict(
        connection,
        f"""
        SELECT
            count({column}) AS n,
            avg({column}) AS mean_h,
            stddev_samp({column}) AS std_h,
            min({column}) AS min_h,
            quantile_cont({column}, 0.01) AS p01_h,
            quantile_cont({column}, 0.05) AS p05_h,
            quantile_cont({column}, 0.50) AS p50_h,
            quantile_cont({column}, 0.95) AS p95_h,
            quantile_cont({column}, 0.99) AS p99_h,
            max({column}) AS max_h,
            100.0 * count(*) FILTER (WHERE {column} < -1)
                / nullif(count({column}), 0) AS early_gt1h_pct,
            100.0 * count(*) FILTER (WHERE abs({column}) <= 1)
                / nullif(count({column}), 0) AS within_1h_pct,
            100.0 * count(*) FILTER (WHERE {column} > 1)
                / nullif(count({column}), 0) AS late_gt1h_pct
        FROM canonical_port_calls
        """,
    )


def _quality_report(
    connection: duckdb.DuckDBPyConnection,
    wave_min: datetime | None,
    wave_max: datetime | None,
) -> dict[str, Any]:
    report = _fetch_dict(
        connection,
        """
        SELECT
            count(*) AS port_calls,
            count(*) FILTER (WHERE has_arrival_label) AS arrival_labeled_calls,
            count(*) FILTER (WHERE has_departure_label) AS departure_labeled_calls,
            100.0 * count(*) FILTER (WHERE has_arrival_label)
                / nullif(count(*), 0) AS arrival_label_pct_calls,
            100.0 * count(*) FILTER (WHERE has_departure_label)
                / nullif(count(*), 0) AS departure_label_pct_calls,
            count(*) FILTER (WHERE has_arrival_wave_coverage)
                AS arrival_wave_covered_calls,
            count(*) FILTER (WHERE has_departure_wave_coverage)
                AS departure_wave_covered_calls,
            100.0 * count(*) FILTER (WHERE has_arrival_wave_coverage)
                / nullif(count(*), 0) AS arrival_wave_coverage_pct_calls,
            100.0 * count(*) FILTER (WHERE has_departure_wave_coverage)
                / nullif(count(*), 0) AS departure_wave_coverage_pct_calls,
            count(*) FILTER (WHERE multiple_eta_flag) AS multiple_eta_calls,
            count(*) FILTER (WHERE multiple_etd_flag) AS multiple_etd_calls,
            count(*) FILTER (WHERE multiple_rta_flag) AS multiple_rta_calls,
            count(*) FILTER (WHERE multiple_rtd_flag) AS multiple_rtd_calls,
            100.0 * count(*) FILTER (WHERE multiple_eta_flag)
                / nullif(count(*), 0) AS multiple_eta_pct_calls,
            100.0 * count(*) FILTER (WHERE multiple_rta_flag)
                / nullif(count(*), 0) AS multiple_rta_pct_calls,
            count(*) FILTER (WHERE invalid_planned_sequence)
                AS invalid_planned_sequence_calls,
            count(*) FILTER (WHERE invalid_actual_sequence)
                AS invalid_actual_sequence_calls,
            count(*) FILTER (WHERE arrival_delay_abs_gt_30d_flag)
                AS arrival_delay_abs_gt_30d_calls,
            count(*) FILTER (WHERE departure_delay_abs_gt_30d_flag)
                AS departure_delay_abs_gt_30d_calls,
            count(*) FILTER (
                WHERE invalid_planned_sequence
                   OR invalid_actual_sequence
                   OR arrival_delay_abs_gt_30d_flag
                   OR departure_delay_abs_gt_30d_flag
            ) AS chronology_review_calls,
            count(*) FILTER (
                WHERE multiple_eta_flag
                   OR multiple_etd_flag
                   OR multiple_rta_flag
                   OR multiple_rtd_flag
            ) AS revision_review_calls,
            min(planned_eta) AS first_planned_eta,
            max(planned_eta) AS last_planned_eta
        FROM canonical_port_calls
        """,
    )
    report["arrival_delay_distribution"] = _delay_stats(
        connection, "arrival_delay_h"
    )
    report["departure_delay_distribution"] = _delay_stats(
        connection, "departure_delay_h"
    )
    report["wave_min"] = wave_min
    report["wave_max"] = wave_max
    report["generated_at_utc"] = datetime.now(timezone.utc)
    report["target_note"] = (
        "Negative delay is valid and means early arrival/departure. "
        "Target-derived columns and quality flags are audit/label columns, "
        "not model features."
    )
    report["canonicalization_note"] = (
        "Canonical ETA/ETD/RTA/RTD use the mode across TIR units. "
        "Min/max, distinct counts and spreads are retained for revision audit."
    )

    total = max(1, int(report["port_calls"] or 0))
    invalid_pct = 100.0 * (
        int(report["invalid_planned_sequence_calls"] or 0)
        + int(report["invalid_actual_sequence_calls"] or 0)
    ) / total
    arrival_ready = (
        float(report["arrival_label_pct_calls"] or 0) >= 80.0
        and float(report["arrival_wave_coverage_pct_calls"] or 0) >= 80.0
        and invalid_pct < 1.0
    )
    departure_ready = (
        float(report["departure_label_pct_calls"] or 0) >= 60.0
        and float(report["departure_wave_coverage_pct_calls"] or 0) >= 60.0
        and invalid_pct < 1.0
    )
    multi_eta_pct = float(report["multiple_eta_pct_calls"] or 0)
    revision_risk = "LOW" if multi_eta_pct <= 1 else "MEDIUM"
    if multi_eta_pct > 5:
        revision_risk = "HIGH"
    report["decision"] = {
        "arrival_target_ready": arrival_ready,
        "departure_target_ready": departure_ready,
        "revision_risk": revision_risk,
        "next_block": (
            "B54C_TEMPORAL_WAVE_JOIN"
            if arrival_ready
            else "REVIEW_PORT_CALL_TARGET_QUALITY"
        ),
    }
    return report


def _write_outputs(
    connection: duckdb.DuckDBPyConnection,
    work_dir: Path,
) -> dict[str, Path]:
    outputs = {
        "quality_calls": work_dir / "port_calls_quality_checked_v1.parquet",
        "chronology_review": work_dir / "port_calls_chronology_review_v1.parquet",
        "revision_review": work_dir / "port_call_revision_conflicts_v1.parquet",
    }
    connection.execute(
        f"COPY canonical_port_calls TO '{_sql_path(outputs['quality_calls'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY chronology_review TO '{_sql_path(outputs['chronology_review'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    connection.execute(
        f"COPY revision_review TO '{_sql_path(outputs['revision_review'])}' "
        "(FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return outputs


def _none_if_missing(value):
    return None if pd.isna(value) else value


def _upsert_canonical_port_calls(path: Path) -> int:
    frame = pd.read_parquet(path)
    if frame.empty:
        return 0
    rows = []
    for row in frame.itertuples(index=False):
        imo = _none_if_missing(row.imo)
        rows.append(
            (
                "MAPTM",
                int(imo) if imo is not None else None,
                _none_if_missing(row.vessel_name),
                _none_if_missing(row.escale),
                _none_if_missing(row.planned_eta),
                _none_if_missing(row.planned_etd),
                _none_if_missing(row.actual_ata),
                _none_if_missing(row.actual_atd),
                _none_if_missing(row.cargo_type),
                _none_if_missing(row.vessel_type),
                CANONICAL_SOURCE_NAME,
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


def process_port_call_quality_gate(
    source_bucket: str,
    source_key: str,
    output_bucket: str = "silver-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    client = s3_client()
    source_uri = f"s3://{source_bucket}/{source_key}"
    run_id: str | None = None
    with tempfile.TemporaryDirectory(prefix="b54b-") as temporary:
        work_dir = Path(temporary)
        source_path = work_dir / "tir_unit_events_clean_v1.parquet"
        _download(client, source_bucket, source_key, source_path)
        checksum = _hash_file(source_path)

        if not force:
            previous = _previous_success(checksum)
            if previous is not None:
                previous_run_id, metadata = previous
                return {
                    "status": "SKIPPED_ALREADY_PROCESSED",
                    "run_id": previous_run_id,
                    "source_uri": source_uri,
                    "checksum": checksum,
                    "quality": metadata,
                    "outputs": metadata.get("output_uris", {}),
                }

        run_id = _start_run(source_uri, checksum)
        try:
            wave_min, wave_max = _wave_bounds()
            database_path = work_dir / "quality_gate.duckdb"
            connection = duckdb.connect(str(database_path))
            try:
                _configure_duckdb(connection, work_dir / "duckdb-tmp")
                _build_quality_tables(
                    connection, source_path, wave_min, wave_max
                )
                report = _quality_report(connection, wave_min, wave_max)
                local_outputs = _write_outputs(connection, work_dir)
            finally:
                connection.close()

            prefix = output_prefix.strip("/") or "version=1"
            keys = {
                "quality_calls": (
                    f"port_calls_quality/{prefix}/"
                    "port_calls_quality_checked_v1.parquet"
                ),
                "chronology_review": (
                    f"quarantine/{prefix}/port_calls_chronology_review_v1.parquet"
                ),
                "revision_review": (
                    f"reviews/{prefix}/port_call_revision_conflicts_v1.parquet"
                ),
                "report": (
                    f"audits/{prefix}/port_calls_quality_gate_report_v1.json"
                ),
            }
            report.update(
                {
                    "run_id": run_id,
                    "source_uri": source_uri,
                    "source_checksum": checksum,
                    "output_bucket": output_bucket,
                }
            )
            output_uris = {
                "quality_calls": _upload(
                    client,
                    local_outputs["quality_calls"],
                    output_bucket,
                    keys["quality_calls"],
                    "application/x-parquet",
                ),
                "chronology_review": _upload(
                    client,
                    local_outputs["chronology_review"],
                    output_bucket,
                    keys["chronology_review"],
                    "application/x-parquet",
                ),
                "revision_review": _upload(
                    client,
                    local_outputs["revision_review"],
                    output_bucket,
                    keys["revision_review"],
                    "application/x-parquet",
                ),
            }
            database_rows = _upsert_canonical_port_calls(
                local_outputs["quality_calls"]
            )
            output_uris["report"] = f"s3://{output_bucket}/{keys['report']}"
            report["database_port_calls_upserted"] = database_rows
            report["output_uris"] = output_uris
            report_path = work_dir / "port_calls_quality_gate_report_v1.json"
            report_path.write_text(
                json.dumps(report, indent=2, default=_json_default),
                encoding="utf-8",
            )
            _upload(
                client,
                report_path,
                output_bucket,
                keys["report"],
                "application/json",
            )
            _finish_run(
                run_id,
                "SUCCESS",
                row_count=int(report["port_calls"]),
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
                _finish_run(run_id, "FAILED", error_message=str(exc))
            raise
