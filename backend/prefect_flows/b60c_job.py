from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import psycopg2
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype
from psycopg2 import sql
from psycopg2.extras import Json

from prefect_flows.b60c_core import (
    CONTRACT_VERSION,
    DATASET_VERSION,
    SCENARIO_VERSION,
    build_operational_port_call_dataset,
    clean_json,
)


SOURCE_NAME = "b60c_operational_port_call_dataset"
DATASET_NAME = "maritime_port_call_landmark_v1"
TARGET_RELATION = "features.maritime_port_call_landmark_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"


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


def _query_frame(query: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _required_relations() -> None:
    required = (
        "core.port_call",
        "core.maritime_observation",
        "features.maritime_external_weather_hourly_v1",
        "reference.business_event",
        "audit.ingestion_run",
    )
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            for relation in required:
                cursor.execute("SELECT to_regclass(%s)", (relation,))
                if cursor.fetchone()[0] is None:
                    raise RuntimeError(f"Required relation does not exist: {relation}")


def load_port_calls() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT
            port_call_id::text AS port_call_id,
            port_code,
            terminal_code,
            mmsi::text AS mmsi,
            imo::text AS imo,
            vessel_name,
            voyage_id,
            planned_eta,
            planned_etb,
            planned_etd,
            actual_ata,
            actual_atb,
            actual_atd,
            cargo_type,
            vessel_type,
            source,
            created_at,
            updated_at
        FROM core.port_call
        ORDER BY actual_ata NULLS LAST, port_call_id
        """
    )
    if frame.empty:
        raise RuntimeError("core.port_call is empty")
    return frame


def load_research_weather() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT
            e.observed_at,
            w.wave_height_m,
            w.wave_period_s,
            w.wave_direction_deg,
            e.wind_speed_ms,
            e.wind_direction_deg,
            e.surface_current_ms,
            e.visibility_m,
            e.pressure_hpa,
            e.wind_gusts_10m,
            e.temperature_2m,
            e.relative_humidity_2m,
            e.precipitation,
            e.cloud_cover,
            e.sea_surface_temperature
        FROM features.maritime_external_weather_hourly_v1 e
        LEFT JOIN core.maritime_observation w
          ON w.observed_at=e.observed_at
         AND w.source='copernicus_ibi_wave'
        WHERE e.dataset_version='b58cb-external-weather-hourly-v1'
        ORDER BY e.observed_at
        """
    )


def load_business_events() -> pd.DataFrame:
    return _query_frame(
        """
        SELECT
            event_id,
            event_name,
            event_type,
            start_date,
            end_date,
            affected_flow,
            knowledge_policy,
            source,
            confidence
        FROM reference.business_event
        ORDER BY start_date, event_id
        """
    )


def _source_signature(
    port_calls: pd.DataFrame,
    weather: pd.DataFrame,
    business_events: pd.DataFrame,
) -> str:
    digest = hashlib.sha256(CONTRACT_VERSION.encode("ascii"))
    call_columns = [
        "port_call_id",
        "planned_eta",
        "planned_etd",
        "actual_ata",
        "actual_atb",
        "actual_atd",
        "cargo_type",
        "terminal_code",
        "source",
    ]
    hashed_calls = pd.util.hash_pandas_object(port_calls[call_columns], index=False)
    digest.update(hashed_calls.to_numpy(dtype="uint64").tobytes())
    if not weather.empty:
        hashed_weather = pd.util.hash_pandas_object(
            weather[["observed_at", "wave_height_m", "wind_speed_ms", "visibility_m"]],
            index=False,
        )
        digest.update(hashed_weather.to_numpy(dtype="uint64").tobytes())
    if not business_events.empty:
        hashed_events = pd.util.hash_pandas_object(
            business_events[["event_id", "start_date", "end_date"]], index=False
        )
        digest.update(hashed_events.to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT run_id, metadata
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
                """
                SELECT COUNT(*)
                FROM features.maritime_port_call_landmark_v1
                WHERE dataset_version=%s
                """,
                (DATASET_VERSION,),
            )
            if int(cursor.fetchone()[0]) == 0:
                return None
    return str(row[0]), dict(row[1] or {})


def _start_run(checksum: str) -> str:
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "dataset_version": DATASET_VERSION,
        "scenario_version": SCENARIO_VERSION,
        "orchestrator": "PREFECT",
        "source_modified": False,
        "main_dataset_synthetic_rows": 0,
        "targets_imputed": False,
        "training_executed": False,
        "historical_replay_allowed": False,
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
                    "postgresql://maritime/features.maritime_port_call_landmark_v1",
                    checksum,
                    Json(metadata),
                ),
            )
            return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    progress = clean_json(
        {"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details}
    )
    with _db_connection() as connection:
        with connection.cursor() as cursor:
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
                    Json(clean_json(metadata)),
                    error_message,
                    run_id,
                ),
            )


def _sql_type(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series.dtype):
        return "TIMESTAMPTZ"
    if is_bool_dtype(series.dtype):
        return "BOOLEAN"
    if is_numeric_dtype(series.dtype):
        return "DOUBLE PRECISION"
    return "TEXT"


def _materialize_dataset(frame: pd.DataFrame) -> int:
    schema_name, table_name = TARGET_RELATION.split(".", 1)
    columns = list(frame.columns)
    for column in columns:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", column):
            raise ValueError(f"Unsafe materialized column name: {column}")
    definitions = [
        sql.SQL("{} {}").format(
            sql.Identifier(column), sql.SQL(_sql_type(frame[column]))
        )
        for column in columns
    ]
    with _db_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                    sql.Identifier(schema_name)
                )
            )
            cursor.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(definitions),
                )
            )
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema=%s AND table_name=%s
                """,
                (schema_name, table_name),
            )
            existing = {row[0] for row in cursor.fetchall()}
            for definition, column in zip(definitions, columns):
                if column not in existing:
                    cursor.execute(
                        sql.SQL("ALTER TABLE {}.{} ADD COLUMN {}").format(
                            sql.Identifier(schema_name),
                            sql.Identifier(table_name),
                            definition,
                        )
                    )
            cursor.execute(
                sql.SQL(
                    "CREATE TEMP TABLE b60c_stage (LIKE {}.{} INCLUDING DEFAULTS) "
                    "ON COMMIT DROP"
                ).format(sql.Identifier(schema_name), sql.Identifier(table_name))
            )
            column_sql = sql.SQL(", ").join(map(sql.Identifier, columns)).as_string(cursor)
            copy_sql = (
                f"COPY b60c_stage ({column_sql}) FROM STDIN "
                "WITH (FORMAT CSV, NULL '\\N')"
            )
            for start in range(0, len(frame), 2_500):
                chunk = frame.iloc[start : start + 2_500].copy()
                stream = io.StringIO()
                chunk.to_csv(stream, index=False, header=False, na_rep="\\N")
                stream.seek(0)
                cursor.copy_expert(copy_sql, stream)
            cursor.execute(
                sql.SQL("DELETE FROM {}.{} WHERE dataset_version=%s").format(
                    sql.Identifier(schema_name), sql.Identifier(table_name)
                ),
                (DATASET_VERSION,),
            )
            cursor.execute(
                sql.SQL("INSERT INTO {}.{} ({}) SELECT {} FROM b60c_stage").format(
                    sql.Identifier(schema_name),
                    sql.Identifier(table_name),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                    sql.SQL(", ").join(map(sql.Identifier, columns)),
                )
            )
            cursor.execute(
                sql.SQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS "
                    "b60c_port_call_landmark_version_call_time_uidx "
                    "ON {}.{} (dataset_version, port_call_id, landmark_at)"
                ).format(sql.Identifier(schema_name), sql.Identifier(table_name))
            )
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS b60c_port_call_landmark_split_time_idx "
                    "ON {}.{} (split, landmark_at)"
                ).format(sql.Identifier(schema_name), sql.Identifier(table_name))
            )
            cursor.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS b60c_port_call_landmark_vessel_time_idx "
                    "ON {}.{} (vessel_key, landmark_at)"
                ).format(sql.Identifier(schema_name), sql.Identifier(table_name))
            )
            cursor.execute(
                sql.SQL("ANALYZE {}.{}").format(
                    sql.Identifier(schema_name), sql.Identifier(table_name)
                )
            )
    return len(frame)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _upload(client, path: Path, key: str) -> str:
    content_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
        ".parquet": "application/octet-stream",
    }.get(path.suffix.lower(), "application/octet-stream")
    client.upload_file(
        str(path), OUTPUT_BUCKET, key, ExtraArgs={"ContentType": content_type}
    )
    client.head_object(Bucket=OUTPUT_BUCKET, Key=key)
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _readme(decision: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# B60C operational port-call landmark dataset",
            "",
            f"Decision: {decision['decision']}",
            "",
            "The entity is a real vessel port call from core.port_call. Each row is",
            "an hourly decision landmark strictly before actual departure. Historical",
            "vessel, terminal and port-state statistics use only calls completed before",
            "the current call started.",
            "",
            "Primary objective: warn before planned departure that the call will exceed",
            "planned ETD by more than three hours. Secondary labels support ordinal",
            "classification, remaining-time regression and survival analysis.",
            "",
            "The main Parquet and Timescale relation contain real retrospective rows",
            "only. Counterfactual stress scenarios are stored in a separate Parquet,",
            "contain no targets, and are forbidden for training, validation and testing.",
            "",
            "Final historical ETA/ETD snapshots and retrospective weather are research",
            "limitations. Production promotion stays blocked until prospective issue-time",
            "snapshots and plan revisions have accumulated enough history.",
        ]
    )


def run_b60c_dataset_build(force: bool = False) -> dict[str, Any]:
    _required_relations()
    port_calls = load_port_calls()
    weather = load_research_weather()
    business_events = load_business_events()
    checksum = _source_signature(port_calls, weather, business_events)
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
        _update_progress(
            run_id,
            "BUILDING_STRICT_PORT_CALL_LANDMARKS",
            source_calls=len(port_calls),
            research_weather_rows=len(weather),
        )
        result = build_operational_port_call_dataset(
            port_calls, weather, business_events
        )
        dataset = result.dataset.copy()
        dataset["materialization_run_id"] = run_id
        scenarios = result.scenarios.copy()
        if not scenarios.empty:
            scenarios["materialization_run_id"] = run_id

        _update_progress(
            run_id,
            "MATERIALIZING_REAL_DATASET",
            rows=len(dataset),
            calls=dataset["port_call_id"].nunique(),
            columns=len(dataset.columns),
        )
        materialized_rows = _materialize_dataset(dataset)

        _update_progress(
            run_id,
            "WRITING_VERSIONED_ARTIFACTS",
            scenario_rows=len(scenarios),
        )
        client = _s3_client()
        outputs: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix="b60c-") as temporary:
            directory = Path(temporary)
            dataset_path = directory / "maritime_port_call_landmark_v1.parquet"
            scenario_path = directory / "counterfactual_stress_scenarios_v1.parquet"
            dataset.to_parquet(dataset_path, index=False, compression="zstd")
            scenarios.to_parquet(scenario_path, index=False, compression="zstd")
            outputs[dataset_path.name] = _upload(
                client,
                dataset_path,
                f"datasets/b60c/{OUTPUT_PREFIX}/{dataset_path.name}",
            )
            outputs[scenario_path.name] = _upload(
                client,
                scenario_path,
                f"scenarios/b60c/{OUTPUT_PREFIX}/{scenario_path.name}",
            )

            for name, report in result.reports.items():
                path = directory / name
                report.to_csv(path, index=False)
                outputs[name] = _upload(
                    client, path, f"reports/b60c/{OUTPUT_PREFIX}/{name}"
                )

            decision_path = directory / "10_b60c_final_decision.json"
            decision_path.write_text(
                json.dumps(clean_json(result.decision), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[decision_path.name] = _upload(
                client,
                decision_path,
                f"configs/b60c/{OUTPUT_PREFIX}/{decision_path.name}",
            )
            feature_path = directory / "11_feature_sets.json"
            feature_path.write_text(
                json.dumps(clean_json(result.feature_sets), indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            outputs[feature_path.name] = _upload(
                client,
                feature_path,
                f"configs/b60c/{OUTPUT_PREFIX}/{feature_path.name}",
            )
            readme_path = directory / "README_B60C.md"
            readme_path.write_text(_readme(result.decision), encoding="utf-8")
            outputs[readme_path.name] = _upload(
                client,
                readme_path,
                f"reports/b60c/{OUTPUT_PREFIX}/{readme_path.name}",
            )

            manifest_rows = []
            for path in sorted(directory.iterdir()):
                manifest_rows.append(
                    {
                        "file": path.name,
                        "bytes": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                )
            manifest_path = directory / "12_artifact_manifest.csv"
            pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
            outputs[manifest_path.name] = _upload(
                client,
                manifest_path,
                f"reports/b60c/{OUTPUT_PREFIX}/{manifest_path.name}",
            )

        metadata = {
            **result.decision,
            "checksum": checksum,
            "orchestrator": "PREFECT",
            "prefect_flow": "b60c-operational-port-call-dataset",
            "materialized_relation": TARGET_RELATION,
            "materialized_rows": materialized_rows,
            "dataset_uri": outputs["maritime_port_call_landmark_v1.parquet"],
            "scenario_uri": outputs["counterfactual_stress_scenarios_v1.parquet"],
            "feature_sets_uri": outputs["11_feature_sets.json"],
            "report_prefix": f"s3://{OUTPUT_BUCKET}/reports/b60c/{OUTPUT_PREFIX}/",
            "outputs": outputs,
        }
        _finish_run(run_id, "SUCCESS", len(dataset), metadata)
        return {
            "status": "SUCCESS",
            "run_id": run_id,
            "reused": False,
            "results": clean_json(metadata),
        }
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "contract_version": CONTRACT_VERSION,
                "dataset_version": DATASET_VERSION,
                "scenario_version": SCENARIO_VERSION,
                "orchestrator": "PREFECT",
            },
            str(exc),
        )
        raise


def verify_b60c_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B60C status: {result.get('status')}")
    metadata = result.get("results") or {}
    expected = "READY_FOR_RETROSPECTIVE_PORT_CALL_RISK_MODELING"
    if metadata.get("decision") != expected:
        raise RuntimeError(f"B60C quality gates failed: {metadata.get('decision')}")
    if metadata.get("main_dataset_synthetic_rows") not in (0, "0"):
        raise RuntimeError("B60C main dataset contains synthetic rows")
    if metadata.get("counterfactual_training_allowed") not in (False, "false"):
        raise RuntimeError("B60C counterfactual rows were marked trainable")
    if metadata.get("targets_imputed") not in (False, "false"):
        raise RuntimeError("B60C targets were imputed")
    for field in ("historical_replay_allowed", "production_promotion_allowed"):
        if metadata.get(field) not in (False, "false"):
            raise RuntimeError(f"B60C retrospective safety violation: {field}")
    return {
        "run_id": result["run_id"],
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "row_count": metadata.get("landmark_rows"),
        "eligible_calls": metadata.get("eligible_calls"),
        "train_calls": metadata.get("train_calls"),
        "valid_calls": metadata.get("valid_calls"),
        "test_calls": metadata.get("test_calls"),
        "counterfactual_rows": metadata.get("counterfactual_rows"),
        "quality_gates_passed": metadata.get("quality_gates_passed"),
        "next_block": metadata.get("next_block"),
        "safety_contract": "PASS",
    }
