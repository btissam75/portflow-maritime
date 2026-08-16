from __future__ import annotations

import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from prefect_flows.b62_core import (
    HORIZONS_H,
    MODEL_VERSION,
    WAVE_TARGETS,
    WEATHER_TARGETS,
    apply_frozen_selection,
    assign_temporal_roles,
    build_vessel_impact_shadow,
    data_signature,
    forecast_metrics,
    historical_severity_thresholds,
    issue_time_readiness,
    json_ready,
    metocean_severity,
    prepare_hourly_frame,
    quality_gates,
    rolling_origins,
    seasonal_predictions,
    select_valid_models,
    source_coverage_report,
    temporal_split_report,
    weather_wave_coupling_report,
)
from prefect_flows.b62_models import (
    autogluon_runtime,
    cascade_backtest,
    cascade_forecast_origin,
    fit_cascade,
)


SOURCE_NAME = "b62_weather_wave_vessel_autogluon"
DATASET_NAME = "maritime_weather_wave_vessel_cascade_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"
FORECAST_TABLE = "serving.maritime_metocean_forecast_shadow_v1"
IMPACT_TABLE = "serving.maritime_metocean_vessel_impact_shadow_v1"
WATCHLIST_TABLE = "serving.maritime_capacity_watchlist_shadow_v1"
MIN_VALID_ORIGINS = 18


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
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()
        columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _relation_exists(relation: str) -> bool:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (relation,))
        return cursor.fetchone()[0] is not None


def _require_sources() -> None:
    required = (
        "core.maritime_observation",
        "features.maritime_external_weather_hourly_v1",
        "audit.ingestion_run",
    )
    missing = [relation for relation in required if not _relation_exists(relation)]
    if missing:
        raise RuntimeError(f"B62 required relations are missing: {missing}")


def load_wave_source() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT observed_at,
               AVG(wave_height_m)::double precision AS wave_height_m,
               AVG(wave_period_s)::double precision AS wave_period_s,
               MOD((DEGREES(ATAN2(
                   AVG(SIN(RADIANS(wave_direction_deg))),
                   AVG(COS(RADIANS(wave_direction_deg)))
               )) + 360.0)::numeric, 360.0)::double precision AS wave_direction_deg
        FROM core.maritime_observation
        WHERE quality_flag=0
          AND wave_height_m IS NOT NULL
          AND wave_period_s IS NOT NULL
          AND wave_direction_deg IS NOT NULL
        GROUP BY observed_at
        ORDER BY observed_at
        """
    )
    if frame.empty:
        raise RuntimeError("No quality-controlled wave observations were found")
    return frame


def load_atmosphere_source() -> pd.DataFrame:
    frame = _query_frame(
        """
        SELECT observed_at,wind_speed_ms,wind_direction_deg,pressure_hpa,
               visibility_m,temperature_2m,relative_humidity_2m,
               precipitation,cloud_cover,wind_gusts_10m,
               surface_current_ms,sea_surface_temperature,
               availability_semantics,dataset_version
        FROM features.maritime_external_weather_hourly_v1
        ORDER BY observed_at
        """
    )
    if frame.empty:
        raise RuntimeError("B58C-B external atmosphere dataset is empty")
    return frame


def load_issue_time_forecasts() -> pd.DataFrame:
    relation = "features.maritime_issue_time_weather_forecast_v1"
    if not _relation_exists(relation):
        return pd.DataFrame()
    return _query_frame(
        f"""
        SELECT provider,provider_model,issue_at,available_at,valid_at,lead_time_h,
               wind_speed_ms,wind_direction_deg,pressure_hpa,visibility_m,
               air_temperature_c,wave_height_m,wave_direction_deg,wave_period_s,
               ocean_current_ms,sea_surface_temperature_c,
               atmosphere_available_flag,wave_available_flag,
               full_weather_available_flag
        FROM {relation}
        WHERE available_at>=issue_at AND valid_at>=available_at
        ORDER BY issue_at,valid_at
        """
    )


def load_watchlist() -> pd.DataFrame:
    if not _relation_exists(WATCHLIST_TABLE):
        return pd.DataFrame()
    return _query_frame(
        f"""
        SELECT port_call_id,vessel_name,port_code,terminal_code,vessel_type,cargo_group,
               landmark_at,decision_at,evaluation_role,risk_score,rank_in_window,
               active_calls,capacity,watchlist_selected,action_tier,
               p_delay_gt3,p_gt3_breach_within_24h,remaining_p50_h
        FROM {WATCHLIST_TABLE}
        WHERE policy_version='b61e-capacity-aware-temporal-ranking-v1'
          AND evaluation_role='TEST_DIAGNOSTIC_ONLY'
        ORDER BY decision_at,rank_in_window
        """
    )


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT run_id,metadata FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s AND checksum=%s AND status='SUCCESS'
            ORDER BY finished_at DESC LIMIT 1
            """,
            (SOURCE_NAME, DATASET_NAME, checksum),
        )
        row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _start_run(checksum: str, parameters: dict[str, Any]) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "orchestrator": "PREFECT",
        "parameters": parameters,
        "test_used_for_selection": False,
        "retrospective_weather_production_allowed": False,
        "production_promotion_allowed": False,
        "automatic_action_allowed": False,
    }
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO audit.ingestion_run
                (source_name,dataset_name,object_uri,checksum,metadata)
            VALUES (%s,%s,%s,%s,%s) RETURNING run_id
            """,
            (
                SOURCE_NAME,
                DATASET_NAME,
                "postgresql://maritime/core.maritime_observation+features.maritime_external_weather_hourly_v1",
                checksum,
                Json(metadata),
            ),
        )
        return str(cursor.fetchone()[0])


def _progress(run_id: str, stage: str, **details: Any) -> None:
    payload = {
        "progress": {
            "stage": stage,
            "updated_at": pd.Timestamp.now(tz="UTC"),
            **details,
        }
    }
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
            (Json(json_ready(payload)), run_id),
        )


def _finish(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error: str | None = None,
) -> None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE audit.ingestion_run
            SET status=%s,row_count=%s,finished_at=NOW(),metadata=metadata || %s,
                error_message=%s
            WHERE run_id=%s
            """,
            (status, row_count, Json(json_ready(metadata)), error, run_id),
        )


def _put_bytes(client, key: str, payload: bytes, content_type: str) -> str:
    client.put_object(
        Bucket=OUTPUT_BUCKET, Key=key, Body=payload, ContentType=content_type
    )
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _put_json(client, key: str, payload: Any) -> str:
    return _put_bytes(
        client,
        key,
        json.dumps(json_ready(payload), indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )


def _put_csv(client, key: str, frame: pd.DataFrame) -> str:
    return _put_bytes(client, key, frame.to_csv(index=False).encode("utf-8"), "text/csv")


def _put_parquet(client, key: str, frame: pd.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return _put_bytes(
        client, key, buffer.getvalue(), "application/vnd.apache.parquet"
    )


def _latest_provider_forecast(
    forecasts: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    if forecasts.empty:
        return pd.DataFrame()
    source = forecasts.copy()
    for column in ("issue_at", "available_at", "valid_at"):
        source[column] = pd.to_datetime(source[column], errors="coerce", utc=True)
    issue_at = source["issue_at"].max()
    source = source.loc[source["issue_at"].eq(issue_at)].copy()
    source["completeness"] = source[
        ["wind_speed_ms", "pressure_hpa", "visibility_m", "wave_height_m", "wave_period_s"]
    ].notna().sum(axis=1)
    selected = []
    for horizon_h in HORIZONS_H:
        candidates = source.assign(
            distance=(pd.to_numeric(source["lead_time_h"], errors="coerce") - horizon_h).abs()
        ).sort_values(["distance", "completeness"], ascending=[True, False])
        if candidates.empty or float(candidates.iloc[0]["distance"]) > 3.1:
            continue
        row = candidates.iloc[0].copy()
        row["horizon_h"] = horizon_h
        selected.append(row)
    if not selected:
        return pd.DataFrame()
    output = pd.DataFrame(selected).reset_index(drop=True)
    output["temperature_2m"] = pd.to_numeric(output["air_temperature_c"], errors="coerce")
    output = metocean_severity(output, thresholds)
    output["track"] = "ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT"
    output["uncertainty_status"] = "DETERMINISTIC_PROVIDER_NOT_LOCALLY_CALIBRATED"
    output["production_claim_allowed"] = False
    return output


def _provider_long(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    mappings = {
        "wind_speed_ms": "wind_speed_ms",
        "pressure_hpa": "pressure_hpa",
        "visibility_m": "visibility_m",
        "temperature_2m": "temperature_2m",
        "wave_height_m": "wave_height_m",
        "wave_period_s": "wave_period_s",
        "wave_direction_deg": "wave_direction_deg",
    }
    rows = []
    for row in frame.itertuples(index=False):
        for variable, attribute in mappings.items():
            value = getattr(row, attribute, None)
            if value is None or not np.isfinite(float(value)):
                continue
            rows.append(
                {
                    "track": row.track,
                    "issue_at": row.issue_at,
                    "valid_at": row.valid_at,
                    "horizon_h": int(row.horizon_h),
                    "variable": variable,
                    "p10": np.nan,
                    "p50": float(value),
                    "p90": np.nan,
                    "source_model": f"{row.provider}:{row.provider_model}",
                    "uncertainty_status": row.uncertainty_status,
                    "operationally_available": True,
                    "production_claim_allowed": False,
                }
            )
    return pd.DataFrame(rows)


def _research_long(weather: pd.DataFrame, wave: pd.DataFrame) -> pd.DataFrame:
    source = pd.concat([weather, wave], ignore_index=True)
    if source.empty:
        return source
    source["track"] = "RESEARCH_REANALYSIS_SHADOW"
    source["source_model"] = "AUTOGLUON_CHRONOS2_SMALL_CASCADE"
    source["uncertainty_status"] = "MODEL_NATIVE_P10_P50_P90_RESEARCH_ONLY"
    source["operationally_available"] = False
    source["production_claim_allowed"] = False
    return source[
        [
            "track", "issue_at", "valid_at", "horizon_h", "variable",
            "p10", "p50", "p90", "source_model", "uncertainty_status",
            "operationally_available", "production_claim_allowed",
        ]
    ]


def _cascade_wide(
    weather: pd.DataFrame,
    wave: pd.DataFrame,
    thresholds: dict[str, float],
    track: str,
) -> pd.DataFrame:
    source = pd.concat([weather, wave], ignore_index=True)
    if source.empty:
        return pd.DataFrame()
    source = source.loc[source["horizon_h"].isin(HORIZONS_H)].copy()
    wide = source.pivot_table(
        index=["issue_at", "valid_at", "horizon_h"],
        columns="variable",
        values="p50",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    wide = metocean_severity(wide, thresholds)
    wide["track"] = track
    wide["uncertainty_status"] = "MODEL_NATIVE_P10_P50_P90_RESEARCH_ONLY"
    wide["production_claim_allowed"] = False
    return wide


def _materialize_forecasts(frame: pd.DataFrame, run_id: str) -> int:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS serving;
    CREATE TABLE IF NOT EXISTS {FORECAST_TABLE} (
        model_version TEXT NOT NULL,track TEXT NOT NULL,issue_at TIMESTAMPTZ NOT NULL,
        valid_at TIMESTAMPTZ NOT NULL,horizon_h INTEGER NOT NULL,variable TEXT NOT NULL,
        p10 DOUBLE PRECISION,p50 DOUBLE PRECISION NOT NULL,p90 DOUBLE PRECISION,
        source_model TEXT NOT NULL,uncertainty_status TEXT NOT NULL,
        operationally_available BOOLEAN NOT NULL,production_claim_allowed BOOLEAN NOT NULL,
        materialization_run_id TEXT NOT NULL,
        PRIMARY KEY(model_version,track,issue_at,valid_at,variable)
    );
    """
    columns = [
        "model_version", "track", "issue_at", "valid_at", "horizon_h", "variable",
        "p10", "p50", "p90", "source_model", "uncertainty_status",
        "operationally_available", "production_claim_allowed", "materialization_run_id",
    ]
    source = frame.copy()
    source["model_version"] = MODEL_VERSION
    source["materialization_run_id"] = run_id
    source = source[columns]
    source = source.replace({np.nan: None})
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(ddl)
        cursor.execute(f"DELETE FROM {FORECAST_TABLE} WHERE model_version=%s", (MODEL_VERSION,))
        if not source.empty:
            execute_values(
                cursor,
                f"INSERT INTO {FORECAST_TABLE} ({','.join(columns)}) VALUES %s",
                [tuple(row) for row in source.itertuples(index=False, name=None)],
                page_size=1_000,
            )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS ix_b62_forecast_latest ON {FORECAST_TABLE} "
            "(track,issue_at DESC,valid_at,variable)"
        )
    return len(source)


def _materialize_impacts(frame: pd.DataFrame, run_id: str) -> int:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS serving;
    CREATE TABLE IF NOT EXISTS {IMPACT_TABLE} (
        model_version TEXT NOT NULL,port_call_id TEXT NOT NULL,vessel_name TEXT,
        port_code TEXT,terminal_code TEXT,vessel_type TEXT,cargo_group TEXT,
        source_decision_at TIMESTAMPTZ NOT NULL,forecast_issue_at TIMESTAMPTZ NOT NULL,
        valid_at TIMESTAMPTZ NOT NULL,horizon_h INTEGER NOT NULL,
        base_temporal_risk DOUBLE PRECISION NOT NULL,metocean_severity DOUBLE PRECISION NOT NULL,
        vessel_exposure DOUBLE PRECISION NOT NULL,combined_priority_score DOUBLE PRECISION NOT NULL,
        metocean_tier TEXT NOT NULL,priority_tier TEXT NOT NULL,forecast_track TEXT NOT NULL,
        score_semantics TEXT NOT NULL,
        automatic_action_allowed BOOLEAN NOT NULL,production_claim_allowed BOOLEAN NOT NULL,
        materialization_run_id TEXT NOT NULL,
        PRIMARY KEY(model_version,port_call_id,forecast_issue_at,valid_at)
    );
    ALTER TABLE {IMPACT_TABLE}
        ADD COLUMN IF NOT EXISTS forecast_track TEXT NOT NULL
        DEFAULT 'UNSPECIFIED_SHADOW';
    """
    columns = [
        "model_version", "port_call_id", "vessel_name", "port_code", "terminal_code",
        "vessel_type", "cargo_group", "source_decision_at", "forecast_issue_at",
        "valid_at", "horizon_h", "base_temporal_risk", "metocean_severity",
        "vessel_exposure", "combined_priority_score", "metocean_tier", "priority_tier",
        "forecast_track", "score_semantics", "automatic_action_allowed", "production_claim_allowed",
        "materialization_run_id",
    ]
    source = pd.DataFrame(columns=columns)
    if not frame.empty:
        source = pd.DataFrame(
            {
                "model_version": MODEL_VERSION,
                "port_call_id": frame["port_call_id"],
                "vessel_name": frame["vessel_name"],
                "port_code": frame["port_code"],
                "terminal_code": frame["terminal_code"],
                "vessel_type": frame["vessel_type"],
                "cargo_group": frame["cargo_group"],
                "source_decision_at": frame["decision_at"],
                "forecast_issue_at": frame["issue_at"],
                "valid_at": frame["valid_at"],
                "horizon_h": frame["horizon_h"].astype(int),
                "base_temporal_risk": frame["base_temporal_risk"],
                "metocean_severity": frame["metocean_severity"],
                "vessel_exposure": frame["vessel_exposure"],
                "combined_priority_score": frame["combined_priority_score"],
                "metocean_tier": frame["metocean_tier"],
                "priority_tier": frame["priority_tier"],
                "forecast_track": frame["forecast_track"],
                "score_semantics": frame["score_semantics"],
                "automatic_action_allowed": False,
                "production_claim_allowed": False,
                "materialization_run_id": run_id,
            }
        )
    source = source.replace({np.nan: None})
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(ddl)
        cursor.execute(f"DELETE FROM {IMPACT_TABLE} WHERE model_version=%s", (MODEL_VERSION,))
        if not source.empty:
            execute_values(
                cursor,
                f"INSERT INTO {IMPACT_TABLE} ({','.join(columns)}) VALUES %s",
                [tuple(row) for row in source.itertuples(index=False, name=None)],
                page_size=1_000,
            )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS ix_b62_impact_priority ON {IMPACT_TABLE} "
            "(forecast_issue_at DESC,horizon_h,combined_priority_score DESC)"
        )
    return len(source)


def _log_mlflow(metadata: dict[str, Any], metrics: pd.DataFrame) -> str:
    try:
        import mlflow

        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("maritime-b62-metocean-cascade")
        with mlflow.start_run(run_name="b62-v1"):
            mlflow.log_params(
                {
                    "model_version": MODEL_VERSION,
                    "forecast_model": "autogluon-chronos2-small",
                    "selection_role": "VALID_ONLY",
                    "test_role": "TEST_DIAGNOSTIC_ONLY",
                }
            )
            mlflow.log_metric("selected_chronos_tasks", metadata["selected_chronos_tasks"])
            for row in metrics.itertuples(index=False):
                if row.evaluation_role != "VALID" or row.model != "CHRONOS2_SMALL":
                    continue
                name = f"valid_mae_{row.family}_{row.variable}_h{row.horizon_h}"
                mlflow.log_metric(name.replace("-", "_"), float(row.MAE))
        return "LOGGED"
    except Exception as exc:
        return f"SKIPPED:{type(exc).__name__}:{exc}"


def run_b62(
    force: bool = False,
    validation_days: int = 365,
    test_days: int = 365,
    backtest_step_h: int = 168,
    preset: str = "chronos2_small",
) -> dict[str, Any]:
    _require_sources()
    waves = load_wave_source()
    atmosphere = load_atmosphere_source()
    issue_forecasts = load_issue_time_forecasts()
    watchlist = load_watchlist()
    parameters = {
        "validation_days": validation_days,
        "test_days": test_days,
        "backtest_step_h": backtest_step_h,
        "preset": preset,
    }
    checksum = data_signature(
        waves, atmosphere, issue_forecasts, watchlist, parameters=parameters
    )
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {"status": "SUCCESS", "run_id": previous[0], "reused": True, "results": previous[1]}

    run_id = _start_run(checksum, parameters)
    client = _s3_client()
    try:
        _progress(run_id, "PREPARING_CANONICAL_HOURLY_GRID", wave_rows=len(waves), atmosphere_rows=len(atmosphere))
        hourly, invalid_bounds = prepare_hourly_frame(waves, atmosphere)
        hourly, boundaries = assign_temporal_roles(hourly, validation_days, test_days)
        coverage = source_coverage_report(hourly)
        split_report = temporal_split_report(hourly)
        coupling = weather_wave_coupling_report(hourly)
        readiness = issue_time_readiness(issue_forecasts)
        runtime = autogluon_runtime()
        if not runtime["ready"]:
            raise RuntimeError(f"AutoGluon runtime is not ready: {runtime['error']}")

        valid_origins = rolling_origins(hourly, "VALID", backtest_step_h)
        test_origins = rolling_origins(
            hourly, "TEST_DIAGNOSTIC_ONLY", max(backtest_step_h * 2, 336)
        )
        if len(valid_origins) < MIN_VALID_ORIGINS:
            raise RuntimeError(f"Only {len(valid_origins)} VALID origins; at least {MIN_VALID_ORIGINS} required")

        thresholds = historical_severity_thresholds(hourly)
        research_impact_forecast = pd.DataFrame()

        with tempfile.TemporaryDirectory(prefix="b62-models-") as temporary:
            model_root = Path(temporary)
            _progress(run_id, "LOADING_CHRONOS2_SMALL", valid_origins=len(valid_origins), test_origins=len(test_origins))
            weather_model, wave_model, covariate_fill, runtime = fit_cascade(
                hourly, boundaries.train_end, model_root, preset=preset
            )

            def valid_progress(index: int, total: int, origin: pd.Timestamp) -> None:
                if index == 1 or index == total or index % 5 == 0:
                    _progress(run_id, "VALID_CASCADE_BACKTEST", completed=index, total=total, origin=origin)

            valid_chronos = cascade_backtest(
                weather_model, wave_model, hourly, valid_origins, "VALID", covariate_fill, valid_progress
            )
            valid_baseline = pd.concat(
                [
                    seasonal_predictions(hourly, valid_origins, WEATHER_TARGETS, "WEATHER"),
                    seasonal_predictions(hourly, valid_origins, WAVE_TARGETS, "WAVE"),
                ],
                ignore_index=True,
            )
            valid_predictions = pd.concat([valid_baseline, valid_chronos], ignore_index=True)
            valid_metrics = forecast_metrics(valid_predictions)
            selection = select_valid_models(valid_metrics)
            if selection.empty:
                raise RuntimeError("VALID selection is empty")

            _progress(run_id, "FROZEN_TEST_DIAGNOSTIC", selected_tasks=len(selection), test_origins=len(test_origins))

            def test_progress(index: int, total: int, origin: pd.Timestamp) -> None:
                _progress(
                    run_id,
                    "FROZEN_TEST_CASCADE_BACKTEST",
                    completed=index,
                    total=total,
                    origin=origin,
                    selected_tasks=len(selection),
                )

            test_chronos = cascade_backtest(
                weather_model, wave_model, hourly, test_origins,
                "TEST_DIAGNOSTIC_ONLY", covariate_fill, test_progress,
            )
            test_baseline = pd.concat(
                [
                    seasonal_predictions(hourly, test_origins, WEATHER_TARGETS, "WEATHER"),
                    seasonal_predictions(hourly, test_origins, WAVE_TARGETS, "WAVE"),
                ],
                ignore_index=True,
            )
            test_all = pd.concat([test_baseline, test_chronos], ignore_index=True)
            test_selected = apply_frozen_selection(test_all, selection, "TEST_DIAGNOSTIC_ONLY")
            test_metrics = forecast_metrics(test_selected)

            research_origin = hourly["observed_at"].max()
            research_weather, research_wave = cascade_forecast_origin(
                weather_model, wave_model, hourly, research_origin, covariate_fill
            )

            if not watchlist.empty:
                impact_origin = pd.Timestamp(watchlist["decision_at"].max()).floor("h")
                if impact_origin.tzinfo is None:
                    impact_origin = impact_origin.tz_localize("UTC")
                else:
                    impact_origin = impact_origin.tz_convert("UTC")
                latest_replay_origin = hourly["observed_at"].max() - pd.Timedelta(
                    hours=max(HORIZONS_H)
                )
                if hourly["observed_at"].min() <= impact_origin <= latest_replay_origin:
                    impact_weather, impact_wave = cascade_forecast_origin(
                        weather_model, wave_model, hourly, impact_origin, covariate_fill
                    )
                    research_impact_forecast = _cascade_wide(
                        impact_weather,
                        impact_wave,
                        thresholds,
                        "RESEARCH_REANALYSIS_SHADOW",
                    )

        provider = _latest_provider_forecast(issue_forecasts, thresholds)
        provider_impacts = build_vessel_impact_shadow(watchlist, provider)
        research_impacts = build_vessel_impact_shadow(
            watchlist, research_impact_forecast
        )
        impacts = (
            provider_impacts if not provider_impacts.empty else research_impacts
        )
        impact_track = (
            str(impacts["forecast_track"].iloc[0])
            if not impacts.empty
            else "NO_TIME_ALIGNED_FORECAST"
        )
        serving_forecasts = pd.concat(
            [_research_long(research_weather, research_wave), _provider_long(provider)],
            ignore_index=True,
        )
        gates = quality_gates(hourly, invalid_bounds, selection, readiness)
        critical_gates_passed = bool(gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all())
        selected_chronos = int(selection["chronos_accepted"].sum())
        decision = (
            "READY_FOR_FRESH_FORWARD_METOCEAN_SHADOW_VALIDATION"
            if readiness["issue_time_ready"] and critical_gates_passed
            else "RESEARCH_CASCADE_READY_COLLECT_ISSUE_TIME_HISTORY"
        )
        next_block = (
            "B62B_FORWARD_WEATHER_WAVE_VESSEL_MONITOR"
            if readiness["issue_time_ready"]
            else "B58C_D_CONTINUE_HOURLY_COLLECTION_AND_B62_MONTHLY_REPLAY"
        )
        forecast_rows = _materialize_forecasts(serving_forecasts, run_id)
        impact_rows = _materialize_impacts(impacts, run_id)

        all_metrics = pd.concat([valid_metrics, test_metrics], ignore_index=True)
        metadata = {
            "model_version": MODEL_VERSION,
            "decision": decision,
            "rows": len(hourly),
            "valid_origins": len(valid_origins),
            "test_origins": len(test_origins),
            "selected_chronos_tasks": selected_chronos,
            "selected_tasks": len(selection),
            "issue_time_ready": readiness["issue_time_ready"],
            "issue_time_span_days": readiness["span_days"],
            "critical_gates_passed": critical_gates_passed,
            "test_used_for_selection": False,
            "test_role": "TEST_DIAGNOSTIC_ONLY_ONCE",
            "research_reanalysis_production_allowed": False,
            "production_promotion_allowed": False,
            "automatic_action_allowed": False,
            "serving_forecast_rows": forecast_rows,
            "serving_impact_rows": impact_rows,
            "vessel_impact_forecast_track": impact_track,
            "runtime": runtime,
            "next_block": next_block,
        }
        metadata["mlflow_status"] = _log_mlflow(metadata, valid_metrics)

        _progress(run_id, "WRITING_VERSIONED_ARTIFACTS", forecast_rows=forecast_rows, impact_rows=impact_rows)
        report_root = f"reports/b62/{OUTPUT_PREFIX}"
        prediction_root = f"predictions/b62/{OUTPUT_PREFIX}"
        model_root_key = f"models/b62/{OUTPUT_PREFIX}"
        artifacts = {
            "source_coverage": _put_csv(client, f"{report_root}/01_source_coverage.csv", coverage),
            "temporal_split": _put_csv(client, f"{report_root}/02_temporal_split.csv", split_report),
            "physical_integrity": _put_csv(client, f"{report_root}/03_physical_integrity.csv", invalid_bounds),
            "valid_metrics": _put_csv(client, f"{report_root}/04_valid_metrics.csv", valid_metrics),
            "valid_selection": _put_csv(client, f"{report_root}/05_valid_model_selection.csv", selection),
            "test_diagnostic": _put_csv(client, f"{report_root}/06_test_diagnostic.csv", test_metrics),
            "weather_wave_coupling": _put_csv(client, f"{report_root}/07_weather_wave_coupling.csv", coupling),
            "quality_gates": _put_csv(client, f"{report_root}/08_quality_gates.csv", gates),
            "vessel_impact": _put_csv(client, f"{report_root}/09_vessel_impact_shadow.csv", impacts),
            "valid_predictions": _put_parquet(client, f"{prediction_root}/valid_predictions.parquet", valid_predictions),
            "test_predictions": _put_parquet(client, f"{prediction_root}/test_selected_predictions.parquet", test_selected),
            "serving_forecasts": _put_parquet(client, f"{prediction_root}/serving_forecasts.parquet", serving_forecasts),
        }
        model_card = {
            **metadata,
            "architecture": "AUTOGLUON_CHRONOS2_WEATHER_TO_WAVE_CASCADE_PLUS_B61E_VESSEL_PRIORITY",
            "weather_targets": list(WEATHER_TARGETS),
            "wave_targets": list(WAVE_TARGETS),
            "horizons_h": list(HORIZONS_H),
            "selection_rule": "VALID MAE gain >=2%, P10-P90 coverage within 15 points of 80%",
            "provider_uncertainty_disclosure": "Provider forecasts have no locally calibrated P10/P90 and remain deterministic inputs",
            "vessel_score_disclosure": "Combined score is a shadow ranking priority, not a calibrated probability",
            "historical_availability_disclosure": "Retrospective reanalysis is never marked operationally available",
            "issue_time_readiness": readiness,
            "thresholds": thresholds,
            "artifacts": artifacts,
        }
        artifacts["model_card"] = _put_json(client, f"{model_root_key}/model_card.json", model_card)
        artifacts["selection_config"] = _put_json(
            client,
            f"{model_root_key}/selection_config.json",
            {"model_version": MODEL_VERSION, "selection": selection.to_dict(orient="records")},
        )
        metadata["artifacts"] = artifacts
        metadata["progress"] = {"stage": "COMPLETE", "updated_at": pd.Timestamp.now(tz="UTC")}
        _finish(run_id, "SUCCESS", len(hourly), metadata)
        return {"status": "SUCCESS", "run_id": run_id, "reused": False, "results": json_ready(metadata)}
    except Exception as exc:
        _finish(
            run_id,
            "FAILED",
            None,
            {
                "model_version": MODEL_VERSION,
                "decision": "FAILED",
                "test_used_for_selection": False,
                "production_promotion_allowed": False,
                "automatic_action_allowed": False,
            },
            f"{type(exc).__name__}: {exc}",
        )
        raise
