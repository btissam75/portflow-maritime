\set ON_ERROR_STOP on

SELECT 'CREATE DATABASE mlflow'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec

ALTER DATABASE mlflow SET timezone TO 'UTC';

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS serving;
CREATE SCHEMA IF NOT EXISTS audit;

CREATE TABLE IF NOT EXISTS audit.ingestion_run (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'RUNNING',
    object_uri TEXT,
    row_count BIGINT,
    checksum TEXT,
    error_message TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS core.maritime_observation (
    observed_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    wave_height_m REAL,
    wave_period_s REAL,
    wave_direction_deg REAL,
    wind_speed_ms REAL,
    wind_direction_deg REAL,
    surface_current_ms REAL,
    visibility_m REAL,
    pressure_hpa REAL,
    quality_flag SMALLINT NOT NULL DEFAULT 0,
    ingestion_run_id UUID REFERENCES audit.ingestion_run(run_id),
    PRIMARY KEY (observed_at, source, latitude, longitude)
);

SELECT create_hypertable(
    'core.maritime_observation',
    by_range('observed_at'),
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS core.vessel_position (
    observed_at TIMESTAMPTZ NOT NULL,
    mmsi BIGINT NOT NULL,
    imo BIGINT,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    speed_over_ground_kn REAL,
    course_over_ground_deg REAL,
    heading_deg REAL,
    navigation_status TEXT,
    destination TEXT,
    reported_eta TIMESTAMPTZ,
    ingestion_run_id UUID REFERENCES audit.ingestion_run(run_id),
    PRIMARY KEY (observed_at, mmsi)
);

SELECT create_hypertable(
    'core.vessel_position',
    by_range('observed_at'),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS vessel_position_mmsi_time_idx
    ON core.vessel_position (mmsi, observed_at DESC);

CREATE TABLE IF NOT EXISTS core.port_call (
    port_call_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    port_code TEXT NOT NULL DEFAULT 'MAPTM',
    terminal_code TEXT,
    mmsi BIGINT,
    imo BIGINT,
    vessel_name TEXT,
    voyage_id TEXT,
    planned_eta TIMESTAMPTZ,
    planned_etb TIMESTAMPTZ,
    planned_etd TIMESTAMPTZ,
    actual_ata TIMESTAMPTZ,
    actual_atb TIMESTAMPTZ,
    actual_atd TIMESTAMPTZ,
    cargo_type TEXT,
    vessel_type TEXT,
    source TEXT NOT NULL,
    source_record_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_record_id)
);

CREATE INDEX IF NOT EXISTS port_call_imo_eta_idx
    ON core.port_call (imo, planned_eta DESC);

CREATE INDEX IF NOT EXISTS port_call_mmsi_eta_idx
    ON core.port_call (mmsi, planned_eta DESC);

CREATE TABLE IF NOT EXISTS features.vessel_snapshot (
    snapshot_at TIMESTAMPTZ NOT NULL,
    port_call_id UUID NOT NULL REFERENCES core.port_call(port_call_id),
    distance_to_port_nm REAL,
    speed_over_ground_kn REAL,
    route_wave_exposure REAL,
    route_wind_exposure REAL,
    wave_height_now_m REAL,
    wave_height_forecast_6h_m REAL,
    wave_height_forecast_12h_m REAL,
    port_congestion_index REAL,
    berth_occupancy_ratio REAL,
    vessels_waiting INTEGER,
    feature_version TEXT NOT NULL,
    feature_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (snapshot_at, port_call_id, feature_version)
);

SELECT create_hypertable(
    'features.vessel_snapshot',
    by_range('snapshot_at'),
    if_not_exists => TRUE
);

CREATE TABLE IF NOT EXISTS serving.delay_prediction (
    predicted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    prediction_id UUID NOT NULL DEFAULT gen_random_uuid(),
    port_call_id UUID NOT NULL REFERENCES core.port_call(port_call_id),
    snapshot_at TIMESTAMPTZ NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    eta_p50 TIMESTAMPTZ,
    eta_p80 TIMESTAMPTZ,
    eta_p95 TIMESTAMPTZ,
    delay_p50_minutes REAL,
    delay_p80_minutes REAL,
    probability_delay_2h REAL,
    probability_delay_6h REAL,
    risk_level TEXT,
    explanation JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (predicted_at, prediction_id)
);

SELECT create_hypertable(
    'serving.delay_prediction',
    by_range('predicted_at'),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS delay_prediction_call_time_idx
    ON serving.delay_prediction (port_call_id, predicted_at DESC);
