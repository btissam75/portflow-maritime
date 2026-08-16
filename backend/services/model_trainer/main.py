from __future__ import annotations

import os
import threading
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from model_trainer import __version__
from model_trainer.influence_study import (
    STUDY_VERSION as INFLUENCE_STUDY_VERSION,
    run_b54g_influence_study,
)
from model_trainer.dependency_audit import (
    AUDIT_VERSION,
    run_b54ea_dependency_audit,
)
from model_trainer.model_stress import (
    STRESS_VERSION as MODEL_STRESS_VERSION,
    run_b54fd1_model_stress,
)
from model_trainer.one_row_audit import (
    AUDIT_VERSION as ONE_ROW_AUDIT_VERSION,
    run_b54fb_one_row_audit,
)
from model_trainer.split_stress import (
    SPLIT_VERSION,
    run_b54fc_split_stress_audit,
)
from model_trainer.train_readiness import (
    AUDIT_VERSION as TRAIN_READINESS_VERSION,
    run_b54fd0_train_readiness,
)
from model_trainer.operational_feasibility import (
    AUDIT_VERSION as OPERATIONAL_FEASIBILITY_VERSION,
    run_b56a_operational_feasibility,
)
from model_trainer.arrival_flow_baselines import (
    TRAINING_VERSION as ARRIVAL_FLOW_BASELINE_VERSION,
    run_b56b_arrival_flow_baselines,
)
from model_trainer.arrival_flow_enrichment import (
    ENRICHMENT_VERSION as ARRIVAL_FLOW_ENRICHMENT_VERSION,
    run_b56c_arrival_flow_enrichment,
)
from model_trainer.temporal_regime_audit import (
    AUDIT_VERSION as TEMPORAL_REGIME_AUDIT_VERSION,
    run_b57a_temporal_regime_audit,
)
from model_trainer.event_aware_baselines import (
    TRAINING_VERSION as EVENT_AWARE_BASELINE_VERSION,
    run_b57c_event_aware_baselines,
)
from model_trainer.operational_daily_cycle import (
    CYCLE_VERSION as OPERATIONAL_DAILY_CYCLE_VERSION,
    monitoring_b57f,
    run_b57f_operational_cycle,
)
from model_trainer.adaptive_recalibration import (
    API_VERSION as ADAPTIVE_RECALIBRATION_VERSION,
    forecast_b57e_daily,
    initialize_b57e,
    monitoring_b57e,
    recalibrate_b57e,
    register_b57e_observations,
    reload_b57e_runtime,
    runtime_status_b57e,
)
from model_trainer.probabilistic_forecast import (
    API_VERSION as PROBABILISTIC_FORECAST_API_VERSION,
    forecast_b57d_daily,
    promote_b57d_probabilistic_forecaster,
    reload_b57d_runtime,
    runtime_status_b57d,
)
from model_trainer.arrival_flow_probabilistic_ensemble import (
    ENSEMBLE_VERSION as ARRIVAL_FLOW_PROBABILISTIC_ENSEMBLE_VERSION,
    run_b56e_arrival_flow_probabilistic_ensemble,
)
from model_trainer.arrival_flow_expert_count import (
    EXPERT_VERSION as ARRIVAL_FLOW_EXPERT_COUNT_VERSION,
    run_b56g_expert_probabilistic_count,
)
from model_trainer.arrival_flow_hybrid_calibration import (
    HYBRID_VERSION as ARRIVAL_FLOW_HYBRID_CALIBRATION_VERSION,
    run_b56g_v2_hybrid_calibration,
)
from model_trainer.arrival_flow_asymmetric_calibration import (
    ASYMMETRIC_VERSION as ARRIVAL_FLOW_ASYMMETRIC_CALIBRATION_VERSION,
    run_b56g_v21_asymmetric_calibration,
)
from model_trainer.arrival_flow_shadow_monitor import (
    MONITOR_VERSION as ARRIVAL_FLOW_SHADOW_MONITOR_VERSION,
    register_shadow_forecast,
    register_shadow_observation,
    run_b56g_v21_shadow_monitor,
)
from model_trainer.weather_timeseries_audit import (
    AUDIT_VERSION as WEATHER_TIMESERIES_AUDIT_VERSION,
    run_b58a_weather_timeseries_audit,
)
from model_trainer.wave_rolling_backtest import (
    MODEL_VERSION as WAVE_ROLLING_BACKTEST_VERSION,
    run_b58b_wave_rolling_backtest,
)
from model_trainer.wave_sequence_challengers import (
    MODEL_VERSION as WAVE_SEQUENCE_CHALLENGER_VERSION,
    run_b58b1_wave_sequence_challengers,
)
from model_trainer.trainer import check_dependencies, train_b54d


app = FastAPI(
    title="Smart Port Maritime Model Trainer",
    version=__version__,
    description=(
        "Train leakage-safe temporal maritime delay baselines and run "
        "dependency, split, weather, train-readiness and fair model-stress audits."
    ),
)

JOB_LOCK = threading.Lock()


B57C_ASYNC_LOCK = threading.Lock()
B57C_ASYNC_STATE = {
    "state": "IDLE",
    "started_at": None,
    "finished_at": None,
    "result": None,
    "error": None,
}


class B54DRequest(BaseModel):
    source_bucket: str = Field(default="silver-maritime", min_length=3)
    source_key: str = Field(min_length=3)
    report_key: str = Field(min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    train_fraction: float = Field(default=0.70, gt=0.50, lt=0.85)
    valid_fraction: float = Field(default=0.15, gt=0.05, lt=0.30)
    force: bool = False


class B54EARequest(BaseModel):
    source_bucket: str = Field(default="silver-maritime", min_length=3)
    source_key: str = Field(min_length=3)
    artifacts_bucket: str = Field(default="gold-maritime", min_length=3)
    split_key: str = Field(min_length=3)
    feature_config_key: str = Field(min_length=3)
    decision_key: str = Field(min_length=3)
    valid_predictions_key: str = Field(min_length=3)
    test_predictions_key: str = Field(min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class B54FBRequest(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    full_key: str = Field(min_length=3)
    model_ready_key: str = Field(min_length=3)
    feature_config_key: str = Field(min_length=3)
    build_report_key: str = Field(min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class B54FCRequest(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    model_ready_key: str = Field(min_length=3)
    feature_config_key: str = Field(min_length=3)
    upstream_decision_key: str = Field(min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    train_fraction: float = Field(default=0.70, gt=0.50, lt=0.85)
    valid_fraction: float = Field(default=0.15, gt=0.05, lt=0.30)
    purge_hours: int = Field(default=72, ge=72, le=336)
    random_seed: int = 42
    force: bool = False


class B54FD0Request(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    model_ready_key: str = Field(min_length=3)
    feature_config_key: str = Field(min_length=3)
    split_assignments_key: str = Field(min_length=3)
    split_decision_key: str = Field(min_length=3)
    build_report_key: str = Field(min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    sample_size: int = Field(default=600, ge=120, le=3000)
    numeric_atol: float = Field(default=0.002, gt=0, le=0.1)
    numeric_rtol: float = Field(default=0.0001, ge=0, le=0.01)
    force: bool = False


class B54FD1Request(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    model_ready_key: str = Field(min_length=3)
    split_assignments_key: str = Field(min_length=3)
    split_decision_key: str = Field(min_length=3)
    readiness_config_key: str = Field(min_length=3)
    readiness_decision_key: str = Field(min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=2"
    force: bool = False


class B54GRequest(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    model_ready_key: str = Field(min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class B56ARequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_timescale: bool = True
    force: bool = False


class B56BRequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class B56CRequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_timescale: bool = True
    force: bool = False


class B57ARequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_event_calendar: bool = True
    force: bool = False


class B57CRequest(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    source_key: str = Field(
        default="datasets/b57b/version=1/tir_daily_predictive_gold_v1.parquet",
        min_length=3,
    )
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class B57DPromotionRequest(BaseModel):
    artifact_bucket: str = Field(default="gold-maritime", min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class B57DForecastRequest(BaseModel):
    prediction_date: str = Field(min_length=10, max_length=35)


class B57EInitializationRequest(BaseModel):
    artifact_bucket: str = Field(default="gold-maritime", min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class B57EForecastRequest(BaseModel):
    prediction_date: str = Field(min_length=10, max_length=35)


class B57EObservationRequest(BaseModel):
    prediction_date: str = Field(min_length=10, max_length=35)
    available_at: str = Field(min_length=20, max_length=40)
    source: str = Field(min_length=3, max_length=100)
    values: dict[str, float]


class B57ERecalibrationRequest(BaseModel):
    as_of: str | None = None


class B57FOperationalCycleRequest(BaseModel):
    artifact_bucket: str = Field(default="gold-maritime", min_length=3)


class B56ERequest(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_timescale: bool = True
    force: bool = False


class B56GRequest(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_timescale: bool = True
    simulation_samples: int = Field(default=400, ge=200, le=2000)
    force: bool = False


class B56GV2Request(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=2"
    materialize_timescale: bool = True
    force: bool = False


class B56GV21Request(BaseModel):
    source_bucket: str = Field(default="gold-maritime", min_length=3)
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_timescale: bool = True
    force: bool = False


class B56GV21ShadowForecastRequest(BaseModel):
    as_of_time: datetime
    horizon_h: int = Field(ge=6, le=24)
    selected_policy: str = Field(min_length=3, max_length=100)
    point_prediction: float = Field(ge=0)
    p10: float = Field(ge=0)
    p50: float = Field(ge=0)
    p90: float = Field(ge=0)
    issued_at: datetime
    source_snapshot_time: datetime
    payload: dict = Field(default_factory=dict)


class B56GV21ShadowObservationRequest(BaseModel):
    forecast_id: str = Field(min_length=32, max_length=40)
    actual_arrivals: float = Field(ge=0)
    source: str = Field(min_length=3, max_length=100)
    source_watermark: datetime
    available_at: datetime
    payload: dict = Field(default_factory=dict)


class B56GV21ShadowMonitorRequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    as_of: datetime | None = None
    auto_capture_observations: bool = True
    force: bool = False


class B58ARequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_timescale: bool = True
    force: bool = False


class B58BRequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_timescale: bool = True
    force: bool = False


class B58B1Request(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "model-trainer", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    try:
        dependencies = check_dependencies()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "dependencies": dependencies}


@app.get("/config")
def config() -> dict:
    return {
        "training_version": "b54d-temporal-catboost-v1",
        "dependency_audit_version": AUDIT_VERSION,
        "one_row_audit_version": ONE_ROW_AUDIT_VERSION,
        "split_stress_version": SPLIT_VERSION,
        "train_readiness_version": TRAIN_READINESS_VERSION,
        "model_stress_version": MODEL_STRESS_VERSION,
        "influence_study_version": INFLUENCE_STUDY_VERSION,
        "operational_feasibility_version": OPERATIONAL_FEASIBILITY_VERSION,
        "weather_timeseries_audit_version": WEATHER_TIMESERIES_AUDIT_VERSION,
        "wave_rolling_backtest_version": WAVE_ROLLING_BACKTEST_VERSION,
        "wave_sequence_challenger_version": WAVE_SEQUENCE_CHALLENGER_VERSION,
        "arrival_flow_baseline_version": ARRIVAL_FLOW_BASELINE_VERSION,
        "arrival_flow_enrichment_version": ARRIVAL_FLOW_ENRICHMENT_VERSION,
        "arrival_flow_probabilistic_ensemble_version": ARRIVAL_FLOW_PROBABILISTIC_ENSEMBLE_VERSION,
        "arrival_flow_expert_count_version": ARRIVAL_FLOW_EXPERT_COUNT_VERSION,
        "arrival_flow_hybrid_calibration_version": ARRIVAL_FLOW_HYBRID_CALIBRATION_VERSION,
        "arrival_flow_asymmetric_calibration_version": ARRIVAL_FLOW_ASYMMETRIC_CALIBRATION_VERSION,
        "arrival_flow_shadow_monitor_version": ARRIVAL_FLOW_SHADOW_MONITOR_VERSION,
        "temporal_regime_audit_version": TEMPORAL_REGIME_AUDIT_VERSION,
        "event_aware_baseline_version": EVENT_AWARE_BASELINE_VERSION,
        "probabilistic_forecast_api_version": PROBABILISTIC_FORECAST_API_VERSION,
        "adaptive_recalibration_version": ADAPTIVE_RECALIBRATION_VERSION,
        "operational_daily_cycle_version": OPERATIONAL_DAILY_CYCLE_VERSION,
        "split_policy": "TEMPORAL_BY_PORT_CALL_ID_70_15_15",
        "selection_policy": "SELECT_ON_TRAIN_VALIDATE_ON_VALID_TEST_DIAGNOSTIC_ONLY",
        "b54f_policy": "AUDIT_FULL_NO_SPLIT_NO_TRAINING",
        "b54fc_policy": (
            "FREEZE_RANDOM_IID_RANDOM_BY_IMO_TEMPORAL_PURGED_ROLLING_CV_"
            "NO_TARGET_MODEL_TRAINING"
        ),
        "b54fd0_policy": (
            "INDEPENDENT_SOURCE_RECALCULATION_TRAIN_ONLY_DEPENDENCY_ANALYSIS_"
            "NO_MODEL_TRAINING"
        ),
        "b54fd1_policy": (
            "SAME_FROZEN_FEATURES_SAME_MODEL_CAPACITY_VALID_ONLY_SELECTION_"
            "RANDOM_DIAGNOSTIC_TEMPORAL_PURGED_OFFICIAL"
        ),
        "b54g_policy": (
            "SAFE_T24_SEPARATE_FROM_ORACLE_EXPLANATORY_TRACKS_"
            "ADJUSTED_ASSOCIATION_NOT_CAUSATION_NO_PREDICTIVE_TRAINING"
        ),
        "b56a_policy": (
            "FULL_HOURLY_FEASIBILITY_AUDIT_NO_SPLIT_NO_TRAINING_"
            "NO_BRONZE_MUTATION_STRICT_PAST_FEATURES"
        ),
        "b56b_policy": (
            "TEMPORAL_70_15_15_PURGED_24H_VALID_SELECTION_"
            "TEST_FINAL_COUNT_MODELS_BLOCK_BOOTSTRAP_WAVE_ABLATION"
        ),
        "b56c_policy": (
            "STRICT_PAST_ENRICHMENT_NO_FINAL_ETA_TEMPORAL_PURGED_"
            "VALID_SELECTION_TEST_FINAL_BOOTSTRAP_PROMOTION"
        ),
        "b57a_policy": (
            "SEASONALITY_CHANGE_POINTS_EVENT_STUDY_SOURCE_BREAKS_"
            "NO_SPLIT_NO_TRAINING_NO_BRONZE_MUTATION"
        ),
        "b57c_policy": (
            "WALK_FORWARD_PURGED_7D_CV_ONLY_SELECTION_2026_TEST_DIAGNOSTIC_"
            "FULL_NO_PORT_OFFICIAL_PRE_BREAK_PORT_DIAGNOSTIC"
        ),
        "b57d_policy": (
            "OOF_CONFORMAL_TEST_RELIABILITY_NO_RETRAINING_"
            "REJECT_DATES_WITHOUT_CANONICAL_FEATURES_NO_TARGET_EXPOSURE"
        ),
        "b57e_policy": (
            "PAST_ONLY_ASYMMETRIC_WEIGHTED_CONFORMAL_"
            "BITEMPORAL_LIVE_LABELS_NO_TEST_TUNING_"
            "DRIFT_GUARD_FALLBACK"
        ),
        "b57f_policy": (
            "ONE_FUTURE_DAY_NO_TARGET_STABLE_OUTCOME_TWICE_"
            "FORECAST_PRECEDES_AVAILABILITY_NO_BACKFILL"
        ),
        "b56e_policy": (
            "REUSE_FROZEN_B56C_PREDICTIONS_VALID_SELECTION_"
            "TEST_LOCKED_PAST_ONLY_ADAPTATION_NO_RETRAINING_"
            "SOURCE_FRESHNESS_BLOCKS_LIVE_SERVING"
        ),
        "b56g_policy": (
            "TRAIN_ONLY_INCREMENTAL_COUNTS_VALID_SELECTION_"
            "TEST_LOCKED_MATURED_ONLINE_ADAPTIVE_CQR_"
            "COHERENT_6H_12H_24H_SHADOW_ONLY"
        ),
        "b56g_v2_policy": (
            "PRESERVE_B56E_POINT_VALID_ONLY_CALIBRATION_"
            "MATURED_LABELS_COHERENT_INTERVALS_SHADOW_ONLY"
        ),
        "b56g_v21_policy": (
            "PRESERVE_B56E_POINT_VALID_ONLY_ASYMMETRIC_ACI_"
            "MATURED_LABELS_ROLLING_GATES_SHADOW_ONLY"
        ),
        "b56g_v21_shadow_policy": (
            "PROSPECTIVE_ONLY_IMMUTABLE_LEDGER_"
            "MATURED_AVAILABLE_AT_LABELS_NO_TEST_REUSE"
        ),
        "b58a_policy": (
            "HOURLY_WEATHER_AUDIT_PAST_ONLY_NO_INTERPOLATION_"
            "NO_SPLIT_NO_TRAINING_NO_SOURCE_MUTATION_"
            "FORMAL_REPLAY_BLOCKED_WITHOUT_AVAILABLE_AT"
        ),
        "b58b_policy": (
            "WAVE_ONLY_GLOBAL_DIRECT_ROLLING_ORIGIN_"
            "LATENCY_1_3_6H_PURGE_72H_VALID_SELECTION_"
            "TEST_DIAGNOSTIC_ADAPTIVE_CONFORMAL_SHADOW_ONLY"
        ),
        "b58b1_policy": (
            "NHITS_PATCHTST_CPU_CHALLENGERS_SAME_TEMPORAL_"
            "BOUNDARIES_PURGE_72H_LATENCY_3H_VALID_SELECTION_"
            "TEST_DIAGNOSTIC_REPLACE_ONLY_IF_GAIN_GE_5PCT"
        ),
        "task_type": os.getenv("B54D_TASK_TYPE", "CPU"),
        "thread_count": int(os.getenv("B54D_THREAD_COUNT", "2")),
    }


def _locked_job(callback):
    if not JOB_LOCK.acquire(blocking=False):
        raise RuntimeError("Another Model Trainer job is currently active")
    try:
        return callback()
    finally:
        JOB_LOCK.release()


@app.post("/v1/train/b54d")
async def train_temporal_baselines(request: B54DRequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: train_b54d(
                source_bucket=request.source_bucket,
                source_key=request.source_key,
                report_key=request.report_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                train_fraction=request.train_fraction,
                valid_fraction=request.valid_fraction,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B54D training failed: {exc}") from exc


@app.post("/v1/audit/b54ea")
async def dependency_weather_audit(request: B54EARequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b54ea_dependency_audit(
                source_bucket=request.source_bucket,
                source_key=request.source_key,
                artifacts_bucket=request.artifacts_bucket,
                split_key=request.split_key,
                feature_config_key=request.feature_config_key,
                decision_key=request.decision_key,
                valid_predictions_key=request.valid_predictions_key,
                test_predictions_key=request.test_predictions_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B54E-A audit failed: {exc}") from exc


@app.post("/v1/audit/b54fb")
async def one_row_structure_dependency_audit(request: B54FBRequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b54fb_one_row_audit(
                source_bucket=request.source_bucket,
                full_key=request.full_key,
                model_ready_key=request.model_ready_key,
                feature_config_key=request.feature_config_key,
                build_report_key=request.build_report_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B54F-B audit failed: {exc}") from exc


@app.post("/v1/splits/b54fc")
async def random_temporal_split_stress_audit(request: B54FCRequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b54fc_split_stress_audit(
                source_bucket=request.source_bucket,
                model_ready_key=request.model_ready_key,
                feature_config_key=request.feature_config_key,
                upstream_decision_key=request.upstream_decision_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                train_fraction=request.train_fraction,
                valid_fraction=request.valid_fraction,
                purge_hours=request.purge_hours,
                random_seed=request.random_seed,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B54F-C split audit failed: {exc}") from exc


@app.post("/v1/audit/b54fd0")
async def independent_train_readiness_audit(request: B54FD0Request) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b54fd0_train_readiness(
                source_bucket=request.source_bucket,
                model_ready_key=request.model_ready_key,
                feature_config_key=request.feature_config_key,
                split_assignments_key=request.split_assignments_key,
                split_decision_key=request.split_decision_key,
                build_report_key=request.build_report_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                sample_size=request.sample_size,
                numeric_atol=request.numeric_atol,
                numeric_rtol=request.numeric_rtol,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B54F-D0 audit failed: {exc}") from exc


@app.post("/v1/train/b54fd1")
async def fair_split_model_stress(request: B54FD1Request) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b54fd1_model_stress(
                source_bucket=request.source_bucket,
                model_ready_key=request.model_ready_key,
                split_assignments_key=request.split_assignments_key,
                split_decision_key=request.split_decision_key,
                readiness_config_key=request.readiness_config_key,
                readiness_decision_key=request.readiness_decision_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B54F-D1 model stress failed: {exc}"
        ) from exc


@app.post("/v1/audit/b54g")
async def maritime_delay_influence_study(request: B54GRequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b54g_influence_study(
                source_bucket=request.source_bucket,
                model_ready_key=request.model_ready_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B54G influence study failed: {exc}"
        ) from exc

@app.post("/v1/audit/b56a")
async def operational_forecast_dataset_feasibility(request: B56ARequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b56a_operational_feasibility(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B56A feasibility audit failed: {exc}"
        ) from exc

@app.post("/v1/train/b56b")
async def train_arrival_flow_temporal_baselines(request: B56BRequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b56b_arrival_flow_baselines(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B56B arrival-flow training failed: {exc}"
        ) from exc

@app.post("/v1/train/b56c")
async def train_arrival_flow_enrichment(request: B56CRequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b56c_arrival_flow_enrichment(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B56C arrival-flow enrichment failed: {exc}"
        ) from exc

@app.post("/v1/audit/b57a")
async def temporal_regime_and_data_break_audit(request: B57ARequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b57a_temporal_regime_audit(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_event_calendar=request.materialize_event_calendar,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57A temporal regime audit failed: {exc}"
        ) from exc

@app.post("/v1/train/b57c")
async def train_event_aware_temporal_baselines(request: B57CRequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b57c_event_aware_baselines(
                source_bucket=request.source_bucket,
                source_key=request.source_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57C event-aware training failed: {exc}"
        ) from exc

def _run_b57c_async(payload: dict) -> None:
    with B57C_ASYNC_LOCK:
        B57C_ASYNC_STATE.update(
            state="RUNNING", result=None, error=None
        )
    try:
        result = _locked_job(
            lambda: run_b57c_event_aware_baselines(**payload)
        )
        with B57C_ASYNC_LOCK:
            B57C_ASYNC_STATE.update(
                state="SUCCESS",
                finished_at=datetime.now(timezone.utc).isoformat(),
                result=result,
            )
    except Exception as exc:
        with B57C_ASYNC_LOCK:
            B57C_ASYNC_STATE.update(
                state="FAILED",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=str(exc),
            )


@app.post("/v1/train/b57c/start")
async def start_event_aware_temporal_baselines(request: B57CRequest) -> dict:
    with B57C_ASYNC_LOCK:
        if B57C_ASYNC_STATE["state"] == "RUNNING":
            return {"started": False, **B57C_ASYNC_STATE}
        payload = {
            "source_bucket": request.source_bucket,
            "source_key": request.source_key,
            "output_bucket": request.output_bucket,
            "output_prefix": request.output_prefix,
            "force": request.force,
        }
        B57C_ASYNC_STATE.update(
            state="STARTING",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            result=None,
            error=None,
        )
        thread = threading.Thread(
            target=_run_b57c_async,
            args=(payload,),
            name="b57c-event-aware-training",
            daemon=True,
        )
        thread.start()
        return {"started": True, **B57C_ASYNC_STATE}


@app.get("/v1/train/b57c/status")
async def event_aware_temporal_baselines_status() -> dict:
    with B57C_ASYNC_LOCK:
        return dict(B57C_ASYNC_STATE)

@app.post("/v1/promote/b57d")
async def promote_probabilistic_forecast_api(
    request: B57DPromotionRequest,
) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: promote_b57d_probabilistic_forecaster(
                artifact_bucket=request.artifact_bucket,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57D promotion failed: {exc}"
        ) from exc


@app.get("/v1/forecast/b57d/status")
async def probabilistic_forecast_status() -> dict:
    status = runtime_status_b57d()
    if status.get("status") == "NOT_LOADED":
        try:
            return await run_in_threadpool(reload_b57d_runtime)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"B57D runtime is not available: {exc}"
            ) from exc
    return status


@app.post("/v1/forecast/b57d/reload")
async def reload_probabilistic_forecast() -> dict:
    try:
        return await run_in_threadpool(reload_b57d_runtime)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"B57D reload failed: {exc}"
        ) from exc


@app.post("/v1/forecast/b57d/daily")
async def daily_probabilistic_forecast(request: B57DForecastRequest) -> dict:
    try:
        return await run_in_threadpool(
            forecast_b57d_daily, request.prediction_date
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57D forecast failed: {exc}"
        ) from exc

@app.post("/v1/forecast/b57e/initialize")
async def initialize_adaptive_recalibration(
    request: B57EInitializationRequest,
) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: initialize_b57e(
                artifact_bucket=request.artifact_bucket,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57E initialization failed: {exc}"
        ) from exc


@app.get("/v1/forecast/b57e/status")
async def adaptive_recalibration_status() -> dict:
    status = runtime_status_b57e()
    if status.get("status") == "NOT_LOADED":
        try:
            return await run_in_threadpool(reload_b57e_runtime)
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail=f"B57E runtime is unavailable: {exc}"
            ) from exc
    return status


@app.post("/v1/forecast/b57e/daily")
async def daily_adaptive_forecast(request: B57EForecastRequest) -> dict:
    try:
        return await run_in_threadpool(
            forecast_b57e_daily, request.prediction_date
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57E forecast failed: {exc}"
        ) from exc


@app.post("/v1/forecast/b57e/observations")
async def register_adaptive_observations(
    request: B57EObservationRequest,
) -> dict:
    try:
        return await run_in_threadpool(
            register_b57e_observations,
            request.prediction_date,
            request.available_at,
            request.source,
            request.values,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57E observation registration failed: {exc}"
        ) from exc


@app.post("/v1/forecast/b57e/recalibrate")
async def run_adaptive_recalibration(
    request: B57ERecalibrationRequest,
) -> dict:
    try:
        return await run_in_threadpool(recalibrate_b57e, request.as_of)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57E recalibration failed: {exc}"
        ) from exc


@app.get("/v1/forecast/b57e/monitoring")
async def adaptive_recalibration_monitoring() -> dict:
    try:
        return await run_in_threadpool(monitoring_b57e)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57E monitoring failed: {exc}"
        ) from exc

@app.post("/v1/forecast/b57f/run")
async def run_operational_daily_cycle(
    request: B57FOperationalCycleRequest,
) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b57f_operational_cycle(
                artifact_bucket=request.artifact_bucket,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57F operational cycle failed: {exc}"
        ) from exc


@app.get("/v1/forecast/b57f/monitoring")
async def operational_daily_cycle_monitoring() -> dict:
    try:
        return await run_in_threadpool(monitoring_b57f)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"B57F monitoring failed: {exc}"
        ) from exc

@app.post("/v1/audit/b56e")
async def audit_arrival_flow_probabilistic_ensemble(
    request: B56ERequest,
) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b56e_arrival_flow_probabilistic_ensemble(
                source_bucket=request.source_bucket,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"B56E probabilistic ensemble audit failed: {exc}",
        ) from exc

@app.post("/v1/train/b56g")
async def train_arrival_flow_expert_count(
    request: B56GRequest,
) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b56g_expert_probabilistic_count(
                source_bucket=request.source_bucket,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                simulation_samples=request.simulation_samples,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"B56G expert count forecasting failed: {exc}",
        ) from exc

@app.post("/v1/recalibrate/b56g-v2")
async def recalibrate_arrival_flow_hybrid(
    request: B56GV2Request,
) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b56g_v2_hybrid_calibration(
                source_bucket=request.source_bucket,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"B56G-v2 hybrid calibration failed: {exc}",
        ) from exc

@app.post("/v1/recalibrate/b56g-v2-1")
async def recalibrate_arrival_flow_asymmetric(
    request: B56GV21Request,
) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b56g_v21_asymmetric_calibration(
                source_bucket=request.source_bucket,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"B56G-v2.1 asymmetric calibration failed: {exc}",
        ) from exc

@app.post("/v1/shadow/b56g-v2-1/forecasts")
async def record_arrival_flow_shadow_forecast(
    request: B56GV21ShadowForecastRequest,
) -> dict:
    try:
        return await run_in_threadpool(
            lambda: register_shadow_forecast(
                as_of_time=request.as_of_time,
                horizon_h=request.horizon_h,
                selected_policy=request.selected_policy,
                point_prediction=request.point_prediction,
                p10=request.p10,
                p50=request.p50,
                p90=request.p90,
                issued_at=request.issued_at,
                source_snapshot_time=request.source_snapshot_time,
                payload=request.payload,
            )
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Shadow forecast registration failed: {exc}",
        ) from exc


@app.post("/v1/shadow/b56g-v2-1/observations")
async def record_arrival_flow_shadow_observation(
    request: B56GV21ShadowObservationRequest,
) -> dict:
    try:
        return await run_in_threadpool(
            lambda: register_shadow_observation(
                forecast_id=request.forecast_id,
                actual_arrivals=request.actual_arrivals,
                source=request.source,
                source_watermark=request.source_watermark,
                available_at=request.available_at,
                payload=request.payload,
            )
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Shadow observation registration failed: {exc}",
        ) from exc


@app.post("/v1/monitor/b56g-v2-1-shadow")
async def monitor_arrival_flow_shadow(
    request: B56GV21ShadowMonitorRequest,
) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b56g_v21_shadow_monitor(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                as_of=request.as_of,
                auto_capture_observations=(
                    request.auto_capture_observations
                ),
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"B56G-v2.1 shadow monitor failed: {exc}",
        ) from exc

@app.post("/v1/audit/b58a")
async def audit_weather_timeseries(request: B58ARequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b58a_weather_timeseries_audit(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"B58A weather time-series audit failed: {exc}",
        ) from exc

@app.post("/v1/train/b58b")
async def train_wave_rolling_backtest(request: B58BRequest) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b58b_wave_rolling_backtest(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"B58B wave rolling backtest failed: {exc}",
        ) from exc

@app.post("/v1/train/b58b1")
async def train_wave_sequence_challengers(request: B58B1Request) -> dict:
    def run() -> dict:
        return _locked_job(
            lambda: run_b58b1_wave_sequence_challengers(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        )

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently active" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"B58B.1 native sequence challengers failed: {exc}",
        ) from exc
