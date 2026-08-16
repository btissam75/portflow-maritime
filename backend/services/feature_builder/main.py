from __future__ import annotations

import os
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from feature_builder import __version__
from feature_builder.event_aware_gold import (
    FEATURE_VERSION as EVENT_AWARE_GOLD_VERSION,
    build_b57b_event_aware_gold,
)
from feature_builder.one_row_features import (
    FEATURE_VERSION as ONE_ROW_FEATURE_VERSION,
    process_one_row_port_call_features,
)
from feature_builder.pipeline import check_dependencies, process_netcdf_object
from feature_builder.port_calls import process_port_call_object
from feature_builder.quality_gate import process_port_call_quality_gate
from feature_builder.wave_features import process_multi_horizon_wave_features
from feature_builder.tir_incremental import (
    COLLECTOR_VERSION as TIR_INCREMENTAL_COLLECTOR_VERSION,
    ingest_tir_increment,
    list_incoming_tir_objects,
)


app = FastAPI(
    title="Smart Port Maritime Feature Builder",
    version=__version__,
    description=(
        "Build Copernicus observations, TIR port calls, revision-aware quality "
        "gates, legacy multi-horizon features and B54F one-row datasets."
    ),
)

NETCDF_PROCESS_LOCK = threading.Lock()
PORT_CALL_PROCESS_LOCK = threading.Lock()
QUALITY_GATE_PROCESS_LOCK = threading.Lock()
WAVE_FEATURE_PROCESS_LOCK = threading.Lock()
ONE_ROW_FEATURE_PROCESS_LOCK = threading.Lock()
EVENT_AWARE_GOLD_LOCK = threading.Lock()
TIR_INCREMENTAL_INGEST_LOCK = threading.Lock()


class ProcessRequest(BaseModel):
    source_bucket: str = Field(default="bronze-maritime", min_length=3)
    source_key: str = Field(min_length=3)
    output_bucket: str = Field(default="silver-maritime", min_length=3)
    source_last_modified: datetime | None = None
    force: bool = False


class PortCallProcessRequest(BaseModel):
    source_bucket: str = Field(default="bronze-maritime", min_length=3)
    source_key: str = Field(min_length=3)
    manifest_key: str = Field(min_length=3)
    output_bucket: str = Field(default="silver-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class PortCallQualityGateRequest(BaseModel):
    source_bucket: str = Field(default="silver-maritime", min_length=3)
    source_key: str = Field(min_length=3)
    output_bucket: str = Field(default="silver-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class MultiHorizonWaveFeatureRequest(BaseModel):
    output_bucket: str = Field(default="silver-maritime", min_length=3)
    output_prefix: str = "version=1"
    horizons_h: list[int] = Field(default_factory=lambda: [24, 12, 6, 3])
    force: bool = False


class OneRowFeatureRequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    force: bool = False


class EventAwareGoldRequest(BaseModel):
    output_bucket: str = Field(default="gold-maritime", min_length=3)
    output_prefix: str = "version=1"
    materialize_timescale: bool = True
    force: bool = False


class TirIncrementalIngestRequest(BaseModel):
    source_bucket: str = Field(default="bronze-maritime", min_length=3)
    source_key: str = Field(min_length=3)
    delete_source: bool = True
    allowed_lateness_days: int = Field(default=3, ge=0, le=31)


@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "service": "feature-builder", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    try:
        dependencies = check_dependencies()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready", "dependencies": dependencies}


@app.post("/v1/process")
async def process(request: ProcessRequest) -> dict:
    def run() -> dict:
        if not NETCDF_PROCESS_LOCK.acquire(blocking=False):
            raise RuntimeError("Another NetCDF object is currently being processed")
        try:
            return process_netcdf_object(
                source_bucket=request.source_bucket,
                source_key=request.source_key,
                output_bucket=request.output_bucket,
                source_last_modified=request.source_last_modified,
                force=request.force,
            )
        finally:
            NETCDF_PROCESS_LOCK.release()

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently being processed" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Feature build failed: {exc}") from exc


@app.post("/v1/port-calls/process")
async def process_port_calls(request: PortCallProcessRequest) -> dict:
    def run() -> dict:
        if not PORT_CALL_PROCESS_LOCK.acquire(blocking=False):
            raise RuntimeError("Another port-call dataset is currently being processed")
        try:
            return process_port_call_object(
                source_bucket=request.source_bucket,
                source_key=request.source_key,
                manifest_key=request.manifest_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        finally:
            PORT_CALL_PROCESS_LOCK.release()

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently being processed" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Port-call build failed: {exc}") from exc


@app.post("/v1/port-calls/quality-gate")
async def quality_gate_port_calls(request: PortCallQualityGateRequest) -> dict:
    def run() -> dict:
        if not QUALITY_GATE_PROCESS_LOCK.acquire(blocking=False):
            raise RuntimeError("Another port-call quality gate is currently running")
        try:
            return process_port_call_quality_gate(
                source_bucket=request.source_bucket,
                source_key=request.source_key,
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        finally:
            QUALITY_GATE_PROCESS_LOCK.release()

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently running" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Port-call quality gate failed: {exc}") from exc


@app.post("/v1/port-calls/wave-features")
async def build_multi_horizon_wave_features(
    request: MultiHorizonWaveFeatureRequest,
) -> dict:
    def run() -> dict:
        if not WAVE_FEATURE_PROCESS_LOCK.acquire(blocking=False):
            raise RuntimeError("Another B54C wave feature build is currently running")
        try:
            return process_multi_horizon_wave_features(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                horizons_h=request.horizons_h,
                force=request.force,
            )
        finally:
            WAVE_FEATURE_PROCESS_LOCK.release()

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently running" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B54C wave feature build failed: {exc}") from exc


@app.post("/v1/port-calls/one-row-features")
async def build_one_row_features(request: OneRowFeatureRequest) -> dict:
    def run() -> dict:
        if not ONE_ROW_FEATURE_PROCESS_LOCK.acquire(blocking=False):
            raise RuntimeError("Another B54F-A one-row build is currently running")
        try:
            return process_one_row_port_call_features(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                force=request.force,
            )
        finally:
            ONE_ROW_FEATURE_PROCESS_LOCK.release()

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently running" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B54F-A build failed: {exc}") from exc


@app.post("/v1/tir/event-aware-gold")
async def build_event_aware_gold(request: EventAwareGoldRequest) -> dict:
    def run() -> dict:
        if not EVENT_AWARE_GOLD_LOCK.acquire(blocking=False):
            raise RuntimeError("Another B57B event-aware Gold build is currently running")
        try:
            return build_b57b_event_aware_gold(
                output_bucket=request.output_bucket,
                output_prefix=request.output_prefix,
                materialize_timescale=request.materialize_timescale,
                force=request.force,
            )
        finally:
            EVENT_AWARE_GOLD_LOCK.release()

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently running" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B57B Gold build failed: {exc}") from exc


@app.get("/v1/tir/incoming")
def list_tir_incoming(limit: int = 20) -> dict:
    try:
        objects = list_incoming_tir_objects(limit=limit)
        return {"status": "SUCCESS", "objects": objects, "count": len(objects)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B57G listing failed: {exc}") from exc


@app.post("/v1/tir/incremental-ingest")
async def ingest_tir_incremental(request: TirIncrementalIngestRequest) -> dict:
    def run() -> dict:
        if not TIR_INCREMENTAL_INGEST_LOCK.acquire(blocking=False):
            raise RuntimeError("Another B57G TIR ingestion is currently running")
        try:
            return ingest_tir_increment(
                source_bucket=request.source_bucket,
                source_key=request.source_key,
                delete_source=request.delete_source,
                allowed_lateness_days=request.allowed_lateness_days,
            )
        finally:
            TIR_INCREMENTAL_INGEST_LOCK.release()

    try:
        return await run_in_threadpool(run)
    except RuntimeError as exc:
        status_code = 409 if "currently running" in str(exc) else 422
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"B57G ingestion failed: {exc}") from exc


@app.get("/config")
def config() -> dict:
    return {
        "target_port": os.getenv("TARGET_PORT_CODE", "MAPTM"),
        "target_latitude": float(os.getenv("TARGET_LATITUDE", "35.8892")),
        "target_longitude": float(os.getenv("TARGET_LONGITUDE", "-5.5000")),
        "feature_version": os.getenv("FEATURE_VERSION", "copernicus-point-v1"),
        "source_timezone": os.getenv("SOURCE_TIMEZONE", "Africa/Casablanca"),
        "quality_gate_version": "b54b-revision-aware-v1",
        "wave_feature_version": "b54c-wave-history-v1",
        "wave_feature_horizons_h": [24, 12, 6, 3],
        "wave_feature_policy": "ASOF_PAST_ONLY_NO_SPLIT",
        "one_row_feature_version": ONE_ROW_FEATURE_VERSION,
        "event_aware_gold_version": EVENT_AWARE_GOLD_VERSION,
        "tir_incremental_collector_version": TIR_INCREMENTAL_COLLECTOR_VERSION,
        "b57b_grain": "ONE_ROW_PER_UTC_DAY",
        "b57b_prediction_at": "UTC_DAY_START",
        "b57b_policy": "PREDICTIVE_PAST_ONLY_EXPLANATORY_REALIZED_NO_SPLIT_NO_TRAINING",
        "one_row_grain": "ONE_ROW_PER_PORT_CALL",
        "one_row_cutoff": "PLANNED_ETA_MINUS_24H",
        "one_row_split_policy": "NO_SPLIT_IN_B54F_A_B",
    }
