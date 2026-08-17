from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel

from platform_api import local_demo


MODEL_VERSION = "b62a-governed-metocean-tail-challenger-v1"
AUDIT_SOURCE = "b62a_governed_metocean_augmentation"
SELECTION_TABLE = "serving.maritime_metocean_tail_challenger_shadow_v1"


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


_pool: SimpleConnectionPool | None = None


def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        dsn = (
            f"host={_env('SMART_PORT_DB_HOST', 'timescaledb')} "
            f"port={_env('SMART_PORT_DB_PORT', '5432')} "
            f"dbname={_env('SMART_PORT_DB_NAME', 'maritime')} "
            f"user={_env('SMART_PORT_DB_USER', 'smartport')} "
            f"password={_env('SMART_PORT_DB_PASSWORD')}"
        )
        _pool = SimpleConnectionPool(1, int(os.getenv("B62A_DB_POOL_SIZE", "4")), dsn=dsn)
    return _pool


@contextmanager
def _connection() -> Iterator[Any]:
    pool = _get_pool()
    connection = pool.getconn()
    try:
        connection.set_session(readonly=True, autocommit=True)
        yield connection
    finally:
        pool.putconn(connection)


class B62AStatus(BaseModel):
    audit_status: str
    decision: str
    model_version: str
    synthetic_rows: int
    synthetic_weight: float
    accepted_challenger_tasks: int
    challenger_tasks: int
    weekly_real_origins: int
    frozen_test_origins: int
    stress_scenarios: int
    critical_gates_passed: bool
    valid_modified: bool
    test_modified: bool
    test_used_for_selection: bool
    production_promotion_allowed: bool
    automatic_action_allowed: bool
    next_block: str | None = None
    finished_at: datetime | None = None


class TaskSelection(BaseModel):
    variable: str
    horizon_h: int
    b62_model: str
    selected_model: str
    challenger_accepted: bool
    valid_b62_mae: float | None = None
    valid_challenger_mae: float | None = None
    valid_challenger_gain_pct: float | None = None
    valid_challenger_coverage: float | None = None
    test_model: str | None = None
    test_mae: float | None = None
    test_bias: float | None = None
    test_coverage: float | None = None
    selection_role: str
    test_role: str
    production_promotion_allowed: bool


router = APIRouter(
    prefix="/api/v1/maritime/metocean-augmentation",
    tags=["Maritime governed metocean augmentation"],
)


@router.get("/status", response_model=B62AStatus)
def status() -> B62AStatus:
    if local_demo.enabled():
        selections = local_demo.metocean_selections()
        return B62AStatus(
            audit_status="DEMO",
            decision="LOCAL_DEMO_CHALLENGER_REVIEW",
            model_version=local_demo.DEMO_METOCEAN_VERSION,
            synthetic_rows=0,
            synthetic_weight=0.0,
            accepted_challenger_tasks=sum(row["challenger_accepted"] for row in selections),
            challenger_tasks=len(selections),
            weekly_real_origins=0,
            frozen_test_origins=0,
            stress_scenarios=0,
            critical_gates_passed=True,
            valid_modified=False,
            test_modified=False,
            test_used_for_selection=False,
            production_promotion_allowed=False,
            automatic_action_allowed=False,
            next_block="CONNECT_REAL_B62A_ARTIFACTS",
            finished_at=local_demo.metocean_forecast(
                "ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT"
            )[0]["issue_at"],
        )
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT status,metadata,finished_at FROM audit.ingestion_run
            WHERE source_name=%s ORDER BY started_at DESC LIMIT 1
            """,
            (AUDIT_SOURCE,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B62A audit unavailable")
    metadata = dict(row[1] or {})
    return B62AStatus(
        audit_status=str(row[0]),
        decision=str(metadata.get("decision", "UNKNOWN")),
        model_version=str(metadata.get("model_version", MODEL_VERSION)),
        synthetic_rows=int(metadata.get("synthetic_rows", 0)),
        synthetic_weight=float(metadata.get("synthetic_weight", 0.0)),
        accepted_challenger_tasks=int(metadata.get("accepted_challenger_tasks", 0)),
        challenger_tasks=int(metadata.get("challenger_tasks", 0)),
        weekly_real_origins=int(metadata.get("weekly_real_origins", 0)),
        frozen_test_origins=int(metadata.get("frozen_test_origins", 0)),
        stress_scenarios=int(metadata.get("stress_scenarios", 0)),
        critical_gates_passed=bool(metadata.get("critical_gates_passed", False)),
        valid_modified=bool(metadata.get("valid_modified", False)),
        test_modified=bool(metadata.get("test_modified", False)),
        test_used_for_selection=bool(metadata.get("test_used_for_selection", False)),
        production_promotion_allowed=bool(metadata.get("production_promotion_allowed", False)),
        automatic_action_allowed=bool(metadata.get("automatic_action_allowed", False)),
        next_block=metadata.get("next_block"),
        finished_at=row[2],
    )


@router.get("/selection", response_model=list[TaskSelection])
def selection() -> list[TaskSelection]:
    if local_demo.enabled():
        return [TaskSelection(**row) for row in local_demo.metocean_selections()]
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT variable,horizon_h,b62_model,selected_model,challenger_accepted,
                   valid_b62_mae,valid_challenger_mae,valid_challenger_gain_pct,
                   valid_challenger_coverage,test_model,test_mae,test_bias,test_coverage,
                   selection_role,test_role,production_promotion_allowed
            FROM {SELECTION_TABLE}
            WHERE model_version=%s
            ORDER BY variable,horizon_h
            """,
            (MODEL_VERSION,),
        )
        rows = cursor.fetchall()
    return [TaskSelection(**dict(zip(TaskSelection.model_fields, row))) for row in rows]


@router.get("/model-card")
def model_card() -> dict[str, Any]:
    if local_demo.enabled():
        return {
            "model_version": local_demo.DEMO_METOCEAN_VERSION,
            "source_b62_model_version": local_demo.DEMO_METOCEAN_VERSION,
            "decision": "LOCAL_DEMO_CHALLENGER_REVIEW",
            "synthetic_scope": "NONE_LOCAL_DEMO",
            "test_role": "NOT_CONSUMED",
            "weekly_replay_role": "LOCAL_DEMO",
            "stress_role": "LOCAL_DEMO",
            "production_promotion_allowed": False,
            "automatic_action_allowed": False,
            "artifacts": {},
        }
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT metadata FROM audit.ingestion_run
            WHERE source_name=%s AND status='SUCCESS'
            ORDER BY finished_at DESC LIMIT 1
            """,
            (AUDIT_SOURCE,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B62A model card unavailable")
    metadata = dict(row[0] or {})
    return {
        "model_version": metadata.get("model_version", MODEL_VERSION),
        "source_b62_model_version": metadata.get("source_b62_model_version"),
        "decision": metadata.get("decision"),
        "synthetic_scope": metadata.get("synthetic_scope"),
        "test_role": metadata.get("test_role"),
        "weekly_replay_role": metadata.get("weekly_replay_role"),
        "stress_role": metadata.get("stress_role"),
        "production_promotion_allowed": False,
        "automatic_action_allowed": False,
        "artifacts": metadata.get("artifacts", {}),
    }
