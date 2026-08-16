from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel


POLICY_VERSION = "b61c-dynamic-alert-policy-v1.1"
DECISION_TABLE = "serving.maritime_port_call_decision_shadow_v1"
POLICY_TABLE = "serving.maritime_decision_policy_v1"
SCORECARD_TABLE = "serving.maritime_decision_policy_scorecard_v1"
AUDIT_SOURCE = "b61c_historical_replay_shadow_decision"
AUDIT_DATASET = "maritime_port_call_dynamic_decision_shadow_v1"


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _database_dsn() -> str:
    return (
        f"host={_env('SMART_PORT_DB_HOST', 'timescaledb')} "
        f"port={_env('SMART_PORT_DB_PORT', '5432')} "
        f"dbname={_env('SMART_PORT_DB_NAME', 'maritime')} "
        f"user={_env('SMART_PORT_DB_USER', 'smartport')} "
        f"password={_env('SMART_PORT_DB_PASSWORD')}"
    )


_pool: SimpleConnectionPool | None = None


def _get_pool() -> SimpleConnectionPool:
    global _pool
    if _pool is None:
        _pool = SimpleConnectionPool(
            minconn=1,
            maxconn=int(os.getenv("B61C_DB_POOL_SIZE", "6")),
            dsn=_database_dsn(),
        )
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


def _normalize_time(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class DecisionStatus(BaseModel):
    audit_status: str
    decision: str
    policy_version: str
    selected_policy_id: str | None
    temporal_model: str | None
    shadow_api_allowed: bool
    dynamic_alert_shadow_allowed: bool
    production_promotion_allowed: bool
    automatic_action_allowed: bool
    fresh_forward_confirmation_required: bool
    finished_at: datetime | None


class SelectedPolicy(BaseModel):
    policy_version: str
    source_model_version: str
    policy_id: str
    selected_on_role: str
    parameters: dict[str, Any]
    quality_gates: list[dict[str, Any]]
    source_mode: str
    production_claim_allowed: bool
    automatic_action_allowed: bool
    fresh_forward_confirmation_required: bool
    created_at: datetime


class DecisionItem(BaseModel):
    port_call_id: str
    landmark_at: datetime
    decision_at: datetime
    evaluation_role: str
    regime: str
    state: str
    previous_state: str
    state_changed: bool
    rank_in_bucket: int
    active_calls: int
    temporal_priority_score: float
    critical_priority_score: float
    p_delay_gt3: float
    p_delay_gt6: float
    p_breach_6h: float
    p_breach_12h: float
    p_breach_24h: float
    remaining_p10_h: float
    remaining_p50_h: float
    remaining_p90_h: float
    alert_active: bool
    new_alert: bool
    action_code: str
    source_mode: str
    production_claim_allowed: bool
    automatic_action_allowed: bool


class DecisionSnapshot(BaseModel):
    requested_at: datetime | None
    resolved_at: datetime
    evaluation_role: str
    policy_version: str
    total_active_calls: int
    alerts: int
    critical: int
    decisions: list[DecisionItem]
    source_mode: str = "HISTORICAL_REPLAY_SHADOW"
    live: bool = False


class PolicyScore(BaseModel):
    policy_id: str
    role: str
    gt3_budget_pct: float
    gt6_budget_pct: float
    objective: float | None
    selected: bool
    metrics: dict[str, Any]


router = APIRouter(
    prefix="/api/v1/maritime/decision",
    tags=["B61C temporal decision shadow"],
)


def _resolved_decision_at(
    requested: datetime | None, evaluation_role: str
) -> datetime:
    requested = _normalize_time(requested)
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT MAX(decision_at)
            FROM {DECISION_TABLE}
            WHERE policy_version=%s
              AND evaluation_role=%s
              AND (%s::timestamptz IS NULL OR decision_at<=%s)
            """,
            (POLICY_VERSION, evaluation_role, requested, requested),
        )
        row = cursor.fetchone()
    if row is None or row[0] is None:
        raise HTTPException(status_code=404, detail="No B61C decision snapshot found")
    return row[0]


def _decision_items(
    decision_at: datetime,
    evaluation_role: str,
    alerts_only: bool,
    limit: int,
) -> list[DecisionItem]:
    alert_filter = "AND alert_active=true" if alerts_only else ""
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                port_call_id,
                landmark_at,
                decision_at,
                evaluation_role,
                regime,
                state,
                previous_state,
                state_changed,
                rank_in_bucket,
                active_calls,
                temporal_priority_score,
                critical_priority_score,
                p_delay_gt3,
                p_delay_gt6,
                p_gt3_breach_within_6h,
                p_gt3_breach_within_12h,
                p_gt3_breach_within_24h,
                remaining_p10_h,
                remaining_p50_h,
                remaining_p90_h,
                alert_active,
                new_alert,
                action_code,
                source_mode,
                production_claim_allowed,
                automatic_action_allowed
            FROM {DECISION_TABLE}
            WHERE policy_version=%s
              AND evaluation_role=%s
              AND decision_at=%s
              {alert_filter}
            ORDER BY
                CASE state
                    WHEN 'CRITICAL' THEN 1
                    WHEN 'HIGH_RISK' THEN 2
                    WHEN 'WATCH' THEN 3
                    ELSE 4
                END,
                rank_in_bucket
            LIMIT %s
            """,
            (POLICY_VERSION, evaluation_role, decision_at, limit),
        )
        rows = cursor.fetchall()
    return [_row_to_decision_item(row) for row in rows]


def _row_to_decision_item(row: tuple[Any, ...]) -> DecisionItem:
    return DecisionItem(
        port_call_id=row[0],
        landmark_at=row[1],
        decision_at=row[2],
        evaluation_role=row[3],
        regime=row[4],
        state=row[5],
        previous_state=row[6],
        state_changed=bool(row[7]),
        rank_in_bucket=int(row[8]),
        active_calls=int(row[9]),
        temporal_priority_score=float(row[10]),
        critical_priority_score=float(row[11]),
        p_delay_gt3=float(row[12]),
        p_delay_gt6=float(row[13]),
        p_breach_6h=float(row[14]),
        p_breach_12h=float(row[15]),
        p_breach_24h=float(row[16]),
        remaining_p10_h=float(row[17]),
        remaining_p50_h=float(row[18]),
        remaining_p90_h=float(row[19]),
        alert_active=bool(row[20]),
        new_alert=bool(row[21]),
        action_code=row[22],
        source_mode=row[23],
        production_claim_allowed=bool(row[24]),
        automatic_action_allowed=bool(row[25]),
    )


@router.get("/status", response_model=DecisionStatus)
def status() -> DecisionStatus:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT status, finished_at, metadata
            FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (AUDIT_SOURCE, AUDIT_DATASET),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="B61C audit status unavailable")
    audit_status, finished_at, metadata = row
    metadata = dict(metadata or {})
    return DecisionStatus(
        audit_status=audit_status,
        decision=str(metadata.get("decision", "UNKNOWN")),
        policy_version=str(metadata.get("policy_version", POLICY_VERSION)),
        selected_policy_id=metadata.get("selected_policy_id"),
        temporal_model=metadata.get("temporal_model"),
        shadow_api_allowed=bool(metadata.get("shadow_api_allowed", False)),
        dynamic_alert_shadow_allowed=bool(
            metadata.get("dynamic_alert_shadow_allowed", False)
        ),
        production_promotion_allowed=False,
        automatic_action_allowed=False,
        fresh_forward_confirmation_required=True,
        finished_at=finished_at,
    )


@router.get("/policy", response_model=SelectedPolicy)
def policy() -> SelectedPolicy:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                policy_version,
                source_model_version,
                policy_id,
                selected_on_role,
                parameters,
                quality_gates,
                source_mode,
                production_claim_allowed,
                automatic_action_allowed,
                fresh_forward_confirmation_required,
                created_at
            FROM {POLICY_TABLE}
            WHERE policy_version=%s
            """,
            (POLICY_VERSION,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B61C selected policy unavailable")
    return SelectedPolicy(
        policy_version=row[0],
        source_model_version=row[1],
        policy_id=row[2],
        selected_on_role=row[3],
        parameters=dict(row[4]),
        quality_gates=list(row[5]),
        source_mode=row[6],
        production_claim_allowed=bool(row[7]),
        automatic_action_allowed=bool(row[8]),
        fresh_forward_confirmation_required=bool(row[9]),
        created_at=row[10],
    )


@router.get("/snapshot", response_model=DecisionSnapshot)
def snapshot(
    at: datetime | None = Query(default=None),
    evaluation_role: str = Query(default="TEST_DIAGNOSTIC_ONLY"),
    alerts_only: bool = Query(default=False),
    limit: int = Query(default=250, ge=1, le=1_000),
) -> DecisionSnapshot:
    allowed_roles = {"VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"}
    if evaluation_role not in allowed_roles:
        raise HTTPException(status_code=422, detail="Unsupported evaluation_role")
    resolved = _resolved_decision_at(at, evaluation_role)
    decisions = _decision_items(resolved, evaluation_role, alerts_only, limit)
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT COUNT(*), COUNT(*) FILTER (WHERE alert_active),
                   COUNT(*) FILTER (WHERE state='CRITICAL')
            FROM {DECISION_TABLE}
            WHERE policy_version=%s AND evaluation_role=%s AND decision_at=%s
            """,
            (POLICY_VERSION, evaluation_role, resolved),
        )
        counts = cursor.fetchone()
    return DecisionSnapshot(
        requested_at=_normalize_time(at),
        resolved_at=resolved,
        evaluation_role=evaluation_role,
        policy_version=POLICY_VERSION,
        total_active_calls=int(counts[0]),
        alerts=int(counts[1]),
        critical=int(counts[2]),
        decisions=decisions,
    )


@router.get("/alerts", response_model=DecisionSnapshot)
def alerts(
    at: datetime | None = Query(default=None),
    evaluation_role: str = Query(default="TEST_DIAGNOSTIC_ONLY"),
    limit: int = Query(default=100, ge=1, le=500),
) -> DecisionSnapshot:
    return snapshot(at=at, evaluation_role=evaluation_role, alerts_only=True, limit=limit)


@router.get("/port-calls/{port_call_id}/timeline", response_model=list[DecisionItem])
def port_call_timeline(
    port_call_id: str,
    limit: int = Query(default=250, ge=1, le=1_000),
) -> list[DecisionItem]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                port_call_id, landmark_at, decision_at, evaluation_role, regime,
                state, previous_state, state_changed, rank_in_bucket, active_calls,
                temporal_priority_score, critical_priority_score,
                p_delay_gt3, p_delay_gt6,
                p_gt3_breach_within_6h, p_gt3_breach_within_12h,
                p_gt3_breach_within_24h,
                remaining_p10_h, remaining_p50_h, remaining_p90_h,
                alert_active, new_alert, action_code, source_mode,
                production_claim_allowed, automatic_action_allowed
            FROM {DECISION_TABLE}
            WHERE policy_version=%s AND port_call_id=%s
            ORDER BY decision_at DESC
            LIMIT %s
            """,
            (POLICY_VERSION, port_call_id, limit),
        )
        rows = cursor.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="Port-call decision timeline not found")
    return [_row_to_decision_item(row) for row in reversed(rows)]


@router.get("/scorecard", response_model=list[PolicyScore])
def scorecard(
    role: str = Query(default="VALID_CALIBRATE"),
) -> list[PolicyScore]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT policy_id, role, gt3_budget_pct, gt6_budget_pct,
                   objective, selected, metrics
            FROM {SCORECARD_TABLE}
            WHERE policy_version=%s AND role=%s
            ORDER BY selected DESC, objective DESC NULLS LAST
            """,
            (POLICY_VERSION, role),
        )
        rows = cursor.fetchall()
    return [
        PolicyScore(
            policy_id=row[0],
            role=row[1],
            gt3_budget_pct=float(row[2]),
            gt6_budget_pct=float(row[3]),
            objective=float(row[4]) if row[4] is not None else None,
            selected=bool(row[5]),
            metrics=dict(row[6]),
        )
        for row in rows
    ]
