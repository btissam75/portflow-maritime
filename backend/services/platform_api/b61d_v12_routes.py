from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel


POLICY_VERSION = "b61d-v1.2-state-conditional-policy-v1"
AUDIT_SOURCE = "b61d_v12_state_conditional_policy"
DECISION_TABLE = "serving.maritime_port_call_state_policy_shadow_v12"
SCORECARD_TABLE = "serving.maritime_state_policy_scorecard_v12"
MODEL_CARD_TABLE = "serving.maritime_state_policy_model_card_v12"


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
        _pool = SimpleConnectionPool(
            1, int(os.getenv("B61D_V12_DB_POOL_SIZE", "4")), dsn=dsn
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


class StatePolicyStatus(BaseModel):
    audit_status: str
    decision: str
    policy_version: str
    selected_candidate_id: str | None
    policy_mode: str | None
    alert_budget_pct: float | None
    hold_windows: int | None
    constraints_passed: bool
    shadow_api_allowed: bool
    production_promotion_allowed: bool = False
    automatic_action_allowed: bool = False
    next_block: str | None
    finished_at: datetime | None


class StatePolicyDecision(BaseModel):
    port_call_id: str
    landmark_at: datetime
    decision_at: datetime
    evaluation_role: str
    hsmm_state: str
    hsmm_state_confidence: float
    critical_state_probability: float
    state_priority_score: float
    policy_candidate: bool
    candidate_reason: str
    decision_state: str
    previous_state: str
    rank_in_bucket: int
    alert_capacity: int
    alert_active: bool
    new_alert: bool
    action_code: str
    p_delay_gt3: float
    p_delay_gt6: float
    hazard_6h: float
    hazard_12h: float
    hazard_24h: float
    production_claim_allowed: bool
    automatic_action_allowed: bool


class StatePolicySnapshot(BaseModel):
    requested_at: datetime | None
    resolved_at: datetime
    evaluation_role: str
    total_calls: int
    alerts: int
    watch_calls: int
    hidden_state_counts: dict[str, int]
    decisions: list[StatePolicyDecision]
    source_mode: str = "HISTORICAL_REPLAY_SHADOW"
    live: bool = False


router = APIRouter(
    prefix="/api/v1/maritime/state-policy",
    tags=["B61D-v1.2 state-conditional shadow policy"],
)


@router.get("/status", response_model=StatePolicyStatus)
def status() -> StatePolicyStatus:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT status,finished_at,metadata FROM audit.ingestion_run
            WHERE source_name=%s ORDER BY started_at DESC LIMIT 1
            """,
            (AUDIT_SOURCE,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=503, detail="B61D-v1.2 status unavailable")
    metadata = dict(row[2] or {})
    budget = metadata.get("alert_budget_pct")
    hold = metadata.get("hold_windows")
    return StatePolicyStatus(
        audit_status=row[0],
        decision=str(metadata.get("decision", "UNKNOWN")),
        policy_version=str(metadata.get("policy_version", POLICY_VERSION)),
        selected_candidate_id=metadata.get("selected_candidate_id"),
        policy_mode=metadata.get("policy_mode"),
        alert_budget_pct=float(budget) if budget is not None else None,
        hold_windows=int(hold) if hold is not None else None,
        constraints_passed=bool(metadata.get("constraints_passed", False)),
        shadow_api_allowed=bool(metadata.get("shadow_api_allowed", False)),
        next_block=metadata.get("next_block"),
        finished_at=row[1],
    )


@router.get("/model-card")
def model_card() -> dict[str, Any]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT policy_version,source_hsmm_version,source_policy_version,
                   selected_candidate_id,selected_policy,baseline_metrics,
                   bootstrap_intervals,model_card,quality_gates,shadow_api_allowed,
                   production_claim_allowed,automatic_action_allowed,
                   fresh_forward_confirmation_required,created_at
            FROM {MODEL_CARD_TABLE} WHERE policy_version=%s
            """,
            (POLICY_VERSION,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B61D-v1.2 model card unavailable")
    return {
        "policy_version": row[0], "source_hsmm_version": row[1],
        "source_policy_version": row[2], "selected_candidate_id": row[3],
        "selected_policy": dict(row[4]), "baseline_metrics": dict(row[5]),
        "bootstrap_intervals": dict(row[6]), "model_card": dict(row[7]),
        "quality_gates": list(row[8]), "shadow_api_allowed": bool(row[9]),
        "production_claim_allowed": bool(row[10]),
        "automatic_action_allowed": bool(row[11]),
        "fresh_forward_confirmation_required": bool(row[12]),
        "created_at": row[13],
    }


def _resolved_at(requested: datetime | None, role: str) -> datetime:
    requested = _normalize_time(requested)
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT MAX(decision_at) FROM {DECISION_TABLE}
            WHERE policy_version=%s AND evaluation_role=%s
              AND (%s::timestamptz IS NULL OR decision_at<=%s)
            """,
            (POLICY_VERSION, role, requested, requested),
        )
        row = cursor.fetchone()
    if row is None or row[0] is None:
        raise HTTPException(status_code=404, detail="No state-policy snapshot found")
    return row[0]


def _row_to_decision(row: tuple[Any, ...]) -> StatePolicyDecision:
    return StatePolicyDecision(
        port_call_id=row[0], landmark_at=row[1], decision_at=row[2],
        evaluation_role=row[3], hsmm_state=row[4],
        hsmm_state_confidence=float(row[5]),
        critical_state_probability=float(row[6]),
        state_priority_score=float(row[7]), policy_candidate=bool(row[8]),
        candidate_reason=row[9], decision_state=row[10], previous_state=row[11],
        rank_in_bucket=int(row[12]), alert_capacity=int(row[13]),
        alert_active=bool(row[14]), new_alert=bool(row[15]), action_code=row[16],
        p_delay_gt3=float(row[17]), p_delay_gt6=float(row[18]),
        hazard_6h=float(row[19]), hazard_12h=float(row[20]), hazard_24h=float(row[21]),
        production_claim_allowed=bool(row[22]), automatic_action_allowed=bool(row[23]),
    )


def _decisions(
    where: str, parameters: tuple[Any, ...], limit: int
) -> list[StatePolicyDecision]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT port_call_id,landmark_at,decision_at,evaluation_role,
                   hsmm_state,hsmm_state_confidence,p_state_critical_disruption,
                   state_priority_score,policy_candidate,candidate_reason,state,
                   previous_state,rank_in_bucket,alert_capacity,alert_active,
                   new_alert,action_code,p_delay_gt3,p_delay_gt6,
                   p_gt3_breach_within_6h,p_gt3_breach_within_12h,
                   p_gt3_breach_within_24h,production_claim_allowed,
                   automatic_action_allowed
            FROM {DECISION_TABLE}
            WHERE policy_version=%s AND {where}
            ORDER BY decision_at DESC,rank_in_bucket LIMIT %s
            """,
            (POLICY_VERSION, *parameters, limit),
        )
        rows = cursor.fetchall()
    return [_row_to_decision(row) for row in rows]


@router.get("/snapshot", response_model=StatePolicySnapshot)
def snapshot(
    at: datetime | None = Query(default=None),
    evaluation_role: str = Query(default="TEST_DIAGNOSTIC_ONLY"),
    alerts_only: bool = Query(default=False),
    limit: int = Query(default=250, ge=1, le=1_000),
) -> StatePolicySnapshot:
    allowed = {"VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"}
    if evaluation_role not in allowed:
        raise HTTPException(status_code=422, detail="Unsupported evaluation_role")
    resolved = _resolved_at(at, evaluation_role)
    alert_filter = " AND alert_active=true" if alerts_only else ""
    decisions = _decisions(
        f"evaluation_role=%s AND decision_at=%s{alert_filter}",
        (evaluation_role, resolved), limit,
    )
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT COUNT(*),COUNT(*) FILTER(WHERE alert_active),
                   COUNT(*) FILTER(WHERE state='WATCH'),hsmm_state
            FROM {DECISION_TABLE}
            WHERE policy_version=%s AND evaluation_role=%s AND decision_at=%s
            GROUP BY hsmm_state
            """,
            (POLICY_VERSION, evaluation_role, resolved),
        )
        rows = cursor.fetchall()
    return StatePolicySnapshot(
        requested_at=_normalize_time(at), resolved_at=resolved,
        evaluation_role=evaluation_role,
        total_calls=sum(int(row[0]) for row in rows),
        alerts=sum(int(row[1]) for row in rows),
        watch_calls=sum(int(row[2]) for row in rows),
        hidden_state_counts={str(row[3]): int(row[0]) for row in rows},
        decisions=decisions,
    )


@router.get("/port-calls/{port_call_id}/timeline", response_model=list[StatePolicyDecision])
def timeline(
    port_call_id: str,
    limit: int = Query(default=250, ge=1, le=1_000),
) -> list[StatePolicyDecision]:
    rows = _decisions("port_call_id=%s", (port_call_id,), limit)
    if not rows:
        raise HTTPException(status_code=404, detail="State-policy timeline not found")
    return list(reversed(rows))


@router.get("/scorecard")
def scorecard(role: str = Query(default="VALID_CALIBRATE")) -> list[dict[str, Any]]:
    allowed = {"VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"}
    if role not in allowed:
        raise HTTPException(status_code=422, detail="Unsupported role")
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT candidate_id,role,policy_mode,critical_probability_threshold,
                   alert_budget_pct,hold_windows,objective,passes_constraints,
                   selected,metrics
            FROM {SCORECARD_TABLE}
            WHERE policy_version=%s AND role=%s
            ORDER BY selected DESC,passes_constraints DESC,objective DESC
            """,
            (POLICY_VERSION, role),
        )
        rows = cursor.fetchall()
    return [
        {
            "candidate_id": row[0], "role": row[1], "policy_mode": row[2],
            "critical_probability_threshold": row[3], "alert_budget_pct": row[4],
            "hold_windows": row[5], "objective": row[6],
            "passes_constraints": row[7], "selected": row[8],
            "metrics": dict(row[9]),
        }
        for row in rows
    ]
