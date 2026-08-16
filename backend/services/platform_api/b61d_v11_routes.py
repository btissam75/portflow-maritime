from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel


MODEL_VERSION = "b61d-v1.1-anchored-hsmm-v1"
AUDIT_SOURCE = "b61d_v11_anchored_hsmm"
DECISION_TABLE = "serving.maritime_port_call_anchored_hsmm_shadow_v11"
SCORECARD_TABLE = "serving.maritime_anchored_hsmm_policy_scorecard_v11"
MODEL_CARD_TABLE = "serving.maritime_anchored_hsmm_model_card_v11"


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
            1, int(os.getenv("B61D_V11_DB_POOL_SIZE", "4")), dsn=dsn
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


class AnchoredHSMMStatus(BaseModel):
    audit_status: str
    decision: str
    model_version: str
    control_candidate_id: str | None
    selected_candidate_id: str | None
    selected_hsmm_weight: float | None
    challenger_accepted: bool
    shadow_api_allowed: bool
    production_promotion_allowed: bool = False
    automatic_action_allowed: bool = False
    next_block: str | None
    finished_at: datetime | None


class AnchoredHSMMDecision(BaseModel):
    port_call_id: str
    landmark_at: datetime
    decision_at: datetime
    evaluation_role: str
    hsmm_state: str
    hsmm_state_confidence: float
    hsmm_risk_score: float
    hsmm_escalation_probability: float
    hsmm_dwell_steps: int
    decision_state: str
    previous_state: str
    rank_in_bucket: int
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


class AnchoredHSMMSnapshot(BaseModel):
    requested_at: datetime | None
    resolved_at: datetime
    evaluation_role: str
    total_calls: int
    alerts: int
    hidden_state_counts: dict[str, int]
    decisions: list[AnchoredHSMMDecision]
    source_mode: str = "HISTORICAL_REPLAY_SHADOW"
    live: bool = False


router = APIRouter(
    prefix="/api/v1/maritime/anchored-hsmm",
    tags=["B61D-v1.1 anchored HSMM shadow"],
)


@router.get("/status", response_model=AnchoredHSMMStatus)
def status() -> AnchoredHSMMStatus:
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
        raise HTTPException(status_code=503, detail="B61D-v1.1 status unavailable")
    metadata = dict(row[2] or {})
    weight = metadata.get("selected_hsmm_weight")
    return AnchoredHSMMStatus(
        audit_status=row[0],
        decision=str(metadata.get("decision", "UNKNOWN")),
        model_version=str(metadata.get("model_version", MODEL_VERSION)),
        control_candidate_id=metadata.get("control_candidate_id"),
        selected_candidate_id=metadata.get("selected_candidate_id"),
        selected_hsmm_weight=float(weight) if weight is not None else None,
        challenger_accepted=bool(metadata.get("challenger_accepted", False)),
        shadow_api_allowed=bool(metadata.get("shadow_api_allowed", False)),
        next_block=metadata.get("next_block"),
        finished_at=row[1],
    )


@router.get("/model-card")
def model_card() -> dict[str, Any]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT model_version,source_model_version,source_policy_version,
                   source_hsmm_version,selected_candidate_id,control_candidate_id,
                   model_card,quality_gates,production_claim_allowed,
                   automatic_action_allowed,fresh_forward_confirmation_required,
                   created_at
            FROM {MODEL_CARD_TABLE} WHERE model_version=%s
            """,
            (MODEL_VERSION,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B61D-v1.1 model card unavailable")
    return {
        "model_version": row[0], "source_model_version": row[1],
        "source_policy_version": row[2], "source_hsmm_version": row[3],
        "selected_candidate_id": row[4], "control_candidate_id": row[5],
        "model_card": dict(row[6]), "quality_gates": list(row[7]),
        "production_claim_allowed": bool(row[8]),
        "automatic_action_allowed": bool(row[9]),
        "fresh_forward_confirmation_required": bool(row[10]),
        "created_at": row[11],
    }


def _resolved_at(requested: datetime | None, role: str) -> datetime:
    requested = _normalize_time(requested)
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT MAX(decision_at) FROM {DECISION_TABLE}
            WHERE model_version=%s AND evaluation_role=%s
              AND (%s::timestamptz IS NULL OR decision_at<=%s)
            """,
            (MODEL_VERSION, role, requested, requested),
        )
        row = cursor.fetchone()
    if row is None or row[0] is None:
        raise HTTPException(status_code=404, detail="No anchored HSMM snapshot found")
    return row[0]


def _row_to_decision(row: tuple[Any, ...]) -> AnchoredHSMMDecision:
    return AnchoredHSMMDecision(
        port_call_id=row[0], landmark_at=row[1], decision_at=row[2],
        evaluation_role=row[3], hsmm_state=row[4],
        hsmm_state_confidence=float(row[5]), hsmm_risk_score=float(row[6]),
        hsmm_escalation_probability=float(row[7]), hsmm_dwell_steps=int(row[8]),
        decision_state=row[9], previous_state=row[10], rank_in_bucket=int(row[11]),
        alert_active=bool(row[12]), new_alert=bool(row[13]), action_code=row[14],
        p_delay_gt3=float(row[15]), p_delay_gt6=float(row[16]),
        hazard_6h=float(row[17]), hazard_12h=float(row[18]), hazard_24h=float(row[19]),
        production_claim_allowed=bool(row[20]), automatic_action_allowed=bool(row[21]),
    )


def _decisions(where: str, parameters: tuple[Any, ...], limit: int) -> list[AnchoredHSMMDecision]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT port_call_id,landmark_at,decision_at,evaluation_role,
                   hsmm_state,hsmm_state_confidence,hsmm_risk_score,
                   hsmm_escalation_probability,hsmm_dwell_steps,state,
                   previous_state,rank_in_bucket,alert_active,new_alert,action_code,
                   p_delay_gt3,p_delay_gt6,p_gt3_breach_within_6h,
                   p_gt3_breach_within_12h,p_gt3_breach_within_24h,
                   production_claim_allowed,automatic_action_allowed
            FROM {DECISION_TABLE}
            WHERE model_version=%s AND {where}
            ORDER BY decision_at DESC,rank_in_bucket LIMIT %s
            """,
            (MODEL_VERSION, *parameters, limit),
        )
        rows = cursor.fetchall()
    return [_row_to_decision(row) for row in rows]


@router.get("/snapshot", response_model=AnchoredHSMMSnapshot)
def snapshot(
    at: datetime | None = Query(default=None),
    evaluation_role: str = Query(default="TEST_DIAGNOSTIC_ONLY"),
    alerts_only: bool = Query(default=False),
    limit: int = Query(default=250, ge=1, le=1_000),
) -> AnchoredHSMMSnapshot:
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
            SELECT COUNT(*),COUNT(*) FILTER(WHERE alert_active),hsmm_state
            FROM {DECISION_TABLE}
            WHERE model_version=%s AND evaluation_role=%s AND decision_at=%s
            GROUP BY hsmm_state
            """,
            (MODEL_VERSION, evaluation_role, resolved),
        )
        rows = cursor.fetchall()
    return AnchoredHSMMSnapshot(
        requested_at=_normalize_time(at), resolved_at=resolved,
        evaluation_role=evaluation_role,
        total_calls=sum(int(row[0]) for row in rows),
        alerts=sum(int(row[1]) for row in rows),
        hidden_state_counts={str(row[2]): int(row[0]) for row in rows},
        decisions=decisions,
    )


@router.get("/port-calls/{port_call_id}/timeline", response_model=list[AnchoredHSMMDecision])
def timeline(
    port_call_id: str,
    limit: int = Query(default=250, ge=1, le=1_000),
) -> list[AnchoredHSMMDecision]:
    rows = _decisions("port_call_id=%s", (port_call_id,), limit)
    if not rows:
        raise HTTPException(status_code=404, detail="Anchored HSMM timeline not found")
    return list(reversed(rows))


@router.get("/scorecard")
def scorecard(role: str = Query(default="VALID_CALIBRATE")) -> list[dict[str, Any]]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT candidate_id,role,hsmm_weight,gt3_budget_pct,gt6_budget_pct,
                   objective,control_selected,challenger_selected,metrics
            FROM {SCORECARD_TABLE}
            WHERE model_version=%s AND role=%s
            ORDER BY challenger_selected DESC,control_selected DESC,objective DESC
            """,
            (MODEL_VERSION, role),
        )
        rows = cursor.fetchall()
    return [
        {
            "candidate_id": row[0], "role": row[1], "hsmm_weight": row[2],
            "gt3_budget_pct": row[3], "gt6_budget_pct": row[4],
            "objective": row[5], "control_selected": row[6],
            "challenger_selected": row[7], "metrics": dict(row[8]),
        }
        for row in rows
    ]
