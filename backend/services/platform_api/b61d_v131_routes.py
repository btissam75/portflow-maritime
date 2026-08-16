from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel


POLICY_VERSION = "b61d-v1.3.1-contract-recalibration-v1"
AUDIT_SOURCE = "b61d_v131_contract_recalibration"
DECISION_TABLE = "serving.maritime_port_call_dual_stage_shadow_v131"
SCORECARD_TABLE = "serving.maritime_dual_stage_contract_scorecard_v131"
ROLE_SCORECARD_TABLE = "serving.maritime_dual_stage_role_scorecard_v131"
MODEL_CARD_TABLE = "serving.maritime_dual_stage_model_card_v131"


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
            1, int(os.getenv("B61D_V131_DB_POOL_SIZE", "4")), dsn=dsn
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


class ContractRecalibrationStatus(BaseModel):
    audit_status: str
    decision: str
    policy_version: str
    selected_candidate_id: str | None = None
    early_mode: str | None = None
    early_top_k: int | None = None
    critical_top_k: int | None = None
    robust_contracts_passed: bool
    early_stage_validated: bool
    critical_stage_validated: bool
    fresh_forward_allowed: bool
    shadow_api_allowed: bool
    next_block: str | None = None
    finished_at: datetime | None = None


class DualStageDecision(BaseModel):
    port_call_id: str
    landmark_at: datetime
    decision_at: datetime
    evaluation_role: str
    hsmm_state: str
    hsmm_state_confidence: float
    early_warning_score: float
    critical_action_score: float
    early_candidate: bool
    critical_candidate: bool
    decision_state: str
    previous_state: str
    early_rank_in_bucket: int
    critical_rank_in_bucket: int
    early_warning: bool
    critical_action: bool
    new_alert: bool
    action_code: str
    p_delay_gt3: float
    p_delay_gt6: float
    hazard_6h: float
    hazard_12h: float
    hazard_24h: float
    production_claim_allowed: bool
    automatic_action_allowed: bool


class DualStageSnapshot(BaseModel):
    requested_at: datetime | None
    resolved_at: datetime
    evaluation_role: str
    total_calls: int
    early_warnings: int
    critical_actions: int
    watch_calls: int
    hidden_state_counts: dict[str, int]
    decisions: list[DualStageDecision]


router = APIRouter(
    prefix="/api/v1/maritime/dual-stage-contracts",
    tags=["Maritime dual-stage contract recalibration"],
)


@router.get("/status", response_model=ContractRecalibrationStatus)
def status() -> ContractRecalibrationStatus:
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
        raise HTTPException(status_code=404, detail="B61D-v1.3.1 audit unavailable")
    metadata = dict(row[1])
    return ContractRecalibrationStatus(
        audit_status=row[0],
        decision=str(metadata.get("decision", "UNKNOWN")),
        policy_version=str(metadata.get("policy_version", POLICY_VERSION)),
        selected_candidate_id=metadata.get("selected_candidate_id"),
        early_mode=metadata.get("early_mode"),
        early_top_k=metadata.get("early_top_k"),
        critical_top_k=metadata.get("critical_top_k"),
        robust_contracts_passed=bool(metadata.get("robust_contracts_passed", False)),
        early_stage_validated=bool(metadata.get("early_stage_validated", False)),
        critical_stage_validated=bool(metadata.get("critical_stage_validated", False)),
        fresh_forward_allowed=bool(metadata.get("fresh_forward_allowed", False)),
        shadow_api_allowed=bool(metadata.get("shadow_api_allowed", False)),
        next_block=metadata.get("next_block"),
        finished_at=row[2],
    )


@router.get("/model-card")
def model_card() -> dict[str, Any]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT policy_version,source_hsmm_version,source_policy_version,
                   selected_candidate_id,source_dual_stage_version,
                   selected_policy,baseline_metrics,
                   bootstrap_intervals,model_card,quality_gates,shadow_api_allowed,
                   production_claim_allowed,automatic_action_allowed,
                   fresh_forward_confirmation_required,created_at
            FROM {MODEL_CARD_TABLE} WHERE policy_version=%s
            """,
            (POLICY_VERSION,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B61D-v1.3.1 model card unavailable")
    return {
        "policy_version": row[0], "source_hsmm_version": row[1],
        "source_policy_version": row[2], "selected_candidate_id": row[3],
        "source_dual_stage_version": row[4], "selected_policy": dict(row[5]),
        "baseline_metrics": dict(row[6]), "bootstrap_intervals": dict(row[7]),
        "model_card": dict(row[8]), "quality_gates": list(row[9]),
        "shadow_api_allowed": bool(row[10]),
        "production_claim_allowed": bool(row[11]),
        "automatic_action_allowed": bool(row[12]),
        "fresh_forward_confirmation_required": bool(row[13]),
        "created_at": row[14],
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
        raise HTTPException(status_code=404, detail="No dual-stage snapshot found")
    return row[0]


def _row_to_decision(row: tuple[Any, ...]) -> DualStageDecision:
    return DualStageDecision(
        port_call_id=row[0], landmark_at=row[1], decision_at=row[2],
        evaluation_role=row[3], hsmm_state=row[4],
        hsmm_state_confidence=float(row[5]), early_warning_score=float(row[6]),
        critical_action_score=float(row[7]), early_candidate=bool(row[8]),
        critical_candidate=bool(row[9]), decision_state=row[10],
        previous_state=row[11], early_rank_in_bucket=int(row[12]),
        critical_rank_in_bucket=int(row[13]), early_warning=bool(row[14]),
        critical_action=bool(row[15]), new_alert=bool(row[16]), action_code=row[17],
        p_delay_gt3=float(row[18]), p_delay_gt6=float(row[19]),
        hazard_6h=float(row[20]), hazard_12h=float(row[21]), hazard_24h=float(row[22]),
        production_claim_allowed=bool(row[23]), automatic_action_allowed=bool(row[24]),
    )


def _decisions(where: str, parameters: tuple[Any, ...], limit: int) -> list[DualStageDecision]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT port_call_id,landmark_at,decision_at,evaluation_role,hsmm_state,
                   hsmm_state_confidence,early_warning_score,critical_action_score,
                   early_candidate,critical_candidate,state,previous_state,
                   early_rank_in_bucket,critical_rank_in_bucket,early_warning,
                   critical_action,new_alert,action_code,p_delay_gt3,p_delay_gt6,
                   p_gt3_breach_within_6h,p_gt3_breach_within_12h,
                   p_gt3_breach_within_24h,production_claim_allowed,
                   automatic_action_allowed
            FROM {DECISION_TABLE}
            WHERE policy_version=%s AND {where}
            ORDER BY decision_at DESC,critical_action DESC,early_warning DESC,
                     critical_rank_in_bucket,early_rank_in_bucket LIMIT %s
            """,
            (POLICY_VERSION, *parameters, limit),
        )
        rows = cursor.fetchall()
    return [_row_to_decision(row) for row in rows]


@router.get("/snapshot", response_model=DualStageSnapshot)
def snapshot(
    at: datetime | None = Query(default=None),
    evaluation_role: str = Query(default="TEST_DIAGNOSTIC_ONLY"),
    actions_only: bool = Query(default=False),
    limit: int = Query(default=250, ge=1, le=1_000),
) -> DualStageSnapshot:
    allowed = {"VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"}
    if evaluation_role not in allowed:
        raise HTTPException(status_code=422, detail="Unsupported evaluation_role")
    resolved = _resolved_at(at, evaluation_role)
    action_filter = " AND alert_active=true" if actions_only else ""
    decisions = _decisions(
        f"evaluation_role=%s AND decision_at=%s{action_filter}",
        (evaluation_role, resolved), limit,
    )
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT COUNT(*),COUNT(*) FILTER(WHERE early_warning),
                   COUNT(*) FILTER(WHERE critical_action),
                   COUNT(*) FILTER(WHERE state='WATCH'),hsmm_state
            FROM {DECISION_TABLE}
            WHERE policy_version=%s AND evaluation_role=%s AND decision_at=%s
            GROUP BY hsmm_state
            """,
            (POLICY_VERSION, evaluation_role, resolved),
        )
        rows = cursor.fetchall()
    return DualStageSnapshot(
        requested_at=_normalize_time(at), resolved_at=resolved,
        evaluation_role=evaluation_role,
        total_calls=sum(int(row[0]) for row in rows),
        early_warnings=sum(int(row[1]) for row in rows),
        critical_actions=sum(int(row[2]) for row in rows),
        watch_calls=sum(int(row[3]) for row in rows),
        hidden_state_counts={str(row[4]): int(row[0]) for row in rows},
        decisions=decisions,
    )


@router.get("/port-calls/{port_call_id}/timeline", response_model=list[DualStageDecision])
def timeline(
    port_call_id: str,
    limit: int = Query(default=250, ge=1, le=1_000),
) -> list[DualStageDecision]:
    rows = _decisions("port_call_id=%s", (port_call_id,), limit)
    if not rows:
        raise HTTPException(status_code=404, detail="Dual-stage timeline not found")
    return list(reversed(rows))


@router.get("/scorecard")
def scorecard() -> list[dict[str, Any]]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT candidate_id,early_mode,early_min_score,early_top_k,
                   critical_top_k,hold_windows,robust_objective,
                   robust_gates_passed,robust_gates_total,
                   passes_robust_contracts,selected,metrics
            FROM {SCORECARD_TABLE}
            WHERE policy_version=%s
            ORDER BY selected DESC,passes_robust_contracts DESC,robust_objective DESC
            """,
            (POLICY_VERSION,),
        )
        rows = cursor.fetchall()
    return [
        {
            "candidate_id": row[0], "early_mode": row[1],
            "early_min_score": row[2], "early_top_k": row[3],
            "critical_top_k": row[4], "hold_windows": row[5],
            "robust_objective": row[6], "robust_gates_passed": row[7],
            "robust_gates_total": row[8], "passes_robust_contracts": row[9],
            "selected": row[10], "metrics": dict(row[11]),
        }
        for row in rows
    ]


@router.get("/role-scorecard")
def role_scorecard(
    role: str = Query(default="VALID_CALIBRATE"),
) -> list[dict[str, Any]]:
    allowed = {"VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"}
    if role not in allowed:
        raise HTTPException(status_code=422, detail="Unsupported role")
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT candidate_id,role,selected,passes_role_contracts,
                   role_gates_passed,role_gates_total,metrics
            FROM {ROLE_SCORECARD_TABLE}
            WHERE policy_version=%s AND role=%s
            ORDER BY selected DESC,passes_role_contracts DESC,role_gates_passed DESC
            """,
            (POLICY_VERSION, role),
        )
        rows = cursor.fetchall()
    return [
        {
            "candidate_id": row[0], "role": row[1], "selected": row[2],
            "passes_role_contracts": row[3], "role_gates_passed": row[4],
            "role_gates_total": row[5], "metrics": dict(row[6]),
        }
        for row in rows
    ]
