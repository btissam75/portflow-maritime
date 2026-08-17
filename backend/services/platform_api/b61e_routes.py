from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from fastapi import APIRouter, HTTPException, Query
from psycopg2.pool import SimpleConnectionPool
from pydantic import BaseModel

from platform_api import local_demo


POLICY_VERSION = "b61e-capacity-aware-temporal-ranking-v1"
AUDIT_SOURCE = "b61e_capacity_aware_temporal_ranking"
DECISION_TABLE = "serving.maritime_capacity_watchlist_shadow_v1"
SCORECARD_TABLE = "serving.maritime_capacity_ranking_scorecard_v1"
MODEL_CARD_TABLE = "serving.maritime_capacity_ranking_model_card_v1"
ALLOWED_ROLES = {"VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"}


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
            1, int(os.getenv("B61E_DB_POOL_SIZE", "4")), dsn=dsn
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


def _validate_role(role: str) -> str:
    if role not in ALLOWED_ROLES:
        raise HTTPException(status_code=422, detail="Unsupported evaluation_role")
    return role


class RankingStatus(BaseModel):
    audit_status: str
    decision: str
    policy_version: str
    selected_candidate_id: str | None = None
    selected_score: str | None = None
    selected_top_k: int | None = None
    bucket_hours: int | None = None
    contracts_passed: bool
    integrity_passed: bool
    shadow_api_allowed: bool
    production_promotion_allowed: bool
    automatic_action_allowed: bool
    serving_rows: int
    next_block: str | None = None
    finished_at: datetime | None = None


class WatchlistDecision(BaseModel):
    port_call_id: str
    vessel_name: str | None = None
    port_code: str | None = None
    terminal_code: str | None = None
    vessel_type: str | None = None
    cargo_group: str | None = None
    landmark_at: datetime
    decision_at: datetime
    evaluation_role: str
    risk_score: float
    rank_in_window: int
    active_calls: int
    capacity: int
    watchlist_selected: bool
    action_tier: str
    reason_code: str
    p_delay_gt3: float
    hazard_6h: float
    hazard_12h: float
    hazard_24h: float
    remaining_p10_h: float
    remaining_p50_h: float
    remaining_p90_h: float
    hsmm_state: str | None = None
    hsmm_state_confidence: float | None = None
    production_claim_allowed: bool
    automatic_action_allowed: bool


class WatchlistSnapshot(BaseModel):
    requested_at: datetime | None
    resolved_at: datetime
    evaluation_role: str
    active_calls: int
    capacity: int
    selected_calls: int
    decisions: list[WatchlistDecision]


router = APIRouter(
    prefix="/api/v1/maritime/capacity-ranking",
    tags=["Maritime capacity-aware temporal ranking"],
)


@router.get("/status", response_model=RankingStatus)
def status() -> RankingStatus:
    if local_demo.enabled():
        resolved = local_demo.capacity_snapshot_times()[-1]
        return RankingStatus(
            audit_status="DEMO",
            decision="LOCAL_DEMO_SHADOW_REPLAY",
            policy_version=local_demo.DEMO_POLICY_VERSION,
            selected_candidate_id="DEMO_CAPACITY_RANKER",
            selected_score="TEMPORAL_RISK_DEMO",
            selected_top_k=6,
            bucket_hours=6,
            contracts_passed=True,
            integrity_passed=True,
            shadow_api_allowed=True,
            production_promotion_allowed=False,
            automatic_action_allowed=False,
            serving_rows=len(local_demo.VESSELS) * len(local_demo.capacity_snapshot_times()),
            next_block="CONNECT_REAL_B61E_MATERIALIZATION",
            finished_at=resolved,
        )
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT status,row_count,metadata,finished_at FROM audit.ingestion_run
            WHERE source_name=%s ORDER BY started_at DESC LIMIT 1
            """,
            (AUDIT_SOURCE,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B61E audit unavailable")
    metadata = dict(row[2])
    return RankingStatus(
        audit_status=str(row[0]),
        decision=str(metadata.get("decision", "UNKNOWN")),
        policy_version=str(metadata.get("policy_version", POLICY_VERSION)),
        selected_candidate_id=metadata.get("selected_candidate_id"),
        selected_score=metadata.get("selected_score"),
        selected_top_k=metadata.get("selected_top_k"),
        bucket_hours=metadata.get("bucket_hours"),
        contracts_passed=bool(metadata.get("contracts_passed", False)),
        integrity_passed=bool(metadata.get("integrity_passed", False)),
        shadow_api_allowed=bool(metadata.get("shadow_api_allowed", False)),
        production_promotion_allowed=bool(metadata.get("production_promotion_allowed", False)),
        automatic_action_allowed=bool(metadata.get("automatic_action_allowed", False)),
        serving_rows=int(row[1] or metadata.get("serving_rows", 0)),
        next_block=metadata.get("next_block"),
        finished_at=row[3],
    )


def _resolved_at(requested: datetime | None, role: str) -> datetime:
    requested = _normalize_time(requested)
    if local_demo.enabled():
        resolved = local_demo.resolve_capacity_time(requested)
        if resolved is None:
            raise HTTPException(status_code=404, detail="No demo ranking snapshot found")
        return resolved
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
        raise HTTPException(status_code=404, detail="No B61E ranking snapshot found")
    return row[0]


def _row_to_decision(row: tuple[Any, ...]) -> WatchlistDecision:
    return WatchlistDecision(
        port_call_id=row[0], vessel_name=row[1], port_code=row[2], terminal_code=row[3],
        vessel_type=row[4], cargo_group=row[5], landmark_at=row[6], decision_at=row[7],
        evaluation_role=row[8], risk_score=float(row[9]), rank_in_window=int(row[10]),
        active_calls=int(row[11]), capacity=int(row[12]), watchlist_selected=bool(row[13]),
        action_tier=row[14], reason_code=row[15], p_delay_gt3=float(row[16]),
        hazard_6h=float(row[17]), hazard_12h=float(row[18]), hazard_24h=float(row[19]),
        remaining_p10_h=float(row[20]), remaining_p50_h=float(row[21]),
        remaining_p90_h=float(row[22]), hsmm_state=row[23],
        hsmm_state_confidence=float(row[24]) if row[24] is not None else None,
        production_claim_allowed=bool(row[25]), automatic_action_allowed=bool(row[26]),
    )


def _decisions(where: str, parameters: tuple[Any, ...], limit: int) -> list[WatchlistDecision]:
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT port_call_id,vessel_name,port_code,terminal_code,vessel_type,cargo_group,
                   landmark_at,decision_at,evaluation_role,risk_score,rank_in_window,
                   active_calls,capacity,watchlist_selected,action_tier,reason_code,
                   p_delay_gt3,p_gt3_breach_within_6h,p_gt3_breach_within_12h,
                   p_gt3_breach_within_24h,remaining_p10_h,remaining_p50_h,
                   remaining_p90_h,hsmm_state,hsmm_state_confidence,
                   production_claim_allowed,automatic_action_allowed
            FROM {DECISION_TABLE}
            WHERE policy_version=%s AND {where}
            ORDER BY decision_at DESC,watchlist_selected DESC,rank_in_window LIMIT %s
            """,
            (POLICY_VERSION, *parameters, limit),
        )
        rows = cursor.fetchall()
    return [_row_to_decision(row) for row in rows]


@router.get("/snapshot", response_model=WatchlistSnapshot)
def snapshot(
    at: datetime | None = Query(default=None),
    evaluation_role: str = Query(default="TEST_DIAGNOSTIC_ONLY"),
    selected_only: bool = Query(default=False),
    limit: int = Query(default=250, ge=1, le=1_000),
) -> WatchlistSnapshot:
    evaluation_role = _validate_role(evaluation_role)
    resolved = _resolved_at(at, evaluation_role)
    if local_demo.enabled():
        all_decisions = [
            WatchlistDecision(**row)
            for row in local_demo.capacity_decisions(resolved, evaluation_role)
        ]
        decisions = [row for row in all_decisions if row.watchlist_selected] if selected_only else all_decisions
        return WatchlistSnapshot(
            requested_at=_normalize_time(at),
            resolved_at=resolved,
            evaluation_role=evaluation_role,
            active_calls=len(all_decisions),
            capacity=6,
            selected_calls=sum(row.watchlist_selected for row in all_decisions),
            decisions=decisions[:limit],
        )
    selection_filter = " AND watchlist_selected=true" if selected_only else ""
    decisions = _decisions(
        f"evaluation_role=%s AND decision_at=%s{selection_filter}",
        (evaluation_role, resolved),
        limit,
    )
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT COUNT(*),MAX(capacity),COUNT(*) FILTER(WHERE watchlist_selected)
            FROM {DECISION_TABLE}
            WHERE policy_version=%s AND evaluation_role=%s AND decision_at=%s
            """,
            (POLICY_VERSION, evaluation_role, resolved),
        )
        counts = cursor.fetchone()
    return WatchlistSnapshot(
        requested_at=_normalize_time(at), resolved_at=resolved,
        evaluation_role=evaluation_role, active_calls=int(counts[0]),
        capacity=int(counts[1] or 0), selected_calls=int(counts[2]), decisions=decisions,
    )


@router.get("/watchlist", response_model=list[WatchlistDecision])
def watchlist(
    at: datetime | None = Query(default=None),
    evaluation_role: str = Query(default="TEST_DIAGNOSTIC_ONLY"),
) -> list[WatchlistDecision]:
    evaluation_role = _validate_role(evaluation_role)
    resolved = _resolved_at(at, evaluation_role)
    if local_demo.enabled():
        return [
            WatchlistDecision(**row)
            for row in local_demo.capacity_decisions(resolved, evaluation_role)
            if row["watchlist_selected"]
        ]
    return _decisions(
        "evaluation_role=%s AND decision_at=%s AND watchlist_selected=true",
        (evaluation_role, resolved),
        100,
    )


@router.get("/port-calls/{port_call_id}/timeline", response_model=list[WatchlistDecision])
def timeline(
    port_call_id: str,
    limit: int = Query(default=250, ge=1, le=1_000),
) -> list[WatchlistDecision]:
    if local_demo.enabled():
        rows = local_demo.capacity_timeline(port_call_id)[-limit:]
        if not rows:
            raise HTTPException(status_code=404, detail="Demo port-call timeline not found")
        return [WatchlistDecision(**row) for row in rows]
    rows = _decisions("port_call_id=%s", (port_call_id,), limit)
    if not rows:
        raise HTTPException(status_code=404, detail="B61E port-call timeline not found")
    return list(reversed(rows))


@router.get("/scorecard")
def scorecard() -> list[dict[str, Any]]:
    if local_demo.enabled():
        return [
            {
                "candidate_id": "DEMO_CAPACITY_RANKER",
                "stage": "LOCAL_DEMO",
                "role": "VALID_SELECT",
                "selected": True,
                "metrics": {"scientific_claim_allowed": False},
            }
        ]
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT candidate_id,stage,role,selected,metrics
            FROM {SCORECARD_TABLE} WHERE policy_version=%s
            ORDER BY stage,role,selected DESC,candidate_id
            """,
            (POLICY_VERSION,),
        )
        rows = cursor.fetchall()
    return [
        {
            "candidate_id": row[0], "stage": row[1], "role": row[2],
            "selected": bool(row[3]), "metrics": dict(row[4]),
        }
        for row in rows
    ]


@router.get("/model-card")
def model_card() -> dict[str, Any]:
    if local_demo.enabled():
        return {
            "policy_version": local_demo.DEMO_POLICY_VERSION,
            "source_model_version": "LOCAL_DEMO_ONLY",
            "source_hsmm_version": "LOCAL_DEMO_ONLY",
            "selected_candidate_id": "DEMO_CAPACITY_RANKER",
            "selected_policy": {"top_k": 6, "bucket_hours": 6},
            "bootstrap_intervals": {},
            "model_card": {"mode": "LOCAL_DEMO", "scientific_claim_allowed": False},
            "quality_gates": [],
            "shadow_api_allowed": True,
            "production_claim_allowed": False,
            "automatic_action_allowed": False,
            "fresh_forward_confirmation_required": True,
            "created_at": local_demo.capacity_snapshot_times()[-1],
        }
    with _connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT policy_version,source_model_version,source_hsmm_version,
                   selected_candidate_id,selected_policy,bootstrap_intervals,
                   model_card,quality_gates,shadow_api_allowed,
                   production_claim_allowed,automatic_action_allowed,
                   fresh_forward_confirmation_required,created_at
            FROM {MODEL_CARD_TABLE} WHERE policy_version=%s
            """,
            (POLICY_VERSION,),
        )
        row = cursor.fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="B61E model card unavailable")
    return {
        "policy_version": row[0], "source_model_version": row[1],
        "source_hsmm_version": row[2], "selected_candidate_id": row[3],
        "selected_policy": dict(row[4]), "bootstrap_intervals": dict(row[5]),
        "model_card": dict(row[6]), "quality_gates": list(row[7]),
        "shadow_api_allowed": bool(row[8]), "production_claim_allowed": bool(row[9]),
        "automatic_action_allowed": bool(row[10]),
        "fresh_forward_confirmation_required": bool(row[11]), "created_at": row[12],
    }
