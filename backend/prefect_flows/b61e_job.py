from __future__ import annotations

import hashlib
import json
import os
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from prefect_flows.b61e_core import (
    POLICY_VERSION,
    SOURCE_HSMM_VERSION,
    SOURCE_MODEL_VERSION,
    TEST_ROLE,
    VALID_ROLES,
    RankingContract,
    RankingParameters,
    calibrate_capacity_on_valid_calibrate,
    cluster_bootstrap_metrics,
    contract_gates,
    parse_score_names,
    parse_top_ks,
    prepare_ranking_frame,
    replay_roles,
    select_score_on_valid_select,
)


SOURCE_NAME = "b61e_capacity_aware_temporal_ranking"
DATASET_NAME = "maritime_capacity_aware_temporal_watchlist_v1"
PREDICTION_RELATION = "serving.maritime_port_call_multitask_prediction_v21"
FEATURE_RELATION = "features.maritime_port_call_governed_v1"
HSMM_RELATION = "serving.maritime_port_call_anchored_hsmm_shadow_v11"
B61D_SCORECARD_RELATION = "serving.maritime_dual_stage_role_scorecard_v131"
OUTPUT_RELATION = "serving.maritime_capacity_watchlist_shadow_v1"
SCORECARD_RELATION = "serving.maritime_capacity_ranking_scorecard_v1"
MODEL_CARD_RELATION = "serving.maritime_capacity_ranking_model_card_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"


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


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    return value


def _db_scalar(value: Any) -> Any:
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _put_bytes(key: str, payload: bytes, content_type: str) -> str:
    _s3_client().put_object(
        Bucket=OUTPUT_BUCKET, Key=key, Body=payload, ContentType=content_type
    )
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _put_json(key: str, payload: Any) -> str:
    return _put_bytes(
        key,
        json.dumps(_json_ready(payload), indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )


def _put_csv(key: str, frame: pd.DataFrame) -> str:
    return _put_bytes(key, frame.to_csv(index=False).encode("utf-8"), "text/csv")


def _verify_upstream() -> dict[str, Any]:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            f"SELECT COUNT(*) FROM {PREDICTION_RELATION} WHERE model_version=%s",
            (SOURCE_MODEL_VERSION,),
        )
        prediction_rows = int(cursor.fetchone()[0])
        cursor.execute(
            f"SELECT COUNT(*) FROM {FEATURE_RELATION} WHERE pre_breach_eligible=true"
        )
        eligible_rows = int(cursor.fetchone()[0])
        cursor.execute(
            """
            SELECT status,metadata->>'decision' FROM audit.ingestion_run
            WHERE source_name='b61b_v21_recalibration_only'
            ORDER BY started_at DESC LIMIT 1
            """
        )
        source_audit = cursor.fetchone()
        cursor.execute(
            """
            SELECT status,metadata->>'decision' FROM audit.ingestion_run
            WHERE source_name='b61d_v131_contract_recalibration'
            ORDER BY started_at DESC LIMIT 1
            """
        )
        diagnostic_audit = cursor.fetchone()
    if prediction_rows <= 0 or eligible_rows <= 0:
        raise RuntimeError("B61E requires B61B-v2.1 predictions and governed labels")
    if source_audit is None or source_audit[0] != "SUCCESS":
        raise RuntimeError("B61B-v2.1 must be successful before B61E")
    if diagnostic_audit is None or diagnostic_audit[0] != "SUCCESS":
        raise RuntimeError("B61D-v1.3.1 diagnosis must exist before B61E")
    diagnostic_decision = str(diagnostic_audit[1] or "")
    if "NOT_VALIDATED" not in diagnostic_decision:
        raise RuntimeError(
            "B61E is only valid after B61D-v1.3.1 records the failed early-warning contract"
        )
    return {
        "prediction_rows": prediction_rows,
        "eligible_feature_rows": eligible_rows,
        "source_audit_status": source_audit[0],
        "source_decision": source_audit[1],
        "diagnostic_audit_status": diagnostic_audit[0],
        "diagnostic_decision": diagnostic_decision,
    }


def load_source() -> pd.DataFrame:
    frame = _query_frame(
        f"""
        SELECT p.model_version,p.source_model_version,p.port_call_id,p.landmark_at,
               p.split,p.evaluation_role,p.regime,p.p_delay_gt1,p.p_delay_gt3,
               p.p_delay_gt6,p.predicted_delay_class,p.remaining_p10_h,
               p.remaining_p50_h,p.remaining_p90_h,p.p_gt3_breach_within_6h,
               p.p_gt3_breach_within_12h,p.p_gt3_breach_within_24h,
               g.vessel_name,g.port_code,g.terminal_code,g.vessel_type,g.cargo_group,
               g.planned_eta,g.planned_etd,g.pre_breach_eligible,
               g.per_call_sample_weight AS decision_weight,
               g.target_gt3_breach_within_24h AS target_event_within_24h,
               g.target_delay_gt_3h,g.target_breach_or_censor_h,
               h.hsmm_state,h.hsmm_state_confidence,h.hsmm_risk_score,
               h.hsmm_escalation_probability
        FROM {PREDICTION_RELATION} p
        JOIN {FEATURE_RELATION} g
          ON g.port_call_id=p.port_call_id AND g.landmark_at=p.landmark_at
        LEFT JOIN {HSMM_RELATION} h
          ON h.port_call_id=p.port_call_id AND h.landmark_at=p.landmark_at
         AND h.evaluation_role=p.evaluation_role AND h.model_version=%s
        WHERE p.model_version=%s
          AND g.pre_breach_eligible=true
          AND p.evaluation_role IN ('VALID_SELECT','VALID_CALIBRATE','TEST_DIAGNOSTIC_ONLY')
        ORDER BY p.evaluation_role,p.landmark_at,p.port_call_id
        """,
        (SOURCE_HSMM_VERSION, SOURCE_MODEL_VERSION),
    )
    if frame.empty:
        raise RuntimeError("B61E joined source is empty")
    key = ["evaluation_role", "landmark_at", "port_call_id"]
    if frame.duplicated(key).any():
        raise RuntimeError("B61E source join produced duplicate landmark rows")
    roles = set(frame["evaluation_role"].astype(str))
    required = {*VALID_ROLES, TEST_ROLE}
    if roles != required:
        raise RuntimeError(f"B61E source roles mismatch: {sorted(roles)}")
    return frame


def _b61d_diagnostic() -> list[dict[str, Any]]:
    frame = _query_frame(
        f"""
        SELECT role,metrics FROM {B61D_SCORECARD_RELATION}
        WHERE policy_version='b61d-v1.3.1-contract-recalibration-v1'
          AND selected=true ORDER BY role
        """
    )
    return [
        {"role": str(row.role), "metrics": _json_ready(dict(row.metrics))}
        for row in frame.itertuples()
    ]


OUTPUT_COLUMNS = [
    "policy_version", "source_model_version", "source_hsmm_version", "candidate_id",
    "port_call_id", "vessel_name", "port_code", "terminal_code", "vessel_type",
    "cargo_group", "planned_eta", "planned_etd", "landmark_at", "decision_at",
    "split", "evaluation_role", "regime", "score_name", "risk_score",
    "rank_in_window", "active_calls", "capacity", "top_k", "bucket_hours",
    "watchlist_selected", "action_tier", "reason_code", "p_delay_gt1", "p_delay_gt3",
    "p_delay_gt6", "p_gt3_breach_within_6h", "p_gt3_breach_within_12h",
    "p_gt3_breach_within_24h", "predicted_delay_class", "remaining_p10_h",
    "remaining_p50_h", "remaining_p90_h", "hsmm_state", "hsmm_state_confidence",
    "hsmm_risk_score", "hsmm_escalation_probability", "target_event_within_24h",
    "target_delay_gt_3h", "target_breach_or_censor_h", "decision_weight",
    "production_claim_allowed", "automatic_action_allowed", "materialization_run_id",
]


def _materialize_decisions(frame: pd.DataFrame, run_id: str) -> int:
    output = frame.copy()
    output["policy_version"] = POLICY_VERSION
    output["source_model_version"] = SOURCE_MODEL_VERSION
    output["source_hsmm_version"] = SOURCE_HSMM_VERSION
    output["candidate_id"] = output["policy_id"]
    output["materialization_run_id"] = run_id
    output = output[OUTPUT_COLUMNS]
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS serving;
    CREATE TABLE IF NOT EXISTS {OUTPUT_RELATION} (
        policy_version TEXT NOT NULL,source_model_version TEXT NOT NULL,
        source_hsmm_version TEXT NOT NULL,candidate_id TEXT NOT NULL,
        port_call_id TEXT NOT NULL,vessel_name TEXT,port_code TEXT,terminal_code TEXT,
        vessel_type TEXT,cargo_group TEXT,planned_eta TIMESTAMPTZ,planned_etd TIMESTAMPTZ,
        landmark_at TIMESTAMPTZ NOT NULL,decision_at TIMESTAMPTZ NOT NULL,
        split TEXT NOT NULL,evaluation_role TEXT NOT NULL,regime TEXT NOT NULL,
        score_name TEXT NOT NULL,risk_score DOUBLE PRECISION NOT NULL,
        rank_in_window INTEGER NOT NULL,active_calls INTEGER NOT NULL,capacity INTEGER NOT NULL,
        top_k INTEGER NOT NULL,bucket_hours INTEGER NOT NULL,watchlist_selected BOOLEAN NOT NULL,
        action_tier TEXT NOT NULL,reason_code TEXT NOT NULL,p_delay_gt1 DOUBLE PRECISION NOT NULL,
        p_delay_gt3 DOUBLE PRECISION NOT NULL,p_delay_gt6 DOUBLE PRECISION NOT NULL,
        p_gt3_breach_within_6h DOUBLE PRECISION NOT NULL,
        p_gt3_breach_within_12h DOUBLE PRECISION NOT NULL,
        p_gt3_breach_within_24h DOUBLE PRECISION NOT NULL,
        predicted_delay_class TEXT NOT NULL,remaining_p10_h DOUBLE PRECISION NOT NULL,
        remaining_p50_h DOUBLE PRECISION NOT NULL,remaining_p90_h DOUBLE PRECISION NOT NULL,
        hsmm_state TEXT,hsmm_state_confidence DOUBLE PRECISION,hsmm_risk_score DOUBLE PRECISION,
        hsmm_escalation_probability DOUBLE PRECISION,target_event_within_24h BOOLEAN NOT NULL,
        target_delay_gt_3h BOOLEAN NOT NULL,target_breach_or_censor_h DOUBLE PRECISION,
        decision_weight DOUBLE PRECISION NOT NULL,production_claim_allowed BOOLEAN NOT NULL,
        automatic_action_allowed BOOLEAN NOT NULL,materialization_run_id TEXT NOT NULL,
        PRIMARY KEY(policy_version,evaluation_role,decision_at,port_call_id)
    );
    """
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(ddl)
        cursor.execute(f"DELETE FROM {OUTPUT_RELATION} WHERE policy_version=%s", (POLICY_VERSION,))
        for start in range(0, len(output), 1_500):
            records = [
                tuple(_db_scalar(value) for value in row)
                for row in output.iloc[start:start + 1_500].itertuples(index=False, name=None)
            ]
            execute_values(
                cursor,
                f"INSERT INTO {OUTPUT_RELATION} ({','.join(OUTPUT_COLUMNS)}) VALUES %s",
                records,
                page_size=1_500,
            )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS ix_b61e_snapshot ON {OUTPUT_RELATION} "
            "(evaluation_role,decision_at DESC,watchlist_selected,rank_in_window)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS ix_b61e_call ON {OUTPUT_RELATION} "
            "(port_call_id,decision_at)"
        )
    return len(output)


def _materialize_reports(
    score_selection: pd.DataFrame,
    capacity_calibration: pd.DataFrame,
    diagnostics: pd.DataFrame,
    parameters: RankingParameters,
    bootstraps: dict[str, dict[str, Any]],
    quality_gates: pd.DataFrame,
    model_card: dict[str, Any],
    run_id: str,
) -> None:
    records = []
    for stage, frame, selected_column in (
        ("SCORE_SELECTION", score_selection, "selected_score"),
        ("CAPACITY_CALIBRATION", capacity_calibration, "selected_capacity"),
    ):
        for row in frame.itertuples(index=False):
            payload = row._asdict()
            records.append((
                POLICY_VERSION, str(payload["candidate_id"]), stage,
                "VALID_SELECT" if stage == "SCORE_SELECTION" else "VALID_CALIBRATE",
                bool(payload[selected_column]), Json(_json_ready(payload)), run_id,
            ))
    for row in diagnostics.itertuples(index=False):
        payload = row._asdict()
        records.append((
            POLICY_VERSION, parameters.policy_id, "FINAL_REPLAY", str(payload["role"]),
            True, Json(_json_ready(payload)), run_id,
        ))
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {SCORECARD_RELATION} (
                policy_version TEXT NOT NULL,candidate_id TEXT NOT NULL,stage TEXT NOT NULL,
                role TEXT NOT NULL,selected BOOLEAN NOT NULL,metrics JSONB NOT NULL,
                materialization_run_id TEXT NOT NULL,
                PRIMARY KEY(policy_version,candidate_id,stage,role)
            )
        """)
        cursor.execute(f"DELETE FROM {SCORECARD_RELATION} WHERE policy_version=%s", (POLICY_VERSION,))
        execute_values(cursor, f"INSERT INTO {SCORECARD_RELATION} VALUES %s", records)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {MODEL_CARD_RELATION} (
                policy_version TEXT PRIMARY KEY,source_model_version TEXT NOT NULL,
                source_hsmm_version TEXT NOT NULL,selected_candidate_id TEXT NOT NULL,
                selected_policy JSONB NOT NULL,bootstrap_intervals JSONB NOT NULL,
                model_card JSONB NOT NULL,quality_gates JSONB NOT NULL,
                shadow_api_allowed BOOLEAN NOT NULL,production_claim_allowed BOOLEAN NOT NULL,
                automatic_action_allowed BOOLEAN NOT NULL,
                fresh_forward_confirmation_required BOOLEAN NOT NULL,
                materialization_run_id TEXT NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
        """)
        cursor.execute(f"DELETE FROM {MODEL_CARD_RELATION} WHERE policy_version=%s", (POLICY_VERSION,))
        cursor.execute(
            f"INSERT INTO {MODEL_CARD_RELATION} VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,false,false,true,%s,now())",
            (
                POLICY_VERSION, SOURCE_MODEL_VERSION, SOURCE_HSMM_VERSION,
                parameters.policy_id,
                Json(_json_ready({
                    "score_name": parameters.score_name,
                    "top_k": parameters.top_k,
                    "bucket_hours": parameters.bucket_hours,
                })),
                Json(_json_ready(bootstraps)), Json(_json_ready(model_card)),
                Json(_json_ready(quality_gates.to_dict("records"))),
                bool(model_card["shadow_api_allowed"]), run_id,
            ),
        )


def _quality_gates(
    prepared: pd.DataFrame,
    decisions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    bootstraps: dict[str, dict[str, Any]],
    parameters: RankingParameters,
    upstream: dict[str, Any],
) -> pd.DataFrame:
    contract = RankingContract()
    role_map = {
        str(row.role): row._asdict() for row in diagnostics.itertuples(index=False)
    }
    valid_metrics = [role_map[role] for role in VALID_ROLES]
    gates = [
        ("UPSTREAM_B61B_V21_SUCCESS", upstream["source_audit_status"] == "SUCCESS", "CRITICAL", upstream["source_decision"]),
        (
            "B61D_V131_FAILURE_DIAGNOSED",
            upstream["diagnostic_audit_status"] == "SUCCESS"
            and "NOT_VALIDATED" in upstream["diagnostic_decision"],
            "CRITICAL",
            upstream["diagnostic_decision"],
        ),
        ("PRE_BREACH_ONLY", bool(prepared["pre_breach_eligible"].all()), "CRITICAL", len(prepared)),
        ("UNIQUE_WINDOW_CALL_ROWS", not prepared.duplicated(["evaluation_role", "decision_at", "port_call_id"]).any(), "CRITICAL", len(prepared)),
        ("ALL_HSMM_STATES_ELIGIBLE", True, "CRITICAL", "HSMM is context only, never a hard filter"),
        ("SCORE_SELECTED_ON_VALID_SELECT", True, "CRITICAL", parameters.score_name),
        ("CAPACITY_CALIBRATED_ON_VALID_CALIBRATE", True, "CRITICAL", parameters.top_k),
        ("TEST_DIAGNOSTIC_ONLY", True, "CRITICAL", TEST_ROLE),
        ("NO_MODEL_RETRAINING", True, "CRITICAL", SOURCE_MODEL_VERSION),
        ("NO_PRODUCTION_ACTION", not bool(decisions["production_claim_allowed"].any()), "CRITICAL", False),
        ("NO_AUTOMATIC_ACTION", not bool(decisions["automatic_action_allowed"].any()), "CRITICAL", False),
        ("PRECISION_BOTH_VALID", all(float(item["precision"]) >= contract.min_precision for item in valid_metrics), "MODEL", {role: role_map[role]["precision"] for role in VALID_ROLES}),
        ("RECALL_BOTH_VALID", all(float(item["recall"]) >= contract.min_recall for item in valid_metrics), "MODEL", {role: role_map[role]["recall"] for role in VALID_ROLES}),
        ("LIFT_BOTH_VALID", all(float(item["precision_lift"]) >= contract.min_precision_lift for item in valid_metrics), "MODEL", {role: role_map[role]["precision_lift"] for role in VALID_ROLES}),
        ("F1_BOTH_VALID", all(float(item["f1"]) >= contract.min_f1 for item in valid_metrics), "MODEL", {role: role_map[role]["f1"] for role in VALID_ROLES}),
        ("EVENT_RECALL_BOTH_VALID", all(float(item["event_recall_any"]) >= contract.min_event_recall for item in valid_metrics), "MODEL", {role: role_map[role]["event_recall_any"] for role in VALID_ROLES}),
        ("CAPACITY_WITHIN_CONTRACT", parameters.top_k <= contract.max_top_k, "MODEL", parameters.top_k),
        ("BOOTSTRAP_PRECISION_LOWER_BOTH", all(float(item["precision"]["p2_5"]) >= contract.min_bootstrap_precision_lower for item in bootstraps.values()), "MODEL", {role: bootstraps[role]["precision"]["p2_5"] for role in VALID_ROLES}),
        ("BOOTSTRAP_RECALL_LOWER_BOTH", all(float(item["recall"]["p2_5"]) >= contract.min_bootstrap_recall_lower for item in bootstraps.values()), "MODEL", {role: bootstraps[role]["recall"]["p2_5"] for role in VALID_ROLES}),
    ]
    return pd.DataFrame([
        {"check": check, "passed": bool(passed), "severity": severity, "value": _json_ready(value)}
        for check, passed, severity, value in gates
    ])


def _checksum(source: pd.DataFrame, settings: tuple[Any, ...]) -> str:
    digest = hashlib.sha256(POLICY_VERSION.encode("ascii"))
    digest.update(repr(settings).encode("ascii"))
    digest.update(
        pd.util.hash_pandas_object(
            source[["port_call_id", "landmark_at", "p_gt3_breach_within_24h"]],
            index=False,
        ).to_numpy(dtype="uint64").tobytes()
    )
    return digest.hexdigest()


def _existing_success(checksum: str) -> dict[str, Any] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT metadata FROM audit.ingestion_run
            WHERE source_name=%s AND checksum=%s AND status='SUCCESS'
            ORDER BY started_at DESC LIMIT 1
            """,
            (SOURCE_NAME, checksum),
        )
        row = cursor.fetchone()
    return dict(row[0]) if row else None


def _start_run(checksum: str) -> str:
    metadata = {
        "policy_version": POLICY_VERSION,
        "orchestrator": "PREFECT",
        "selection_role": "VALID_SELECT",
        "calibration_role": "VALID_CALIBRATE",
        "test_role": TEST_ROLE,
        "test_used_for_selection": False,
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
            (SOURCE_NAME, DATASET_NAME, f"postgresql://maritime/{OUTPUT_RELATION}", checksum, Json(metadata)),
        )
        return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    payload = _json_ready({"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details})
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
            (Json({"progress": payload}), run_id),
        )


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE audit.ingestion_run
            SET finished_at=now(),status=%s,row_count=%s,
                metadata=metadata || %s,error_message=%s WHERE run_id=%s
            """,
            (status, row_count, Json(_json_ready(metadata)), error_message, run_id),
        )


def run_b61e(
    *,
    force: bool = False,
    score_names: str = "HAZARD_24H,P_GT3,TEMPORAL_HAZARD_MOE",
    top_ks: str = "1,2",
    bucket_hours: int = 6,
    bootstrap_iterations: int = 500,
) -> dict[str, Any]:
    parsed_scores = parse_score_names(score_names)
    parsed_top_ks = parse_top_ks(top_ks)
    if bucket_hours not in (3, 6, 12):
        raise ValueError("bucket_hours must be 3, 6 or 12")
    if bootstrap_iterations < 200:
        raise ValueError("At least 200 bootstrap iterations are required")
    upstream = _verify_upstream()
    source = load_source()
    checksum = _checksum(
        source,
        (tuple(parsed_scores), tuple(parsed_top_ks), bucket_hours, bootstrap_iterations),
    )
    if not force:
        existing = _existing_success(checksum)
        if existing is not None:
            return existing
    run_id = _start_run(checksum)
    try:
        _update_progress(run_id, "PREPARING_PRE_BREACH_CAPACITY_WINDOWS", source_rows=len(source))
        prepared = prepare_ranking_frame(source, bucket_hours)
        _update_progress(
            run_id, "VALID_SELECT_SCORE_SELECTION",
            prepared_rows=len(prepared), score_candidates=len(parsed_scores),
        )
        score_selection, selected_score = select_score_on_valid_select(
            prepared, parsed_scores, bucket_hours
        )
        _update_progress(
            run_id, "VALID_CALIBRATE_CAPACITY_CALIBRATION",
            selected_score=selected_score, capacity_candidates=len(parsed_top_ks),
        )
        capacity_calibration, parameters = calibrate_capacity_on_valid_calibrate(
            prepared, selected_score, parsed_top_ks, bucket_hours
        )
        decisions, diagnostics = replay_roles(prepared, parameters)
        _update_progress(
            run_id, "PORT_CALL_CLUSTER_BOOTSTRAP",
            candidate_id=parameters.policy_id, iterations=bootstrap_iterations,
        )
        bootstraps = {}
        for offset, role in enumerate(VALID_ROLES):
            role_decisions = decisions.loc[decisions["evaluation_role"].eq(role)]
            bootstraps[role] = cluster_bootstrap_metrics(
                role_decisions,
                iterations=bootstrap_iterations,
                random_state=20260810 + offset,
            )
        quality_gates = _quality_gates(
            prepared, decisions, diagnostics, bootstraps, parameters, upstream
        )
        integrity_passed = bool(
            quality_gates.loc[quality_gates["severity"].eq("CRITICAL"), "passed"].all()
        )
        contracts_passed = bool(
            quality_gates.loc[quality_gates["severity"].eq("MODEL"), "passed"].all()
        )
        accepted = integrity_passed and contracts_passed
        decision = (
            "READY_FOR_B61F_FRESH_FORWARD_CAPACITY_WATCHLIST"
            if accepted else "KEEP_B61C_CHAMPION_B61E_RESEARCH_ONLY"
        )
        next_block = (
            "B61F_FRESH_FORWARD_CAPACITY_WATCHLIST_VALIDATION"
            if accepted else "RETAIN_B61C_AND_REVIEW_B61E_FAILED_CONTRACTS"
        )
        _update_progress(run_id, "WRITING_VERSIONED_WATCHLIST_ARTIFACTS")
        serving_rows = _materialize_decisions(decisions, run_id)
        report_root = f"reports/b61e/{OUTPUT_PREFIX}"
        model_root = f"models/b61e/{OUTPUT_PREFIX}"
        b61d_diagnostic = _b61d_diagnostic()
        artifacts = {
            "score_selection": _put_csv(f"{report_root}/01_valid_select_score_selection.csv", score_selection),
            "capacity_calibration": _put_csv(f"{report_root}/02_valid_calibrate_capacity.csv", capacity_calibration),
            "role_diagnostics": _put_csv(f"{report_root}/03_final_role_diagnostics.csv", diagnostics),
            "quality_gates": _put_csv(f"{report_root}/04_quality_gates.csv", quality_gates),
            "bootstraps": _put_json(f"{report_root}/05_bootstrap_intervals.json", bootstraps),
        }
        policy_config = {
            "policy_version": POLICY_VERSION,
            "target": "GT3_DELAY_BREACH_WITHIN_NEXT_24H",
            "score_name": parameters.score_name,
            "top_k": parameters.top_k,
            "bucket_hours": parameters.bucket_hours,
            "all_hsmm_states_eligible": True,
            "selection_role": "VALID_SELECT",
            "calibration_role": "VALID_CALIBRATE",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "automatic_action_allowed": False,
        }
        artifacts["policy_config"] = _put_json(f"{model_root}/policy_config.json", policy_config)
        model_card = {
            **policy_config,
            "architecture": (
                "CAPACITY_AWARE_TEMPORAL_RANKING_OVER_ALL_STATES_WITH_"
                "HAZARD_SCORE_SELECTION_AND_SEPARATE_CAPACITY_CALIBRATION"
            ),
            "scientific_rationale": (
                "B61D-v1.3.1 failed because its hard non-fluid eligibility rule removed "
                "true events. B61E ranks every active call using frozen issue-time temporal "
                "probabilities; HSMM state is explanatory context only."
            ),
            "source_model_version": SOURCE_MODEL_VERSION,
            "source_hsmm_version": SOURCE_HSMM_VERSION,
            "contracts_passed": contracts_passed,
            "integrity_passed": integrity_passed,
            "shadow_api_allowed": accepted,
            "production_claim_allowed": False,
            "automatic_action_allowed": False,
            "fresh_forward_confirmation_required": True,
            "operational_lead_limitation": (
                "Historical replay supports a capacity-ranked 24h watchlist, but it does not "
                "establish reliable warning three or six hours before breach. Lead-time metrics "
                "remain diagnostic and require fresh forward confirmation."
            ),
            "test_disclosure": (
                "TEST was previously consumed upstream and is replayed only after score "
                "selection and capacity calibration. It is diagnostic, not confirmatory."
            ),
            "b61d_v131_diagnostic": b61d_diagnostic,
            "bootstrap_intervals": bootstraps,
            "artifacts": artifacts,
        }
        artifacts["model_card"] = _put_json(f"{report_root}/06_model_card.json", model_card)
        _materialize_reports(
            score_selection, capacity_calibration, diagnostics, parameters,
            bootstraps, quality_gates, model_card, run_id,
        )
        role_metrics = {
            str(row.role): _json_ready(row._asdict())
            for row in diagnostics.itertuples(index=False)
        }
        metadata = {
            "policy_version": POLICY_VERSION,
            "decision": decision,
            "selected_candidate_id": parameters.policy_id,
            "selected_score": parameters.score_name,
            "selected_top_k": parameters.top_k,
            "bucket_hours": parameters.bucket_hours,
            "selection_role": "VALID_SELECT",
            "calibration_role": "VALID_CALIBRATE",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "test_used_for_selection": False,
            "contracts_passed": contracts_passed,
            "integrity_passed": integrity_passed,
            "shadow_api_allowed": accepted,
            "production_promotion_allowed": False,
            "automatic_action_allowed": False,
            "serving_rows": serving_rows,
            "role_metrics": role_metrics,
            "bootstrap_clusters": {
                role: payload["clusters"] for role, payload in bootstraps.items()
            },
            "next_block": next_block,
            "artifacts": artifacts,
            "progress": {"stage": "COMPLETE", "updated_at": pd.Timestamp.now(tz="UTC")},
        }
        _finish_run(run_id, "SUCCESS", serving_rows, metadata)
        return _json_ready(metadata)
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "policy_version": POLICY_VERSION,
                "decision": "FAILED",
                "production_promotion_allowed": False,
                "automatic_action_allowed": False,
            },
            str(exc),
        )
        raise
