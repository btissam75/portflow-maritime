from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from prefect_flows.b61c_core import evaluate_policy
from prefect_flows.b61d_v12_core import (
    POLICY_VERSION,
    SOURCE_HSMM_VERSION,
    SOURCE_POLICY_VERSION,
    StatePolicyParameters,
    add_selection_constraints,
    apply_state_conditional_policy,
    choose_constrained_policy,
    cluster_bootstrap_metrics,
    event_lead_metrics,
    parse_integer_grid,
    parse_modes,
    parse_numeric_grid,
    policy_objective,
)


SOURCE_NAME = "b61d_v12_state_conditional_policy"
DATASET_NAME = "maritime_state_conditional_policy_shadow_v12"
SOURCE_RELATION = "serving.maritime_port_call_anchored_hsmm_shadow_v11"
SOURCE_MODEL_CARD_RELATION = "serving.maritime_anchored_hsmm_model_card_v11"
BASELINE_DECISION_RELATION = "serving.maritime_port_call_decision_shadow_v1"
BASELINE_SCORECARD_RELATION = "serving.maritime_decision_policy_scorecard_v1"
OUTPUT_RELATION = "serving.maritime_port_call_state_policy_shadow_v12"
SCORECARD_RELATION = "serving.maritime_state_policy_scorecard_v12"
MODEL_CARD_RELATION = "serving.maritime_state_policy_model_card_v12"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1.2"


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
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
            f"SELECT COUNT(*) FROM {SOURCE_RELATION} WHERE model_version=%s",
            (SOURCE_HSMM_VERSION,),
        )
        source_rows = int(cursor.fetchone()[0])
        cursor.execute(
            f"SELECT COUNT(*) FROM {BASELINE_DECISION_RELATION} WHERE policy_version=%s",
            (SOURCE_POLICY_VERSION,),
        )
        baseline_rows = int(cursor.fetchone()[0])
        cursor.execute(
            """
            SELECT status,metadata->>'decision'
            FROM audit.ingestion_run WHERE source_name='b61d_v11_anchored_hsmm'
            ORDER BY started_at DESC LIMIT 1
            """
        )
        audit = cursor.fetchone()
    if source_rows <= 0 or baseline_rows <= 0:
        raise RuntimeError("B61D-v1.2 requires frozen B61D-v1.1 and B61C-v1.1 rows")
    if audit is None or audit[0] != "SUCCESS":
        raise RuntimeError("B61D-v1.1 must be successful before state-policy replay")
    return {
        "source_rows": source_rows,
        "baseline_rows": baseline_rows,
        "source_audit_status": audit[0],
        "source_decision": audit[1],
    }


def load_source() -> pd.DataFrame:
    frame = _query_frame(
        f"""
        SELECT * FROM {SOURCE_RELATION}
        WHERE model_version=%s
          AND evaluation_role IN (
              'VALID_SELECT','VALID_CALIBRATE','TEST_DIAGNOSTIC_ONLY'
          )
        ORDER BY evaluation_role,decision_at,port_call_id
        """,
        (SOURCE_HSMM_VERSION,),
    )
    if frame.empty:
        raise RuntimeError("Frozen B61D-v1.1 decision source is empty")
    key = ["evaluation_role", "decision_at", "port_call_id"]
    if frame.duplicated(key).any():
        raise RuntimeError("Frozen B61D-v1.1 source has duplicate decision rows")
    for column in ("landmark_at", "decision_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    for column in (
        "target_delay_gt_3h", "target_delay_gt_6h",
        "production_claim_allowed", "automatic_action_allowed",
    ):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def _baseline_decisions() -> pd.DataFrame:
    return _query_frame(
        f"""
        SELECT port_call_id,evaluation_role,decision_at,alert_active,state,
               target_delay_gt_3h,target_delay_gt_6h,
               target_breach_or_censor_h,decision_weight
        FROM {BASELINE_DECISION_RELATION}
        WHERE policy_version=%s
        ORDER BY evaluation_role,port_call_id,decision_at
        """,
        (SOURCE_POLICY_VERSION,),
    )


def _baseline_metrics() -> dict[str, dict[str, Any]]:
    scorecard = _query_frame(
        f"""
        SELECT role,metrics FROM {BASELINE_SCORECARD_RELATION}
        WHERE policy_version=%s AND selected=true
        """,
        (SOURCE_POLICY_VERSION,),
    )
    result = {str(row.role): dict(row.metrics) for row in scorecard.itertuples()}
    for role, role_frame in _baseline_decisions().groupby("evaluation_role", sort=False):
        result.setdefault(str(role), {}).update(event_lead_metrics(role_frame))
    required = {"VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"}
    if not required.issubset(result):
        raise RuntimeError(f"B61C baseline misses roles: {sorted(required.difference(result))}")
    return result


def _parameters_from_row(row: dict[str, Any], bucket_hours: int) -> StatePolicyParameters:
    return StatePolicyParameters(
        mode=str(row["policy_mode"]),
        critical_probability_threshold=float(row["critical_probability_threshold"]),
        alert_budget_pct=float(row["alert_budget_pct"]),
        hold_windows=int(row["hold_windows"]),
        bucket_hours=bucket_hours,
    )


def _evaluate(decisions: pd.DataFrame, parameters: StatePolicyParameters) -> dict[str, Any]:
    metrics = evaluate_policy(decisions, parameters)
    metrics.update(event_lead_metrics(decisions))
    metrics.update(
        {
            "candidate_id": parameters.policy_id,
            "policy_mode": parameters.mode,
            "critical_probability_threshold": parameters.critical_probability_threshold,
            "alert_budget_pct": parameters.alert_budget_pct,
            "hold_windows": parameters.hold_windows,
        }
    )
    metrics["objective"] = policy_objective(metrics)
    return metrics


def _candidate_replay(
    source: pd.DataFrame,
    modes: list[str],
    probability_thresholds: list[float],
    alert_budgets: list[float],
    hold_windows: list[int],
    bucket_hours: int,
    baseline: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = source.loc[source["evaluation_role"].eq("VALID_CALIBRATE")].copy()
    if valid.empty:
        raise RuntimeError("VALID_CALIBRATE rows are required for policy selection")
    rows: list[dict[str, Any]] = []
    for mode in modes:
        for threshold in probability_thresholds:
            for budget in alert_budgets:
                for hold in hold_windows:
                    parameters = StatePolicyParameters(
                        mode=mode,
                        critical_probability_threshold=threshold,
                        alert_budget_pct=budget,
                        hold_windows=hold,
                        bucket_hours=bucket_hours,
                    )
                    decisions = apply_state_conditional_policy(valid, parameters)
                    rows.append(_evaluate(decisions, parameters))
    scorecard = add_selection_constraints(pd.DataFrame(rows), baseline)
    selected = choose_constrained_policy(scorecard)
    scorecard["selected"] = scorecard["candidate_id"].eq(selected["candidate_id"])
    return scorecard, selected


def _selected_replay(
    source: pd.DataFrame,
    selected: dict[str, Any],
    bucket_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, StatePolicyParameters]:
    parameters = _parameters_from_row(selected, bucket_hours)
    decisions = apply_state_conditional_policy(source, parameters)
    diagnostics = []
    for _, role_frame in decisions.groupby("evaluation_role", sort=False):
        diagnostics.append(_evaluate(role_frame, parameters))
    return decisions, pd.DataFrame(diagnostics), parameters


def _quality_gates(
    source: pd.DataFrame,
    decisions: pd.DataFrame,
    selected: dict[str, Any],
    bootstrap: dict[str, Any],
    upstream: dict[str, Any],
) -> pd.DataFrame:
    selected_passes = bool(selected["passes_constraints"])
    roles = set(source["evaluation_role"].astype(str))
    gates = [
        ("UPSTREAM_V11_FROZEN_SUCCESS", upstream["source_audit_status"] == "SUCCESS", "CRITICAL", upstream),
        ("UNIQUE_DECISION_ROWS", not source.duplicated(["evaluation_role", "decision_at", "port_call_id"]).any(), "CRITICAL", len(source)),
        ("VALID_CALIBRATE_ONLY_SELECTION", True, "CRITICAL", "VALID_CALIBRATE"),
        ("TEST_DIAGNOSTIC_ONLY", "TEST_DIAGNOSTIC_ONLY" in roles, "CRITICAL", "not used for selection"),
        ("NO_HSMM_OR_PREDICTOR_RETRAINING", True, "CRITICAL", SOURCE_HSMM_VERSION),
        ("NO_PRODUCTION_ACTION", not bool(decisions["production_claim_allowed"].any()), "CRITICAL", False),
        ("NO_AUTOMATIC_ACTION", not bool(decisions["automatic_action_allowed"].any()), "CRITICAL", False),
        ("SELECTED_LIFT_CONSTRAINT", bool(selected["gate_lift"]), "MODEL", selected["gt3_precision_lift"]),
        ("SELECTED_RECALL_CONSTRAINT", bool(selected["gate_recall"]), "MODEL", selected["gt3_recall"]),
        ("SELECTED_STABILITY_CONSTRAINT", bool(selected["gate_stability"]), "MODEL", selected["state_stability"]),
        ("SELECTED_COST_CONSTRAINT", bool(selected["gate_cost"]), "MODEL", selected["cost_reduction_pct"]),
        ("SELECTED_LEAD_CONSTRAINT", bool(selected["gate_lead"]), "MODEL", selected.get("median_event_lead_h")),
        ("ALL_SELECTION_CONSTRAINTS", selected_passes, "MODEL", selected["candidate_id"]),
        ("CLUSTER_BOOTSTRAP_REPORTED", int(bootstrap["iterations"]) >= 200, "MODEL", bootstrap["iterations"]),
        ("BOOTSTRAP_LIFT_LOWER_ABOVE_RANDOM", float(bootstrap["gt3_precision_lift"]["p2_5"]) > 1.0, "MODEL", bootstrap["gt3_precision_lift"]["p2_5"]),
    ]
    return pd.DataFrame(
        [
            {"check": check, "passed": bool(passed), "severity": severity, "value": _json_ready(value)}
            for check, passed, severity, value in gates
        ]
    )


def _materialize_decisions(frame: pd.DataFrame, run_id: str) -> int:
    columns = [
        "policy_version", "source_hsmm_version", "source_policy_version",
        "candidate_id", "port_call_id", "landmark_at", "decision_at", "split",
        "evaluation_role", "regime", "hsmm_state", "hsmm_state_confidence",
        "hsmm_risk_score", "hsmm_escalation_probability", "hsmm_dwell_steps",
        "p_state_fluid", "p_state_pressure_building", "p_state_congested",
        "p_state_critical_disruption", "p_state_recovery",
        "base_temporal_priority_score", "base_critical_priority_score",
        "state_priority_score", "effective_priority_score", "policy_candidate",
        "candidate_reason", "rank_in_bucket", "active_calls", "alert_capacity",
        "policy_mode", "critical_probability_threshold", "alert_budget_pct",
        "hold_windows", "state", "previous_state", "state_changed", "alert_active",
        "new_alert", "action_code", "p_delay_gt3", "p_delay_gt6",
        "p_gt3_breach_within_6h", "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h", "remaining_p10_h", "remaining_p50_h",
        "remaining_p90_h", "target_delay_gt_3h", "target_delay_gt_6h",
        "target_breach_or_censor_h", "decision_weight", "source_mode",
        "production_claim_allowed", "automatic_action_allowed",
        "materialization_run_id",
    ]
    output = frame.copy()
    output["policy_version"] = POLICY_VERSION
    output["source_hsmm_version"] = SOURCE_HSMM_VERSION
    output["source_policy_version"] = SOURCE_POLICY_VERSION
    output["candidate_id"] = output["policy_id"]
    output["materialization_run_id"] = run_id
    output = output[columns]
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS serving;
    CREATE TABLE IF NOT EXISTS {OUTPUT_RELATION} (
        policy_version TEXT NOT NULL,source_hsmm_version TEXT NOT NULL,
        source_policy_version TEXT NOT NULL,candidate_id TEXT NOT NULL,
        port_call_id TEXT NOT NULL,landmark_at TIMESTAMPTZ NOT NULL,
        decision_at TIMESTAMPTZ NOT NULL,split TEXT NOT NULL,
        evaluation_role TEXT NOT NULL,regime TEXT NOT NULL,hsmm_state TEXT NOT NULL,
        hsmm_state_confidence DOUBLE PRECISION NOT NULL,
        hsmm_risk_score DOUBLE PRECISION NOT NULL,
        hsmm_escalation_probability DOUBLE PRECISION NOT NULL,
        hsmm_dwell_steps INTEGER NOT NULL,p_state_fluid DOUBLE PRECISION NOT NULL,
        p_state_pressure_building DOUBLE PRECISION NOT NULL,
        p_state_congested DOUBLE PRECISION NOT NULL,
        p_state_critical_disruption DOUBLE PRECISION NOT NULL,
        p_state_recovery DOUBLE PRECISION NOT NULL,
        base_temporal_priority_score DOUBLE PRECISION NOT NULL,
        base_critical_priority_score DOUBLE PRECISION NOT NULL,
        state_priority_score DOUBLE PRECISION NOT NULL,
        effective_priority_score DOUBLE PRECISION NOT NULL,
        policy_candidate BOOLEAN NOT NULL,candidate_reason TEXT NOT NULL,
        rank_in_bucket INTEGER NOT NULL,active_calls INTEGER NOT NULL,
        alert_capacity INTEGER NOT NULL,policy_mode TEXT NOT NULL,
        critical_probability_threshold DOUBLE PRECISION NOT NULL,
        alert_budget_pct DOUBLE PRECISION NOT NULL,hold_windows INTEGER NOT NULL,
        state TEXT NOT NULL,previous_state TEXT NOT NULL,state_changed BOOLEAN NOT NULL,
        alert_active BOOLEAN NOT NULL,new_alert BOOLEAN NOT NULL,action_code TEXT NOT NULL,
        p_delay_gt3 DOUBLE PRECISION NOT NULL,p_delay_gt6 DOUBLE PRECISION NOT NULL,
        p_gt3_breach_within_6h DOUBLE PRECISION NOT NULL,
        p_gt3_breach_within_12h DOUBLE PRECISION NOT NULL,
        p_gt3_breach_within_24h DOUBLE PRECISION NOT NULL,
        remaining_p10_h DOUBLE PRECISION NOT NULL,remaining_p50_h DOUBLE PRECISION NOT NULL,
        remaining_p90_h DOUBLE PRECISION NOT NULL,target_delay_gt_3h BOOLEAN NOT NULL,
        target_delay_gt_6h BOOLEAN NOT NULL,target_breach_or_censor_h DOUBLE PRECISION,
        decision_weight DOUBLE PRECISION NOT NULL,source_mode TEXT NOT NULL,
        production_claim_allowed BOOLEAN NOT NULL,automatic_action_allowed BOOLEAN NOT NULL,
        materialization_run_id TEXT NOT NULL,
        PRIMARY KEY(policy_version,evaluation_role,decision_at,port_call_id)
    );
    """
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(ddl)
        cursor.execute(f"DELETE FROM {OUTPUT_RELATION} WHERE policy_version=%s", (POLICY_VERSION,))
        for start in range(0, len(output), 1_500):
            records = list(output.iloc[start:start + 1_500].itertuples(index=False, name=None))
            execute_values(
                cursor,
                f"INSERT INTO {OUTPUT_RELATION} ({','.join(columns)}) VALUES %s",
                records,
                page_size=1_500,
            )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS ix_b61dv12_snapshot ON {OUTPUT_RELATION} "
            "(decision_at DESC,state,rank_in_bucket)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS ix_b61dv12_call ON {OUTPUT_RELATION} "
            "(port_call_id,decision_at)"
        )
    return len(output)


def _materialize_scorecard_and_card(
    scorecard: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    bootstrap: dict[str, Any],
    gates: pd.DataFrame,
    model_card: dict[str, Any],
    run_id: str,
) -> None:
    records = []
    selected_id = str(selected["candidate_id"])
    for row in scorecard.itertuples(index=False):
        payload = row._asdict()
        records.append(
            (
                POLICY_VERSION, str(payload["candidate_id"]), "VALID_CALIBRATE",
                str(payload["policy_mode"]),
                float(payload["critical_probability_threshold"]),
                float(payload["alert_budget_pct"]), int(payload["hold_windows"]),
                float(payload["objective"]), bool(payload["passes_constraints"]),
                bool(payload["selected"]), Json(_json_ready(payload)), run_id,
            )
        )
    for row in diagnostics.itertuples(index=False):
        payload = row._asdict()
        if str(payload["role"]) == "VALID_CALIBRATE":
            continue
        records.append(
            (
                POLICY_VERSION, selected_id, str(payload["role"]),
                str(payload["policy_mode"]),
                float(payload["critical_probability_threshold"]),
                float(payload["alert_budget_pct"]), int(payload["hold_windows"]),
                float(payload["objective"]), bool(selected["passes_constraints"]),
                True, Json(_json_ready(payload)), run_id,
            )
        )
    dedup = {(item[0], item[1], item[2]): item for item in records}
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCORECARD_RELATION} (
                policy_version TEXT NOT NULL,candidate_id TEXT NOT NULL,role TEXT NOT NULL,
                policy_mode TEXT NOT NULL,critical_probability_threshold DOUBLE PRECISION NOT NULL,
                alert_budget_pct DOUBLE PRECISION NOT NULL,hold_windows INTEGER NOT NULL,
                objective DOUBLE PRECISION NOT NULL,passes_constraints BOOLEAN NOT NULL,
                selected BOOLEAN NOT NULL,metrics JSONB NOT NULL,
                materialization_run_id TEXT NOT NULL,
                PRIMARY KEY(policy_version,candidate_id,role)
            )
            """
        )
        cursor.execute(f"DELETE FROM {SCORECARD_RELATION} WHERE policy_version=%s", (POLICY_VERSION,))
        execute_values(cursor, f"INSERT INTO {SCORECARD_RELATION} VALUES %s", list(dedup.values()))
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MODEL_CARD_RELATION} (
                policy_version TEXT PRIMARY KEY,source_hsmm_version TEXT NOT NULL,
                source_policy_version TEXT NOT NULL,selected_candidate_id TEXT NOT NULL,
                selected_policy JSONB NOT NULL,baseline_metrics JSONB NOT NULL,
                bootstrap_intervals JSONB NOT NULL,model_card JSONB NOT NULL,
                quality_gates JSONB NOT NULL,shadow_api_allowed BOOLEAN NOT NULL,
                production_claim_allowed BOOLEAN NOT NULL,
                automatic_action_allowed BOOLEAN NOT NULL,
                fresh_forward_confirmation_required BOOLEAN NOT NULL,
                materialization_run_id TEXT NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(f"DELETE FROM {MODEL_CARD_RELATION} WHERE policy_version=%s", (POLICY_VERSION,))
        cursor.execute(
            f"INSERT INTO {MODEL_CARD_RELATION} VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,false,false,true,%s,now())",
            (
                POLICY_VERSION, SOURCE_HSMM_VERSION, SOURCE_POLICY_VERSION,
                selected_id, Json(_json_ready(selected)), Json(_json_ready(baseline)),
                Json(_json_ready(bootstrap)), Json(_json_ready(model_card)),
                Json(_json_ready(gates.to_dict("records"))),
                bool(model_card["shadow_api_allowed"]), run_id,
            ),
        )


def _checksum(source: pd.DataFrame, settings: tuple[Any, ...]) -> str:
    digest = hashlib.sha256(POLICY_VERSION.encode("ascii"))
    digest.update(repr(settings).encode("ascii"))
    digest.update(
        pd.util.hash_pandas_object(
            source[["port_call_id", "decision_at", "hsmm_state", "p_state_critical_disruption"]],
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
        "fit_role": "NONE_FROZEN_HSMM_AND_PREDICTOR",
        "selection_role": "VALID_CALIBRATE",
        "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
        "test_used_for_fit_or_selection": False,
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
            (
                SOURCE_NAME, DATASET_NAME, f"postgresql://maritime/{OUTPUT_RELATION}",
                checksum, Json(metadata),
            ),
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
                metadata=metadata || %s,error_message=%s
            WHERE run_id=%s
            """,
            (status, row_count, Json(_json_ready(metadata)), error_message, run_id),
        )


def run_b61d_v12(
    *,
    force: bool = False,
    policy_modes: str = "STATE_STRICT,POSTERIOR_STRICT,CRITICAL_WITH_CONGESTED_BACKSTOP",
    probability_thresholds: str = "0.2,0.3,0.4,0.5",
    alert_budgets: str = "1,2,3",
    hold_windows: str = "0,1,2",
    bucket_hours: int = 6,
    bootstrap_iterations: int = 500,
) -> dict[str, Any]:
    modes = parse_modes(policy_modes)
    thresholds = parse_numeric_grid(
        probability_thresholds, minimum=0.0, maximum=1.0,
        name="probability_thresholds",
    )
    budgets = parse_numeric_grid(
        alert_budgets, minimum=0.1, maximum=25.0, name="alert_budgets"
    )
    holds = parse_integer_grid(
        hold_windows, allowed={0, 1, 2, 3}, name="hold_windows"
    )
    if bucket_hours not in (3, 6):
        raise ValueError("bucket_hours must be 3 or 6")
    if bootstrap_iterations < 200:
        raise ValueError("At least 200 cluster bootstrap iterations are required")
    upstream = _verify_upstream()
    source = load_source()
    settings = (
        tuple(modes), tuple(thresholds), tuple(budgets), tuple(holds),
        bucket_hours, bootstrap_iterations,
    )
    checksum = _checksum(source, settings)
    if not force:
        existing = _existing_success(checksum)
        if existing is not None:
            return existing
    run_id = _start_run(checksum)
    try:
        baseline = _baseline_metrics()
        candidate_count = len(modes) * len(thresholds) * len(budgets) * len(holds)
        _update_progress(
            run_id, "VALID_CONSTRAINED_POLICY_SEARCH",
            rows=len(source), candidates=candidate_count,
        )
        scorecard, selected = _candidate_replay(
            source, modes, thresholds, budgets, holds, bucket_hours,
            baseline["VALID_CALIBRATE"],
        )
        _update_progress(
            run_id, "SELECTED_POLICY_REPLAY_ALL_ROLES",
            selected_candidate_id=selected["candidate_id"],
            passes_constraints=bool(selected["passes_constraints"]),
        )
        decisions, diagnostics, parameters = _selected_replay(
            source, selected, bucket_hours
        )
        valid_decisions = decisions.loc[
            decisions["evaluation_role"].eq("VALID_CALIBRATE")
        ]
        bootstrap = cluster_bootstrap_metrics(
            valid_decisions,
            iterations=bootstrap_iterations,
            random_state=20260810,
        )
        gates = _quality_gates(source, decisions, selected, bootstrap, upstream)
        critical_passed = bool(
            gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all()
        )
        model_passed = bool(
            gates.loc[gates["severity"].eq("MODEL"), "passed"].all()
        )
        accepted = critical_passed and model_passed
        decision = (
            "READY_FOR_B61D_V12_FRESH_FORWARD_SHADOW"
            if accepted else "B61D_V12_NOT_ACCEPTED_KEEP_B61C_V11"
        )
        next_block = (
            "B61D_V12_FRESH_FORWARD_STATE_POLICY_VALIDATION"
            if accepted else "KEEP_B61C_V11_AND_FREEZE_B61D_V12_DIAGNOSTIC"
        )
        _update_progress(run_id, "WRITING_VERSIONED_SHADOW_ARTIFACTS")
        serving_rows = _materialize_decisions(decisions, run_id)
        selected_payload = {
            key: _json_ready(value) for key, value in selected.items()
        }
        policy_config = {
            "policy_version": POLICY_VERSION,
            "source_hsmm_version": SOURCE_HSMM_VERSION,
            "source_policy_version": SOURCE_POLICY_VERSION,
            "parameters": {
                "mode": parameters.mode,
                "critical_probability_threshold": parameters.critical_probability_threshold,
                "alert_budget_pct": parameters.alert_budget_pct,
                "hold_windows": parameters.hold_windows,
                "bucket_hours": parameters.bucket_hours,
            },
            "selection_role": "VALID_CALIBRATE",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "automatic_action_allowed": False,
        }
        report_root = f"reports/b61dv12/{OUTPUT_PREFIX}"
        model_root = f"models/b61dv12/{OUTPUT_PREFIX}"
        artifacts = {
            "scorecard": _put_csv(f"{report_root}/01_valid_candidate_scorecard.csv", scorecard),
            "selected_diagnostics": _put_csv(f"{report_root}/02_selected_role_diagnostics.csv", diagnostics),
            "quality_gates": _put_csv(f"{report_root}/03_quality_gates.csv", gates),
            "bootstrap": _put_json(f"{report_root}/04_cluster_bootstrap.json", bootstrap),
            "policy_config": _put_json(f"{model_root}/policy_config.json", policy_config),
        }
        model_card = {
            "policy_version": POLICY_VERSION,
            "architecture": (
                "FROZEN_ANCHORED_HSMM_STATE_FILTER_PLUS_B61C_SECONDARY_RANKING_"
                "CAPACITY_HYSTERESIS"
            ),
            "selected_policy": selected_payload,
            "baseline_valid": baseline["VALID_CALIBRATE"],
            "baseline_test": baseline["TEST_DIAGNOSTIC_ONLY"],
            "bootstrap_intervals": bootstrap,
            "selection_constraints": {
                "lift": ">=80% of B61C VALID lift",
                "recall": ">=90% of B61C VALID recall",
                "lead": ">=B61C+1h or >=B61C recall@6h+0.02",
                "stability": ">=0.80",
                "research_cost_reduction_pct": ">=0",
            },
            "test_disclosure": (
                "TEST was already consumed upstream and is diagnostic only; it did not "
                "select or tune this policy."
            ),
            "shadow_api_allowed": accepted,
            "production_claim_allowed": False,
            "automatic_action_allowed": False,
            "artifacts": artifacts,
        }
        artifacts["model_card"] = _put_json(
            f"{report_root}/05_model_card.json", model_card
        )
        _materialize_scorecard_and_card(
            scorecard, diagnostics, selected, baseline, bootstrap,
            gates, model_card, run_id,
        )
        metadata = {
            "policy_version": POLICY_VERSION,
            "decision": decision,
            "selected_candidate_id": selected["candidate_id"],
            "policy_mode": selected["policy_mode"],
            "critical_probability_threshold": selected["critical_probability_threshold"],
            "alert_budget_pct": selected["alert_budget_pct"],
            "hold_windows": selected["hold_windows"],
            "candidate_count": candidate_count,
            "selection_role": "VALID_CALIBRATE",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "test_used_for_fit_or_selection": False,
            "constraints_passed": bool(selected["passes_constraints"]),
            "critical_gates_passed": critical_passed,
            "model_gates_passed": model_passed,
            "shadow_api_allowed": accepted,
            "replay_allowed": accepted,
            "production_promotion_allowed": False,
            "automatic_action_allowed": False,
            "serving_rows": serving_rows,
            "bootstrap_clusters": bootstrap["clusters"],
            "next_block": next_block,
            "artifacts": artifacts,
            "progress": {
                "stage": "COMPLETE",
                "updated_at": pd.Timestamp.now(tz="UTC"),
            },
        }
        _finish_run(run_id, "SUCCESS", serving_rows, metadata)
        return _json_ready(metadata)
    except Exception as exc:
        _finish_run(
            run_id, "FAILED", None,
            {
                "policy_version": POLICY_VERSION,
                "decision": "FAILED",
                "production_promotion_allowed": False,
                "automatic_action_allowed": False,
            },
            str(exc),
        )
        raise
