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

from prefect_flows.b61c_core import (
    POLICY_VERSION,
    SOURCE_MODEL_VERSION,
    PolicyParameters,
    add_temporal_scores,
    apply_dynamic_policy,
    build_decision_grid,
    choose_policy,
    evaluate_policy,
    parse_budgets,
    policy_objective,
)


SOURCE_NAME = "b61c_historical_replay_shadow_decision"
DATASET_NAME = "maritime_port_call_dynamic_decision_shadow_v1"
SOURCE_RELATION = "serving.maritime_port_call_multitask_prediction_v21"
FEATURE_RELATION = "features.maritime_port_call_governed_v1"
DECISION_RELATION = "serving.maritime_port_call_decision_shadow_v1"
SCORECARD_RELATION = "serving.maritime_decision_policy_scorecard_v1"
POLICY_RELATION = "serving.maritime_decision_policy_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"
SOURCE_DECISION_KEY = "configs/b61bv21/version=2.1/final_decision.json"
GOVERNED_DATASET_VERSION = "b61a-governed-port-call-enrichment-v1"


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


def _relation_exists(relation: str) -> bool:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (relation,))
        return cursor.fetchone()[0] is not None


def _query_frame(query: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()
        columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _get_json(key: str) -> dict[str, Any]:
    payload = _s3_client().get_object(Bucket=OUTPUT_BUCKET, Key=key)["Body"].read()
    return json.loads(payload.decode("utf-8"))


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


def load_replay_source() -> pd.DataFrame:
    source_decision = _get_json(SOURCE_DECISION_KEY)
    if source_decision.get("model_version") != SOURCE_MODEL_VERSION:
        raise RuntimeError("B61B-v2.1 source model version mismatch")
    if not source_decision.get("replay_allowed", False):
        raise RuntimeError("B61B-v2.1 did not authorize historical replay")
    frame = _query_frame(
        f"""
        SELECT
            p.port_call_id,
            p.landmark_at,
            p.split,
            p.evaluation_role,
            p.regime,
            p.p_delay_gt1,
            p.p_delay_gt3,
            p.p_delay_gt6,
            p.remaining_p10_h,
            p.remaining_p50_h,
            p.remaining_p90_h,
            p.p_gt3_breach_within_6h,
            p.p_gt3_breach_within_12h,
            p.p_gt3_breach_within_24h,
            g.early_warning_eligible,
            g.pre_breach_eligible,
            g.target_delay_gt_3h,
            g.target_delay_gt_6h,
            g.target_breach_gt3_observed,
            g.target_breach_or_censor_h,
            g.per_call_sample_weight
        FROM {SOURCE_RELATION} p
        JOIN {FEATURE_RELATION} g
          ON g.port_call_id=p.port_call_id
         AND g.landmark_at=p.landmark_at
        WHERE p.model_version=%s
          AND g.dataset_version=%s
          AND p.evaluation_role IN (
              'VALID_SELECT', 'VALID_CALIBRATE', 'TEST_DIAGNOSTIC_ONLY'
          )
          AND g.synthetic_row=false
          AND g.targets_imputed=false
        ORDER BY p.evaluation_role, p.landmark_at, p.port_call_id
        """,
        (SOURCE_MODEL_VERSION, GOVERNED_DATASET_VERSION),
    )
    if frame.empty:
        raise RuntimeError("B61C source join returned no rows")
    if frame.duplicated(["port_call_id", "landmark_at"]).any():
        raise RuntimeError("B61C source contains duplicate port-call landmarks")
    for column in (
        "target_delay_gt_3h",
        "target_delay_gt_6h",
        "early_warning_eligible",
        "pre_breach_eligible",
    ):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def _candidate_scorecard(
    valid_grid: pd.DataFrame,
    gt3_budgets: list[float],
    gt6_budgets: list[float],
    bucket_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for gt3_budget in gt3_budgets:
        for gt6_budget in gt6_budgets:
            if gt6_budget > gt3_budget:
                continue
            watch_budget = min(25.0, gt3_budget * 2.0)
            parameters = PolicyParameters(
                gt3_budget_pct=gt3_budget,
                gt6_budget_pct=gt6_budget,
                bucket_hours=bucket_hours,
                high_score_threshold=float(
                    valid_grid["temporal_priority_score"].quantile(
                        1.0 - gt3_budget / 100.0,
                        interpolation="higher",
                    )
                ),
                critical_score_threshold=float(
                    valid_grid["critical_priority_score"].quantile(
                        1.0 - gt6_budget / 100.0,
                        interpolation="higher",
                    )
                ),
                watch_score_threshold=float(
                    valid_grid["temporal_priority_score"].quantile(
                        1.0 - watch_budget / 100.0,
                        interpolation="higher",
                    )
                ),
            )
            decisions = apply_dynamic_policy(valid_grid, parameters)
            metrics = evaluate_policy(decisions, parameters)
            metrics["objective"] = policy_objective(metrics)
            rows.append(metrics)
    scorecard = pd.DataFrame(rows)
    selected = choose_policy(scorecard)
    scorecard["selected"] = scorecard["policy_id"].eq(selected["policy_id"])
    return scorecard, selected


def _selected_replay(
    grid: pd.DataFrame,
    selected: dict[str, Any],
    bucket_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, PolicyParameters]:
    parameters = PolicyParameters(
        gt3_budget_pct=float(selected["gt3_budget_pct"]),
        gt6_budget_pct=float(selected["gt6_budget_pct"]),
        bucket_hours=bucket_hours,
        high_score_threshold=float(selected["high_score_threshold"]),
        critical_score_threshold=float(selected["critical_score_threshold"]),
        watch_score_threshold=float(selected["watch_score_threshold"]),
    )
    decisions = apply_dynamic_policy(grid, parameters)
    diagnostics = []
    for role, role_frame in decisions.groupby("evaluation_role", sort=False):
        metrics = evaluate_policy(role_frame, parameters)
        metrics["objective"] = policy_objective(metrics)
        diagnostics.append(metrics)
    return decisions, pd.DataFrame(diagnostics), parameters


def _quality_gates(
    source: pd.DataFrame,
    decisions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    parameters: PolicyParameters,
) -> pd.DataFrame:
    valid = diagnostics.loc[diagnostics["role"].eq("VALID_CALIBRATE")]
    if valid.empty:
        raise RuntimeError("Selected policy has no VALID diagnostics")
    valid_row = valid.iloc[0]
    buckets = list(
        decisions.groupby(["evaluation_role", "decision_at"], sort=False)
    )
    high_capacity_ok = all(
        int(item["alert_active"].sum()) <= int(item["high_capacity"].iloc[0])
        for _, item in buckets
    )
    critical_capacity_ok = all(
        int(item["state"].eq("CRITICAL").sum())
        <= int(item["critical_capacity"].iloc[0])
        for _, item in buckets
    )
    rows = [
        {"check": "SOURCE_B61BV21_REPLAY_ALLOWED", "passed": True, "severity": "CRITICAL", "value": True},
        {"check": "NO_PREDICTIVE_MODEL_TRAINING", "passed": True, "severity": "CRITICAL", "value": False},
        {"check": "POLICY_SELECTED_ON_VALID_ONLY", "passed": True, "severity": "CRITICAL", "value": "VALID_CALIBRATE"},
        {"check": "TEST_NOT_USED_FOR_POLICY_SELECTION", "passed": True, "severity": "CRITICAL", "value": False},
        {"check": "TEST_REUSED_DIAGNOSTIC_NOT_CONFIRMATORY", "passed": True, "severity": "CRITICAL", "value": True},
        {"check": "SOURCE_PORT_CALL_LANDMARK_UNIQUE", "passed": not source.duplicated(["port_call_id", "landmark_at"]).any(), "severity": "CRITICAL", "value": None},
        {"check": "DECISION_CALL_BUCKET_UNIQUE", "passed": not decisions.duplicated(["evaluation_role", "decision_at", "port_call_id"]).any(), "severity": "CRITICAL", "value": None},
        {"check": "TEMPORAL_ORDER_PRESERVED", "passed": bool(decisions.groupby(["evaluation_role", "port_call_id"])["decision_at"].apply(lambda values: values.is_monotonic_increasing).all()), "severity": "CRITICAL", "value": None},
        {"check": "FINITE_TEMPORAL_SCORES", "passed": bool(np.isfinite(decisions[["temporal_priority_score", "critical_priority_score"]].to_numpy()).all()), "severity": "CRITICAL", "value": None},
        {"check": "GT3_CAPACITY_RESPECTED_EACH_BUCKET", "passed": high_capacity_ok, "severity": "CRITICAL", "value": parameters.gt3_budget_pct},
        {"check": "GT6_CAPACITY_RESPECTED_EACH_BUCKET", "passed": critical_capacity_ok, "severity": "CRITICAL", "value": parameters.gt6_budget_pct},
        {"check": "AUTOMATIC_ACTION_EXECUTION_BLOCKED", "passed": bool(~decisions["automatic_action_allowed"].any()), "severity": "CRITICAL", "value": False},
        {"check": "VALID_GT3_ALERTS_NONZERO", "passed": int(valid_row["alert_rows"]) > 0, "severity": "POLICY", "value": int(valid_row["alert_rows"])},
        {"check": "VALID_GT6_ALERTS_NONZERO", "passed": int(valid_row["critical_rows"]) > 0, "severity": "POLICY", "value": int(valid_row["critical_rows"])},
        {"check": "VALID_GT3_RECALL_NONZERO", "passed": float(valid_row["gt3_recall"]) > 0.0, "severity": "POLICY", "value": float(valid_row["gt3_recall"])},
        {"check": "VALID_GT6_RECALL_NONZERO", "passed": float(valid_row["gt6_recall"]) > 0.0, "severity": "POLICY", "value": float(valid_row["gt6_recall"])},
        {"check": "VALID_GT3_PRECISION_LIFT_ABOVE_ONE", "passed": float(valid_row["gt3_precision_lift"]) > 1.0, "severity": "POLICY", "value": float(valid_row["gt3_precision_lift"])},
        {"check": "VALID_GT6_PRECISION_LIFT_ABOVE_ONE", "passed": float(valid_row["gt6_precision_lift"]) > 1.0, "severity": "POLICY", "value": float(valid_row["gt6_precision_lift"])},
        {"check": "VALID_POLICY_COST_REDUCTION_POSITIVE", "passed": float(valid_row["cost_reduction_pct"]) > 0.0, "severity": "POLICY", "value": float(valid_row["cost_reduction_pct"])},
        {"check": "VALID_STATE_STABILITY_ABOVE_50PCT", "passed": float(valid_row["state_stability"]) >= 0.50, "severity": "POLICY", "value": float(valid_row["state_stability"])},
        {"check": "FRESH_FORWARD_SHADOW_REQUIRED", "passed": True, "severity": "GOVERNANCE", "value": True},
        {"check": "PRODUCTION_PROMOTION_BLOCKED", "passed": True, "severity": "GOVERNANCE", "value": False},
    ]
    return pd.DataFrame(rows)


def _decision_output(decisions: pd.DataFrame, run_id: str) -> pd.DataFrame:
    columns = [
        "port_call_id",
        "landmark_at",
        "decision_at",
        "split",
        "evaluation_role",
        "regime",
        "temporal_priority_score",
        "critical_priority_score",
        "rank_in_bucket",
        "active_calls",
        "high_capacity",
        "critical_capacity",
        "state",
        "previous_state",
        "state_changed",
        "alert_active",
        "new_alert",
        "action_code",
        "p_delay_gt1",
        "p_delay_gt3",
        "p_delay_gt6",
        "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h",
        "remaining_p10_h",
        "remaining_p50_h",
        "remaining_p90_h",
        "target_delay_gt_3h",
        "target_delay_gt_6h",
        "target_breach_or_censor_h",
        "decision_weight",
        "gt3_budget_pct",
        "gt6_budget_pct",
        "policy_id",
        "source_mode",
        "production_claim_allowed",
        "automatic_action_allowed",
    ]
    output = decisions[columns].copy()
    output.insert(0, "policy_version", POLICY_VERSION)
    output.insert(1, "source_model_version", SOURCE_MODEL_VERSION)
    output["materialization_run_id"] = run_id
    return output


def _materialize_decisions(frame: pd.DataFrame) -> int:
    columns = list(frame.columns)
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {DECISION_RELATION} (
                policy_version TEXT NOT NULL,
                source_model_version TEXT NOT NULL,
                port_call_id TEXT NOT NULL,
                landmark_at TIMESTAMPTZ NOT NULL,
                decision_at TIMESTAMPTZ NOT NULL,
                split TEXT NOT NULL,
                evaluation_role TEXT NOT NULL,
                regime TEXT NOT NULL,
                temporal_priority_score DOUBLE PRECISION NOT NULL,
                critical_priority_score DOUBLE PRECISION NOT NULL,
                rank_in_bucket INTEGER NOT NULL,
                active_calls INTEGER NOT NULL,
                high_capacity INTEGER NOT NULL,
                critical_capacity INTEGER NOT NULL,
                state TEXT NOT NULL,
                previous_state TEXT NOT NULL,
                state_changed BOOLEAN NOT NULL,
                alert_active BOOLEAN NOT NULL,
                new_alert BOOLEAN NOT NULL,
                action_code TEXT NOT NULL,
                p_delay_gt1 DOUBLE PRECISION NOT NULL,
                p_delay_gt3 DOUBLE PRECISION NOT NULL,
                p_delay_gt6 DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_6h DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_12h DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_24h DOUBLE PRECISION NOT NULL,
                remaining_p10_h DOUBLE PRECISION NOT NULL,
                remaining_p50_h DOUBLE PRECISION NOT NULL,
                remaining_p90_h DOUBLE PRECISION NOT NULL,
                target_delay_gt_3h BOOLEAN NOT NULL,
                target_delay_gt_6h BOOLEAN NOT NULL,
                target_breach_or_censor_h DOUBLE PRECISION,
                decision_weight DOUBLE PRECISION NOT NULL,
                gt3_budget_pct DOUBLE PRECISION NOT NULL,
                gt6_budget_pct DOUBLE PRECISION NOT NULL,
                policy_id TEXT NOT NULL,
                source_mode TEXT NOT NULL,
                production_claim_allowed BOOLEAN NOT NULL,
                automatic_action_allowed BOOLEAN NOT NULL,
                materialization_run_id TEXT NOT NULL,
                PRIMARY KEY (policy_version, evaluation_role, decision_at, port_call_id)
            )
            """
        )
        cursor.execute(
            f"DELETE FROM {DECISION_RELATION} WHERE policy_version=%s",
            (POLICY_VERSION,),
        )
        column_sql = ", ".join(columns)
        for start in range(0, len(frame), 2_000):
            records = list(
                frame.iloc[start : start + 2_000].itertuples(index=False, name=None)
            )
            execute_values(
                cursor,
                f"INSERT INTO {DECISION_RELATION} ({column_sql}) VALUES %s",
                records,
                page_size=2_000,
            )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_b61c_decision_snapshot "
            f"ON {DECISION_RELATION} (decision_at DESC, state, rank_in_bucket)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_b61c_decision_call "
            f"ON {DECISION_RELATION} (port_call_id, decision_at)"
        )
    return len(frame)


def _materialize_policy(
    scorecard: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected: dict[str, Any],
    gates: pd.DataFrame,
    run_id: str,
) -> None:
    selected_policy_id = str(selected["policy_id"])
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCORECARD_RELATION} (
                policy_version TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                role TEXT NOT NULL,
                gt3_budget_pct DOUBLE PRECISION NOT NULL,
                gt6_budget_pct DOUBLE PRECISION NOT NULL,
                objective DOUBLE PRECISION,
                selected BOOLEAN NOT NULL,
                metrics JSONB NOT NULL,
                materialization_run_id TEXT NOT NULL,
                PRIMARY KEY (policy_version, policy_id, role)
            )
            """
        )
        cursor.execute(
            f"DELETE FROM {SCORECARD_RELATION} WHERE policy_version=%s",
            (POLICY_VERSION,),
        )
        records = []
        for row in scorecard.itertuples(index=False):
            payload = row._asdict()
            records.append(
                (
                    POLICY_VERSION,
                    str(payload["policy_id"]),
                    str(payload["role"]),
                    float(payload["gt3_budget_pct"]),
                    float(payload["gt6_budget_pct"]),
                    float(payload["objective"]),
                    bool(payload["selected"]),
                    Json(_json_ready(payload)),
                    run_id,
                )
            )
        for row in diagnostics.itertuples(index=False):
            payload = row._asdict()
            records.append(
                (
                    POLICY_VERSION,
                    selected_policy_id,
                    str(payload["role"]),
                    float(payload["gt3_budget_pct"]),
                    float(payload["gt6_budget_pct"]),
                    float(payload["objective"]),
                    True,
                    Json(_json_ready(payload)),
                    run_id,
                )
            )
        deduplicated = {}
        for record in records:
            deduplicated[(record[0], record[1], record[2])] = record
        execute_values(
            cursor,
            f"INSERT INTO {SCORECARD_RELATION} VALUES %s",
            list(deduplicated.values()),
            page_size=500,
        )
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {POLICY_RELATION} (
                policy_version TEXT PRIMARY KEY,
                source_model_version TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                selected_on_role TEXT NOT NULL,
                parameters JSONB NOT NULL,
                quality_gates JSONB NOT NULL,
                source_mode TEXT NOT NULL,
                production_claim_allowed BOOLEAN NOT NULL,
                automatic_action_allowed BOOLEAN NOT NULL,
                fresh_forward_confirmation_required BOOLEAN NOT NULL,
                materialization_run_id TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(f"DELETE FROM {POLICY_RELATION} WHERE policy_version=%s", (POLICY_VERSION,))
        cursor.execute(
            f"""
            INSERT INTO {POLICY_RELATION}
            VALUES (%s, %s, %s, %s, %s, %s, %s, false, false, true, %s, now())
            """,
            (
                POLICY_VERSION,
                SOURCE_MODEL_VERSION,
                selected_policy_id,
                "VALID_CALIBRATE",
                Json(_json_ready(selected)),
                Json(_json_ready(gates.to_dict("records"))),
                "HISTORICAL_REPLAY_SHADOW",
                run_id,
            ),
        )


def _source_signature(
    source: pd.DataFrame,
    gt3_budgets: list[float],
    gt6_budgets: list[float],
    bucket_hours: int,
) -> str:
    digest = hashlib.sha256(POLICY_VERSION.encode("ascii"))
    digest.update(str((gt3_budgets, gt6_budgets, bucket_hours)).encode("ascii"))
    digest.update(
        pd.util.hash_pandas_object(
            source[
                [
                    "port_call_id",
                    "landmark_at",
                    "p_delay_gt3",
                    "p_delay_gt6",
                    "target_delay_gt_3h",
                    "target_delay_gt_6h",
                ]
            ],
            index=False,
        ).to_numpy(dtype="uint64").tobytes()
    )
    return digest.hexdigest()


def _previous_success(checksum: str) -> dict[str, Any] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT metadata
            FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s AND checksum=%s
              AND status='SUCCESS'
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (SOURCE_NAME, DATASET_NAME, checksum),
        )
        row = cursor.fetchone()
        if row is None or not _relation_exists(DECISION_RELATION):
            return None
        cursor.execute(
            f"SELECT COUNT(*) FROM {DECISION_RELATION} WHERE policy_version=%s",
            (POLICY_VERSION,),
        )
        return dict(row[0]) if int(cursor.fetchone()[0]) > 0 else None


def _start_run(checksum: str) -> str:
    metadata = {
        "policy_version": POLICY_VERSION,
        "source_model_version": SOURCE_MODEL_VERSION,
        "orchestrator": "PREFECT",
        "model_training_executed": False,
        "policy_selection_role": "VALID_CALIBRATE",
        "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
        "test_used_for_policy_selection": False,
        "production_promotion_allowed": False,
        "automatic_action_allowed": False,
    }
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO audit.ingestion_run
                (source_name, dataset_name, object_uri, checksum, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING run_id
            """,
            (
                SOURCE_NAME,
                DATASET_NAME,
                f"postgresql://maritime/{DECISION_RELATION}",
                checksum,
                Json(metadata),
            ),
        )
        return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    payload = _json_ready(
        {"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details}
    )
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
            SET finished_at=now(), status=%s, row_count=%s,
                metadata=metadata || %s, error_message=%s
            WHERE run_id=%s
            """,
            (status, row_count, Json(_json_ready(metadata)), error_message, run_id),
        )


def run_b61c_replay(
    force: bool = False,
    gt3_budgets: str = "1,2,3,5",
    gt6_budgets: str = "0.25,0.5,1",
    bucket_hours: int = 6,
) -> dict[str, Any]:
    if bucket_hours not in (1, 3, 6, 12):
        raise ValueError("bucket_hours must be 1, 3, 6, or 12")
    for relation in (SOURCE_RELATION, FEATURE_RELATION, "audit.ingestion_run"):
        if not _relation_exists(relation):
            raise RuntimeError(f"Required B61C relation is missing: {relation}")
    gt3_values = parse_budgets(gt3_budgets)
    gt6_values = parse_budgets(gt6_budgets)
    source = load_replay_source()
    checksum = _source_signature(source, gt3_values, gt6_values, bucket_hours)
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            return {**previous, "reused": True}
    run_id = _start_run(checksum)
    try:
        _update_progress(
            run_id,
            "BUILDING_TEMPORAL_DECISION_GRID",
            source_rows=len(source),
            source_calls=source["port_call_id"].nunique(),
        )
        grid = add_temporal_scores(build_decision_grid(source, bucket_hours))
        valid_grid = grid.loc[grid["evaluation_role"].eq("VALID_CALIBRATE")].copy()
        if valid_grid.empty:
            raise RuntimeError("VALID_CALIBRATE decision grid is empty")
        _update_progress(
            run_id,
            "REPLAYING_CAPACITY_CONSTRAINED_POLICIES_ON_VALID",
            candidates=len(gt3_values) * len(gt6_values),
        )
        scorecard, selected = _candidate_scorecard(
            valid_grid, gt3_values, gt6_values, bucket_hours
        )
        decisions, diagnostics, parameters = _selected_replay(
            grid, selected, bucket_hours
        )
        gates = _quality_gates(source, decisions, diagnostics, parameters)
        critical_passed = bool(
            gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all()
        )
        policy_passed = bool(
            gates.loc[gates["severity"].eq("POLICY"), "passed"].all()
        )
        shadow_api_allowed = critical_passed
        dynamic_alert_shadow_allowed = critical_passed and policy_passed
        output = _decision_output(decisions, run_id)
        decision_rows = _materialize_decisions(output)
        _materialize_policy(scorecard, diagnostics, selected, gates, run_id)
        if dynamic_alert_shadow_allowed:
            decision = "READY_FOR_DYNAMIC_ALERT_SHADOW_API"
        elif shadow_api_allowed:
            decision = "READY_FOR_SCORE_ONLY_SHADOW_API_POLICY_REFINEMENT_REQUIRED"
        else:
            decision = "RESEARCH_ONLY_B61C_INTEGRITY_REFINEMENT_REQUIRED"
        metadata = {
            "decision": decision,
            "policy_version": POLICY_VERSION,
            "source_model_version": SOURCE_MODEL_VERSION,
            "row_count": int(len(source)),
            "source_calls": int(source["port_call_id"].nunique()),
            "decision_rows": int(decision_rows),
            "decision_calls": int(decisions["port_call_id"].nunique()),
            "decision_buckets": int(decisions["decision_at"].nunique()),
            "selected_policy_id": parameters.policy_id,
            "gt3_budget_pct": parameters.gt3_budget_pct,
            "gt6_budget_pct": parameters.gt6_budget_pct,
            "bucket_hours": bucket_hours,
            "temporal_model": "B61BV21_DISCRETE_HAZARD_6_12_24H_ENSEMBLE",
            "temporal_state_machine": "CAPACITY_CONSTRAINED_HYSTERETIC_MULTI_STATE_V1",
            "model_training_executed": False,
            "policy_selection_role": "VALID_CALIBRATE",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "test_used_for_policy_selection": False,
            "critical_gates_passed": critical_passed,
            "policy_gates_passed": policy_passed,
            "shadow_api_allowed": shadow_api_allowed,
            "dynamic_alert_shadow_allowed": dynamic_alert_shadow_allowed,
            "production_promotion_allowed": False,
            "automatic_action_allowed": False,
            "fresh_forward_confirmation_required": True,
            "source_mode": "HISTORICAL_REPLAY_SHADOW",
            "limitations": [
                "Policy costs are normalized research units, not approved enterprise costs.",
                "The previous TEST window remains diagnostic and non-confirmatory.",
                "Actions are recommendations only and cannot be executed automatically.",
                "Fresh forward shadow outcomes are required before production promotion.",
            ],
            "next_block": (
                "B61C_FRESH_FORWARD_SHADOW_MONITORING"
                if dynamic_alert_shadow_allowed
                else "B61C_POLICY_COST_AND_CAPACITY_REFINEMENT"
            ),
        }
        reports = {
            "policy_candidate_scorecard": scorecard,
            "selected_policy_diagnostics": diagnostics,
            "quality_gates": gates,
            "state_distribution": decisions.groupby(
                ["evaluation_role", "state"], as_index=False
            ).agg(rows=("port_call_id", "size"), calls=("port_call_id", "nunique")),
            "transition_matrix": decisions.groupby(
                ["evaluation_role", "previous_state", "state"], as_index=False
            ).agg(transitions=("port_call_id", "size")),
        }
        _update_progress(run_id, "WRITING_B61C_REPLAY_AND_POLICY_ARTIFACTS")
        for name, report in reports.items():
            _put_csv(f"reports/b61c/{OUTPUT_PREFIX}/{name}.csv", report)
        _put_json(f"configs/b61c/{OUTPUT_PREFIX}/selected_policy.json", selected)
        _put_json(f"configs/b61c/{OUTPUT_PREFIX}/final_decision.json", metadata)
        _finish_run(run_id, "SUCCESS", len(source), metadata)
        return metadata
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "decision": "FAILED",
                "model_training_executed": False,
                "test_used_for_policy_selection": False,
                "production_promotion_allowed": False,
                "automatic_action_allowed": False,
                "next_block": "FIX_B61C_AND_RERUN",
            },
            str(exc),
        )
        raise


def verify_b61c_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "decision",
        "policy_version",
        "source_model_version",
        "decision_rows",
        "selected_policy_id",
        "model_training_executed",
        "policy_selection_role",
        "test_role",
        "test_used_for_policy_selection",
        "shadow_api_allowed",
        "production_promotion_allowed",
        "automatic_action_allowed",
        "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"B61C result misses required fields: {missing}")
    if result["model_training_executed"]:
        raise ValueError("B61C must not retrain the predictive model")
    if result["test_used_for_policy_selection"]:
        raise ValueError("B61C policy leakage contract violated")
    if result["policy_selection_role"] != "VALID_CALIBRATE":
        raise ValueError("B61C policy must be selected on VALID_CALIBRATE")
    if result["test_role"] != "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY":
        raise ValueError("B61C must disclose TEST reuse")
    if result["production_promotion_allowed"] or result["automatic_action_allowed"]:
        raise ValueError("B61C cannot promote or execute actions")
    if int(result["decision_rows"]) <= 0:
        raise ValueError("B61C produced no decision rows")
    return result
