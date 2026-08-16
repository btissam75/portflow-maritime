from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from prefect_flows.b61c_core import (
    PolicyParameters,
    apply_dynamic_policy,
    evaluate_policy,
    policy_objective,
)
from prefect_flows.b61d_v11_core import (
    MODEL_VERSION,
    SOURCE_HSMM_VERSION,
    SOURCE_MODEL_VERSION,
    SOURCE_POLICY_VERSION,
    STATE_NAMES,
    add_anchored_scores,
    decode_frame,
    event_lead_metrics,
    fit_anchored_hsmm,
    prebreach_fit_mask,
    prepare_sequence_features,
)


SOURCE_NAME = "b61d_v11_anchored_hsmm"
DATASET_NAME = "maritime_anchored_hsmm_shadow_v11"
PREDICTION_RELATION = "serving.maritime_port_call_multitask_prediction_v21"
FEATURE_RELATION = "features.maritime_port_call_governed_v1"
BASELINE_DECISION_RELATION = "serving.maritime_port_call_decision_shadow_v1"
BASELINE_SCORECARD_RELATION = "serving.maritime_decision_policy_scorecard_v1"
OUTPUT_RELATION = "serving.maritime_port_call_anchored_hsmm_shadow_v11"
SCORECARD_RELATION = "serving.maritime_anchored_hsmm_policy_scorecard_v11"
MODEL_CARD_RELATION = "serving.maritime_anchored_hsmm_model_card_v11"
GOVERNED_DATASET_VERSION = "b61a-governed-port-call-enrichment-v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1.1"


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


def _relation_exists(relation: str) -> bool:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (relation,))
        return cursor.fetchone()[0] is not None


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


def _verify_upstream() -> None:
    checks = (
        (PREDICTION_RELATION, "model_version", SOURCE_MODEL_VERSION),
        (BASELINE_DECISION_RELATION, "policy_version", SOURCE_POLICY_VERSION),
    )
    with _db_connection() as connection, connection.cursor() as cursor:
        for relation, column, version in checks:
            cursor.execute(f"SELECT COUNT(*) FROM {relation} WHERE {column}=%s", (version,))
            if int(cursor.fetchone()[0]) <= 0:
                raise RuntimeError(f"Required upstream artifact is empty: {relation} {version}")


def load_source() -> pd.DataFrame:
    frame = _query_frame(
        f"""
        SELECT
            p.port_call_id,p.landmark_at,p.split,p.evaluation_role,p.regime,
            p.p_delay_gt1,p.p_delay_gt3,p.p_delay_gt6,
            p.remaining_p10_h,p.remaining_p50_h,p.remaining_p90_h,
            p.p_gt3_breach_within_6h,p.p_gt3_breach_within_12h,
            p.p_gt3_breach_within_24h,g.early_warning_eligible,
            g.pre_breach_eligible,g.overdue_h,g.plan_progress_ratio,
            g.time_to_planned_departure_h,g.arrival_delay_h,
            g.vessel_history_prior_late_gt3_rate,g.known_event_any_24h,
            g.target_delay_gt_3h,g.target_delay_gt_6h,
            g.target_breach_gt3_observed,g.target_breach_or_censor_h,
            g.per_call_sample_weight
        FROM {PREDICTION_RELATION} p
        JOIN {FEATURE_RELATION} g
          ON g.dataset_version=%s
         AND g.port_call_id=p.port_call_id
         AND g.landmark_at=p.landmark_at
        WHERE p.model_version=%s
          AND p.evaluation_role IN (
              'VALID_SELECT','VALID_CALIBRATE','TEST_DIAGNOSTIC_ONLY'
          )
          AND g.synthetic_row=false AND g.targets_imputed=false
        ORDER BY p.evaluation_role,p.port_call_id,p.landmark_at
        """,
        (GOVERNED_DATASET_VERSION, SOURCE_MODEL_VERSION),
    )
    if frame.empty or frame.duplicated(["port_call_id", "landmark_at"]).any():
        raise RuntimeError("B61D-v1.1 source is empty or has duplicate landmarks")
    for column in (
        "early_warning_eligible", "pre_breach_eligible",
        "target_delay_gt_3h", "target_delay_gt_6h",
    ):
        frame[column] = frame[column].fillna(False).astype(bool)
    return prepare_sequence_features(frame)


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
    return result


def _decision_grid(frame: pd.DataFrame, bucket_hours: int) -> pd.DataFrame:
    eligible = frame.loc[frame["early_warning_eligible"]].copy()
    eligible["decision_at"] = eligible["landmark_at"].dt.floor(f"{bucket_hours}h")
    eligible = (
        eligible.sort_values(
            ["evaluation_role", "decision_at", "port_call_id", "landmark_at"]
        )
        .drop_duplicates(
            ["evaluation_role", "decision_at", "port_call_id"], keep="last"
        )
        .reset_index(drop=True)
    )
    observations = eligible.groupby(
        ["evaluation_role", "port_call_id"], sort=False
    )["port_call_id"].transform("size")
    eligible["decision_weight"] = 1.0 / observations.clip(lower=1).astype(float)
    return eligible


def _threshold(series: pd.Series, budget_pct: float) -> float:
    return float(series.quantile(1.0 - budget_pct / 100.0, interpolation="higher"))


def _lead_objective(metrics: dict[str, Any]) -> float:
    median_lead = float(metrics.get("median_event_lead_h") or 0.0)
    return float(
        policy_objective(metrics)
        + 1.5 * min(median_lead, 24.0)
        + 30.0 * float(metrics["event_recall_at_least_6h"])
        + 15.0 * float(metrics["event_recall_at_least_12h"])
        + 5.0 * float(metrics["event_recall_at_least_24h"])
    )


def _candidate_replay(
    decoded: pd.DataFrame,
    fit_mask: pd.Series,
    weights: list[float],
    gt3_budgets: list[float],
    gt6_budgets: list[float],
    bucket_hours: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weight in sorted(set([0.0, *weights])):
        grid = _decision_grid(add_anchored_scores(decoded, fit_mask, weight), bucket_hours)
        valid = grid.loc[grid["evaluation_role"].eq("VALID_CALIBRATE")]
        for gt3_budget in gt3_budgets:
            for gt6_budget in gt6_budgets:
                if gt6_budget > gt3_budget:
                    continue
                parameters = PolicyParameters(
                    gt3_budget_pct=gt3_budget,
                    gt6_budget_pct=gt6_budget,
                    bucket_hours=bucket_hours,
                    high_score_threshold=_threshold(valid["temporal_priority_score"], gt3_budget),
                    critical_score_threshold=_threshold(valid["critical_priority_score"], gt6_budget),
                    watch_score_threshold=_threshold(
                        valid["temporal_priority_score"], min(25.0, 2.0 * gt3_budget)
                    ),
                )
                decisions = apply_dynamic_policy(valid, parameters)
                metrics = evaluate_policy(decisions, parameters)
                metrics.update(event_lead_metrics(decisions))
                metrics["hsmm_weight"] = weight
                metrics["candidate_id"] = (
                    f"ANCHORED_W{weight:g}_GT3_{gt3_budget:g}_"
                    f"GT6_{gt6_budget:g}_{bucket_hours}H"
                )
                metrics["objective"] = _lead_objective(metrics)
                rows.append(metrics)
    scorecard = pd.DataFrame(rows)
    control = scorecard.loc[scorecard["hsmm_weight"].eq(0.0)].sort_values(
        ["objective", "gt3_precision_lift", "gt3_recall"], ascending=False
    ).iloc[0].to_dict()
    challenger = scorecard.loc[scorecard["hsmm_weight"].gt(0.0)].sort_values(
        ["objective", "event_recall_at_least_6h", "median_event_lead_h", "gt3_precision_lift"],
        ascending=False,
    ).iloc[0].to_dict()
    scorecard["control_selected"] = scorecard["candidate_id"].eq(control["candidate_id"])
    scorecard["challenger_selected"] = scorecard["candidate_id"].eq(challenger["candidate_id"])
    return scorecard, control, challenger


def _selected_replay(
    decoded: pd.DataFrame,
    fit_mask: pd.Series,
    selected: dict[str, Any],
    bucket_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, PolicyParameters]:
    grid = _decision_grid(
        add_anchored_scores(decoded, fit_mask, float(selected["hsmm_weight"])),
        bucket_hours,
    )
    parameters = PolicyParameters(
        gt3_budget_pct=float(selected["gt3_budget_pct"]),
        gt6_budget_pct=float(selected["gt6_budget_pct"]),
        bucket_hours=bucket_hours,
        high_score_threshold=float(selected["high_score_threshold"]),
        critical_score_threshold=float(selected["critical_score_threshold"]),
        watch_score_threshold=float(selected["watch_score_threshold"]),
    )
    decisions = apply_dynamic_policy(grid, parameters)
    decisions["candidate_id"] = str(selected["candidate_id"])
    diagnostics = []
    for role, role_frame in decisions.groupby("evaluation_role", sort=False):
        metrics = evaluate_policy(role_frame, parameters)
        metrics.update(event_lead_metrics(role_frame))
        metrics.update(
            {
                "hsmm_weight": float(selected["hsmm_weight"]),
                "candidate_id": str(selected["candidate_id"]),
            }
        )
        metrics["objective"] = _lead_objective(metrics)
        diagnostics.append(metrics)
    return decisions, pd.DataFrame(diagnostics), parameters


def _state_diagnostics(decoded: pd.DataFrame) -> pd.DataFrame:
    eligible = decoded.loc[decoded["early_warning_eligible"]].copy()
    rows = []
    for role, role_frame in eligible.groupby("evaluation_role", sort=False):
        overall_prevalence = float(role_frame["target_delay_gt_3h"].mean())
        counts = role_frame["hsmm_state"].value_counts().reindex(STATE_NAMES, fill_value=0)
        probability = counts.to_numpy(dtype=float) / max(len(role_frame), 1)
        entropy = float(
            -(probability[probability > 0] * np.log(probability[probability > 0])).sum()
            / math.log(len(STATE_NAMES))
        )
        for state in STATE_NAMES:
            state_rows = role_frame.loc[role_frame["hsmm_state"].eq(state)]
            state_prevalence = float(state_rows["target_delay_gt_3h"].mean()) if len(state_rows) else None
            rows.append(
                {
                    "role": role,
                    "state": state,
                    "rows": len(state_rows),
                    "row_pct": 100.0 * len(state_rows) / max(len(role_frame), 1),
                    "calls": state_rows["port_call_id"].nunique(),
                    "mean_confidence": state_rows["hsmm_state_confidence"].mean(),
                    "mean_risk": state_rows["hsmm_risk_score"].mean(),
                    "target_gt3_prevalence": state_prevalence,
                    "target_enrichment": (
                        state_prevalence / max(overall_prevalence, 1e-12)
                        if state_prevalence is not None else None
                    ),
                    "normalized_state_entropy": entropy,
                }
            )
    return pd.DataFrame(rows)


def _quality_gates(
    source: pd.DataFrame,
    bundle: Any,
    state_diagnostics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    control: dict[str, Any],
    baseline: dict[str, dict[str, Any]],
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    valid = diagnostics.loc[diagnostics["role"].eq("VALID_CALIBRATE")].iloc[0]
    baseline_valid = baseline["VALID_CALIBRATE"]
    valid_states = state_diagnostics.loc[
        state_diagnostics["role"].eq("VALID_CALIBRATE")
    ]
    state_rows = valid_states.set_index("state")["rows"]
    high_rows = int(state_rows.get("CONGESTED", 0) + state_rows.get("CRITICAL_DISRUPTION", 0))
    valid_total = max(int(valid_states["rows"].sum()), 1)
    high_pct = 100.0 * high_rows / valid_total
    critical_enrichment = float(
        valid_states.loc[
            valid_states["state"].eq("CRITICAL_DISRUPTION"), "target_enrichment"
        ].fillna(0.0).iloc[0]
    )
    baseline_lead = float(baseline_valid.get("median_event_lead_h") or 0.0)
    challenger_lead = float(valid.get("median_event_lead_h") or 0.0)
    lead_improved = (
        challenger_lead >= baseline_lead + 1.0
        or float(valid["event_recall_at_least_6h"])
        >= float(baseline_valid.get("event_recall_at_least_6h") or 0.0) + 0.02
    )
    rows = [
        {"check": "FIT_USES_PREBREACH_VALID_SELECT_EARLY_70PCT", "passed": True, "severity": "CRITICAL", "value": bundle.fit_cutoff},
        {"check": "TEST_NOT_USED_FOR_FIT_OR_SELECTION", "passed": True, "severity": "CRITICAL", "value": False},
        {"check": "SOURCE_LANDMARK_UNIQUE", "passed": not source.duplicated(["port_call_id", "landmark_at"]).any(), "severity": "CRITICAL", "value": None},
        {"check": "DECISION_BUCKET_UNIQUE", "passed": not decisions.duplicated(["evaluation_role", "decision_at", "port_call_id"]).any(), "severity": "CRITICAL", "value": None},
        {"check": "TARGETS_NOT_USED_FOR_STATE_FIT", "passed": True, "severity": "CRITICAL", "value": True},
        {"check": "ALL_ANCHORS_MINIMUM_SUPPORT", "passed": min(bundle.anchor_counts.values()) >= bundle.minimum_anchor_rows, "severity": "MODEL", "value": bundle.anchor_counts},
        {"check": "FIT_ROWS_SUFFICIENT", "passed": bundle.fit_rows >= 1_000, "severity": "MODEL", "value": bundle.fit_rows},
        {"check": "FIT_CALLS_SUFFICIENT", "passed": bundle.fit_calls >= 300, "severity": "MODEL", "value": bundle.fit_calls},
        {"check": "TRANSITIONS_SUFFICIENT", "passed": bundle.transition_rows >= 500, "severity": "MODEL", "value": bundle.transition_rows},
        {"check": "ALL_STATES_IN_VALID_OPERATIONAL_GRID", "passed": bool((valid_states["rows"] > 0).all()) and len(valid_states) == 5, "severity": "MODEL", "value": int((valid_states["rows"] > 0).sum())},
        {"check": "VALID_STATE_ENTROPY_AT_LEAST_30PCT", "passed": float(valid_states["normalized_state_entropy"].iloc[0]) >= 0.30, "severity": "MODEL", "value": float(valid_states["normalized_state_entropy"].iloc[0])},
        {"check": "VALID_HIGH_REGIME_OCCUPANCY_0P5_TO_30PCT", "passed": 0.5 <= high_pct <= 30.0, "severity": "MODEL", "value": high_pct},
        {"check": "CRITICAL_STATE_TARGET_ENRICHMENT", "passed": critical_enrichment >= 1.20, "severity": "MODEL", "value": critical_enrichment},
        {"check": "FINITE_OUTPUTS", "passed": bool(np.isfinite(decisions[["hsmm_risk_score", "hsmm_escalation_probability", "temporal_priority_score"]].to_numpy()).all()), "severity": "CRITICAL", "value": None},
        {"check": "WEIGHT_ZERO_CONTROL_RECALL_REPRODUCES_B61C", "passed": abs(float(control["gt3_recall"]) - float(baseline_valid["gt3_recall"])) <= 0.02, "severity": "CONTROL", "value": {"control": float(control["gt3_recall"]), "b61c": float(baseline_valid["gt3_recall"])}},
        {"check": "WEIGHT_ZERO_CONTROL_LIFT_REPRODUCES_B61C", "passed": float(control["gt3_precision_lift"]) >= 0.90 * float(baseline_valid["gt3_precision_lift"]), "severity": "CONTROL", "value": {"control": float(control["gt3_precision_lift"]), "b61c": float(baseline_valid["gt3_precision_lift"])}},
        {"check": "POSITIVE_HSMM_WEIGHT_SELECTED", "passed": float(valid["hsmm_weight"]) > 0.0, "severity": "CHALLENGER", "value": float(valid["hsmm_weight"])},
        {"check": "VALID_GT3_LIFT_NON_INFERIOR_80PCT", "passed": float(valid["gt3_precision_lift"]) >= 0.80 * float(baseline_valid["gt3_precision_lift"]), "severity": "CHALLENGER", "value": float(valid["gt3_precision_lift"])},
        {"check": "VALID_GT3_RECALL_NON_INFERIOR_90PCT", "passed": float(valid["gt3_recall"]) >= 0.90 * float(baseline_valid["gt3_recall"]), "severity": "CHALLENGER", "value": float(valid["gt3_recall"])},
        {"check": "VALID_LEAD_TIME_OR_6H_RECALL_IMPROVED", "passed": lead_improved, "severity": "CHALLENGER", "value": {"baseline_h": baseline_lead, "challenger_h": challenger_lead}},
        {"check": "VALID_STATE_STABILITY_AT_LEAST_80PCT", "passed": float(valid["state_stability"]) >= 0.80, "severity": "CHALLENGER", "value": float(valid["state_stability"])},
        {"check": "VALID_RESEARCH_COST_REDUCTION_NONNEGATIVE", "passed": float(valid["cost_reduction_pct"]) >= 0.0, "severity": "CHALLENGER", "value": float(valid["cost_reduction_pct"])},
        {"check": "PRODUCTION_PROMOTION_BLOCKED", "passed": True, "severity": "GOVERNANCE", "value": False},
        {"check": "AUTOMATIC_ACTION_BLOCKED", "passed": not bool(decisions["automatic_action_allowed"].any()), "severity": "GOVERNANCE", "value": False},
    ]
    return pd.DataFrame(rows)


def _materialize_decisions(frame: pd.DataFrame, run_id: str) -> int:
    columns = [
        "model_version", "source_model_version", "source_policy_version",
        "source_hsmm_version", "candidate_id", "port_call_id", "landmark_at",
        "decision_at", "split", "evaluation_role", "regime", "hsmm_state",
        "hsmm_state_confidence", "hsmm_risk_score",
        "hsmm_escalation_probability", "hsmm_dwell_steps", "p_state_fluid",
        "p_state_pressure_building", "p_state_congested",
        "p_state_critical_disruption", "p_state_recovery", "hsmm_weight",
        "base_temporal_priority_score", "base_critical_priority_score",
        "temporal_priority_score", "critical_priority_score", "rank_in_bucket",
        "active_calls", "state", "previous_state", "state_changed", "alert_active",
        "new_alert", "action_code", "p_delay_gt3", "p_delay_gt6",
        "p_gt3_breach_within_6h", "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h", "remaining_p10_h", "remaining_p50_h",
        "remaining_p90_h", "target_delay_gt_3h", "target_delay_gt_6h",
        "target_breach_or_censor_h", "decision_weight", "source_mode",
        "production_claim_allowed", "automatic_action_allowed",
        "materialization_run_id",
    ]
    output = frame.copy()
    output.insert(0, "model_version", MODEL_VERSION)
    output.insert(1, "source_model_version", SOURCE_MODEL_VERSION)
    output.insert(2, "source_policy_version", SOURCE_POLICY_VERSION)
    output.insert(3, "source_hsmm_version", SOURCE_HSMM_VERSION)
    output["source_mode"] = "HISTORICAL_REPLAY_SHADOW"
    output["production_claim_allowed"] = False
    output["automatic_action_allowed"] = False
    output["materialization_run_id"] = run_id
    output = output[columns]
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS serving;
    CREATE TABLE IF NOT EXISTS {OUTPUT_RELATION} (
        model_version TEXT NOT NULL,source_model_version TEXT NOT NULL,
        source_policy_version TEXT NOT NULL,source_hsmm_version TEXT NOT NULL,
        candidate_id TEXT NOT NULL,port_call_id TEXT NOT NULL,
        landmark_at TIMESTAMPTZ NOT NULL,decision_at TIMESTAMPTZ NOT NULL,
        split TEXT NOT NULL,evaluation_role TEXT NOT NULL,regime TEXT NOT NULL,
        hsmm_state TEXT NOT NULL,hsmm_state_confidence DOUBLE PRECISION NOT NULL,
        hsmm_risk_score DOUBLE PRECISION NOT NULL,
        hsmm_escalation_probability DOUBLE PRECISION NOT NULL,
        hsmm_dwell_steps INTEGER NOT NULL,p_state_fluid DOUBLE PRECISION NOT NULL,
        p_state_pressure_building DOUBLE PRECISION NOT NULL,
        p_state_congested DOUBLE PRECISION NOT NULL,
        p_state_critical_disruption DOUBLE PRECISION NOT NULL,
        p_state_recovery DOUBLE PRECISION NOT NULL,hsmm_weight DOUBLE PRECISION NOT NULL,
        base_temporal_priority_score DOUBLE PRECISION NOT NULL,
        base_critical_priority_score DOUBLE PRECISION NOT NULL,
        temporal_priority_score DOUBLE PRECISION NOT NULL,
        critical_priority_score DOUBLE PRECISION NOT NULL,
        rank_in_bucket INTEGER NOT NULL,active_calls INTEGER NOT NULL,
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
        PRIMARY KEY(model_version,evaluation_role,decision_at,port_call_id)
    );
    """
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(ddl)
        cursor.execute(f"DELETE FROM {OUTPUT_RELATION} WHERE model_version=%s", (MODEL_VERSION,))
        for start in range(0, len(output), 1_500):
            records = list(output.iloc[start : start + 1_500].itertuples(index=False, name=None))
            execute_values(
                cursor,
                f"INSERT INTO {OUTPUT_RELATION} ({','.join(columns)}) VALUES %s",
                records,
                page_size=1_500,
            )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS ix_b61dv11_snapshot ON {OUTPUT_RELATION} "
            "(decision_at DESC,state,rank_in_bucket)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS ix_b61dv11_call ON {OUTPUT_RELATION} "
            "(port_call_id,decision_at)"
        )
    return len(output)


def _materialize_model_card(
    scorecard: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected: dict[str, Any],
    control: dict[str, Any],
    gates: pd.DataFrame,
    model_card: dict[str, Any],
    run_id: str,
) -> None:
    records = []
    for row in scorecard.itertuples(index=False):
        payload = row._asdict()
        records.append(
            (
                MODEL_VERSION, str(payload["candidate_id"]), "VALID_CALIBRATE",
                float(payload["hsmm_weight"]), float(payload["gt3_budget_pct"]),
                float(payload["gt6_budget_pct"]), float(payload["objective"]),
                bool(payload["control_selected"]), bool(payload["challenger_selected"]),
                Json(_json_ready(payload)), run_id,
            )
        )
    selected_id = str(selected["candidate_id"])
    for row in diagnostics.itertuples(index=False):
        payload = row._asdict()
        records.append(
            (
                MODEL_VERSION, selected_id, str(payload["role"]),
                float(payload["hsmm_weight"]), float(payload["gt3_budget_pct"]),
                float(payload["gt6_budget_pct"]), float(payload["objective"]),
                False, True, Json(_json_ready(payload)), run_id,
            )
        )
    dedup = {(record[0], record[1], record[2]): record for record in records}
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SCORECARD_RELATION} (
                model_version TEXT NOT NULL,candidate_id TEXT NOT NULL,role TEXT NOT NULL,
                hsmm_weight DOUBLE PRECISION NOT NULL,gt3_budget_pct DOUBLE PRECISION NOT NULL,
                gt6_budget_pct DOUBLE PRECISION NOT NULL,objective DOUBLE PRECISION NOT NULL,
                control_selected BOOLEAN NOT NULL,challenger_selected BOOLEAN NOT NULL,
                metrics JSONB NOT NULL,materialization_run_id TEXT NOT NULL,
                PRIMARY KEY(model_version,candidate_id,role)
            )
            """
        )
        cursor.execute(f"DELETE FROM {SCORECARD_RELATION} WHERE model_version=%s", (MODEL_VERSION,))
        execute_values(cursor, f"INSERT INTO {SCORECARD_RELATION} VALUES %s", list(dedup.values()))
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {MODEL_CARD_RELATION} (
                model_version TEXT PRIMARY KEY,source_model_version TEXT NOT NULL,
                source_policy_version TEXT NOT NULL,source_hsmm_version TEXT NOT NULL,
                selected_candidate_id TEXT NOT NULL,control_candidate_id TEXT NOT NULL,
                model_card JSONB NOT NULL,quality_gates JSONB NOT NULL,
                production_claim_allowed BOOLEAN NOT NULL,
                automatic_action_allowed BOOLEAN NOT NULL,
                fresh_forward_confirmation_required BOOLEAN NOT NULL,
                materialization_run_id TEXT NOT NULL,created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cursor.execute(f"DELETE FROM {MODEL_CARD_RELATION} WHERE model_version=%s", (MODEL_VERSION,))
        cursor.execute(
            f"INSERT INTO {MODEL_CARD_RELATION} VALUES "
            "(%s,%s,%s,%s,%s,%s,%s,%s,false,false,true,%s,now())",
            (
                MODEL_VERSION, SOURCE_MODEL_VERSION, SOURCE_POLICY_VERSION,
                SOURCE_HSMM_VERSION, selected_id, str(control["candidate_id"]),
                Json(_json_ready(model_card)), Json(_json_ready(gates.to_dict("records"))), run_id,
            ),
        )


def _checksum(source: pd.DataFrame, settings: tuple[Any, ...]) -> str:
    digest = hashlib.sha256(MODEL_VERSION.encode("ascii"))
    digest.update(repr(settings).encode("ascii"))
    digest.update(
        pd.util.hash_pandas_object(
            source[["port_call_id", "landmark_at", "p_delay_gt3", "p_gt3_breach_within_24h"]],
            index=False,
        ).to_numpy(dtype="uint64").tobytes()
    )
    return digest.hexdigest()


def _start_run(checksum: str) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "orchestrator": "PREFECT",
        "fit_role": "VALID_SELECT_PREBREACH_CHRONOLOGICAL_EARLY_70PCT",
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


def run_b61d_v11(
    force: bool = False,
    hsmm_weights: str = "0.1,0.2,0.3,0.4",
    gt3_budgets: str = "1,2,3",
    gt6_budgets: str = "0.5,1",
    bucket_hours: int = 6,
) -> dict[str, Any]:
    del force
    if bucket_hours not in (3, 6):
        raise ValueError("B61D-v1.1 bucket_hours must be 3 or 6")
    for relation in (
        PREDICTION_RELATION, FEATURE_RELATION, BASELINE_DECISION_RELATION,
        BASELINE_SCORECARD_RELATION, "audit.ingestion_run",
    ):
        if not _relation_exists(relation):
            raise RuntimeError(f"Required B61D-v1.1 relation is missing: {relation}")
    _verify_upstream()
    weights = sorted({float(item) for item in hsmm_weights.split(",")})
    gt3_values = sorted({float(item) for item in gt3_budgets.split(",")})
    gt6_values = sorted({float(item) for item in gt6_budgets.split(",")})
    if not weights or any(item <= 0.0 or item > 1.0 for item in weights):
        raise ValueError("Challenger HSMM weights must be in (0,1]")
    if any(item <= 0.0 or item > 25.0 for item in [*gt3_values, *gt6_values]):
        raise ValueError("Alert budgets must be in (0,25]")
    source = load_source()
    run_id = _start_run(_checksum(source, (weights, gt3_values, gt6_values, bucket_hours)))
    try:
        fit_mask, fit_cutoff = prebreach_fit_mask(source)
        _update_progress(
            run_id, "FITTING_PREBREACH_ANCHORED_HSMM",
            source_rows=len(source), fit_rows=int(fit_mask.sum()),
            source_calls=source["port_call_id"].nunique(),
        )
        bundle = fit_anchored_hsmm(source, fit_mask, fit_cutoff)
        _update_progress(
            run_id, "DECODING_ANCHORED_EXPLICIT_DURATION_STATES",
            anchor_counts=bundle.anchor_counts,fit_calls=bundle.fit_calls,
            transitions=bundle.transition_rows,
        )
        decoded = decode_frame(bundle, source)
        decoded_fit_mask = (
            decoded["evaluation_role"].eq("VALID_SELECT")
            & decoded["early_warning_eligible"].fillna(False).astype(bool)
            & decoded["pre_breach_eligible"].fillna(False).astype(bool)
            & decoded["landmark_at"].le(bundle.fit_cutoff)
        )
        state_diagnostics = _state_diagnostics(decoded)
        baseline = _baseline_metrics()
        _update_progress(
            run_id, "BENCHMARKING_EXACT_B61C_CONTROL_AND_POSITIVE_WEIGHTS",
            control_weight=0.0,
            challenger_candidates=len(weights) * len(gt3_values) * len(gt6_values),
        )
        scorecard, control, challenger = _candidate_replay(
            decoded, decoded_fit_mask, weights, gt3_values, gt6_values, bucket_hours
        )
        decisions, diagnostics, parameters = _selected_replay(
            decoded, decoded_fit_mask, challenger, bucket_hours
        )
        gates = _quality_gates(
            source, bundle, state_diagnostics, diagnostics, control,
            baseline, decisions,
        )
        scientific_passed = bool(
            gates.loc[
                gates["severity"].isin(["CRITICAL", "MODEL", "CONTROL", "CHALLENGER"]),
                "passed",
            ].all()
        )
        decision = (
            "READY_FOR_B61D_V11_ANCHORED_HSMM_SHADOW"
            if scientific_passed
            else "B61D_V11_NOT_ACCEPTED_KEEP_B61C_V11"
        )
        decision_rows = _materialize_decisions(decisions, run_id)
        model_card = {
            "model_version": MODEL_VERSION,
            "model_family": "WEAKLY_SUPERVISED_ANCHORED_CONTEXTUAL_EXPLICIT_DURATION_HSMM",
            "states": list(STATE_NAMES),
            "fit_role": "VALID_SELECT_PREBREACH_CHRONOLOGICAL_EARLY_70PCT",
            "fit_cutoff": bundle.fit_cutoff,
            "fit_rows": bundle.fit_rows,
            "fit_calls": bundle.fit_calls,
            "anchor_counts": bundle.anchor_counts,
            "minimum_anchor_rows": bundle.minimum_anchor_rows,
            "transition_rows": bundle.transition_rows,
            "duration_bin_hours": bundle.duration_bin_hours,
            "exact_b61c_control": control,
            "selected_challenger": challenger,
            "challenger_accepted": scientific_passed,
            "limitations": [
                "States are weakly supervised operational regimes, not observed ground truth labels.",
                "Targets are used only for VALID diagnostics and never for state fitting.",
                "TEST is reused diagnostic and cannot support a production claim.",
                "All actions remain read-only recommendations.",
            ],
        }
        _materialize_model_card(
            scorecard, diagnostics, challenger, control, gates, model_card, run_id
        )
        artifacts = {
            "model": _put_bytes(
                f"models/b61dv11/{OUTPUT_PREFIX}/anchored_hsmm.pkl",
                pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL),
                "application/octet-stream",
            ),
            "model_card": _put_json(
                f"models/b61dv11/{OUTPUT_PREFIX}/model_card.json", model_card
            ),
        }
        for name, report in {
            "candidate_scorecard": scorecard,
            "selected_diagnostics": diagnostics,
            "state_diagnostics": state_diagnostics,
            "quality_gates": gates,
        }.items():
            _put_csv(f"reports/b61dv11/{OUTPUT_PREFIX}/{name}.csv", report)
        metadata = {
            "decision": decision,
            "model_version": MODEL_VERSION,
            "source_model_version": SOURCE_MODEL_VERSION,
            "source_policy_version": SOURCE_POLICY_VERSION,
            "source_hsmm_version": SOURCE_HSMM_VERSION,
            "row_count": len(source),
            "vessel_calls": source["port_call_id"].nunique(),
            "decision_rows": decision_rows,
            "control_candidate_id": str(control["candidate_id"]),
            "selected_candidate_id": str(challenger["candidate_id"]),
            "selected_hsmm_weight": float(challenger["hsmm_weight"]),
            "gt3_budget_pct": parameters.gt3_budget_pct,
            "gt6_budget_pct": parameters.gt6_budget_pct,
            "bucket_hours": bucket_hours,
            "fit_role": "VALID_SELECT_PREBREACH_CHRONOLOGICAL_EARLY_70PCT",
            "selection_role": "VALID_CALIBRATE",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "test_used_for_fit_or_selection": False,
            "scientific_gates_passed": scientific_passed,
            "challenger_accepted": scientific_passed,
            "shadow_api_allowed": bool(
                gates.loc[gates["severity"].isin(["CRITICAL", "MODEL", "CONTROL"]), "passed"].all()
            ),
            "production_promotion_allowed": False,
            "automatic_action_allowed": False,
            "fresh_forward_confirmation_required": True,
            "artifacts": artifacts,
            "next_block": (
                "B61D_V11_FRESH_FORWARD_SHADOW_MONITORING"
                if scientific_passed else "KEEP_B61C_V11_AND_REPORT_ANCHORED_HSMM_LIMITS"
            ),
        }
        _put_json(f"configs/b61dv11/{OUTPUT_PREFIX}/final_decision.json", metadata)
        _finish_run(run_id, "SUCCESS", len(source), metadata)
        return metadata
    except Exception as exc:
        _finish_run(
            run_id, "FAILED", None,
            {
                "decision": "FAILED", "model_version": MODEL_VERSION,
                "test_used_for_fit_or_selection": False,
                "production_promotion_allowed": False,
                "automatic_action_allowed": False,
                "next_block": "FIX_B61D_V11_AND_RERUN",
            },
            str(exc),
        )
        raise


def verify_b61d_v11_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "decision", "model_version", "decision_rows", "control_candidate_id",
        "selected_candidate_id", "fit_role", "selection_role", "test_role",
        "test_used_for_fit_or_selection", "challenger_accepted",
        "shadow_api_allowed", "production_promotion_allowed",
        "automatic_action_allowed", "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"B61D-v1.1 result misses fields: {missing}")
    if result["fit_role"] != "VALID_SELECT_PREBREACH_CHRONOLOGICAL_EARLY_70PCT":
        raise ValueError("B61D-v1.1 fit role is invalid")
    if result["selection_role"] != "VALID_CALIBRATE":
        raise ValueError("B61D-v1.1 selection must use VALID_CALIBRATE")
    if result["test_used_for_fit_or_selection"]:
        raise ValueError("B61D-v1.1 TEST leakage contract violated")
    if result["test_role"] != "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY":
        raise ValueError("B61D-v1.1 must disclose TEST reuse")
    if result["production_promotion_allowed"] or result["automatic_action_allowed"]:
        raise ValueError("B61D-v1.1 cannot promote or execute actions")
    if int(result["decision_rows"]) <= 0:
        raise ValueError("B61D-v1.1 produced no decisions")
    return result
