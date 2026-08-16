from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from prefect_flows.b61d_v13_core import (
    SOURCE_HSMM_VERSION,
    SOURCE_POLICY_VERSION,
    DualStageParameters,
    apply_dual_stage_policy,
    evaluate_dual_policy,
    event_lead_metrics,
    parse_integer_grid,
    parse_modes,
    parse_numeric_grid,
)


POLICY_VERSION = "b61d-v1.3.1-contract-recalibration-v1"
SOURCE_DUAL_STAGE_VERSION = "b61d-v1.3-dual-stage-policy-v1"
VALID_ROLES = ("VALID_SELECT", "VALID_CALIBRATE")


@dataclass(frozen=True)
class ContractThresholds:
    early_min_lift: float = 2.0
    early_min_recall: float = 0.05
    early_baseline_recall_ratio: float = 0.20
    early_min_detected_positive_calls: int = 5
    early_min_recall_at_6h: float = 0.02
    critical_precision_ratio: float = 0.80
    critical_recall_ratio: float = 1.00
    critical_lift_ratio: float = 0.80
    min_stability: float = 0.80
    max_alert_rate_pct: float = 10.0
    max_alert_rate_delta_pct: float = 3.0


def _role_contracts(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: ContractThresholds,
) -> dict[str, Any]:
    early_required_recall = max(
        thresholds.early_min_recall,
        thresholds.early_baseline_recall_ratio * float(baseline["gt3_recall"]),
    )
    baseline_alert_rate = float(
        baseline.get("alert_rate_pct", baseline.get("gt3_alert_rate_pct", 5.0))
    )
    gates = {
        "gate_early_only_lift": (
            float(metrics["early_gt3_precision_lift"]) >= thresholds.early_min_lift
        ),
        "gate_early_only_recall": (
            float(metrics["early_gt3_recall"]) >= early_required_recall
        ),
        "gate_early_detected_calls": (
            int(metrics.get("early_alerted_positive_calls", 0))
            >= thresholds.early_min_detected_positive_calls
        ),
        "gate_early_lead_6h": (
            float(metrics.get("early_event_recall_at_least_6h", 0.0))
            >= thresholds.early_min_recall_at_6h
        ),
        "gate_critical_precision_noninferior": (
            float(metrics["gt6_precision"])
            >= thresholds.critical_precision_ratio * float(baseline["gt6_precision"])
        ),
        "gate_critical_recall_noninferior": (
            float(metrics["gt6_recall"])
            >= thresholds.critical_recall_ratio * float(baseline["gt6_recall"])
        ),
        "gate_critical_lift_noninferior": (
            float(metrics["gt6_precision_lift"])
            >= thresholds.critical_lift_ratio * float(baseline["gt6_precision_lift"])
        ),
        "gate_stability": float(metrics["state_stability"]) >= thresholds.min_stability,
        "gate_alert_burden": (
            float(metrics["alert_rate_pct"])
            <= max(
                thresholds.max_alert_rate_pct,
                baseline_alert_rate + thresholds.max_alert_rate_delta_pct,
            )
        ),
        "gate_research_cost": float(metrics["cost_reduction_pct"]) >= 0.0,
    }
    result = dict(metrics)
    result.update(gates)
    result["early_required_recall"] = early_required_recall
    result["passes_role_contracts"] = all(gates.values())
    result["role_gates_passed"] = int(sum(bool(value) for value in gates.values()))
    result["role_gates_total"] = int(len(gates))
    return result


def evaluate_candidate_roles(
    source: pd.DataFrame,
    parameters: DualStageParameters,
    baselines: dict[str, dict[str, Any]],
    thresholds: ContractThresholds | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    thresholds = thresholds or ContractThresholds()
    role_rows: list[dict[str, Any]] = []
    for role in VALID_ROLES:
        role_source = source.loc[source["evaluation_role"].eq(role)].copy()
        if role_source.empty:
            raise ValueError(f"Missing required validation role: {role}")
        decisions = apply_dual_stage_policy(role_source, parameters)
        metrics = evaluate_dual_policy(decisions, parameters)
        metrics["role"] = role
        role_rows.append(_role_contracts(metrics, baselines[role], thresholds))
    role_frame = pd.DataFrame(role_rows)
    robust = aggregate_robust_contracts(role_frame, parameters)
    return role_frame, robust


def aggregate_robust_contracts(
    role_frame: pd.DataFrame,
    parameters: DualStageParameters,
) -> dict[str, Any]:
    roles = set(role_frame["role"].astype(str))
    if roles != set(VALID_ROLES):
        raise ValueError(f"Robust contracts require exactly {VALID_ROLES}, got {sorted(roles)}")
    gate_columns = [column for column in role_frame if column.startswith("gate_")]
    robust: dict[str, Any] = {
        "candidate_id": parameters.policy_id,
        "early_mode": parameters.early_mode,
        "early_min_score": parameters.early_min_score,
        "early_top_k": parameters.early_top_k,
        "critical_top_k": parameters.critical_top_k,
        "hold_windows": parameters.hold_windows,
        "validation_roles": list(VALID_ROLES),
        "mean_objective": float(role_frame["objective"].mean()),
        "worst_early_precision": float(role_frame["early_gt3_precision"].min()),
        "worst_early_recall": float(role_frame["early_gt3_recall"].min()),
        "worst_early_lift": float(role_frame["early_gt3_precision_lift"].min()),
        "worst_early_detected_calls": int(role_frame["early_alerted_positive_calls"].min()),
        "worst_early_recall_6h": float(role_frame["early_event_recall_at_least_6h"].min()),
        "worst_critical_precision": float(role_frame["gt6_precision"].min()),
        "worst_critical_recall": float(role_frame["gt6_recall"].min()),
        "worst_critical_lift": float(role_frame["gt6_precision_lift"].min()),
        "worst_stability": float(role_frame["state_stability"].min()),
        "worst_cost_reduction_pct": float(role_frame["cost_reduction_pct"].min()),
        "max_alert_rate_pct": float(role_frame["alert_rate_pct"].max()),
    }
    for gate in gate_columns:
        robust[f"robust_{gate}"] = bool(role_frame[gate].all())
    robust_gate_columns = [key for key in robust if key.startswith("robust_gate_")]
    robust["robust_gates_passed"] = int(sum(bool(robust[key]) for key in robust_gate_columns))
    robust["robust_gates_total"] = int(len(robust_gate_columns))
    robust["passes_robust_contracts"] = bool(
        robust_gate_columns and all(bool(robust[key]) for key in robust_gate_columns)
    )
    robust["robust_objective"] = float(
        robust["mean_objective"]
        + 40.0 * robust["worst_early_recall"]
        + 3.0 * min(robust["worst_early_lift"], 20.0)
        + 20.0 * robust["worst_early_recall_6h"]
        + 25.0 * robust["worst_critical_recall"]
        + 2.0 * min(robust["worst_critical_lift"], 20.0)
        + 2.0 * robust["robust_gates_passed"]
    )
    return robust


def choose_robust_candidate(scorecard: pd.DataFrame) -> dict[str, Any]:
    if scorecard.empty:
        raise ValueError("Robust contract scorecard is empty")
    eligible = scorecard.loc[scorecard["passes_robust_contracts"]]
    pool = eligible if not eligible.empty else scorecard
    selected = pool.sort_values(
        [
            "robust_gates_passed", "robust_objective", "worst_early_recall",
            "worst_early_lift", "worst_critical_lift", "max_alert_rate_pct",
        ],
        ascending=[False, False, False, False, False, True],
    ).iloc[0]
    return selected.to_dict()


def _weighted_binary(
    actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    tp = float(weight[actual & predicted].sum())
    fp = float(weight[~actual & predicted].sum())
    fn = float(weight[actual & ~predicted].sum())
    tn = float(weight[~actual & ~predicted].sum())
    total = max(tp + fp + fn + tn, 1e-12)
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    prevalence = (tp + fn) / total
    return {
        "precision": precision,
        "recall": recall,
        "precision_lift": precision / max(prevalence, 1e-12),
        "positive_rate_pct": 100.0 * float(weight[predicted].sum()) / total,
    }


def cluster_bootstrap_stage_metrics(
    decisions: pd.DataFrame,
    *,
    iterations: int = 500,
    random_state: int = 20260810,
) -> dict[str, Any]:
    calls = decisions["port_call_id"].astype(str).drop_duplicates().to_numpy()
    if len(calls) < 20:
        raise ValueError("Cluster bootstrap requires at least 20 port calls")
    frame = decisions.copy()
    frame["_call"] = frame["port_call_id"].astype(str)
    rng = np.random.default_rng(random_state)
    names = (
        "early_gt3_precision", "early_gt3_recall", "early_gt3_precision_lift",
        "critical_gt6_precision", "critical_gt6_recall", "critical_gt6_precision_lift",
        "early_warning_rate_pct", "critical_action_rate_pct",
    )
    samples: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(iterations):
        sampled = rng.choice(calls, size=len(calls), replace=True)
        multiplicity = pd.Series(sampled).value_counts()
        bootstrap = frame.loc[frame["_call"].isin(multiplicity.index)].copy()
        weight = (
            pd.to_numeric(bootstrap["decision_weight"], errors="coerce").fillna(0.0)
            * bootstrap["_call"].map(multiplicity).astype(float)
        ).to_numpy(float)
        early = _weighted_binary(
            bootstrap["target_delay_gt_3h"].astype(bool).to_numpy(),
            bootstrap["early_warning"].astype(bool).to_numpy(),
            weight,
        )
        critical = _weighted_binary(
            bootstrap["target_delay_gt_6h"].astype(bool).to_numpy(),
            bootstrap["critical_action"].astype(bool).to_numpy(),
            weight,
        )
        for metric in ("precision", "recall", "precision_lift"):
            samples[f"early_gt3_{metric}"].append(early[metric])
            samples[f"critical_gt6_{metric}"].append(critical[metric])
        samples["early_warning_rate_pct"].append(early["positive_rate_pct"])
        samples["critical_action_rate_pct"].append(critical["positive_rate_pct"])
    result: dict[str, Any] = {
        "method": "PORT_CALL_CLUSTER_BOOTSTRAP_SEPARATE_STAGES",
        "iterations": iterations,
        "seed": random_state,
        "clusters": int(len(calls)),
    }
    for name, values in samples.items():
        result[name] = {
            "p2_5": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.50)),
            "p97_5": float(np.quantile(values, 0.975)),
        }
    return result


def verify_b61d_v131_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "policy_version", "decision", "selected_candidate_id",
        "selection_roles", "test_role", "test_used_for_fit_or_selection",
        "early_stage_validated", "critical_stage_validated",
        "production_promotion_allowed", "automatic_action_allowed",
        "serving_rows", "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(f"B61D-v1.3.1 result contract misses: {missing}")
    if result["policy_version"] != POLICY_VERSION:
        raise RuntimeError("B61D-v1.3.1 policy version mismatch")
    if tuple(result["selection_roles"]) != VALID_ROLES:
        raise RuntimeError("B61D-v1.3.1 must require both validation roles")
    if result["test_used_for_fit_or_selection"]:
        raise RuntimeError("B61D-v1.3.1 TEST leakage contract failed")
    if result["production_promotion_allowed"] or result["automatic_action_allowed"]:
        raise RuntimeError("B61D-v1.3.1 cannot enable production or automatic action")
    if int(result["serving_rows"]) <= 0:
        raise RuntimeError("B61D-v1.3.1 did not materialize research decisions")
    return result


__all__ = [
    "POLICY_VERSION", "SOURCE_DUAL_STAGE_VERSION", "SOURCE_HSMM_VERSION",
    "SOURCE_POLICY_VERSION", "VALID_ROLES", "ContractThresholds",
    "DualStageParameters", "apply_dual_stage_policy", "evaluate_dual_policy",
    "event_lead_metrics",
    "parse_integer_grid", "parse_modes", "parse_numeric_grid",
    "evaluate_candidate_roles", "aggregate_robust_contracts",
    "choose_robust_candidate", "cluster_bootstrap_stage_metrics",
    "verify_b61d_v131_result",
]
