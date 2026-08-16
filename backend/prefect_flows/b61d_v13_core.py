from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


POLICY_VERSION = "b61d-v1.3-dual-stage-policy-v1"
SOURCE_HSMM_VERSION = "b61d-v1.1-anchored-hsmm-v1"
SOURCE_POLICY_VERSION = "b61c-dynamic-alert-policy-v1.1"

EARLY_MODES = (
    "PRESSURE_CONGESTED",
    "TRANSITION_AWARE",
    "NON_FLUID",
)


@dataclass(frozen=True)
class DualStageParameters:
    early_mode: str
    early_min_score: float
    early_top_k: int
    critical_top_k: int
    hold_windows: int
    bucket_hours: int = 6
    transition_threshold: float = 0.15
    hysteresis_bonus: float = 0.05
    false_positive_cost_gt3: float = 1.0
    false_negative_cost_gt3: float = 15.0
    false_positive_cost_gt6: float = 2.0
    false_negative_cost_gt6: float = 40.0
    early_action_cost: float = 0.10
    critical_action_cost: float = 0.20

    def __post_init__(self) -> None:
        if self.early_mode not in EARLY_MODES:
            raise ValueError(f"Unsupported early-warning mode: {self.early_mode}")
        if not 0.0 <= self.early_min_score <= 1.0:
            raise ValueError("early_min_score must be in [0,1]")
        if self.early_top_k not in range(1, 11):
            raise ValueError("early_top_k must be between 1 and 10")
        if self.critical_top_k not in range(1, 11):
            raise ValueError("critical_top_k must be between 1 and 10")
        if self.hold_windows not in (0, 1, 2, 3):
            raise ValueError("hold_windows must be 0, 1, 2 or 3")
        if self.bucket_hours not in (3, 6):
            raise ValueError("bucket_hours must be 3 or 6")

    @property
    def policy_id(self) -> str:
        mode = {
            "PRESSURE_CONGESTED": "PC",
            "TRANSITION_AWARE": "TA",
            "NON_FLUID": "NF",
        }[self.early_mode]
        return (
            f"DUAL_{mode}_S{self.early_min_score:g}_E{self.early_top_k}_"
            f"C{self.critical_top_k}_H{self.hold_windows}_{self.bucket_hours}H"
        )


def parse_numeric_grid(
    value: str | Iterable[float], *, minimum: float, maximum: float, name: str
) -> list[float]:
    parsed = (
        [float(item.strip()) for item in value.split(",") if item.strip()]
        if isinstance(value, str)
        else [float(item) for item in value]
    )
    result = sorted(set(parsed))
    if not result or any(item < minimum or item > maximum for item in result):
        raise ValueError(f"{name} must contain values in [{minimum},{maximum}]")
    return result


def parse_integer_grid(
    value: str | Iterable[int], *, allowed: set[int], name: str
) -> list[int]:
    parsed = (
        [int(item.strip()) for item in value.split(",") if item.strip()]
        if isinstance(value, str)
        else [int(item) for item in value]
    )
    result = sorted(set(parsed))
    if not result or any(item not in allowed for item in result):
        raise ValueError(f"{name} must be a subset of {sorted(allowed)}")
    return result


def parse_modes(value: str | Iterable[str]) -> list[str]:
    parsed = (
        [item.strip().upper() for item in value.split(",") if item.strip()]
        if isinstance(value, str)
        else [str(item).strip().upper() for item in value]
    )
    result = list(dict.fromkeys(parsed))
    invalid = sorted(set(result).difference(EARLY_MODES))
    if not result or invalid:
        raise ValueError(f"Unsupported early-warning modes: {invalid}")
    return result


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(float)


def _early_candidate_mask(
    hidden: np.ndarray,
    escalation: np.ndarray,
    parameters: DualStageParameters,
) -> np.ndarray:
    pressure = np.isin(hidden, ["PRESSURE_BUILDING", "CONGESTED"])
    if parameters.early_mode == "PRESSURE_CONGESTED":
        result = pressure
    elif parameters.early_mode == "TRANSITION_AWARE":
        result = pressure | (
            (hidden != "CRITICAL_DISRUPTION")
            & (escalation >= parameters.transition_threshold)
        )
    else:
        result = np.isin(hidden, ["PRESSURE_BUILDING", "CONGESTED", "RECOVERY"])
    return result & (hidden != "CRITICAL_DISRUPTION")


def _action_for_state(state: str) -> str:
    return {
        "NORMAL": "MONITOR",
        "WATCH": "REVIEW_STATE_AND_NEXT_WINDOW",
        "EARLY_WARNING": "PREPARE_BERTH_AND_REVIEW_NEXT_WINDOW",
        "CRITICAL": "ESCALATE_TO_OPERATIONS_CONTROL",
    }[state]


def apply_dual_stage_policy(
    frame: pd.DataFrame,
    parameters: DualStageParameters,
) -> pd.DataFrame:
    required = {
        "evaluation_role", "decision_at", "port_call_id", "hsmm_state",
        "hsmm_escalation_probability", "hsmm_risk_score",
        "base_temporal_priority_score", "base_critical_priority_score",
        "p_delay_gt3", "p_delay_gt6", "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h", "p_gt3_breach_within_24h",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dual-stage source misses columns: {missing}")
    if frame.empty:
        return frame.copy()

    output = frame.sort_values(
        ["evaluation_role", "decision_at", "port_call_id"]
    ).reset_index(drop=True).copy()
    rows = len(output)
    roles = output["evaluation_role"].astype(str).to_numpy()
    identifiers = output["port_call_id"].astype(str).to_numpy()
    hidden = output["hsmm_state"].astype(str).to_numpy()
    decision_keys = pd.to_datetime(output["decision_at"], utc=True).astype("int64").to_numpy()

    escalation = np.clip(_numeric(output, "hsmm_escalation_probability"), 0.0, 1.0)
    hsmm_risk = np.clip(_numeric(output, "hsmm_risk_score"), 0.0, 1.0)
    base_temporal = np.clip(_numeric(output, "base_temporal_priority_score"), 0.0, 1.0)
    base_critical = np.clip(_numeric(output, "base_critical_priority_score"), 0.0, 1.0)
    p3 = np.clip(_numeric(output, "p_delay_gt3"), 0.0, 1.0)
    p6 = np.clip(_numeric(output, "p_delay_gt6"), 0.0, 1.0)
    h6 = np.clip(_numeric(output, "p_gt3_breach_within_6h"), 0.0, 1.0)
    h12 = np.clip(_numeric(output, "p_gt3_breach_within_12h"), 0.0, 1.0)
    h24 = np.clip(_numeric(output, "p_gt3_breach_within_24h"), 0.0, 1.0)

    grouping = [output["evaluation_role"], output["port_call_id"]]
    delta_p3 = (
        output.assign(_value=p3).groupby(grouping, sort=False)["_value"]
        .diff().fillna(0.0).clip(lower=0.0).to_numpy(float)
    )
    delta_h12 = (
        output.assign(_value=h12).groupby(grouping, sort=False)["_value"]
        .diff().fillna(0.0).clip(lower=0.0).to_numpy(float)
    )
    early_score = np.clip(
        0.22 * h24 + 0.18 * h12 + 0.08 * h6 + 0.18 * escalation
        + 0.14 * delta_p3 + 0.10 * delta_h12 + 0.10 * base_temporal,
        0.0, 1.0,
    )
    critical_score = np.clip(
        0.40 * base_critical + 0.25 * p6 + 0.20 * h6 + 0.15 * hsmm_risk,
        0.0, 1.0,
    )
    base_early = (
        _early_candidate_mask(hidden, escalation, parameters)
        & (early_score >= parameters.early_min_score)
    )
    # Critical actions are anchored to the decoded critical state. They are never
    # filtered by the posterior magnitude that failed in B61D-v1.2.
    base_critical_candidate = hidden == "CRITICAL_DISRUPTION"

    previous = np.full(rows, "NORMAL", dtype=object)
    state = np.full(rows, "NORMAL", dtype=object)
    early_candidate = np.zeros(rows, dtype=bool)
    critical_candidate = base_critical_candidate.copy()
    retained = np.zeros(rows, dtype=bool)
    early_selected = np.zeros(rows, dtype=bool)
    critical_selected = np.zeros(rows, dtype=bool)
    early_rank = np.zeros(rows, dtype=int)
    critical_rank = np.zeros(rows, dtype=int)
    active_calls = np.zeros(rows, dtype=int)
    early_capacity = np.zeros(rows, dtype=int)
    critical_capacity = np.zeros(rows, dtype=int)
    early_reason = np.full(rows, "NOT_CANDIDATE", dtype=object)

    bucket_start = np.ones(rows, dtype=bool)
    bucket_start[1:] = (roles[1:] != roles[:-1]) | (decision_keys[1:] != decision_keys[:-1])
    starts = np.flatnonzero(bucket_start)
    ends = np.r_[starts[1:], rows]
    previous_state: dict[str, str] = {}
    hold_remaining: dict[str, int] = {}
    current_role: str | None = None

    for start, end in zip(starts, ends):
        role = roles[start]
        if role != current_role:
            previous_state.clear()
            hold_remaining.clear()
            current_role = role
        size = end - start
        ids = identifiers[start:end]
        bucket_previous = np.asarray(
            [previous_state.get(identifier, "NORMAL") for identifier in ids],
            dtype=object,
        )
        holds = np.fromiter(
            (hold_remaining.get(identifier, 0) for identifier in ids),
            dtype=int, count=size,
        )
        previous[start:end] = bucket_previous
        bucket_retained = (
            (holds > 0)
            & np.isin(bucket_previous, ["EARLY_WARNING", "CRITICAL"])
            & (hidden[start:end] != "CRITICAL_DISRUPTION")
            & (early_score[start:end] >= max(0.05, parameters.early_min_score * 0.75))
        )
        retained[start:end] = bucket_retained
        bucket_early_candidate = base_early[start:end] | bucket_retained
        early_candidate[start:end] = bucket_early_candidate
        early_reason[start:end][base_early[start:end]] = "EARLY_STATE_AND_SCORE"
        early_reason[start:end][bucket_retained] = "HYSTERESIS_HOLD"

        bucket_critical = base_critical_candidate[start:end]
        critical_positions = np.flatnonzero(bucket_critical)
        critical_order = critical_positions[
            np.argsort(-critical_score[start:end][critical_positions], kind="stable")
        ]
        critical_choice = critical_order[: parameters.critical_top_k]
        critical_selected[start + critical_choice] = True
        if len(critical_positions):
            ranks = np.empty(len(critical_positions), dtype=int)
            ranks[np.argsort(-critical_score[start:end][critical_positions], kind="stable")] = (
                np.arange(1, len(critical_positions) + 1)
            )
            critical_rank[start + critical_positions] = ranks

        early_positions = np.flatnonzero(bucket_early_candidate & ~bucket_critical)
        effective_early = early_score[start:end].copy()
        effective_early += parameters.hysteresis_bonus * bucket_retained.astype(float)
        early_order = early_positions[
            np.argsort(-effective_early[early_positions], kind="stable")
        ]
        early_choice = early_order[: parameters.early_top_k]
        early_selected[start + early_choice] = True
        if len(early_positions):
            ranks = np.empty(len(early_positions), dtype=int)
            ranks[np.argsort(-effective_early[early_positions], kind="stable")] = (
                np.arange(1, len(early_positions) + 1)
            )
            early_rank[start + early_positions] = ranks

        bucket_critical_selected = np.zeros(size, dtype=bool)
        bucket_critical_selected[critical_choice] = True
        bucket_early_selected = np.zeros(size, dtype=bool)
        bucket_early_selected[early_choice] = True
        watch = (
            np.isin(hidden[start:end], ["PRESSURE_BUILDING", "CONGESTED", "CRITICAL_DISRUPTION"])
            | bucket_early_candidate
        ) & ~bucket_critical_selected & ~bucket_early_selected
        bucket_state = np.where(
            bucket_critical_selected, "CRITICAL",
            np.where(bucket_early_selected, "EARLY_WARNING", np.where(watch, "WATCH", "NORMAL")),
        )
        state[start:end] = bucket_state
        active_calls[start:end] = size
        early_capacity[start:end] = parameters.early_top_k
        critical_capacity[start:end] = parameters.critical_top_k

        for position, identifier in enumerate(ids):
            previous_state[identifier] = str(bucket_state[position])
            if bucket_critical_selected[position]:
                hold_remaining[identifier] = parameters.hold_windows
            elif bucket_early_selected[position] and base_early[start + position]:
                hold_remaining[identifier] = parameters.hold_windows
            elif bucket_early_selected[position] and bucket_retained[position]:
                hold_remaining[identifier] = max(int(holds[position]) - 1, 0)
            else:
                hold_remaining[identifier] = max(int(holds[position]) - 1, 0)

    output["delta_p_delay_gt3"] = delta_p3
    output["delta_hazard_12h"] = delta_h12
    output["early_warning_score"] = early_score
    output["critical_action_score"] = critical_score
    output["early_candidate"] = early_candidate
    output["critical_candidate"] = critical_candidate
    output["early_candidate_reason"] = early_reason
    output["hysteresis_retained"] = retained
    output["early_rank_in_bucket"] = early_rank
    output["critical_rank_in_bucket"] = critical_rank
    output["early_warning"] = early_selected
    output["critical_action"] = critical_selected
    output["alert_active"] = early_selected | critical_selected
    output["previous_state"] = previous
    output["state"] = state
    output["state_changed"] = state != previous
    output["new_alert"] = output["alert_active"] & ~np.isin(previous, ["EARLY_WARNING", "CRITICAL"])
    output["action_code"] = [_action_for_state(item) for item in state]
    output["active_calls"] = active_calls
    output["early_capacity"] = early_capacity
    output["critical_capacity"] = critical_capacity
    output["early_mode"] = parameters.early_mode
    output["early_min_score"] = parameters.early_min_score
    output["early_top_k"] = parameters.early_top_k
    output["critical_top_k"] = parameters.critical_top_k
    output["hold_windows"] = parameters.hold_windows
    output["policy_id"] = parameters.policy_id
    output["source_mode"] = "HISTORICAL_REPLAY_SHADOW"
    output["production_claim_allowed"] = False
    output["automatic_action_allowed"] = False
    return output


def _weighted_binary_metrics(
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
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "precision_lift": precision / max(prevalence, 1e-12),
        "positive_rate_pct": 100.0 * float(weight[predicted].sum()) / total,
    }


def event_lead_metrics(
    decisions: pd.DataFrame, *, alert_column: str = "alert_active", prefix: str = ""
) -> dict[str, Any]:
    positive = decisions.loc[decisions["target_delay_gt_3h"].astype(bool)]
    positive_calls = int(positive["port_call_id"].nunique())
    alerted = positive.loc[
        positive[alert_column].astype(bool)
        & pd.to_numeric(positive["target_breach_or_censor_h"], errors="coerce").ge(0.0)
    ]
    leads = (
        alerted.groupby("port_call_id")["target_breach_or_censor_h"].max()
        if not alerted.empty else pd.Series(dtype=float)
    )
    result: dict[str, Any] = {
        f"{prefix}positive_calls": positive_calls,
        f"{prefix}alerted_positive_calls": int(len(leads)),
        f"{prefix}event_recall_any": float(len(leads) / max(positive_calls, 1)),
        f"{prefix}median_event_lead_h": float(leads.median()) if not leads.empty else None,
        f"{prefix}p25_event_lead_h": float(leads.quantile(0.25)) if not leads.empty else None,
    }
    for horizon in (6, 12, 24):
        result[f"{prefix}event_recall_at_least_{horizon}h"] = float(
            leads.ge(horizon).sum() / max(positive_calls, 1)
        )
    return result


def evaluate_dual_policy(
    decisions: pd.DataFrame, parameters: DualStageParameters
) -> dict[str, Any]:
    weight = pd.to_numeric(decisions["decision_weight"], errors="coerce").fillna(0.0).to_numpy(float)
    gt3 = decisions["target_delay_gt_3h"].astype(bool).to_numpy()
    gt6 = decisions["target_delay_gt_6h"].astype(bool).to_numpy()
    attention = decisions["alert_active"].astype(bool).to_numpy()
    early = decisions["early_warning"].astype(bool).to_numpy()
    critical = decisions["critical_action"].astype(bool).to_numpy()
    m3 = _weighted_binary_metrics(gt3, attention, weight)
    me = _weighted_binary_metrics(gt3, early, weight)
    m6 = _weighted_binary_metrics(gt6, critical, weight)
    reference_cost = (
        parameters.false_negative_cost_gt3 * float(weight[gt3].sum())
        + parameters.false_negative_cost_gt6 * float(weight[gt6].sum())
    )
    policy_cost = (
        parameters.false_positive_cost_gt3 * m3["fp"]
        + parameters.false_negative_cost_gt3 * m3["fn"]
        + parameters.false_positive_cost_gt6 * m6["fp"]
        + parameters.false_negative_cost_gt6 * m6["fn"]
        + parameters.early_action_cost * float(weight[early].sum())
        + parameters.critical_action_cost * float(weight[critical].sum())
    )
    metrics: dict[str, Any] = {
        "rows": int(len(decisions)),
        "vessel_calls": int(decisions["port_call_id"].nunique()),
        "gt3_precision": m3["precision"],
        "gt3_recall": m3["recall"],
        "gt3_precision_lift": m3["precision_lift"],
        "early_gt3_precision": me["precision"],
        "early_gt3_recall": me["recall"],
        "early_gt3_precision_lift": me["precision_lift"],
        "gt6_precision": m6["precision"],
        "gt6_recall": m6["recall"],
        "gt6_precision_lift": m6["precision_lift"],
        "alert_rate_pct": m3["positive_rate_pct"],
        "early_warning_rate_pct": me["positive_rate_pct"],
        "critical_action_rate_pct": m6["positive_rate_pct"],
        "early_warning_rows": int(early.sum()),
        "critical_action_rows": int(critical.sum()),
        "state_stability": float(1.0 - decisions["state_changed"].astype(float).mean()),
        "reference_cost": reference_cost,
        "policy_cost": policy_cost,
        "cost_reduction_pct": 100.0 * (reference_cost - policy_cost) / max(reference_cost, 1e-12),
        "candidate_id": parameters.policy_id,
        "early_mode": parameters.early_mode,
        "early_min_score": parameters.early_min_score,
        "early_top_k": parameters.early_top_k,
        "critical_top_k": parameters.critical_top_k,
        "hold_windows": parameters.hold_windows,
    }
    metrics.update(event_lead_metrics(decisions))
    metrics.update(event_lead_metrics(decisions, alert_column="early_warning", prefix="early_"))
    metrics["objective"] = policy_objective(metrics)
    return metrics


def policy_objective(metrics: dict[str, Any]) -> float:
    lead = float(metrics.get("median_event_lead_h") or 0.0)
    return float(
        35.0 * float(metrics["gt3_recall"])
        + 4.0 * min(float(metrics["gt3_precision_lift"]), 25.0)
        + 3.0 * min(float(metrics["gt6_precision_lift"]), 25.0)
        + 20.0 * float(metrics["gt6_recall"])
        + 25.0 * float(metrics.get("event_recall_at_least_6h", 0.0))
        + 1.5 * min(lead, 24.0)
        + 5.0 * float(metrics["state_stability"])
        + 0.20 * float(metrics["cost_reduction_pct"])
        - 0.25 * float(metrics["alert_rate_pct"])
    )


def add_selection_constraints(
    scorecard: pd.DataFrame, baseline: dict[str, Any]
) -> pd.DataFrame:
    output = scorecard.copy()
    baseline_lift3 = float(baseline["gt3_precision_lift"])
    baseline_recall3 = float(baseline["gt3_recall"])
    baseline_lift6 = float(baseline.get("gt6_precision_lift", 1.0))
    baseline_recall6 = float(baseline.get("gt6_recall", 0.0))
    baseline_lead = float(baseline.get("median_event_lead_h") or 0.0)
    baseline_recall_6h = float(baseline.get("event_recall_at_least_6h") or 0.0)
    baseline_alert_rate = float(baseline.get("gt3_alert_rate_pct", baseline.get("alert_rate_pct", 5.0)))
    output["gate_early_lift"] = output["gt3_precision_lift"].ge(max(2.0, 0.25 * baseline_lift3))
    output["gate_early_recall"] = output["gt3_recall"].ge(0.90 * baseline_recall3)
    output["gate_early_lead"] = (
        output["median_event_lead_h"].fillna(0.0).ge(baseline_lead + 1.0)
        | output["event_recall_at_least_6h"].ge(baseline_recall_6h + 0.02)
    )
    output["gate_critical_lift"] = output["gt6_precision_lift"].ge(max(2.0, 0.80 * baseline_lift6))
    output["gate_critical_recall"] = output["gt6_recall"].ge(max(0.05, 0.75 * baseline_recall6))
    output["gate_alert_burden"] = output["alert_rate_pct"].le(max(10.0, baseline_alert_rate + 3.0))
    output["gate_stability"] = output["state_stability"].ge(0.80)
    output["gate_cost"] = output["cost_reduction_pct"].ge(0.0)
    gate_columns = [
        "gate_early_lift", "gate_early_recall", "gate_early_lead",
        "gate_critical_lift", "gate_critical_recall", "gate_alert_burden",
        "gate_stability", "gate_cost",
    ]
    output["passes_constraints"] = output[gate_columns].all(axis=1)
    return output


def choose_constrained_policy(scorecard: pd.DataFrame) -> dict[str, Any]:
    if scorecard.empty:
        raise ValueError("Dual-stage policy scorecard is empty")
    eligible = scorecard.loc[scorecard["passes_constraints"]]
    pool = eligible if not eligible.empty else scorecard
    selected = pool.sort_values(
        ["objective", "gt3_recall", "gt6_precision_lift", "median_event_lead_h", "alert_rate_pct"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    return selected.to_dict()


def cluster_bootstrap_dual_metrics(
    decisions: pd.DataFrame, *, iterations: int = 500, random_state: int = 20260810
) -> dict[str, Any]:
    calls = decisions["port_call_id"].astype(str).drop_duplicates().to_numpy()
    if len(calls) < 20:
        raise ValueError("Cluster bootstrap requires at least 20 port calls")
    frame = decisions.copy()
    frame["_call"] = frame["port_call_id"].astype(str)
    rng = np.random.default_rng(random_state)
    names = (
        "gt3_precision", "gt3_recall", "gt3_precision_lift",
        "gt6_precision", "gt6_recall", "gt6_precision_lift", "alert_rate_pct",
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
        m3 = _weighted_binary_metrics(
            bootstrap["target_delay_gt_3h"].astype(bool).to_numpy(),
            bootstrap["alert_active"].astype(bool).to_numpy(), weight,
        )
        m6 = _weighted_binary_metrics(
            bootstrap["target_delay_gt_6h"].astype(bool).to_numpy(),
            bootstrap["critical_action"].astype(bool).to_numpy(), weight,
        )
        for prefix, metrics in (("gt3", m3), ("gt6", m6)):
            samples[f"{prefix}_precision"].append(metrics["precision"])
            samples[f"{prefix}_recall"].append(metrics["recall"])
            samples[f"{prefix}_precision_lift"].append(metrics["precision_lift"])
        samples["alert_rate_pct"].append(m3["positive_rate_pct"])
    result: dict[str, Any] = {
        "method": "PORT_CALL_CLUSTER_BOOTSTRAP_DUAL_CONTRACT",
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


def verify_b61d_v13_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "policy_version", "decision", "selected_candidate_id", "selection_role",
        "test_role", "test_used_for_fit_or_selection", "production_promotion_allowed",
        "automatic_action_allowed", "serving_rows", "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(f"B61D-v1.3 result contract misses: {missing}")
    if result["policy_version"] != POLICY_VERSION:
        raise RuntimeError("B61D-v1.3 policy version mismatch")
    if result["selection_role"] != "VALID_CALIBRATE":
        raise RuntimeError("B61D-v1.3 selection must use VALID_CALIBRATE only")
    if result["test_used_for_fit_or_selection"]:
        raise RuntimeError("B61D-v1.3 TEST leakage contract failed")
    if result["production_promotion_allowed"] or result["automatic_action_allowed"]:
        raise RuntimeError("B61D-v1.3 cannot enable production or automatic action")
    if int(result["serving_rows"]) <= 0:
        raise RuntimeError("B61D-v1.3 did not materialize shadow decisions")
    return result
