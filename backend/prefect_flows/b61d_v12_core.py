from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


POLICY_VERSION = "b61d-v1.2-state-conditional-policy-v1"
SOURCE_HSMM_VERSION = "b61d-v1.1-anchored-hsmm-v1"
SOURCE_POLICY_VERSION = "b61c-dynamic-alert-policy-v1.1"

HIDDEN_STATES = (
    "FLUID",
    "PRESSURE_BUILDING",
    "CONGESTED",
    "CRITICAL_DISRUPTION",
    "RECOVERY",
)
POLICY_MODES = (
    "STATE_STRICT",
    "POSTERIOR_STRICT",
    "CRITICAL_WITH_CONGESTED_BACKSTOP",
)


@dataclass(frozen=True)
class StatePolicyParameters:
    mode: str
    critical_probability_threshold: float
    alert_budget_pct: float
    hold_windows: int
    bucket_hours: int = 6
    congested_probability_threshold: float = 0.20
    congested_score_threshold: float = 0.65
    hysteresis_bonus: float = 0.05
    false_positive_cost_gt3: float = 1.0
    false_negative_cost_gt3: float = 15.0
    false_positive_cost_gt6: float = 2.0
    false_negative_cost_gt6: float = 40.0
    action_cost: float = 0.20
    high_score_threshold: float | None = None
    critical_score_threshold: float | None = None
    watch_score_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in POLICY_MODES:
            raise ValueError(f"Unsupported state-policy mode: {self.mode}")
        if not 0.0 <= self.critical_probability_threshold <= 1.0:
            raise ValueError("Critical probability threshold must be in [0,1]")
        if not 0.0 < self.alert_budget_pct <= 25.0:
            raise ValueError("Alert budget must be in (0,25]")
        if self.hold_windows not in (0, 1, 2, 3):
            raise ValueError("Hold windows must be 0, 1, 2 or 3")

    @property
    def gt3_budget_pct(self) -> float:
        return self.alert_budget_pct

    @property
    def gt6_budget_pct(self) -> float:
        return self.alert_budget_pct

    @property
    def policy_id(self) -> str:
        mode = {
            "STATE_STRICT": "STATE",
            "POSTERIOR_STRICT": "POST",
            "CRITICAL_WITH_CONGESTED_BACKSTOP": "BACKSTOP",
        }[self.mode]
        return (
            f"{mode}_P{self.critical_probability_threshold:g}_"
            f"B{self.alert_budget_pct:g}_H{self.hold_windows}_"
            f"{self.bucket_hours}H"
        )


def parse_numeric_grid(
    value: str | Iterable[float],
    *,
    minimum: float,
    maximum: float,
    name: str,
) -> list[float]:
    if isinstance(value, str):
        parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        parsed = [float(item) for item in value]
    result = sorted(set(parsed))
    if not result or any(item < minimum or item > maximum for item in result):
        raise ValueError(f"{name} must contain values in [{minimum},{maximum}]")
    return result


def parse_integer_grid(
    value: str | Iterable[int], *, allowed: set[int], name: str
) -> list[int]:
    if isinstance(value, str):
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        parsed = [int(item) for item in value]
    result = sorted(set(parsed))
    if not result or any(item not in allowed for item in result):
        raise ValueError(f"{name} must be a subset of {sorted(allowed)}")
    return result


def parse_modes(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        parsed = [item.strip().upper() for item in value.split(",") if item.strip()]
    else:
        parsed = [str(item).strip().upper() for item in value]
    result = list(dict.fromkeys(parsed))
    invalid = sorted(set(result).difference(POLICY_MODES))
    if not result or invalid:
        raise ValueError(f"Unsupported policy modes: {invalid}")
    return result


def _capacity(rows: int, budget_pct: float) -> int:
    if rows <= 0:
        return 0
    return max(1, int(math.ceil(rows * budget_pct / 100.0)))


def _candidate_mask(
    bucket: pd.DataFrame, parameters: StatePolicyParameters
) -> pd.Series:
    hidden = bucket["hsmm_state"].astype(str)
    critical_probability = pd.to_numeric(
        bucket["p_state_critical_disruption"], errors="coerce"
    ).fillna(0.0)
    threshold = parameters.critical_probability_threshold
    if parameters.mode == "STATE_STRICT":
        return hidden.eq("CRITICAL_DISRUPTION") & critical_probability.ge(threshold)
    if parameters.mode == "POSTERIOR_STRICT":
        return critical_probability.ge(threshold)
    congested_backstop = (
        hidden.eq("CONGESTED")
        & critical_probability.ge(parameters.congested_probability_threshold)
        & pd.to_numeric(
            bucket["base_critical_priority_score"], errors="coerce"
        ).fillna(0.0).ge(parameters.congested_score_threshold)
    )
    return (
        hidden.eq("CRITICAL_DISRUPTION") & critical_probability.ge(threshold)
    ) | congested_backstop


def _action_for_state(state: str) -> str:
    return {
        "NORMAL": "MONITOR",
        "WATCH": "REVIEW_STATE_AND_NEXT_WINDOW",
        "CRITICAL": "ESCALATE_TO_OPERATIONS_CONTROL",
    }[state]


def apply_state_conditional_policy(
    frame: pd.DataFrame,
    parameters: StatePolicyParameters,
) -> pd.DataFrame:
    required = {
        "evaluation_role", "decision_at", "port_call_id", "hsmm_state",
        "p_state_critical_disruption", "p_state_congested",
        "base_temporal_priority_score", "base_critical_priority_score",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"State-policy source misses columns: {missing}")
    if frame.empty:
        return frame.copy()
    output = frame.sort_values(
        ["evaluation_role", "decision_at", "port_call_id"]
    ).reset_index(drop=True).copy()
    rows = len(output)
    identifiers = output["port_call_id"].astype(str).to_numpy()
    roles = output["evaluation_role"].astype(str).to_numpy()
    hidden_states = output["hsmm_state"].astype(str).to_numpy()
    decision_keys = pd.to_datetime(output["decision_at"], utc=True).astype("int64").to_numpy()
    critical_probability = pd.to_numeric(
        output["p_state_critical_disruption"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    base_critical = pd.to_numeric(
        output["base_critical_priority_score"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    base_temporal = pd.to_numeric(
        output["base_temporal_priority_score"], errors="coerce"
    ).fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=float)
    priority = np.clip(
        0.55 * critical_probability + 0.25 * base_critical + 0.20 * base_temporal,
        0.0,
        1.0,
    )
    base_candidate = _candidate_mask(output, parameters).to_numpy(dtype=bool)

    previous = np.full(rows, "NORMAL", dtype=object)
    prior_hold = np.zeros(rows, dtype=int)
    retained = np.zeros(rows, dtype=bool)
    candidate = np.zeros(rows, dtype=bool)
    reason = np.full(rows, "NOT_CANDIDATE", dtype=object)
    effective = priority.copy()
    rank = np.zeros(rows, dtype=int)
    selected = np.zeros(rows, dtype=bool)
    state = np.full(rows, "NORMAL", dtype=object)
    active_calls = np.zeros(rows, dtype=int)
    capacities = np.zeros(rows, dtype=int)

    bucket_start = np.ones(rows, dtype=bool)
    bucket_start[1:] = (
        (roles[1:] != roles[:-1]) | (decision_keys[1:] != decision_keys[:-1])
    )
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
        bucket_ids = identifiers[start:end]
        bucket_hidden = hidden_states[start:end]
        bucket_previous = np.asarray(
            [previous_state.get(identifier, "NORMAL") for identifier in bucket_ids],
            dtype=object,
        )
        bucket_hold = np.fromiter(
            (hold_remaining.get(identifier, 0) for identifier in bucket_ids),
            dtype=int,
            count=end - start,
        )
        previous[start:end] = bucket_previous
        prior_hold[start:end] = bucket_hold
        bucket_retained = (
            (bucket_hold > 0)
            & (bucket_previous == "CRITICAL")
            & np.isin(bucket_hidden, ["CRITICAL_DISRUPTION", "RECOVERY"])
            & (priority[start:end] >= 0.25)
        )
        retained[start:end] = bucket_retained
        bucket_candidate = base_candidate[start:end] | bucket_retained
        candidate[start:end] = bucket_candidate
        reason[start:end][bucket_retained] = "HYSTERESIS_HOLD"
        reason[start:end][base_candidate[start:end]] = "STATE_OR_POSTERIOR"
        bucket_effective = (
            priority[start:end]
            + parameters.hysteresis_bonus * bucket_retained.astype(float)
        )
        effective[start:end] = bucket_effective
        order = np.argsort(-bucket_effective, kind="stable")
        bucket_rank = np.empty(end - start, dtype=int)
        bucket_rank[order] = np.arange(1, end - start + 1)
        rank[start:end] = bucket_rank
        capacity = _capacity(end - start, parameters.alert_budget_pct)
        active_calls[start:end] = end - start
        capacities[start:end] = capacity
        candidate_positions = np.flatnonzero(bucket_candidate)
        selected_positions = candidate_positions[
            np.argsort(-bucket_effective[candidate_positions], kind="stable")[:capacity]
        ]
        bucket_selected = np.zeros(end - start, dtype=bool)
        bucket_selected[selected_positions] = True
        selected[start:end] = bucket_selected
        bucket_watch = (
            np.isin(
                bucket_hidden,
                ["PRESSURE_BUILDING", "CONGESTED", "CRITICAL_DISRUPTION"],
            )
            | ((bucket_hidden == "RECOVERY") & (bucket_hold > 0))
        ) & ~bucket_selected
        bucket_state = np.where(
            bucket_selected, "CRITICAL", np.where(bucket_watch, "WATCH", "NORMAL")
        )
        state[start:end] = bucket_state
        for position, identifier in enumerate(bucket_ids):
            previous_state[identifier] = str(bucket_state[position])
            if bucket_selected[position] and base_candidate[start + position]:
                hold_remaining[identifier] = parameters.hold_windows
            elif bucket_selected[position]:
                hold_remaining[identifier] = max(int(bucket_hold[position]) - 1, 0)
            else:
                hold_remaining[identifier] = 0

    output["previous_state"] = previous
    output["state_priority_score"] = priority
    output["policy_candidate"] = candidate
    output["candidate_reason"] = reason
    output["effective_priority_score"] = effective
    output["rank_in_bucket"] = rank
    output["state"] = state
    output["state_changed"] = state != previous
    output["alert_active"] = selected
    output["new_alert"] = selected & (previous != "CRITICAL")
    output["action_code"] = [_action_for_state(item) for item in state]
    output["active_calls"] = active_calls
    output["alert_capacity"] = capacities
    output["hold_windows"] = parameters.hold_windows
    output["alert_budget_pct"] = parameters.alert_budget_pct
    output["critical_probability_threshold"] = parameters.critical_probability_threshold
    output["policy_mode"] = parameters.mode
    output["policy_id"] = parameters.policy_id
    output["source_mode"] = "HISTORICAL_REPLAY_SHADOW"
    output["production_claim_allowed"] = False
    output["automatic_action_allowed"] = False
    return output


def event_lead_metrics(decisions: pd.DataFrame) -> dict[str, Any]:
    positive = decisions.loc[decisions["target_delay_gt_3h"].astype(bool)]
    positive_calls = int(positive["port_call_id"].nunique())
    alerted = positive.loc[
        positive["alert_active"]
        & pd.to_numeric(
            positive["target_breach_or_censor_h"], errors="coerce"
        ).ge(0.0)
    ]
    leads = (
        alerted.groupby("port_call_id")["target_breach_or_censor_h"].max()
        if not alerted.empty else pd.Series(dtype=float)
    )
    result: dict[str, Any] = {
        "positive_calls": positive_calls,
        "alerted_positive_calls": int(len(leads)),
        "event_recall_any": float(len(leads) / max(positive_calls, 1)),
        "median_event_lead_h": float(leads.median()) if not leads.empty else None,
        "p25_event_lead_h": float(leads.quantile(0.25)) if not leads.empty else None,
    }
    for horizon in (6, 12, 24):
        result[f"event_recall_at_least_{horizon}h"] = float(
            leads.ge(horizon).sum() / max(positive_calls, 1)
        )
    return result


def policy_objective(metrics: dict[str, Any]) -> float:
    lead = float(metrics.get("median_event_lead_h") or 0.0)
    return float(
        float(metrics["cost_reduction_pct"])
        + 30.0 * float(metrics["gt3_recall"])
        + 3.0 * min(float(metrics["gt3_precision_lift"]), 25.0)
        + 25.0 * float(metrics.get("event_recall_at_least_6h", 0.0))
        + 1.5 * min(lead, 24.0)
        + 5.0 * float(metrics["state_stability"])
    )


def add_selection_constraints(
    scorecard: pd.DataFrame,
    baseline: dict[str, Any],
) -> pd.DataFrame:
    output = scorecard.copy()
    baseline_lift = float(baseline["gt3_precision_lift"])
    baseline_recall = float(baseline["gt3_recall"])
    baseline_lead = float(baseline.get("median_event_lead_h") or 0.0)
    baseline_recall_6h = float(baseline.get("event_recall_at_least_6h") or 0.0)
    output["gate_lift"] = output["gt3_precision_lift"].ge(0.80 * baseline_lift)
    output["gate_recall"] = output["gt3_recall"].ge(0.90 * baseline_recall)
    output["gate_stability"] = output["state_stability"].ge(0.80)
    output["gate_cost"] = output["cost_reduction_pct"].ge(0.0)
    output["gate_lead"] = (
        output["median_event_lead_h"].fillna(0.0).ge(baseline_lead + 1.0)
        | output["event_recall_at_least_6h"].ge(baseline_recall_6h + 0.02)
    )
    output["passes_constraints"] = output[
        ["gate_lift", "gate_recall", "gate_stability", "gate_cost", "gate_lead"]
    ].all(axis=1)
    return output


def choose_constrained_policy(scorecard: pd.DataFrame) -> dict[str, Any]:
    if scorecard.empty:
        raise ValueError("State-policy scorecard is empty")
    eligible = scorecard.loc[scorecard["passes_constraints"]]
    pool = eligible if not eligible.empty else scorecard
    selected = pool.sort_values(
        [
            "objective", "gt3_precision_lift", "gt3_recall",
            "event_recall_at_least_6h", "alert_budget_pct",
        ],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    return selected.to_dict()


def cluster_bootstrap_metrics(
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
    samples: dict[str, list[float]] = {
        "gt3_precision": [], "gt3_recall": [], "gt3_precision_lift": [],
        "alert_rate_pct": [],
    }
    for _ in range(iterations):
        sampled = rng.choice(calls, size=len(calls), replace=True)
        multiplicity = pd.Series(sampled).value_counts()
        bootstrap = frame.loc[frame["_call"].isin(multiplicity.index)].copy()
        bootstrap["_weight"] = (
            pd.to_numeric(bootstrap["decision_weight"], errors="coerce").fillna(0.0)
            * bootstrap["_call"].map(multiplicity).astype(float)
        )
        actual = bootstrap["target_delay_gt_3h"].astype(bool).to_numpy()
        predicted = bootstrap["alert_active"].astype(bool).to_numpy()
        weight = bootstrap["_weight"].to_numpy(dtype=float)
        tp = float(weight[actual & predicted].sum())
        fp = float(weight[~actual & predicted].sum())
        fn = float(weight[actual & ~predicted].sum())
        total = float(weight.sum())
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        prevalence = (tp + fn) / max(total, 1e-12)
        samples["gt3_precision"].append(precision)
        samples["gt3_recall"].append(recall)
        samples["gt3_precision_lift"].append(precision / max(prevalence, 1e-12))
        samples["alert_rate_pct"].append(
            100.0 * float(weight[predicted].sum()) / max(total, 1e-12)
        )
    result: dict[str, Any] = {
        "method": "PORT_CALL_CLUSTER_BOOTSTRAP",
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


def verify_b61d_v12_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "policy_version", "decision", "selected_candidate_id",
        "selection_role", "test_role", "test_used_for_fit_or_selection",
        "production_promotion_allowed", "automatic_action_allowed",
        "serving_rows", "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(f"B61D-v1.2 result contract misses: {missing}")
    if result["policy_version"] != POLICY_VERSION:
        raise RuntimeError("B61D-v1.2 policy version mismatch")
    if result["selection_role"] != "VALID_CALIBRATE":
        raise RuntimeError("B61D-v1.2 selection must use VALID_CALIBRATE only")
    if result["test_used_for_fit_or_selection"]:
        raise RuntimeError("B61D-v1.2 TEST leakage contract failed")
    if result["production_promotion_allowed"] or result["automatic_action_allowed"]:
        raise RuntimeError("B61D-v1.2 cannot enable production or automatic action")
    if int(result["serving_rows"]) <= 0:
        raise RuntimeError("B61D-v1.2 did not materialize shadow decisions")
    return result
