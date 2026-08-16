from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


POLICY_VERSION = "b61c-dynamic-alert-policy-v1.1"
SOURCE_MODEL_VERSION = "b61b-v2.1-maritime-recalibration-only-v1"
STATES = ("NORMAL", "WATCH", "HIGH_RISK", "CRITICAL")


@dataclass(frozen=True)
class PolicyParameters:
    gt3_budget_pct: float
    gt6_budget_pct: float
    bucket_hours: int = 6
    hysteresis_bonus: float = 0.03
    watch_multiplier: float = 2.0
    false_positive_cost_gt3: float = 1.0
    false_negative_cost_gt3: float = 15.0
    false_positive_cost_gt6: float = 2.0
    false_negative_cost_gt6: float = 40.0
    action_cost: float = 0.20
    high_score_threshold: float | None = None
    critical_score_threshold: float | None = None
    watch_score_threshold: float | None = None

    @property
    def policy_id(self) -> str:
        return (
            f"GT3_{self.gt3_budget_pct:g}PCT_"
            f"GT6_{self.gt6_budget_pct:g}PCT_"
            f"{self.bucket_hours}H"
        )


def parse_budgets(value: str | Iterable[float]) -> list[float]:
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
        budgets = [float(item) for item in parts]
    else:
        budgets = [float(item) for item in value]
    result = sorted(set(budgets))
    if not result or any(not 0.0 < item <= 25.0 for item in result):
        raise ValueError("Alert budgets must be unique percentages in (0, 25]")
    return result


def build_decision_grid(frame: pd.DataFrame, bucket_hours: int = 6) -> pd.DataFrame:
    required = {
        "port_call_id",
        "landmark_at",
        "evaluation_role",
        "early_warning_eligible",
        "p_delay_gt3",
        "p_delay_gt6",
        "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h",
        "target_delay_gt_3h",
        "target_delay_gt_6h",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Decision source misses columns: {missing}")
    output = frame.loc[
        frame["early_warning_eligible"].fillna(False).astype(bool)
    ].copy()
    output["landmark_at"] = pd.to_datetime(output["landmark_at"], utc=True)
    output["decision_at"] = output["landmark_at"].dt.floor(f"{bucket_hours}h")
    output = (
        output.sort_values(["evaluation_role", "decision_at", "port_call_id", "landmark_at"])
        .drop_duplicates(
            ["evaluation_role", "decision_at", "port_call_id"], keep="last"
        )
        .reset_index(drop=True)
    )
    if output.empty:
        raise ValueError("No eligible decision landmarks are available")
    observations = output.groupby(
        ["evaluation_role", "port_call_id"], sort=False
    )["port_call_id"].transform("size")
    output["decision_weight"] = 1.0 / observations.clip(lower=1).astype(float)
    return output


def add_temporal_scores(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    components = (
        "p_delay_gt3",
        "p_delay_gt6",
        "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h",
    )
    for column in components:
        output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)
        output[column] = output[column].clip(0.0, 1.0)
    output["temporal_priority_score"] = (
        0.45 * output["p_delay_gt3"]
        + 0.30 * output["p_gt3_breach_within_6h"]
        + 0.15 * output["p_gt3_breach_within_12h"]
        + 0.10 * output["p_gt3_breach_within_24h"]
    )
    output["critical_priority_score"] = (
        0.65 * output["p_delay_gt6"]
        + 0.35 * output["p_gt3_breach_within_6h"]
    )
    return output


def _capacity(rows: int, budget_pct: float, upper: int | None = None) -> int:
    if rows <= 0:
        return 0
    value = max(1, int(math.ceil(rows * budget_pct / 100.0)))
    return min(value, upper) if upper is not None else value


def _action_for_state(state: str) -> str:
    return {
        "NORMAL": "MONITOR",
        "WATCH": "REVIEW_NEXT_DECISION_WINDOW",
        "HIGH_RISK": "VERIFY_BERTH_EQUIPMENT_AND_PRIORITY",
        "CRITICAL": "ESCALATE_TO_OPERATIONS_CONTROL",
    }[state]


def apply_dynamic_policy(
    frame: pd.DataFrame,
    parameters: PolicyParameters,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ordered = frame.sort_values(
        ["evaluation_role", "decision_at", "port_call_id"]
    ).copy()
    decisions: list[pd.DataFrame] = []
    for role, role_frame in ordered.groupby("evaluation_role", sort=False):
        previous_state: dict[str, str] = {}
        for decision_at, bucket in role_frame.groupby("decision_at", sort=True):
            bucket = bucket.copy()
            active_calls = len(bucket)
            high_capacity = _capacity(active_calls, parameters.gt3_budget_pct)
            critical_capacity = _capacity(
                active_calls, parameters.gt6_budget_pct, upper=high_capacity
            )
            prior = bucket["port_call_id"].astype(str).map(previous_state).fillna("NORMAL")
            bucket["previous_state"] = prior
            retained = prior.isin(["HIGH_RISK", "CRITICAL"]).astype(float)
            bucket["effective_priority_score"] = (
                bucket["temporal_priority_score"]
                + parameters.hysteresis_bonus * retained
            )
            bucket["rank_in_bucket"] = bucket["effective_priority_score"].rank(
                method="first", ascending=False
            ).astype(int)
            high_threshold = (
                parameters.high_score_threshold
                if parameters.high_score_threshold is not None
                else -math.inf
            )
            high_candidates = bucket.loc[
                bucket["effective_priority_score"].ge(high_threshold)
            ]
            high_ids = set(
                high_candidates.nlargest(
                    min(high_capacity, len(high_candidates)),
                    "effective_priority_score",
                )[
                    "port_call_id"
                ].astype(str)
            )
            high = bucket["port_call_id"].astype(str).isin(high_ids)
            critical_candidates = bucket.loc[high].copy()
            critical_candidates["effective_critical_score"] = (
                critical_candidates["critical_priority_score"]
                + parameters.hysteresis_bonus
                * critical_candidates["previous_state"].eq("CRITICAL").astype(float)
            )
            critical_threshold = (
                parameters.critical_score_threshold
                if parameters.critical_score_threshold is not None
                else -math.inf
            )
            critical_candidates = critical_candidates.loc[
                critical_candidates["effective_critical_score"].ge(
                    critical_threshold
                )
            ]
            critical_ids = set(
                critical_candidates.nlargest(
                    min(critical_capacity, len(critical_candidates)),
                    "effective_critical_score",
                )["port_call_id"].astype(str)
            )
            watch_capacity = min(
                active_calls,
                max(high_capacity, int(math.ceil(high_capacity * parameters.watch_multiplier))),
            )
            watch_threshold = (
                parameters.watch_score_threshold
                if parameters.watch_score_threshold is not None
                else -math.inf
            )
            watch_candidates = bucket.loc[
                bucket["effective_priority_score"].ge(watch_threshold)
            ]
            watch_ids = set(
                watch_candidates.nlargest(
                    min(watch_capacity, len(watch_candidates)),
                    "effective_priority_score",
                )[
                    "port_call_id"
                ].astype(str)
            )
            identifiers = bucket["port_call_id"].astype(str)
            state = np.select(
                [identifiers.isin(critical_ids), identifiers.isin(high_ids), identifiers.isin(watch_ids)],
                ["CRITICAL", "HIGH_RISK", "WATCH"],
                default="NORMAL",
            )
            bucket["state"] = state
            bucket["state_changed"] = bucket["state"].ne(bucket["previous_state"])
            bucket["alert_active"] = bucket["state"].isin(["HIGH_RISK", "CRITICAL"])
            bucket["new_alert"] = bucket["alert_active"] & ~bucket[
                "previous_state"
            ].isin(["HIGH_RISK", "CRITICAL"])
            bucket["action_code"] = bucket["state"].map(_action_for_state)
            bucket["active_calls"] = active_calls
            bucket["high_capacity"] = high_capacity
            bucket["critical_capacity"] = critical_capacity
            bucket["gt3_budget_pct"] = parameters.gt3_budget_pct
            bucket["gt6_budget_pct"] = parameters.gt6_budget_pct
            bucket["policy_id"] = parameters.policy_id
            bucket["source_mode"] = "HISTORICAL_REPLAY_SHADOW"
            bucket["production_claim_allowed"] = False
            bucket["automatic_action_allowed"] = False
            decisions.append(bucket)
            previous_state.update(
                dict(zip(identifiers, bucket["state"].astype(str)))
            )
    return pd.concat(decisions, ignore_index=True)


def _weighted_confusion(
    actual: np.ndarray, predicted: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    actual = np.asarray(actual, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    weight = np.asarray(weight, dtype=float)
    tp = float(weight[actual & predicted].sum())
    fp = float(weight[~actual & predicted].sum())
    fn = float(weight[actual & ~predicted].sum())
    tn = float(weight[~actual & ~predicted].sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    prevalence = (tp + fn) / max(tp + fp + fn + tn, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "prevalence": prevalence,
        "precision_lift": precision / max(prevalence, 1e-12),
    }


def evaluate_policy(
    decisions: pd.DataFrame,
    parameters: PolicyParameters,
) -> dict[str, Any]:
    if decisions.empty:
        raise ValueError("Cannot evaluate an empty policy replay")
    weight = decisions["decision_weight"].to_numpy(dtype=float)
    gt3 = _weighted_confusion(
        decisions["target_delay_gt_3h"].astype(bool).to_numpy(),
        decisions["alert_active"].astype(bool).to_numpy(),
        weight,
    )
    gt6 = _weighted_confusion(
        decisions["target_delay_gt_6h"].astype(bool).to_numpy(),
        decisions["state"].eq("CRITICAL").to_numpy(),
        weight,
    )
    total_weight = float(weight.sum())
    alert_weight = float(weight[decisions["alert_active"].to_numpy()].sum())
    critical_weight = float(weight[decisions["state"].eq("CRITICAL").to_numpy()].sum())
    baseline_cost = (
        gt3["tp"] + gt3["fn"]
    ) * parameters.false_negative_cost_gt3 + (
        gt6["tp"] + gt6["fn"]
    ) * parameters.false_negative_cost_gt6
    policy_cost = (
        gt3["fp"] * parameters.false_positive_cost_gt3
        + gt3["fn"] * parameters.false_negative_cost_gt3
        + gt6["fp"] * parameters.false_positive_cost_gt6
        + gt6["fn"] * parameters.false_negative_cost_gt6
        + alert_weight * parameters.action_cost
    )
    ordered = decisions.sort_values(
        ["evaluation_role", "port_call_id", "decision_at"]
    )
    previous = ordered.groupby(["evaluation_role", "port_call_id"], sort=False)[
        "state"
    ].shift()
    comparable = previous.notna()
    transitions = int(
        (ordered.loc[comparable, "state"].to_numpy() != previous.loc[comparable].to_numpy()).sum()
    )
    possible_transitions = max(int(comparable.sum()), 1)
    true_alert = decisions["alert_active"] & decisions["target_delay_gt_3h"].astype(bool)
    lead = pd.to_numeric(
        decisions.loc[true_alert, "target_breach_or_censor_h"], errors="coerce"
    )
    lead = lead.loc[lead.ge(0.0)]
    days = max(
        (
            pd.to_datetime(decisions["decision_at"], utc=True).max()
            - pd.to_datetime(decisions["decision_at"], utc=True).min()
        ).total_seconds()
        / 86400.0,
        1.0,
    )
    return {
        "policy_id": parameters.policy_id,
        "role": str(decisions["evaluation_role"].iloc[0]),
        "gt3_budget_pct": parameters.gt3_budget_pct,
        "gt6_budget_pct": parameters.gt6_budget_pct,
        "high_score_threshold": parameters.high_score_threshold,
        "critical_score_threshold": parameters.critical_score_threshold,
        "watch_score_threshold": parameters.watch_score_threshold,
        "decision_rows": int(len(decisions)),
        "port_calls": int(decisions["port_call_id"].nunique()),
        "decision_buckets": int(decisions["decision_at"].nunique()),
        "alert_rows": int(decisions["alert_active"].sum()),
        "critical_rows": int(decisions["state"].eq("CRITICAL").sum()),
        "new_alert_rows": int(decisions["new_alert"].sum()),
        "alert_rate_pct": 100.0 * alert_weight / max(total_weight, 1e-12),
        "critical_rate_pct": 100.0 * critical_weight / max(total_weight, 1e-12),
        "new_alerts_per_day": float(decisions["new_alert"].sum()) / days,
        "state_stability": 1.0 - transitions / possible_transitions,
        "median_true_alert_lead_h": float(lead.median()) if not lead.empty else None,
        "baseline_cost_units": baseline_cost,
        "policy_cost_units": policy_cost,
        "cost_reduction_pct": 100.0 * (baseline_cost - policy_cost) / max(baseline_cost, 1e-12),
        **{f"gt3_{name}": value for name, value in gt3.items()},
        **{f"gt6_{name}": value for name, value in gt6.items()},
    }


def policy_objective(metrics: dict[str, Any]) -> float:
    penalty = 0.0
    if metrics["alert_rows"] <= 0:
        penalty += 10_000.0
    if metrics["critical_rows"] <= 0:
        penalty += 5_000.0
    if metrics["gt3_precision_lift"] < 1.0:
        penalty += 1_000.0 * (1.0 - metrics["gt3_precision_lift"])
    if metrics["gt6_precision_lift"] < 1.0:
        penalty += 500.0 * (1.0 - metrics["gt6_precision_lift"])
    return float(
        metrics["cost_reduction_pct"]
        + 20.0 * metrics["gt3_recall"]
        + 10.0 * metrics["gt6_recall"]
        + 5.0 * metrics["state_stability"]
        - penalty
    )


def choose_policy(scorecard: pd.DataFrame) -> dict[str, Any]:
    valid = scorecard.loc[scorecard["role"].eq("VALID_CALIBRATE")].copy()
    if valid.empty:
        raise ValueError("VALID_CALIBRATE policy scorecard is empty")
    valid["objective"] = valid.apply(
        lambda row: policy_objective(row.to_dict()), axis=1
    )
    selected = valid.sort_values(
        ["objective", "gt3_precision_lift", "gt3_recall", "gt3_budget_pct"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return selected.to_dict()
