from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


POLICY_VERSION = "b61e-capacity-aware-temporal-ranking-v1"
SOURCE_MODEL_VERSION = "b61b-v2.1-maritime-recalibration-only-v1"
SOURCE_HSMM_VERSION = "b61d-v1.1-anchored-hsmm-v1"
VALID_ROLES = ("VALID_SELECT", "VALID_CALIBRATE")
TEST_ROLE = "TEST_DIAGNOSTIC_ONLY"
SCORE_NAMES = ("HAZARD_24H", "P_GT3", "TEMPORAL_HAZARD_MOE")


@dataclass(frozen=True)
class RankingParameters:
    score_name: str
    top_k: int
    bucket_hours: int = 6

    def __post_init__(self) -> None:
        if self.score_name not in SCORE_NAMES:
            raise ValueError(f"Unsupported score_name: {self.score_name}")
        if self.top_k not in range(1, 6):
            raise ValueError("top_k must be between 1 and 5")
        if self.bucket_hours not in (3, 6, 12):
            raise ValueError("bucket_hours must be 3, 6 or 12")

    @property
    def policy_id(self) -> str:
        return f"RANK_{self.score_name}_K{self.top_k}_{self.bucket_hours}H"


@dataclass(frozen=True)
class RankingContract:
    min_precision: float = 0.30
    min_recall: float = 0.55
    min_precision_lift: float = 2.0
    min_f1: float = 0.40
    min_event_recall: float = 0.55
    max_top_k: int = 2
    min_bootstrap_precision_lower: float = 0.20
    min_bootstrap_recall_lower: float = 0.45


def parse_score_names(value: str | Iterable[str]) -> list[str]:
    values = value.split(",") if isinstance(value, str) else list(value)
    parsed = [str(item).strip().upper() for item in values if str(item).strip()]
    invalid = sorted(set(parsed).difference(SCORE_NAMES))
    if not parsed or invalid:
        raise ValueError(f"Invalid ranking scores: {invalid or parsed}")
    return list(dict.fromkeys(parsed))


def parse_top_ks(value: str | Iterable[int]) -> list[int]:
    values = value.split(",") if isinstance(value, str) else list(value)
    parsed = [int(item) for item in values]
    if not parsed or any(item not in range(1, 6) for item in parsed):
        raise ValueError("top_ks must contain integers between 1 and 5")
    return list(dict.fromkeys(parsed))


def prepare_ranking_frame(frame: pd.DataFrame, bucket_hours: int = 6) -> pd.DataFrame:
    required = {
        "evaluation_role", "port_call_id", "landmark_at", "p_delay_gt3",
        "p_gt3_breach_within_6h", "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h", "target_event_within_24h",
        "target_delay_gt_3h", "target_breach_or_censor_h",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Ranking source misses columns: {missing}")
    if frame.empty:
        raise ValueError("Ranking source is empty")
    output = frame.copy()
    output["landmark_at"] = pd.to_datetime(output["landmark_at"], utc=True)
    output["decision_at"] = output["landmark_at"].dt.floor(f"{bucket_hours}h")
    output = output.sort_values(
        ["evaluation_role", "decision_at", "port_call_id", "landmark_at"],
        ascending=[True, True, True, False],
    )
    output = output.drop_duplicates(
        ["evaluation_role", "decision_at", "port_call_id"], keep="first"
    ).reset_index(drop=True)
    output["target_event_within_24h"] = output["target_event_within_24h"].fillna(False).astype(bool)
    output["target_delay_gt_3h"] = output["target_delay_gt_3h"].fillna(False).astype(bool)
    return output


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return (
        pd.to_numeric(frame[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .clip(0.0, 1.0)
        .to_numpy(float)
    )


def ranking_score(frame: pd.DataFrame, score_name: str) -> np.ndarray:
    if score_name == "HAZARD_24H":
        return _numeric(frame, "p_gt3_breach_within_24h")
    if score_name == "P_GT3":
        return _numeric(frame, "p_delay_gt3")
    if score_name == "TEMPORAL_HAZARD_MOE":
        h6 = _numeric(frame, "p_gt3_breach_within_6h")
        h12 = _numeric(frame, "p_gt3_breach_within_12h")
        h24 = _numeric(frame, "p_gt3_breach_within_24h")
        p3 = _numeric(frame, "p_delay_gt3")
        return np.clip(0.10 * h6 + 0.20 * h12 + 0.50 * h24 + 0.20 * p3, 0.0, 1.0)
    raise ValueError(f"Unsupported score_name: {score_name}")


def apply_capacity_ranking(
    prepared: pd.DataFrame, parameters: RankingParameters
) -> pd.DataFrame:
    output = prepared.copy()
    output["risk_score"] = ranking_score(output, parameters.score_name)
    output = output.sort_values(
        ["evaluation_role", "decision_at", "risk_score", "port_call_id"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    grouping = output.groupby(["evaluation_role", "decision_at"], sort=False)
    output["rank_in_window"] = grouping.cumcount() + 1
    output["active_calls"] = grouping["port_call_id"].transform("size").astype(int)
    output["capacity"] = np.minimum(parameters.top_k, output["active_calls"]).astype(int)
    output["watchlist_selected"] = output["rank_in_window"] <= output["capacity"]
    output["score_name"] = parameters.score_name
    output["top_k"] = parameters.top_k
    output["bucket_hours"] = parameters.bucket_hours
    output["policy_id"] = parameters.policy_id
    output["action_tier"] = np.where(
        output["watchlist_selected"], "PRIORITY_REVIEW", "MONITOR"
    )
    output["reason_code"] = np.where(
        output["watchlist_selected"],
        "TOP_K_TEMPORAL_RISK_ALL_STATES",
        "BELOW_WINDOW_CAPACITY",
    )
    output["production_claim_allowed"] = False
    output["automatic_action_allowed"] = False
    return output


def _binary_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    tp = int(np.sum(actual & predicted))
    fp = int(np.sum(~actual & predicted))
    fn = int(np.sum(actual & ~predicted))
    tn = int(np.sum(~actual & ~predicted))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    prevalence = (tp + fn) / max(tp + fp + fn + tn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "prevalence": prevalence,
        "precision_lift": precision / max(prevalence, 1e-12),
    }


def evaluate_ranking(decisions: pd.DataFrame, parameters: RankingParameters) -> dict[str, Any]:
    actual = decisions["target_event_within_24h"].astype(bool).to_numpy()
    selected = decisions["watchlist_selected"].astype(bool).to_numpy()
    binary = _binary_metrics(actual, selected)
    positive_calls = decisions.loc[actual, "port_call_id"].astype(str).nunique()
    detected = decisions.loc[actual & selected].copy()
    lead = pd.to_numeric(detected["target_breach_or_censor_h"], errors="coerce")
    detected = detected.assign(_lead=lead).loc[lead.ge(0.0)]
    call_lead = (
        detected.groupby(detected["port_call_id"].astype(str))["_lead"].max()
        if not detected.empty else pd.Series(dtype=float)
    )
    event_recall = len(call_lead) / max(positive_calls, 1)
    metrics: dict[str, Any] = {
        **binary,
        "rows": int(len(decisions)),
        "vessel_calls": int(decisions["port_call_id"].nunique()),
        "windows": int(decisions["decision_at"].nunique()),
        "selected_rows": int(selected.sum()),
        "review_rate_pct": 100.0 * float(selected.mean()),
        "average_active_calls": float(
            decisions.groupby("decision_at")["active_calls"].first().mean()
        ),
        "average_reviews_per_window": float(
            decisions.groupby("decision_at")["watchlist_selected"].sum().mean()
        ),
        "positive_calls": int(positive_calls),
        "detected_positive_calls": int(len(call_lead)),
        "event_recall_any": float(event_recall),
        "event_recall_at_least_1h": float(call_lead.ge(1.0).sum() / max(positive_calls, 1)),
        "event_recall_at_least_3h": float(call_lead.ge(3.0).sum() / max(positive_calls, 1)),
        "event_recall_at_least_6h": float(call_lead.ge(6.0).sum() / max(positive_calls, 1)),
        "median_lead_h": float(call_lead.median()) if not call_lead.empty else None,
        "score_name": parameters.score_name,
        "top_k": parameters.top_k,
        "bucket_hours": parameters.bucket_hours,
        "candidate_id": parameters.policy_id,
        "all_states_eligible": True,
    }
    metrics["selection_objective"] = float(
        50.0 * metrics["f1"]
        + 20.0 * metrics["event_recall_any"]
        + 2.0 * min(metrics["precision_lift"], 10.0)
        - 0.05 * metrics["review_rate_pct"]
    )
    return metrics


def contract_gates(metrics: dict[str, Any], contract: RankingContract | None = None) -> dict[str, bool]:
    contract = contract or RankingContract()
    return {
        "gate_precision": float(metrics["precision"]) >= contract.min_precision,
        "gate_recall": float(metrics["recall"]) >= contract.min_recall,
        "gate_precision_lift": float(metrics["precision_lift"]) >= contract.min_precision_lift,
        "gate_f1": float(metrics["f1"]) >= contract.min_f1,
        "gate_event_recall": float(metrics["event_recall_any"]) >= contract.min_event_recall,
        "gate_capacity": int(metrics["top_k"]) <= contract.max_top_k,
        "gate_all_states": bool(metrics["all_states_eligible"]),
    }


def select_score_on_valid_select(
    prepared: pd.DataFrame, score_names: list[str], bucket_hours: int
) -> tuple[pd.DataFrame, str]:
    source = prepared.loc[prepared["evaluation_role"].eq("VALID_SELECT")]
    rows = []
    for score_name in score_names:
        parameters = RankingParameters(score_name=score_name, top_k=1, bucket_hours=bucket_hours)
        rows.append(evaluate_ranking(apply_capacity_ranking(source, parameters), parameters))
    scorecard = pd.DataFrame(rows)
    selected = scorecard.sort_values(
        ["selection_objective", "f1", "precision", "event_recall_any"],
        ascending=False,
    ).iloc[0]
    scorecard["selected_score"] = scorecard["score_name"].eq(selected["score_name"])
    scorecard["selection_stage"] = "VALID_SELECT_SCORE_SELECTION"
    return scorecard, str(selected["score_name"])


def calibrate_capacity_on_valid_calibrate(
    prepared: pd.DataFrame,
    score_name: str,
    top_ks: list[int],
    bucket_hours: int,
    contract: RankingContract | None = None,
) -> tuple[pd.DataFrame, RankingParameters]:
    contract = contract or RankingContract()
    source = prepared.loc[prepared["evaluation_role"].eq("VALID_CALIBRATE")]
    rows = []
    for top_k in top_ks:
        parameters = RankingParameters(score_name=score_name, top_k=top_k, bucket_hours=bucket_hours)
        metrics = evaluate_ranking(apply_capacity_ranking(source, parameters), parameters)
        gates = contract_gates(metrics, contract)
        rows.append({**metrics, **gates, "passes_contract": all(gates.values())})
    scorecard = pd.DataFrame(rows)
    eligible = scorecard.loc[scorecard["passes_contract"]]
    pool = eligible if not eligible.empty else scorecard
    selected = pool.sort_values(
        ["passes_contract", "selection_objective", "f1", "precision", "top_k"],
        ascending=[False, False, False, False, True],
    ).iloc[0]
    parameters = RankingParameters(
        score_name=score_name,
        top_k=int(selected["top_k"]),
        bucket_hours=bucket_hours,
    )
    scorecard["selected_capacity"] = scorecard["candidate_id"].eq(parameters.policy_id)
    scorecard["selection_stage"] = "VALID_CALIBRATE_CAPACITY_CALIBRATION"
    return scorecard, parameters


def replay_roles(
    prepared: pd.DataFrame, parameters: RankingParameters
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decisions = apply_capacity_ranking(prepared, parameters)
    rows = []
    for role, role_frame in decisions.groupby("evaluation_role", sort=False):
        metrics = evaluate_ranking(role_frame, parameters)
        metrics["role"] = str(role)
        metrics.update(contract_gates(metrics))
        metrics["passes_contract"] = all(
            bool(value) for key, value in metrics.items() if key.startswith("gate_")
        )
        rows.append(metrics)
    return decisions, pd.DataFrame(rows)


def cluster_bootstrap_metrics(
    decisions: pd.DataFrame,
    *,
    iterations: int = 500,
    random_state: int = 20260810,
) -> dict[str, Any]:
    calls = decisions["port_call_id"].astype(str).drop_duplicates().to_numpy()
    if len(calls) < 30:
        raise ValueError("Cluster bootstrap requires at least 30 port calls")
    frame = decisions.copy()
    frame["_call"] = frame["port_call_id"].astype(str)
    rng = np.random.default_rng(random_state)
    samples = {name: [] for name in (
        "precision", "recall", "f1", "precision_lift", "event_recall_any",
        "review_rate_pct",
    )}
    for _ in range(iterations):
        sampled = rng.choice(calls, size=len(calls), replace=True)
        multiplicity = pd.Series(sampled).value_counts()
        bootstrap = frame.loc[frame["_call"].isin(multiplicity.index)].copy()
        weight = bootstrap["_call"].map(multiplicity).to_numpy(float)
        actual = bootstrap["target_event_within_24h"].astype(bool).to_numpy()
        selected = bootstrap["watchlist_selected"].astype(bool).to_numpy()
        tp = float(weight[actual & selected].sum())
        fp = float(weight[~actual & selected].sum())
        fn = float(weight[actual & ~selected].sum())
        total = float(weight.sum())
        precision = tp / max(tp + fp, 1e-12)
        recall = tp / max(tp + fn, 1e-12)
        prevalence = float(weight[actual].sum()) / max(total, 1e-12)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        bootstrap["_hit"] = (
            bootstrap["target_event_within_24h"].astype(bool)
            & bootstrap["watchlist_selected"].astype(bool)
        )
        call_summary = bootstrap.groupby("_call").agg(
            positive=("target_event_within_24h", "max"),
            detected=("_hit", "max"),
        )
        call_weight = call_summary.index.to_series().map(multiplicity).to_numpy(float)
        positive = call_summary["positive"].astype(bool).to_numpy()
        detected = call_summary["detected"].astype(bool).to_numpy()
        event_recall = float(call_weight[detected].sum() / max(call_weight[positive].sum(), 1e-12))
        samples["precision"].append(precision)
        samples["recall"].append(recall)
        samples["f1"].append(f1)
        samples["precision_lift"].append(precision / max(prevalence, 1e-12))
        samples["event_recall_any"].append(event_recall)
        samples["review_rate_pct"].append(100.0 * float(weight[selected].sum()) / max(total, 1e-12))
    result: dict[str, Any] = {
        "method": "PORT_CALL_CLUSTER_BOOTSTRAP",
        "iterations": iterations,
        "seed": random_state,
        "clusters": int(len(calls)),
    }
    for name, values in samples.items():
        result[name] = {
            "p2_5": float(np.quantile(values, 0.025)),
            "median": float(np.quantile(values, 0.5)),
            "p97_5": float(np.quantile(values, 0.975)),
        }
    return result


def verify_b61e_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "policy_version", "decision", "selected_score", "selected_top_k",
        "selection_role", "calibration_role", "test_role",
        "test_used_for_selection", "contracts_passed", "serving_rows",
        "production_promotion_allowed", "automatic_action_allowed", "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(f"B61E result contract misses: {missing}")
    if result["policy_version"] != POLICY_VERSION:
        raise RuntimeError("B61E policy version mismatch")
    if result["selection_role"] != "VALID_SELECT":
        raise RuntimeError("B61E score selection must use VALID_SELECT")
    if result["calibration_role"] != "VALID_CALIBRATE":
        raise RuntimeError("B61E capacity calibration must use VALID_CALIBRATE")
    if result["test_used_for_selection"]:
        raise RuntimeError("B61E TEST leakage contract failed")
    if result["production_promotion_allowed"] or result["automatic_action_allowed"]:
        raise RuntimeError("B61E remains shadow-only")
    if int(result["serving_rows"]) <= 0:
        raise RuntimeError("B61E did not materialize watchlist rows")
    return result


__all__ = [
    "POLICY_VERSION", "SOURCE_MODEL_VERSION", "SOURCE_HSMM_VERSION",
    "VALID_ROLES", "TEST_ROLE", "SCORE_NAMES", "RankingParameters",
    "RankingContract", "parse_score_names", "parse_top_ks",
    "prepare_ranking_frame", "ranking_score", "apply_capacity_ranking",
    "evaluate_ranking", "contract_gates", "select_score_on_valid_select",
    "calibrate_capacity_on_valid_calibrate", "replay_roles",
    "cluster_bootstrap_metrics", "verify_b61e_result",
]
