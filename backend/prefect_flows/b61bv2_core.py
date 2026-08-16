from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from prefect_flows.b61b_core import (
    HAZARD_HORIZONS,
    HAZARD_TARGETS,
    QUANTILES,
    RISK_NAMES,
    RISK_TARGETS,
    apply_conformal,
    clean_json,
    enforce_hazard_order,
    enforce_quantile_order,
    enforce_risk_order,
    pinball_loss,
)


RANDOM_SEED = 20260808
MODEL_VERSION = "b61b-v2-maritime-rare-event-hybrid-v2"
REAL_DATASET_VERSION = "b61a-governed-port-call-enrichment-v1"
SYNTHETIC_DATASET_VERSION = "b61ax-governed-rare-tail-augmentation-v1"
TRACKS = (
    "REAL_REFERENCE",
    "REAL_COST_SENSITIVE",
    "EVT_LOW_WEIGHT",
    "SEQUENCE_REAL_ONLY",
)


def _weights(sample_weight: np.ndarray | pd.Series | None, size: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(size, dtype="float64")
    values = np.asarray(sample_weight, dtype="float64")
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    if values.sum() <= 0.0:
        return np.ones(size, dtype="float64")
    return values


def weighted_binary_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    threshold: float = 0.5,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> dict[str, float | int | None]:
    actual = np.asarray(actual, dtype="int64")
    probability = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1.0 - 1e-6)
    weight = _weights(sample_weight, len(actual))
    predicted = probability >= threshold
    positive = actual == 1
    tp = float(weight[predicted & positive].sum())
    fp = float(weight[predicted & ~positive].sum())
    fn = float(weight[~predicted & positive].sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    has_two_classes = len(np.unique(actual)) == 2
    return {
        "rows": int(len(actual)),
        "effective_calls": float(weight.sum()),
        "positives": int(positive.sum()),
        "positive_rate": float(np.average(actual, weights=weight)),
        "roc_auc": float(roc_auc_score(actual, probability, sample_weight=weight)) if has_two_classes else None,
        "average_precision": float(average_precision_score(actual, probability, sample_weight=weight)) if has_two_classes else None,
        "brier": float(brier_score_loss(actual, probability, sample_weight=weight)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1], sample_weight=weight)),
        "ece_10": weighted_ece(actual, probability, weight),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def weighted_ece(actual: np.ndarray, probability: np.ndarray, weight: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = max(float(weight.sum()), 1e-12)
    score = 0.0
    for index in range(bins):
        upper = probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        mask = (probability >= edges[index]) & upper & (weight > 0.0)
        if mask.any():
            mass = float(weight[mask].sum())
            observed = float(np.average(actual[mask], weights=weight[mask]))
            forecast = float(np.average(probability[mask], weights=weight[mask]))
            score += mass / total * abs(observed - forecast)
    return float(score)


def weighted_regression_metrics(
    actual: np.ndarray,
    quantiles: np.ndarray,
    sample_weight: np.ndarray | pd.Series | None = None,
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype="float64")
    predicted = enforce_quantile_order(quantiles)
    weight = _weights(sample_weight, len(actual))
    residual = predicted[:, 1] - actual
    losses = []
    for index, quantile in enumerate(QUANTILES):
        error = actual - predicted[:, index]
        loss = np.maximum(quantile * error, (quantile - 1.0) * error)
        losses.append(float(np.average(loss, weights=weight)))
    covered = (actual >= predicted[:, 0]) & (actual <= predicted[:, 2])
    return {
        "rows": int(len(actual)),
        "effective_calls": float(weight.sum()),
        "mae_p50": float(np.average(np.abs(residual), weights=weight)),
        "rmse_p50": float(np.sqrt(np.average(np.square(residual), weights=weight))),
        "bias_p50": float(np.average(residual, weights=weight)),
        "pinball_p10": losses[0],
        "pinball_p50": losses[1],
        "pinball_p90": losses[2],
        "mean_pinball": float(np.mean(losses)),
        "coverage_p10_p90": float(np.average(covered, weights=weight)),
        "mean_interval_width": float(np.average(predicted[:, 2] - predicted[:, 0], weights=weight)),
    }


def binary_selection_objective(metrics: dict[str, Any]) -> float:
    ap = float(metrics.get("average_precision") or 0.0)
    prevalence = max(float(metrics.get("positive_rate") or 0.0), 1e-4)
    relative_ap = min(ap / prevalence, 10.0)
    return float(metrics["brier"] + 0.20 * metrics["log_loss"] - 0.015 * relative_ap)


def duration_selection_objective(metrics: dict[str, Any]) -> float:
    undercoverage = max(0.0, 0.78 - float(metrics["coverage_p10_p90"]))
    return float(metrics["mean_pinball"] + 0.25 * undercoverage * metrics["mae_p50"])


def select_threshold(
    actual: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
    false_negative_cost: float,
) -> float:
    best = (float("inf"), 0.5)
    for threshold in np.linspace(0.02, 0.90, 89):
        predicted = probability >= threshold
        positive = actual == 1
        fp = float(sample_weight[predicted & ~positive].sum())
        fn = float(sample_weight[~predicted & positive].sum())
        cost = (fp + false_negative_cost * fn) / max(float(sample_weight.sum()), 1e-12)
        if cost < best[0]:
            best = (cost, float(threshold))
    return best[1]


@dataclass
class BinaryCalibrator:
    method: str
    model: Any = None

    def predict(self, probability: np.ndarray) -> np.ndarray:
        values = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1.0 - 1e-6)
        if self.method == "ISOTONIC":
            return np.clip(self.model.predict(values), 0.0, 1.0)
        if self.method == "PLATT":
            logits = np.log(values / (1.0 - values)).reshape(-1, 1)
            return np.clip(self.model.predict_proba(logits)[:, 1], 0.0, 1.0)
        return values


def fit_binary_calibrator(
    actual: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
) -> BinaryCalibrator:
    actual = np.asarray(actual, dtype="int64")
    probability = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1.0 - 1e-6)
    positive = int(actual.sum())
    negative = int(len(actual) - positive)
    # Isotonic calibration creates wide probability plateaus. Reserve it for
    # well-supported tasks; Platt scaling preserves ranking for rare risks.
    if len(actual) >= 1_000 and positive >= 200 and negative >= 200:
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(probability, actual, sample_weight=sample_weight)
        return BinaryCalibrator("ISOTONIC", model)
    if len(actual) >= 50 and positive >= 5 and negative >= 5:
        logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
        model = LogisticRegression(C=1.0, solver="lbfgs", random_state=RANDOM_SEED)
        model.fit(logits, actual, sample_weight=sample_weight)
        return BinaryCalibrator("PLATT", model)
    return BinaryCalibrator("IDENTITY_INSUFFICIENT_SUPPORT")


def adaptive_conformal_policy(
    frame: pd.DataFrame,
    actual: np.ndarray,
    quantiles: np.ndarray,
    calibration_mask: np.ndarray,
    minimum_regime_rows: int = 150,
) -> dict[str, Any]:
    weights = _weights(
        frame.get("per_call_sample_weight", pd.Series(1.0, index=frame.index)).to_numpy()[calibration_mask],
        int(calibration_mask.sum()),
    )
    lower, upper = weighted_conformal_corrections(
        actual[calibration_mask],
        quantiles[calibration_mask, 0],
        quantiles[calibration_mask, 2],
        weights,
    )
    policy: dict[str, Any] = {
        "__GLOBAL__": {"lower_h": lower, "upper_h": upper, "rows": int(calibration_mask.sum())}
    }
    regimes = frame["regime"].astype(str).to_numpy()
    for regime in sorted(set(regimes[calibration_mask])):
        mask = calibration_mask & (regimes == regime)
        if mask.sum() < minimum_regime_rows:
            continue
        local_lower, local_upper = weighted_conformal_corrections(
            actual[mask],
            quantiles[mask, 0],
            quantiles[mask, 2],
            _weights(
                frame.get("per_call_sample_weight", pd.Series(1.0, index=frame.index)).to_numpy()[mask],
                int(mask.sum()),
            ),
        )
        shrinkage = mask.sum() / (mask.sum() + minimum_regime_rows)
        policy[regime] = {
            "lower_h": float(shrinkage * local_lower + (1.0 - shrinkage) * lower),
            "upper_h": float(shrinkage * local_upper + (1.0 - shrinkage) * upper),
            "rows": int(mask.sum()),
        }
    return policy


def weighted_quantile(values: np.ndarray, quantile: float, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype="float64")
    weights = _weights(weights, len(values))
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= max(float(weights.sum()), 1e-12)
    return float(np.interp(quantile, cumulative, values))


def weighted_conformal_corrections(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: np.ndarray,
    alpha: float = 0.2,
) -> tuple[float, float]:
    if len(actual) < 20:
        return 0.0, 0.0
    lower_error = np.maximum(np.asarray(lower, dtype="float64") - actual, 0.0)
    upper_error = np.maximum(actual - np.asarray(upper, dtype="float64"), 0.0)
    tail_level = 1.0 - alpha / 2.0
    return (
        weighted_quantile(lower_error, tail_level, weights),
        weighted_quantile(upper_error, tail_level, weights),
    )


def apply_adaptive_conformal(
    frame: pd.DataFrame, quantiles: np.ndarray, policy: dict[str, Any]
) -> np.ndarray:
    result = enforce_quantile_order(quantiles).copy()
    regimes = frame["regime"].astype(str).to_numpy()
    global_policy = policy["__GLOBAL__"]
    for regime in set(regimes):
        correction = policy.get(regime, global_policy)
        mask = regimes == regime
        result[mask] = apply_conformal(
            result[mask], float(correction["lower_h"]), float(correction["upper_h"])
        )
    return enforce_quantile_order(result)


def approximate_concordance_index(
    duration: np.ndarray,
    event_observed: np.ndarray,
    risk: np.ndarray,
    maximum_pairs: int = 100_000,
) -> float | None:
    duration = np.asarray(duration, dtype="float64")
    event_observed = np.asarray(event_observed, dtype=bool)
    risk = np.asarray(risk, dtype="float64")
    valid = np.isfinite(duration) & np.isfinite(risk)
    positions = np.flatnonzero(valid)
    if len(positions) < 2 or not event_observed[positions].any():
        return None
    rng = np.random.default_rng(RANDOM_SEED)
    first = rng.choice(positions, size=maximum_pairs, replace=True)
    second = rng.choice(positions, size=maximum_pairs, replace=True)
    comparable = (duration[first] < duration[second]) & event_observed[first]
    if not comparable.any():
        return None
    score = np.where(
        risk[first][comparable] > risk[second][comparable],
        1.0,
        np.where(risk[first][comparable] == risk[second][comparable], 0.5, 0.0),
    )
    return float(score.mean())


def grouped_bootstrap_ci(
    frame: pd.DataFrame,
    metric: Callable[[np.ndarray], float],
    replicates: int = 300,
) -> tuple[float, float, float, int]:
    groups = frame.groupby("port_call_id", sort=False).indices
    identifiers = np.asarray(list(groups), dtype=object)
    if len(identifiers) < 20:
        value = float(metric(frame.index.to_numpy()))
        return value, value, value, 0
    positions = {key: np.asarray(groups[key], dtype="int64") for key in identifiers}
    rng = np.random.default_rng(RANDOM_SEED)
    estimates = []
    for _ in range(replicates):
        sampled = rng.choice(identifiers, size=len(identifiers), replace=True)
        index = np.concatenate([positions[key] for key in sampled])
        try:
            value = float(metric(index))
        except ValueError:
            continue
        if math.isfinite(value):
            estimates.append(value)
    point = float(metric(frame.index.to_numpy()))
    if len(estimates) < 20:
        return point, point, point, len(estimates)
    return (
        point,
        float(np.quantile(estimates, 0.025)),
        float(np.quantile(estimates, 0.975)),
        len(estimates),
    )


def coherent_outputs(
    risk: np.ndarray, quantiles: np.ndarray, hazard: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "risk": enforce_risk_order(risk),
        "quantiles": enforce_quantile_order(quantiles),
        "hazard": enforce_hazard_order(hazard),
    }


def json_ready(value: Any) -> Any:
    return clean_json(value)
