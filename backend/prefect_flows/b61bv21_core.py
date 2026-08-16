from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from prefect_flows.b61bv2_core import BinaryCalibrator, weighted_binary_metrics


RANDOM_SEED = 20260809
MODEL_VERSION = "b61b-v2.1-maritime-recalibration-only-v1"
SOURCE_MODEL_VERSION = "b61b-v2-maritime-rare-event-hybrid-v2"
RARE_RISK_TASKS = ("gt3", "gt6")


@dataclass(frozen=True)
class RecalibrationEvidence:
    task: str
    method: str
    rows: int
    positives: int
    negatives: int
    rank_preserving: bool
    roc_auc_delta: float | None
    average_precision_delta: float | None
    brier_delta: float | None
    ece_delta: float | None


def fit_rank_preserving_platt(
    actual: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
    minimum_rows: int = 50,
    minimum_class_rows: int = 5,
) -> BinaryCalibrator:
    """Fit a monotone Platt map without changing the underlying score order."""
    actual = np.asarray(actual, dtype="int64")
    probability = np.clip(np.asarray(probability, dtype="float64"), 1e-6, 1.0 - 1e-6)
    sample_weight = np.asarray(sample_weight, dtype="float64")
    positives = int(actual.sum())
    negatives = int(len(actual) - positives)
    if (
        len(actual) < minimum_rows
        or positives < minimum_class_rows
        or negatives < minimum_class_rows
    ):
        return BinaryCalibrator("IDENTITY_INSUFFICIENT_SUPPORT")
    logits = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_SEED,
        max_iter=1_000,
    )
    model.fit(logits, actual, sample_weight=sample_weight)
    if float(model.coef_[0, 0]) <= 0.0:
        return BinaryCalibrator("IDENTITY_NON_POSITIVE_PLATT_SLOPE")
    return BinaryCalibrator("PLATT", model)


def build_recalibration_evidence(
    task: str,
    method: str,
    actual: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
    sample_weight: np.ndarray,
) -> RecalibrationEvidence:
    raw = weighted_binary_metrics(actual, raw_probability, sample_weight=sample_weight)
    calibrated = weighted_binary_metrics(
        actual, calibrated_probability, sample_weight=sample_weight
    )
    order = np.argsort(np.asarray(raw_probability), kind="mergesort")
    ordered = np.asarray(calibrated_probability)[order]
    rank_preserving = bool(np.all(np.diff(ordered) >= -1e-12))

    def delta(name: str) -> float | None:
        before = raw.get(name)
        after = calibrated.get(name)
        if before is None or after is None:
            return None
        return float(after) - float(before)

    actual = np.asarray(actual, dtype="int64")
    return RecalibrationEvidence(
        task=task,
        method=(
            "PLATT_RANK_PRESERVING" if method == "PLATT" else method
        ),
        rows=int(len(actual)),
        positives=int(actual.sum()),
        negatives=int(len(actual) - actual.sum()),
        rank_preserving=rank_preserving,
        roc_auc_delta=delta("roc_auc"),
        average_precision_delta=delta("average_precision"),
        brier_delta=delta("brier"),
        ece_delta=delta("ece_10"),
    )


def evidence_as_dict(evidence: RecalibrationEvidence) -> dict[str, Any]:
    return {
        "task": evidence.task,
        "method": evidence.method,
        "rows": evidence.rows,
        "positives": evidence.positives,
        "negatives": evidence.negatives,
        "rank_preserving": evidence.rank_preserving,
        "roc_auc_delta": evidence.roc_auc_delta,
        "average_precision_delta": evidence.average_precision_delta,
        "brier_delta": evidence.brier_delta,
        "ece_delta": evidence.ece_delta,
    }


def calibration_gate_rows(evidence: list[RecalibrationEvidence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evidence:
        rows.extend(
            [
                {
                    "check": f"{item.task.upper()}_USES_RANK_PRESERVING_PLATT",
                    "passed": item.method == "PLATT_RANK_PRESERVING",
                    "severity": "CALIBRATION",
                    "value": item.method,
                },
                {
                    "check": f"{item.task.upper()}_RANK_FIDELITY",
                    "passed": item.rank_preserving
                    and abs(item.roc_auc_delta or 0.0) <= 1e-10
                    and abs(item.average_precision_delta or 0.0) <= 1e-10,
                    "severity": "CALIBRATION",
                    "value": item.average_precision_delta,
                },
                {
                    "check": f"{item.task.upper()}_CALIBRATION_SUPPORT",
                    "passed": item.rows >= 50
                    and item.positives >= 5
                    and item.negatives >= 5,
                    "severity": "CALIBRATION",
                    "value": item.positives,
                },
                {
                    "check": f"{item.task.upper()}_BRIER_NON_INFERIOR",
                    "passed": item.brier_delta is not None
                    and item.brier_delta <= 0.001,
                    "severity": "CALIBRATION",
                    "value": item.brier_delta,
                },
                {
                    "check": f"{item.task.upper()}_ECE_NON_INFERIOR",
                    "passed": item.ece_delta is not None and item.ece_delta <= 0.005,
                    "severity": "CALIBRATION",
                    "value": item.ece_delta,
                },
            ]
        )
    return rows
