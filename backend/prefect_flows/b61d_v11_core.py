from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


MODEL_VERSION = "b61d-v1.1-anchored-hsmm-v1"
SOURCE_MODEL_VERSION = "b61b-v2.1-maritime-recalibration-only-v1"
SOURCE_POLICY_VERSION = "b61c-dynamic-alert-policy-v1.1"
SOURCE_HSMM_VERSION = "b61d-contextual-hsmm-v1"

STATE_NAMES = (
    "FLUID",
    "PRESSURE_BUILDING",
    "CONGESTED",
    "CRITICAL_DISRUPTION",
    "RECOVERY",
)
STATE_RISK = np.array([0.05, 0.35, 0.70, 1.00, 0.25], dtype=float)

OBSERVATION_COLUMNS = (
    "p_delay_gt3",
    "p_delay_gt6",
    "p_gt3_breach_within_6h",
    "p_gt3_breach_within_12h",
    "p_gt3_breach_within_24h",
    "remaining_p50_h",
    "remaining_interval_width_h",
    "overdue_h",
    "plan_progress_ratio",
    "time_to_planned_departure_h",
    "arrival_delay_h",
    "vessel_history_prior_late_gt3_rate",
    "known_event_any_24h",
    "delta_p_delay_gt3",
    "delta_hazard_12h",
    "delta_overdue_h",
)

CONTEXT_COLUMNS = (
    "p_gt3_breach_within_12h",
    "p_gt3_breach_within_24h",
    "overdue_h",
    "plan_progress_ratio",
    "known_event_any_24h",
    "delta_p_delay_gt3",
    "delta_overdue_h",
)

TOPOLOGY_WEIGHT = np.array(
    [
        [1.00, 0.80, 0.25, 0.03, 0.10],
        [0.40, 1.00, 0.85, 0.35, 0.15],
        [0.10, 0.40, 1.00, 0.85, 0.55],
        [0.02, 0.10, 0.50, 1.00, 0.90],
        [0.95, 0.55, 0.15, 0.05, 1.00],
    ],
    dtype=float,
)


@dataclass
class AnchoredContextualHSMM:
    imputer: SimpleImputer
    scaler: RobustScaler
    emission_mean: np.ndarray
    emission_variance: np.ndarray
    emission_prior: np.ndarray
    emission_temperature: float
    transition_model: Pipeline
    transition_classes: np.ndarray
    duration_pmf: np.ndarray
    start_probability: np.ndarray
    topology_weight: np.ndarray
    observation_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    max_duration_bins: int
    duration_bin_hours: int
    fit_rows: int
    fit_calls: int
    transition_rows: int
    fit_cutoff: pd.Timestamp
    anchor_counts: dict[str, int]
    minimum_anchor_rows: int


def prepare_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "port_call_id", "landmark_at", "evaluation_role", "p_delay_gt3",
        "p_delay_gt6", "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h", "p_gt3_breach_within_24h",
        "remaining_p10_h", "remaining_p50_h", "remaining_p90_h",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"B61D-v1.1 source misses columns: {missing}")
    output = frame.copy()
    output["landmark_at"] = pd.to_datetime(output["landmark_at"], utc=True)
    output = output.sort_values(
        ["evaluation_role", "port_call_id", "landmark_at"]
    ).reset_index(drop=True)
    output["remaining_interval_width_h"] = (
        pd.to_numeric(output["remaining_p90_h"], errors="coerce")
        - pd.to_numeric(output["remaining_p10_h"], errors="coerce")
    ).clip(lower=0.0)
    grouped = output.groupby(["evaluation_role", "port_call_id"], sort=False)
    output["delta_p_delay_gt3"] = grouped["p_delay_gt3"].diff().fillna(0.0)
    output["delta_hazard_12h"] = grouped[
        "p_gt3_breach_within_12h"
    ].diff().fillna(0.0)
    output["delta_overdue_h"] = grouped["overdue_h"].diff().fillna(0.0)
    for column in OBSERVATION_COLUMNS:
        if column not in output:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def prebreach_fit_mask(
    frame: pd.DataFrame,
    fraction: float = 0.70,
) -> tuple[pd.Series, pd.Timestamp]:
    eligible = (
        frame["evaluation_role"].eq("VALID_SELECT")
        & frame["early_warning_eligible"].fillna(False).astype(bool)
        & frame["pre_breach_eligible"].fillna(False).astype(bool)
    )
    rows = frame.loc[eligible]
    if len(rows) < 1_000:
        raise ValueError("Insufficient pre-breach VALID_SELECT rows for anchored HSMM")
    cutoff = pd.Timestamp(rows["landmark_at"].quantile(fraction))
    mask = eligible & frame["landmark_at"].le(cutoff)
    return mask, cutoff


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _rank01(value: pd.Series) -> np.ndarray:
    return value.rank(method="average", pct=True).to_numpy(dtype=float)


def anchor_score_matrix(frame: pd.DataFrame) -> np.ndarray:
    p3 = _numeric(frame, "p_delay_gt3").clip(0.0, 1.0).to_numpy()
    p6 = _numeric(frame, "p_delay_gt6").clip(0.0, 1.0).to_numpy()
    h6 = _numeric(frame, "p_gt3_breach_within_6h").clip(0.0, 1.0).to_numpy()
    h12 = _numeric(frame, "p_gt3_breach_within_12h").clip(0.0, 1.0).to_numpy()
    h24 = _numeric(frame, "p_gt3_breach_within_24h").clip(0.0, 1.0).to_numpy()
    overdue = _rank01(_numeric(frame, "overdue_h"))
    progress = _rank01(_numeric(frame, "plan_progress_ratio"))
    increase = _rank01(
        _numeric(frame, "delta_p_delay_gt3")
        + 0.5 * _numeric(frame, "delta_hazard_12h")
        + 0.2 * _numeric(frame, "delta_overdue_h")
    )
    decrease = _rank01(
        -_numeric(frame, "delta_p_delay_gt3")
        - 0.5 * _numeric(frame, "delta_hazard_12h")
        - 0.2 * _numeric(frame, "delta_overdue_h")
    )
    previous_p3 = (
        frame.groupby("port_call_id", sort=False)["p_delay_gt3"]
        .shift()
        .fillna(frame["p_delay_gt3"])
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .clip(0.0, 1.0)
        .to_numpy()
    )
    scores = np.column_stack(
        [
            1.25 * (1.0 - p3) + 0.90 * (1.0 - h24) + 0.45 * (1.0 - overdue),
            1.20 * increase + 0.80 * h24 + 0.45 * np.maximum(h24 - h6, 0.0)
            + 0.20 * p3 - 0.35 * p6,
            1.10 * p3 + 0.75 * h12 + 0.55 * overdue + 0.30 * progress
            - 0.20 * decrease,
            1.25 * p6 + 1.05 * h6 + 0.60 * overdue + 0.35 * progress,
            1.25 * decrease + 0.65 * previous_p3 + 0.35 * (previous_p3 - p3).clip(0.0)
            + 0.20 * overdue,
        ]
    )
    median = np.median(scores, axis=0)
    q75 = np.quantile(scores, 0.75, axis=0)
    q25 = np.quantile(scores, 0.25, axis=0)
    return (scores - median) / np.maximum(q75 - q25, 1e-6)


def balanced_anchor_labels(
    scores: np.ndarray,
    minimum_rows: int | None = None,
) -> tuple[np.ndarray, int]:
    rows, states = scores.shape
    if states != len(STATE_NAMES) or rows < 500:
        raise ValueError("Anchored state assignment has insufficient support")
    if minimum_rows is None:
        minimum_rows = min(max(50, int(math.ceil(rows * 0.02))), rows // 10)
    labels = scores.argmax(axis=1).astype(int)
    claimed = np.zeros(rows, dtype=bool)
    for state in (3, 4, 1, 2, 0):
        order = np.argsort(scores[:, state])[::-1]
        selected = order[~claimed[order]][:minimum_rows]
        labels[selected] = state
        claimed[selected] = True
    counts = np.bincount(labels, minlength=states)
    if int(counts.min()) < minimum_rows:
        raise ValueError(f"Unable to anchor all states: counts={counts.tolist()}")
    return labels, minimum_rows


def _transition_rows(
    fit_frame: pd.DataFrame,
    labels: np.ndarray,
    duration_bin_hours: int,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    indexed = fit_frame.copy()
    indexed["_anchor_state"] = labels
    features: list[np.ndarray] = []
    targets: list[int] = []
    run_states: list[int] = []
    run_durations: list[int] = []
    eye = np.eye(len(STATE_NAMES), dtype=float)
    for _, sequence in indexed.groupby("port_call_id", sort=False):
        sequence = sequence.sort_values("landmark_at")
        states = sequence["_anchor_state"].to_numpy(dtype=int)
        if len(states) == 0:
            continue
        start = 0
        for position in range(1, len(states) + 1):
            if position == len(states) or states[position] != states[start]:
                elapsed_h = (
                    sequence["landmark_at"].iloc[position - 1]
                    - sequence["landmark_at"].iloc[start]
                ).total_seconds() / 3600.0
                run_states.append(int(states[start]))
                run_durations.append(
                    max(1, int(math.ceil(max(elapsed_h, 0.0) / duration_bin_hours)) + 1)
                )
                start = position
        for position in range(1, len(sequence)):
            gap_h = (
                sequence["landmark_at"].iloc[position]
                - sequence["landmark_at"].iloc[position - 1]
            ).total_seconds() / 3600.0
            if gap_h <= 0.0 or gap_h > 72.0:
                continue
            previous = int(states[position - 1])
            context = sequence.iloc[position][list(CONTEXT_COLUMNS)].to_numpy(dtype=float)
            context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(np.concatenate([eye[previous], context]))
            targets.append(int(states[position]))
    if len(features) < 500:
        raise ValueError("Insufficient contextual transitions for anchored HSMM")
    return np.vstack(features), np.asarray(targets), run_states, run_durations


def _duration_pmf(
    states: list[int], durations: list[int], maximum: int
) -> np.ndarray:
    counts = np.ones((len(STATE_NAMES), maximum), dtype=float)
    for state, duration in zip(states, durations):
        counts[state, min(maximum, duration) - 1] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def fit_anchored_hsmm(
    frame: pd.DataFrame,
    fit_mask: pd.Series,
    fit_cutoff: pd.Timestamp,
    random_state: int = 20260809,
    max_duration_bins: int = 12,
    duration_bin_hours: int = 6,
    emission_temperature: float = 1.75,
) -> AnchoredContextualHSMM:
    fit_frame = frame.loc[fit_mask].copy()
    fit_frame = fit_frame.loc[
        fit_frame.groupby("port_call_id")["port_call_id"].transform("size").ge(2)
    ].copy()
    if len(fit_frame) < 1_000 or fit_frame["port_call_id"].nunique() < 300:
        raise ValueError("Anchored HSMM needs 1,000 rows and 300 multi-landmark calls")
    scores = anchor_score_matrix(fit_frame)
    labels, minimum_rows = balanced_anchor_labels(scores)
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    transformed = scaler.fit_transform(
        imputer.fit_transform(fit_frame[list(OBSERVATION_COLUMNS)].to_numpy(dtype=float))
    )
    means = np.vstack([transformed[labels == state].mean(axis=0) for state in range(5)])
    variances = np.vstack([transformed[labels == state].var(axis=0) for state in range(5)])
    variances = np.maximum(variances, 0.05)
    empirical_prior = np.bincount(labels, minlength=5).astype(float) / len(labels)
    priors = 0.50 * empirical_prior + 0.50 / len(STATE_NAMES)
    transition_x, transition_y, run_states, run_durations = _transition_rows(
        fit_frame, labels, duration_bin_hours
    )
    if len(np.unique(transition_y)) < len(STATE_NAMES):
        raise ValueError(
            f"All anchored transition destinations are required: {np.unique(transition_y)}"
        )
    transition = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1_000,
                    class_weight="balanced",
                    C=0.60,
                    random_state=random_state,
                ),
            ),
        ]
    )
    transition.fit(transition_x, transition_y)
    start_counts = np.ones(len(STATE_NAMES), dtype=float)
    label_lookup = pd.Series(labels, index=fit_frame.index)
    for _, sequence in fit_frame.groupby("port_call_id", sort=False):
        start_counts[int(label_lookup.loc[sequence["landmark_at"].idxmin()])] += 1.0
    counts = np.bincount(labels, minlength=5)
    return AnchoredContextualHSMM(
        imputer=imputer,
        scaler=scaler,
        emission_mean=means,
        emission_variance=variances,
        emission_prior=priors,
        emission_temperature=emission_temperature,
        transition_model=transition,
        transition_classes=np.asarray(transition.named_steps["model"].classes_, dtype=int),
        duration_pmf=_duration_pmf(run_states, run_durations, max_duration_bins),
        start_probability=start_counts / start_counts.sum(),
        topology_weight=TOPOLOGY_WEIGHT.copy(),
        observation_columns=OBSERVATION_COLUMNS,
        context_columns=CONTEXT_COLUMNS,
        max_duration_bins=max_duration_bins,
        duration_bin_hours=duration_bin_hours,
        fit_rows=len(fit_frame),
        fit_calls=int(fit_frame["port_call_id"].nunique()),
        transition_rows=len(transition_y),
        fit_cutoff=fit_cutoff,
        anchor_counts={STATE_NAMES[i]: int(counts[i]) for i in range(5)},
        minimum_anchor_rows=minimum_rows,
    )


def _softmax(log_probability: np.ndarray) -> np.ndarray:
    shifted = log_probability - log_probability.max(axis=1, keepdims=True)
    probability = np.exp(shifted)
    return probability / probability.sum(axis=1, keepdims=True)


def emission_probability(
    bundle: AnchoredContextualHSMM, sequence: pd.DataFrame
) -> np.ndarray:
    transformed = bundle.scaler.transform(
        bundle.imputer.transform(
            sequence[list(bundle.observation_columns)].to_numpy(dtype=float)
        )
    )
    difference = transformed[:, None, :] - bundle.emission_mean[None, :, :]
    log_probability = -0.5 * (
        np.log(2.0 * np.pi * bundle.emission_variance)[None, :, :]
        + difference**2 / bundle.emission_variance[None, :, :]
    ).sum(axis=2)
    log_probability += np.log(np.clip(bundle.emission_prior, 1e-12, 1.0))[None, :]
    return np.clip(_softmax(log_probability / bundle.emission_temperature), 1e-12, 1.0)


def _transition_tensor(
    bundle: AnchoredContextualHSMM, sequence: pd.DataFrame
) -> np.ndarray:
    states = len(STATE_NAMES)
    result = np.full((len(sequence), states, states), 0.01, dtype=float)
    eye = np.eye(states, dtype=float)
    for position in range(len(sequence)):
        context = sequence.iloc[position][list(bundle.context_columns)].to_numpy(dtype=float)
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        inputs = np.vstack(
            [np.concatenate([eye[previous], context]) for previous in range(states)]
        )
        predicted = bundle.transition_model.predict_proba(inputs)
        for class_position, state in enumerate(bundle.transition_classes):
            result[position, :, int(state)] = predicted[:, class_position]
        result[position] *= bundle.topology_weight
        result[position] += 0.08 * eye
        result[position] /= result[position].sum(axis=1, keepdims=True)
    return np.clip(result, 1e-12, 1.0)


def _duration_bin(
    timestamps: pd.Series,
    start: int,
    end: int,
    bin_hours: int,
    maximum: int,
) -> int:
    elapsed_h = (timestamps.iloc[end] - timestamps.iloc[start]).total_seconds() / 3600.0
    return min(maximum, max(1, int(math.ceil(max(elapsed_h, 0.0) / bin_hours)) + 1))


def decode_sequence(
    bundle: AnchoredContextualHSMM, sequence: pd.DataFrame
) -> pd.DataFrame:
    sequence = sequence.sort_values("landmark_at").copy()
    emission = emission_probability(bundle, sequence)
    transition = _transition_tensor(bundle, sequence)
    log_emission = np.log(emission)
    cumulative = np.vstack([np.zeros((1, 5)), np.cumsum(log_emission, axis=0)])
    length = len(sequence)
    score = np.full((length, 5), -np.inf)
    back_state = np.full((length, 5), -1, dtype=int)
    back_start = np.full((length, 5), -1, dtype=int)
    timestamps = sequence["landmark_at"].reset_index(drop=True)
    for end in range(length):
        for state in range(5):
            for start in range(end + 1):
                duration = _duration_bin(
                    timestamps, start, end, bundle.duration_bin_hours,
                    bundle.max_duration_bins,
                )
                segment = cumulative[end + 1, state] - cumulative[start, state]
                duration_score = math.log(bundle.duration_pmf[state, duration - 1])
                if start == 0:
                    candidate = math.log(bundle.start_probability[state]) + segment + duration_score
                    previous = -1
                else:
                    previous_scores = score[start - 1] + np.log(transition[start, :, state])
                    previous = int(np.argmax(previous_scores))
                    candidate = previous_scores[previous] + segment + duration_score
                if candidate > score[end, state]:
                    score[end, state] = candidate
                    back_state[end, state] = previous
                    back_start[end, state] = start
    path = np.zeros(length, dtype=int)
    end = length - 1
    state = int(np.argmax(score[end]))
    while end >= 0:
        start = int(back_start[end, state])
        path[start : end + 1] = state
        state = int(back_state[end, state])
        end = start - 1
    expected_risk = emission @ STATE_RISK
    decoded_risk = STATE_RISK[path]
    escalation = np.zeros(length, dtype=float)
    for position, current in enumerate(path):
        escalation[position] = transition[
            position, current, STATE_RISK > STATE_RISK[current]
        ].sum()
    dwell = np.ones(length, dtype=int)
    for position in range(1, length):
        dwell[position] = dwell[position - 1] + 1 if path[position] == path[position - 1] else 1
    sequence["hsmm_state_index"] = path
    sequence["hsmm_state"] = [STATE_NAMES[item] for item in path]
    sequence["hsmm_state_confidence"] = emission[np.arange(length), path]
    sequence["hsmm_risk_score"] = np.clip(0.60 * expected_risk + 0.40 * decoded_risk, 0.0, 1.0)
    sequence["hsmm_escalation_probability"] = np.clip(escalation, 0.0, 1.0)
    sequence["hsmm_dwell_steps"] = dwell
    for state_index, name in enumerate(STATE_NAMES):
        sequence[f"p_state_{name.lower()}"] = emission[:, state_index]
    return sequence


def decode_frame(bundle: AnchoredContextualHSMM, frame: pd.DataFrame) -> pd.DataFrame:
    decoded = [
        decode_sequence(bundle, sequence)
        for _, sequence in frame.groupby(["evaluation_role", "port_call_id"], sort=False)
    ]
    if not decoded:
        raise ValueError("Anchored HSMM received no sequences")
    return pd.concat(decoded, ignore_index=True)


def _percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    if len(reference) == 0:
        raise ValueError("Empty HSMM calibration reference")
    return np.searchsorted(reference, np.asarray(values, dtype=float), side="right") / len(reference)


def add_anchored_scores(
    decoded: pd.DataFrame,
    fit_mask: pd.Series,
    hsmm_weight: float,
) -> pd.DataFrame:
    if not 0.0 <= hsmm_weight <= 1.0:
        raise ValueError("HSMM weight must be in [0,1]")
    output = decoded.copy()
    output["base_temporal_priority_score"] = (
        0.45 * output["p_delay_gt3"]
        + 0.30 * output["p_gt3_breach_within_6h"]
        + 0.15 * output["p_gt3_breach_within_12h"]
        + 0.10 * output["p_gt3_breach_within_24h"]
    )
    output["base_critical_priority_score"] = (
        0.65 * output["p_delay_gt6"]
        + 0.35 * output["p_gt3_breach_within_6h"]
    )
    fit = np.asarray(fit_mask, dtype=bool)
    risk_rank = _percentile(
        output.loc[fit, "hsmm_risk_score"].to_numpy(),
        output["hsmm_risk_score"].to_numpy(),
    )
    escalation_rank = _percentile(
        output.loc[fit, "hsmm_escalation_probability"].to_numpy(),
        output["hsmm_escalation_probability"].to_numpy(),
    )
    critical_probability = output["p_state_critical_disruption"].to_numpy(dtype=float)
    anchored_temporal = 0.75 * risk_rank + 0.25 * escalation_rank
    anchored_critical = 0.65 * risk_rank + 0.35 * critical_probability
    output["temporal_priority_score"] = (
        (1.0 - hsmm_weight) * output["base_temporal_priority_score"]
        + hsmm_weight * anchored_temporal
    )
    output["critical_priority_score"] = (
        (1.0 - hsmm_weight) * output["base_critical_priority_score"]
        + hsmm_weight * anchored_critical
    )
    output["hsmm_weight"] = hsmm_weight
    return output


def event_lead_metrics(decisions: pd.DataFrame) -> dict[str, Any]:
    positive = decisions.loc[decisions["target_delay_gt_3h"].astype(bool)]
    positive_calls = int(positive["port_call_id"].nunique())
    alerted = positive.loc[
        positive["alert_active"]
        & pd.to_numeric(positive["target_breach_or_censor_h"], errors="coerce").ge(0)
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
