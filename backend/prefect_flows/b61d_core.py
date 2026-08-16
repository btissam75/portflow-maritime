from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler


MODEL_VERSION = "b61d-contextual-hsmm-v1"
SOURCE_MODEL_VERSION = "b61b-v2.1-maritime-recalibration-only-v1"
SOURCE_POLICY_VERSION = "b61c-dynamic-alert-policy-v1.1"
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


@dataclass
class ContextualHSMM:
    imputer: SimpleImputer
    scaler: RobustScaler
    emission_model: GaussianMixture
    component_to_state: np.ndarray
    transition_model: Pipeline
    transition_classes: np.ndarray
    duration_pmf: np.ndarray
    start_probability: np.ndarray
    observation_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    max_duration_bins: int
    duration_bin_hours: int
    fit_rows: int
    fit_calls: int
    transition_rows: int
    fit_cutoff: pd.Timestamp


def prepare_sequence_features(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "port_call_id",
        "landmark_at",
        "evaluation_role",
        "p_delay_gt3",
        "p_delay_gt6",
        "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h",
        "remaining_p10_h",
        "remaining_p50_h",
        "remaining_p90_h",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"B61D source misses columns: {missing}")
    output = frame.copy()
    output["landmark_at"] = pd.to_datetime(output["landmark_at"], utc=True)
    output = output.sort_values(
        ["evaluation_role", "port_call_id", "landmark_at"]
    ).reset_index(drop=True)
    output["remaining_interval_width_h"] = (
        pd.to_numeric(output["remaining_p90_h"], errors="coerce")
        - pd.to_numeric(output["remaining_p10_h"], errors="coerce")
    ).clip(lower=0.0)
    grouped = output.groupby(
        ["evaluation_role", "port_call_id"], sort=False
    )
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


def chronological_fit_mask(
    frame: pd.DataFrame,
    role: str = "VALID_SELECT",
    fraction: float = 0.70,
) -> tuple[pd.Series, pd.Timestamp]:
    role_rows = frame.loc[frame["evaluation_role"].eq(role)]
    if role_rows.empty:
        raise ValueError(f"No rows for HSMM fit role {role}")
    cutoff = role_rows["landmark_at"].quantile(fraction)
    mask = frame["evaluation_role"].eq(role) & frame["landmark_at"].le(cutoff)
    if int(mask.sum()) < 1_000:
        raise ValueError("Insufficient chronological rows for contextual HSMM fit")
    return mask, pd.Timestamp(cutoff)


def _state_mapping(
    component_means_original: np.ndarray,
    columns: tuple[str, ...],
) -> np.ndarray:
    index = {name: position for position, name in enumerate(columns)}
    risk = (
        0.30 * component_means_original[:, index["p_delay_gt3"]]
        + 0.20 * component_means_original[:, index["p_delay_gt6"]]
        + 0.20
        * component_means_original[:, index["p_gt3_breach_within_6h"]]
        + 0.15
        * component_means_original[:, index["p_gt3_breach_within_12h"]]
        + 0.10
        * component_means_original[:, index["p_gt3_breach_within_24h"]]
        + 0.05
        * np.maximum(component_means_original[:, index["plan_progress_ratio"]], 0.0)
    )
    ordered = list(np.argsort(risk))
    fluid = ordered[0]
    critical = ordered[-1]
    congested = ordered[-2]
    middle = [item for item in ordered if item not in (fluid, congested, critical)]
    delta_index = index["delta_p_delay_gt3"]
    pressure = max(middle, key=lambda item: component_means_original[item, delta_index])
    recovery = next(item for item in middle if item != pressure)
    component_to_state = np.empty(len(ordered), dtype=int)
    component_to_state[fluid] = 0
    component_to_state[pressure] = 1
    component_to_state[congested] = 2
    component_to_state[critical] = 3
    component_to_state[recovery] = 4
    return component_to_state


def _pseudo_state_probabilities(
    model: GaussianMixture,
    transformed: np.ndarray,
    component_to_state: np.ndarray,
) -> np.ndarray:
    component_probability = model.predict_proba(transformed)
    state_probability = np.zeros_like(component_probability)
    for component, state in enumerate(component_to_state):
        state_probability[:, state] = component_probability[:, component]
    return np.clip(state_probability, 1e-12, 1.0)


def _transition_rows(
    fit_frame: pd.DataFrame,
    pseudo_state: np.ndarray,
    duration_bin_hours: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[int]]:
    indexed = fit_frame.copy()
    indexed["_pseudo_state"] = pseudo_state
    features: list[np.ndarray] = []
    targets: list[int] = []
    run_states: list[np.ndarray] = []
    run_durations: list[int] = []
    for _, sequence in indexed.groupby("port_call_id", sort=False):
        sequence = sequence.sort_values("landmark_at")
        states = sequence["_pseudo_state"].to_numpy(dtype=int)
        if len(states) == 0:
            continue
        start = 0
        for position in range(1, len(states) + 1):
            if position == len(states) or states[position] != states[start]:
                elapsed = (
                    sequence["landmark_at"].iloc[position - 1]
                    - sequence["landmark_at"].iloc[start]
                ).total_seconds() / 3600.0
                run_states.append(np.array([states[start]], dtype=int))
                run_durations.append(
                    max(1, int(math.ceil(elapsed / duration_bin_hours)) + 1)
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
            one_hot = np.eye(len(STATE_NAMES), dtype=float)[previous]
            context = sequence.iloc[position][list(CONTEXT_COLUMNS)].to_numpy(
                dtype=float
            )
            features.append(np.concatenate([one_hot, context]))
            targets.append(int(states[position]))
    if not features:
        raise ValueError("No valid transitions are available for B61D")
    return np.vstack(features), np.asarray(targets), run_states, run_durations


def _duration_distribution(
    run_states: list[np.ndarray],
    run_durations: list[int],
    max_duration_bins: int,
) -> np.ndarray:
    counts = np.ones((len(STATE_NAMES), max_duration_bins), dtype=float)
    for state_array, duration in zip(run_states, run_durations):
        state = int(state_array[0])
        counts[state, min(duration, max_duration_bins) - 1] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def fit_contextual_hsmm(
    frame: pd.DataFrame,
    fit_mask: pd.Series,
    fit_cutoff: pd.Timestamp,
    random_state: int = 20260809,
    max_duration_bins: int = 12,
    duration_bin_hours: int = 6,
) -> ContextualHSMM:
    fit_frame = frame.loc[fit_mask].copy()
    fit_frame = fit_frame.loc[
        fit_frame.groupby("port_call_id")["port_call_id"].transform("size").ge(2)
    ].copy()
    if len(fit_frame) < 1_000 or fit_frame["port_call_id"].nunique() < 300:
        raise ValueError("B61D needs at least 1,000 rows and 300 multi-landmark calls")
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    raw = fit_frame[list(OBSERVATION_COLUMNS)].to_numpy(dtype=float)
    imputed = imputer.fit_transform(raw)
    transformed = scaler.fit_transform(imputed)
    emission = GaussianMixture(
        n_components=len(STATE_NAMES),
        covariance_type="diag",
        reg_covar=1e-4,
        n_init=5,
        max_iter=400,
        random_state=random_state,
    )
    emission.fit(transformed)
    means_original = scaler.inverse_transform(emission.means_)
    component_to_state = _state_mapping(means_original, OBSERVATION_COLUMNS)
    state_probability = _pseudo_state_probabilities(
        emission, transformed, component_to_state
    )
    pseudo_state = state_probability.argmax(axis=1)
    transition_x, transition_y, run_states, run_durations = _transition_rows(
        fit_frame, pseudo_state, duration_bin_hours
    )
    if len(np.unique(transition_y)) < 2:
        raise ValueError("Contextual HSMM needs at least two transition states")
    transition = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=800,
                    class_weight="balanced",
                    C=0.75,
                    random_state=random_state,
                ),
            ),
        ]
    )
    transition.fit(transition_x, transition_y)
    start_counts = np.ones(len(STATE_NAMES), dtype=float)
    state_lookup = pd.Series(pseudo_state, index=fit_frame.index)
    for _, sequence in fit_frame.groupby("port_call_id", sort=False):
        first_index = sequence.sort_values("landmark_at").index[0]
        start_counts[int(state_lookup.loc[first_index])] += 1.0
    start_probability = start_counts / start_counts.sum()
    duration_pmf = _duration_distribution(
        run_states, run_durations, max_duration_bins
    )
    return ContextualHSMM(
        imputer=imputer,
        scaler=scaler,
        emission_model=emission,
        component_to_state=component_to_state,
        transition_model=transition,
        transition_classes=np.asarray(
            transition.named_steps["model"].classes_, dtype=int
        ),
        duration_pmf=duration_pmf,
        start_probability=start_probability,
        observation_columns=OBSERVATION_COLUMNS,
        context_columns=CONTEXT_COLUMNS,
        max_duration_bins=max_duration_bins,
        duration_bin_hours=duration_bin_hours,
        fit_rows=len(fit_frame),
        fit_calls=int(fit_frame["port_call_id"].nunique()),
        transition_rows=len(transition_y),
        fit_cutoff=fit_cutoff,
    )


def _transition_tensor(bundle: ContextualHSMM, sequence: pd.DataFrame) -> np.ndarray:
    states = len(STATE_NAMES)
    result = np.full((len(sequence), states, states), 1e-12, dtype=float)
    eye = np.eye(states, dtype=float)
    for position in range(len(sequence)):
        context = sequence.iloc[position][list(bundle.context_columns)].to_numpy(
            dtype=float
        )
        context = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
        inputs = np.vstack(
            [np.concatenate([eye[previous], context]) for previous in range(states)]
        )
        predicted = bundle.transition_model.predict_proba(inputs)
        for class_position, state in enumerate(bundle.transition_classes):
            result[position, :, int(state)] = predicted[:, class_position]
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
    bundle: ContextualHSMM,
    sequence: pd.DataFrame,
) -> pd.DataFrame:
    sequence = sequence.sort_values("landmark_at").copy()
    raw = sequence[list(bundle.observation_columns)].to_numpy(dtype=float)
    transformed = bundle.scaler.transform(bundle.imputer.transform(raw))
    emission = _pseudo_state_probabilities(
        bundle.emission_model, transformed, bundle.component_to_state
    )
    transition = _transition_tensor(bundle, sequence)
    log_emission = np.log(emission)
    cumulative = np.vstack(
        [np.zeros((1, len(STATE_NAMES))), np.cumsum(log_emission, axis=0)]
    )
    length = len(sequence)
    states = len(STATE_NAMES)
    score = np.full((length, states), -np.inf)
    back_state = np.full((length, states), -1, dtype=int)
    back_start = np.full((length, states), -1, dtype=int)
    timestamps = sequence["landmark_at"].reset_index(drop=True)
    for end in range(length):
        for state in range(states):
            for start in range(0, end + 1):
                duration = _duration_bin(
                    timestamps,
                    start,
                    end,
                    bundle.duration_bin_hours,
                    bundle.max_duration_bins,
                )
                segment_score = cumulative[end + 1, state] - cumulative[start, state]
                duration_score = math.log(bundle.duration_pmf[state, duration - 1])
                if start == 0:
                    candidate = (
                        math.log(bundle.start_probability[state])
                        + segment_score
                        + duration_score
                    )
                    previous_state = -1
                else:
                    previous_scores = score[start - 1] + np.log(
                        transition[start, :, state]
                    )
                    previous_state = int(np.argmax(previous_scores))
                    candidate = (
                        previous_scores[previous_state]
                        + segment_score
                        + duration_score
                    )
                if candidate > score[end, state]:
                    score[end, state] = candidate
                    back_state[end, state] = previous_state
                    back_start[end, state] = start
    path = np.zeros(length, dtype=int)
    end = length - 1
    state = int(np.argmax(score[end]))
    while end >= 0:
        start = int(back_start[end, state])
        path[start : end + 1] = state
        previous_state = int(back_state[end, state])
        end = start - 1
        if end >= 0:
            state = previous_state
    confidence = emission[np.arange(length), path]
    expected_risk = emission @ STATE_RISK
    decoded_risk = STATE_RISK[path]
    hsmm_risk = 0.65 * expected_risk + 0.35 * decoded_risk
    escalation = np.zeros(length, dtype=float)
    for position, current in enumerate(path):
        current_risk = STATE_RISK[current]
        escalation[position] = transition[position, current, STATE_RISK > current_risk].sum()
    dwell = np.ones(length, dtype=int)
    for position in range(1, length):
        dwell[position] = dwell[position - 1] + 1 if path[position] == path[position - 1] else 1
    sequence["hsmm_state_index"] = path
    sequence["hsmm_state"] = [STATE_NAMES[item] for item in path]
    sequence["hsmm_state_confidence"] = confidence
    sequence["hsmm_risk_score"] = np.clip(hsmm_risk, 0.0, 1.0)
    sequence["hsmm_escalation_probability"] = np.clip(escalation, 0.0, 1.0)
    sequence["hsmm_dwell_steps"] = dwell
    for state, name in enumerate(STATE_NAMES):
        sequence[f"p_state_{name.lower()}"] = emission[:, state]
    return sequence


def decode_frame(bundle: ContextualHSMM, frame: pd.DataFrame) -> pd.DataFrame:
    decoded = []
    for _, sequence in frame.groupby(
        ["evaluation_role", "port_call_id"], sort=False
    ):
        decoded.append(decode_sequence(bundle, sequence))
    if not decoded:
        raise ValueError("B61D decoder received no port-call sequences")
    return pd.concat(decoded, ignore_index=True)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    if len(reference) == 0:
        raise ValueError("Empty percentile reference")
    return np.searchsorted(reference, values, side="right") / float(len(reference))


def add_lead_aware_scores(
    decoded: pd.DataFrame,
    fit_mask: pd.Series,
    hsmm_weight: float,
) -> pd.DataFrame:
    output = decoded.copy()
    early_raw = (
        0.50 * output["p_gt3_breach_within_24h"]
        + 0.30 * output["p_gt3_breach_within_12h"]
        + 0.10 * output["p_gt3_breach_within_6h"]
        + 0.10 * output["p_delay_gt3"]
    )
    critical_raw = (
        0.60 * output["p_gt3_breach_within_6h"]
        + 0.40 * output["p_delay_gt6"]
    )
    fit_positions = np.asarray(fit_mask, dtype=bool)
    base_percentile = empirical_percentile(
        early_raw.to_numpy()[fit_positions], early_raw.to_numpy()
    )
    hsmm_percentile = empirical_percentile(
        output["hsmm_risk_score"].to_numpy()[fit_positions],
        output["hsmm_risk_score"].to_numpy(),
    )
    critical_percentile = empirical_percentile(
        critical_raw.to_numpy()[fit_positions], critical_raw.to_numpy()
    )
    escalation_percentile = empirical_percentile(
        output["hsmm_escalation_probability"].to_numpy()[fit_positions],
        output["hsmm_escalation_probability"].to_numpy(),
    )
    output["temporal_priority_score"] = (
        (1.0 - hsmm_weight) * base_percentile
        + hsmm_weight * (0.75 * hsmm_percentile + 0.25 * escalation_percentile)
    )
    output["critical_priority_score"] = (
        (1.0 - hsmm_weight) * critical_percentile
        + hsmm_weight * hsmm_percentile
    )
    output["hsmm_weight"] = hsmm_weight
    return output


def event_lead_metrics(decisions: pd.DataFrame) -> dict[str, Any]:
    positive = decisions.loc[decisions["target_delay_gt_3h"].astype(bool)].copy()
    positive_calls = int(positive["port_call_id"].nunique())
    alerted = positive.loc[
        positive["alert_active"]
        & pd.to_numeric(positive["target_breach_or_censor_h"], errors="coerce").ge(0)
    ].copy()
    if alerted.empty:
        leads = pd.Series(dtype=float)
    else:
        leads = alerted.groupby("port_call_id")["target_breach_or_censor_h"].max()
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
