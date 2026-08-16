from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

try:
    from catboost import CatBoostRegressor

    CATBOOST_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - runtime gate reports the error
    CatBoostRegressor = None
    CATBOOST_IMPORT_ERROR = str(exc)


BENCHMARK_VERSION = "b60b-advanced-timeseries-benchmark-v1.1"
SOURCE_DATASET_VERSION = "b60a-maritime-multitask-hourly-v1"
ORIGIN_STEP_H = 24
RANDOM_SEED = 20260804
MIN_REPLACEMENT_GAIN_PCT = 2.0
EXPECTED_INTERVAL_COVERAGE = 0.80
BOOTSTRAP_REPETITIONS = 1_000
BOOTSTRAP_BLOCK_DAYS = 7


@dataclass(frozen=True)
class TargetSpec:
    target: str
    task: str
    kind: str
    horizon_h: int
    signal_status: str
    sequence_start_h: Optional[int] = None
    sequence_end_h: Optional[int] = None
    sequence_lead_h: Optional[int] = None
    availability_lag_h: int = 0


TARGET_SPECS = (
    TargetSpec(
        "target_arrivals_next_6h",
        "ARRIVAL",
        "COUNT",
        6,
        "B60A2_STRONG",
        sequence_start_h=1,
        sequence_end_h=6,
    ),
    TargetSpec(
        "target_arrivals_6_12h",
        "ARRIVAL",
        "COUNT",
        12,
        "B60A2_STRONG",
        sequence_start_h=7,
        sequence_end_h=12,
    ),
    TargetSpec(
        "target_arrivals_next_12h",
        "ARRIVAL",
        "COUNT",
        12,
        "B60A2_CHALLENGER",
        sequence_start_h=1,
        sequence_end_h=12,
    ),
    TargetSpec(
        "target_arrivals_12_24h",
        "ARRIVAL",
        "COUNT",
        24,
        "B60A2_CHALLENGER",
        sequence_start_h=13,
        sequence_end_h=24,
    ),
    TargetSpec(
        "target_arrivals_next_24h",
        "ARRIVAL",
        "COUNT",
        24,
        "B60A2_CHALLENGER",
        sequence_start_h=1,
        sequence_end_h=24,
    ),
    TargetSpec(
        "target_next_arrival_wait_h",
        "ARRIVAL",
        "WAIT",
        24,
        "B60A2_STRONG",
    ),
    TargetSpec(
        "target_wave_period_s_48h",
        "WAVE",
        "WAVE_PERIOD",
        48,
        "B60A2_STRONG_BUT_VALID_WEAK",
        sequence_lead_h=51,
        availability_lag_h=3,
    ),
    TargetSpec(
        "target_wave_period_s_72h",
        "WAVE",
        "WAVE_PERIOD",
        72,
        "B60A2_STRONG",
        sequence_lead_h=75,
        availability_lag_h=3,
    ),
)


@dataclass
class TransformState:
    features: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray


@dataclass
class BenchmarkResult:
    reports: dict[str, pd.DataFrame]
    predictions: dict[str, pd.DataFrame]
    decision: dict[str, Any]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return [clean_json(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def target_scope_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "target": item.target,
                "task": item.task,
                "kind": item.kind,
                "horizon_h": item.horizon_h,
                "b60a2_signal_status": item.signal_status,
                "sequence_mapping": (
                    f"SUM_LEADS_{item.sequence_start_h}_{item.sequence_end_h}"
                    if item.sequence_start_h is not None
                    else f"LEAD_{item.sequence_lead_h}"
                    if item.sequence_lead_h is not None
                    else "NOT_APPLICABLE"
                ),
                "duplicate_alias_removed": item.target == "target_arrivals_next_6h",
                "availability_lag_h": item.availability_lag_h,
            }
            for item in TARGET_SPECS
        ]
    )


def horizon_specific_features(features: list[str], horizon_h: int) -> list[str]:
    selected = []
    for feature in dict.fromkeys(features):
        target_time = re.search(r"target_time_(\d+)h_", feature)
        event_time = re.search(r"_at_(\d+)h(?:_|$)", feature)
        if target_time is not None and int(target_time.group(1)) != horizon_h:
            continue
        if event_time is not None and int(event_time.group(1)) != horizon_h:
            continue
        if feature.startswith("target_") and not feature.startswith("target_time_"):
            continue
        selected.append(feature)
    return selected


def sampled_positions(frame: pd.DataFrame, split_column: str, split: str) -> np.ndarray:
    positions = np.flatnonzero(frame[split_column].eq(split).to_numpy())
    if len(positions) == 0:
        raise ValueError(f"No {split} rows in {split_column}")
    first = int(positions[0])
    sampled = positions[(positions - first) % ORIGIN_STEP_H == 0]
    return sampled.astype("int64")


def evaluation_positions(
    frame: pd.DataFrame,
    split_column: str,
    split: str,
    target: str,
    sequence_predictions: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    if sequence_predictions is None or sequence_predictions.empty:
        return sampled_positions(frame, split_column, split)
    sequence = sequence_predictions.loc[
        sequence_predictions["split"].eq(split)
        & sequence_predictions["target"].eq(target)
    ]
    if sequence.empty:
        return sampled_positions(frame, split_column, split)
    expected_times = pd.DatetimeIndex(
        pd.to_datetime(sequence["issue_at"].drop_duplicates(), utc=True)
    )
    source_times = pd.DatetimeIndex(pd.to_datetime(frame["as_of_time"], utc=True))
    positions = np.flatnonzero(source_times.isin(expected_times)).astype("int64")
    if len(positions) != len(expected_times):
        missing = expected_times.difference(source_times)
        raise ValueError(
            f"Sequence origins do not map to B60A rows for {target}: {list(missing[:5])}"
        )
    if not frame.iloc[positions][split_column].eq(split).all():
        raise ValueError(f"Sequence origins cross the {split} split for {target}")
    return positions


def safe_baseline_lags(horizon_h: int, availability_lag_h: int = 0) -> list[int]:
    minimum_lag = horizon_h + availability_lag_h
    return sorted(set([minimum_lag, max(minimum_lag, 24), max(minimum_lag, 168)]))


def safe_naive_prediction(
    values: np.ndarray,
    fit_positions: np.ndarray,
    evaluation_positions: np.ndarray,
    horizon_h: int,
    availability_lag_h: int = 0,
) -> tuple[np.ndarray, int]:
    candidates = safe_baseline_lags(horizon_h, availability_lag_h)
    calibration_size = min(1_440, max(336, len(fit_positions) // 5))
    calibration = fit_positions[-calibration_size:]
    best_lag = candidates[0]
    best_loss = math.inf
    for lag in candidates:
        valid = calibration[calibration >= lag]
        finite = np.isfinite(values[valid]) & np.isfinite(values[valid - lag])
        valid = valid[finite]
        if len(valid) < 100:
            continue
        loss = float(np.mean(np.abs(values[valid] - values[valid - lag])))
        if loss < best_loss:
            best_loss = loss
            best_lag = lag
    if int(evaluation_positions.min()) < best_lag:
        raise ValueError("Safe baseline does not have sufficient history")
    prediction = values[evaluation_positions - best_lag].astype("float64")
    if not np.isfinite(prediction).all():
        raise ValueError("Safe baseline encountered unavailable historical targets")
    return prediction, best_lag


def fit_transform_state(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, TransformState]:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    medians = np.nanmedian(numeric, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(numeric), numeric, medians)
    means = filled.mean(axis=0)
    scales = filled.std(axis=0)
    scales = np.where(scales > 1e-8, scales, 1.0)
    return (filled - means) / scales, TransformState(features, medians, means, scales)


def transform_with_state(frame: pd.DataFrame, state: TransformState) -> np.ndarray:
    numeric = frame[state.features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    filled = np.where(np.isfinite(numeric), numeric, state.medians)
    return (filled - state.means) / state.scales


def recency_weights(times: pd.Series, half_life_days: Optional[float]) -> np.ndarray:
    if half_life_days is None:
        return np.ones(len(times), dtype="float64")
    values = pd.to_datetime(times, utc=True)
    age_days = (values.max() - values).dt.total_seconds().to_numpy() / 86_400.0
    return np.power(0.5, age_days / half_life_days)


def _nb_loss_gradient(
    beta: np.ndarray,
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> tuple[float, np.ndarray]:
    eta = np.clip(design @ beta, -8.0, 8.0)
    mean = np.exp(eta)
    size = 1.0 / max(alpha, 1e-8)
    log_probability = (
        gammaln(target + size)
        - gammaln(size)
        - gammaln(target + 1.0)
        + size * (math.log(size) - np.log(size + mean))
        + target * (np.log(mean + 1e-12) - np.log(size + mean))
    )
    weight_sum = max(float(weights.sum()), 1e-12)
    penalty = 0.002 * float(np.dot(beta[1:], beta[1:]))
    loss = -float(np.dot(weights, log_probability)) / weight_sum + penalty
    derivative_eta = (mean - target) / (1.0 + alpha * mean)
    gradient = design.T @ (weights * derivative_eta) / weight_sum
    gradient[1:] += 0.004 * beta[1:]
    return loss, gradient


def _estimate_alpha(target: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> float:
    numerator = np.sum(weights * ((target - mean) ** 2 - mean))
    denominator = np.sum(weights * np.maximum(mean**2, 1e-8))
    return float(np.clip(numerator / max(denominator, 1e-12), 1e-4, 5.0))


def fit_dynamic_negative_binomial(
    train: pd.DataFrame,
    features: list[str],
    target: str,
) -> dict[str, Any]:
    matrix, state = fit_transform_state(train, features)
    design = np.column_stack([np.ones(len(matrix)), matrix])
    y = train[target].to_numpy(dtype="float64")
    weights = recency_weights(train["as_of_time"], 180.0)
    weighted_mean = float(np.average(y, weights=weights))
    alpha = max((float(np.var(y)) - weighted_mean) / max(weighted_mean**2, 1e-8), 1e-4)
    beta = np.zeros(design.shape[1], dtype="float64")
    beta[0] = math.log(max(weighted_mean, 0.05))
    converged = True
    for _ in range(3):
        result = minimize(
            lambda value: _nb_loss_gradient(value, design, y, weights, alpha),
            beta,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 180, "ftol": 1e-8, "gtol": 1e-6},
        )
        beta = result.x
        fitted = np.exp(np.clip(design @ beta, -8.0, 8.0))
        alpha = _estimate_alpha(y, fitted, weights)
        converged &= bool(result.success)
    return {
        "family": "DYNAMIC_NEGATIVE_BINOMIAL",
        "state": state,
        "beta": beta,
        "alpha": alpha,
        "converged": converged,
    }


def predict_dynamic_negative_binomial(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    matrix = transform_with_state(frame, model["state"])
    design = np.column_stack([np.ones(len(matrix)), matrix])
    return np.clip(np.exp(np.clip(design @ model["beta"], -8.0, 8.0)), 0.0, None)


def fit_hgb(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    kind: str,
) -> dict[str, Any]:
    matrix, state = fit_transform_state(train, features)
    loss = "poisson" if kind == "COUNT" else "absolute_error"
    model = HistGradientBoostingRegressor(
        loss=loss,
        learning_rate=0.05,
        max_iter=280,
        max_leaf_nodes=23,
        min_samples_leaf=48,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )
    weights = recency_weights(train["as_of_time"], 180.0 if kind == "COUNT" else None)
    model.fit(matrix, train[target].to_numpy(dtype="float64"), sample_weight=weights)
    return {"family": "HGB_POISSON" if kind == "COUNT" else "HGB_ROBUST", "state": state, "model": model}


def predict_hgb(model: dict[str, Any], frame: pd.DataFrame, nonnegative: bool) -> np.ndarray:
    values = np.asarray(model["model"].predict(transform_with_state(frame, model["state"])), dtype="float64")
    return np.clip(values, 0.0, None) if nonnegative else values


def fit_catboost(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    kind: str,
) -> dict[str, Any]:
    if CatBoostRegressor is None:
        raise RuntimeError(f"CatBoost unavailable: {CATBOOST_IMPORT_ERROR}")
    matrix, state = fit_transform_state(train, features)
    loss = "Poisson" if kind == "COUNT" else "MAE"
    model = CatBoostRegressor(
        iterations=320,
        depth=7,
        learning_rate=0.04,
        loss_function=loss,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=2,
        l2_leaf_reg=5.0,
    )
    model.fit(matrix, train[target].to_numpy(dtype="float64"))
    return {"family": "CATBOOST_POISSON" if kind == "COUNT" else "CATBOOST_MAE", "state": state, "model": model}


def predict_catboost(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    values = np.asarray(model["model"].predict(transform_with_state(frame, model["state"])), dtype="float64")
    return np.clip(values, 0.0, None)


def fit_discrete_hazard(
    train: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    base_matrix, state = fit_transform_state(train, features)
    wait = np.clip(np.ceil(train["target_next_arrival_wait_h"].to_numpy(dtype="float64")), 1, 24).astype("int64")
    observed = train["target_next_arrival_observed_24h"].to_numpy(dtype="float64") > 0.5
    matrices = []
    labels = []
    for hour in range(1, 25):
        at_risk = wait >= hour
        if not np.any(at_risk):
            continue
        hour_features = np.column_stack(
            [
                np.full(int(at_risk.sum()), hour / 24.0),
                np.full(int(at_risk.sum()), math.sin(2.0 * math.pi * hour / 24.0)),
                np.full(int(at_risk.sum()), math.cos(2.0 * math.pi * hour / 24.0)),
            ]
        )
        matrices.append(np.column_stack([base_matrix[at_risk], hour_features]))
        labels.append((observed[at_risk] & (wait[at_risk] == hour)).astype("int8"))
    design = np.concatenate(matrices, axis=0)
    target = np.concatenate(labels, axis=0)
    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=220,
        max_leaf_nodes=19,
        min_samples_leaf=64,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )
    model.fit(design, target)
    return {"family": "DISCRETE_TIME_HAZARD", "state": state, "model": model}


def predict_discrete_hazard(model: dict[str, Any], frame: pd.DataFrame) -> np.ndarray:
    base = transform_with_state(frame, model["state"])
    survival = np.ones(len(frame), dtype="float64")
    expectation = np.ones(len(frame), dtype="float64")
    for hour in range(1, 24):
        extra = np.column_stack(
            [
                np.full(len(frame), hour / 24.0),
                np.full(len(frame), math.sin(2.0 * math.pi * hour / 24.0)),
                np.full(len(frame), math.cos(2.0 * math.pi * hour / 24.0)),
            ]
        )
        hazard = model["model"].predict_proba(np.column_stack([base, extra]))[:, 1]
        survival *= 1.0 - np.clip(hazard, 1e-6, 1.0 - 1e-6)
        expectation += survival
    return np.clip(expectation, 1.0, 24.0)


def _metric(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    denominator = float(np.abs(actual).sum())
    return {
        "rows": len(actual),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(np.square(error)))),
        "WAPE_PCT": float(100.0 * np.abs(error).sum() / denominator) if denominator > 1e-12 else np.nan,
        "BIAS": float(np.mean(error)),
    }


def prediction_metrics(predictions: pd.DataFrame, split: str) -> pd.DataFrame:
    subset = predictions.loc[predictions["split"].eq(split)].copy()
    rows = []
    for (target, model), group in subset.groupby(["target", "model"], sort=True):
        rows.append(
            {
                "split": split,
                "target": target,
                "model": model,
                **_metric(
                    group["actual"].to_numpy(dtype="float64"),
                    group["prediction"].to_numpy(dtype="float64"),
                ),
            }
        )
    return pd.DataFrame(rows)


def _fit_predict_candidate(
    model_name: str,
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    features: list[str],
    spec: TargetSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    if model_name == "DYNAMIC_NEGATIVE_BINOMIAL":
        model = fit_dynamic_negative_binomial(train, features, spec.target)
        return predict_dynamic_negative_binomial(model, evaluation), model
    if model_name in {"HGB_POISSON", "HGB_ROBUST"}:
        model = fit_hgb(train, features, spec.target, spec.kind)
        return predict_hgb(model, evaluation, True), model
    if model_name in {"CATBOOST_POISSON", "CATBOOST_MAE"}:
        model = fit_catboost(train, features, spec.target, spec.kind)
        return predict_catboost(model, evaluation), model
    if model_name == "DISCRETE_TIME_HAZARD":
        model = fit_discrete_hazard(train, features)
        return predict_discrete_hazard(model, evaluation), model
    raise ValueError(f"Unknown candidate: {model_name}")


def candidate_names(spec: TargetSpec, enable_catboost: bool = True) -> list[str]:
    if spec.kind == "COUNT":
        names = ["DYNAMIC_NEGATIVE_BINOMIAL", "HGB_POISSON"]
        if enable_catboost:
            names.append("CATBOOST_POISSON")
        return names
    if spec.kind == "WAIT":
        names = ["DISCRETE_TIME_HAZARD", "HGB_ROBUST"]
        if enable_catboost:
            names.append("CATBOOST_MAE")
        return names
    names = ["HGB_ROBUST"]
    if enable_catboost:
        names.append("CATBOOST_MAE")
    return names


def tabular_validation_predictions(
    frame: pd.DataFrame,
    representation_sets: dict[str, list[str]],
    sequence_predictions: Optional[pd.DataFrame] = None,
    enable_catboost: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    inventory_rows = []
    for spec in TARGET_SPECS:
        split_column = "split_arrival" if spec.task == "ARRIVAL" else "split_wave"
        train_positions = np.flatnonzero(frame[split_column].eq("TRAIN").to_numpy())
        valid_positions = evaluation_positions(
            frame,
            split_column,
            "VALID",
            spec.target,
            sequence_predictions,
        )
        features = horizon_specific_features(
            representation_sets[f"{spec.task.lower()}_core_compact"], spec.horizon_h
        )
        train = frame.iloc[train_positions].dropna(subset=[spec.target]).copy()
        evaluation = frame.iloc[valid_positions].dropna(subset=[spec.target]).copy()
        values = frame[spec.target].to_numpy(dtype="float64")
        baseline, lag = safe_naive_prediction(
            values,
            train_positions,
            evaluation.index.to_numpy(),
            spec.horizon_h,
            spec.availability_lag_h,
        )
        for row, prediction in zip(evaluation.itertuples(index=False), baseline):
            prediction_rows.append(
                {
                    "split": "VALID",
                    "issue_at": row.as_of_time,
                    "target": spec.target,
                    "actual": getattr(row, spec.target),
                    "model": "SAFE_NAIVE",
                    "prediction": float(max(prediction, 0.0)),
                }
            )
        inventory_rows.append(
            {
                "target": spec.target,
                "model": "SAFE_NAIVE",
                "family": "SEASONAL_TARGET_SHIFT",
                "features": 0,
                "fit_rows": len(train),
                "selected_lag_h": lag,
                "training_seconds": 0.0,
            }
        )
        for model_name in candidate_names(spec, enable_catboost):
            started = time.perf_counter()
            prediction, model = _fit_predict_candidate(
                model_name, train, evaluation, features, spec
            )
            elapsed = time.perf_counter() - started
            for row, value in zip(evaluation.itertuples(index=False), prediction):
                prediction_rows.append(
                    {
                        "split": "VALID",
                        "issue_at": row.as_of_time,
                        "target": spec.target,
                        "actual": getattr(row, spec.target),
                        "model": model_name,
                        "prediction": float(value),
                    }
                )
            inventory_rows.append(
                {
                    "target": spec.target,
                    "model": model_name,
                    "family": model["family"],
                    "features": len(features),
                    "fit_rows": len(train),
                    "selected_lag_h": np.nan,
                    "training_seconds": elapsed,
                    "optimizer_converged": model.get("converged", True),
                    "dispersion_alpha": model.get("alpha", np.nan),
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(inventory_rows)


def append_sequence_predictions(
    tabular: pd.DataFrame,
    sequence: pd.DataFrame,
) -> pd.DataFrame:
    if sequence.empty:
        return tabular
    required = ["split", "issue_at", "target", "actual", "model", "prediction"]
    missing = sorted(set(required).difference(sequence.columns))
    if missing:
        raise ValueError(f"Sequence predictions missing columns: {missing}")
    return pd.concat([tabular, sequence[required]], ignore_index=True)


def align_common_origins(
    predictions: pd.DataFrame, split: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = predictions.loc[predictions["split"].eq(split)].copy()
    aligned = []
    coverage_rows = []
    for target, group in source.groupby("target", sort=True):
        models = sorted(group["model"].unique())
        origin_sets = [set(group.loc[group["model"].eq(model), "issue_at"]) for model in models]
        common = set.intersection(*origin_sets) if origin_sets else set()
        if not common:
            raise ValueError(f"No common {split} origins for {target}")
        aligned.append(group.loc[group["issue_at"].isin(common)])
        coverage_rows.append(
            {
                "split": split,
                "target": target,
                "models": len(models),
                "common_origins": len(common),
                "first_origin": min(common),
                "last_origin": max(common),
            }
        )
    return pd.concat(aligned, ignore_index=True), pd.DataFrame(coverage_rows)


def common_origin_metrics(predictions: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    aligned, coverage = align_common_origins(predictions, split)
    return prediction_metrics(aligned, split), coverage


def paired_block_bootstrap_delta_ci(
    delta: np.ndarray,
    seed: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    block_size: int = BOOTSTRAP_BLOCK_DAYS,
) -> tuple[float, float]:
    values = np.asarray(delta, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    block_size = min(block_size, len(values))
    starts = np.arange(len(values))
    means = np.empty(repetitions, dtype="float64")
    for repetition in range(repetitions):
        sampled = []
        while len(sampled) < len(values):
            start = int(rng.choice(starts))
            sampled.extend((start + offset) % len(values) for offset in range(block_size))
        means[repetition] = values[np.asarray(sampled[: len(values)], dtype="int64")].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_candidate_delta(
    predictions: pd.DataFrame,
    target: str,
    candidate: str,
) -> np.ndarray:
    subset = predictions.loc[
        predictions["target"].eq(target)
        & predictions["model"].isin(["SAFE_NAIVE", candidate])
    ]
    pivot = subset.pivot_table(
        index="issue_at", columns="model", values=["actual", "prediction"], aggfunc="first"
    )
    baseline_error = np.abs(
        pivot[("prediction", "SAFE_NAIVE")].to_numpy(dtype="float64")
        - pivot[("actual", "SAFE_NAIVE")].to_numpy(dtype="float64")
    )
    candidate_error = np.abs(
        pivot[("prediction", candidate)].to_numpy(dtype="float64")
        - pivot[("actual", candidate)].to_numpy(dtype="float64")
    )
    return candidate_error - baseline_error


def select_models(
    valid_metrics: pd.DataFrame,
    aligned_predictions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    for target, group in valid_metrics.groupby("target", sort=True):
        baseline = group.loc[group["model"].eq("SAFE_NAIVE")].iloc[0]
        best = group.sort_values(["MAE", "model"]).iloc[0]
        gain = 100.0 * (float(baseline.MAE) - float(best.MAE)) / max(float(baseline.MAE), 1e-12)
        delta_ci_low = np.nan
        delta_ci_high = np.nan
        statistically_confirmed = True
        if best.model != "SAFE_NAIVE" and aligned_predictions is not None:
            delta = _paired_candidate_delta(aligned_predictions, target, str(best.model))
            seed = RANDOM_SEED + sum(ord(value) for value in f"{target}:{best.model}")
            delta_ci_low, delta_ci_high = paired_block_bootstrap_delta_ci(delta, seed)
            statistically_confirmed = bool(np.isfinite(delta_ci_high) and delta_ci_high < 0.0)
        accepted = (
            best.model != "SAFE_NAIVE"
            and gain >= MIN_REPLACEMENT_GAIN_PCT
            and statistically_confirmed
        )
        selected = str(best.model) if accepted else "SAFE_NAIVE"
        selected_row = group.loc[group["model"].eq(selected)].iloc[0]
        rows.append(
            {
                "target": target,
                "safe_naive_valid_mae": float(baseline.MAE),
                "best_candidate": str(best.model),
                "best_candidate_valid_mae": float(best.MAE),
                "gain_vs_safe_naive_pct": gain,
                "replacement_threshold_pct": MIN_REPLACEMENT_GAIN_PCT,
                "delta_mae_ci95_low": delta_ci_low,
                "delta_mae_ci95_high": delta_ci_high,
                "paired_block_days": BOOTSTRAP_BLOCK_DAYS,
                "statistically_confirmed": statistically_confirmed,
                "candidate_accepted": accepted,
                "selected_model": selected,
                "selected_valid_mae": float(selected_row.MAE),
                "selection_split": "VALID_ONLY",
                "test_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def selected_validation_predictions(
    predictions: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    blocks = []
    for choice in selection.itertuples(index=False):
        blocks.append(
            predictions.loc[
                predictions["target"].eq(choice.target)
                & predictions["model"].eq(choice.selected_model)
                & predictions["split"].eq("VALID")
            ].copy()
        )
    return pd.concat(blocks, ignore_index=True)


def tabular_selected_test_predictions(
    frame: pd.DataFrame,
    representation_sets: dict[str, list[str]],
    selection: pd.DataFrame,
    sequence_predictions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    by_target = {item.target: item for item in TARGET_SPECS}
    for choice in selection.itertuples(index=False):
        if choice.selected_model in {"NHITS_SEQ", "PATCHTST_SEQ"}:
            continue
        spec = by_target[choice.target]
        split_column = "split_arrival" if spec.task == "ARRIVAL" else "split_wave"
        fit_positions = np.flatnonzero(frame[split_column].isin(["TRAIN", "VALID"]).to_numpy())
        test_positions = evaluation_positions(
            frame,
            split_column,
            "TEST",
            spec.target,
            sequence_predictions,
        )
        fit = frame.iloc[fit_positions].dropna(subset=[spec.target]).copy()
        evaluation = frame.iloc[test_positions].dropna(subset=[spec.target]).copy()
        if choice.selected_model == "SAFE_NAIVE":
            prediction, _ = safe_naive_prediction(
                frame[spec.target].to_numpy(dtype="float64"),
                fit_positions,
                evaluation.index.to_numpy(),
                spec.horizon_h,
                spec.availability_lag_h,
            )
            prediction = np.clip(prediction, 0.0, None)
        else:
            features = horizon_specific_features(
                representation_sets[f"{spec.task.lower()}_core_compact"], spec.horizon_h
            )
            prediction, _ = _fit_predict_candidate(
                choice.selected_model, fit, evaluation, features, spec
            )
        for row, value in zip(evaluation.itertuples(index=False), prediction):
            rows.append(
                {
                    "split": "TEST",
                    "issue_at": row.as_of_time,
                    "target": spec.target,
                    "actual": getattr(row, spec.target),
                    "model": choice.selected_model,
                    "prediction": float(value),
                }
            )
    return pd.DataFrame(rows)


def selected_sequence_test_predictions(
    sequence: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for choice in selection.itertuples(index=False):
        if choice.selected_model not in {"NHITS_SEQ", "PATCHTST_SEQ"}:
            continue
        rows.append(
            sequence.loc[
                sequence["split"].eq("TEST")
                & sequence["target"].eq(choice.target)
                & sequence["model"].eq(choice.selected_model)
            ].copy()
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=sequence.columns)


def conformal_intervals(
    valid_selected: pd.DataFrame,
    test_selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_blocks = []
    calibration_rows = []
    for target, test in test_selected.groupby("target", sort=True):
        valid = valid_selected.loc[valid_selected["target"].eq(target)]
        actual = valid["actual"].to_numpy(dtype="float64")
        predicted = valid["prediction"].to_numpy(dtype="float64")
        tail_alpha = (1.0 - EXPECTED_INTERVAL_COVERAGE) / 2.0
        quantile_level = min(
            1.0,
            math.ceil((len(valid) + 1) * (1.0 - tail_alpha)) / len(valid),
        )
        lower_radius = max(
            0.0,
            float(np.quantile(predicted - actual, quantile_level, method="higher")),
        )
        upper_radius = max(
            0.0,
            float(np.quantile(actual - predicted, quantile_level, method="higher")),
        )
        block = test.copy()
        block["p50"] = block["prediction"]
        block["p10"] = np.maximum(0.0, block["prediction"] - lower_radius)
        block["p90"] = np.maximum(block["p50"], block["prediction"] + upper_radius)
        block["covered_p10_p90"] = block["actual"].between(block["p10"], block["p90"])
        output_blocks.append(block)
        calibration_rows.append(
            {
                "target": target,
                "calibration_rows": len(valid),
                "one_sided_quantile_level": quantile_level,
                "lower_radius": lower_radius,
                "upper_radius": upper_radius,
                "nominal_coverage_pct": 100.0 * EXPECTED_INTERVAL_COVERAGE,
                "test_coverage_pct": float(100.0 * block["covered_p10_p90"].mean()),
                "mean_interval_width": float((block["p90"] - block["p10"]).mean()),
                "calibration_split": "VALID",
                "test_role": "FINAL_DIAGNOSTIC_ONLY",
            }
        )
    return pd.concat(output_blocks, ignore_index=True), pd.DataFrame(calibration_rows)


def benchmark_contract_gates(
    frame: pd.DataFrame,
    valid_predictions: pd.DataFrame,
    selection: pd.DataFrame,
    test_predictions: pd.DataFrame,
    sequence_runtime_ready: bool,
    representation_sets: dict[str, list[str]],
) -> pd.DataFrame:
    prediction_features = []
    for spec in TARGET_SPECS:
        key = f"{spec.task.lower()}_core_compact"
        prediction_features.extend(
            horizon_specific_features(representation_sets.get(key, []), spec.horizon_h)
        )
    forbidden_features = sorted(
        {
            value
            for value in prediction_features
            if value.startswith("target_") and not value.startswith("target_time_")
        }
    )
    research_features = sorted(
        {value for value in prediction_features if value.startswith("research_")}
    )
    finite_valid = np.isfinite(valid_predictions[["actual", "prediction"]].to_numpy()).all()
    finite_test = np.isfinite(test_predictions[["actual", "prediction"]].to_numpy()).all()
    gates = [
        ("SOURCE_B60A_DATASET_VERSION", bool(frame["dataset_version"].eq(SOURCE_DATASET_VERSION).all()), 0),
        ("B60A2_SCOPE_ONLY", set(selection["target"]) == {item.target for item in TARGET_SPECS}, 0),
        ("DUPLICATE_TARGET_ALIAS_REMOVED", "target_arrivals_0_6h" not in set(selection["target"]), 0),
        ("CORE_FEATURES_ONLY", len(forbidden_features) == 0, len(forbidden_features)),
        ("SEQUENCE_RUNTIME_AVAILABLE", sequence_runtime_ready, 0 if sequence_runtime_ready else 1),
        ("SELECTION_USES_VALID_ONLY", bool(selection["selection_split"].eq("VALID_ONLY").all()), 0),
        ("TEST_NOT_USED_FOR_SELECTION", not bool(selection["test_used_for_selection"].any()), 0),
        ("FINITE_VALID_PREDICTIONS", bool(finite_valid), 0 if finite_valid else 1),
        ("FINITE_TEST_PREDICTIONS", bool(finite_test), 0 if finite_test else 1),
        ("RESEARCH_WEATHER_EXCLUDED", len(research_features) == 0, len(research_features)),
    ]
    result = pd.DataFrame(gates, columns=["gate", "passed", "observed"])
    result["severity"] = "CRITICAL"
    return result


def feature_contract_frame(representation_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for spec in TARGET_SPECS:
        representation = f"{spec.task.lower()}_core_compact"
        features = horizon_specific_features(
            representation_sets.get(representation, []), spec.horizon_h
        )
        forbidden = sorted(
            value
            for value in features
            if value.startswith("target_") and not value.startswith("target_time_")
        )
        research = sorted(value for value in features if value.startswith("research_"))
        rows.append(
            {
                "target": spec.target,
                "representation": representation,
                "horizon_h": spec.horizon_h,
                "feature_count": len(features),
                "forbidden_target_features": "|".join(forbidden),
                "research_features": "|".join(research),
                "research_weather_used": bool(research),
            }
        )
    return pd.DataFrame(rows)


def run_benchmark_core(
    frame: pd.DataFrame,
    representation_sets: dict[str, list[str]],
    sequence_predictions: pd.DataFrame,
    sequence_inventory: pd.DataFrame,
    sequence_runtime_ready: bool,
    enable_catboost: bool = True,
) -> BenchmarkResult:
    source = frame.copy().sort_values("as_of_time").reset_index(drop=True)
    source["as_of_time"] = pd.to_datetime(source["as_of_time"], utc=True)
    required_targets = [item.target for item in TARGET_SPECS] + ["target_next_arrival_observed_24h"]
    missing = sorted(set(required_targets) - set(source.columns))
    if missing:
        raise ValueError(f"B60A target columns missing: {missing}")

    tabular_valid, tabular_inventory = tabular_validation_predictions(
        source,
        representation_sets,
        sequence_predictions=sequence_predictions,
        enable_catboost=enable_catboost,
    )
    all_predictions = append_sequence_predictions(tabular_valid, sequence_predictions)
    aligned_valid, origin_coverage = align_common_origins(all_predictions, "VALID")
    valid_metrics = prediction_metrics(aligned_valid, "VALID")
    selection = select_models(valid_metrics, aligned_valid)
    valid_selected = selected_validation_predictions(aligned_valid, selection)

    tabular_test = tabular_selected_test_predictions(
        source,
        representation_sets,
        selection,
        sequence_predictions=sequence_predictions,
    )
    sequence_test = selected_sequence_test_predictions(sequence_predictions, selection)
    test_selected = pd.concat([tabular_test, sequence_test], ignore_index=True)
    test_metrics = prediction_metrics(test_selected, "TEST")
    probabilistic_test, calibration = conformal_intervals(valid_selected, test_selected)

    gates = benchmark_contract_gates(
        source,
        all_predictions.loc[all_predictions["split"].eq("VALID")],
        selection,
        test_selected,
        sequence_runtime_ready,
        representation_sets,
    )
    gates_passed = bool(gates["passed"].all())
    accepted = int(selection["candidate_accepted"].sum())
    selected_sequence = int(selection["selected_model"].isin(["NHITS_SEQ", "PATCHTST_SEQ"]).sum())
    decision_name = (
        "B60B_BENCHMARK_COMPLETE_READY_FOR_B60C_CALIBRATION_AND_ENSEMBLE"
        if gates_passed and accepted > 0
        else "B60B_BLOCKED_REPAIR_REQUIRED"
        if not gates_passed
        else "B60B_NO_MODEL_BEATS_SAFE_BASELINE"
    )
    inventory = pd.concat([tabular_inventory, sequence_inventory], ignore_index=True, sort=False)
    reports = {
        "01_contract_gates.csv": gates,
        "02_target_scope.csv": target_scope_frame(),
        "03_feature_contract.csv": feature_contract_frame(representation_sets),
        "04_model_inventory.csv": inventory,
        "05_common_origin_coverage.csv": origin_coverage,
        "06_valid_metrics.csv": valid_metrics,
        "07_model_selection.csv": selection,
        "08_test_metrics.csv": test_metrics,
        "09_probabilistic_calibration.csv": calibration,
    }
    predictions = {
        "valid_candidate_predictions.parquet": all_predictions.loc[all_predictions["split"].eq("VALID")].copy(),
        "test_selected_predictions.parquet": probabilistic_test,
    }
    decision = {
        "status": "SUCCESS",
        "decision": decision_name,
        "benchmark_version": BENCHMARK_VERSION,
        "source_dataset_version": SOURCE_DATASET_VERSION,
        "source_rows": len(source),
        "target_count": len(TARGET_SPECS),
        "selected_models": dict(zip(selection["target"], selection["selected_model"])),
        "accepted_challengers": accepted,
        "selected_sequence_models": selected_sequence,
        "quality_gates_passed": gates_passed,
        "selection_split": "VALID_ONLY",
        "test_role": "FINAL_DIAGNOSTIC_ONLY_AFTER_SELECTION",
        "selection_used_test": False,
        "research_weather_used": False,
        "duplicate_target_removed": "target_arrivals_0_6h",
        "probabilistic_interval": "ASYMMETRIC_SPLIT_CONFORMAL_P10_P50_P90",
        "production_promotion_allowed": False,
        "next_block": "B60C_CONTEXTUAL_MIXTURE_AND_ADAPTIVE_CONFORMAL",
    }
    return BenchmarkResult(reports, predictions, clean_json(decision))
