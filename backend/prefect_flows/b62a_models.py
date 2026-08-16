from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from prefect_flows.b62a_core import (
    CHALLENGER_TARGETS,
    EXPECTED_COVERAGE,
    HORIZONS_H,
    feature_columns,
    target_column,
    target_role_column,
)
from prefect_flows.b62_core import enforce_forecast_constraints, forecast_metrics


MODEL_NAME = "AUGMENTED_QUANTILE_HGB_CONFORMAL"
QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class FittedTask:
    variable: str
    horizon_h: int
    feature_columns: list[str]
    models: dict[float, HistGradientBoostingRegressor]
    bias_correction: float
    conformal_expansion: float
    calibration_rows: int


def _quantile_higher(values: np.ndarray, probability: float) -> float:
    clean = np.asarray(values, dtype="float64")
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return 0.0
    rank_probability = min(1.0, math.ceil((len(clean) + 1) * probability) / len(clean))
    try:
        return float(np.quantile(clean, rank_probability, method="higher"))
    except TypeError:  # pragma: no cover - NumPy < 1.22 compatibility
        return float(np.quantile(clean, rank_probability, interpolation="higher"))


def _training_rows(frame: pd.DataFrame, variable: str, horizon_h: int) -> pd.DataFrame:
    target = target_column(variable, horizon_h)
    role = target_role_column(horizon_h)
    source = frame.loc[frame[role].eq("TRAIN")].copy()
    source[target] = pd.to_numeric(source[target], errors="coerce")
    return source.loc[source[target].notna()].copy()


def fit_tail_challenger(
    real_model_train: pd.DataFrame,
    synthetic_train: pd.DataFrame,
    calibration_real: pd.DataFrame,
    max_iter: int = 120,
    random_seed: int = 20260811,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> dict[tuple[str, int], FittedTask]:
    features = feature_columns(real_model_train)
    if not features:
        raise ValueError("B62A has no engineered feature columns")
    combined = pd.concat([real_model_train, synthetic_train], ignore_index=True)
    tasks: dict[tuple[str, int], FittedTask] = {}
    total_tasks = len(CHALLENGER_TARGETS) * len(HORIZONS_H)
    completed = 0
    for variable in CHALLENGER_TARGETS:
        for horizon_h in HORIZONS_H:
            target = target_column(variable, horizon_h)
            train = _training_rows(combined, variable, horizon_h)
            calibration = _training_rows(calibration_real, variable, horizon_h)
            if len(train) < 5_000 or len(calibration) < 500:
                raise ValueError(
                    f"Insufficient rows for {variable} h{horizon_h}: "
                    f"train={len(train)} calibration={len(calibration)}"
                )
            x_train = train[features]
            y_train = train[target].to_numpy(dtype="float64")
            weights = pd.to_numeric(train["sample_weight"], errors="coerce").fillna(1.0)
            models: dict[float, HistGradientBoostingRegressor] = {}
            for quantile in QUANTILES:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=quantile,
                    learning_rate=0.06,
                    max_iter=max_iter,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.10,
                    n_iter_no_change=15,
                    random_state=random_seed + horizon_h + int(100 * quantile),
                )
                model.fit(x_train, y_train, sample_weight=weights)
                models[quantile] = model

            x_calibration = calibration[features]
            actual = calibration[target].to_numpy(dtype="float64")
            raw10 = models[0.1].predict(x_calibration)
            raw50 = models[0.5].predict(x_calibration)
            raw90 = models[0.9].predict(x_calibration)
            bias = float(np.nanmedian(actual - raw50))
            shifted10 = raw10 + bias
            shifted90 = raw90 + bias
            nonconformity = np.maximum(shifted10 - actual, actual - shifted90)
            expansion = max(0.0, _quantile_higher(nonconformity, EXPECTED_COVERAGE))
            tasks[(variable, horizon_h)] = FittedTask(
                variable=variable,
                horizon_h=horizon_h,
                feature_columns=features,
                models=models,
                bias_correction=bias,
                conformal_expansion=expansion,
                calibration_rows=len(calibration),
            )
            completed += 1
            if progress is not None:
                progress(completed, total_tasks, variable, horizon_h)
    return tasks


def predict_origins(
    tasks: dict[tuple[str, int], FittedTask],
    supervised: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
    role: str,
) -> pd.DataFrame:
    origin_index = pd.DatetimeIndex(pd.to_datetime(list(origins), errors="coerce", utc=True))
    source = supervised.loc[supervised["issue_at"].isin(origin_index)].copy()
    rows: list[dict[str, Any]] = []
    for (variable, horizon_h), task in tasks.items():
        target = target_column(variable, horizon_h)
        target_role = target_role_column(horizon_h)
        eligible = source.loc[source[target_role].eq(source["evaluation_role"])].copy()
        eligible[target] = pd.to_numeric(eligible[target], errors="coerce")
        eligible = eligible.loc[eligible[target].notna()]
        if eligible.empty:
            continue
        x = eligible[task.feature_columns]
        p10 = task.models[0.1].predict(x) + task.bias_correction - task.conformal_expansion
        p50 = task.models[0.5].predict(x) + task.bias_correction
        p90 = task.models[0.9].predict(x) + task.bias_correction + task.conformal_expansion
        for index, item in enumerate(eligible.itertuples(index=False)):
            issue_at = pd.Timestamp(item.issue_at)
            rows.append(
                {
                    "family": "WAVE",
                    "evaluation_role": role,
                    "issue_at": issue_at,
                    "valid_at": issue_at + pd.Timedelta(hours=horizon_h),
                    "horizon_h": horizon_h,
                    "variable": variable,
                    "model": MODEL_NAME,
                    "actual": float(getattr(item, target)),
                    "p10": float(p10[index]),
                    "p50": float(p50[index]),
                    "p90": float(p90[index]),
                }
            )
    return enforce_forecast_constraints(pd.DataFrame(rows))


def predict_stress(
    tasks: dict[tuple[str, int], FittedTask],
    stress: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variable, horizon_h), task in tasks.items():
        x = stress[task.feature_columns]
        p10 = task.models[0.1].predict(x) + task.bias_correction - task.conformal_expansion
        p50 = task.models[0.5].predict(x) + task.bias_correction
        p90 = task.models[0.9].predict(x) + task.bias_correction + task.conformal_expansion
        for index, scenario in enumerate(stress["stress_scenario_id"]):
            rows.append(
                {
                    "stress_scenario_id": scenario,
                    "stress_role": "SYNTHETIC_STRESS_NO_PERFORMANCE_CLAIM",
                    "family": "WAVE",
                    "variable": variable,
                    "horizon_h": horizon_h,
                    "model": MODEL_NAME,
                    "p10": float(p10[index]),
                    "p50": float(p50[index]),
                    "p90": float(p90[index]),
                }
            )
    output = pd.DataFrame(rows)
    output = enforce_forecast_constraints(output)
    return output


def select_against_frozen_b62(
    frozen_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
) -> pd.DataFrame:
    frozen_metrics = forecast_metrics(frozen_predictions)
    challenger_metrics = forecast_metrics(challenger_predictions)
    rows = []
    for variable in CHALLENGER_TARGETS:
        for horizon_h in HORIZONS_H:
            frozen = frozen_metrics.loc[
                frozen_metrics["variable"].eq(variable)
                & frozen_metrics["horizon_h"].eq(horizon_h)
            ]
            challenger = challenger_metrics.loc[
                challenger_metrics["variable"].eq(variable)
                & challenger_metrics["horizon_h"].eq(horizon_h)
            ]
            if frozen.empty or challenger.empty:
                continue
            reference = frozen.iloc[0]
            candidate = challenger.iloc[0]
            gain = 100.0 * (float(reference.MAE) - float(candidate.MAE)) / max(
                float(reference.MAE), 1e-12
            )
            reference_gap = abs(float(reference.P10_P90_COVERAGE) - EXPECTED_COVERAGE)
            candidate_gap = abs(float(candidate.P10_P90_COVERAGE) - EXPECTED_COVERAGE)
            accepted = bool(
                int(candidate.rows) >= 18
                and gain >= 2.0
                and candidate_gap <= min(0.15, reference_gap + 1e-12)
                and int(candidate.quantile_crossings) == 0
            )
            rows.append(
                {
                    "family": "WAVE",
                    "variable": variable,
                    "horizon_h": horizon_h,
                    "valid_rows": int(candidate.rows),
                    "b62_model": str(reference.model),
                    "b62_mae": float(reference.MAE),
                    "b62_coverage": float(reference.P10_P90_COVERAGE),
                    "challenger_mae": float(candidate.MAE),
                    "challenger_coverage": float(candidate.P10_P90_COVERAGE),
                    "challenger_bias": float(candidate.BIAS),
                    "challenger_gain_pct": float(gain),
                    "selected_model": MODEL_NAME if accepted else str(reference.model),
                    "challenger_accepted": accepted,
                    "selection_role": "VALID_REAL_ONLY",
                    "test_used_for_selection": False,
                    "stress_used_for_selection": False,
                }
            )
    return pd.DataFrame(rows)


def apply_wave_selection(
    frozen_predictions: pd.DataFrame,
    challenger_predictions: pd.DataFrame,
    selection: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    output = frozen_predictions.copy()
    for row in selection.itertuples(index=False):
        if not bool(row.challenger_accepted):
            continue
        mask = output["variable"].eq(row.variable) & output["horizon_h"].eq(row.horizon_h)
        output = output.loc[~mask]
        challenger = challenger_predictions.loc[
            challenger_predictions["variable"].eq(row.variable)
            & challenger_predictions["horizon_h"].eq(row.horizon_h)
        ]
        output = pd.concat([output, challenger], ignore_index=True)
    output["evaluation_role"] = role
    return output.sort_values(["family", "variable", "horizon_h", "issue_at"]).reset_index(drop=True)


def task_calibration_report(tasks: dict[tuple[str, int], FittedTask]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "variable": variable,
                "horizon_h": horizon_h,
                "calibration_rows": task.calibration_rows,
                "bias_correction": task.bias_correction,
                "conformal_expansion": task.conformal_expansion,
                "calibration_role": "TRAIN_CALIBRATION_REAL_ONLY",
            }
            for (variable, horizon_h), task in sorted(tasks.items())
        ]
    )
