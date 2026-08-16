from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


MODEL_VERSION = "b62b-vintage-weather-wave-shadow-v1"
DATASET_VERSION = "b62b-open-meteo-previous-runs-v1"
TARGET = "wave_period_s"
HORIZON_H = 24
EXPECTED_COVERAGE = 0.80
QUANTILES = (0.1, 0.5, 0.9)

ARCHIVE_VARIABLES = (
    "temperature_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_10m",
)

FEATURE_COLUMNS = (
    "temperature_2m",
    "pressure_msl",
    "wind_speed_10m",
    "wind_direction_sin",
    "wind_direction_cos",
    "wave_period_issue",
    "wave_period_lag6",
    "wave_period_lag24",
    "wave_period_lag168",
    "wave_period_mean24",
    "wave_period_std24",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
)


@dataclass
class VintageTask:
    feature_columns: list[str]
    models: dict[float, HistGradientBoostingRegressor]
    bias_correction: float
    conformal_expansion: float
    calibration_rows: int


def _numeric(values: Any) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("float64")


def _quantile_higher(values: np.ndarray, probability: float) -> float:
    clean = np.asarray(values, dtype="float64")
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0:
        return 0.0
    rank_probability = min(1.0, math.ceil((len(clean) + 1) * probability) / len(clean))
    try:
        return float(np.quantile(clean, rank_probability, method="higher"))
    except TypeError:  # pragma: no cover - NumPy < 1.22
        return float(np.quantile(clean, rank_probability, interpolation="higher"))


def previous_run_variable(variable: str, lead_days: int = 1) -> str:
    return f"{variable}_previous_day{int(lead_days)}"


def normalize_previous_runs_payload(
    payload: dict[str, Any],
    lead_days: int = 1,
    provider_model: str = "gfs_seamless",
) -> pd.DataFrame:
    """Normalize fixed-lead archive values without inventing availability timestamps."""
    hourly = dict(payload.get("hourly") or {})
    times = pd.to_datetime(hourly.get("time", []), errors="coerce", utc=True)
    if len(times) == 0 or times.isna().any():
        raise ValueError("Previous-runs payload has no valid hourly timestamps")
    frame = pd.DataFrame({"valid_at": times})
    for variable in ARCHIVE_VARIABLES:
        key = previous_run_variable(variable, lead_days)
        values = hourly.get(key)
        if values is None:
            frame[variable] = np.nan
        elif len(values) != len(frame):
            raise ValueError(f"Previous-runs field length mismatch: {key}")
        else:
            frame[variable] = _numeric(values)
    frame["issue_at"] = frame["valid_at"] - pd.Timedelta(days=lead_days)
    frame["lead_time_h"] = 24 * int(lead_days)
    frame["provider"] = "OPEN_METEO_PREVIOUS_RUNS"
    frame["provider_model"] = provider_model
    frame["availability_semantics"] = "FIXED_LEAD_ARCHIVE_NO_EXACT_AVAILABLE_AT"
    frame["operationally_available_at_issue"] = False
    frame["requested_latitude"] = payload.get("latitude")
    frame["requested_longitude"] = payload.get("longitude")
    return frame


def attach_wave_truth_and_lags(
    archive: pd.DataFrame,
    wave_observations: pd.DataFrame,
    origin_step_h: int = 6,
) -> pd.DataFrame:
    source = archive.copy()
    for column in ("issue_at", "valid_at"):
        source[column] = pd.to_datetime(source[column], errors="coerce", utc=True)
    wave = wave_observations[["observed_at", TARGET]].copy()
    wave["observed_at"] = pd.to_datetime(wave["observed_at"], errors="coerce", utc=True)
    wave[TARGET] = _numeric(wave[TARGET])
    wave = wave.dropna().drop_duplicates("observed_at").set_index("observed_at").sort_index()
    if wave.empty:
        raise ValueError("No real wave-period observations are available")

    target = wave[TARGET]
    rolling_mean = target.rolling(24, min_periods=12).mean()
    rolling_std = target.rolling(24, min_periods=12).std()

    def lookup(index: pd.Series, values: pd.Series) -> np.ndarray:
        return values.reindex(pd.DatetimeIndex(index)).to_numpy(dtype="float64")

    source["actual"] = lookup(source["valid_at"], target)
    source["wave_period_issue"] = lookup(source["issue_at"], target)
    source["wave_period_lag6"] = lookup(source["issue_at"] - pd.Timedelta(hours=6), target)
    source["wave_period_lag24"] = lookup(source["issue_at"] - pd.Timedelta(hours=24), target)
    source["wave_period_lag168"] = lookup(source["issue_at"] - pd.Timedelta(hours=168), target)
    source["wave_period_mean24"] = lookup(source["issue_at"], rolling_mean)
    source["wave_period_std24"] = lookup(source["issue_at"], rolling_std)

    direction = np.deg2rad(_numeric(source["wind_direction_10m"]))
    source["wind_direction_sin"] = np.sin(direction)
    source["wind_direction_cos"] = np.cos(direction)
    issue = source["issue_at"]
    source["hour_sin"] = np.sin(2.0 * np.pi * issue.dt.hour / 24.0)
    source["hour_cos"] = np.cos(2.0 * np.pi * issue.dt.hour / 24.0)
    source["dow_sin"] = np.sin(2.0 * np.pi * issue.dt.dayofweek / 7.0)
    source["dow_cos"] = np.cos(2.0 * np.pi * issue.dt.dayofweek / 7.0)
    source["year_sin"] = np.sin(2.0 * np.pi * issue.dt.dayofyear / 365.25)
    source["year_cos"] = np.cos(2.0 * np.pi * issue.dt.dayofyear / 365.25)
    source = source.loc[source["issue_at"].dt.hour.mod(origin_step_h).eq(0)].copy()
    source = source.dropna(subset=["actual", *FEATURE_COLUMNS])
    return source.sort_values("issue_at").drop_duplicates("issue_at").reset_index(drop=True)


def assign_frozen_temporal_roles(
    frame: pd.DataFrame,
    valid_days: int = 180,
    test_days: int = 180,
    purge_h: int = HORIZON_H,
) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    if frame.empty:
        raise ValueError("Cannot split an empty B62B archive")
    output = frame.sort_values("issue_at").copy()
    end = pd.Timestamp(output["valid_at"].max()).floor("D") + pd.Timedelta(days=1)
    test_start = end - pd.Timedelta(days=test_days)
    valid_end = test_start - pd.Timedelta(hours=purge_h)
    valid_start = valid_end - pd.Timedelta(days=valid_days)
    train_end = valid_start - pd.Timedelta(hours=purge_h)
    output["evaluation_role"] = "PURGED"
    output.loc[output["valid_at"].lt(train_end), "evaluation_role"] = "TRAIN_ARCHIVE"
    output.loc[
        output["issue_at"].ge(valid_start) & output["valid_at"].lt(valid_end),
        "evaluation_role",
    ] = "VALID_ARCHIVE"
    output.loc[output["issue_at"].ge(test_start), "evaluation_role"] = "TEST_CONFIRMATORY_ONCE"
    counts = output["evaluation_role"].value_counts()
    if int(counts.get("TRAIN_ARCHIVE", 0)) < 500:
        raise ValueError("B62B requires at least 500 archive TRAIN rows")
    if int(counts.get("VALID_ARCHIVE", 0)) < 120:
        raise ValueError("B62B requires at least 120 archive VALID rows")
    if int(counts.get("TEST_CONFIRMATORY_ONCE", 0)) < 120:
        raise ValueError("B62B requires at least 120 frozen TEST rows")
    return output, {
        "train_end": train_end,
        "valid_start": valid_start,
        "valid_end": valid_end,
        "test_start": test_start,
        "test_end": end,
    }


def split_train_calibration(
    frame: pd.DataFrame, calibration_days: int = 90
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    train = frame.loc[frame["evaluation_role"].eq("TRAIN_ARCHIVE")].copy()
    cutoff = pd.Timestamp(train["valid_at"].max()) - pd.Timedelta(days=calibration_days)
    model_train = train.loc[train["valid_at"].lt(cutoff)].copy()
    calibration = train.loc[train["issue_at"].ge(cutoff)].copy()
    if len(model_train) < 300 or len(calibration) < 120:
        raise ValueError(
            f"Insufficient archive fit/calibration rows: {len(model_train)}/{len(calibration)}"
        )
    return model_train, calibration, cutoff


def fit_vintage_task(
    model_train: pd.DataFrame,
    calibration: pd.DataFrame,
    max_iter: int = 160,
    seed: int = 20260811,
) -> VintageTask:
    features = list(FEATURE_COLUMNS)
    x_train = model_train[features]
    y_train = _numeric(model_train["actual"]).to_numpy(dtype="float64")
    models: dict[float, HistGradientBoostingRegressor] = {}
    for quantile in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=quantile,
            learning_rate=0.05,
            max_iter=max_iter,
            max_leaf_nodes=24,
            min_samples_leaf=24,
            l2_regularization=1.5,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=20,
            random_state=seed + int(100 * quantile),
        )
        model.fit(x_train, y_train)
        models[quantile] = model
    x_cal = calibration[features]
    actual = _numeric(calibration["actual"]).to_numpy(dtype="float64")
    raw10 = models[0.1].predict(x_cal)
    raw50 = models[0.5].predict(x_cal)
    raw90 = models[0.9].predict(x_cal)
    bias = float(np.nanmedian(actual - raw50))
    nonconformity = np.maximum(raw10 + bias - actual, actual - raw90 - bias)
    expansion = max(0.0, _quantile_higher(nonconformity, EXPECTED_COVERAGE))
    return VintageTask(features, models, bias, expansion, len(calibration))


def predict_vintage_task(task: VintageTask, frame: pd.DataFrame, role: str) -> pd.DataFrame:
    source = frame.copy()
    x = source[task.feature_columns]
    p10 = task.models[0.1].predict(x) + task.bias_correction - task.conformal_expansion
    p50 = task.models[0.5].predict(x) + task.bias_correction
    p90 = task.models[0.9].predict(x) + task.bias_correction + task.conformal_expansion
    low = np.minimum.reduce([p10, p50, p90])
    high = np.maximum.reduce([p10, p50, p90])
    return pd.DataFrame(
        {
            "evaluation_role": role,
            "issue_at": source["issue_at"].to_numpy(),
            "valid_at": source["valid_at"].to_numpy(),
            "horizon_h": HORIZON_H,
            "variable": TARGET,
            "model": "VINTAGE_WEATHER_TO_WAVE_HGB_CONFORMAL",
            "actual": _numeric(source["actual"]).to_numpy(),
            "p10": np.clip(low, 0.5, 40.0),
            "p50": np.clip(p50, 0.5, 40.0),
            "p90": np.clip(high, 0.5, 40.0),
        }
    )


def fit_seasonal_interval(calibration: pd.DataFrame) -> tuple[float, float]:
    residual = _numeric(calibration["actual"] - calibration["wave_period_lag168"])
    return float(residual.quantile(0.1)), float(residual.quantile(0.9))


def seasonal_predictions(
    frame: pd.DataFrame, role: str, residual_quantiles: tuple[float, float]
) -> pd.DataFrame:
    p50 = _numeric(frame["wave_period_lag168"])
    return pd.DataFrame(
        {
            "evaluation_role": role,
            "issue_at": frame["issue_at"].to_numpy(),
            "valid_at": frame["valid_at"].to_numpy(),
            "horizon_h": HORIZON_H,
            "variable": TARGET,
            "model": "B62_SEASONAL_NAIVE_168H",
            "actual": _numeric(frame["actual"]).to_numpy(),
            "p10": np.clip(p50 + residual_quantiles[0], 0.5, 40.0),
            "p50": np.clip(p50, 0.5, 40.0),
            "p90": np.clip(p50 + residual_quantiles[1], 0.5, 40.0),
        }
    )


def predict_frozen_b62a(task: Any, features: pd.DataFrame, frame: pd.DataFrame, role: str) -> pd.DataFrame:
    x = features[task.feature_columns]
    p10 = task.models[0.1].predict(x) + task.bias_correction - task.conformal_expansion
    p50 = task.models[0.5].predict(x) + task.bias_correction
    p90 = task.models[0.9].predict(x) + task.bias_correction + task.conformal_expansion
    return pd.DataFrame(
        {
            "evaluation_role": role,
            "issue_at": frame["issue_at"].to_numpy(),
            "valid_at": frame["valid_at"].to_numpy(),
            "horizon_h": HORIZON_H,
            "variable": TARGET,
            "model": "B62A_AUGMENTED_QUANTILE_HGB_CONFORMAL_FROZEN",
            "actual": _numeric(frame["actual"]).to_numpy(),
            "p10": np.clip(np.minimum.reduce([p10, p50, p90]), 0.5, 40.0),
            "p50": np.clip(p50, 0.5, 40.0),
            "p90": np.clip(np.maximum.reduce([p10, p50, p90]), 0.5, 40.0),
        }
    )


def forecast_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (role, model), group in predictions.groupby(["evaluation_role", "model"], sort=True):
        actual = _numeric(group["actual"]).to_numpy()
        p10 = _numeric(group["p10"]).to_numpy()
        p50 = _numeric(group["p50"]).to_numpy()
        p90 = _numeric(group["p90"]).to_numpy()
        finite = np.isfinite(np.column_stack([actual, p10, p50, p90])).all(axis=1)
        actual, p10, p50, p90 = actual[finite], p10[finite], p50[finite], p90[finite]
        error = p50 - actual
        covered = (actual >= p10) & (actual <= p90)
        rows.append(
            {
                "evaluation_role": role,
                "model": model,
                "rows": len(actual),
                "origins": int(group.loc[finite, "issue_at"].nunique()),
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(np.square(error)))),
                "BIAS": float(np.mean(error)),
                "P10_P90_COVERAGE": float(np.mean(covered)),
                "MEAN_INTERVAL_WIDTH": float(np.mean(p90 - p10)),
                "quantile_crossings": int(np.sum((p10 > p50) | (p50 > p90))),
            }
        )
    return pd.DataFrame(rows)


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    reference_model: str,
    candidate_model: str,
    iterations: int = 500,
    seed: int = 20260811,
) -> dict[str, float | int | str]:
    source = predictions.loc[predictions["model"].isin([reference_model, candidate_model])]
    wide = source.pivot_table(
        index="issue_at", columns="model", values=["actual", "p50"], aggfunc="first"
    )
    required = [("actual", reference_model), ("p50", reference_model), ("p50", candidate_model)]
    if any(column not in wide for column in required):
        raise ValueError("Bootstrap models do not share aligned origins")
    actual = wide[("actual", reference_model)]
    reference_error = (wide[("p50", reference_model)] - actual).abs()
    candidate_error = (wide[("p50", candidate_model)] - actual).abs()
    paired = pd.DataFrame(
        {"reference": reference_error, "candidate": candidate_error}, index=wide.index
    ).dropna()
    cluster_time = pd.DatetimeIndex(paired.index)
    if cluster_time.tz is not None:
        cluster_time = cluster_time.tz_convert("UTC").tz_localize(None)
    paired["cluster"] = cluster_time.to_period("W-SUN").astype(str)
    clusters = list(paired["cluster"].unique())
    if len(clusters) < 4:
        return {
            "reference_model": reference_model,
            "candidate_model": candidate_model,
            "origins": len(paired),
            "clusters": len(clusters),
            "gain_median_pct": float("nan"),
            "gain_ci_low_pct": float("nan"),
            "gain_ci_high_pct": float("nan"),
            "probability_gain_positive": float("nan"),
        }
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        draw = pd.concat([paired.loc[paired["cluster"].eq(cluster)] for cluster in sampled])
        ref = float(draw["reference"].mean())
        cand = float(draw["candidate"].mean())
        gains.append(100.0 * (ref - cand) / max(ref, 1e-12))
    return {
        "reference_model": reference_model,
        "candidate_model": candidate_model,
        "origins": len(paired),
        "clusters": len(clusters),
        "gain_median_pct": float(np.median(gains)),
        "gain_ci_low_pct": float(np.quantile(gains, 0.025)),
        "gain_ci_high_pct": float(np.quantile(gains, 0.975)),
        "probability_gain_positive": float(np.mean(np.asarray(gains) > 0.0)),
    }


def select_on_valid(
    predictions: pd.DataFrame,
    bootstrap_iterations: int = 500,
    min_gain_pct: float = 5.0,
    seed: int = 20260811,
    reference: str = "B62A_AUGMENTED_QUANTILE_HGB_CONFORMAL_FROZEN",
    candidate: str = "VINTAGE_WEATHER_TO_WAVE_HGB_CONFORMAL",
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = forecast_metrics(predictions)
    valid = metrics.loc[metrics["evaluation_role"].eq("VALID_ARCHIVE")].copy()
    candidate_match = valid.loc[valid["model"].eq(candidate)]
    reference_match = valid.loc[valid["model"].eq(reference)]
    if candidate_match.empty or reference_match.empty:
        raise ValueError("B62B VALID is missing the frozen incumbent or vintage challenger")
    boot = paired_block_bootstrap(predictions, reference, candidate, bootstrap_iterations, seed)
    candidate_row = candidate_match.iloc[0]
    reference_row = reference_match.iloc[0]
    point_gain = 100.0 * (float(reference_row.MAE) - float(candidate_row.MAE)) / max(
        float(reference_row.MAE), 1e-12
    )
    coverage = float(candidate_row.P10_P90_COVERAGE)
    accepted = bool(
        int(candidate_row.rows) >= 120
        and point_gain >= min_gain_pct
        and float(boot["gain_ci_low_pct"]) > 0.0
        and 0.70 <= coverage <= 0.90
        and int(candidate_row.quantile_crossings) == 0
    )
    decision = {
        "selected_model": candidate if accepted else reference,
        "best_candidate": candidate,
        "reference_model": reference,
        "candidate_point_gain_pct": point_gain,
        "candidate_coverage": coverage,
        "valid_accepted": accepted,
        "selection_role": "VALID_ARCHIVE_ONLY",
        "test_used_for_selection": False,
        **boot,
    }
    return decision, metrics


def production_contract(
    archive_test_predictions: pd.DataFrame,
    fresh_predictions: pd.DataFrame,
    selected_model: str,
    reference_model: str,
    fresh_reference_model: str,
    min_fresh_origins: int,
    min_fresh_days: int,
    bootstrap_iterations: int,
    min_gain_pct: float,
    seed: int,
) -> dict[str, Any]:
    test_boot = paired_block_bootstrap(
        archive_test_predictions,
        reference_model,
        selected_model,
        bootstrap_iterations,
        seed + 1,
    )
    test_metrics = forecast_metrics(archive_test_predictions)
    selected_test = test_metrics.loc[
        test_metrics["model"].eq(selected_model)
        & test_metrics["evaluation_role"].eq("TEST_CONFIRMATORY_ONCE")
    ]
    archive_confirmed = False
    if not selected_test.empty:
        row = selected_test.iloc[0]
        archive_confirmed = bool(
            int(row.rows) >= 120
            and float(test_boot["gain_ci_low_pct"]) > 0.0
            and float(test_boot["gain_median_pct"]) >= min_gain_pct
            and 0.70 <= float(row.P10_P90_COVERAGE) <= 0.90
            and int(row.quantile_crossings) == 0
        )

    fresh_origins = int(fresh_predictions["issue_at"].nunique()) if not fresh_predictions.empty else 0
    fresh_span_days = 0.0
    fresh_boot: dict[str, Any] = {}
    fresh_confirmed = False
    if fresh_origins:
        span = pd.Timestamp(fresh_predictions["issue_at"].max()) - pd.Timestamp(
            fresh_predictions["issue_at"].min()
        )
        fresh_span_days = float(span.total_seconds() / 86_400.0)
    if fresh_origins >= min_fresh_origins and fresh_span_days >= min_fresh_days:
        fresh_boot = paired_block_bootstrap(
            fresh_predictions,
            fresh_reference_model,
            selected_model,
            bootstrap_iterations,
            seed + 2,
        )
        fresh_metrics = forecast_metrics(fresh_predictions)
        selected_fresh = fresh_metrics.loc[fresh_metrics["model"].eq(selected_model)]
        if not selected_fresh.empty:
            row = selected_fresh.iloc[0]
            fresh_confirmed = bool(
                float(fresh_boot["gain_ci_low_pct"]) > 0.0
                and 0.70 <= float(row.P10_P90_COVERAGE) <= 0.90
            )
    return {
        "archive_confirmed": archive_confirmed,
        "fresh_confirmed": fresh_confirmed,
        "fresh_origins": fresh_origins,
        "fresh_span_days": fresh_span_days,
        "minimum_fresh_origins": min_fresh_origins,
        "minimum_fresh_days": min_fresh_days,
        "test_bootstrap": test_boot,
        "fresh_bootstrap": fresh_boot,
        "fresh_reference_model": fresh_reference_model,
        "limited_production_pilot_allowed": bool(archive_confirmed and fresh_confirmed),
        "automatic_action_allowed": False,
    }
