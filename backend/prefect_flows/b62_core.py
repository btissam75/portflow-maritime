from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable

import numpy as np
import pandas as pd


MODEL_VERSION = "b62-weather-wave-vessel-autogluon-v1"
DATASET_VERSION = "b62-metocean-hourly-research-v1"
HORIZONS_H = (6, 12, 24, 48, 72)
QUANTILES = (0.1, 0.5, 0.9)
PURGE_HOURS = max(HORIZONS_H)
MIN_TRAIN_HOURS = 24 * 365 * 2
MIN_VALID_ROWS = 18
MIN_MODEL_GAIN_PCT = 2.0
EXPECTED_COVERAGE = 0.80

WEATHER_TARGETS = (
    "wind_speed_ms",
    "pressure_hpa",
    "visibility_m",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_direction_sin",
    "wind_direction_cos",
)
WAVE_TARGETS = (
    "wave_height_m",
    "wave_period_s",
    "wave_direction_sin",
    "wave_direction_cos",
)
WEATHER_COVARIATES = (
    "wind_speed_ms",
    "pressure_hpa",
    "visibility_m",
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "cloud_cover",
    "wind_direction_sin",
    "wind_direction_cos",
)
CALENDAR_COVARIATES = (
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "year_sin",
    "year_cos",
)

PHYSICAL_BOUNDS = {
    "wave_height_m": (0.0, 30.0),
    "wave_period_s": (0.5, 40.0),
    "wind_speed_ms": (0.0, 100.0),
    "pressure_hpa": (800.0, 1_100.0),
    "visibility_m": (0.0, 200_000.0),
    "temperature_2m": (-80.0, 65.0),
    "relative_humidity_2m": (0.0, 100.0),
    "precipitation": (0.0, 1_000.0),
    "cloud_cover": (0.0, 100.0),
}


@dataclass(frozen=True)
class TemporalBoundaries:
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def _circular_components(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    radians = np.deg2rad(pd.to_numeric(values, errors="coerce"))
    return np.sin(radians), np.cos(radians)


def circular_degrees(sine: np.ndarray | pd.Series, cosine: np.ndarray | pd.Series) -> np.ndarray:
    angle = np.rad2deg(np.arctan2(np.asarray(sine), np.asarray(cosine)))
    return np.mod(angle + 360.0, 360.0)


def add_calendar_features(frame: pd.DataFrame, time_column: str = "observed_at") -> pd.DataFrame:
    output = frame.copy()
    times = pd.to_datetime(output[time_column], errors="coerce", utc=True)
    output["hour_sin"] = np.sin(2.0 * np.pi * times.dt.hour / 24.0)
    output["hour_cos"] = np.cos(2.0 * np.pi * times.dt.hour / 24.0)
    output["dow_sin"] = np.sin(2.0 * np.pi * times.dt.dayofweek / 7.0)
    output["dow_cos"] = np.cos(2.0 * np.pi * times.dt.dayofweek / 7.0)
    output["year_sin"] = np.sin(2.0 * np.pi * times.dt.dayofyear / 365.25)
    output["year_cos"] = np.cos(2.0 * np.pi * times.dt.dayofyear / 365.25)
    return output


def prepare_hourly_frame(
    waves: pd.DataFrame,
    atmosphere: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if waves.empty:
        raise ValueError("The canonical wave source is empty")
    if atmosphere.empty:
        raise ValueError("The external atmosphere source is empty")

    wave = waves.copy()
    weather = atmosphere.copy()
    wave["observed_at"] = pd.to_datetime(wave["observed_at"], errors="coerce", utc=True)
    weather["observed_at"] = pd.to_datetime(
        weather["observed_at"], errors="coerce", utc=True
    )
    if wave["observed_at"].isna().any() or weather["observed_at"].isna().any():
        raise ValueError("Metocean sources contain invalid timestamps")
    if wave["observed_at"].duplicated().any():
        raise ValueError("Wave source is not unique by observed_at")
    if weather["observed_at"].duplicated().any():
        raise ValueError("Atmosphere source is not unique by observed_at")

    first = max(wave["observed_at"].min(), weather["observed_at"].min())
    last = min(wave["observed_at"].max(), weather["observed_at"].max())
    if last <= first:
        raise ValueError("Wave and atmosphere sources do not overlap")
    grid = pd.DataFrame({"observed_at": pd.date_range(first, last, freq="h", tz="UTC")})
    output = grid.merge(wave, on="observed_at", how="left", validate="one_to_one")
    output = output.merge(
        weather, on="observed_at", how="left", validate="one_to_one", suffixes=("", "_ext")
    )

    raw_variables = {
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "wind_speed_ms",
        "wind_direction_deg",
        "pressure_hpa",
        "visibility_m",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
    }
    for column in raw_variables:
        output[column] = _numeric(output, column)

    invalid_rows: list[dict[str, Any]] = []
    for column, (lower, upper) in PHYSICAL_BOUNDS.items():
        invalid = output[column].notna() & ~output[column].between(lower, upper)
        invalid_rows.append(
            {
                "variable": column,
                "lower_inclusive": lower,
                "upper_inclusive": upper,
                "outside_bound_rows": int(invalid.sum()),
            }
        )
        output.loc[invalid, column] = np.nan

    for column in ("wave_direction_deg", "wind_direction_deg"):
        invalid = output[column].notna() & ~output[column].between(0.0, 360.0, inclusive="left")
        invalid_rows.append(
            {
                "variable": column,
                "lower_inclusive": 0.0,
                "upper_inclusive": 360.0,
                "outside_bound_rows": int(invalid.sum()),
            }
        )
        output.loc[invalid, column] = np.nan

    output["wave_direction_sin"], output["wave_direction_cos"] = _circular_components(
        output["wave_direction_deg"]
    )
    output["wind_direction_sin"], output["wind_direction_cos"] = _circular_components(
        output["wind_direction_deg"]
    )
    output = add_calendar_features(output)
    output["source_wave_present"] = output[list(WAVE_TARGETS[:2])].notna().all(axis=1)
    output["source_weather_present"] = output[
        ["wind_speed_ms", "pressure_hpa", "visibility_m"]
    ].notna().all(axis=1)
    output["dataset_version"] = DATASET_VERSION
    output = output.sort_values("observed_at").reset_index(drop=True)
    return output, pd.DataFrame(invalid_rows)


def assign_temporal_roles(
    frame: pd.DataFrame,
    validation_days: int = 365,
    test_days: int = 365,
) -> tuple[pd.DataFrame, TemporalBoundaries]:
    if validation_days < 90 or test_days < 90:
        raise ValueError("VALID and TEST must each contain at least 90 days")
    output = frame.copy().sort_values("observed_at").reset_index(drop=True)
    times = pd.to_datetime(output["observed_at"], utc=True)
    test_end = times.max()
    test_start = test_end - pd.Timedelta(days=test_days) + pd.Timedelta(hours=1)
    valid_end = test_start - pd.Timedelta(hours=PURGE_HOURS + 1)
    valid_start = valid_end - pd.Timedelta(days=validation_days) + pd.Timedelta(hours=1)
    train_end = valid_start - pd.Timedelta(hours=PURGE_HOURS + 1)
    if (train_end - times.min()).total_seconds() / 3600 < MIN_TRAIN_HOURS:
        raise ValueError("Less than two years remain in TRAIN after temporal purging")

    role = np.full(len(output), "PURGED", dtype=object)
    role[times.le(train_end)] = "TRAIN"
    role[times.between(valid_start, valid_end)] = "VALID"
    role[times.ge(test_start)] = "TEST_DIAGNOSTIC_ONLY"
    output["evaluation_role"] = role
    boundaries = TemporalBoundaries(train_end, valid_start, valid_end, test_start, test_end)
    return output, boundaries


def temporal_split_report(frame: pd.DataFrame) -> pd.DataFrame:
    report = (
        frame.groupby("evaluation_role", sort=False)
        .agg(
            rows=("observed_at", "size"),
            first_at=("observed_at", "min"),
            last_at=("observed_at", "max"),
            wave_complete_pct=("source_wave_present", lambda value: 100.0 * value.mean()),
            weather_complete_pct=("source_weather_present", lambda value: 100.0 * value.mean()),
        )
        .reset_index()
    )
    order = {"TRAIN": 0, "PURGED": 1, "VALID": 2, "TEST_DIAGNOSTIC_ONLY": 3}
    report["_order"] = report["evaluation_role"].map(order)
    return report.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def source_coverage_report(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family, variables in (("WEATHER", WEATHER_TARGETS), ("WAVE", WAVE_TARGETS)):
        for variable in variables:
            available = int(frame[variable].notna().sum())
            rows.append(
                {
                    "family": family,
                    "variable": variable,
                    "rows": len(frame),
                    "available_rows": available,
                    "coverage_pct": 100.0 * available / max(len(frame), 1),
                    "distinct_values": int(frame[variable].nunique(dropna=True)),
                    "target_imputed": False,
                }
            )
    return pd.DataFrame(rows)


def rolling_origins(
    frame: pd.DataFrame,
    role: str,
    step_h: int,
    minimum_context_h: int = 24 * 180,
) -> list[pd.Timestamp]:
    if step_h < 24:
        raise ValueError("Rolling-origin step must be at least 24 hours")
    source = frame.loc[frame["evaluation_role"].eq(role), "observed_at"]
    if source.empty:
        return []
    first = source.min()
    last = source.max() - pd.Timedelta(hours=max(HORIZONS_H))
    dataset_start = frame["observed_at"].min()
    first = max(first, dataset_start + pd.Timedelta(hours=minimum_context_h))
    if last < first:
        return []
    return list(pd.date_range(first, last, freq=f"{step_h}h"))


def seasonal_residual_quantiles(
    frame: pd.DataFrame,
    targets: Iterable[str],
    lag_h: int = 168,
) -> dict[str, tuple[float, float]]:
    train = frame.loc[frame["evaluation_role"].eq("TRAIN")]
    values: dict[str, tuple[float, float]] = {}
    for target in targets:
        residual = train[target] - train[target].shift(lag_h)
        residual = residual.replace([np.inf, -np.inf], np.nan).dropna()
        if residual.empty:
            values[target] = (0.0, 0.0)
        else:
            values[target] = (
                float(residual.quantile(0.1)),
                float(residual.quantile(0.9)),
            )
    return values


def seasonal_predictions(
    frame: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
    targets: Iterable[str],
    family: str,
    lag_h: int = 168,
) -> pd.DataFrame:
    lookup = frame.set_index("observed_at")
    residuals = seasonal_residual_quantiles(frame, targets, lag_h)
    rows = []
    for origin in origins:
        role = str(lookup.at[origin, "evaluation_role"])
        for horizon_h in HORIZONS_H:
            valid_at = origin + pd.Timedelta(hours=horizon_h)
            baseline_at = valid_at - pd.Timedelta(hours=lag_h)
            if valid_at not in lookup.index or baseline_at not in lookup.index:
                continue
            for target in targets:
                actual = lookup.at[valid_at, target]
                baseline = lookup.at[baseline_at, target]
                if not np.isfinite(actual) or not np.isfinite(baseline):
                    continue
                lower_residual, upper_residual = residuals[target]
                p50 = float(baseline)
                rows.append(
                    {
                        "family": family,
                        "evaluation_role": role,
                        "issue_at": origin,
                        "valid_at": valid_at,
                        "horizon_h": horizon_h,
                        "variable": target,
                        "model": "SEASONAL_NAIVE_168H",
                        "actual": float(actual),
                        "p10": float(p50 + lower_residual),
                        "p50": p50,
                        "p90": float(p50 + upper_residual),
                    }
                )
    return enforce_forecast_constraints(pd.DataFrame(rows))


def enforce_forecast_constraints(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    output = predictions.copy()
    quantiles = output[["p10", "p50", "p90"]].to_numpy(dtype="float64")
    quantiles.sort(axis=1)
    output[["p10", "p50", "p90"]] = quantiles
    nonnegative = ~output["variable"].str.contains(
        r"direction_(?:sin|cos)", regex=True
    )
    bounded_percent = output["variable"].isin(["relative_humidity_2m", "cloud_cover"])
    for column in ("p10", "p50", "p90"):
        output.loc[nonnegative, column] = output.loc[nonnegative, column].clip(lower=0.0)
        output.loc[bounded_percent, column] = output.loc[bounded_percent, column].clip(0.0, 100.0)
        output.loc[output["variable"].str.endswith(("_sin", "_cos")), column] = output.loc[
            output["variable"].str.endswith(("_sin", "_cos")), column
        ].clip(-1.0, 1.0)
    return output


def forecast_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    keys = ["family", "evaluation_role", "variable", "horizon_h", "model"]
    for key, group in predictions.groupby(keys, sort=True):
        actual = group["actual"].to_numpy(dtype="float64")
        p10 = group["p10"].to_numpy(dtype="float64")
        p50 = group["p50"].to_numpy(dtype="float64")
        p90 = group["p90"].to_numpy(dtype="float64")
        error = p50 - actual
        coverage = np.mean((actual >= p10) & (actual <= p90))
        width = np.mean(p90 - p10)
        pinball10 = np.mean(np.maximum(0.1 * (actual - p10), -0.9 * (actual - p10)))
        pinball90 = np.mean(np.maximum(0.9 * (actual - p90), -0.1 * (actual - p90)))
        rows.append(
            {
                **dict(zip(keys, key)),
                "rows": len(group),
                "MAE": float(np.mean(np.abs(error))),
                "RMSE": float(np.sqrt(np.mean(np.square(error)))),
                "BIAS": float(np.mean(error)),
                "P10_P90_COVERAGE": float(coverage),
                "MEAN_INTERVAL_WIDTH": float(width),
                "WQL_PROXY": float(pinball10 + pinball90),
                "quantile_crossings": int(((p10 > p50) | (p50 > p90)).sum()),
            }
        )
    return pd.DataFrame(rows)


def select_valid_models(metrics: pd.DataFrame) -> pd.DataFrame:
    valid = metrics.loc[metrics["evaluation_role"].eq("VALID")].copy()
    rows = []
    for (family, variable, horizon_h), group in valid.groupby(
        ["family", "variable", "horizon_h"], sort=True
    ):
        baseline_rows = group.loc[group["model"].eq("SEASONAL_NAIVE_168H")]
        challenger_rows = group.loc[group["model"].eq("CHRONOS2_SMALL")]
        if baseline_rows.empty or challenger_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        challenger = challenger_rows.iloc[0]
        gain = 100.0 * (baseline.MAE - challenger.MAE) / max(float(baseline.MAE), 1e-12)
        coverage_gap = abs(float(challenger.P10_P90_COVERAGE) - EXPECTED_COVERAGE)
        accepted = bool(
            int(challenger.rows) >= MIN_VALID_ROWS
            and gain >= MIN_MODEL_GAIN_PCT
            and coverage_gap <= 0.15
            and int(challenger.quantile_crossings) == 0
        )
        selected = "CHRONOS2_SMALL" if accepted else "SEASONAL_NAIVE_168H"
        rows.append(
            {
                "family": family,
                "variable": variable,
                "horizon_h": int(horizon_h),
                "valid_rows": int(challenger.rows),
                "baseline_mae": float(baseline.MAE),
                "chronos_mae": float(challenger.MAE),
                "chronos_gain_pct": float(gain),
                "chronos_coverage": float(challenger.P10_P90_COVERAGE),
                "selected_model": selected,
                "chronos_accepted": accepted,
                "selection_role": "VALID_ONLY",
                "test_used_for_selection": False,
            }
        )
    return pd.DataFrame(rows)


def apply_frozen_selection(
    predictions: pd.DataFrame,
    selection: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    source = predictions.loc[predictions["evaluation_role"].eq(role)].copy()
    if source.empty:
        return source
    keys = ["family", "variable", "horizon_h"]
    selected = selection[keys + ["selected_model"]]
    output = source.merge(selected, on=keys, how="inner", validate="many_to_one")
    output = output.loc[output["model"].eq(output["selected_model"])]
    return output.drop(columns="selected_model").reset_index(drop=True)


def weather_wave_coupling_report(frame: pd.DataFrame) -> pd.DataFrame:
    train = frame.loc[frame["evaluation_role"].eq("TRAIN")]
    rows = []
    for weather_variable in (
        "wind_speed_ms",
        "pressure_hpa",
        "visibility_m",
        "precipitation",
    ):
        for wave_variable in ("wave_height_m", "wave_period_s"):
            for lag_h in (0, 3, 6, 12, 24):
                pair = pd.DataFrame(
                    {
                        "weather": train[weather_variable].shift(lag_h),
                        "wave": train[wave_variable],
                    }
                ).dropna()
                rows.append(
                    {
                        "weather_variable": weather_variable,
                        "wave_variable": wave_variable,
                        "weather_lag_h": lag_h,
                        "rows": len(pair),
                        "spearman": (
                            float(pair.corr(method="spearman").iloc[0, 1])
                            if len(pair) >= 100
                            else np.nan
                        ),
                        "interpretation": "ASSOCIATION_NOT_CAUSALITY",
                    }
                )
    return pd.DataFrame(rows)


def historical_severity_thresholds(frame: pd.DataFrame) -> dict[str, float]:
    train = frame.loc[frame["evaluation_role"].eq("TRAIN")]
    return {
        "wave_height_q75": float(train["wave_height_m"].quantile(0.75)),
        "wave_height_q90": float(train["wave_height_m"].quantile(0.90)),
        "wind_q75": float(train["wind_speed_ms"].quantile(0.75)),
        "wind_q90": float(train["wind_speed_ms"].quantile(0.90)),
        "visibility_q25": float(train["visibility_m"].quantile(0.25)),
        "visibility_q10": float(train["visibility_m"].quantile(0.10)),
        "period_q90": float(train["wave_period_s"].quantile(0.90)),
    }


def _high_score(value: pd.Series, q75: float, q90: float) -> pd.Series:
    scale = max(q90 - q75, 1e-6)
    return ((value - q75) / scale).clip(0.0, 1.0).fillna(0.0)


def _low_score(value: pd.Series, q25: float, q10: float) -> pd.Series:
    scale = max(q25 - q10, 1e-6)
    return ((q25 - value) / scale).clip(0.0, 1.0).fillna(0.0)


def metocean_severity(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    output = frame.copy()
    hs = _high_score(
        _numeric(output, "wave_height_m"),
        thresholds["wave_height_q75"],
        thresholds["wave_height_q90"],
    )
    wind = _high_score(
        _numeric(output, "wind_speed_ms"), thresholds["wind_q75"], thresholds["wind_q90"]
    )
    visibility = _low_score(
        _numeric(output, "visibility_m"),
        thresholds["visibility_q25"],
        thresholds["visibility_q10"],
    )
    period = (
        _numeric(output, "wave_period_s") / max(thresholds["period_q90"], 1e-6)
    ).sub(0.75).div(0.25).clip(0.0, 1.0).fillna(0.0)
    output["wave_component"] = hs
    output["wind_component"] = wind
    output["visibility_component"] = visibility
    output["period_component"] = period
    output["metocean_severity"] = (
        0.45 * hs + 0.25 * wind + 0.20 * visibility + 0.10 * period
    ).clip(0.0, 1.0)
    output["metocean_tier"] = pd.cut(
        output["metocean_severity"],
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["NORMAL", "WATCH", "ADVERSE", "SEVERE"],
    ).astype(str)
    return output


def vessel_exposure(vessel_type: Any, cargo_group: Any) -> float:
    text = f"{vessel_type or ''} {cargo_group or ''}".upper()
    if any(token in text for token in ("PASSENGER", "FERRY", "LNG", "LPG")):
        return 1.00
    if any(token in text for token in ("TANKER", "CONTAINER", "RO-RO", "RORO")):
        return 0.85
    if any(token in text for token in ("BULK", "GENERAL CARGO")):
        return 0.75
    return 0.65


def build_vessel_impact_shadow(
    watchlist: pd.DataFrame,
    operational_forecast: pd.DataFrame,
) -> pd.DataFrame:
    if watchlist.empty or operational_forecast.empty:
        return pd.DataFrame()
    latest_decision = watchlist["decision_at"].max()
    vessels = watchlist.loc[watchlist["decision_at"].eq(latest_decision)].copy()
    forecast = operational_forecast.copy()
    forecast["issue_at"] = pd.to_datetime(forecast["issue_at"], errors="coerce", utc=True)
    decision_at = pd.Timestamp(latest_decision)
    if decision_at.tzinfo is None:
        decision_at = decision_at.tz_localize("UTC")
    else:
        decision_at = decision_at.tz_convert("UTC")
    context_gap_h = (
        forecast["issue_at"].sub(decision_at).abs().dt.total_seconds() / 3_600.0
    )
    forecast = forecast.loc[context_gap_h.le(24.0)].copy()
    if forecast.empty:
        return pd.DataFrame()
    if "track" not in forecast:
        forecast["track"] = "UNSPECIFIED_SHADOW"
    if "metocean_severity" not in forecast:
        raise ValueError("Operational forecast is missing metocean_severity")
    forecast = forecast.sort_values("valid_at").drop_duplicates("horizon_h")
    vessels["_join"] = 1
    forecast["_join"] = 1
    output = vessels.merge(forecast, on="_join", how="inner").drop(columns="_join")
    output["vessel_exposure"] = [
        vessel_exposure(vessel_type, cargo_group)
        for vessel_type, cargo_group in zip(output["vessel_type"], output["cargo_group"])
    ]
    output["base_temporal_risk"] = _numeric(output, "risk_score").clip(0.0, 1.0)
    output["combined_priority_score"] = (
        0.75 * output["base_temporal_risk"]
        + 0.25 * output["metocean_severity"] * output["vessel_exposure"]
    ).clip(0.0, 1.0)
    output["priority_tier"] = pd.cut(
        output["combined_priority_score"],
        bins=[-np.inf, 0.35, 0.60, 0.80, np.inf],
        labels=["ROUTINE", "REVIEW", "PRIORITY", "CRITICAL_REVIEW"],
    ).astype(str)
    output["forecast_track"] = output["track"]
    output["score_semantics"] = "SHADOW_PRIORITY_NOT_CALIBRATED_PROBABILITY"
    output["automatic_action_allowed"] = False
    output["production_claim_allowed"] = False
    return output


def issue_time_readiness(forecasts: pd.DataFrame) -> dict[str, Any]:
    if forecasts.empty:
        return {
            "collections": 0,
            "span_days": 0.0,
            "issue_time_ready": False,
            "reason": "NO_ISSUE_TIME_FORECASTS",
        }
    issue = pd.to_datetime(forecasts["issue_at"], utc=True)
    span_days = (issue.max() - issue.min()).total_seconds() / 86_400.0
    collections = int(issue.nunique())
    ready = bool(span_days >= 180.0 and collections >= 180)
    return {
        "collections": collections,
        "first_issue_at": issue.min(),
        "last_issue_at": issue.max(),
        "span_days": span_days,
        "issue_time_ready": ready,
        "reason": "READY" if ready else "REQUIRES_180_DAYS_OF_FRESH_ARCHIVE",
    }


def data_signature(*frames: pd.DataFrame, parameters: dict[str, Any] | None = None) -> str:
    digest = hashlib.sha256(MODEL_VERSION.encode("ascii"))
    for frame in frames:
        if frame.empty:
            digest.update(b"EMPTY")
            continue
        hashed = pd.util.hash_pandas_object(frame, index=True)
        digest.update(hashed.to_numpy(dtype="uint64").tobytes())
    if parameters:
        for key, value in sorted(parameters.items()):
            digest.update(f"{key}={value}".encode("utf-8"))
    return digest.hexdigest()


def quality_gates(
    frame: pd.DataFrame,
    invalid_bounds: pd.DataFrame,
    selection: pd.DataFrame,
    issue_readiness: dict[str, Any],
) -> pd.DataFrame:
    critical = [
        (
            "HOURLY_GRID_UNIQUE",
            not frame["observed_at"].duplicated().any(),
            "CRITICAL",
            int(frame["observed_at"].duplicated().sum()),
        ),
        (
            "TRAIN_AT_LEAST_TWO_YEARS",
            int(frame["evaluation_role"].eq("TRAIN").sum()) >= MIN_TRAIN_HOURS,
            "CRITICAL",
            int(frame["evaluation_role"].eq("TRAIN").sum()),
        ),
        (
            "NO_TEST_MODEL_SELECTION",
            selection.empty or not selection["test_used_for_selection"].any(),
            "CRITICAL",
            int(selection.get("test_used_for_selection", pd.Series(dtype=bool)).sum()),
        ),
        (
            "QUANTILE_SELECTION_AVAILABLE",
            not selection.empty,
            "CRITICAL",
            len(selection),
        ),
        (
            "PHYSICAL_BOUNDS_REPAIRED_WITHOUT_TARGET_IMPUTATION",
            True,
            "CRITICAL",
            int(invalid_bounds["outside_bound_rows"].sum()),
        ),
        (
            "ISSUE_TIME_ARCHIVE_180_DAYS",
            bool(issue_readiness["issue_time_ready"]),
            "PRODUCTION_BLOCKER",
            float(issue_readiness["span_days"]),
        ),
    ]
    return pd.DataFrame(
        critical, columns=["check", "passed", "severity", "observed_value"]
    )


def verify_b62_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B62 status: {result.get('status')}")
    metadata = result.get("results") or result
    if metadata.get("critical_gates_passed") not in (True, "true"):
        raise RuntimeError("B62 critical scientific contract failed")
    if metadata.get("test_used_for_selection") not in (False, "false"):
        raise RuntimeError("B62 leakage violation: TEST was used for selection")
    if metadata.get("automatic_action_allowed") not in (False, "false"):
        raise RuntimeError("B62 cannot enable automatic vessel actions")
    return {
        "run_id": result.get("run_id"),
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "selected_chronos_tasks": metadata.get("selected_chronos_tasks"),
        "issue_time_ready": metadata.get("issue_time_ready"),
        "production_allowed": metadata.get("production_promotion_allowed"),
        "serving_forecast_rows": metadata.get("serving_forecast_rows"),
        "serving_impact_rows": metadata.get("serving_impact_rows"),
        "next_block": metadata.get("next_block"),
    }
