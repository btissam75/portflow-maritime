from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from prefect_flows.b62_core import (
    CALENDAR_COVARIATES,
    HORIZONS_H,
    QUANTILES,
    WAVE_TARGETS,
    WEATHER_COVARIATES,
    WEATHER_TARGETS,
    enforce_forecast_constraints,
)


try:
    import autogluon
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

    AUTOGLUON_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - reported through the runtime gate
    autogluon = None
    TimeSeriesDataFrame = None
    TimeSeriesPredictor = None
    AUTOGLUON_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def autogluon_runtime() -> dict[str, Any]:
    try:
        autogluon_version = version("autogluon.timeseries")
    except PackageNotFoundError:
        autogluon_version = None
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - runtime diagnostic
        torch_version = f"UNAVAILABLE:{type(exc).__name__}"
        cuda_available = False
    return {
        "ready": TimeSeriesPredictor is not None,
        "error": AUTOGLUON_IMPORT_ERROR,
        "autogluon_version": autogluon_version,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "model_preset": "chronos2_small",
        "device_policy": "CPU_ONLY",
    }


def _naive_utc(times: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(times, errors="coerce", utc=True))
    return values.tz_convert("UTC").tz_localize(None)


def _panel(
    frame: pd.DataFrame,
    targets: Iterable[str],
    covariates: Iterable[str],
    end_at: pd.Timestamp,
    covariate_fill: dict[str, float] | None = None,
) -> pd.DataFrame:
    source = frame.loc[frame["observed_at"].le(end_at)].copy()
    source = source.sort_values("observed_at")
    rows = []
    covariates = list(covariates)
    fill = covariate_fill or {}
    for target in targets:
        item = pd.DataFrame(
            {
                "item_id": target,
                "timestamp": _naive_utc(source["observed_at"]),
                "target": pd.to_numeric(source[target], errors="coerce").to_numpy(
                    dtype="float64"
                ),
            }
        )
        for covariate in covariates:
            values = pd.to_numeric(source[covariate], errors="coerce")
            if covariate in fill:
                values = values.fillna(fill[covariate])
            item[covariate] = values.to_numpy(dtype="float64")
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def _future_calendar(
    item_ids: Iterable[str],
    origin: pd.Timestamp,
    prediction_length: int,
) -> pd.DataFrame:
    times = pd.date_range(
        origin + pd.Timedelta(hours=1), periods=prediction_length, freq="h", tz="UTC"
    )
    base = pd.DataFrame(
        {
            "timestamp": _naive_utc(times),
            "hour_sin": np.sin(2.0 * np.pi * times.hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * times.hour / 24.0),
            "dow_sin": np.sin(2.0 * np.pi * times.dayofweek / 7.0),
            "dow_cos": np.cos(2.0 * np.pi * times.dayofweek / 7.0),
            "year_sin": np.sin(2.0 * np.pi * times.dayofyear / 365.25),
            "year_cos": np.cos(2.0 * np.pi * times.dayofyear / 365.25),
        }
    )
    rows = []
    for item_id in item_ids:
        item = base.copy()
        item.insert(0, "item_id", item_id)
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def training_covariate_fill(frame: pd.DataFrame) -> dict[str, float]:
    train = frame.loc[frame["evaluation_role"].eq("TRAIN")]
    fill = {}
    for column in WEATHER_COVARIATES:
        values = pd.to_numeric(train[column], errors="coerce")
        median = float(values.median())
        fill[column] = median if np.isfinite(median) else 0.0
    return fill


def _as_timeseries(frame: pd.DataFrame):
    if TimeSeriesDataFrame is None:
        raise RuntimeError(f"AutoGluon is unavailable: {AUTOGLUON_IMPORT_ERROR}")
    return TimeSeriesDataFrame.from_data_frame(
        frame,
        id_column="item_id",
        timestamp_column="timestamp",
    )


def fit_chronos_predictor(
    frame: pd.DataFrame,
    targets: Iterable[str],
    covariates: Iterable[str],
    train_end: pd.Timestamp,
    path: Path,
    preset: str = "chronos2_small",
    covariate_fill: dict[str, float] | None = None,
) -> Any:
    runtime = autogluon_runtime()
    if not runtime["ready"]:
        raise RuntimeError(f"AutoGluon runtime unavailable: {runtime['error']}")
    path.mkdir(parents=True, exist_ok=True)
    panel = _panel(frame, targets, covariates, train_end, covariate_fill)
    train_data = _as_timeseries(panel)
    predictor = TimeSeriesPredictor(
        path=str(path),
        prediction_length=max(HORIZONS_H),
        target="target",
        known_covariates_names=list(covariates),
        quantile_levels=list(QUANTILES),
        eval_metric="WQL",
        freq="h",
        verbosity=2,
    )
    predictor.fit(train_data=train_data, presets=preset)
    return predictor


def _prediction_frame(predictions: Any, origin: pd.Timestamp) -> pd.DataFrame:
    # AutoGluon preserves its TimeSeriesDataFrame subclass after reset_index().
    # Convert to a plain DataFrame before using pandas reshape operations.
    source = pd.DataFrame(predictions.reset_index())
    source["valid_at"] = pd.to_datetime(source["timestamp"], errors="coerce", utc=True)
    source["issue_at"] = origin
    source["horizon_h"] = (
        (source["valid_at"] - origin).dt.total_seconds() / 3_600.0
    ).round().astype("int16")
    source = source.rename(
        columns={
            "item_id": "variable",
            "0.1": "p10",
            "0.5": "p50",
            "0.9": "p90",
            0.1: "p10",
            0.5: "p50",
            0.9: "p90",
        }
    )
    if "p50" not in source and "mean" in source:
        source["p50"] = source["mean"]
    return source[["issue_at", "valid_at", "horizon_h", "variable", "p10", "p50", "p90"]]


def predict_weather_origin(
    predictor: Any,
    frame: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    context = _panel(frame, WEATHER_TARGETS, CALENDAR_COVARIATES, origin)
    future = _future_calendar(WEATHER_TARGETS, origin, max(HORIZONS_H))
    result = predictor.predict(
        _as_timeseries(context),
        known_covariates=_as_timeseries(future),
        use_cache=False,
        random_seed=20260811,
    )
    return _prediction_frame(result, origin)


def _wave_future_covariates(
    weather_predictions: pd.DataFrame,
    covariate_fill: dict[str, float],
    origin: pd.Timestamp,
) -> pd.DataFrame:
    weather_source = pd.DataFrame(weather_predictions).copy()
    weather = weather_source.pivot_table(
        index="valid_at", columns="variable", values="p50", aggfunc="first"
    ).sort_index()
    expected = pd.date_range(
        origin + pd.Timedelta(hours=1), periods=max(HORIZONS_H), freq="h", tz="UTC"
    )
    weather = weather.reindex(expected)
    for column in WEATHER_COVARIATES:
        if column not in weather:
            weather[column] = covariate_fill[column]
        weather[column] = pd.to_numeric(weather[column], errors="coerce").fillna(
            covariate_fill[column]
        )
    calendar = _future_calendar(["_calendar"], origin, max(HORIZONS_H)).drop(
        columns="item_id"
    )
    base = pd.DataFrame({"timestamp": _naive_utc(expected)})
    for column in WEATHER_COVARIATES:
        base[column] = weather[column].to_numpy(dtype="float64")
    for column in CALENDAR_COVARIATES:
        base[column] = calendar[column].to_numpy(dtype="float64")
    rows = []
    for target in WAVE_TARGETS:
        item = base.copy()
        item.insert(0, "item_id", target)
        rows.append(item)
    return pd.concat(rows, ignore_index=True)


def predict_wave_origin(
    predictor: Any,
    frame: pd.DataFrame,
    weather_predictions: pd.DataFrame,
    origin: pd.Timestamp,
    covariate_fill: dict[str, float],
) -> pd.DataFrame:
    covariates = [*WEATHER_COVARIATES, *CALENDAR_COVARIATES]
    context = _panel(frame, WAVE_TARGETS, covariates, origin, covariate_fill)
    future = _wave_future_covariates(weather_predictions, covariate_fill, origin)
    result = predictor.predict(
        _as_timeseries(context),
        known_covariates=_as_timeseries(future),
        use_cache=False,
        random_seed=20260811,
    )
    return _prediction_frame(result, origin)


def _attach_actuals(
    forecasts: pd.DataFrame,
    frame: pd.DataFrame,
    family: str,
    role: str,
) -> pd.DataFrame:
    actual = frame.set_index("observed_at")
    source = forecasts.loc[forecasts["horizon_h"].isin(HORIZONS_H)].copy()
    source["actual"] = [
        actual.at[valid_at, variable]
        if valid_at in actual.index and variable in actual.columns
        else np.nan
        for valid_at, variable in zip(source["valid_at"], source["variable"])
    ]
    source = source.loc[pd.to_numeric(source["actual"], errors="coerce").notna()].copy()
    source["family"] = family
    source["evaluation_role"] = role
    source["model"] = "CHRONOS2_SMALL"
    return source[
        [
            "family",
            "evaluation_role",
            "issue_at",
            "valid_at",
            "horizon_h",
            "variable",
            "model",
            "actual",
            "p10",
            "p50",
            "p90",
        ]
    ]


def cascade_forecast_origin(
    weather_predictor: Any,
    wave_predictor: Any,
    frame: pd.DataFrame,
    origin: pd.Timestamp,
    covariate_fill: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    weather = predict_weather_origin(weather_predictor, frame, origin)
    wave = predict_wave_origin(
        wave_predictor, frame, weather, origin, covariate_fill
    )
    return enforce_forecast_constraints(weather), enforce_forecast_constraints(wave)


def cascade_backtest(
    weather_predictor: Any,
    wave_predictor: Any,
    frame: pd.DataFrame,
    origins: Iterable[pd.Timestamp],
    role: str,
    covariate_fill: dict[str, float],
    progress: Callable[[int, int, pd.Timestamp], None] | None = None,
) -> pd.DataFrame:
    rows = []
    origins = list(origins)
    for index, origin in enumerate(origins, start=1):
        weather, wave = cascade_forecast_origin(
            weather_predictor, wave_predictor, frame, origin, covariate_fill
        )
        rows.append(_attach_actuals(weather, frame, "WEATHER", role))
        rows.append(_attach_actuals(wave, frame, "WAVE", role))
        if progress is not None:
            progress(index, len(origins), origin)
    if not rows:
        return pd.DataFrame()
    return enforce_forecast_constraints(pd.concat(rows, ignore_index=True))


def fit_cascade(
    frame: pd.DataFrame,
    train_end: pd.Timestamp,
    model_root: Path,
    preset: str = "chronos2_small",
) -> tuple[Any, Any, dict[str, float], dict[str, Any]]:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    fill = training_covariate_fill(frame)
    weather = fit_chronos_predictor(
        frame,
        WEATHER_TARGETS,
        CALENDAR_COVARIATES,
        train_end,
        model_root / "weather",
        preset=preset,
    )
    wave = fit_chronos_predictor(
        frame,
        WAVE_TARGETS,
        [*WEATHER_COVARIATES, *CALENDAR_COVARIATES],
        train_end,
        model_root / "wave",
        preset=preset,
        covariate_fill=fill,
    )
    runtime = autogluon_runtime()
    runtime["weather_items"] = len(WEATHER_TARGETS)
    runtime["wave_items"] = len(WAVE_TARGETS)
    runtime["prediction_length"] = max(HORIZONS_H)
    runtime["weather_to_wave_covariates"] = list(WEATHER_COVARIATES)
    return weather, wave, fill, runtime
