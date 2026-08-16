from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from prefect_flows.b60b_core import ORIGIN_STEP_H, RANDOM_SEED, TARGET_SPECS

try:
    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE
    from neuralforecast.models import NHITS, PatchTST

    SEQUENCE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - reported by runtime gate
    torch = None
    NeuralForecast = None
    MAE = None
    NHITS = None
    PatchTST = None
    SEQUENCE_IMPORT_ERROR = str(exc)


ALIASES = ("NHITS_SEQ", "PATCHTST_SEQ")


def sequence_runtime() -> dict[str, Any]:
    return {
        "ready": NeuralForecast is not None,
        "error": SEQUENCE_IMPORT_ERROR,
        "torch_version": getattr(torch, "__version__", None),
        "neuralforecast_version": (
            __import__("neuralforecast").__version__
            if NeuralForecast is not None
            else None
        ),
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
    }


def _calendar(times: pd.Series) -> pd.DataFrame:
    values = pd.to_datetime(times, utc=True)
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2.0 * np.pi * values.dt.hour / 24.0),
            "hour_cos": np.cos(2.0 * np.pi * values.dt.hour / 24.0),
            "day_sin": np.sin(2.0 * np.pi * values.dt.dayofweek / 7.0),
            "day_cos": np.cos(2.0 * np.pi * values.dt.dayofweek / 7.0),
        }
    )


def _panel(
    frame: pd.DataFrame,
    unique_id: str,
    value_column: str,
    split_column: str,
    max_lead_h: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_or_test = frame[split_column].isin(["VALID", "TEST"])
    last_issue = frame.loc[valid_or_test, "as_of_time"].max()
    source = frame.loc[
        frame["as_of_time"] <= last_issue + pd.Timedelta(hours=max_lead_h)
    ].copy()
    source = source.loc[pd.to_numeric(source[value_column], errors="coerce").notna()].copy()
    panel = pd.DataFrame(
        {
            "unique_id": unique_id,
            "ds": pd.to_datetime(source["as_of_time"], utc=True),
            "y": pd.to_numeric(source[value_column], errors="coerce").to_numpy(dtype="float64"),
        }
    )
    calendar = _calendar(source["as_of_time"])
    for column in calendar:
        panel[column] = calendar[column].to_numpy(dtype="float64")
    return panel.reset_index(drop=True), source.reset_index(drop=True)


def _models(horizon_h: int, max_steps: int) -> list[Any]:
    if NeuralForecast is None:
        raise RuntimeError(f"NeuralForecast unavailable: {SEQUENCE_IMPORT_ERROR}")
    common = {
        "h": horizon_h,
        "loss": MAE(),
        "valid_loss": MAE(),
        "max_steps": max_steps,
        "learning_rate": 0.001,
        "early_stop_patience_steps": 5,
        "val_check_steps": 50,
        "batch_size": 1,
        "valid_batch_size": 1,
        "windows_batch_size": 256,
        "inference_windows_batch_size": 128,
        "scaler_type": "robust",
        "random_seed": RANDOM_SEED,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "logger": False,
        "accelerator": "cpu",
        "devices": 1,
    }
    return [
        NHITS(
            input_size=24 * 14,
            n_blocks=[1, 1, 1],
            mlp_units=[[256, 256], [256, 256], [256, 256]],
            n_pool_kernel_size=[24, 6, 1],
            n_freq_downsample=[24, 6, 1],
            dropout_prob_theta=0.10,
            futr_exog_list=["hour_sin", "hour_cos", "day_sin", "day_cos"],
            alias="NHITS_SEQ",
            **common,
        ),
        PatchTST(
            input_size=24 * 30,
            encoder_layers=3,
            n_heads=4,
            hidden_size=64,
            linear_hidden_size=128,
            dropout=0.10,
            fc_dropout=0.10,
            attn_dropout=0.05,
            patch_len=24,
            stride=12,
            revin=True,
            alias="PATCHTST_SEQ",
            **common,
        ),
    ]


def _cross_validation(
    panel: pd.DataFrame,
    first_valid_at: pd.Timestamp,
    horizon_h: int,
    max_steps: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_valid_positions = np.flatnonzero(panel["ds"].ge(first_valid_at).to_numpy())
    if len(first_valid_positions) == 0:
        raise ValueError("Sequence panel does not reach VALID")
    first_valid = int(first_valid_positions[0])
    raw_backtest = len(panel) - first_valid
    windows = max(1, math.ceil(max(raw_backtest - horizon_h, 0) / ORIGIN_STEP_H))
    test_size = horizon_h + windows * ORIGIN_STEP_H
    test_size = min(test_size, len(panel) - 24 * 365 * 3)
    fit_history = len(panel) - test_size
    validation_size = min(24 * 180, max(24 * 60, fit_history // 5))
    if fit_history - validation_size < 24 * 365 * 2:
        raise ValueError("Insufficient history for neural sequence validation")
    models = _models(horizon_h, max_steps)
    started = time.perf_counter()
    forecast = NeuralForecast(models=models, freq="h")
    cv = forecast.cross_validation(
        df=panel,
        val_size=validation_size,
        test_size=test_size,
        n_windows=None,
        step_size=ORIGIN_STEP_H,
        refit=False,
        verbose=0,
    )
    elapsed = time.perf_counter() - started
    inventory = []
    for model in forecast.models:
        parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        inventory.append(
            {
                "model": str(model.alias or type(model).__name__),
                "family": type(model).__name__,
                "features": 4 if getattr(model, "futr_exog_list", []) else 0,
                "fit_rows": fit_history,
                "internal_validation_rows": validation_size,
                "training_seconds": elapsed / len(forecast.models),
                "trainable_parameters": int(parameters),
                "forecast_lead_h": horizon_h,
                "max_steps": max_steps,
                "fit_policy": "FROZEN_PRE_VALID_FIT_DAILY_ROLLING_ORIGINS",
            }
        )
    return cv, pd.DataFrame(inventory)


def _split_lookup(frame: pd.DataFrame, split_column: str) -> dict[pd.Timestamp, str]:
    return dict(
        zip(
            pd.to_datetime(frame["as_of_time"], utc=True),
            frame[split_column].astype(str),
        )
    )


def _actual_lookup(frame: pd.DataFrame, target: str) -> dict[pd.Timestamp, float]:
    values = pd.to_numeric(frame[target], errors="coerce")
    return dict(zip(pd.to_datetime(frame["as_of_time"], utc=True), values))


def _arrival_predictions(frame: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    source = cv.copy()
    source["cutoff"] = pd.to_datetime(source["cutoff"], utc=True)
    source["ds"] = pd.to_datetime(source["ds"], utc=True)
    source["lead_h"] = ((source["ds"] - source["cutoff"]).dt.total_seconds() / 3600.0).round().astype(int)
    split_lookup = _split_lookup(frame, "split_arrival")
    specs = [item for item in TARGET_SPECS if item.kind == "COUNT"]
    actual = {item.target: _actual_lookup(frame, item.target) for item in specs}
    rows = []
    for cutoff, group in source.groupby("cutoff", sort=True):
        split = split_lookup.get(cutoff)
        if split not in {"VALID", "TEST"}:
            continue
        by_lead = group.set_index("lead_h")
        for spec in specs:
            if spec.sequence_end_h not in by_lead.index:
                continue
            target_actual = actual[spec.target].get(cutoff)
            if target_actual is None or not np.isfinite(target_actual):
                continue
            leads = range(int(spec.sequence_start_h), int(spec.sequence_end_h) + 1)
            for model in ALIASES:
                prediction = float(np.clip(by_lead.loc[list(leads), model].to_numpy(dtype="float64"), 0.0, None).sum())
                rows.append(
                    {
                        "split": split,
                        "issue_at": cutoff,
                        "target": spec.target,
                        "actual": float(target_actual),
                        "model": model,
                        "prediction": prediction,
                    }
                )
    return pd.DataFrame(rows)


def _wave_predictions(frame: pd.DataFrame, cv: pd.DataFrame) -> pd.DataFrame:
    source = cv.copy()
    source["cutoff"] = pd.to_datetime(source["cutoff"], utc=True)
    source["ds"] = pd.to_datetime(source["ds"], utc=True)
    source["lead_h"] = ((source["ds"] - source["cutoff"]).dt.total_seconds() / 3600.0).round().astype(int)
    split_lookup = _split_lookup(frame, "split_wave")
    specs = [item for item in TARGET_SPECS if item.kind == "WAVE_PERIOD"]
    actual = {item.target: _actual_lookup(frame, item.target) for item in specs}
    rows = []
    for cutoff, group in source.groupby("cutoff", sort=True):
        split = split_lookup.get(cutoff)
        if split not in {"VALID", "TEST"}:
            continue
        by_lead = group.set_index("lead_h")
        for spec in specs:
            if spec.sequence_lead_h not in by_lead.index:
                continue
            target_actual = actual[spec.target].get(cutoff)
            if target_actual is None or not np.isfinite(target_actual):
                continue
            for model in ALIASES:
                rows.append(
                    {
                        "split": split,
                        "issue_at": cutoff,
                        "target": spec.target,
                        "actual": float(target_actual),
                        "model": model,
                        "prediction": float(max(by_lead.loc[int(spec.sequence_lead_h), model], 0.0)),
                    }
                )
    return pd.DataFrame(rows)


def run_sequence_models(
    frame: pd.DataFrame,
    max_steps: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    runtime = sequence_runtime()
    if not runtime["ready"]:
        raise RuntimeError(f"Sequence runtime unavailable: {runtime['error']}")
    torch.set_num_threads(2)
    source = frame.copy().sort_values("as_of_time").reset_index(drop=True)
    source["as_of_time"] = pd.to_datetime(source["as_of_time"], utc=True)

    arrival_panel, _ = _panel(
        source,
        "ARRIVAL_COUNTS",
        "arrivals_prev_1h",
        "split_arrival",
        24,
    )
    arrival_valid_at = source.loc[source["split_arrival"].eq("VALID"), "as_of_time"].min()
    arrival_cv, arrival_inventory = _cross_validation(
        arrival_panel, arrival_valid_at, 24, max_steps
    )
    arrival_inventory["target"] = "ARRIVAL_COUNT_SEQUENCE_PANEL"

    wave_panel, _ = _panel(
        source,
        "WAVE_PERIOD_SAFE_LAG3",
        "wave_period_lag_3h",
        "split_wave",
        75,
    )
    wave_valid_at = source.loc[source["split_wave"].eq("VALID"), "as_of_time"].min()
    wave_cv, wave_inventory = _cross_validation(
        wave_panel, wave_valid_at, 75, max_steps
    )
    wave_inventory["target"] = "WAVE_PERIOD_SEQUENCE_PANEL"

    predictions = pd.concat(
        [
            _arrival_predictions(source, arrival_cv),
            _wave_predictions(source, wave_cv),
        ],
        ignore_index=True,
    )
    inventory = pd.concat([arrival_inventory, wave_inventory], ignore_index=True)
    runtime["prediction_rows"] = len(predictions)
    runtime["arrival_cv_rows"] = len(arrival_cv)
    runtime["wave_cv_rows"] = len(wave_cv)
    return predictions, inventory, runtime
