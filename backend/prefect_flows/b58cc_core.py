from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def circular_absolute_error(
    actual: Iterable[float], predicted: Iterable[float]
) -> np.ndarray:
    actual_array = np.asarray(actual, dtype="float64")
    predicted_array = np.asarray(predicted, dtype="float64")
    return np.abs((predicted_array - actual_array + 180.0) % 360.0 - 180.0)


def assign_purged_split(
    issue_at: pd.Series,
    target_at: pd.Series,
    train_boundary: pd.Timestamp,
    valid_boundary: pd.Timestamp,
    purge_h: int,
) -> pd.Series:
    issue = pd.to_datetime(issue_at, utc=True)
    target = pd.to_datetime(target_at, utc=True)
    purge = pd.Timedelta(hours=purge_h)
    result = np.full(len(issue), "EXCLUDED_PURGE", dtype=object)
    result[target < train_boundary] = "TRAIN"
    result[
        (issue >= train_boundary + purge) & (target < valid_boundary)
    ] = "VALID"
    result[issue >= valid_boundary + purge] = "TEST"
    return pd.Series(result, index=issue_at.index, dtype="object")


def block_bootstrap_error_delta(
    frame: pd.DataFrame,
    baseline_error: str,
    candidate_error: str,
    time_column: str = "target_at",
    iterations: int = 500,
    seed: int = 20260802,
) -> dict[str, float | int]:
    sample = frame[[time_column, baseline_error, candidate_error]].dropna().copy()
    if sample.empty:
        return {
            "rows": 0,
            "days": 0,
            "baseline_mae": np.nan,
            "candidate_mae": np.nan,
            "gain_pct": np.nan,
            "delta_mae_ci95_low": np.nan,
            "delta_mae_ci95_high": np.nan,
        }
    sample["day"] = pd.to_datetime(sample[time_column], utc=True).dt.floor("D")
    daily = (
        sample.groupby("day", as_index=False)
        .agg(
            baseline_sum=(baseline_error, "sum"),
            candidate_sum=(candidate_error, "sum"),
            rows=(baseline_error, "size"),
        )
        .sort_values("day")
    )
    baseline_mae = float(sample[baseline_error].mean())
    candidate_mae = float(sample[candidate_error].mean())
    gain_pct = (
        100.0 * (baseline_mae - candidate_mae) / baseline_mae
        if baseline_mae > 0
        else np.nan
    )
    generator = np.random.default_rng(seed)
    positions = np.arange(len(daily))
    deltas = np.empty(iterations, dtype="float64")
    for index in range(iterations):
        selected = daily.iloc[generator.choice(positions, size=len(daily), replace=True)]
        count = float(selected["rows"].sum())
        deltas[index] = (
            float(selected["baseline_sum"].sum())
            - float(selected["candidate_sum"].sum())
        ) / count
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "rows": int(len(sample)),
        "days": int(len(daily)),
        "baseline_mae": baseline_mae,
        "candidate_mae": candidate_mae,
        "gain_pct": float(gain_pct),
        "delta_mae_ci95_low": float(low),
        "delta_mae_ci95_high": float(high),
    }


def decide_marginal_family(
    cells: pd.DataFrame,
    minimum_gain_pct: float = 2.0,
    minimum_supported_fraction: float = 0.5,
    severe_degradation_pct: float = -5.0,
) -> dict[str, float | int | str | bool]:
    eligible = cells.dropna(
        subset=["gain_pct", "delta_mae_ci95_low", "delta_mae_ci95_high"]
    )
    if eligible.empty:
        return {
            "decision": "INSUFFICIENT_EVIDENCE",
            "keep": False,
            "cells": 0,
            "supported_cells": 0,
            "severely_degraded_cells": 0,
            "median_gain_pct": np.nan,
        }
    supported = int(eligible["delta_mae_ci95_low"].gt(0.0).sum())
    severe = int(eligible["gain_pct"].lt(severe_degradation_pct).sum())
    required = int(np.ceil(len(eligible) * minimum_supported_fraction))
    median_gain = float(eligible["gain_pct"].median())
    keep = median_gain >= minimum_gain_pct and supported >= required and severe == 0
    return {
        "decision": "KEEP" if keep else "REJECT_OR_REPAIR",
        "keep": bool(keep),
        "cells": int(len(eligible)),
        "supported_cells": supported,
        "required_supported_cells": required,
        "severely_degraded_cells": severe,
        "median_gain_pct": median_gain,
    }
