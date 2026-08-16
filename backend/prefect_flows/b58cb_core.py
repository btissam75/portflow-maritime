from __future__ import annotations

from typing import Any

import pandas as pd


def normalize_direction_degrees(values: pd.Series) -> pd.Series:
    """Return circular directions in the canonical [0, 360) interval."""
    return pd.to_numeric(values, errors="coerce").mod(360.0)


def normalize_open_meteo_payload(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        reason = payload.get("reason", "hourly payload is missing")
        raise ValueError(f"Invalid Open-Meteo response: {reason}")
    lengths = {
        key: len(value)
        for key, value in hourly.items()
        if isinstance(value, list)
    }
    if not lengths or len(set(lengths.values())) != 1:
        raise ValueError(f"Inconsistent hourly arrays: {lengths}")
    frame = pd.DataFrame(hourly)
    frame["observed_at"] = pd.to_datetime(
        frame.pop("time"), errors="coerce", utc=True
    )
    if frame["observed_at"].isna().any():
        raise ValueError("Open-Meteo returned invalid timestamps")
    for column in frame.columns.difference(["observed_at"]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.sort_values("observed_at")
        .drop_duplicates("observed_at", keep="last")
    )
