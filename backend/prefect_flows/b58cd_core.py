from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from decimal import Decimal
from typing import Any, Iterable

import numpy as np
import pandas as pd


WEATHER_VARIABLE_MAP = {
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_direction_deg",
    "pressure_msl": "pressure_hpa",
    "visibility": "visibility_m",
    "temperature_2m": "air_temperature_c",
}
MARINE_VARIABLE_MAP = {
    "wave_height": "wave_height_m",
    "wave_direction": "wave_direction_deg",
    "wave_period": "wave_period_s",
    "ocean_current_velocity": "ocean_current_ms",
    "ocean_current_direction": "ocean_current_direction_deg",
    "sea_surface_temperature": "sea_surface_temperature_c",
}
DIRECTION_COLUMNS = (
    "wind_direction_deg",
    "wave_direction_deg",
    "ocean_current_direction_deg",
)


def clean_json_value(value: Any) -> Any:
    """Convert database and dataframe scalars into strict JSON values."""
    if isinstance(value, dict):
        return {str(key): clean_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (Decimal, np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def canonical_payload_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def payload_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload_bytes(payload)).hexdigest()


def normalize_direction_degrees(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.mod(360.0)


def normalize_hourly_payload(
    payload: dict[str, Any],
    variable_map: dict[str, str],
) -> pd.DataFrame:
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise ValueError("Open-Meteo payload has no hourly time array")
    times = hourly["time"]
    frame = pd.DataFrame(
        {"valid_at": pd.to_datetime(times, utc=True, errors="coerce")}
    )
    if frame["valid_at"].isna().any() or frame["valid_at"].duplicated().any():
        raise ValueError("Forecast valid times are invalid or duplicated")
    for source, canonical in variable_map.items():
        values = hourly.get(source, [None] * len(times))
        if not isinstance(values, list) or len(values) != len(times):
            raise ValueError(f"Inconsistent hourly array: {source}")
        frame[canonical] = pd.to_numeric(pd.Series(values), errors="coerce")
    return frame


def _empty_hourly(columns: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["valid_at", *columns])


def build_issue_time_forecast_frame(
    weather_payload: dict[str, Any] | None,
    marine_payload: dict[str, Any] | None,
    issue_at: datetime | pd.Timestamp,
    weather_available_at: datetime | pd.Timestamp | None,
    marine_available_at: datetime | pd.Timestamp | None,
    forecast_hours: int,
) -> pd.DataFrame:
    issue = pd.Timestamp(issue_at)
    if issue.tzinfo is None:
        issue = issue.tz_localize("UTC")
    else:
        issue = issue.tz_convert("UTC")

    weather = (
        normalize_hourly_payload(weather_payload, WEATHER_VARIABLE_MAP)
        if weather_payload is not None
        else _empty_hourly(WEATHER_VARIABLE_MAP.values())
    )
    marine = (
        normalize_hourly_payload(marine_payload, MARINE_VARIABLE_MAP)
        if marine_payload is not None
        else _empty_hourly(MARINE_VARIABLE_MAP.values())
    )
    if weather.empty and marine.empty:
        raise ValueError("Both weather and marine forecasts are empty")

    weather_available = (
        pd.Timestamp(weather_available_at).tz_convert("UTC")
        if weather_available_at is not None
        else pd.NaT
    )
    marine_available = (
        pd.Timestamp(marine_available_at).tz_convert("UTC")
        if marine_available_at is not None
        else pd.NaT
    )
    available_candidates = [
        value for value in (weather_available, marine_available) if pd.notna(value)
    ]
    if not available_candidates:
        raise ValueError("At least one real provider availability timestamp is required")
    combined_available = max(available_candidates)

    frame = weather.merge(marine, on="valid_at", how="outer", validate="one_to_one")
    frame = frame.sort_values("valid_at").reset_index(drop=True)
    first_valid = max(issue, combined_available).ceil("h")
    last_valid = issue + pd.Timedelta(hours=forecast_hours)
    frame = frame.loc[
        frame["valid_at"].between(first_valid, last_valid, inclusive="both")
    ].copy()
    if frame.empty:
        raise ValueError("Forecast payload has no usable future rows after availability")

    for column in DIRECTION_COLUMNS:
        frame[column] = normalize_direction_degrees(frame[column])

    frame.insert(0, "issue_at", issue)
    frame.insert(1, "available_at", combined_available)
    frame["weather_available_at"] = weather_available
    frame["marine_available_at"] = marine_available
    frame["lead_time_h"] = (
        frame["valid_at"] - frame["issue_at"]
    ).dt.total_seconds() / 3600.0
    frame["atmosphere_available_flag"] = frame[
        ["wind_speed_ms", "wind_direction_deg", "pressure_hpa"]
    ].notna().all(axis=1)
    frame["visibility_available_flag"] = frame["visibility_m"].notna()
    frame["wave_available_flag"] = frame[
        ["wave_height_m", "wave_direction_deg", "wave_period_s"]
    ].notna().all(axis=1)
    frame["marine_current_available_flag"] = frame[
        [
            "ocean_current_ms",
            "ocean_current_direction_deg",
            "sea_surface_temperature_c",
        ]
    ].notna().all(axis=1)
    frame["full_weather_available_flag"] = frame[
        [
            "atmosphere_available_flag",
            "visibility_available_flag",
            "wave_available_flag",
            "marine_current_available_flag",
        ]
    ].all(axis=1)
    if (frame["available_at"] < frame["issue_at"]).any():
        raise ValueError("Provider availability precedes collection issue time")
    if (frame["valid_at"] < frame["available_at"]).any():
        raise ValueError("Forecast valid time precedes provider availability")
    return frame.reset_index(drop=True)
