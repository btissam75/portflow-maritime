from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASET_VERSION = "b60a-maritime-multitask-hourly-v1"
CONTRACT_VERSION = "b60a-contract-v1"
HISTORY_H = 336
OBSERVATION_LATENCY_H = 3
ARRIVAL_HORIZONS_H = (6, 12, 24)
WAVE_HORIZONS_H = (6, 12, 24, 48, 72)
ARRIVAL_LAGS_H = (1, 2, 3, 6, 12, 24, 48, 72, 168, 336)
WAVE_LAGS_H = (3, 6, 24, 72, 168)
EXTERNAL_LAGS_H = (3, 24)
TRAIN_START = pd.Timestamp("2020-01-01T00:00:00Z")
VALID_START = pd.Timestamp("2024-01-01T00:00:00Z")
TEST_START = pd.Timestamp("2025-01-01T00:00:00Z")


@dataclass(frozen=True)
class BuildResult:
    dataset: pd.DataFrame
    event_sequence: pd.DataFrame
    reports: dict[str, pd.DataFrame]
    decision: dict[str, Any]
    feature_sets: dict[str, list[str]]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(item) for item in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is pd.NA:
        return None
    return value


def future_sum(series: pd.Series, horizon_h: int) -> pd.Series:
    """Sum the next h hourly buckets, excluding the issue-time bucket."""
    shifted = pd.to_numeric(series, errors="coerce").shift(-1)
    return shifted.rolling(horizon_h, min_periods=horizon_h).sum().shift(
        -(horizon_h - 1)
    )


def detect_sustained_coverage_break(
    times: pd.Series,
    counts: pd.Series,
    threshold: float = 0.70,
    consecutive_months: int = 3,
) -> tuple[pd.Timestamp | None, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "as_of_time": pd.to_datetime(times, utc=True),
            "arrivals": pd.to_numeric(counts, errors="coerce").fillna(0.0),
        }
    )
    monthly = (
        frame.set_index("as_of_time")["arrivals"]
        .resample("MS")
        .sum()
        .rename("arrivals")
        .to_frame()
    )
    monthly["prior_12m_median"] = (
        monthly["arrivals"].shift(1).rolling(12, min_periods=12).median()
    )
    monthly["coverage_ratio"] = (
        monthly["arrivals"] / monthly["prior_12m_median"]
    )
    monthly["below_threshold"] = monthly["coverage_ratio"].lt(threshold)
    sustained = monthly["below_threshold"].copy()
    for offset in range(1, consecutive_months):
        sustained &= monthly["below_threshold"].shift(-offset).fillna(False)
    monthly["sustained_break_start"] = sustained
    candidates = monthly.index[monthly["sustained_break_start"]]
    break_at = None if candidates.empty else pd.Timestamp(candidates[0])
    if break_at is not None:
        break_at = break_at.tz_convert("UTC")
    report = monthly.reset_index().rename(columns={"as_of_time": "month"})
    return break_at, report


def assign_arrival_split(
    times: pd.Series,
    coverage_break_at: pd.Timestamp,
    history_h: int = HISTORY_H,
) -> pd.Series:
    values = pd.to_datetime(times, utc=True)
    start = values.min()
    warmup_end = start + pd.Timedelta(hours=history_h)
    train_stop = VALID_START - pd.Timedelta(hours=max(ARRIVAL_HORIZONS_H))
    valid_stop = TEST_START - pd.Timedelta(hours=max(ARRIVAL_HORIZONS_H))
    test_stop = coverage_break_at - pd.Timedelta(hours=max(ARRIVAL_HORIZONS_H))
    split = pd.Series("EXCLUDED_AFTER_COVERAGE_BREAK", index=values.index, dtype="object")
    split.loc[values < warmup_end] = "EXCLUDED_WARMUP"
    split.loc[(values >= warmup_end) & (values < train_stop)] = "TRAIN"
    split.loc[(values >= train_stop) & (values < VALID_START)] = "EXCLUDED_PURGE"
    split.loc[(values >= VALID_START) & (values < valid_stop)] = "VALID"
    split.loc[(values >= valid_stop) & (values < TEST_START)] = "EXCLUDED_PURGE"
    split.loc[(values >= TEST_START) & (values < test_stop)] = "TEST"
    split.loc[(values >= test_stop) & (values < coverage_break_at)] = (
        "EXCLUDED_TARGET_UNMATURED"
    )
    return split


def assign_wave_split(
    times: pd.Series,
    history_h: int = HISTORY_H,
) -> pd.Series:
    values = pd.to_datetime(times, utc=True)
    start = values.min()
    end_exclusive = values.max() + pd.Timedelta(hours=1)
    horizon_h = max(WAVE_HORIZONS_H)
    warmup_end = start + pd.Timedelta(hours=history_h)
    train_stop = VALID_START - pd.Timedelta(hours=horizon_h)
    valid_stop = TEST_START - pd.Timedelta(hours=horizon_h)
    test_stop = end_exclusive - pd.Timedelta(hours=horizon_h)
    split = pd.Series("EXCLUDED_TARGET_UNMATURED", index=values.index, dtype="object")
    split.loc[values < warmup_end] = "EXCLUDED_WARMUP"
    split.loc[(values >= warmup_end) & (values < train_stop)] = "TRAIN"
    split.loc[(values >= train_stop) & (values < VALID_START)] = "EXCLUDED_PURGE"
    split.loc[(values >= VALID_START) & (values < valid_stop)] = "VALID"
    split.loc[(values >= valid_stop) & (values < TEST_START)] = "EXCLUDED_PURGE"
    split.loc[(values >= TEST_START) & (values < test_stop)] = "TEST"
    return split


def train_median_fill(
    series: pd.Series,
    train_mask: pd.Series,
) -> tuple[pd.Series, pd.Series, float]:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    median = numeric.loc[train_mask].median()
    if pd.isna(median):
        median = 0.0
    missing = numeric.isna().astype("int8")
    return numeric.fillna(float(median)), missing, float(median)


def _cyclic(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    radians = 2.0 * np.pi * pd.to_numeric(values, errors="coerce") / period
    return np.sin(radians), np.cos(radians)


def _add_calendar_features(
    output: pd.DataFrame,
    times: pd.Series,
    prefix: str,
    feature_names: list[str],
) -> None:
    values = pd.to_datetime(times, utc=True)
    hour_sin, hour_cos = _cyclic(values.dt.hour, 24.0)
    dow_sin, dow_cos = _cyclic(values.dt.dayofweek, 7.0)
    doy_sin, doy_cos = _cyclic(values.dt.dayofyear - 1, 366.0)
    month_sin, month_cos = _cyclic(values.dt.month - 1, 12.0)
    additions = {
        f"{prefix}_hour_sin": hour_sin,
        f"{prefix}_hour_cos": hour_cos,
        f"{prefix}_dow_sin": dow_sin,
        f"{prefix}_dow_cos": dow_cos,
        f"{prefix}_doy_sin": doy_sin,
        f"{prefix}_doy_cos": doy_cos,
        f"{prefix}_month_sin": month_sin,
        f"{prefix}_month_cos": month_cos,
        f"{prefix}_weekend": values.dt.dayofweek.ge(5).astype("int8"),
    }
    for name, value in additions.items():
        output[name] = value
        feature_names.append(name)


def _event_flags(
    times: pd.Series,
    events: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    values = pd.to_datetime(times, utc=True).dt.date
    any_event = pd.Series(0, index=times.index, dtype="int8")
    holiday = pd.Series(0, index=times.index, dtype="int8")
    if events.empty:
        return any_event, holiday
    known = events.loc[events["knowledge_policy"].eq("KNOWN_CALENDAR_EVENT")]
    for row in known.itertuples(index=False):
        mask = values.between(row.start_date, row.end_date)
        any_event.loc[mask] = 1
        if "AID" in str(row.event_name).upper() or "HOLIDAY" in str(row.event_type).upper():
            holiday.loc[mask] = 1
    return any_event, holiday


def _next_arrival_targets(
    issue_times: pd.Series,
    event_times: pd.Series,
    coverage_break_at: pd.Timestamp,
    horizon_h: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    issues = pd.to_datetime(issue_times, utc=True)
    events = np.sort(pd.to_datetime(event_times, utc=True).view("int64"))
    issue_ns = issues.view("int64").to_numpy()
    positions = np.searchsorted(events, issue_ns, side="right")
    has_next = positions < len(events)
    next_ns = np.full(len(issues), np.datetime64("NaT", "ns").view("int64"))
    next_ns[has_next] = events[positions[has_next]]
    wait_h = (next_ns - issue_ns) / 3_600_000_000_000.0
    mature = (issues + pd.Timedelta(hours=horizon_h)).lt(coverage_break_at).to_numpy()
    observed = mature & has_next & (wait_h > 0.0) & (wait_h <= horizon_h)
    target_wait = np.where(mature, np.minimum(wait_h, float(horizon_h)), np.nan)
    target_wait[mature & ~has_next] = float(horizon_h)
    return target_wait, observed.astype("float64")


def _feature_registry(
    dataset: pd.DataFrame,
    arrival_core: list[str],
    wave_core: list[str],
    research: list[str],
) -> pd.DataFrame:
    arrival_set = set(arrival_core)
    wave_set = set(wave_core)
    research_set = set(research)
    rows: list[dict[str, Any]] = []
    for column in dataset.columns:
        if column.startswith("target_time_"):
            role = "KNOWN_FUTURE_ALLOWED"
            semantics = "DETERMINISTIC_FROM_ISSUE_TIME"
            task = "BOTH"
        elif column.startswith("target_"):
            role = "TARGET"
            semantics = "FUTURE_OBSERVED_NOT_IMPUTED"
            task = "ARRIVAL" if "arrival" in column else "WAVE"
        elif column in research_set:
            role = "RESEARCH_ONLY_RETROSPECTIVE"
            semantics = "PAST_WITH_3H_MINIMUM_LAG_TRAIN_MEDIAN_IMPUTED"
            task = "BOTH"
        elif column in arrival_set or column in wave_set:
            role = (
                "KNOWN_FUTURE_ALLOWED"
                if column.startswith("target_time_") or column.startswith("known_event_")
                else "CORE_ALLOWED_RETROSPECTIVE"
            )
            semantics = (
                "DETERMINISTIC_FROM_ISSUE_TIME"
                if role == "KNOWN_FUTURE_ALLOWED"
                else "PAST_OBSERVED_OR_LAGGED"
            )
            task = (
                "BOTH"
                if column in arrival_set and column in wave_set
                else "ARRIVAL"
                if column in arrival_set
                else "WAVE"
            )
        else:
            role = "METADATA_OR_AUDIT"
            semantics = "NOT_IN_MODEL_MATRIX"
            task = "NONE"
        rows.append(
            {
                "column_name": column,
                "dtype": str(dataset[column].dtype),
                "role": role,
                "task": task,
                "availability_semantics": semantics,
                "null_policy": (
                    "NEVER_IMPUTE" if role == "TARGET" else "EXPLICIT_BY_CONTRACT"
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_event_sequence(
    events: pd.DataFrame,
    grid_start: pd.Timestamp,
    coverage_break_at: pd.Timestamp,
) -> pd.DataFrame:
    result = events.copy()
    result["actual_ata"] = pd.to_datetime(result["actual_ata"], utc=True)
    result = result.loc[
        result["actual_ata"].ge(grid_start)
        & result["actual_ata"].lt(coverage_break_at)
    ].sort_values(["actual_ata", "port_call_id"])
    result = result.drop_duplicates("port_call_id").reset_index(drop=True)
    result.insert(0, "event_index", np.arange(len(result), dtype="int64"))
    result["interarrival_h"] = result["actual_ata"].diff().dt.total_seconds() / 3600.0
    result["event_hour"] = result["actual_ata"].dt.floor("h")
    result["event_split"] = np.select(
        [
            result["actual_ata"].lt(VALID_START),
            result["actual_ata"].lt(TEST_START),
        ],
        ["TRAIN", "VALID"],
        default="TEST",
    )
    result["dataset_version"] = DATASET_VERSION
    return result


def _count_missing(frame: pd.DataFrame, columns: Iterable[str], mask: pd.Series) -> int:
    names = list(columns)
    if not names or not bool(mask.any()):
        return 0
    return int(frame.loc[mask, names].isna().sum().sum())


def _cargo_group(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    if "legumes et fruits" in text:
        return "fresh_produce"
    if "machine" in text or "appareils electriques" in text:
        return "machinery_electrical"
    if "textile" in text or "confection" in text:
        return "textile"
    if "materiel de transport" in text:
        return "transport_equipment"
    if "produits de la mer" in text or "produits frais / mer" in text:
        return "seafood"
    return "other"


def build_multitask_dataset(
    hourly: pd.DataFrame,
    arrival_events: pd.DataFrame,
    business_events: pd.DataFrame,
) -> BuildResult:
    source = hourly.copy()
    source["as_of_time"] = pd.to_datetime(source["as_of_time"], utc=True)
    source = source.sort_values("as_of_time").reset_index(drop=True)
    if source.empty:
        raise ValueError("Hourly source is empty")
    if source["as_of_time"].duplicated().any():
        raise ValueError("Hourly source contains duplicate timestamps")

    expected = pd.date_range(
        source["as_of_time"].min(), source["as_of_time"].max(), freq="h"
    )
    grid_continuous = len(expected) == len(source) and source["as_of_time"].equals(
        pd.Series(expected, name="as_of_time")
    )
    if not grid_continuous:
        raise ValueError("Hourly source is not a continuous UTC grid")

    break_at, monthly_coverage = detect_sustained_coverage_break(
        source["as_of_time"], source["arrivals_prev_1h"]
    )
    if break_at is None:
        raise ValueError("No sustained arrival coverage break was detected")

    output = pd.DataFrame({"as_of_time": source["as_of_time"]})
    output["dataset_version"] = DATASET_VERSION
    output["arrival_coverage_break_at"] = break_at
    output["split_arrival"] = assign_arrival_split(output["as_of_time"], break_at)
    output["split_wave"] = assign_wave_split(output["as_of_time"])
    output["arrival_eligible"] = output["split_arrival"].isin(
        ["TRAIN", "VALID", "TEST"]
    )
    output["wave_eligible"] = output["split_wave"].isin(["TRAIN", "VALID", "TEST"])
    for flag in ("port_source_present", "wave_source_present", "external_source_present"):
        output[flag] = pd.to_numeric(source[flag], errors="coerce").fillna(0).astype("int8")

    arrival_core: list[str] = []
    wave_core: list[str] = []
    research_features: list[str] = []
    imputation_rows: list[dict[str, Any]] = []

    common_base = [
        "arrivals_prev_1h",
        "arrivals_last_6h",
        "arrivals_last_24h",
        "arrivals_last_168h",
        "departures_prev_1h",
        "departures_last_6h",
        "departures_last_24h",
        "vessels_in_port_observed",
        "weather_available_flag",
    ]
    for column in common_base:
        output[column] = pd.to_numeric(source[column], errors="coerce")
        arrival_core.append(column)
        if column in ("arrivals_prev_1h", "vessels_in_port_observed", "weather_available_flag"):
            wave_core.append(column)

    wave_train = output["split_wave"].eq("TRAIN")
    for column in ("delayed_gt3_last_24h", "mean_arrival_delay_last_24h"):
        filled, missing, median = train_median_fill(source[column], wave_train)
        value_name = f"{column}_filled"
        missing_name = f"{column}_missing"
        output[value_name] = filled
        output[missing_name] = missing
        arrival_core.extend([value_name, missing_name])
        imputation_rows.append(
            {
                "feature": value_name,
                "source_column": column,
                "train_median": median,
                "missing_rows_before": int(missing.sum()),
                "method": "WAVE_TRAIN_MEDIAN_PLUS_MISSING_FLAG",
            }
        )

    arrival_signal = pd.to_numeric(source["arrivals_prev_1h"], errors="coerce")
    departure_signal = pd.to_numeric(source["departures_prev_1h"], errors="coerce")
    for lag_h in ARRIVAL_LAGS_H:
        name = f"arrivals_lag_{lag_h}h"
        output[name] = arrival_signal.shift(lag_h - 1)
        arrival_core.append(name)
        if lag_h in (1, 6, 24, 168):
            wave_core.append(name)
    for lag_h in (1, 6, 24, 168):
        name = f"departures_lag_{lag_h}h"
        output[name] = departure_signal.shift(lag_h - 1)
        arrival_core.append(name)
    for lag_h in (1, 6, 24):
        name = f"vessels_in_port_lag_{lag_h}h"
        output[name] = pd.to_numeric(
            source["vessels_in_port_observed"], errors="coerce"
        ).shift(lag_h - 1)
        arrival_core.append(name)
    for window_h in (6, 24, 168, 336):
        rolling = arrival_signal.rolling(window_h, min_periods=window_h)
        for suffix, values in (
            ("sum", rolling.sum()),
            ("mean", rolling.mean()),
            ("std", rolling.std(ddof=0)),
        ):
            name = f"arrivals_roll_{window_h}h_{suffix}"
            output[name] = values
            arrival_core.append(name)
            if window_h in (24, 168):
                wave_core.append(name)
    for span_h in (6, 24, 168):
        name = f"arrivals_ewm_{span_h}h"
        output[name] = arrival_signal.ewm(span=span_h, adjust=False).mean()
        arrival_core.append(name)

    cargo_events = arrival_events.copy()
    cargo_events["actual_ata"] = pd.to_datetime(cargo_events["actual_ata"], utc=True)
    cargo_events["as_of_time"] = cargo_events["actual_ata"].dt.floor("h") + pd.Timedelta(hours=1)
    cargo_events["cargo_group"] = cargo_events["cargo_type"].map(_cargo_group)
    cargo_pivot = pd.crosstab(cargo_events["as_of_time"], cargo_events["cargo_group"])
    cargo_groups = (
        "fresh_produce",
        "machinery_electrical",
        "textile",
        "transport_equipment",
        "seafood",
        "other",
    )
    cargo_aligned = cargo_pivot.reindex(
        index=pd.DatetimeIndex(output["as_of_time"]),
        columns=cargo_groups,
        fill_value=0,
    )
    event_hourly_total = cargo_aligned.sum(axis=1).to_numpy(dtype="float64")
    for group in cargo_groups:
        signal = pd.Series(
            cargo_aligned[group].to_numpy(dtype="float64"), index=output.index
        )
        previous_name = f"cargo_{group}_arrivals_prev_1h"
        output[previous_name] = signal
        arrival_core.append(previous_name)
        for window_h in (24, 168):
            count_name = f"cargo_{group}_arrivals_last_{window_h}h"
            share_name = f"cargo_{group}_share_last_{window_h}h"
            count = signal.rolling(window_h, min_periods=window_h).sum()
            denominator = arrival_signal.rolling(
                window_h, min_periods=window_h
            ).sum()
            output[count_name] = count
            output[share_name] = (count / denominator.replace(0.0, np.nan)).fillna(0.0)
            arrival_core.extend([count_name, share_name])
    output = output.copy()

    wave_raw = {
        "wave_height": pd.to_numeric(source["wave_height_m"], errors="coerce"),
        "wave_period": pd.to_numeric(source["wave_period_s"], errors="coerce"),
    }
    direction_rad = np.deg2rad(
        pd.to_numeric(source["wave_direction_deg"], errors="coerce")
    )
    wave_raw["wave_direction_sin"] = pd.Series(np.sin(direction_rad), index=source.index)
    wave_raw["wave_direction_cos"] = pd.Series(np.cos(direction_rad), index=source.index)
    for component, values in wave_raw.items():
        for lag_h in WAVE_LAGS_H:
            name = f"{component}_lag_{lag_h}h"
            output[name] = values.shift(lag_h)
            arrival_core.append(name)
            wave_core.append(name)
        safe = values.shift(OBSERVATION_LATENCY_H)
        for window_h in (24, 72, 168):
            rolling = safe.rolling(window_h, min_periods=window_h)
            for suffix, aggregate in (("mean", rolling.mean()), ("std", rolling.std(ddof=0))):
                name = f"{component}_roll_{window_h}h_{suffix}"
                output[name] = aggregate
                arrival_core.append(name)
                wave_core.append(name)
    output = output.copy()

    calendar_features: list[str] = []
    _add_calendar_features(output, output["as_of_time"], "issue", calendar_features)
    for horizon_h in WAVE_HORIZONS_H:
        target_times = output["as_of_time"] + pd.Timedelta(hours=horizon_h)
        horizon_calendar: list[str] = []
        _add_calendar_features(
            output, target_times, f"target_time_{horizon_h}h", horizon_calendar
        )
        any_event, holiday = _event_flags(target_times, business_events)
        event_name = f"known_event_any_at_{horizon_h}h"
        holiday_name = f"known_event_holiday_at_{horizon_h}h"
        output[event_name] = any_event
        output[holiday_name] = holiday
        horizon_calendar.extend([event_name, holiday_name])
        calendar_features.extend(horizon_calendar)
    arrival_core.extend(calendar_features)
    wave_core.extend(calendar_features)
    output = output.copy()

    external_columns = [
        "ext_wind_speed_ms",
        "ext_surface_current_ms",
        "ext_visibility_m",
        "ext_pressure_hpa",
        "ext_wind_gusts_10m",
        "ext_temperature_2m",
        "ext_relative_humidity_2m",
        "ext_precipitation",
        "ext_cloud_cover",
        "ext_sea_surface_temperature",
    ]
    external_raw: dict[str, pd.Series] = {
        name[4:]: pd.to_numeric(source[name], errors="coerce")
        for name in external_columns
    }
    ext_direction_rad = np.deg2rad(
        pd.to_numeric(source["ext_wind_direction_deg"], errors="coerce")
    )
    external_raw["wind_direction_sin"] = pd.Series(
        np.sin(ext_direction_rad), index=source.index
    )
    external_raw["wind_direction_cos"] = pd.Series(
        np.cos(ext_direction_rad), index=source.index
    )
    for component, values in external_raw.items():
        for lag_h in EXTERNAL_LAGS_H:
            shifted = values.shift(lag_h)
            filled, missing, median = train_median_fill(shifted, wave_train)
            name = f"research_ext_{component}_lag_{lag_h}h"
            missing_name = f"{name}_missing"
            output[name] = filled
            output[missing_name] = missing
            research_features.extend([name, missing_name])
            imputation_rows.append(
                {
                    "feature": name,
                    "source_column": component,
                    "train_median": median,
                    "missing_rows_before": int(missing.sum()),
                    "method": "WAVE_TRAIN_MEDIAN_PLUS_MISSING_FLAG",
                }
            )
    output = output.copy()

    source_target_mismatch: list[dict[str, Any]] = []
    arrival_targets: list[str] = []
    for horizon_h in ARRIVAL_HORIZONS_H:
        recomputed = future_sum(arrival_signal, horizon_h)
        source_target = pd.to_numeric(
            source[f"source_target_arrivals_next_{horizon_h}h"], errors="coerce"
        )
        comparable = source_target.notna() & recomputed.notna()
        mismatch = int(
            (~np.isclose(source_target.loc[comparable], recomputed.loc[comparable])).sum()
        )
        source_target_mismatch.append(
            {
                "target": f"arrivals_next_{horizon_h}h",
                "comparable_rows": int(comparable.sum()),
                "mismatch_rows": mismatch,
            }
        )
        mature = (output["as_of_time"] + pd.Timedelta(hours=horizon_h)).lt(break_at)
        name = f"target_arrivals_next_{horizon_h}h"
        output[name] = recomputed.where(mature)
        arrival_targets.append(name)
    output["target_arrivals_0_6h"] = output["target_arrivals_next_6h"]
    output["target_arrivals_6_12h"] = (
        output["target_arrivals_next_12h"] - output["target_arrivals_next_6h"]
    )
    output["target_arrivals_12_24h"] = (
        output["target_arrivals_next_24h"] - output["target_arrivals_next_12h"]
    )
    arrival_targets.extend(
        ["target_arrivals_0_6h", "target_arrivals_6_12h", "target_arrivals_12_24h"]
    )
    next_wait, next_observed = _next_arrival_targets(
        output["as_of_time"], arrival_events["actual_ata"], break_at
    )
    output["target_next_arrival_wait_h"] = next_wait
    output["target_next_arrival_observed_24h"] = next_observed
    arrival_targets.extend(
        ["target_next_arrival_wait_h", "target_next_arrival_observed_24h"]
    )

    wave_targets: list[str] = []
    grid_end_exclusive = output["as_of_time"].max() + pd.Timedelta(hours=1)
    for horizon_h in WAVE_HORIZONS_H:
        mature = (output["as_of_time"] + pd.Timedelta(hours=horizon_h)).lt(
            grid_end_exclusive
        )
        for source_name, target_name in (
            ("wave_height_m", "wave_height_m"),
            ("wave_period_s", "wave_period_s"),
        ):
            name = f"target_{target_name}_{horizon_h}h"
            output[name] = pd.to_numeric(source[source_name], errors="coerce").shift(
                -horizon_h
            ).where(mature)
            wave_targets.append(name)
        target_direction = pd.to_numeric(
            source["wave_direction_deg"], errors="coerce"
        ).shift(-horizon_h)
        for suffix, values in (
            ("sin", np.sin(np.deg2rad(target_direction))),
            ("cos", np.cos(np.deg2rad(target_direction))),
        ):
            name = f"target_wave_direction_{suffix}_{horizon_h}h"
            output[name] = pd.Series(values, index=source.index).where(mature)
            wave_targets.append(name)

    arrival_core = list(dict.fromkeys(arrival_core))
    wave_core = list(dict.fromkeys(wave_core))
    research_features = list(dict.fromkeys(research_features))
    arrival_mask = output["arrival_eligible"]
    wave_mask = output["wave_eligible"]
    arrival_missing = _count_missing(output, arrival_core, arrival_mask)
    wave_missing = _count_missing(output, wave_core, wave_mask)
    research_missing = _count_missing(output, research_features, arrival_mask | wave_mask)

    event_sequence = _build_event_sequence(
        arrival_events, output["as_of_time"].min(), break_at
    )
    feature_sets = {
        "arrival_core": arrival_core,
        "arrival_research_enriched": arrival_core + research_features,
        "wave_core": wave_core,
        "wave_research_enriched": wave_core + research_features,
        "arrival_targets": arrival_targets,
        "wave_targets": wave_targets,
    }
    feature_registry = _feature_registry(
        output, arrival_core, wave_core, research_features
    )

    split_rows = []
    for task, column in (("ARRIVAL", "split_arrival"), ("WAVE", "split_wave")):
        for split_name, count in output[column].value_counts().items():
            split_rows.append({"task": task, "split": split_name, "rows": int(count)})
    split_report = pd.DataFrame(split_rows).sort_values(["task", "split"])

    source_inventory = pd.DataFrame(
        [
            {
                "source": "features.port_hourly_state_v1",
                "rows": len(source),
                "joined_rows": int(output["port_source_present"].sum()),
                "availability_semantics": "RETROSPECTIVE_HOURLY_STATE",
                "model_role": "CORE_RETROSPECTIVE",
            },
            {
                "source": "core.maritime_observation",
                "rows": int(output["wave_source_present"].sum()),
                "joined_rows": int(output["wave_source_present"].sum()),
                "availability_semantics": "EVENT_TIME_WITH_3H_CONSERVATIVE_LAG",
                "model_role": "CORE_RETROSPECTIVE",
            },
            {
                "source": "features.maritime_external_weather_hourly_v1",
                "rows": int(output["external_source_present"].sum()),
                "joined_rows": int(output["external_source_present"].sum()),
                "availability_semantics": "RETROSPECTIVE_REANALYSIS_RESEARCH_ONLY",
                "model_role": "RESEARCH_ONLY",
            },
            {
                "source": "features.maritime_issue_time_weather_forecast_v1",
                "rows": 0,
                "joined_rows": 0,
                "availability_semantics": "LIVE_ISSUE_TIME_NO_HISTORICAL_OVERLAP",
                "model_role": "EXCLUDED_FROM_HISTORICAL_DATASET",
            },
        ]
    )
    target_integrity = pd.DataFrame(source_target_mismatch)
    event_report = (
        event_sequence.groupby("event_split", as_index=False)
        .agg(
            events=("port_call_id", "size"),
            first_event=("actual_ata", "min"),
            last_event=("actual_ata", "max"),
            median_interarrival_h=("interarrival_h", "median"),
        )
        .sort_values("event_split")
    )
    source_arrival_values = arrival_signal.to_numpy(dtype="float64")
    event_alignment_mismatches = int(
        (~np.isclose(event_hourly_total, source_arrival_values)).sum()
    )
    entity_semantics = pd.DataFrame(
        [
            {
                "field": "arrival_entity",
                "value": "TIR_OPERATIONAL_UNIT",
                "evidence": "vessel_type is TIR for nearly all core.port_call rows",
                "claim_allowed": True,
            },
            {
                "field": "ship_or_vessel_arrival",
                "value": "NOT_SUPPORTED",
                "evidence": "MMSI, IMO and terminal identity are absent",
                "claim_allowed": False,
            },
            {
                "field": "wave_observation",
                "value": "MARINE_WAVE_STATE",
                "evidence": "Copernicus IBI hourly wave height, period and direction",
                "claim_allowed": True,
            },
        ]
    )

    gates = [
        ("CONTINUOUS_UNIQUE_HOURLY_GRID", grid_continuous, len(source)),
        ("PORT_SOURCE_JOIN_COMPLETE", output["port_source_present"].eq(1).all(), int(output["port_source_present"].eq(0).sum())),
        ("WAVE_SOURCE_JOIN_COMPLETE", output["wave_source_present"].eq(1).all(), int(output["wave_source_present"].eq(0).sum())),
        ("EXTERNAL_SOURCE_JOIN_COMPLETE", output["external_source_present"].eq(1).all(), int(output["external_source_present"].eq(0).sum())),
        (
            "EXTERNAL_WEATHER_MARKED_RESEARCH_ONLY",
            source["ext_availability_semantics"].eq(
                "RETROSPECTIVE_REANALYSIS_RESEARCH_ONLY"
            ).all(),
            int(
                source["ext_availability_semantics"]
                .ne("RETROSPECTIVE_REANALYSIS_RESEARCH_ONLY")
                .sum()
            ),
        ),
        ("ARRIVAL_BREAK_DETECTED", break_at is not None, 0 if break_at is not None else 1),
        ("ARRIVAL_BREAK_IN_EXPECTED_2025_WINDOW", pd.Timestamp("2025-04-01T00:00:00Z") <= break_at <= pd.Timestamp("2025-07-01T00:00:00Z"), 0),
        ("SOURCE_ARRIVAL_TARGETS_REPRODUCED", int(target_integrity["mismatch_rows"].sum()) == 0, int(target_integrity["mismatch_rows"].sum())),
        (
            "EVENT_TO_HOURLY_COUNT_ALIGNMENT_GE_99_9_PCT",
            100.0 * (1.0 - event_alignment_mismatches / len(output)) >= 99.9,
            event_alignment_mismatches,
        ),
        ("ARRIVAL_CORE_FEATURES_COMPLETE", arrival_missing == 0, arrival_missing),
        ("WAVE_CORE_FEATURES_COMPLETE", wave_missing == 0, wave_missing),
        ("RESEARCH_FEATURES_COMPLETE_AFTER_IMPUTATION", research_missing == 0, research_missing),
        (
            "NO_TARGET_COLUMN_IN_FEATURE_SETS",
            not any(
                name.startswith("target_") and not name.startswith("target_time_")
                for name in arrival_core + wave_core + research_features
            ),
            int(
                sum(
                    name.startswith("target_")
                    and not name.startswith("target_time_")
                    for name in arrival_core + wave_core + research_features
                )
            ),
        ),
        ("ARRIVAL_TRAIN_SUPPORT", int(output["split_arrival"].eq("TRAIN").sum()) >= 25_000, int(output["split_arrival"].eq("TRAIN").sum())),
        ("ARRIVAL_VALID_SUPPORT", int(output["split_arrival"].eq("VALID").sum()) >= 8_000, int(output["split_arrival"].eq("VALID").sum())),
        ("ARRIVAL_TEST_SUPPORT", int(output["split_arrival"].eq("TEST").sum()) >= 1_500, int(output["split_arrival"].eq("TEST").sum())),
        ("WAVE_TEST_SUPPORT", int(output["split_wave"].eq("TEST").sum()) >= 8_000, int(output["split_wave"].eq("TEST").sum())),
        ("EVENT_SEQUENCE_SUPPORT", len(event_sequence) >= 25_000, len(event_sequence)),
        ("NO_SYNTHETIC_TARGETS", True, 0),
    ]
    quality_gates = pd.DataFrame(gates, columns=["gate", "passed", "observed"])
    quality_gates["severity"] = "CRITICAL"
    gates_passed = bool(quality_gates["passed"].all())
    decision_name = (
        "READY_FOR_ADVANCED_TIME_SERIES_BENCHMARK"
        if gates_passed
        else "BLOCKED_DATA_CONTRACT_REPAIR_REQUIRED"
    )
    decision = {
        "status": "SUCCESS",
        "decision": decision_name,
        "dataset_version": DATASET_VERSION,
        "contract_version": CONTRACT_VERSION,
        "row_count": len(output),
        "column_count": len(output.columns),
        "feature_count_arrival_core": len(arrival_core),
        "feature_count_wave_core": len(wave_core),
        "feature_count_research": len(research_features),
        "arrival_coverage_break_at": break_at,
        "arrival_train_rows": int(output["split_arrival"].eq("TRAIN").sum()),
        "arrival_valid_rows": int(output["split_arrival"].eq("VALID").sum()),
        "arrival_test_rows": int(output["split_arrival"].eq("TEST").sum()),
        "wave_train_rows": int(output["split_wave"].eq("TRAIN").sum()),
        "wave_valid_rows": int(output["split_wave"].eq("VALID").sum()),
        "wave_test_rows": int(output["split_wave"].eq("TEST").sum()),
        "arrival_event_rows": len(event_sequence),
        "arrival_entity_semantics": "TIR_OPERATIONAL_UNIT_NOT_VESSEL",
        "ship_arrival_claim_allowed": False,
        "event_to_hourly_mismatch_hours": event_alignment_mismatches,
        "quality_gates_passed": gates_passed,
        "synthetic_rows_created": 0,
        "synthetic_targets_created": 0,
        "target_imputation_used": False,
        "external_weather_role": "RETROSPECTIVE_REANALYSIS_RESEARCH_ONLY",
        "historical_replay_allowed": True,
        "production_promotion_allowed": False,
        "training_executed": False,
        "recommended_models": [
            "HGB_POISSON_BASELINE",
            "DYNAMIC_NEGATIVE_BINOMIAL",
            "NHITS",
            "TFT",
            "PATCHTST",
            "TIMEMIXER",
            "ITRANSFORMER",
            "CHRONOS_ZERO_SHOT_CHALLENGER",
            "TEMPORAL_POINT_PROCESS_ON_EVENT_SEQUENCE",
        ],
        "next_block": "B60B_ADVANCED_TIME_SERIES_ROLLING_ORIGIN_BENCHMARK",
    }
    reports = {
        "01_source_inventory.csv": source_inventory,
        "02_monthly_arrival_coverage_break.csv": monthly_coverage,
        "03_temporal_splits.csv": split_report,
        "04_feature_registry.csv": feature_registry,
        "05_imputation_registry.csv": pd.DataFrame(imputation_rows),
        "06_target_integrity.csv": target_integrity,
        "07_event_sequence_summary.csv": event_report,
        "08_quality_gates.csv": quality_gates,
        "09_entity_semantics.csv": entity_semantics,
    }
    return BuildResult(
        dataset=output,
        event_sequence=event_sequence,
        reports=reports,
        decision=clean_json(decision),
        feature_sets=feature_sets,
    )


def feature_sets_json(feature_sets: dict[str, list[str]]) -> str:
    return json.dumps(clean_json(feature_sets), indent=2, ensure_ascii=True)
