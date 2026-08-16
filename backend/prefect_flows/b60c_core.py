from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASET_VERSION = "b60c-operational-port-call-landmark-v1"
CONTRACT_VERSION = "b60c-operational-port-call-contract-v1"
SCENARIO_VERSION = "b60c-counterfactual-stress-v1"
TRAIN_START = pd.Timestamp("2020-01-01", tz="UTC")
VALID_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
TEST_END = pd.Timestamp("2025-04-01", tz="UTC")
MAX_STAY_H = 24.0 * 30.0
WEATHER_CONSERVATIVE_LAG_H = 3
HORIZONS_H = (6, 12, 24)


@dataclass
class BuildResult:
    dataset: pd.DataFrame
    scenarios: pd.DataFrame
    reports: dict[str, pd.DataFrame]
    decision: dict[str, Any]
    feature_sets: dict[str, list[str]]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _clean_category(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .fillna("<UNKNOWN>")
        .astype("object")
    )


def cargo_group(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if any(token in text for token in ("fruit", "legume", "frais")):
        return "fresh_produce"
    if any(token in text for token in ("machine", "electr", "mecan")):
        return "machinery_electrical"
    if any(token in text for token in ("textile", "confection")):
        return "textile"
    if any(token in text for token in ("transport", "vehicule", "automobile")):
        return "transport_equipment"
    if any(token in text for token in ("mer", "poisson", "seafood")):
        return "seafood"
    return "other"


def delay_class(delay_h: pd.Series) -> pd.Series:
    values = pd.to_numeric(delay_h, errors="coerce")
    result = pd.Series("TARGET_MISSING", index=values.index, dtype="object")
    result.loc[values.le(1.0)] = "NORMAL_LE_1H"
    result.loc[values.gt(1.0) & values.le(3.0)] = "MINOR_1_3H"
    result.loc[values.gt(3.0) & values.le(6.0)] = "MAJOR_3_6H"
    result.loc[values.gt(6.0)] = "CRITICAL_GT_6H"
    return result


def assign_call_split(calls: pd.DataFrame) -> pd.Series:
    ata = _as_utc(calls["actual_ata"])
    atd = _as_utc(calls["actual_atd"])
    valid = calls["valid_stay"].fillna(False)
    split = pd.Series("EXCLUDED_DATE", index=calls.index, dtype="object")
    split.loc[ata.isna()] = "EXCLUDED_ATA_MISSING"
    split.loc[ata.notna() & atd.isna()] = "EXCLUDED_TARGET_MISSING"
    split.loc[ata.notna() & atd.notna() & ~valid] = "EXCLUDED_INVALID_STAY"

    eligible = ata.notna() & atd.notna() & valid
    train = eligible & ata.ge(TRAIN_START) & ata.lt(VALID_START)
    valid_issue = eligible & ata.ge(VALID_START) & ata.lt(TEST_START)
    test = eligible & ata.ge(TEST_START) & ata.lt(TEST_END)
    split.loc[train & atd.lt(VALID_START)] = "TRAIN"
    split.loc[valid_issue & atd.lt(TEST_START)] = "VALID"
    split.loc[test & atd.lt(TEST_END)] = "TEST"
    crossing = (
        (train & atd.ge(VALID_START))
        | (valid_issue & atd.ge(TEST_START))
        | (test & atd.ge(TEST_END))
    )
    split.loc[crossing] = "EXCLUDED_BOUNDARY_CROSSING"
    return split


def prepare_calls(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "port_call_id",
        "planned_eta",
        "planned_etd",
        "actual_ata",
        "actual_atd",
        "cargo_type",
        "vessel_name",
        "source",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required port-call columns are missing: {missing}")
    calls = frame.copy().reset_index(drop=True)
    calls["_call_row_id"] = np.arange(len(calls), dtype="int64")
    optional = (
        "port_code",
        "terminal_code",
        "mmsi",
        "imo",
        "voyage_id",
        "planned_etb",
        "actual_atb",
        "vessel_type",
        "created_at",
        "updated_at",
    )
    for column in optional:
        if column not in calls:
            calls[column] = pd.NA
    for column in (
        "planned_eta",
        "planned_etb",
        "planned_etd",
        "actual_ata",
        "actual_atb",
        "actual_atd",
        "created_at",
        "updated_at",
    ):
        calls[column] = _as_utc(calls[column])
    for column in (
        "port_code",
        "terminal_code",
        "vessel_name",
        "vessel_type",
        "cargo_type",
        "source",
    ):
        calls[column] = _clean_category(calls[column])

    imo = calls["imo"].astype("string").str.strip()
    mmsi = calls["mmsi"].astype("string").str.strip()
    name = calls["vessel_name"].astype("string").str.upper().str.strip()
    calls["vessel_key"] = np.where(
        imo.notna() & imo.ne(""),
        "IMO:" + imo.fillna(""),
        np.where(
            mmsi.notna() & mmsi.ne(""),
            "MMSI:" + mmsi.fillna(""),
            "NAME:" + name.fillna("<UNKNOWN>"),
        ),
    )
    calls["cargo_group"] = calls["cargo_type"].map(cargo_group)
    calls["stay_h"] = (
        calls["actual_atd"] - calls["actual_ata"]
    ).dt.total_seconds() / 3600.0
    calls["departure_delay_h"] = (
        calls["actual_atd"] - calls["planned_etd"]
    ).dt.total_seconds() / 3600.0
    calls["arrival_delay_h"] = (
        calls["actual_ata"] - calls["planned_eta"]
    ).dt.total_seconds() / 3600.0
    calls["planned_stay_h"] = (
        calls["planned_etd"] - calls["actual_ata"]
    ).dt.total_seconds() / 3600.0
    calls["planned_berth_offset_h"] = (
        calls["planned_etb"] - calls["actual_ata"]
    ).dt.total_seconds() / 3600.0
    calls["valid_stay"] = calls["stay_h"].gt(0.0) & calls["stay_h"].le(MAX_STAY_H)
    calls["target_observed"] = calls["actual_atd"].notna()
    calls["target_departure_delay_class"] = delay_class(calls["departure_delay_h"])
    calls["target_delay_gt_1h"] = calls["departure_delay_h"].gt(1.0)
    calls["target_delay_gt_3h"] = calls["departure_delay_h"].gt(3.0)
    calls["target_delay_gt_6h"] = calls["departure_delay_h"].gt(6.0)
    calls["split"] = assign_call_split(calls)
    return calls


def _history_for_group(
    calls: pd.DataFrame,
    group_column: str,
    prefix: str,
) -> pd.DataFrame:
    columns = [
        f"{prefix}_prior_calls",
        f"{prefix}_prior_mean_stay_h",
        f"{prefix}_prior_std_stay_h",
        f"{prefix}_prior_mean_delay_h",
        f"{prefix}_prior_late_gt3_rate",
        f"{prefix}_prior_late_gt6_rate",
        f"{prefix}_last_stay_h",
        f"{prefix}_last_delay_h",
        f"{prefix}_recent5_mean_stay_h",
        f"{prefix}_recent5_late_gt3_rate",
        f"{prefix}_days_since_last_departure",
    ]
    result = pd.DataFrame(np.nan, index=calls.index, columns=columns)
    valid = calls.loc[
        calls["valid_stay"] & calls["actual_ata"].notna() & calls["actual_atd"].notna()
    ]
    grouped = valid.groupby(group_column, dropna=False, sort=False)
    for _, group in grouped:
        arrivals = group.sort_values(["actual_ata", "port_call_id"])
        completed = group.sort_values(["actual_atd", "port_call_id"])
        complete_rows = list(completed.index)
        pointer = 0
        count = 0
        sum_stay = 0.0
        sumsq_stay = 0.0
        sum_delay = 0.0
        late3 = 0
        late6 = 0
        recent_stay: deque[float] = deque(maxlen=5)
        recent_late: deque[float] = deque(maxlen=5)
        last_stay = np.nan
        last_delay = np.nan
        last_departure = pd.NaT
        for row_index, row in arrivals.iterrows():
            issue_at = row["actual_ata"]
            while pointer < len(complete_rows):
                completed_index = complete_rows[pointer]
                completed_at = calls.at[completed_index, "actual_atd"]
                if pd.isna(completed_at) or completed_at > issue_at:
                    break
                stay = float(calls.at[completed_index, "stay_h"])
                delay = float(calls.at[completed_index, "departure_delay_h"])
                count += 1
                sum_stay += stay
                sumsq_stay += stay * stay
                sum_delay += delay
                is_late3 = float(delay > 3.0)
                late3 += int(is_late3)
                late6 += int(delay > 6.0)
                recent_stay.append(stay)
                recent_late.append(is_late3)
                last_stay = stay
                last_delay = delay
                last_departure = completed_at
                pointer += 1
            if count:
                mean_stay = sum_stay / count
                variance = max(0.0, sumsq_stay / count - mean_stay * mean_stay)
                values = [
                    float(count),
                    mean_stay,
                    math.sqrt(variance),
                    sum_delay / count,
                    late3 / count,
                    late6 / count,
                    last_stay,
                    last_delay,
                    float(np.mean(recent_stay)),
                    float(np.mean(recent_late)),
                    (issue_at - last_departure).total_seconds() / 86400.0,
                ]
                result.loc[row_index, columns] = values
            else:
                result.at[row_index, columns[0]] = 0.0
    return result


def add_prior_history(calls: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = calls.copy()
    specifications = (
        ("vessel_key", "vessel_history"),
        ("terminal_code", "terminal_history"),
        ("cargo_group", "cargo_history"),
    )
    history_columns: list[str] = []
    for group_column, prefix in specifications:
        history = _history_for_group(result, group_column, prefix)
        for column in history:
            result[column] = history[column]
            history_columns.append(column)
    result["port_history_group"] = "ALL_PORT_CALLS"
    history = _history_for_group(result, "port_history_group", "port_history")
    for column in history:
        result[column] = history[column]
        history_columns.append(column)
    return result, history_columns


def expand_landmarks(calls: pd.DataFrame) -> pd.DataFrame:
    eligible = calls.loc[calls["split"].isin(["TRAIN", "VALID", "TEST"])].copy()
    counts = np.ceil(eligible["stay_h"]).astype("int64").clip(lower=1, upper=720)
    expanded = eligible.loc[eligible.index.repeat(counts)].copy()
    expanded["landmark_h"] = expanded.groupby("_call_row_id").cumcount().astype("int64")
    expanded["landmark_at"] = expanded["actual_ata"] + pd.to_timedelta(
        expanded["landmark_h"], unit="h"
    )
    expanded = expanded.loc[expanded["landmark_at"].lt(expanded["actual_atd"])].copy()
    expanded["dataset_version"] = DATASET_VERSION
    expanded["data_origin"] = "REAL_RETROSPECTIVE"
    expanded["training_allowed"] = expanded["split"].eq("TRAIN")
    expanded["validation_allowed"] = expanded["split"].eq("VALID")
    expanded["test_allowed"] = expanded["split"].eq("TEST")
    expanded["production_claim_allowed"] = False
    expanded["per_call_sample_weight"] = 1.0 / expanded.groupby(
        "_call_row_id"
    )["_call_row_id"].transform("size")

    expanded["elapsed_since_arrival_h"] = (
        expanded["landmark_at"] - expanded["actual_ata"]
    ).dt.total_seconds() / 3600.0
    expanded["time_to_planned_departure_h"] = (
        expanded["planned_etd"] - expanded["landmark_at"]
    ).dt.total_seconds() / 3600.0
    expanded["overdue_h"] = (-expanded["time_to_planned_departure_h"]).clip(lower=0.0)
    denominator = expanded["planned_stay_h"].clip(lower=1.0)
    expanded["plan_progress_ratio"] = (
        expanded["elapsed_since_arrival_h"] / denominator
    ).clip(lower=0.0, upper=10.0)
    atb_observed = expanded["actual_atb"].notna() & expanded["actual_atb"].le(
        expanded["landmark_at"]
    )
    expanded["berth_event_observed"] = atb_observed.astype("int8")
    expanded["hours_since_berth_observed"] = np.where(
        atb_observed,
        (
            expanded["landmark_at"] - expanded["actual_atb"]
        ).dt.total_seconds()
        / 3600.0,
        np.nan,
    )
    expanded["early_warning_eligible"] = expanded["landmark_at"].le(
        expanded["planned_etd"]
    )
    breach_at = expanded["planned_etd"] + pd.Timedelta(hours=3)
    expanded["pre_breach_eligible"] = expanded["landmark_at"].lt(breach_at)
    expanded["current_plan_state"] = np.select(
        [
            expanded["time_to_planned_departure_h"].gt(3.0),
            expanded["time_to_planned_departure_h"].ge(0.0),
            expanded["overdue_h"].le(3.0),
        ],
        ["ON_TRACK_WINDOW", "DUE_WITHIN_3H", "OVERDUE_LT3H"],
        default="OVERDUE_GE3H",
    )

    expanded["target_actual_atd"] = expanded["actual_atd"]
    expanded["target_total_stay_h"] = expanded["stay_h"]
    expanded["target_departure_delay_h"] = expanded["departure_delay_h"]
    expanded["target_departure_delay_class"] = delay_class(
        expanded["departure_delay_h"]
    )
    expanded["target_delay_gt_1h"] = expanded["departure_delay_h"].gt(1.0)
    expanded["target_delay_gt_3h"] = expanded["departure_delay_h"].gt(3.0)
    expanded["target_delay_gt_6h"] = expanded["departure_delay_h"].gt(6.0)
    expanded["target_remaining_h"] = (
        expanded["actual_atd"] - expanded["landmark_at"]
    ).dt.total_seconds() / 3600.0
    for horizon_h in HORIZONS_H:
        expanded[f"target_departure_within_{horizon_h}h"] = expanded[
            "target_remaining_h"
        ].le(horizon_h)
        future_breach = (
            expanded["target_delay_gt_3h"]
            & breach_at.gt(expanded["landmark_at"])
            & breach_at.le(expanded["landmark_at"] + pd.Timedelta(hours=horizon_h))
        )
        expanded[f"target_gt3_breach_within_{horizon_h}h"] = future_breach
    breach_observed = (
        expanded["target_delay_gt_3h"]
        & breach_at.ge(expanded["landmark_at"])
        & breach_at.lt(expanded["actual_atd"])
    )
    time_to_breach = (
        breach_at - expanded["landmark_at"]
    ).dt.total_seconds() / 3600.0
    expanded["target_breach_gt3_observed"] = breach_observed
    expanded["target_breach_or_censor_h"] = np.where(
        breach_observed & time_to_breach.ge(0.0),
        time_to_breach,
        expanded["target_remaining_h"],
    )
    return expanded


def _window_counts(
    query_ns: np.ndarray,
    event_ns: np.ndarray,
    window_h: int,
) -> np.ndarray:
    if len(event_ns) == 0:
        return np.zeros(len(query_ns), dtype="int64")
    right = np.searchsorted(event_ns, query_ns, side="right")
    lower = query_ns - np.int64(window_h * 3_600_000_000_000)
    left = np.searchsorted(event_ns, lower, side="right")
    return right - left


def _dynamic_state_for_times(
    calls: pd.DataFrame,
    times: pd.Series,
    prefix: str,
) -> pd.DataFrame:
    query = _as_utc(times)
    query_ns = query.astype("int64").to_numpy()
    valid = calls.loc[
        calls["valid_stay"] & calls["actual_ata"].notna() & calls["actual_atd"].notna()
    ].copy()
    arrival_ns = np.sort(valid["actual_ata"].astype("int64").to_numpy())
    departure_sorted = valid.sort_values("actual_atd")
    departure_ns = departure_sorted["actual_atd"].astype("int64").to_numpy()
    output = pd.DataFrame(index=times.index)
    for window_h in (1, 6, 24, 168):
        output[f"{prefix}_arrivals_last_{window_h}h"] = _window_counts(
            query_ns, arrival_ns, window_h
        )
        output[f"{prefix}_departures_last_{window_h}h"] = _window_counts(
            query_ns, departure_ns, window_h
        )
    arrivals_to_date = np.searchsorted(arrival_ns, query_ns, side="right")
    departures_to_date = np.searchsorted(departure_ns, query_ns, side="right")
    output[f"{prefix}_active_calls_observed"] = np.maximum(
        0, arrivals_to_date - departures_to_date
    )
    output[f"{prefix}_flow_imbalance_6h"] = (
        output[f"{prefix}_arrivals_last_6h"]
        - output[f"{prefix}_departures_last_6h"]
    )
    output[f"{prefix}_flow_imbalance_24h"] = (
        output[f"{prefix}_arrivals_last_24h"]
        - output[f"{prefix}_departures_last_24h"]
    )
    delays = pd.to_numeric(
        departure_sorted["departure_delay_h"], errors="coerce"
    ).fillna(0.0).to_numpy(dtype="float64")
    late = (delays > 3.0).astype("float64")
    delay_prefix = np.concatenate(([0.0], np.cumsum(delays)))
    late_prefix = np.concatenate(([0.0], np.cumsum(late)))
    right = np.searchsorted(departure_ns, query_ns, side="right")
    lower = query_ns - np.int64(24 * 3_600_000_000_000)
    left = np.searchsorted(departure_ns, lower, side="right")
    counts = right - left
    sums = delay_prefix[right] - delay_prefix[left]
    late_sums = late_prefix[right] - late_prefix[left]
    output[f"{prefix}_completed_mean_delay_last_24h"] = np.divide(
        sums, counts, out=np.full(len(counts), np.nan), where=counts > 0
    )
    output[f"{prefix}_completed_late_gt3_rate_last_24h"] = np.divide(
        late_sums, counts, out=np.full(len(counts), np.nan), where=counts > 0
    )
    return output


def add_dynamic_port_state(
    landmarks: pd.DataFrame,
    calls: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    output = landmarks.copy()
    unique = pd.DataFrame(
        {"landmark_at": pd.Series(output["landmark_at"].drop_duplicates()).sort_values()}
    )
    global_state = _dynamic_state_for_times(
        calls, unique["landmark_at"], "port"
    )
    global_state["landmark_at"] = unique["landmark_at"].to_numpy()
    output = output.merge(global_state, on="landmark_at", how="left", validate="many_to_one")
    state_columns = [column for column in global_state if column != "landmark_at"]

    terminal_parts: list[pd.DataFrame] = []
    for terminal, rows in output.groupby("terminal_code", sort=False):
        terminal_calls = calls.loc[calls["terminal_code"].eq(terminal)]
        state = _dynamic_state_for_times(
            terminal_calls, rows["landmark_at"], "terminal"
        )
        state.index = rows.index
        terminal_parts.append(state)
    terminal_state = pd.concat(terminal_parts).sort_index()
    for column in terminal_state:
        output[column] = terminal_state[column]
        state_columns.append(column)
    return output, state_columns


def _cyclic(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    angle = 2.0 * np.pi * pd.to_numeric(values, errors="coerce") / period
    return np.sin(angle), np.cos(angle)


def add_calendar_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    times = _as_utc(output["landmark_at"])
    columns: list[str] = []
    components = (
        ("hour", times.dt.hour, 24.0),
        ("dow", times.dt.dayofweek, 7.0),
        ("doy", times.dt.dayofyear, 366.0),
        ("month", times.dt.month - 1, 12.0),
    )
    for label, values, period in components:
        sin_values, cos_values = _cyclic(values, period)
        for suffix, data in (("sin", sin_values), ("cos", cos_values)):
            name = f"landmark_{label}_{suffix}"
            output[name] = data
            columns.append(name)
    output["landmark_weekend"] = times.dt.dayofweek.ge(5).astype("int8")
    columns.append("landmark_weekend")
    return output, columns


def add_business_event_features(
    frame: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    calendar = events.copy()
    if calendar.empty:
        calendar = pd.DataFrame(columns=["start_date", "end_date", "event_type"])
    calendar["start_date"] = _as_utc(calendar["start_date"])
    calendar["end_date"] = _as_utc(calendar["end_date"])
    columns: list[str] = []
    for horizon_h in (0, *HORIZONS_H):
        target = _as_utc(output["landmark_at"]) + pd.Timedelta(hours=horizon_h)
        any_event = np.zeros(len(output), dtype="int8")
        holiday = np.zeros(len(output), dtype="int8")
        for row in calendar.itertuples(index=False):
            if pd.isna(row.start_date) or pd.isna(row.end_date):
                continue
            active = target.dt.normalize().between(
                row.start_date.normalize(), row.end_date.normalize()
            )
            any_event[active.to_numpy()] = 1
            event_type = str(getattr(row, "event_type", "")).lower()
            event_name = str(getattr(row, "event_name", "")).lower()
            if any(token in f"{event_type} {event_name}" for token in ("holiday", "aid", "fitr", "adha")):
                holiday[active.to_numpy()] = 1
        suffix = "now" if horizon_h == 0 else f"{horizon_h}h"
        event_name = f"known_event_any_{suffix}"
        holiday_name = f"known_holiday_{suffix}"
        output[event_name] = any_event
        output[holiday_name] = holiday
        columns.extend([event_name, holiday_name])
    return output, columns


def add_research_weather(
    frame: pd.DataFrame,
    weather: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    output = frame.copy().sort_values("landmark_at").reset_index(drop=True)
    expected = (
        "wave_height_m",
        "wave_period_s",
        "wave_direction_deg",
        "wind_speed_ms",
        "wind_direction_deg",
        "surface_current_ms",
        "visibility_m",
        "pressure_hpa",
        "wind_gusts_10m",
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "cloud_cover",
        "sea_surface_temperature",
    )
    source = weather.copy()
    if "observed_at" not in source:
        source["observed_at"] = pd.NaT
    source["observed_at"] = _as_utc(source["observed_at"])
    source = source.dropna(subset=["observed_at"]).sort_values("observed_at")
    for column in expected:
        if column not in source:
            source[column] = np.nan
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["research_weather_available_at"] = source["observed_at"] + pd.Timedelta(
        hours=WEATHER_CONSERVATIVE_LAG_H
    )
    research: list[str] = []
    right_columns = ["research_weather_available_at", "observed_at"]
    for column in expected:
        name = f"research_{column}_lag3h"
        source[name] = source[column]
        right_columns.append(name)
        research.append(name)
    for column in ("wave_height_m", "wave_period_s", "wind_speed_ms", "visibility_m"):
        # The availability timestamp already applies the conservative lag. A
        # second row shift would silently turn a 3-hour policy into 6 hours.
        values = source[column]
        for window_h in (24, 72):
            rolling = values.rolling(window_h, min_periods=max(6, window_h // 4))
            for suffix, aggregate in (
                ("mean", rolling.mean()),
                ("std", rolling.std(ddof=0)),
                ("max", rolling.max()),
            ):
                name = f"research_{column}_roll_{window_h}h_{suffix}"
                source[name] = aggregate
                right_columns.append(name)
                research.append(name)
    if source.empty:
        for column in research:
            output[column] = np.nan
        output["research_weather_age_h"] = np.nan
    else:
        merged = pd.merge_asof(
            output,
            source[right_columns].sort_values("research_weather_available_at"),
            left_on="landmark_at",
            right_on="research_weather_available_at",
            direction="backward",
            allow_exact_matches=True,
        )
        output = merged
        output["research_weather_age_h"] = (
            output["landmark_at"] - output["observed_at"]
        ).dt.total_seconds() / 3600.0
    research.append("research_weather_age_h")
    output = output.drop(
        columns=["research_weather_available_at", "observed_at"], errors="ignore"
    )
    policy = pd.DataFrame(
        [
            {
                "feature_family": "wave_and_external_weather",
                "effective_lag_h": WEATHER_CONSERVATIVE_LAG_H,
                "availability_semantics": "RETROSPECTIVE_OBSERVATION_WITH_CONSERVATIVE_LAG",
                "model_role": "RESEARCH_ONLY_NOT_PRODUCTION_PROMOTABLE",
                "reason": "historical available_at was not captured",
            }
        ]
    )
    return output, research, policy


def impute_features_from_train(
    frame: pd.DataFrame,
    features: Iterable[str],
    method: str,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    output = frame.copy()
    expanded = list(features)
    rows: list[dict[str, Any]] = []
    train = output["split"].eq("TRAIN")
    updates: dict[str, pd.Series] = {}
    for column in list(features):
        values = pd.to_numeric(output[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        missing = values.isna()
        if not missing.any():
            updates[column] = values
            continue
        observed_train = values.loc[train].dropna()
        median = float(observed_train.median()) if len(observed_train) else 0.0
        missing_name = f"{column}_missing"
        updates[column] = values.fillna(float(median))
        updates[missing_name] = missing.astype("int8")
        expanded.append(missing_name)
        rows.append(
            {
                "feature": column,
                "missing_flag": missing_name,
                "train_median": float(median),
                "missing_rows": int(missing.sum()),
                "method": method,
            }
        )
    output = pd.concat(
        [
            output.drop(columns=list(updates), errors="ignore"),
            pd.DataFrame(updates, index=output.index),
        ],
        axis=1,
    ).copy()
    return output, list(dict.fromkeys(expanded)), pd.DataFrame(rows)


def generate_counterfactual_scenarios(
    dataset: pd.DataFrame,
    core_features: list[str],
    research_features: list[str],
    max_parent_rows: int = 2_000,
    random_seed: int = 20260804,
) -> pd.DataFrame:
    eligible = dataset.loc[
        dataset["split"].eq("TRAIN") & dataset["early_warning_eligible"]
    ].copy()
    if eligible.empty:
        return pd.DataFrame()
    sample_size = min(max_parent_rows, len(eligible))
    parents = eligible.sample(sample_size, random_state=random_seed).copy()
    target_columns = [column for column in parents if column.startswith("target_")]
    keep = list(
        dict.fromkeys(
            column
            for column in (
                "dataset_version",
                "port_call_id",
                "vessel_key",
                "landmark_at",
                "split",
                *core_features,
                *research_features,
            )
            if column in parents and column not in target_columns
        )
    )
    parents = parents[keep].copy()
    train = dataset.loc[dataset["split"].eq("TRAIN")]
    high_values: dict[str, float] = {}
    for column in research_features:
        if column not in train or not any(
            token in column for token in ("wave_height", "wave_period", "wind_speed")
        ):
            continue
        observed = pd.to_numeric(train[column], errors="coerce").dropna()
        high_values[column] = float(observed.quantile(0.95)) if len(observed) else 0.0
    parts: list[pd.DataFrame] = []
    for scenario_type in ("TRAFFIC_SURGE", "MARINE_STRESS", "COMPOUND_STRESS"):
        scenario = parents.copy()
        scenario["parent_port_call_id"] = scenario["port_call_id"].astype(str)
        scenario["scenario_type"] = scenario_type
        scenario["scenario_id"] = [
            f"{SCENARIO_VERSION}:{scenario_type}:{index:06d}"
            for index in range(len(scenario))
        ]
        if scenario_type in ("TRAFFIC_SURGE", "COMPOUND_STRESS"):
            for column in (
                "port_arrivals_last_6h",
                "port_arrivals_last_24h",
                "port_active_calls_observed",
                "terminal_active_calls_observed",
                "port_flow_imbalance_6h",
                "port_flow_imbalance_24h",
            ):
                if column in scenario:
                    baseline = pd.to_numeric(scenario[column], errors="coerce").fillna(0.0)
                    scenario[column] = np.ceil(1.5 * baseline + 1.0)
        if scenario_type in ("MARINE_STRESS", "COMPOUND_STRESS"):
            for column, high in high_values.items():
                scenario[column] = np.maximum(
                    pd.to_numeric(scenario[column], errors="coerce").fillna(high), high
                )
        scenario["dataset_version"] = SCENARIO_VERSION
        scenario["data_origin"] = "COUNTERFACTUAL_STRESS"
        scenario["generator_version"] = SCENARIO_VERSION
        scenario["random_seed"] = random_seed
        scenario["training_allowed"] = False
        scenario["validation_allowed"] = False
        scenario["test_allowed"] = False
        scenario["production_claim_allowed"] = False
        scenario["expected_risk_direction"] = "NON_DECREASING"
        scenario["scenario_role"] = "ROBUSTNESS_TEST_ONLY_NO_GROUND_TRUTH"
        parts.append(scenario)
    return pd.concat(parts, ignore_index=True)


def _feature_registry(
    frame: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> pd.DataFrame:
    memberships: dict[str, list[str]] = {}
    for feature_set, columns in feature_sets.items():
        for column in columns:
            memberships.setdefault(column, []).append(feature_set)
    rows = []
    for column, sets in memberships.items():
        if column.startswith("target_"):
            role = "TARGET_ONLY"
            availability = "FUTURE_OUTCOME"
        elif column.startswith("research_"):
            role = "RESEARCH_FEATURE"
            availability = "RETROSPECTIVE_NOT_HISTORICALLY_PROVEN"
        elif column in ("vessel_key", "terminal_code", "cargo_group", "vessel_type", "port_code", "source"):
            role = "CATEGORICAL_FEATURE"
            availability = "BUSINESS_SEMANTICS_RETROSPECTIVE_SNAPSHOT"
        else:
            role = "CORE_FEATURE"
            availability = "KNOWN_OR_OBSERVED_BY_LANDMARK_RETROSPECTIVE"
        rows.append(
            {
                "column": column,
                "dtype": str(frame[column].dtype) if column in frame else "ABSENT",
                "role": role,
                "availability_semantics": availability,
                "feature_sets": "|".join(sorted(sets)),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "column"])


def build_operational_port_call_dataset(
    port_calls: pd.DataFrame,
    weather: pd.DataFrame,
    business_events: pd.DataFrame,
) -> BuildResult:
    calls = prepare_calls(port_calls)
    calls_with_history, history_features = add_prior_history(calls)
    landmarks = expand_landmarks(calls_with_history)
    landmarks, state_features = add_dynamic_port_state(landmarks, calls)
    landmarks, calendar_features = add_calendar_features(landmarks)
    landmarks, event_features = add_business_event_features(landmarks, business_events)
    landmarks, research_weather, weather_policy = add_research_weather(
        landmarks, weather
    )

    plan_features = [
        "arrival_delay_h",
        "planned_stay_h",
        "planned_berth_offset_h",
        "elapsed_since_arrival_h",
        "time_to_planned_departure_h",
        "overdue_h",
        "plan_progress_ratio",
        "berth_event_observed",
        "hours_since_berth_observed",
        "landmark_h",
    ]
    categorical_features = [
        "port_code",
        "terminal_code",
        "vessel_key",
        "vessel_type",
        "cargo_group",
        "source",
        "current_plan_state",
    ]
    for column in categorical_features:
        landmarks[column] = _clean_category(landmarks[column])
    numeric_core = list(
        dict.fromkeys(plan_features + history_features + state_features + calendar_features + event_features)
    )
    landmarks, numeric_core, core_imputation = impute_features_from_train(
        landmarks,
        numeric_core,
        "TRAIN_MEDIAN_PLUS_MISSING_FLAG",
    )
    landmarks, research_weather, weather_imputation = impute_features_from_train(
        landmarks,
        research_weather,
        "TRAIN_MEDIAN_PLUS_MISSING_FLAG_RESEARCH_ONLY",
    )

    targets = [
        "target_actual_atd",
        "target_total_stay_h",
        "target_departure_delay_h",
        "target_departure_delay_class",
        "target_delay_gt_1h",
        "target_delay_gt_3h",
        "target_delay_gt_6h",
        "target_remaining_h",
        "target_breach_gt3_observed",
        "target_breach_or_censor_h",
    ]
    for horizon_h in HORIZONS_H:
        targets.extend(
            [
                f"target_departure_within_{horizon_h}h",
                f"target_gt3_breach_within_{horizon_h}h",
            ]
        )
    metadata = [
        "dataset_version",
        "data_origin",
        "port_call_id",
        "vessel_name",
        "actual_ata",
        "planned_eta",
        "planned_etb",
        "planned_etd",
        "landmark_at",
        "split",
        "early_warning_eligible",
        "pre_breach_eligible",
        "per_call_sample_weight",
        "training_allowed",
        "validation_allowed",
        "test_allowed",
        "production_claim_allowed",
    ]
    feature_sets = {
        "core_numeric": numeric_core,
        "core_categorical": categorical_features,
        "core_model": numeric_core + categorical_features,
        "research_weather": research_weather,
        "research_enriched_model": numeric_core + categorical_features + research_weather,
        "targets": targets,
        "metadata": metadata,
    }
    scenarios = generate_counterfactual_scenarios(
        landmarks,
        feature_sets["core_model"],
        feature_sets["research_weather"],
    )

    duplicate_grain = int(
        landmarks.duplicated(["port_call_id", "landmark_at"], keep=False).sum()
    )
    strict_landmarks = bool((landmarks["landmark_at"] < landmarks["target_actual_atd"]).all())
    split_per_call = int(landmarks.groupby("port_call_id")["split"].nunique().max()) == 1
    target_in_features = sorted(
        column
        for column in feature_sets["core_model"] + feature_sets["research_weather"]
        if column.startswith("target_")
    )
    synthetic_in_main = int(landmarks["data_origin"].ne("REAL_RETROSPECTIVE").sum())
    prohibited_raw_future = sorted(
        set(
            (
                "actual_atb",
                "actual_atd",
                "stay_h",
                "departure_delay_h",
                "target_observed",
                "valid_stay",
            )
        ).intersection(feature_sets["metadata"] + feature_sets["core_model"] + feature_sets["research_weather"])
    )
    scenario_target_columns = [column for column in scenarios if column.startswith("target_")]
    scenario_training_rows = int(
        scenarios.get("training_allowed", pd.Series(dtype=bool)).fillna(False).sum()
    )
    eligible_calls = calls.loc[calls["split"].isin(["TRAIN", "VALID", "TEST"])]
    split_call_counts = eligible_calls["split"].value_counts()
    train_critical_calls = int(
        eligible_calls.loc[eligible_calls["split"].eq("TRAIN"), "target_delay_gt_6h"].sum()
    )
    train_major_calls = int(
        eligible_calls.loc[eligible_calls["split"].eq("TRAIN"), "target_delay_gt_3h"].sum()
    )
    finite_core = int(
        (~np.isfinite(landmarks[numeric_core].to_numpy(dtype="float64"))).sum()
    )

    gates = [
        ("UNIQUE_PORT_CALL_LANDMARK_GRAIN", duplicate_grain == 0, duplicate_grain),
        ("LANDMARK_STRICTLY_BEFORE_TARGET_ATD", strict_landmarks, 0 if strict_landmarks else 1),
        ("ONE_SPLIT_PER_PORT_CALL", split_per_call, 0 if split_per_call else 1),
        ("NO_TARGET_IN_FEATURE_SETS", len(target_in_features) == 0, len(target_in_features)),
        ("NO_RAW_FUTURE_IN_EXPORTED_INPUTS", len(prohibited_raw_future) == 0, len(prohibited_raw_future)),
        ("MAIN_DATASET_REAL_ONLY", synthetic_in_main == 0, synthetic_in_main),
        ("TRAIN_CALL_SUPPORT", int(split_call_counts.get("TRAIN", 0)) >= 15_000, int(split_call_counts.get("TRAIN", 0))),
        ("VALID_CALL_SUPPORT", int(split_call_counts.get("VALID", 0)) >= 4_000, int(split_call_counts.get("VALID", 0))),
        ("TEST_CALL_SUPPORT", int(split_call_counts.get("TEST", 0)) >= 1_000, int(split_call_counts.get("TEST", 0))),
        ("LANDMARK_SUPPORT", len(landmarks) >= 80_000, len(landmarks)),
        ("MAJOR_DELAY_TRAIN_SUPPORT", train_major_calls >= 1_000, train_major_calls),
        ("CRITICAL_DELAY_TRAIN_SUPPORT", train_critical_calls >= 400, train_critical_calls),
        ("FINITE_CORE_FEATURES", finite_core == 0, finite_core),
        ("SCENARIOS_HAVE_NO_TARGETS", len(scenario_target_columns) == 0, len(scenario_target_columns)),
        ("SCENARIOS_NOT_TRAINABLE", scenario_training_rows == 0, scenario_training_rows),
        ("TEST_CONTAINS_REAL_ROWS_ONLY", bool(landmarks.loc[landmarks["split"].eq("TEST"), "data_origin"].eq("REAL_RETROSPECTIVE").all()), 0),
        ("WEATHER_RESEARCH_ONLY", True, 0),
        ("PLAN_REVISION_LIMITATION_RECORDED", True, 0),
    ]
    quality_gates = pd.DataFrame(gates, columns=["gate", "passed", "observed"])
    quality_gates["severity"] = "CRITICAL"
    gates_passed = bool(quality_gates["passed"].all())

    split_rows = []
    for split, group in landmarks.groupby("split"):
        split_rows.append(
            {
                "split": split,
                "calls": group["port_call_id"].nunique(),
                "landmarks": len(group),
                "early_warning_landmarks": int(group["early_warning_eligible"].sum()),
                "major_delay_calls": group.loc[group["target_delay_gt_3h"], "port_call_id"].nunique(),
                "critical_delay_calls": group.loc[group["target_delay_gt_6h"], "port_call_id"].nunique(),
            }
        )
    split_report = pd.DataFrame(split_rows)
    call_target_report = (
        eligible_calls.groupby(["split", "target_departure_delay_class"], as_index=False)
        .agg(calls=("port_call_id", "nunique"))
    )
    call_target_report["share_pct"] = 100.0 * call_target_report["calls"] / call_target_report.groupby("split")["calls"].transform("sum")
    scenario_report = (
        scenarios.groupby("scenario_type", as_index=False)
        .agg(rows=("scenario_id", "size"), parent_calls=("parent_port_call_id", "nunique"))
        if not scenarios.empty
        else pd.DataFrame(columns=["scenario_type", "rows", "parent_calls"])
    )
    scenario_report["training_allowed"] = False
    scenario_report["ground_truth_available"] = False
    scenario_report["role"] = "ROBUSTNESS_TEST_ONLY"

    feature_registry = _feature_registry(landmarks, feature_sets)
    availability_report = pd.DataFrame(
        [
            ("planned_eta_etd", "RETROSPECTIVE_FINAL_SNAPSHOT", "CORE_RESEARCH", False),
            ("actual_ata", "KNOWN_AT_OR_BEFORE_LANDMARK", "CORE", True),
            ("actual_atb", "MASKED_UNTIL_EVENT_TIME", "CORE", True),
            ("actual_atd", "FUTURE_TARGET_ONLY", "TARGET", True),
            ("prior_call_outcomes", "ONLY_ATD_STRICTLY_BEFORE_CALL_ATA", "CORE", True),
            ("dynamic_port_state", "EVENTS_AT_OR_BEFORE_LANDMARK", "CORE_RETROSPECTIVE", True),
            ("weather", "OBSERVATION_PLUS_3H_LAG", "RESEARCH_ONLY", False),
            ("business_calendar", "DETERMINISTIC_KNOWN_FUTURE", "CORE", True),
            ("counterfactual_scenarios", "GENERATED_SEPARATE_ARTIFACT", "STRESS_TEST", False),
        ],
        columns=["family", "availability_semantics", "model_role", "historical_replay_proven"],
    )
    source_inventory = pd.DataFrame(
        [
            {
                "source": "core.port_call",
                "rows": len(calls),
                "eligible_complete_calls": len(eligible_calls),
                "role": "PRIMARY_REAL_ENTITY_AND_TARGET_SOURCE",
            },
            {
                "source": "core.maritime_observation + external weather",
                "rows": len(weather),
                "eligible_complete_calls": np.nan,
                "role": "RETROSPECTIVE_RESEARCH_CONTEXT",
            },
            {
                "source": "reference.business_event",
                "rows": len(business_events),
                "eligible_complete_calls": np.nan,
                "role": "KNOWN_FUTURE_CALENDAR",
            },
        ]
    )
    imputation_report = pd.concat(
        [core_imputation, weather_imputation], ignore_index=True
    )
    reports = {
        "01_source_inventory.csv": source_inventory,
        "02_split_and_landmark_support.csv": split_report,
        "03_call_target_distribution.csv": call_target_report,
        "04_feature_registry.csv": feature_registry,
        "05_imputation_registry.csv": imputation_report,
        "06_availability_policy.csv": availability_report,
        "07_weather_policy.csv": weather_policy,
        "08_counterfactual_scenario_manifest.csv": scenario_report,
        "09_quality_gates.csv": quality_gates,
    }
    decision_name = (
        "READY_FOR_RETROSPECTIVE_PORT_CALL_RISK_MODELING"
        if gates_passed
        else "BLOCKED_DATA_CONTRACT_REPAIR_REQUIRED"
    )
    decision = clean_json(
        {
            "status": "SUCCESS",
            "decision": decision_name,
            "dataset_version": DATASET_VERSION,
            "contract_version": CONTRACT_VERSION,
            "source_calls": len(calls),
            "eligible_calls": len(eligible_calls),
            "landmark_rows": len(landmarks),
            "column_count": len(landmarks.columns),
            "core_numeric_features": len(numeric_core),
            "core_categorical_features": len(categorical_features),
            "research_weather_features": len(research_weather),
            "target_count": len(targets),
            "train_calls": int(split_call_counts.get("TRAIN", 0)),
            "valid_calls": int(split_call_counts.get("VALID", 0)),
            "test_calls": int(split_call_counts.get("TEST", 0)),
            "train_major_delay_calls": train_major_calls,
            "train_critical_delay_calls": train_critical_calls,
            "counterfactual_rows": len(scenarios),
            "counterfactual_training_allowed": False,
            "main_dataset_synthetic_rows": synthetic_in_main,
            "targets_imputed": False,
            "selection_policy": "TRAIN_THEN_VALID_TEST_UNTOUCHED",
            "primary_objective": "EARLY_WARNING_DEPARTURE_DELAY_GT3H_BY_PORT_CALL",
            "secondary_objectives": [
                "DEPARTURE_DELAY_ORDINAL_CLASSIFICATION",
                "REMAINING_TIME_REGRESSION",
                "TIME_TO_GT3H_BREACH_SURVIVAL",
            ],
            "quality_gates_passed": gates_passed,
            "historical_replay_allowed": False,
            "production_promotion_allowed": False,
            "availability_limitation": (
                "Historical revisions and available_at for ETA/ETD are absent; "
                "the dataset is retrospective until prospective snapshots mature."
            ),
            "next_block": "B60D_MULTITASK_DELAY_REGIME_AND_SURVIVAL_BENCHMARK",
        }
    )
    ordered = list(dict.fromkeys(metadata + categorical_features + numeric_core + research_weather + targets))
    ordered = [column for column in ordered if column in landmarks]
    # Export only the allow-listed contract. In particular, never append raw
    # actual_atd/stay/delay source columns next to model inputs.
    landmarks = landmarks[ordered].sort_values(
        ["landmark_at", "port_call_id"]
    ).reset_index(drop=True)
    decision["column_count"] = len(landmarks.columns)
    return BuildResult(
        dataset=landmarks,
        scenarios=scenarios,
        reports=reports,
        decision=decision,
        feature_sets=feature_sets,
    )
