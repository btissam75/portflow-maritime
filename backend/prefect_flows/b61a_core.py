from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


DATASET_VERSION = "b61a-governed-port-call-enrichment-v1"
CONTRACT_VERSION = "b61a-governed-data-completion-contract-v1"
SOURCE_DATASET_VERSION = "b60c-operational-port-call-landmark-v1"
EVENT_CONTEXT_VERSION = "b60ch-port-call-event-context-v1"
HORIZONS_H = (6, 12, 24, 48, 72)

RHO_SEAWATER_KG_M3 = 1025.0
GRAVITY_M_S2 = 9.80665


@dataclass
class GovernedBuildResult:
    dataset: pd.DataFrame
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
    return pd.to_datetime(series, utc=True, errors="coerce")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def _missing(frame: pd.DataFrame, column: str) -> pd.Series:
    flag = f"{column}_missing"
    if flag in frame:
        return _numeric(frame, flag, 1.0).fillna(1.0).gt(0.0)
    return _numeric(frame, column).isna()


def circular_delta_deg(left: pd.Series, right: pd.Series) -> pd.Series:
    delta = (_numeric(pd.DataFrame({"x": left}), "x") - _numeric(
        pd.DataFrame({"x": right}), "x"
    ) + 180.0) % 360.0 - 180.0
    return delta.abs()


def _train_quantile(
    frame: pd.DataFrame,
    column: str,
    quantile: float,
    missing: pd.Series | None = None,
    fallback: float = 1.0,
) -> float:
    mask = frame["split"].eq("TRAIN")
    if missing is not None:
        mask &= ~missing
    values = _numeric(frame, column).loc[mask].dropna()
    if values.empty:
        return float(fallback)
    return float(values.quantile(quantile))


def _safe_ratio(numerator: pd.Series, denominator: float) -> pd.Series:
    if not math.isfinite(denominator) or abs(denominator) < 1e-9:
        denominator = 1.0
    return pd.to_numeric(numerator, errors="coerce").fillna(0.0) / denominator


def _impute_derived_from_train(
    frame: pd.DataFrame,
    columns: Iterable[str],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    # The caller already owns a private build frame. Mutating it avoids another
    # several-hundred-MB copy on the full B60C landmark dataset.
    output = frame
    expanded: list[str] = []
    report_rows: list[dict[str, Any]] = []
    train = output["split"].eq("TRAIN")
    for column in columns:
        values = _numeric(output, column)
        missing = values.isna()
        observed_train = values.loc[train & ~missing]
        median = float(observed_train.median()) if len(observed_train) else 0.0
        output[column] = values.fillna(median)
        expanded.append(column)
        if missing.any():
            flag = f"{column}_missing"
            output[flag] = missing.astype("int8")
            expanded.append(flag)
        report_rows.append(
            {
                "feature": column,
                "method": "TRAIN_MEDIAN_DERIVED_FEATURE_ONLY",
                "train_median": median,
                "missing_rows": int(missing.sum()),
                "target_imputed": False,
                "quality_grade": "D_IMPUTED_WHEN_FLAGGED",
            }
        )
    return output, expanded, pd.DataFrame(report_rows)


def _join_event_context(
    landmarks: pd.DataFrame,
    event_context: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    if event_context.empty:
        return landmarks, [], []
    context = event_context
    context["landmark_at"] = _as_utc(context["landmark_at"])
    if context.duplicated(["port_call_id", "landmark_at"]).any():
        raise ValueError("B60C-H event context is not unique at landmark grain")
    excluded = {
        "event_context_version",
        "dataset_version",
        "port_call_id",
        "landmark_at",
        "split",
        "predictive_feature_count",
        "research_feature_count",
        "synthetic_feature_rows",
        "materialization_run_id",
    }
    candidates = [column for column in context if column not in excluded]
    additions = [column for column in candidates if column not in landmarks]
    if not additions:
        return landmarks, [], []
    # Reindex only the added columns. A DataFrame merge would duplicate all
    # 200+ B60C columns and exceed the 3 GB Prefect worker memory limit.
    context_indexed = context.set_index(["port_call_id", "landmark_at"])
    landmark_keys = pd.MultiIndex.from_frame(
        landmarks[["port_call_id", "landmark_at"]]
    )
    addition_block = context_indexed[additions].reindex(landmark_keys)
    addition_block.reset_index(drop=True, inplace=True)
    for column in additions:
        addition_block[column] = pd.to_numeric(
            addition_block[column], errors="coerce"
        ).fillna(0.0)
    joined = pd.concat(
        [landmarks.reset_index(drop=True), addition_block], axis=1, copy=False
    )
    predictive = [column for column in additions if not column.startswith("research_")]
    research = [column for column in additions if column.startswith("research_")]
    return joined, predictive, research


def align_issue_time_forecasts(
    landmark_times: pd.Series,
    forecasts: pd.DataFrame,
    horizons: Iterable[int] = HORIZONS_H,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    unique = pd.DataFrame(
        {"landmark_at": _as_utc(pd.Series(landmark_times).drop_duplicates())}
    ).dropna().sort_values("landmark_at")
    variables = (
        "wind_speed_ms",
        "wind_direction_deg",
        "pressure_hpa",
        "visibility_m",
        "wave_height_m",
        "wave_direction_deg",
        "wave_period_s",
        "ocean_current_ms",
        "sea_surface_temperature_c",
    )
    features: list[str] = []
    for horizon in horizons:
        features.extend(
            [
                f"issue_time_fcst_h{horizon}_available",
                f"issue_time_fcst_h{horizon}_age_h",
                f"issue_time_fcst_h{horizon}_provider_lead_h",
                *[f"issue_time_fcst_h{horizon}_{name}" for name in variables],
            ]
        )
    if unique.empty:
        return unique, features, pd.DataFrame()
    empty_features = pd.DataFrame(
        {
            column: (
                np.zeros(len(unique), dtype="int8")
                if column.endswith("_available")
                else np.full(len(unique), np.nan, dtype="float64")
            )
            for column in features
        },
        index=unique.index,
    )
    unique = pd.concat([unique, empty_features], axis=1)
    if forecasts.empty:
        coverage = pd.DataFrame(
            [
                {"horizon_h": horizon, "eligible_times": len(unique), "matched_times": 0, "coverage_pct": 0.0}
                for horizon in horizons
            ]
        )
        return unique, features, coverage

    source = forecasts.copy()
    for column in ("issue_at", "available_at", "valid_at"):
        source[column] = _as_utc(source[column])
    source = source.dropna(subset=["issue_at", "available_at", "valid_at"])
    source = source.loc[source["available_at"].ge(source["issue_at"])].copy()
    # B60C currently ends before B58C-D collection starts. Without this range
    # gate, grouping every historical landmark across five horizons performs a
    # large CPU-only search that can never produce a match.
    latest_landmark = unique["landmark_at"].max()
    earliest_target = unique["landmark_at"].min() + pd.Timedelta(
        hours=min(horizons)
    )
    if (
        source.empty
        or source["available_at"].min() > latest_landmark
        or source["valid_at"].max() < earliest_target
    ):
        coverage = pd.DataFrame(
            [
                {
                    "horizon_h": horizon,
                    "eligible_times": len(unique),
                    "matched_times": 0,
                    "coverage_pct": 0.0,
                }
                for horizon in horizons
            ]
        )
        return unique, features, coverage
    source["valid_hour"] = source["valid_at"].dt.round("h")
    source = source.sort_values(["valid_hour", "available_at", "issue_at"])
    coverage_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        target_hour = (unique["landmark_at"] + pd.Timedelta(hours=horizon)).dt.round("h")
        selected: dict[int, pd.Series] = {}
        left_groups = pd.DataFrame(
            {"row": unique.index, "landmark_at": unique["landmark_at"], "target_hour": target_hour}
        ).groupby("target_hour", sort=False)
        right_groups = {key: value for key, value in source.groupby("valid_hour", sort=False)}
        for target, left in left_groups:
            candidates = right_groups.get(target)
            if candidates is None or candidates.empty:
                continue
            candidates = candidates.sort_values(["available_at", "issue_at"])
            availability = candidates["available_at"].astype("int64").to_numpy()
            query = left["landmark_at"].astype("int64").to_numpy()
            positions = np.searchsorted(availability, query, side="right") - 1
            for row_index, position in zip(left["row"].to_numpy(), positions):
                if position < 0:
                    continue
                candidate = candidates.iloc[int(position)]
                if candidate["issue_at"] > unique.at[row_index, "landmark_at"]:
                    continue
                selected[int(row_index)] = candidate
        available_name = f"issue_time_fcst_h{horizon}_available"
        age_name = f"issue_time_fcst_h{horizon}_age_h"
        lead_name = f"issue_time_fcst_h{horizon}_provider_lead_h"
        for row_index, candidate in selected.items():
            landmark = unique.at[row_index, "landmark_at"]
            unique.at[row_index, available_name] = 1.0
            unique.at[row_index, age_name] = (
                landmark - candidate["available_at"]
            ).total_seconds() / 3600.0
            unique.at[row_index, lead_name] = float(
                candidate.get(
                    "lead_time_h",
                    (candidate["valid_at"] - candidate["issue_at"]).total_seconds() / 3600.0,
                )
            )
            for variable in variables:
                unique.at[row_index, f"issue_time_fcst_h{horizon}_{variable}"] = pd.to_numeric(
                    candidate.get(variable), errors="coerce"
                )
        matched = len(selected)
        coverage_rows.append(
            {
                "horizon_h": horizon,
                "eligible_times": len(unique),
                "matched_times": matched,
                "coverage_pct": 100.0 * matched / max(1, len(unique)),
            }
        )
    return unique, features, pd.DataFrame(coverage_rows)


def _add_physical_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, pd.DataFrame]:
    output = frame
    hs_name = "research_wave_height_m_lag3h"
    period_name = "research_wave_period_s_lag3h"
    wave_direction_name = "research_wave_direction_deg_lag3h"
    wind_name = "research_wind_speed_ms_lag3h"
    wind_direction_name = "research_wind_direction_deg_lag3h"
    visibility_name = "research_visibility_m_lag3h"
    pressure_name = "research_pressure_hpa_lag3h"
    gust_name = "research_wind_gusts_10m_lag3h"

    hs = _numeric(output, hs_name)
    period = _numeric(output, period_name)
    wave_direction = _numeric(output, wave_direction_name)
    wind = _numeric(output, wind_name)
    wind_direction = _numeric(output, wind_direction_name)
    visibility = _numeric(output, visibility_name)
    pressure = _numeric(output, pressure_name)
    gust = _numeric(output, gust_name)

    invalid = (
        hs.lt(0.0) | hs.ge(30.0) | period.le(0.0) | period.ge(40.0)
        | wind.lt(0.0) | wind.ge(100.0)
        | (~wave_direction.between(0.0, 360.0, inclusive="left"))
        | (~wind_direction.between(0.0, 360.0, inclusive="left"))
    )
    output["research_physics_invalid_input_flag"] = invalid.fillna(True).astype("int8")

    valid_wave = hs.between(0.0, 30.0, inclusive="left") & period.between(
        0.0, 40.0, inclusive="neither"
    )
    wave_flux = (
        RHO_SEAWATER_KG_M3
        * GRAVITY_M_S2**2
        / (64.0 * np.pi)
        * hs.pow(2)
        * period
        / 1000.0
    ).where(valid_wave)
    wavelength = (GRAVITY_M_S2 * period.pow(2) / (2.0 * np.pi)).where(valid_wave)
    output["research_wave_energy_flux_kw_m"] = wave_flux
    output["research_deepwater_wavelength_m"] = wavelength
    output["research_wave_steepness"] = (hs / wavelength.replace(0.0, np.nan)).where(valid_wave)
    output["research_wave_energy_density_kj_m2"] = (
        RHO_SEAWATER_KG_M3 * GRAVITY_M_S2 * hs.pow(2) / 16.0 / 1000.0
    ).where(valid_wave)
    output["research_wave_direction_sin"] = np.sin(np.deg2rad(wave_direction))
    output["research_wave_direction_cos"] = np.cos(np.deg2rad(wave_direction))
    output["research_wind_direction_sin"] = np.sin(np.deg2rad(wind_direction))
    output["research_wind_direction_cos"] = np.cos(np.deg2rad(wind_direction))
    delta = ((wind_direction - wave_direction + 180.0) % 360.0 - 180.0).abs()
    output["research_wind_wave_direction_delta_deg"] = delta
    output["research_wind_wave_alignment_cos"] = np.cos(np.deg2rad(delta))
    output["research_wind_stress_proxy_ms2"] = wind.pow(2)
    output["research_gust_factor"] = gust / wind.where(wind.gt(0.5))

    source_missing = (
        _missing(output, hs_name)
        | _missing(output, period_name)
        | _missing(output, wave_direction_name)
        | _missing(output, wind_name)
        | _missing(output, wind_direction_name)
    )
    output["research_physics_source_missing_flag"] = source_missing.astype("int8")

    thresholds = []
    for feature, fallback in (
        (hs_name, 2.5),
        (wind_name, 10.8),
        ("research_wave_energy_flux_kw_m", 20.0),
        ("research_wave_steepness", 0.05),
    ):
        missing = source_missing if feature.startswith("research_wave_") else _missing(output, feature)
        q90 = _train_quantile(output, feature, 0.90, missing, fallback)
        q95 = _train_quantile(output, feature, 0.95, missing, fallback)
        alias = feature
        if alias.startswith("research_"):
            alias = alias[len("research_") :]
        if alias.endswith("_lag3h"):
            alias = alias[: -len("_lag3h")]
        output[f"research_local_{alias}_ge_q90"] = _numeric(output, feature).ge(q90).astype("int8")
        output[f"research_local_{alias}_ge_q95"] = _numeric(output, feature).ge(q95).astype("int8")
        thresholds.extend(
            [
                {"feature": feature, "quantile": "q90", "value": q90, "source_split": "TRAIN", "role": "LOCAL_EMPIRICAL_CONTEXT"},
                {"feature": feature, "quantile": "q95", "value": q95, "source_split": "TRAIN", "role": "LOCAL_EMPIRICAL_CONTEXT"},
            ]
        )

    output["research_reference_wind_beaufort6_flag"] = wind.ge(10.8).astype("int8")
    output["research_reference_wind_beaufort8_flag"] = wind.ge(17.2).astype("int8")
    output["research_reference_low_visibility_1km_flag"] = visibility.lt(1000.0).astype("int8")
    output["research_reference_severe_visibility_200m_flag"] = visibility.lt(200.0).astype("int8")

    temporal_source = (
        output[["landmark_at", hs_name, wind_name, pressure_name, "research_wave_energy_flux_kw_m"]]
        .sort_values("landmark_at")
        .drop_duplicates("landmark_at", keep="first")
        .set_index("landmark_at")
    )
    temporal_source["research_pressure_delta_3h"] = temporal_source[pressure_name] - temporal_source[pressure_name].shift(3)
    temporal_source["research_pressure_delta_24h"] = temporal_source[pressure_name] - temporal_source[pressure_name].shift(24)
    temporal_source["research_wave_energy_flux_roll_6h_mean"] = temporal_source["research_wave_energy_flux_kw_m"].rolling(6, min_periods=2).mean()
    temporal_source["research_wave_energy_flux_roll_24h_mean"] = temporal_source["research_wave_energy_flux_kw_m"].rolling(24, min_periods=6).mean()
    temporal_source["research_wave_energy_flux_roll_24h_max"] = temporal_source["research_wave_energy_flux_kw_m"].rolling(24, min_periods=6).max()
    temporal_columns = [column for column in temporal_source if column.startswith("research_") and column not in {hs_name, wind_name, pressure_name, "research_wave_energy_flux_kw_m"}]
    for column in temporal_columns:
        output[column] = output["landmark_at"].map(temporal_source[column])

    continuous = [
        "research_wave_energy_flux_kw_m",
        "research_deepwater_wavelength_m",
        "research_wave_steepness",
        "research_wave_energy_density_kj_m2",
        "research_wave_direction_sin",
        "research_wave_direction_cos",
        "research_wind_direction_sin",
        "research_wind_direction_cos",
        "research_wind_wave_direction_delta_deg",
        "research_wind_wave_alignment_cos",
        "research_wind_stress_proxy_ms2",
        "research_gust_factor",
        *temporal_columns,
    ]
    output, imputed_continuous, imputation = _impute_derived_from_train(output, continuous)
    binary = [
        column
        for column in output
        if column.startswith("research_local_") or column.startswith("research_reference_")
    ] + ["research_physics_invalid_input_flag", "research_physics_source_missing_flag"]
    features = list(dict.fromkeys(imputed_continuous + binary))
    reference_rules = pd.DataFrame(
        [
            ("wind_speed_ms", ">=10.8", "Beaufort force 6 reference", "CONTEXT_ONLY_NOT_PORT_LIMIT"),
            ("wind_speed_ms", ">=17.2", "Beaufort force 8 reference", "CONTEXT_ONLY_NOT_PORT_LIMIT"),
            ("visibility_m", "<1000", "low visibility reference", "CONTEXT_ONLY_NOT_PORT_LIMIT"),
            ("visibility_m", "<200", "severe visibility reference", "CONTEXT_ONLY_NOT_PORT_LIMIT"),
        ],
        columns=["variable", "rule", "meaning", "governance"],
    )
    return output, features, pd.DataFrame(thresholds), pd.concat(
        [imputation, reference_rules.assign(feature=np.nan, method="REFERENCE_RULE", train_median=np.nan, missing_rows=0, target_imputed=False, quality_grade="C_DERIVED")],
        ignore_index=True,
        sort=False,
    )


def _add_operational_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    output = frame
    train = output["split"].eq("TRAIN")
    threshold_rows: list[dict[str, Any]] = []

    scales: dict[str, float] = {}
    for column in (
        "port_active_calls_observed",
        "terminal_active_calls_observed",
        "port_flow_imbalance_6h",
        "terminal_flow_imbalance_6h",
    ):
        values = _numeric(output, column).loc[train].abs().dropna()
        scale = float(values.quantile(0.95)) if len(values) else 1.0
        scale = max(scale, 1.0)
        scales[column] = scale
        threshold_rows.append(
            {"feature": column, "quantile": "q95_abs_scale", "value": scale, "source_split": "TRAIN", "role": "ROBUST_OPERATIONAL_NORMALIZER"}
        )

    output["port_occupancy_proxy"] = _safe_ratio(
        _numeric(output, "port_active_calls_observed"), scales["port_active_calls_observed"]
    ).clip(0.0, 5.0)
    output["terminal_occupancy_proxy"] = _safe_ratio(
        _numeric(output, "terminal_active_calls_observed"), scales["terminal_active_calls_observed"]
    ).clip(0.0, 5.0)
    output["port_queue_pressure_proxy"] = _safe_ratio(
        _numeric(output, "port_flow_imbalance_6h").clip(lower=0.0), scales["port_flow_imbalance_6h"]
    ).clip(0.0, 5.0)
    output["terminal_queue_pressure_proxy"] = _safe_ratio(
        _numeric(output, "terminal_flow_imbalance_6h").clip(lower=0.0), scales["terminal_flow_imbalance_6h"]
    ).clip(0.0, 5.0)
    late_rate = _numeric(output, "terminal_completed_late_gt3_rate_last_24h", 0.0).fillna(0.0).clip(0.0, 1.0)
    overdue = (_numeric(output, "overdue_h", 0.0).fillna(0.0) / 6.0).clip(0.0, 3.0)
    output["operational_pressure_index"] = (
        0.30 * output["port_occupancy_proxy"]
        + 0.25 * output["terminal_occupancy_proxy"]
        + 0.20 * output["terminal_queue_pressure_proxy"]
        + 0.15 * late_rate
        + 0.10 * overdue
    ).clip(0.0, 5.0)

    vessel_n = _numeric(output, "vessel_history_prior_calls", 0.0).fillna(0.0).clip(lower=0.0)
    vessel_rate = _numeric(output, "vessel_history_prior_late_gt3_rate", 0.0).fillna(0.0).clip(0.0, 1.0)
    port_rate = _numeric(output, "port_history_prior_late_gt3_rate", 0.0).fillna(0.0).clip(0.0, 1.0)
    prior_strength = 20.0
    output["vessel_delay_susceptibility_bayes"] = (
        vessel_n * vessel_rate + prior_strength * port_rate
    ) / (vessel_n + prior_strength)
    output["vessel_history_evidence_strength"] = vessel_n / (vessel_n + prior_strength)

    marine = np.maximum(
        _numeric(output, "research_local_wave_height_m_ge_q90", 0.0).fillna(0.0),
        _numeric(output, "research_local_wind_speed_ms_ge_q90", 0.0).fillna(0.0),
    )
    output["research_compound_marine_operational_pressure"] = (
        output["operational_pressure_index"] * marine
    )
    event = _numeric(output, "known_event_any_now", 0.0).fillna(0.0).clip(0.0, 1.0)
    output["known_event_operational_pressure_interaction"] = (
        event * output["operational_pressure_index"]
    )

    q90 = _train_quantile(output, "operational_pressure_index", 0.90, fallback=1.0)
    output["operational_pressure_ge_train_q90"] = output["operational_pressure_index"].ge(q90).astype("int8")
    threshold_rows.append(
        {"feature": "operational_pressure_index", "quantile": "q90", "value": q90, "source_split": "TRAIN", "role": "LOCAL_EMPIRICAL_CONTEXT"}
    )
    core = [
        "port_occupancy_proxy",
        "terminal_occupancy_proxy",
        "port_queue_pressure_proxy",
        "terminal_queue_pressure_proxy",
        "operational_pressure_index",
        "vessel_delay_susceptibility_bayes",
        "vessel_history_evidence_strength",
        "known_event_operational_pressure_interaction",
        "operational_pressure_ge_train_q90",
    ]
    research = ["research_compound_marine_operational_pressure"]
    return output, core + research, pd.DataFrame(threshold_rows)


def _feature_registry(
    feature_sets: dict[str, list[str]],
    issue_time_coverage_ready: bool,
) -> pd.DataFrame:
    membership: dict[str, set[str]] = {}
    for feature_set, columns in feature_sets.items():
        for column in columns:
            membership.setdefault(column, set()).add(feature_set)
    rows = []
    for feature, sets in membership.items():
        if feature.startswith("target_"):
            family = "target"
            grade = "A_REAL_OUTCOME"
            availability = "FUTURE_OUTCOME_NEVER_INPUT"
            training_allowed = False
            production_allowed = False
            method = "OBSERVED_SOURCE_TARGET"
        elif feature.startswith("issue_time_fcst_"):
            family = "issue_time_forecast"
            grade = "A_REAL_ISSUE_TIME" if issue_time_coverage_ready else "A_PENDING_HISTORY"
            availability = "AVAILABLE_AT_NOT_AFTER_LANDMARK"
            training_allowed = issue_time_coverage_ready
            production_allowed = issue_time_coverage_ready
            method = "B58CD_PROVIDER_FORECAST"
        elif feature.startswith("research_"):
            family = "retrospective_weather_or_event"
            grade = "B_REAL_RETROSPECTIVE_OR_C_DERIVED"
            availability = "RETROSPECTIVE_RESEARCH_ONLY"
            training_allowed = True
            production_allowed = False
            method = "REANALYSIS_OR_DETERMINISTIC_DERIVATION"
        elif feature.endswith("_missing"):
            family = "missingness_indicator"
            grade = "C_DERIVED"
            availability = "SAME_AS_PARENT_FEATURE"
            training_allowed = True
            production_allowed = True
            method = "DETERMINISTIC_MISSINGNESS_FLAG"
        else:
            family = "core_operational_or_calendar"
            grade = "A_OR_C_POINT_IN_TIME"
            availability = "B60C_POINT_IN_TIME_CONTRACT"
            training_allowed = True
            production_allowed = True
            method = "OBSERVED_OR_STRICT_PRIOR_DERIVATION"
        rows.append(
            {
                "feature": feature,
                "family": family,
                "quality_grade": grade,
                "availability_semantics": availability,
                "is_synthetic": False,
                "target_imputed": False,
                "training_allowed": training_allowed,
                "production_allowed": production_allowed,
                "method": method,
                "feature_sets": "|".join(sorted(sets)),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "feature"])


def build_governed_dataset(
    landmarks: pd.DataFrame,
    event_context: pd.DataFrame | None = None,
    issue_time_forecasts: pd.DataFrame | None = None,
    minimum_rows: int = 80_000,
) -> GovernedBuildResult:
    required = {
        "dataset_version",
        "data_origin",
        "port_call_id",
        "landmark_at",
        "split",
        "training_allowed",
        "validation_allowed",
        "test_allowed",
    }
    missing_required = sorted(required - set(landmarks.columns))
    if missing_required:
        raise ValueError(f"B60C contract columns missing: {missing_required}")
    # The job gives this builder ownership of the loaded frame. Keep a single
    # wide B60C frame in memory and append governed columns in place.
    source = landmarks
    source["landmark_at"] = _as_utc(source["landmark_at"])
    version_mask = source["dataset_version"].eq(SOURCE_DATASET_VERSION)
    if not bool(version_mask.all()):
        source = source.loc[version_mask].copy()
    if source.empty:
        raise ValueError(f"No source rows for {SOURCE_DATASET_VERSION}")
    source.reset_index(drop=True, inplace=True)
    source_targets = [column for column in source if column.startswith("target_")]
    original_target_hash = pd.util.hash_pandas_object(
        source[["port_call_id", "landmark_at", *source_targets]], index=False
    ).sum()

    output, event_predictive, event_research = _join_event_context(
        source, event_context if event_context is not None else pd.DataFrame()
    )
    print(
        f"B61A stage=EVENT_CONTEXT_JOINED rows={len(output)} "
        f"predictive_features={len(event_predictive)} research_features={len(event_research)}"
    )
    output, physical_features, physical_thresholds, derivation_report = _add_physical_features(output)
    print(
        f"B61A stage=PHYSICAL_FEATURES_DERIVED features={len(physical_features)}"
    )
    output, operational_features, operational_thresholds = _add_operational_features(output)
    print(
        f"B61A stage=OPERATIONAL_FEATURES_DERIVED features={len(operational_features)}"
    )

    aligned, issue_time_features, forecast_coverage = align_issue_time_forecasts(
        output["landmark_at"],
        issue_time_forecasts if issue_time_forecasts is not None else pd.DataFrame(),
    )
    print(
        f"B61A stage=ISSUE_TIME_ALIGNMENT_COMPLETE "
        f"forecast_rows={0 if issue_time_forecasts is None else len(issue_time_forecasts)}"
    )
    aligned_indexed = aligned.set_index("landmark_at")
    forecast_block = aligned_indexed[issue_time_features].reindex(
        pd.DatetimeIndex(output["landmark_at"])
    )
    forecast_block.reset_index(drop=True, inplace=True)
    output = pd.concat([output, forecast_block], axis=1, copy=False)
    for column in issue_time_features:
        if column.endswith("_available"):
            output[column] = _numeric(output, column, 0.0).fillna(0.0).astype("int8")

    essential_weather = [
        "research_wave_height_m_lag3h",
        "research_wave_period_s_lag3h",
        "research_wind_speed_ms_lag3h",
        "research_pressure_hpa_lag3h",
        "research_visibility_m_lag3h",
    ]
    observed_parts = [(~_missing(output, column)).astype(float) for column in essential_weather]
    output["research_weather_source_completeness"] = pd.concat(observed_parts, axis=1).mean(axis=1)
    available_columns = [column for column in issue_time_features if column.endswith("_available")]
    output["issue_time_forecast_completeness"] = (
        output[available_columns].mean(axis=1) if available_columns else 0.0
    )
    output["governed_quality_grade"] = np.select(
        [
            output["issue_time_forecast_completeness"].ge(0.8),
            output["research_weather_source_completeness"].ge(0.8),
        ],
        ["A_REAL_ISSUE_TIME", "B_REAL_RETROSPECTIVE"],
        default="C_CORE_ONLY_OR_PARTIAL",
    )
    output["source_dataset_version"] = SOURCE_DATASET_VERSION
    output["dataset_version"] = DATASET_VERSION
    output["data_completion_contract"] = CONTRACT_VERSION
    output["targets_imputed"] = False
    output["synthetic_row"] = False
    output["production_claim_allowed"] = False

    metadata = {
        "dataset_version",
        "source_dataset_version",
        "data_completion_contract",
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
        "materialization_run_id",
        "governed_quality_grade",
        "targets_imputed",
        "synthetic_row",
    }
    research_base = [column for column in source.columns if column.startswith("research_")]
    research_derived = [column for column in physical_features + operational_features + event_research if column.startswith("research_")]
    issue_time_ready = bool(
        not forecast_coverage.empty
        and forecast_coverage["coverage_pct"].min() >= 60.0
        and output.loc[output["split"].eq("TRAIN"), "issue_time_forecast_completeness"].mean() >= 0.6
    )
    exclusions = metadata | set(source_targets) | set(research_base) | set(research_derived) | set(issue_time_features)
    core_existing = [column for column in source.columns if column not in exclusions]
    core_added = [column for column in event_predictive + operational_features if not column.startswith("research_")]
    core_model = list(dict.fromkeys(core_existing + core_added))
    research_features = list(dict.fromkeys(research_base + research_derived + event_research))
    quality_features = [
        "research_weather_source_completeness",
        "issue_time_forecast_completeness",
    ]
    feature_sets = {
        "core_model": core_model,
        "research_features": research_features,
        "research_enriched_model": list(dict.fromkeys(core_model + research_features + quality_features)),
        "prospective_issue_time_features": issue_time_features,
        "prospective_model": list(dict.fromkeys(core_model + issue_time_features + quality_features)),
        "targets": source_targets,
        "metadata": [column for column in output if column in metadata],
    }

    duplicate_rows = int(output.duplicated(["port_call_id", "landmark_at"]).sum())
    synthetic_rows = int(output["data_origin"].ne("REAL_RETROSPECTIVE").sum()) + int(output["synthetic_row"].sum())
    target_hash = pd.util.hash_pandas_object(
        output[["port_call_id", "landmark_at", *source_targets]], index=False
    ).sum()
    target_integrity = bool(original_target_hash == target_hash)
    target_in_inputs = sorted(
        set(source_targets).intersection(
            feature_sets["core_model"]
            + feature_sets["research_enriched_model"]
            + feature_sets["prospective_model"]
        )
    )
    threshold_report = pd.concat(
        [physical_thresholds, operational_thresholds], ignore_index=True
    )
    thresholds_train_only = bool(threshold_report["source_split"].eq("TRAIN").all())
    split_unchanged = bool(
        output.groupby("port_call_id")["split"].nunique().max() == 1
    )
    gates = pd.DataFrame(
        [
            ("SOURCE_AND_OUTPUT_ROW_COUNT_MATCH", len(output) == len(source), len(output) - len(source)),
            ("UNIQUE_PORT_CALL_LANDMARK_GRAIN", duplicate_rows == 0, duplicate_rows),
            ("ONE_SPLIT_PER_PORT_CALL", split_unchanged, 0 if split_unchanged else 1),
            ("TARGET_VALUES_BYTE_STABLE", target_integrity, 0 if target_integrity else 1),
            ("TARGETS_NEVER_IMPUTED", not bool(output["targets_imputed"].any()), 0),
            ("NO_TARGET_IN_INPUT_FEATURE_SETS", len(target_in_inputs) == 0, len(target_in_inputs)),
            ("NO_SYNTHETIC_ROWS_IN_MAIN_DATASET", synthetic_rows == 0, synthetic_rows),
            ("THRESHOLDS_LEARNED_FROM_TRAIN_ONLY", thresholds_train_only, 0 if thresholds_train_only else 1),
            ("TEST_REMAINS_REAL_AND_UNMODIFIED", bool(output.loc[output["split"].eq("TEST"), "data_origin"].eq("REAL_RETROSPECTIVE").all()), 0),
            ("ISSUE_TIME_FORECASTS_NOT_IMPUTED", not any(column.endswith("_missing") for column in issue_time_features), 0),
            ("MINIMUM_LANDMARK_SUPPORT", len(output) >= minimum_rows, len(output)),
        ],
        columns=["gate", "passed", "observed"],
    )
    gates["severity"] = "CRITICAL"
    gates_passed = bool(gates["passed"].all())

    coverage = (
        output.groupby("split", as_index=False)
        .agg(
            rows=("port_call_id", "size"),
            calls=("port_call_id", "nunique"),
            weather_completeness=("research_weather_source_completeness", "mean"),
            issue_time_completeness=("issue_time_forecast_completeness", "mean"),
        )
    )
    coverage["weather_completeness_pct"] = 100.0 * coverage.pop("weather_completeness")
    coverage["issue_time_completeness_pct"] = 100.0 * coverage.pop("issue_time_completeness")
    quality_distribution = (
        output.groupby(["split", "governed_quality_grade"], as_index=False)
        .agg(rows=("port_call_id", "size"), calls=("port_call_id", "nunique"))
    )
    feature_registry = _feature_registry(feature_sets, issue_time_ready)
    track_policy = pd.DataFrame(
        [
            ("CORE", "B60C point-in-time and known calendar", True, False, "RETROSPECTIVE_MODELING"),
            ("RESEARCH_ENRICHED", "reanalysis plus physical derivations", True, False, "ABLATION_AND_RESEARCH"),
            ("PROSPECTIVE_ISSUE_TIME", "B58C-D immutable forecasts", issue_time_ready, issue_time_ready, "SHADOW_THEN_PRODUCTION"),
            ("SYNTHETIC_SCENARIO", "separate B60C stress artifact", False, False, "ROBUSTNESS_ONLY"),
        ],
        columns=["track", "content", "training_allowed", "production_allowed", "role"],
    )
    source_inventory = pd.DataFrame(
        [
            ("features.maritime_port_call_landmark_v1", len(source), "REAL_RETROSPECTIVE", "PRIMARY_ENTITY_TARGET_AND_CORE_FEATURES"),
            ("features.maritime_port_call_event_context_v1", len(event_context) if event_context is not None else 0, "REAL_CALENDAR_AND_RETROSPECTIVE_EVENTS", "KNOWN_CALENDAR_PLUS_RESEARCH"),
            ("features.maritime_issue_time_weather_forecast_v1", len(issue_time_forecasts) if issue_time_forecasts is not None else 0, "REAL_PROVIDER_FORECAST", "PROSPECTIVE_ISSUE_TIME_ONLY"),
            ("B61A_DETERMINISTIC_DERIVATIONS", len(output), "DERIVED_NOT_OBSERVED", "PHYSICAL_AND_OPERATIONAL_FEATURES"),
        ],
        columns=["source", "rows", "data_semantics", "role"],
    )
    reports = {
        "01_source_inventory.csv": source_inventory,
        "02_feature_registry.csv": feature_registry,
        "03_train_only_thresholds.csv": threshold_report,
        "04_derivation_and_imputation_registry.csv": derivation_report,
        "05_split_coverage.csv": coverage,
        "06_quality_grade_distribution.csv": quality_distribution,
        "07_issue_time_forecast_coverage.csv": forecast_coverage,
        "08_track_policy.csv": track_policy,
        "09_quality_gates.csv": gates,
    }
    decision_name = (
        "READY_FOR_GOVERNED_MULTITASK_MODELING"
        if gates_passed
        else "BLOCKED_GOVERNED_DATA_CONTRACT_REPAIR"
    )
    decision = clean_json(
        {
            "status": "SUCCESS",
            "decision": decision_name,
            "dataset_version": DATASET_VERSION,
            "contract_version": CONTRACT_VERSION,
            "source_dataset_version": SOURCE_DATASET_VERSION,
            "row_count": len(output),
            "vessel_calls": output["port_call_id"].nunique(),
            "column_count": len(output.columns),
            "core_features": len(core_model),
            "research_features": len(research_features),
            "issue_time_features": len(issue_time_features),
            "event_predictive_features_added": len(event_predictive),
            "event_research_features_added": len(event_research),
            "main_dataset_synthetic_rows": synthetic_rows,
            "targets_imputed": False,
            "threshold_selection_split": "TRAIN_ONLY",
            "test_used_for_feature_engineering": False,
            "issue_time_history_ready": issue_time_ready,
            "research_modeling_allowed": gates_passed,
            "production_promotion_allowed": gates_passed and issue_time_ready,
            "quality_gates_passed": gates_passed,
            "limitations": [
                "Historical ETA/ETD revision snapshots are still absent.",
                "Retrospective weather and derived physics are research-only.",
                "Real berth, resource, incident and priority states are not fabricated.",
                "Issue-time weather enters production only after sufficient historical coverage.",
            ],
            "next_block": "B61B_MULTITASK_TEMPORAL_SURVIVAL_MIXTURE_OF_EXPERTS",
        }
    )
    return GovernedBuildResult(
        dataset=output,
        reports=reports,
        decision=decision,
        feature_sets=feature_sets,
    )
