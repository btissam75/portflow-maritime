from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


AUDIT_VERSION = "b58ca-prefect-weather-missingness-v1"
WEATHER_VARIABLES = (
    "wave_height_m",
    "wave_period_s",
    "wave_direction_deg",
    "wind_speed_ms",
    "wind_direction_deg",
    "surface_current_ms",
    "visibility_m",
    "pressure_hpa",
)
WAVE_VARIABLES = (
    "wave_height_m",
    "wave_period_s",
    "wave_direction_deg",
)
HORIZONS_H = (6, 12, 24, 48, 72)

# Broad integrity bounds, not operational alert thresholds and never generators.
INTEGRITY_BOUNDS = {
    "wave_height_m": (0.0, 30.0),
    "wave_period_s": (0.0, 40.0),
    "wave_direction_deg": (0.0, 360.0),
    "wind_speed_ms": (0.0, 100.0),
    "wind_direction_deg": (0.0, 360.0),
    "surface_current_ms": (0.0, 10.0),
    "visibility_m": (0.0, 200_000.0),
    "pressure_hpa": (800.0, 1_100.0),
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Required source columns are missing: {missing}")


def prepare_source(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "observed_at",
        "source",
        "latitude",
        "longitude",
        "quality_flag",
        *WEATHER_VARIABLES,
    }
    _require_columns(frame, required)
    source = frame.copy()
    source["observed_at"] = pd.to_datetime(
        source["observed_at"], errors="coerce", utc=True
    )
    if source.empty:
        raise ValueError("The weather source is empty")
    if source["observed_at"].isna().any():
        raise ValueError("The weather source contains invalid observed_at values")
    for column in WEATHER_VARIABLES:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["quality_flag"] = pd.to_numeric(
        source["quality_flag"], errors="coerce"
    ).fillna(1).astype("int16")
    return source.sort_values(
        ["observed_at", "source", "latitude", "longitude"],
        kind="stable",
    ).reset_index(drop=True)


def _circular_hourly_mean(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    if valid.empty:
        return float("nan")
    radians = np.deg2rad(valid.to_numpy(dtype="float64"))
    return float(np.mod(np.degrees(np.arctan2(np.sin(radians).mean(),
                                               np.cos(radians).mean())), 360.0))


def build_hourly_grid(source: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source(source)
    quality = source.loc[source["quality_flag"].eq(0)].copy()
    if quality.empty:
        raise ValueError("No quality_flag=0 weather observation is available")
    quality["hour"] = quality["observed_at"].dt.floor("h")

    continuous = [
        variable
        for variable in WEATHER_VARIABLES
        if variable not in {"wave_direction_deg", "wind_direction_deg"}
    ]
    grouped = quality.groupby("hour", sort=True)
    hourly = grouped[continuous].mean().reset_index()
    for variable in ("wave_direction_deg", "wind_direction_deg"):
        circular = grouped[variable].apply(_circular_hourly_mean)
        hourly = hourly.merge(
            circular.rename(variable).reset_index(), on="hour", how="left"
        )
    counts = grouped.agg(
        observation_count=("observed_at", "size"),
        source_count=("source", "nunique"),
    ).reset_index()
    hourly = hourly.merge(counts, on="hour", how="left")

    first_hour = source["observed_at"].min().floor("h")
    last_hour = source["observed_at"].max().floor("h")
    grid = pd.DataFrame(
        {"observed_at": pd.date_range(first_hour, last_hour, freq="h", tz="UTC")}
    )
    hourly = grid.merge(
        hourly.rename(columns={"hour": "observed_at"}),
        on="observed_at",
        how="left",
        validate="one_to_one",
    )
    hourly["observation_count"] = hourly["observation_count"].fillna(0).astype("int32")
    hourly["source_count"] = hourly["source_count"].fillna(0).astype("int16")
    return hourly


def inventory_report(source: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source(source)
    quality = source["quality_flag"].eq(0)
    duplicate_rows = int(
        source.duplicated(
            ["observed_at", "source", "latitude", "longitude"], keep=False
        ).sum()
    )
    rows = [
        ("audit_version", AUDIT_VERSION, "code contract"),
        ("source_relation", "core.maritime_observation", "read-only canonical source"),
        ("source_rows", len(source), "raw rows inspected"),
        ("quality_rows", int(quality.sum()), "quality_flag=0 rows"),
        ("hourly_rows", len(hourly), "complete UTC hourly grid"),
        ("first_observed_at", source["observed_at"].min(), "first event time"),
        ("last_observed_at", source["observed_at"].max(), "last event time"),
        ("source_count", source["source"].nunique(), "distinct providers"),
        ("duplicate_primary_key_rows", duplicate_rows, "must be zero"),
        ("available_at_present", False, "historical availability is not recorded"),
        ("synthetic_rows_created", 0, "audit does not generate data"),
        ("source_modified", False, "Bronze and Core remain immutable"),
    ]
    return pd.DataFrame(rows, columns=["item", "value", "interpretation"])


def variable_coverage_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(hourly)
    for variable in WEATHER_VARIABLES:
        values = hourly[variable]
        available = int(values.notna().sum())
        coverage = available / total if total else 0.0
        if available == 0:
            state = "STRUCTURALLY_ABSENT"
            repair = "EXTERNAL_DATA_REQUIRED"
        elif available == total:
            state = "COMPLETE"
            repair = "NO_IMPUTATION_NEEDED"
        elif coverage >= 0.95:
            state = "SPARSE_GAPS"
            repair = "GAP_REPAIR_BENCHMARK_CANDIDATE"
        elif coverage >= 0.20:
            state = "MATERIAL_GAPS"
            repair = "MODEL_IMPUTATION_BENCHMARK_CANDIDATE"
        else:
            state = "INSUFFICIENT_OBSERVED_SUPPORT"
            repair = "EXTERNAL_DATA_OR_SCOPE_REDUCTION"
        rows.append(
            {
                "variable": variable,
                "family": "WAVE" if variable in WAVE_VARIABLES else "WEATHER_AUXILIARY",
                "hourly_rows": total,
                "available_rows": available,
                "missing_rows": total - available,
                "coverage_pct": 100.0 * coverage,
                "distinct_values": int(values.nunique(dropna=True)),
                "data_state": state,
                "repairability": repair,
                "threshold_generation_allowed": False,
            }
        )
    return pd.DataFrame(rows)


def temporal_coverage_report(hourly: pd.DataFrame) -> pd.DataFrame:
    base = hourly[["observed_at", *WEATHER_VARIABLES]].copy()
    base["period_start"] = pd.to_datetime(
        base["observed_at"].dt.strftime("%Y-%m-01"), utc=True
    )
    rows: list[dict[str, Any]] = []
    for period, block in base.groupby("period_start", sort=True):
        for variable in WEATHER_VARIABLES:
            available = int(block[variable].notna().sum())
            rows.append(
                {
                    "period_start": period,
                    "year": int(period.year),
                    "month": int(period.month),
                    "variable": variable,
                    "hours": int(len(block)),
                    "available_hours": available,
                    "missing_hours": int(len(block) - available),
                    "coverage_pct": 100.0 * available / len(block),
                }
            )
    return pd.DataFrame(rows)


def missing_run_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    timestamps = hourly["observed_at"].reset_index(drop=True)
    for variable in WEATHER_VARIABLES:
        missing = hourly[variable].isna().reset_index(drop=True)
        groups = missing.ne(missing.shift(fill_value=False)).cumsum()
        for _, indexes in missing[missing].groupby(groups[missing]).groups.items():
            positions = np.asarray(list(indexes), dtype="int64")
            start = int(positions.min())
            end = int(positions.max())
            length = int(end - start + 1)
            rows.append(
                {
                    "variable": variable,
                    "gap_start": timestamps.iloc[start],
                    "gap_end": timestamps.iloc[end],
                    "gap_hours": length,
                    "gap_class": (
                        "1_3H" if length <= 3 else
                        "4_12H" if length <= 12 else
                        "13_72H" if length <= 72 else
                        "GT_72H"
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=["variable", "gap_start", "gap_end", "gap_hours", "gap_class"],
    )


def gap_summary_report(gaps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in WEATHER_VARIABLES:
        subset = gaps.loc[gaps["variable"].eq(variable)] if not gaps.empty else gaps
        rows.append(
            {
                "variable": variable,
                "gap_count": int(len(subset)),
                "missing_hours": int(subset["gap_hours"].sum()) if len(subset) else 0,
                "max_gap_hours": int(subset["gap_hours"].max()) if len(subset) else 0,
                "median_gap_hours": float(subset["gap_hours"].median()) if len(subset) else 0.0,
                "gaps_1_3h": int(subset["gap_class"].eq("1_3H").sum()) if len(subset) else 0,
                "gaps_4_12h": int(subset["gap_class"].eq("4_12H").sum()) if len(subset) else 0,
                "gaps_13_72h": int(subset["gap_class"].eq("13_72H").sum()) if len(subset) else 0,
                "gaps_gt_72h": int(subset["gap_class"].eq("GT_72H").sum()) if len(subset) else 0,
            }
        )
    return pd.DataFrame(rows)


def pairwise_availability_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total = len(hourly)
    for left in WEATHER_VARIABLES:
        for right in WEATHER_VARIABLES:
            both = int(hourly[[left, right]].notna().all(axis=1).sum())
            rows.append(
                {
                    "left_variable": left,
                    "right_variable": right,
                    "both_available_rows": both,
                    "both_available_pct": 100.0 * both / total if total else 0.0,
                }
            )
    return pd.DataFrame(rows)


def source_coverage_report(source: pd.DataFrame) -> pd.DataFrame:
    source = prepare_source(source)
    rows: list[dict[str, Any]] = []
    for source_name, block in source.groupby("source", dropna=False, sort=True):
        quality = block.loc[block["quality_flag"].eq(0)]
        for variable in WEATHER_VARIABLES:
            rows.append(
                {
                    "source": source_name,
                    "variable": variable,
                    "source_rows": int(len(block)),
                    "quality_rows": int(len(quality)),
                    "available_quality_rows": int(quality[variable].notna().sum()),
                    "coverage_pct": (
                        100.0 * quality[variable].notna().mean() if len(quality) else 0.0
                    ),
                    "first_observed_at": block["observed_at"].min(),
                    "last_observed_at": block["observed_at"].max(),
                }
            )
    return pd.DataFrame(rows)


def horizon_availability_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable in WEATHER_VARIABLES:
        available = hourly[variable].notna()
        for horizon in HORIZONS_H:
            pair = available & available.shift(-horizon, fill_value=False)
            possible = max(len(hourly) - horizon, 0)
            rows.append(
                {
                    "variable": variable,
                    "horizon_h": horizon,
                    "possible_pairs": possible,
                    "observed_pairs": int(pair.iloc[:possible].sum()) if possible else 0,
                    "pair_coverage_pct": (
                        100.0 * pair.iloc[:possible].mean() if possible else 0.0
                    ),
                    "usable_as_observed_target": bool(
                        possible and pair.iloc[:possible].mean() >= 0.95
                    ),
                }
            )
    return pd.DataFrame(rows)


def missingness_mechanism_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    month = hourly["observed_at"].dt.month
    hour = hourly["observed_at"].dt.hour
    for variable in WEATHER_VARIABLES:
        missing = hourly[variable].isna().astype("float64")
        missing_rate = float(missing.mean())
        if missing_rate == 0.0:
            signal = "COMPLETE"
        elif missing_rate == 1.0:
            signal = "STRUCTURAL_ABSENCE"
        else:
            monthly = missing.groupby(month).mean()
            hourly_rate = missing.groupby(hour).mean()
            spread = max(float(monthly.max() - monthly.min()),
                         float(hourly_rate.max() - hourly_rate.min()))
            signal = (
                "TIME_DEPENDENT_MISSINGNESS_SIGNAL"
                if spread >= 0.05
                else "NO_STRONG_TEMPORAL_PATTERN_DETECTED"
            )
        monthly = missing.groupby(month).mean()
        hourly_rate = missing.groupby(hour).mean()
        lag1 = missing.corr(missing.shift(1)) if missing.nunique() > 1 else np.nan
        rows.append(
            {
                "variable": variable,
                "missing_rate_pct": 100.0 * missing_rate,
                "monthly_missing_spread_pp": 100.0 * float(monthly.max() - monthly.min()),
                "hourly_missing_spread_pp": 100.0 * float(hourly_rate.max() - hourly_rate.min()),
                "missing_flag_lag1_correlation": lag1,
                "diagnostic_signal": signal,
                "mar_mnar_identifiable": False,
                "interpretation": (
                    "Observed data alone cannot prove MAR versus MNAR; this is a diagnostic only."
                ),
            }
        )
    return pd.DataFrame(rows)


def integrity_bounds_report(hourly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variable, (lower, upper) in INTEGRITY_BOUNDS.items():
        values = hourly[variable].dropna()
        invalid = int(((values < lower) | (values >= upper)).sum())
        rows.append(
            {
                "variable": variable,
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "observed_rows": int(len(values)),
                "outside_bound_rows": invalid,
                "outside_bound_pct": 100.0 * invalid / len(values) if len(values) else 0.0,
                "bound_role": "BROAD_INTEGRITY_GUARDRAIL_ONLY",
                "generation_role": "DO_NOT_GENERATE_VALUES_FROM_BOUNDS",
            }
        )
    return pd.DataFrame(rows)


def generation_policy_report() -> pd.DataFrame:
    rows = [
        ("SOURCE_IMMUTABILITY", "PASS", "Bronze and Core are read-only"),
        ("TRAIN_ONLY_FIT", "REQUIRED", "Any future imputer fits on TRAIN only"),
        ("VALID_TEST_REAL_ONLY", "REQUIRED", "VALID and TEST targets remain observed"),
        ("MISSINGNESS_MASK", "REQUIRED", "Keep an is_observed flag per repaired feature"),
        ("UNCERTAINTY", "REQUIRED", "Store uncertainty and generator version"),
        ("STRUCTURAL_ABSENCE", "BLOCK", "A 100% absent variable needs external observations"),
        ("THRESHOLD_SYNTHESIS", "PROHIBITED", "Thresholds constrain values; they do not create measurements"),
        ("EXTREME_TAIL", "SEPARATE_VALIDATION", "Validate rare extremes independently"),
        ("SYNTHETIC_TARGET", "PROHIBITED", "Never replace evaluation targets with synthetic labels"),
    ]
    return pd.DataFrame(rows, columns=["control", "status", "rule"])


def make_decision(
    source: pd.DataFrame,
    hourly: pd.DataFrame,
    coverage: pd.DataFrame,
    gaps: pd.DataFrame,
    integrity: pd.DataFrame,
) -> dict[str, Any]:
    lookup = coverage.set_index("variable")
    absent = coverage.loc[
        coverage["data_state"].eq("STRUCTURALLY_ABSENT"), "variable"
    ].tolist()
    complete = coverage.loc[
        coverage["data_state"].eq("COMPLETE"), "variable"
    ].tolist()
    partial = coverage.loc[
        ~coverage["data_state"].isin(["COMPLETE", "STRUCTURALLY_ABSENT"]),
        "variable",
    ].tolist()
    candidates = coverage.loc[
        coverage["repairability"].isin(
            ["GAP_REPAIR_BENCHMARK_CANDIDATE", "MODEL_IMPUTATION_BENCHMARK_CANDIDATE"]
        ),
        "variable",
    ].tolist()
    low_support = coverage.loc[
        coverage["data_state"].eq("INSUFFICIENT_OBSERVED_SUPPORT"), "variable"
    ].tolist()
    wave_complete = all(
        lookup.loc[variable, "data_state"] == "COMPLETE"
        for variable in WAVE_VARIABLES
    )
    wave_supported = all(
        float(lookup.loc[variable, "coverage_pct"]) >= 95.0
        for variable in WAVE_VARIABLES
    )
    full_weather_supported = bool(coverage["coverage_pct"].ge(95.0).all())
    invalid_rows = int(integrity["outside_bound_rows"].sum())
    if absent:
        decision = "READY_FOR_WAVE_ONLY_NEED_EXTERNAL_DATA_FOR_FULL_WEATHER"
        next_block = "B58C_B_EXTERNAL_WEATHER_INGESTION_AND_GAP_REPAIR_BENCHMARK"
    elif low_support:
        decision = "NEED_EXTERNAL_DATA_OR_SCOPE_REDUCTION"
        next_block = "B58C_B_EXTERNAL_WEATHER_INGESTION_AND_SCOPE_DECISION"
    elif candidates:
        decision = "READY_FOR_TRAIN_ONLY_IMPUTATION_BENCHMARK"
        next_block = "B58C_B_MASKED_IMPUTATION_BACKTEST"
    else:
        decision = "NO_SYNTHETIC_FILL_NEEDED"
        next_block = "B58C_C_FEATURE_ENRICHMENT_WITH_OBSERVED_DATA"
    return clean_json(
        {
            "status": "SUCCESS",
            "decision": decision,
            "audit_version": AUDIT_VERSION,
            "source_rows": len(source),
            "hourly_rows": len(hourly),
            "first_observed_at": hourly["observed_at"].min(),
            "last_observed_at": hourly["observed_at"].max(),
            "wave_track_supported": wave_supported,
            "wave_track_complete": wave_complete,
            "full_weather_track_supported": full_weather_supported,
            "complete_variables": complete,
            "partially_missing_variables": partial,
            "imputation_benchmark_candidates": candidates,
            "insufficient_support_variables": low_support,
            "structurally_absent_variables": absent,
            "maximum_gap_hours": int(gaps["gap_hours"].max()) if len(gaps) else 0,
            "outside_integrity_bound_rows": invalid_rows,
            "audit_gates_passed": True,
            "data_integrity_gates_passed": invalid_rows == 0,
            "synthetic_rows_created": 0,
            "source_modified": False,
            "training_executed": False,
            "threshold_generation_allowed": False,
            "historical_replay_allowed": False,
            "formal_production_promotion": False,
            "imputation_policy": "FIT_TRAIN_ONLY_VALIDATE_AND_TEST_ON_OBSERVED_VALUES",
            "next_block": next_block,
        }
    )


def build_reports(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    source = prepare_source(source)
    hourly = build_hourly_grid(source)
    coverage = variable_coverage_report(hourly)
    gaps = missing_run_report(hourly)
    integrity = integrity_bounds_report(hourly)
    reports = {
        "01_source_inventory.csv": inventory_report(source, hourly),
        "02_variable_coverage.csv": coverage,
        "03_temporal_coverage_monthly.csv": temporal_coverage_report(hourly),
        "04_missing_runs.csv": gaps,
        "05_gap_summary.csv": gap_summary_report(gaps),
        "06_missingness_mechanism.csv": missingness_mechanism_report(hourly),
        "07_pairwise_availability.csv": pairwise_availability_report(hourly),
        "08_source_coverage.csv": source_coverage_report(source),
        "09_horizon_target_availability.csv": horizon_availability_report(hourly),
        "10_integrity_bounds.csv": integrity,
        "11_generation_policy.csv": generation_policy_report(),
    }
    decision = make_decision(source, hourly, coverage, gaps, integrity)
    return hourly, reports, decision
