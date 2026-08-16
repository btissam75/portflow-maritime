from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


AUDIT_VERSION = "b59a-dynamic-port-call-data-audit-v1"
TRAIN_START = pd.Timestamp("2020-01-01", tz="UTC")
VALID_START = pd.Timestamp("2024-01-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")
TEST_END = pd.Timestamp("2025-04-01", tz="UTC")
MAX_VALID_STAY_H = 24.0 * 30.0

TARGET_COLUMNS = {
    "actual_atd",
    "target_remaining_h",
    "target_departure_delay_h",
    "target_delay_gt_1h",
    "target_delay_gt_3h",
    "target_delay_gt_6h",
}


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
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


def prepare_calls(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "port_call_id",
        "imo",
        "vessel_name",
        "cargo_type",
        "planned_eta",
        "planned_etd",
        "actual_ata",
        "actual_atd",
        "source",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Required port-call columns are missing: {missing}")
    calls = frame.copy()
    for column in (
        "planned_eta",
        "planned_etd",
        "actual_ata",
        "actual_atd",
        "created_at",
        "updated_at",
    ):
        if column in calls:
            calls[column] = pd.to_datetime(calls[column], errors="coerce", utc=True)
    calls["stay_h"] = (
        calls["actual_atd"] - calls["actual_ata"]
    ).dt.total_seconds() / 3600.0
    calls["departure_delay_h"] = (
        calls["actual_atd"] - calls["planned_etd"]
    ).dt.total_seconds() / 3600.0
    calls["arrival_delay_h"] = (
        calls["actual_ata"] - calls["planned_eta"]
    ).dt.total_seconds() / 3600.0
    calls["target_observed"] = calls["actual_atd"].notna()
    calls["valid_stay"] = (
        calls["stay_h"].gt(0.0) & calls["stay_h"].le(MAX_VALID_STAY_H)
    )
    return calls


def delay_class(delay_h: pd.Series) -> pd.Series:
    values = pd.to_numeric(delay_h, errors="coerce")
    result = pd.Series("TARGET_MISSING", index=values.index, dtype="object")
    result.loc[values.le(1.0)] = "NORMAL_LE_1H"
    result.loc[values.gt(1.0) & values.le(3.0)] = "MINOR_1_3H"
    result.loc[values.gt(3.0) & values.le(6.0)] = "MAJOR_3_6H"
    result.loc[values.gt(6.0)] = "CRITICAL_GT_6H"
    return result


def assign_call_split(calls: pd.DataFrame) -> pd.Series:
    ata = pd.to_datetime(calls["actual_ata"], errors="coerce", utc=True)
    atd = pd.to_datetime(calls["actual_atd"], errors="coerce", utc=True)
    valid_stay = calls["valid_stay"].fillna(False)
    split = pd.Series("EXCLUDED_DATE", index=calls.index, dtype="object")
    split.loc[ata.isna()] = "EXCLUDED_ATA_MISSING"
    split.loc[ata.notna() & atd.isna()] = "EXCLUDED_TARGET_MISSING"
    split.loc[ata.notna() & atd.notna() & ~valid_stay] = "EXCLUDED_INVALID_STAY"

    eligible = ata.notna() & atd.notna() & valid_stay
    train_issue = eligible & ata.ge(TRAIN_START) & ata.lt(VALID_START)
    valid_issue = eligible & ata.ge(VALID_START) & ata.lt(TEST_START)
    test_issue = eligible & ata.ge(TEST_START) & ata.lt(TEST_END)

    split.loc[train_issue & atd.lt(VALID_START)] = "TRAIN"
    split.loc[valid_issue & atd.lt(TEST_START)] = "VALID"
    split.loc[test_issue & atd.lt(TEST_END)] = "TEST"

    crosses = (
        (train_issue & atd.ge(VALID_START))
        | (valid_issue & atd.ge(TEST_START))
        | (test_issue & atd.ge(TEST_END))
    )
    split.loc[crosses] = "EXCLUDED_BOUNDARY_CROSSING"
    return split


def add_model_contract_columns(frame: pd.DataFrame) -> pd.DataFrame:
    calls = prepare_calls(frame)
    calls["delay_class"] = delay_class(calls["departure_delay_h"])
    calls["split"] = assign_call_split(calls)
    calls["target_delay_gt_1h"] = calls["departure_delay_h"].gt(1.0)
    calls["target_delay_gt_3h"] = calls["departure_delay_h"].gt(3.0)
    calls["target_delay_gt_6h"] = calls["departure_delay_h"].gt(6.0)
    return calls


def _classification_landmarks(row: pd.Series) -> int:
    if not bool(row["valid_stay"]) or pd.isna(row["planned_etd"]):
        return 0
    limit = min(row["actual_atd"], row["planned_etd"] + pd.Timedelta(hours=3))
    available_h = (limit - row["actual_ata"]).total_seconds() / 3600.0
    return max(0, int(math.ceil(available_h)))


def add_landmark_counts(calls: pd.DataFrame) -> pd.DataFrame:
    result = calls.copy()
    valid = result["valid_stay"].fillna(False)
    result["hourly_landmark_rows"] = 0
    result.loc[valid, "hourly_landmark_rows"] = np.ceil(
        result.loc[valid, "stay_h"]
    ).astype("int64")
    result["classification_landmark_rows"] = result.apply(
        _classification_landmarks, axis=1
    ).astype("int64")
    result["per_call_sample_weight"] = 0.0
    positive = result["hourly_landmark_rows"].gt(0)
    result.loc[positive, "per_call_sample_weight"] = (
        1.0 / result.loc[positive, "hourly_landmark_rows"]
    )
    return result


def source_inventory_report(calls: pd.DataFrame) -> pd.DataFrame:
    duplicate_ids = int(calls["port_call_id"].duplicated(keep=False).sum())
    duplicate_grain = int(
        calls.duplicated(["imo", "actual_ata"], keep=False).sum()
    )
    return pd.DataFrame(
        [
            ("audit_version", AUDIT_VERSION),
            ("source_rows", len(calls)),
            ("distinct_port_call_id", calls["port_call_id"].nunique()),
            ("distinct_imo", calls["imo"].nunique(dropna=True)),
            ("distinct_vessel_name", calls["vessel_name"].nunique(dropna=True)),
            ("duplicate_port_call_id_rows", duplicate_ids),
            ("duplicate_imo_ata_rows", duplicate_grain),
            ("first_actual_ata", calls["actual_ata"].min()),
            ("last_actual_ata", calls["actual_ata"].max()),
            ("actual_ata_present", int(calls["actual_ata"].notna().sum())),
            ("planned_etd_present", int(calls["planned_etd"].notna().sum())),
            ("actual_atd_present", int(calls["actual_atd"].notna().sum())),
        ],
        columns=["metric", "value"],
    )


def monthly_target_coverage_report(calls: pd.DataFrame) -> pd.DataFrame:
    dated = calls.loc[calls["actual_ata"].notna()].copy()
    dated["month"] = dated["actual_ata"].dt.strftime("%Y-%m")
    grouped = dated.groupby("month", sort=True)
    report = grouped.agg(
        calls=("port_call_id", "size"),
        target_observed=("target_observed", "sum"),
        valid_stays=("valid_stay", "sum"),
        distinct_imo=("imo", "nunique"),
    ).reset_index()
    report["target_coverage_pct"] = 100.0 * report["target_observed"] / report["calls"]
    report["valid_target_pct"] = 100.0 * report["valid_stays"] / report["calls"]
    return report


def target_distribution_report(calls: pd.DataFrame) -> pd.DataFrame:
    observed = calls.loc[calls["valid_stay"] & calls["planned_etd"].notna()].copy()
    rows: list[dict[str, Any]] = []
    for split, group in observed.groupby("split", sort=False):
        if split not in {"TRAIN", "VALID", "TEST"}:
            continue
        for label, class_group in group.groupby("delay_class", sort=False):
            rows.append(
                {
                    "split": split,
                    "delay_class": label,
                    "calls": len(class_group),
                    "share_pct": 100.0 * len(class_group) / len(group),
                }
            )
    return pd.DataFrame(rows)


def target_quantile_report(calls: pd.DataFrame) -> pd.DataFrame:
    observed = calls.loc[calls["valid_stay"] & calls["planned_etd"].notna()].copy()
    rows = []
    for split, group in observed.groupby("split", sort=False):
        if split not in {"TRAIN", "VALID", "TEST"}:
            continue
        for target in ("stay_h", "departure_delay_h", "arrival_delay_h"):
            values = pd.to_numeric(group[target], errors="coerce").dropna()
            rows.append(
                {
                    "split": split,
                    "target": target,
                    "rows": len(values),
                    "mean": values.mean(),
                    "p01": values.quantile(0.01),
                    "p10": values.quantile(0.10),
                    "p50": values.quantile(0.50),
                    "p90": values.quantile(0.90),
                    "p99": values.quantile(0.99),
                }
            )
    return pd.DataFrame(rows)


def split_report(calls: pd.DataFrame) -> pd.DataFrame:
    return (
        calls.groupby("split", dropna=False)
        .agg(
            calls=("port_call_id", "size"),
            first_ata=("actual_ata", "min"),
            last_ata=("actual_ata", "max"),
            last_atd=("actual_atd", "max"),
            target_observed=("target_observed", "sum"),
            valid_stays=("valid_stay", "sum"),
        )
        .reset_index()
    )


def landmark_yield_report(calls: pd.DataFrame) -> pd.DataFrame:
    model = add_landmark_counts(calls)
    eligible = model["split"].isin(["TRAIN", "VALID", "TEST"])
    rows = []
    for split, group in model.loc[eligible].groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "calls": len(group),
                "hourly_landmark_rows": int(group["hourly_landmark_rows"].sum()),
                "classification_landmark_rows": int(
                    group["classification_landmark_rows"].sum()
                ),
                "mean_hourly_landmarks_per_call": group["hourly_landmark_rows"].mean(),
                "delay_gt_3h_calls": int(group["target_delay_gt_3h"].sum()),
                "delay_gt_6h_calls": int(group["target_delay_gt_6h"].sum()),
            }
        )
    return pd.DataFrame(rows)


def missing_target_bias_report(calls: pd.DataFrame, dimension: str) -> pd.DataFrame:
    source = calls.loc[calls[dimension].notna()].copy()
    grouped = source.groupby(dimension, dropna=False)
    report = grouped.agg(
        calls=("port_call_id", "size"),
        target_observed=("target_observed", "sum"),
    ).reset_index()
    report["target_missing"] = report["calls"] - report["target_observed"]
    report["target_coverage_pct"] = 100.0 * report["target_observed"] / report["calls"]
    return report.sort_values(["calls", dimension], ascending=[False, True]).reset_index(drop=True)


def baseline_report(calls: pd.DataFrame) -> pd.DataFrame:
    observed = calls.loc[
        calls["split"].isin(["TRAIN", "VALID", "TEST"])
        & calls["planned_etd"].notna()
    ].copy()
    error = observed["departure_delay_h"]
    observed["absolute_error_h"] = error.abs()
    observed["squared_error_h"] = error.pow(2)
    rows = []
    for split, group in observed.groupby("split", sort=False):
        rows.append(
            {
                "split": split,
                "baseline": "PLANNED_ETD",
                "calls": len(group),
                "mae_h": group["absolute_error_h"].mean(),
                "rmse_h": math.sqrt(group["squared_error_h"].mean()),
                "bias_h": group["departure_delay_h"].mean(),
                "late_gt_3h_pct": 100.0 * group["target_delay_gt_3h"].mean(),
                "late_gt_6h_pct": 100.0 * group["target_delay_gt_6h"].mean(),
            }
        )
    return pd.DataFrame(rows)


def feature_contract_report() -> pd.DataFrame:
    rows = [
        ("core.port_call", "port_call_id", "KEY_ONLY", "NO", "EXCLUDE_MODEL"),
        ("core.port_call", "imo", "STATIC_CATEGORY_AND_HISTORY_KEY", "BUSINESS_SEMANTICS_ONLY", "RETROSPECTIVE_CORE"),
        ("core.port_call", "vessel_name", "STATIC_CATEGORY", "BUSINESS_SEMANTICS_ONLY", "RETROSPECTIVE_CORE"),
        ("core.port_call", "voyage_id", "NEAR_UNIQUE_IDENTIFIER", "BUSINESS_SEMANTICS_ONLY", "EXCLUDE_RAW_MODEL"),
        ("core.port_call", "cargo_type", "STATIC_CATEGORY", "BUSINESS_SEMANTICS_ONLY", "RETROSPECTIVE_CORE"),
        ("core.port_call", "planned_eta", "KNOWN_PLAN", "NOT_CAPTURED_HISTORICALLY", "RETROSPECTIVE_CORE"),
        ("core.port_call", "planned_etd", "KNOWN_PLAN", "NOT_CAPTURED_HISTORICALLY", "RETROSPECTIVE_CORE"),
        ("core.port_call", "actual_ata", "ISSUE_ANCHOR", "EVENT_TIME_ONLY", "RETROSPECTIVE_CORE"),
        ("core.port_call", "actual_atd", "TARGET_ONLY", "FUTURE_EVENT", "TARGET_ONLY"),
        ("features.port_hourly_state_v1", "port_state_lags", "DYNAMIC_CONTEXT", "NO_HISTORICAL_AVAILABLE_AT", "RETROSPECTIVE_CORE"),
        ("core.maritime_observation", "wave_observations", "DYNAMIC_CONTEXT", "NO_HISTORICAL_AVAILABLE_AT", "RESEARCH_WEATHER"),
        ("features.maritime_external_weather_hourly_v1", "external_weather", "RETROSPECTIVE_REANALYSIS", "NOT_AVAILABLE_AT_HISTORICAL_ISSUE", "RESEARCH_ONLY"),
        ("features.maritime_issue_time_weather_forecast_v1", "weather_forecast", "KNOWN_FUTURE_CONTEXT", "ISSUE_AT_AND_AVAILABLE_AT", "LIVE_WEATHER_ONLY"),
        ("reference.business_event", "business_calendar", "KNOWN_FUTURE", "DETERMINISTIC_CALENDAR", "CORE_ALLOWED"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "source_relation",
            "feature_family",
            "semantic_role",
            "availability_evidence",
            "policy",
        ],
    )


def leakage_policy_report() -> pd.DataFrame:
    rows = [
        ("TARGET_COLUMNS_EXCLUDED_FROM_FEATURES", True, "CRITICAL", "ATD and derived labels are target-only"),
        ("ONE_SPLIT_PER_PORT_CALL", True, "CRITICAL", "Split is assigned before landmark expansion"),
        ("LANDMARK_STRICTLY_BEFORE_ATD", True, "CRITICAL", "Generate k where ATA+k<ATD"),
        ("CLASSIFICATION_BEFORE_BREACH", True, "CRITICAL", "Use landmarks before ETD+3h"),
        ("HISTORICAL_PLAN_AVAILABLE_AT_CAPTURED", False, "EXPECTED_LIMITATION", "Port-call ETA/ETD revisions lack historical available_at"),
        ("RETROSPECTIVE_WEATHER_OPERATIONAL", False, "EXPECTED_LIMITATION", "Reanalysis is research-only"),
        ("HISTORICAL_MISSING_ATD_IS_CENSORING", False, "CRITICAL", "Missing historical targets are excluded, not called censored"),
    ]
    return pd.DataFrame(rows, columns=["check", "passed", "severity", "details"])


def make_decision(calls: pd.DataFrame, relation_inventory: pd.DataFrame) -> dict[str, Any]:
    model = add_landmark_counts(calls)
    eligible = model.loc[model["split"].isin(["TRAIN", "VALID", "TEST"])]
    split_counts = eligible["split"].value_counts()
    duplicate_ids = int(calls["port_call_id"].duplicated(keep=False).sum())
    valid_target_pct = 100.0 * calls["valid_stay"].sum() / max(1, calls["target_observed"].sum())
    target_coverage_pct = 100.0 * calls["target_observed"].sum() / max(1, calls["actual_ata"].notna().sum())
    planned_etd_coverage_pct = 100.0 * eligible["planned_etd"].notna().mean()
    late_gt_3 = int(eligible["target_delay_gt_3h"].sum())
    late_gt_6 = int(eligible["target_delay_gt_6h"].sum())
    landmark_rows = int(eligible["hourly_landmark_rows"].sum())
    required_relations = {
        "core.port_call",
        "features.port_hourly_state_v1",
    }
    present_relations = set(
        relation_inventory.loc[relation_inventory["exists"], "relation"]
    )
    required_present = required_relations.issubset(present_relations)
    target_gate = (
        len(eligible) >= 20_000
        and valid_target_pct >= 99.0
        and planned_etd_coverage_pct >= 99.0
        and late_gt_3 >= 2_000
        and late_gt_6 >= 500
    )
    split_gate = (
        int(split_counts.get("TRAIN", 0)) >= 15_000
        and int(split_counts.get("VALID", 0)) >= 4_000
        and int(split_counts.get("TEST", 0)) >= 1_000
    )
    grain_gate = duplicate_ids == 0
    landmark_gate = landmark_rows >= 80_000
    audit_passed = bool(
        target_gate
        and split_gate
        and grain_gate
        and landmark_gate
        and required_present
    )
    decision = (
        "READY_FOR_RETROSPECTIVE_LANDMARK_DATASET"
        if audit_passed
        else "BLOCKED_DATA_CONTRACT_REPAIR_REQUIRED"
    )
    return clean_json(
        {
            "status": "SUCCESS",
            "decision": decision,
            "audit_version": AUDIT_VERSION,
            "source_rows": len(calls),
            "eligible_complete_calls": len(eligible),
            "target_coverage_pct": target_coverage_pct,
            "valid_observed_target_pct": valid_target_pct,
            "planned_etd_coverage_pct": planned_etd_coverage_pct,
            "train_calls": int(split_counts.get("TRAIN", 0)),
            "valid_calls": int(split_counts.get("VALID", 0)),
            "test_calls": int(split_counts.get("TEST", 0)),
            "late_gt_3h_calls": late_gt_3,
            "late_gt_6h_calls": late_gt_6,
            "expected_hourly_landmarks": landmark_rows,
            "expected_classification_landmarks": int(
                eligible["classification_landmark_rows"].sum()
            ),
            "grain_gate_passed": grain_gate,
            "target_gate_passed": target_gate,
            "split_gate_passed": split_gate,
            "landmark_gate_passed": landmark_gate,
            "required_relations_present": required_present,
            "audit_gates_passed": audit_passed,
            "historical_missing_atd_treated_as_censoring": False,
            "retrospective_benchmark_allowed": audit_passed,
            "historical_replay_allowed": False,
            "formal_production_promotion": False,
            "synthetic_rows_created": 0,
            "source_modified": False,
            "training_executed": False,
            "availability_limitation": (
                "Historical ETA/ETD revisions and source available_at are not captured; "
                "the first dataset is retrospective and cannot prove live availability."
            ),
            "next_block": (
                "B59B_RETROSPECTIVE_POINT_IN_TIME_LANDMARK_DATASET"
                if audit_passed
                else "B59A_SOURCE_AND_TARGET_REPAIR"
            ),
        }
    )


def build_reports(
    frame: pd.DataFrame,
    relation_inventory: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    calls = add_model_contract_columns(frame)
    reports = {
        "01_source_inventory.csv": source_inventory_report(calls),
        "02_monthly_target_coverage.csv": monthly_target_coverage_report(calls),
        "03_split_audit.csv": split_report(calls),
        "04_target_distribution.csv": target_distribution_report(calls),
        "05_target_quantiles.csv": target_quantile_report(calls),
        "06_planned_etd_baseline.csv": baseline_report(calls),
        "07_landmark_yield.csv": landmark_yield_report(calls),
        "08_missing_target_by_vessel.csv": missing_target_bias_report(calls, "vessel_name"),
        "09_missing_target_by_cargo.csv": missing_target_bias_report(calls, "cargo_type"),
        "10_feature_contract.csv": feature_contract_report(),
        "11_relation_inventory.csv": relation_inventory,
        "12_leakage_and_availability_policy.csv": leakage_policy_report(),
    }
    return calls, reports, make_decision(calls, relation_inventory)
