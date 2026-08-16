from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import genpareto


RANDOM_SEED = 20260805
SOURCE_DATASET_VERSION = "b61a-governed-port-call-enrichment-v1"
AUGMENTATION_VERSION = "b61ax-governed-rare-tail-augmentation-v1"
GENERATOR_VERSION = "conditional-evt-local-trajectory-v1"
DEFAULT_EXTERNAL_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "bobaaayoung/container-ship-data-collection"
)
DEFAULT_EXTERNAL_NAME = "KAGGLE_HELSINKI_TALLINN_CONTAINER_SHIPS"
DEFAULT_EXTERNAL_LICENSE = "KAGGLE_DATA_CARD_SOURCE_PORT_AUTHORITIES_LICENSE_REVIEW_REQUIRED"


@dataclass(frozen=True)
class EVTFit:
    threshold_h: float
    shape: float
    scale_h: float
    local_tail_rows: int
    external_tail_rows: int
    local_weight: float
    max_delay_h: float


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


ALIASES = {
    "vessel_id": ("imo", "mmsi", "shipid", "vesselid", "id"),
    "vessel_name": ("ship", "shipname", "vesselname", "name"),
    "departure_port": ("depport", "departureport", "portofdeparture", "fromport"),
    "arrival_port": ("arrport", "arrivalport", "portofarrival", "toport"),
    "scheduled_departure": (
        "etdschedule",
        "scheduledetd",
        "scheduleddeparture",
        "plannedetd",
    ),
    "actual_departure": ("atd", "actualdeparture", "departuretime"),
    "scheduled_arrival": (
        "etaschedule",
        "scheduledeta",
        "scheduledarrival",
        "plannedeta",
    ),
    "actual_arrival": ("ata", "actualarrival", "arrivaltime"),
    "vessel_type": ("vesseltype", "shiptype", "type"),
}


def _find_column(columns: Iterable[str], aliases: tuple[str, ...]) -> str | None:
    lookup = {normalize_column_name(column): column for column in columns}
    for alias in aliases:
        if alias in lookup:
            return str(lookup[alias])
    return None


def standardize_external_voyages(frame: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        target: _find_column(frame.columns, aliases)
        for target, aliases in ALIASES.items()
    }
    required = ("scheduled_departure", "actual_departure")
    if any(mapping[column] is None for column in required):
        return pd.DataFrame()
    result = pd.DataFrame(index=frame.index)
    for target, source in mapping.items():
        result[target] = frame[source] if source is not None else None
    for column in (
        "scheduled_departure",
        "actual_departure",
        "scheduled_arrival",
        "actual_arrival",
    ):
        result[column] = pd.to_datetime(result[column], errors="coerce", utc=True)
    result["vessel_id"] = result["vessel_id"].fillna(result["vessel_name"]).astype(str)
    result["vessel_name"] = result["vessel_name"].fillna("UNKNOWN").astype(str)
    result["departure_port"] = result["departure_port"].fillna("UNKNOWN").astype(str).str.upper()
    result["arrival_port"] = result["arrival_port"].fillna("UNKNOWN").astype(str).str.upper()
    result["departure_delay_h"] = (
        result["actual_departure"] - result["scheduled_departure"]
    ).dt.total_seconds() / 3600.0
    result = result.loc[
        result["actual_departure"].notna()
        & result["scheduled_departure"].notna()
        & result["departure_delay_h"].between(-72.0, 720.0)
    ].copy()
    return result.reset_index(drop=True)


def derive_external_port_calls(voyages: pd.DataFrame) -> pd.DataFrame:
    if voyages.empty:
        return pd.DataFrame()
    source = voyages.sort_values(["vessel_id", "actual_departure"]).copy()
    previous_arrival = source.groupby("vessel_id", sort=False)["actual_arrival"].shift(1)
    previous_port = source.groupby("vessel_id", sort=False)["arrival_port"].shift(1)
    source["actual_ata"] = previous_arrival
    source["port_code"] = source["departure_port"]
    source["port_stay_h"] = (
        source["actual_departure"] - source["actual_ata"]
    ).dt.total_seconds() / 3600.0
    same_port = previous_port.fillna("UNKNOWN").eq(source["departure_port"])
    valid_stay = same_port & source["port_stay_h"].between(0.25, 720.0)
    source.loc[~valid_stay, "port_stay_h"] = np.nan
    source["tail_signal_h"] = pd.to_numeric(
        source["departure_delay_h"], errors="coerce"
    ).clip(lower=0.0)
    missing_delay = source["tail_signal_h"].isna() | source["tail_signal_h"].eq(0.0)
    if source["port_stay_h"].notna().any():
        port_median = source.groupby("port_code")["port_stay_h"].transform("median")
        stay_excess = (source["port_stay_h"] - port_median).clip(lower=0.0)
        source.loc[missing_delay, "tail_signal_h"] = stay_excess[missing_delay]
    return source.reset_index(drop=True)


def fit_governed_evt(
    local_delays_h: np.ndarray,
    external_tail_signal_h: np.ndarray,
    minimum_threshold_h: float = 3.0,
    external_prior_strength: float = 100.0,
    max_delay_h: float = 240.0,
) -> EVTFit:
    local = np.asarray(local_delays_h, dtype="float64")
    external = np.asarray(external_tail_signal_h, dtype="float64")
    local = local[np.isfinite(local)]
    external = external[np.isfinite(external)]
    if len(local) < 100:
        raise ValueError("At least 100 real local calls are required")
    threshold = max(minimum_threshold_h, float(np.quantile(local, 0.95)))
    local_excess = local[local > threshold] - threshold
    external_excess = external[external > threshold] - threshold
    if len(local_excess) < 30:
        raise ValueError("At least 30 real local tail calls are required for EVT")
    if len(external_excess) < 30:
        raise ValueError("At least 30 external tail observations are required")

    def estimate(excess: np.ndarray) -> tuple[float, float]:
        shape, _, scale = genpareto.fit(excess, floc=0.0)
        return float(np.clip(shape, -0.20, 0.50)), float(max(scale, 0.25))

    local_shape, local_scale = estimate(local_excess)
    external_shape, external_scale = estimate(external_excess)
    local_weight = len(local_excess) / (len(local_excess) + external_prior_strength)
    shape = local_weight * local_shape + (1.0 - local_weight) * external_shape
    # External data informs tail shape; local scale remains dominant to preserve units.
    scale_ratio = np.clip(
        np.median(local_excess) / max(np.median(external_excess), 0.25),
        0.25,
        4.0,
    )
    aligned_external_scale = external_scale * scale_ratio
    scale = local_weight * local_scale + (1.0 - local_weight) * aligned_external_scale
    return EVTFit(
        threshold_h=float(threshold),
        shape=float(np.clip(shape, -0.20, 0.50)),
        scale_h=float(max(scale, 0.25)),
        local_tail_rows=len(local_excess),
        external_tail_rows=len(external_excess),
        local_weight=float(local_weight),
        max_delay_h=float(max_delay_h),
    )


def sample_tail_delays(fit: EVTFit, count: int, seed: int = RANDOM_SEED) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype="float64")
    rng = np.random.default_rng(seed)
    excess = genpareto.rvs(
        c=fit.shape,
        loc=0.0,
        scale=fit.scale_h,
        size=count,
        random_state=rng,
    )
    delays = fit.threshold_h + np.maximum(excess, 0.0)
    return np.clip(delays, fit.threshold_h + 1e-3, fit.max_delay_h)


def _stable_call_seed(call_id: str, sequence: int, seed: int) -> str:
    payload = f"{call_id}|{sequence}|{seed}|{GENERATOR_VERSION}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def select_parent_calls(local: pd.DataFrame) -> pd.DataFrame:
    calls = (
        local.sort_values("landmark_at")
        .groupby("port_call_id", as_index=False)
        .agg(
            target_departure_delay_h=("target_departure_delay_h", "first"),
            pressure=("operational_pressure_index", "max"),
            susceptibility=("vessel_delay_susceptibility_bayes", "max"),
            landmarks=("landmark_at", "size"),
        )
    )
    for column in ("pressure", "susceptibility"):
        calls[column] = pd.to_numeric(calls[column], errors="coerce").fillna(0.0)
    pressure_cut = calls["pressure"].quantile(0.60)
    susceptibility_cut = calls["susceptibility"].quantile(0.60)
    candidates = calls.loc[
        calls["target_departure_delay_h"].gt(1.0)
        | calls["pressure"].ge(pressure_cut)
        | calls["susceptibility"].ge(susceptibility_cut)
    ].copy()
    if len(candidates) < 100:
        candidates = calls.nlargest(min(500, len(calls)), ["pressure", "susceptibility"])
    candidates["selection_weight"] = (
        0.1
        + candidates["pressure"].rank(pct=True)
        + candidates["susceptibility"].rank(pct=True)
        + candidates["target_departure_delay_h"].clip(lower=0.0).rank(pct=True)
    )
    candidates["selection_weight"] /= candidates["selection_weight"].sum()
    return candidates


def generate_counterfactual_tail_landmarks(
    local: pd.DataFrame,
    delays_h: np.ndarray,
    external_source_name: str,
    external_checksum: str,
    synthetic_call_weight: float = 0.20,
    seed: int = RANDOM_SEED,
    max_landmarks_per_call: int = 12,
) -> pd.DataFrame:
    if not 0.0 < synthetic_call_weight <= 0.30:
        raise ValueError("Synthetic call weight must be in (0, 0.30]")
    parents = select_parent_calls(local)
    if parents.empty or len(delays_h) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    parent_ids = rng.choice(
        parents["port_call_id"].to_numpy(),
        size=len(delays_h),
        replace=True,
        p=parents["selection_weight"].to_numpy(),
    )
    grouped = {
        str(key): group.sort_values("landmark_at")
        for key, group in local.groupby("port_call_id", sort=False)
    }
    generated = []
    for sequence, (parent_id, delay_h) in enumerate(zip(parent_ids, delays_h)):
        parent = grouped[str(parent_id)].copy()
        if len(parent) > max_landmarks_per_call:
            positions = np.unique(
                np.linspace(0, len(parent) - 1, max_landmarks_per_call).round().astype(int)
            )
            parent = parent.iloc[positions].copy()
        planned_etd = pd.to_datetime(parent["planned_etd"], utc=True, errors="coerce").dropna()
        if planned_etd.empty:
            continue
        planned_etd_value = planned_etd.iloc[0]
        generated_atd = planned_etd_value + pd.Timedelta(hours=float(delay_h))
        parent = parent.loc[parent["landmark_at"].lt(generated_atd)].copy()
        if parent.empty:
            continue
        token = _stable_call_seed(str(parent_id), sequence, seed)
        synthetic_id = f"B61AX-{token}"
        breach_at = planned_etd_value + pd.Timedelta(hours=3.0)
        remaining = (
            generated_atd - pd.to_datetime(parent["landmark_at"], utc=True)
        ).dt.total_seconds() / 3600.0
        until_breach = (
            breach_at - pd.to_datetime(parent["landmark_at"], utc=True)
        ).dt.total_seconds() / 3600.0
        parent["source_parent_port_call_id"] = parent["port_call_id"].astype(str)
        parent["port_call_id"] = synthetic_id
        parent["dataset_version"] = AUGMENTATION_VERSION
        parent["data_origin"] = "SYNTHETIC_TAIL_EXTERNAL_PRIOR"
        parent["split"] = "TRAIN"
        parent["per_call_sample_weight"] = synthetic_call_weight / len(parent)
        parent["training_allowed"] = True
        parent["validation_allowed"] = False
        parent["test_allowed"] = False
        parent["production_claim_allowed"] = False
        parent["synthetic_row"] = True
        parent["targets_imputed"] = False
        parent["target_origin"] = "COUNTERFACTUAL_EVT_TAIL"
        parent["target_actual_atd"] = generated_atd
        parent["target_departure_delay_h"] = float(delay_h)
        parent["target_total_stay_h"] = (
            generated_atd - pd.to_datetime(parent["actual_ata"], utc=True)
        ).dt.total_seconds() / 3600.0
        parent["target_remaining_h"] = remaining.clip(lower=0.0)
        parent["target_delay_gt_1h"] = bool(delay_h > 1.0)
        parent["target_delay_gt_3h"] = bool(delay_h > 3.0)
        parent["target_delay_gt_6h"] = bool(delay_h > 6.0)
        parent["target_departure_delay_class"] = (
            "CRITICAL_GT_6H"
            if delay_h > 6.0
            else "MAJOR_3_6H"
            if delay_h > 3.0
            else "MINOR_1_3H"
        )
        breach_observed = bool(delay_h > 3.0) & until_breach.ge(0.0)
        parent["target_breach_gt3_observed"] = breach_observed
        parent["pre_breach_eligible"] = until_breach.gt(0.0)
        parent["target_breach_or_censor_h"] = np.where(
            breach_observed,
            until_breach.clip(lower=0.0),
            parent["target_remaining_h"],
        )
        for horizon in (6, 12, 24):
            parent[f"target_gt3_breach_within_{horizon}h"] = (
                bool(delay_h > 3.0)
                & until_breach.gt(0.0)
                & until_breach.le(float(horizon))
            )
            parent[f"target_departure_within_{horizon}h"] = remaining.le(float(horizon))
        parent["external_source_name"] = external_source_name
        parent["external_source_checksum"] = external_checksum
        parent["generator_version"] = GENERATOR_VERSION
        parent["generation_seed"] = seed + sequence
        parent["synthetic_tail_delay_h"] = float(delay_h)
        generated.append(parent)
    if not generated:
        return pd.DataFrame()
    result = pd.concat(generated, ignore_index=True)
    return result.sort_values(["port_call_id", "landmark_at"]).reset_index(drop=True)


def distribution_report(
    local_delays: np.ndarray,
    external_signal: np.ndarray,
    synthetic_delays: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, values in (
        ("LOCAL_REAL_TRAIN", local_delays),
        ("EXTERNAL_REAL_REFERENCE", external_signal),
        ("SYNTHETIC_TAIL_TRAIN_ONLY", synthetic_delays),
    ):
        series = pd.Series(values, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "population": name,
                "rows": len(series),
                "mean_h": series.mean(),
                "median_h": series.median(),
                "p90_h": series.quantile(0.90),
                "p95_h": series.quantile(0.95),
                "p99_h": series.quantile(0.99),
                "max_h": series.max(),
                "gt3_pct": 100.0 * series.gt(3.0).mean(),
                "gt6_pct": 100.0 * series.gt(6.0).mean(),
                "gt24_pct": 100.0 * series.gt(24.0).mean(),
            }
        )
    return pd.DataFrame(rows)
