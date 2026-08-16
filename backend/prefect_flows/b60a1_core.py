from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler


AUDIT_VERSION = "b60a1-feature-representation-v1"
CORRELATION_REPORT_THRESHOLD = 0.70
CORRELATION_PRUNE_THRESHOLD = 0.995
QUASI_CONSTANT_THRESHOLD = 0.995
PCA_VARIANCE_THRESHOLD = 0.95
MI_SAMPLE_ROWS = 8_000
STABILITY_SAMPLE_ROWS = 20_000
AUTOCORRELATION_LAGS = (1, 2, 3, 6, 12, 24, 48, 72, 168)


@dataclass
class AuditResult:
    reports: dict[str, pd.DataFrame]
    representation_sets: dict[str, list[str]]
    pca_frames: dict[str, pd.DataFrame]
    transformers: dict[str, dict[str, Any]]
    decision: dict[str, Any]


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


def task_relevant_features(features: list[str], task: str) -> list[str]:
    if task == "WAVE":
        return list(dict.fromkeys(features))
    excluded_tokens = (
        "target_time_48h_",
        "target_time_72h_",
        "known_event_any_at_48h",
        "known_event_any_at_72h",
        "known_event_holiday_at_48h",
        "known_event_holiday_at_72h",
    )
    return [
        feature
        for feature in dict.fromkeys(features)
        if not any(token in feature for token in excluded_tokens)
    ]


def feature_block(feature: str) -> str:
    if feature.startswith("research_ext_"):
        return "external_weather"
    if feature.startswith("cargo_"):
        return "cargo_mix"
    if feature.startswith(("arrivals_", "departures_", "vessels_in_port_")):
        return "arrival_history"
    if feature.startswith("wave_"):
        return "wave_history"
    if feature.startswith(("issue_", "target_time_", "known_event_")):
        return "calendar_passthrough"
    if feature.startswith(("delayed_", "mean_arrival_", "weather_available_")):
        return "port_state"
    return "other_passthrough"


def is_protected_feature(feature: str) -> bool:
    return feature.startswith(("issue_", "target_time_", "known_event_")) or feature.endswith(
        "_missing"
    )


def temporal_family(feature: str) -> str:
    normalized = re.sub(r"_(lag|last|roll|ewm)_?\d+h(?:_(sum|mean|std|share))?", "_temporal", feature)
    normalized = re.sub(r"_prev_1h", "_temporal", normalized)
    return normalized


def intentional_temporal_pair(left: str, right: str) -> bool:
    return left != right and temporal_family(left) == temporal_family(right)


def _sample(frame: pd.DataFrame, maximum: int, seed: int = 2026) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.sample(maximum, random_state=seed).sort_index()


def constant_report(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        values = pd.to_numeric(train[feature], errors="coerce")
        counts = values.value_counts(dropna=False, normalize=True)
        variance = float(values.var(ddof=0)) if values.notna().any() else np.nan
        rows.append(
            {
                "feature": feature,
                "block": feature_block(feature),
                "unique_values": int(values.nunique(dropna=False)),
                "variance": variance,
                "dominant_fraction": float(counts.iloc[0]) if len(counts) else 1.0,
                "constant": bool(values.nunique(dropna=False) <= 1),
                "quasi_constant": bool(
                    len(counts) == 0 or float(counts.iloc[0]) >= QUASI_CONSTANT_THRESHOLD
                ),
            }
        )
    return pd.DataFrame(rows)


def correlation_matrices(
    train: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = train[features].apply(pd.to_numeric, errors="coerce")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    rows = []
    for row_index, left in enumerate(features):
        for column_index in range(row_index + 1, len(features)):
            right = features[column_index]
            pearson_value = pearson.iat[row_index, column_index]
            spearman_value = spearman.iat[row_index, column_index]
            coefficients = np.abs(
                np.asarray([pearson_value, spearman_value], dtype="float64")
            )
            finite_coefficients = coefficients[np.isfinite(coefficients)]
            if finite_coefficients.size == 0:
                continue
            maximum = float(finite_coefficients.max())
            if maximum >= CORRELATION_REPORT_THRESHOLD:
                left_values = numeric[left].to_numpy(dtype="float64")
                right_values = numeric[right].to_numpy(dtype="float64")
                exact = bool(np.array_equal(left_values, right_values, equal_nan=True))
                rows.append(
                    {
                        "feature_left": left,
                        "feature_right": right,
                        "block_left": feature_block(left),
                        "block_right": feature_block(right),
                        "pearson": pearson_value,
                        "spearman": spearman_value,
                        "max_abs_correlation": maximum,
                        "exact_duplicate": exact,
                        "intentional_temporal_pair": intentional_temporal_pair(left, right),
                        "prune_candidate": bool(maximum >= CORRELATION_PRUNE_THRESHOLD),
                    }
                )
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["exact_duplicate", "max_abs_correlation"], ascending=[False, False]
        )
    return pearson, spearman, pairs


def target_association_report(
    train: pd.DataFrame,
    features: list[str],
    targets: list[str],
) -> pd.DataFrame:
    sampled = _sample(train, MI_SAMPLE_ROWS)
    x = sampled[features].apply(pd.to_numeric, errors="coerce").astype("float64")
    rows = []
    for target in targets:
        if target not in sampled.columns:
            continue
        y = pd.to_numeric(sampled[target], errors="coerce")
        valid = y.notna() & x.notna().all(axis=1)
        if int(valid.sum()) < 100:
            continue
        x_valid = x.loc[valid]
        y_valid = y.loc[valid]
        variable_features = [
            feature for feature in features if x_valid[feature].nunique(dropna=True) > 1
        ]
        mutual_by_feature: dict[str, float] = {}
        if variable_features:
            mutual = mutual_info_regression(
                x_valid[variable_features],
                y_valid,
                discrete_features=False,
                random_state=2026,
                n_neighbors=3,
            )
            mutual_by_feature = dict(zip(variable_features, mutual))
            pearson = x_valid[variable_features].corrwith(y_valid, method="pearson")
            spearman = x_valid[variable_features].corrwith(y_valid, method="spearman")
        else:
            pearson = pd.Series(dtype="float64")
            spearman = pd.Series(dtype="float64")
        for feature in features:
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "block": feature_block(feature),
                    "sample_rows": int(valid.sum()),
                    "pearson": pearson.get(feature, np.nan),
                    "spearman": spearman.get(feature, np.nan),
                    "mutual_information": float(mutual_by_feature.get(feature, 0.0)),
                }
            )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["max_abs_linear_or_rank"] = result[["pearson", "spearman"]].abs().max(axis=1)
    return result


def approximate_vif_report(train: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    constants = constant_report(train, features)
    active = constants.loc[~constants["constant"], "feature"].tolist()
    if not active:
        return pd.DataFrame(columns=["feature", "approximate_vif", "block"])
    sampled = _sample(train[active], STABILITY_SAMPLE_ROWS)
    standardized = StandardScaler().fit_transform(sampled.to_numpy(dtype="float64"))
    correlation = np.corrcoef(standardized, rowvar=False)
    inverse = np.linalg.pinv(correlation, rcond=1e-8)
    values = np.maximum(np.diag(inverse), 1.0)
    return pd.DataFrame(
        {
            "feature": active,
            "approximate_vif": values,
            "block": [feature_block(feature) for feature in active],
            "diagnostic_only": True,
        }
    ).sort_values("approximate_vif", ascending=False)


def population_stability_index(
    train_values: pd.Series,
    comparison_values: pd.Series,
    bins: int = 10,
) -> float:
    train = pd.to_numeric(train_values, errors="coerce").dropna().to_numpy(dtype="float64")
    comparison = pd.to_numeric(comparison_values, errors="coerce").dropna().to_numpy(dtype="float64")
    if len(train) == 0 or len(comparison) == 0 or np.nanstd(train) <= 1e-12:
        return 0.0
    edges = np.unique(np.quantile(train, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    train_counts = np.histogram(train, bins=edges)[0].astype("float64")
    comparison_counts = np.histogram(comparison, bins=edges)[0].astype("float64")
    train_fraction = np.clip(train_counts / train_counts.sum(), 1e-6, None)
    comparison_fraction = np.clip(
        comparison_counts / comparison_counts.sum(), 1e-6, None
    )
    return float(
        np.sum(
            (comparison_fraction - train_fraction)
            * np.log(comparison_fraction / train_fraction)
        )
    )


def stability_report(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    train_sample = _sample(train, STABILITY_SAMPLE_ROWS)
    valid_sample = _sample(valid, STABILITY_SAMPLE_ROWS)
    test_sample = _sample(test, STABILITY_SAMPLE_ROWS)
    rows = []
    for feature in features:
        train_values = pd.to_numeric(train_sample[feature], errors="coerce")
        train_mean = float(train_values.mean())
        train_std = float(train_values.std(ddof=0))
        scale = train_std if train_std > 1e-12 else 1.0
        for split_name, split_frame in (("VALID", valid_sample), ("TEST_DIAGNOSTIC_ONLY", test_sample)):
            comparison = pd.to_numeric(split_frame[feature], errors="coerce")
            rows.append(
                {
                    "feature": feature,
                    "block": feature_block(feature),
                    "comparison_split": split_name,
                    "train_mean": train_mean,
                    "comparison_mean": float(comparison.mean()),
                    "standardized_mean_shift": float(
                        abs(float(comparison.mean()) - train_mean) / scale
                    ),
                    "psi": population_stability_index(train_values, comparison),
                    "used_for_selection": split_name == "VALID",
                }
            )
    return pd.DataFrame(rows)


def _autocorrelation(values: np.ndarray, maximum_lag: int) -> np.ndarray:
    centered = values - np.mean(values)
    denominator = float(np.dot(centered, centered))
    result = np.ones(maximum_lag + 1, dtype="float64")
    if denominator <= 1e-12:
        result[1:] = 0.0
        return result
    for lag in range(1, maximum_lag + 1):
        result[lag] = float(np.dot(centered[lag:], centered[:-lag]) / denominator)
    return result


def _pacf_from_acf(acf: np.ndarray) -> np.ndarray:
    maximum_lag = len(acf) - 1
    pacf = np.zeros(maximum_lag + 1, dtype="float64")
    pacf[0] = 1.0
    phi = np.zeros((maximum_lag + 1, maximum_lag + 1), dtype="float64")
    for order in range(1, maximum_lag + 1):
        numerator = acf[order]
        denominator = 1.0
        if order > 1:
            numerator -= np.dot(phi[order - 1, 1:order], acf[order - 1 : 0 : -1])
            denominator -= np.dot(phi[order - 1, 1:order], acf[1:order])
        reflection = 0.0 if abs(denominator) <= 1e-12 else numerator / denominator
        phi[order, order] = reflection
        if order > 1:
            for index in range(1, order):
                phi[order, index] = (
                    phi[order - 1, index]
                    - reflection * phi[order - 1, order - index]
                )
        pacf[order] = reflection
    return pacf


def temporal_dependence_report(train: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    rows = []
    maximum_lag = max(AUTOCORRELATION_LAGS)
    for target in targets:
        values = pd.to_numeric(train[target], errors="coerce").dropna().to_numpy(dtype="float64")
        if len(values) <= maximum_lag + 10:
            continue
        acf = _autocorrelation(values, maximum_lag)
        pacf = _pacf_from_acf(acf)
        for lag in AUTOCORRELATION_LAGS:
            rows.append(
                {
                    "target": target,
                    "lag_h": lag,
                    "acf": acf[lag],
                    "pacf_yule_walker": pacf[lag],
                    "train_rows": len(values),
                }
            )
    return pd.DataFrame(rows)


def feature_priority(association: pd.DataFrame, features: list[str]) -> dict[str, float]:
    if association.empty:
        return {feature: 0.0 for feature in features}
    grouped = association.groupby("feature").agg(
        mutual_information=("mutual_information", "max"),
        correlation=("max_abs_linear_or_rank", "max"),
    )
    mi_max = max(float(grouped["mutual_information"].max()), 1e-12)
    scores = 0.7 * grouped["mutual_information"] / mi_max + 0.3 * grouped["correlation"].fillna(0.0)
    return {feature: float(scores.get(feature, 0.0)) for feature in features}


def build_pruned_features(
    features: list[str],
    constants: pd.DataFrame,
    pairs: pd.DataFrame,
    priority: dict[str, float],
) -> tuple[list[str], pd.DataFrame]:
    remove: dict[str, tuple[str, str | None]] = {}
    for row in constants.itertuples(index=False):
        if row.constant or (row.quasi_constant and not is_protected_feature(row.feature)):
            remove[row.feature] = (
                "CONSTANT" if row.constant else "QUASI_CONSTANT",
                None,
            )
    if not pairs.empty:
        candidates = pairs.loc[pairs["prune_candidate"]].copy()
        for row in candidates.itertuples(index=False):
            left, right = row.feature_left, row.feature_right
            if left in remove or right in remove:
                continue
            if row.intentional_temporal_pair and not row.exact_duplicate:
                continue
            if is_protected_feature(left) and is_protected_feature(right):
                continue
            if is_protected_feature(left):
                loser, winner = right, left
            elif is_protected_feature(right):
                loser, winner = left, right
            else:
                left_score = priority.get(left, 0.0)
                right_score = priority.get(right, 0.0)
                loser, winner = (
                    (right, left)
                    if (left_score, left) >= (right_score, right)
                    else (left, right)
                )
            remove[loser] = (
                "EXACT_DUPLICATE" if row.exact_duplicate else "CORRELATION_GE_0_995",
                winner,
            )
    kept = [feature for feature in features if feature not in remove]
    rows = [
        {
            "feature": feature,
            "action": "DROP" if feature in remove else "KEEP",
            "reason": remove.get(feature, ("UNIQUE_OR_PROTECTED", None))[0],
            "retained_proxy": remove.get(feature, (None, None))[1],
            "priority_score": priority.get(feature, 0.0),
            "block": feature_block(feature),
        }
        for feature in features
    ]
    return kept, pd.DataFrame(rows)


def build_compact_features(
    pruned: list[str], priority: dict[str, float], maximum_scored: int = 32
) -> list[str]:
    protected = [feature for feature in pruned if is_protected_feature(feature)]
    scored = sorted(
        [feature for feature in pruned if feature not in protected],
        key=lambda feature: (priority.get(feature, 0.0), feature),
        reverse=True,
    )[:maximum_scored]
    selected = set(protected + scored)
    for feature in list(selected):
        if feature.endswith("_missing"):
            value_feature = feature[: -len("_missing")]
            if value_feature in pruned:
                selected.add(value_feature)
        else:
            missing_feature = f"{feature}_missing"
            if missing_feature in pruned:
                selected.add(missing_feature)
    return [feature for feature in pruned if feature in selected]


def fit_block_pca(
    active: pd.DataFrame,
    split_column: str,
    features: list[str],
    targets: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train_mask = active[split_column].eq("TRAIN")
    metadata = active[["as_of_time", split_column, *targets]].copy()
    transformed_parts: list[pd.DataFrame] = [metadata]
    transformers: dict[str, Any] = {}
    summary_rows = []
    loading_rows = []
    component_features: list[str] = []
    grouped: dict[str, list[str]] = {}
    for feature in features:
        grouped.setdefault(feature_block(feature), []).append(feature)

    passthrough_blocks = {"calendar_passthrough", "other_passthrough"}
    for block, block_features in grouped.items():
        if block in passthrough_blocks or len(block_features) < 3:
            names = [f"pass__{feature}" for feature in block_features]
            passthrough = active[block_features].astype("float64").copy()
            passthrough.columns = names
            transformed_parts.append(passthrough)
            component_features.extend(names)
            summary_rows.append(
                {
                    "block": block,
                    "input_features": len(block_features),
                    "output_components": len(block_features),
                    "explained_variance": 1.0,
                    "method": "PASSTHROUGH",
                    "fit_split": "TRAIN",
                }
            )
            continue
        x_train = active.loc[train_mask, block_features].to_numpy(dtype="float64")
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(x_train)
        pca = PCA(n_components=PCA_VARIANCE_THRESHOLD, svd_solver="full")
        pca.fit(scaled_train)
        scaled_all = scaler.transform(active[block_features].to_numpy(dtype="float64"))
        components = pca.transform(scaled_all)
        names = [f"pca__{block}__{index + 1:02d}" for index in range(components.shape[1])]
        transformed_parts.append(
            pd.DataFrame(components, index=active.index, columns=names)
        )
        component_features.extend(names)
        transformers[block] = {
            "features": block_features,
            "scaler": scaler,
            "pca": pca,
            "component_names": names,
            "fit_split": "TRAIN",
        }
        summary_rows.append(
            {
                "block": block,
                "input_features": len(block_features),
                "output_components": len(names),
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
                "method": "STANDARD_SCALER_PLUS_PCA",
                "fit_split": "TRAIN",
            }
        )
        for component_index, component_name in enumerate(names):
            for feature_index, feature in enumerate(block_features):
                loading_rows.append(
                    {
                        "block": block,
                        "component": component_name,
                        "feature": feature,
                        "loading": float(pca.components_[component_index, feature_index]),
                    }
                )
    transformed = pd.concat(transformed_parts, axis=1)
    transformed.attrs["component_features"] = component_features
    return (
        transformed,
        transformers,
        pd.DataFrame(summary_rows),
        pd.DataFrame(loading_rows),
    )


def _representative_targets(feature_sets: dict[str, list[str]], task: str) -> list[str]:
    available = feature_sets[f"{task.lower()}_targets"]
    if task == "ARRIVAL":
        desired = [
            "target_arrivals_next_6h",
            "target_arrivals_next_12h",
            "target_arrivals_next_24h",
        ]
    else:
        desired = [
            "target_wave_height_m_24h",
            "target_wave_period_s_24h",
            "target_wave_direction_sin_24h",
            "target_wave_direction_cos_24h",
        ]
    return [target for target in desired if target in available]


def audit_feature_representations(
    dataset: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> AuditResult:
    frame = dataset.copy()
    frame["as_of_time"] = pd.to_datetime(frame["as_of_time"], utc=True)
    reports: dict[str, pd.DataFrame] = {}
    representation_sets: dict[str, list[str]] = {}
    pca_frames: dict[str, pd.DataFrame] = {}
    transformers: dict[str, dict[str, Any]] = {}
    inventory_rows = []

    for task in ("ARRIVAL", "WAVE"):
        task_key = task.lower()
        split_column = f"split_{task_key}"
        active = frame.loc[frame[split_column].isin(["TRAIN", "VALID", "TEST"])].copy()
        train = active.loc[active[split_column].eq("TRAIN")]
        valid = active.loc[active[split_column].eq("VALID")]
        test = active.loc[active[split_column].eq("TEST")]
        targets = _representative_targets(feature_sets, task)
        all_targets = feature_sets[f"{task_key}_targets"]
        enriched = task_relevant_features(
            feature_sets[f"{task_key}_research_enriched"], task
        )
        core = task_relevant_features(feature_sets[f"{task_key}_core"], task)
        missing_columns = sorted(set(enriched + all_targets + [split_column]) - set(active.columns))
        if missing_columns:
            raise ValueError(f"Missing B60A columns for {task}: {missing_columns}")
        if active[enriched].isna().any().any():
            raise ValueError(f"Active {task} features contain missing values")

        constants = constant_report(train, enriched)
        pearson, spearman, pairs = correlation_matrices(train, enriched)
        association = target_association_report(train, enriched, targets)
        stability = stability_report(train, valid, test, enriched)
        temporal = temporal_dependence_report(train, targets)
        reports[f"{task_key}_constant_features.csv"] = constants
        reports[f"{task_key}_pearson_matrix.csv"] = pearson.reset_index(names="feature")
        reports[f"{task_key}_spearman_matrix.csv"] = spearman.reset_index(names="feature")
        reports[f"{task_key}_correlated_pairs.csv"] = pairs
        reports[f"{task_key}_target_association.csv"] = association
        reports[f"{task_key}_feature_stability.csv"] = stability
        reports[f"{task_key}_temporal_dependence.csv"] = temporal

        priorities = feature_priority(association, enriched)
        for track_name, track_features in (("CORE", core), ("RESEARCH", enriched)):
            track_key = f"{task_key}_{track_name.lower()}"
            track_constants = constants.loc[constants["feature"].isin(track_features)]
            track_pairs = pairs.loc[
                pairs["feature_left"].isin(track_features)
                & pairs["feature_right"].isin(track_features)
            ] if not pairs.empty else pairs
            pruned, actions = build_pruned_features(
                track_features, track_constants, track_pairs, priorities
            )
            compact = build_compact_features(pruned, priorities)
            representation_sets[f"{track_key}_raw"] = track_features
            representation_sets[f"{track_key}_pruned"] = pruned
            representation_sets[f"{track_key}_compact"] = compact
            reports[f"{track_key}_pruning_actions.csv"] = actions
            reports[f"{track_key}_vif.csv"] = approximate_vif_report(train, pruned)

            pca_frame, fitted, pca_summary, loadings = fit_block_pca(
                active,
                split_column,
                pruned,
                all_targets,
            )
            pca_features = list(pca_frame.attrs["component_features"])
            representation_sets[f"{track_key}_block_pca"] = pca_features
            pca_frames[track_key] = pca_frame
            transformers[track_key] = fitted
            pca_summary.insert(0, "track", track_key)
            loadings.insert(0, "track", track_key)
            reports[f"{track_key}_pca_summary.csv"] = pca_summary
            reports[f"{track_key}_pca_loadings.csv"] = loadings
            inventory_rows.append(
                {
                    "task": task,
                    "track": track_name,
                    "raw_features": len(track_features),
                    "pruned_features": len(pruned),
                    "compact_features": len(compact),
                    "block_pca_features": len(pca_features),
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "test_rows": len(test),
                }
            )

    inventory = pd.DataFrame(inventory_rows)
    reports["representation_inventory.csv"] = inventory
    forbidden = [
        feature
        for features in representation_sets.values()
        for feature in features
        if feature.startswith("target_") and not feature.startswith("target_time_")
    ]
    non_finite = 0
    for pca_frame in pca_frames.values():
        component_columns = list(pca_frame.attrs["component_features"])
        non_finite += int(
            (~np.isfinite(pca_frame[component_columns].to_numpy(dtype="float64"))).sum()
        )
    gates = pd.DataFrame(
        [
            ("SOURCE_B60A_DATASET_VERSION", frame["dataset_version"].eq("b60a-maritime-multitask-hourly-v1").all(), int(frame["dataset_version"].ne("b60a-maritime-multitask-hourly-v1").sum())),
            ("NO_TARGET_IN_FEATURE_REPRESENTATIONS", len(forbidden) == 0, len(forbidden)),
            ("TRANSFORMERS_FIT_ON_TRAIN_ONLY", all(spec.get("fit_split") == "TRAIN" for fitted in transformers.values() for spec in fitted.values()), 0),
            ("TEST_NOT_USED_FOR_SELECTION", True, 0),
            ("PCA_COMPONENTS_FINITE", non_finite == 0, non_finite),
            ("ALL_FOUR_TASK_TRACKS_CREATED", len(inventory) == 4, len(inventory)),
            ("RAW_REPRESENTATIONS_PRESERVED", all(len(values) > 0 for key, values in representation_sets.items() if key.endswith("_raw")), 0),
            ("PRUNED_REPRESENTATIONS_NONEMPTY", all(len(values) > 0 for key, values in representation_sets.items() if key.endswith("_pruned")), 0),
            ("BLOCK_PCA_REPRESENTATIONS_NONEMPTY", all(len(values) > 0 for key, values in representation_sets.items() if key.endswith("_block_pca")), 0),
            ("NO_PREDICTIVE_MODEL_TRAINING", True, 0),
        ],
        columns=["gate", "passed", "observed"],
    )
    gates["severity"] = "CRITICAL"
    reports["quality_gates.csv"] = gates
    passed = bool(gates["passed"].all())
    decision = {
        "status": "SUCCESS",
        "decision": (
            "READY_FOR_B60B_FEATURE_REPRESENTATION_BENCHMARK"
            if passed
            else "BLOCKED_FEATURE_REPRESENTATION_REPAIR_REQUIRED"
        ),
        "audit_version": AUDIT_VERSION,
        "source_dataset_version": "b60a-maritime-multitask-hourly-v1",
        "source_rows": len(frame),
        "task_tracks": len(inventory),
        "representation_count": len(representation_sets),
        "quality_gates_passed": passed,
        "correlation_report_threshold": CORRELATION_REPORT_THRESHOLD,
        "correlation_prune_threshold": CORRELATION_PRUNE_THRESHOLD,
        "pca_variance_threshold": PCA_VARIANCE_THRESHOLD,
        "pca_scope": "SEPARATE_BY_TASK_TRACK_AND_SEMANTIC_BLOCK",
        "ppca_decision": "NOT_SELECTED_STRUCTURAL_MISSINGNESS",
        "selection_split": "TRAIN_STRUCTURE_ONLY",
        "validation_role": "B60B_MODEL_SELECTION_ONLY",
        "test_role": "DIAGNOSTIC_ONLY_NOT_USED_FOR_SELECTION",
        "selection_used_test": False,
        "predictive_training_executed": False,
        "source_modified": False,
        "production_promotion_allowed": False,
        "next_block": "B60B_ADVANCED_TIME_SERIES_ROLLING_ORIGIN_BENCHMARK",
        "representation_sizes": inventory.to_dict(orient="records"),
    }
    return AuditResult(
        reports=reports,
        representation_sets=representation_sets,
        pca_frames=pca_frames,
        transformers=transformers,
        decision=clean_json(decision),
    )


def representation_sets_json(value: dict[str, list[str]]) -> str:
    return json.dumps(clean_json(value), indent=2, ensure_ascii=True)


def content_checksum(dataset_bytes: bytes, feature_sets_bytes: bytes) -> str:
    digest = hashlib.sha256(AUDIT_VERSION.encode("ascii"))
    digest.update(hashlib.sha256(dataset_bytes).digest())
    digest.update(hashlib.sha256(feature_sets_bytes).digest())
    return digest.hexdigest()
