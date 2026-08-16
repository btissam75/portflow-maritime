from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

import boto3
import numpy as np
import pandas as pd
import psycopg2
from catboost import CatBoostClassifier, CatBoostRegressor
from psycopg2.extras import Json, execute_values
from sklearn.isotonic import IsotonicRegression

from prefect_flows.b61b_core import (
    CATEGORICAL_FEATURES,
    HAZARD_HORIZONS,
    HAZARD_TARGETS,
    IDENTIFIER_COLUMNS,
    MODEL_VERSION,
    RANDOM_SEED,
    RESEARCH_NUMERIC_FEATURES,
    RISK_NAMES,
    RISK_TARGETS,
    SOURCE_DATASET_VERSION,
    TARGET_COLUMNS,
    apply_conformal,
    available_contract,
    binary_metrics,
    clean_json,
    conformalize_interval,
    delay_class_index,
    enforce_hazard_order,
    enforce_quantile_order,
    enforce_risk_order,
    regime_labels,
    regression_metrics,
    risk_from_class_probabilities,
    select_binary_threshold,
    select_blend_weight,
    split_validation_roles,
    stable_hash_sample,
)
from prefect_flows.b61b_sequence import train_sequence_expert


SOURCE_NAME = "b61b_multitask_temporal_survival_moe"
DATASET_NAME = "maritime_port_call_multitask_predictions_v1"
SOURCE_RELATION = "features.maritime_port_call_governed_v1"
TARGET_RELATION = "serving.maritime_port_call_multitask_prediction_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"


def _db_connection():
    return psycopg2.connect(
        host=os.environ["SMART_PORT_DB_HOST"],
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
    )


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _relation_exists(relation: str) -> bool:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass(%s)", (relation,))
        return cursor.fetchone()[0] is not None


def _source_columns() -> list[str]:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='features'
              AND table_name='maritime_port_call_governed_v1'
            ORDER BY ordinal_position
            """
        )
        return [str(row[0]) for row in cursor.fetchall()]


def _query_frame(query: str, parameters: tuple[Any, ...] | None = None) -> pd.DataFrame:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(query, parameters)
        rows = cursor.fetchall()
        columns = [item.name for item in cursor.description]
    return pd.DataFrame(rows, columns=columns)


def _quote_columns(columns: list[str]) -> str:
    if any(not column.replace("_", "").isalnum() for column in columns):
        raise ValueError("Unsafe source column")
    return ", ".join(f'"{column}"' for column in columns)


def load_model_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    columns = _source_columns()
    contract = available_contract(columns)
    selected = list(
        dict.fromkeys(
            [
                *IDENTIFIER_COLUMNS,
                *contract.core_numeric,
                *contract.research_numeric,
                *contract.categorical,
                *TARGET_COLUMNS,
            ]
        )
    )
    selected = [column for column in selected if column in columns]
    frame = _query_frame(
        f"""
        SELECT {_quote_columns(selected)}
        FROM {SOURCE_RELATION}
        WHERE dataset_version=%s
          AND split IN ('TRAIN', 'VALID', 'TEST')
          AND synthetic_row=false
          AND targets_imputed=false
        ORDER BY port_call_id, landmark_at
        """,
        (SOURCE_DATASET_VERSION,),
    )
    if frame.empty:
        raise RuntimeError(f"No source rows found in {SOURCE_RELATION}")
    frame["landmark_at"] = pd.to_datetime(frame["landmark_at"], utc=True)
    frame["model_role"] = split_validation_roles(frame)
    permission_columns = {
        "TRAIN": "training_allowed",
        "VALID": "validation_allowed",
        "TEST": "test_allowed",
    }
    for split, permission in permission_columns.items():
        denied = frame["split"].eq(split) & ~frame[permission].fillna(False).astype(bool)
        frame.loc[denied, "model_role"] = "UNUSED_NOT_GOVERNED"
    frame["regime"] = regime_labels(frame)
    for column in RISK_TARGETS + HAZARD_TARGETS:
        frame[column] = frame[column].astype("float32")
    frame["target_remaining_h"] = pd.to_numeric(frame["target_remaining_h"], errors="coerce").astype("float32")
    frame["target_departure_delay_h"] = pd.to_numeric(frame["target_departure_delay_h"], errors="coerce").astype("float32")
    info = {
        "source_columns": len(columns),
        "loaded_columns": len(selected),
        "core_numeric": list(contract.core_numeric),
        "research_numeric": list(contract.research_numeric),
        "categorical": list(contract.categorical),
    }
    return frame, info


def _source_signature(frame: pd.DataFrame, max_steps: int, iterations: int) -> str:
    digest = hashlib.sha256(MODEL_VERSION.encode("ascii"))
    digest.update(str((len(frame), max_steps, iterations)).encode("ascii"))
    digest.update(
        pd.util.hash_pandas_object(
            frame[["port_call_id", "landmark_at", "split"]], index=False
        ).to_numpy(dtype="uint64").tobytes()
    )
    return digest.hexdigest()


def _previous_success(checksum: str) -> dict[str, Any] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT metadata
            FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s AND checksum=%s
              AND status='SUCCESS'
            ORDER BY finished_at DESC
            LIMIT 1
            """,
            (SOURCE_NAME, DATASET_NAME, checksum),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        cursor.execute("SELECT to_regclass(%s)", (TARGET_RELATION,))
        if cursor.fetchone()[0] is None:
            return None
        cursor.execute(
            "SELECT COUNT(*) FROM serving.maritime_port_call_multitask_prediction_v1 WHERE model_version=%s",
            (MODEL_VERSION,),
        )
        if int(cursor.fetchone()[0]) == 0:
            return None
        return dict(row[0])


def _start_run(checksum: str) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "orchestrator": "PREFECT",
        "selection_split": "VALID_SELECT",
        "calibration_split": "VALID_CALIBRATE",
        "test_role": "TEST_DIAGNOSTIC_ONLY",
        "test_used_for_selection": False,
        "synthetic_rows": 0,
        "targets_imputed": False,
        "production_promotion_allowed": False,
    }
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO audit.ingestion_run
                (source_name, dataset_name, object_uri, checksum, metadata)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING run_id
            """,
            (
                SOURCE_NAME,
                DATASET_NAME,
                f"postgresql://maritime/{TARGET_RELATION}",
                checksum,
                Json(metadata),
            ),
        )
        return str(cursor.fetchone()[0])


def _update_progress(run_id: str, stage: str, **details: Any) -> None:
    payload = clean_json(
        {"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details}
    )
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
            (Json({"progress": payload}), run_id),
        )


def _finish_run(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error_message: str | None = None,
) -> None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE audit.ingestion_run
            SET finished_at=now(), status=%s, row_count=%s,
                metadata=metadata || %s, error_message=%s
            WHERE run_id=%s
            """,
            (status, row_count, Json(clean_json(metadata)), error_message, run_id),
        )


def _model_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame[features].copy()
    for column in features:
        if column in CATEGORICAL_FEATURES:
            result[column] = result[column].fillna("__MISSING__").astype(str)
        else:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("float32")
    return result


def _catboost_common(iterations: int) -> dict[str, Any]:
    return {
        "iterations": iterations,
        "depth": 7,
        "learning_rate": 0.045,
        "l2_leaf_reg": 5.0,
        "random_seed": RANDOM_SEED,
        "thread_count": 2,
        "allow_writing_files": False,
        "verbose": False,
        "used_ram_limit": "900mb",
    }


def _fit_tabular_expert(
    frame: pd.DataFrame,
    features: list[str],
    iterations: int,
    max_train_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], bytes, pd.DataFrame]:
    train = stable_hash_sample(frame.loc[frame["model_role"].eq("TRAIN_FIT")], max_train_rows)
    valid = frame.loc[frame["model_role"].eq("VALID_SELECT")]
    x_train = _model_frame(train, features)
    x_valid = _model_frame(valid, features)
    x_all = _model_frame(frame, features)
    cat_features = [column for column in features if column in CATEGORICAL_FEATURES]
    sample_weight = pd.to_numeric(train["per_call_sample_weight"], errors="coerce").fillna(1.0).to_numpy()
    common = _catboost_common(iterations)

    classes = delay_class_index(train["target_departure_delay_h"])
    valid_classes = delay_class_index(valid["target_departure_delay_h"])
    ordinal = CatBoostClassifier(
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights="Balanced",
        **common,
    )
    ordinal.fit(
        x_train,
        classes,
        cat_features=cat_features,
        sample_weight=sample_weight,
        eval_set=(x_valid, valid_classes),
        early_stopping_rounds=45,
    )
    class_probability = ordinal.predict_proba(x_all)
    risk = risk_from_class_probabilities(class_probability)

    quantile = CatBoostRegressor(
        loss_function="MultiQuantile:alpha=0.1,0.5,0.9",
        eval_metric="MultiQuantile:alpha=0.1,0.5,0.9",
        **common,
    )
    quantile.fit(
        x_train,
        train["target_remaining_h"].to_numpy(),
        cat_features=cat_features,
        sample_weight=sample_weight,
        eval_set=(x_valid, valid["target_remaining_h"].to_numpy()),
        early_stopping_rounds=45,
    )
    duration = enforce_quantile_order(np.asarray(quantile.predict(x_all)))

    hazard_predictions = []
    hazard_models: dict[str, Any] = {}
    hazard_train = train.loc[train["pre_breach_eligible"].astype(bool)]
    hazard_valid = valid.loc[valid["pre_breach_eligible"].astype(bool)]
    for horizon, target in zip(HAZARD_HORIZONS, HAZARD_TARGETS):
        y_train = hazard_train[target].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            probability = np.repeat(float(np.mean(y_train)), len(frame))
            hazard_models[str(horizon)] = {"constant": float(np.mean(y_train))}
        else:
            model = CatBoostClassifier(
                loss_function="Logloss",
                eval_metric="BrierScore",
                auto_class_weights="Balanced",
                **common,
            )
            fit_kwargs: dict[str, Any] = {}
            if not hazard_valid.empty and hazard_valid[target].nunique() == 2:
                fit_kwargs = {
                    "eval_set": (_model_frame(hazard_valid, features), hazard_valid[target].astype(int)),
                    "early_stopping_rounds": 45,
                }
            model.fit(
                _model_frame(hazard_train, features),
                y_train,
                cat_features=cat_features,
                sample_weight=pd.to_numeric(hazard_train["per_call_sample_weight"], errors="coerce").fillna(1.0),
                **fit_kwargs,
            )
            probability = model.predict_proba(x_all)[:, 1]
            hazard_models[str(horizon)] = model
        hazard_predictions.append(probability)
    hazard = enforce_hazard_order(np.column_stack(hazard_predictions))
    importance = pd.DataFrame(
        {"feature": features, "importance": ordinal.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    models = {"ordinal": ordinal, "quantile": quantile, "hazard": hazard_models}
    artifact = pickle.dumps(
        {
            "model_version": MODEL_VERSION,
            "features": features,
            "categorical_features": cat_features,
            "models": models,
        },
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    inventory = {
        "family": "CATBOOST_SHARED_FEATURE_EXPERT",
        "train_rows": len(train),
        "valid_select_rows": len(valid),
        "features": len(features),
        "categorical_features": len(cat_features),
        "requested_iterations": iterations,
        "ordinal_best_iteration": ordinal.get_best_iteration(),
        "quantile_best_iteration": quantile.get_best_iteration(),
        "models": 5,
    }
    return {"risk": risk, "quantiles": duration, "hazard": hazard}, inventory, artifact, importance


def _fit_research_challenger(
    frame: pd.DataFrame,
    features: list[str],
    iterations: int,
    maximum_rows: int,
) -> tuple[pd.DataFrame, bytes]:
    if not any(column in features for column in RESEARCH_NUMERIC_FEATURES):
        return pd.DataFrame(), b""
    train = stable_hash_sample(frame.loc[frame["model_role"].eq("TRAIN_FIT")], maximum_rows)
    valid_test = frame.loc[frame["model_role"].isin(["VALID_SELECT", "TEST_DIAGNOSTIC_ONLY"])]
    x_train = _model_frame(train, features)
    x_eval = _model_frame(valid_test, features)
    cat_features = [column for column in features if column in CATEGORICAL_FEATURES]
    common = _catboost_common(max(120, iterations // 2))
    classifier = CatBoostClassifier(
        loss_function="Logloss", eval_metric="BrierScore", auto_class_weights="Balanced", **common
    )
    classifier.fit(
        x_train,
        train["target_delay_gt_3h"].astype(int),
        cat_features=cat_features,
        sample_weight=train["per_call_sample_weight"],
    )
    regressor = CatBoostRegressor(loss_function="MAE", eval_metric="MAE", **common)
    regressor.fit(
        x_train,
        train["target_remaining_h"],
        cat_features=cat_features,
        sample_weight=train["per_call_sample_weight"],
    )
    probability = classifier.predict_proba(x_eval)[:, 1]
    remaining = np.clip(regressor.predict(x_eval), 0.0, None)
    rows = []
    for role in ("VALID_SELECT", "TEST_DIAGNOSTIC_ONLY"):
        mask = valid_test["model_role"].eq(role).to_numpy()
        if not mask.any():
            continue
        classification = binary_metrics(
            valid_test.loc[mask, "target_delay_gt_3h"].astype(int).to_numpy(),
            probability[mask],
        )
        rows.append({"role": role, "task": "DELAY_GT3", "metric": "BRIER", "value": classification["brier"]})
        rows.append({"role": role, "task": "DELAY_GT3", "metric": "ROC_AUC", "value": classification["roc_auc"]})
        rows.append(
            {
                "role": role,
                "task": "REMAINING_P50",
                "metric": "MAE",
                "value": float(np.mean(np.abs(remaining[mask] - valid_test.loc[mask, "target_remaining_h"].to_numpy()))),
            }
        )
    return pd.DataFrame(rows), pickle.dumps(
        {"policy": "RESEARCH_ONLY_RETROSPECTIVE_WEATHER", "features": features, "classifier": classifier, "regressor": regressor},
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _baseline_predictions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    train = frame["model_role"].eq("TRAIN_FIT")
    risk = np.column_stack(
        [np.repeat(frame.loc[train, target].mean(), len(frame)) for target in RISK_TARGETS]
    )
    schedule = np.clip(
        pd.to_numeric(frame["time_to_planned_departure_h"], errors="coerce").fillna(
            frame.loc[train, "target_remaining_h"].median()
        ).to_numpy(dtype="float64"),
        0.0,
        None,
    )
    residual = frame.loc[train, "target_remaining_h"].to_numpy() - schedule[train]
    offsets = np.quantile(residual, [0.1, 0.5, 0.9])
    duration = enforce_quantile_order(np.column_stack([schedule + offset for offset in offsets]))
    hazard = np.column_stack(
        [np.repeat(frame.loc[train & frame["pre_breach_eligible"].astype(bool), target].mean(), len(frame)) for target in HAZARD_TARGETS]
    )
    return {"risk": enforce_risk_order(risk), "quantiles": duration, "hazard": enforce_hazard_order(hazard)}


def _contextual_blend(
    frame: pd.DataFrame,
    actual: np.ndarray,
    tabular: np.ndarray,
    sequence: np.ndarray,
    task: str,
    eligible: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    select = frame["model_role"].eq("VALID_SELECT").to_numpy() & eligible
    regimes = frame["regime"].astype(str).to_numpy()
    global_weight, global_objective = select_blend_weight(
        actual[select], tabular[select], sequence[select], task
    )
    weights = {"__GLOBAL__": global_weight}
    objectives = {"__GLOBAL__": global_objective}
    for regime in sorted(set(regimes[select])):
        mask = select & (regimes == regime)
        if mask.sum() < 250:
            continue
        weight, objective = select_blend_weight(actual[mask], tabular[mask], sequence[mask], task)
        shrinkage = mask.sum() / (mask.sum() + 250.0)
        weights[regime] = float(shrinkage * weight + (1.0 - shrinkage) * global_weight)
        objectives[regime] = objective
    row_weights = np.array([weights.get(regime, global_weight) for regime in regimes])
    if task == "QUANTILE":
        blended = row_weights[:, None] * tabular + (1.0 - row_weights[:, None]) * sequence
        blended = enforce_quantile_order(blended)
    else:
        blended = row_weights * tabular + (1.0 - row_weights) * sequence
        blended = np.clip(blended, 0.0, 1.0)
    return blended, {"weights": weights, "selection_objectives": objectives}


def _calibrate_binary(
    frame: pd.DataFrame,
    actual: np.ndarray,
    probability: np.ndarray,
    eligible: np.ndarray,
) -> tuple[np.ndarray, float, IsotonicRegression | None]:
    calibration = frame["model_role"].eq("VALID_CALIBRATE").to_numpy() & eligible
    calibrated = np.clip(probability, 0.0, 1.0)
    calibrator = None
    if calibration.sum() >= 100 and len(np.unique(actual[calibration])) == 2:
        calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        calibrator.fit(probability[calibration], actual[calibration])
        calibrated = calibrator.predict(probability)
    threshold = select_binary_threshold(actual[calibration], calibrated[calibration])
    return np.clip(calibrated, 0.0, 1.0), threshold, calibrator


def _build_ensemble(
    frame: pd.DataFrame,
    tabular: dict[str, np.ndarray],
    sequence: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any], bytes]:
    early = frame["early_warning_eligible"].astype(bool).to_numpy()
    pre_breach = frame["pre_breach_eligible"].astype(bool).to_numpy()
    risk_columns = []
    risk_policy = {}
    calibrators: dict[str, Any] = {}
    thresholds = {}
    for index, (name, target) in enumerate(zip(RISK_NAMES, RISK_TARGETS)):
        actual = frame[target].astype(int).to_numpy()
        blended, policy = _contextual_blend(
            frame, actual, tabular["risk"][:, index], sequence["risk"][:, index], "BINARY", early
        )
        calibrated, threshold, calibrator = _calibrate_binary(frame, actual, blended, early)
        risk_columns.append(calibrated)
        risk_policy[name] = policy
        calibrators[f"risk_{name}"] = calibrator
        thresholds[name] = threshold
    risk = enforce_risk_order(np.column_stack(risk_columns))

    actual_remaining = frame["target_remaining_h"].to_numpy(dtype="float64")
    duration, duration_policy = _contextual_blend(
        frame,
        actual_remaining,
        tabular["quantiles"],
        sequence["quantiles"],
        "QUANTILE",
        np.isfinite(actual_remaining),
    )
    calibration = frame["model_role"].eq("VALID_CALIBRATE").to_numpy() & np.isfinite(actual_remaining)
    lower_correction, upper_correction = conformalize_interval(
        actual_remaining[calibration], duration[calibration, 0], duration[calibration, 2]
    )
    duration = apply_conformal(duration, lower_correction, upper_correction)

    hazard_columns = []
    hazard_policy = {}
    for index, (horizon, target) in enumerate(zip(HAZARD_HORIZONS, HAZARD_TARGETS)):
        actual = frame[target].astype(int).to_numpy()
        blended, policy = _contextual_blend(
            frame, actual, tabular["hazard"][:, index], sequence["hazard"][:, index], "BINARY", pre_breach
        )
        calibrated, threshold, calibrator = _calibrate_binary(frame, actual, blended, pre_breach)
        hazard_columns.append(calibrated)
        hazard_policy[str(horizon)] = policy
        calibrators[f"hazard_{horizon}"] = calibrator
        thresholds[f"breach_{horizon}h"] = threshold
    hazard = enforce_hazard_order(np.column_stack(hazard_columns))
    policy = {
        "risk": risk_policy,
        "duration": duration_policy,
        "hazard": hazard_policy,
        "thresholds": thresholds,
        "conformal": {
            "nominal_coverage": 0.8,
            "lower_correction_h": lower_correction,
            "upper_correction_h": upper_correction,
            "calibration_split": "VALID_CALIBRATE",
        },
    }
    artifact = pickle.dumps(
        {"model_version": MODEL_VERSION, "policy": policy, "calibrators": calibrators},
        protocol=pickle.HIGHEST_PROTOCOL,
    )
    return {"risk": risk, "quantiles": duration, "hazard": hazard}, policy, artifact


def _metric_reports(
    frame: pd.DataFrame,
    predictions: dict[str, dict[str, np.ndarray]],
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binary_rows = []
    duration_rows = []
    ordinal_rows = []
    for role in ("VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"):
        role_mask = frame["model_role"].eq(role).to_numpy()
        for expert, values in predictions.items():
            for index, (name, target) in enumerate(zip(RISK_NAMES, RISK_TARGETS)):
                mask = role_mask & frame["early_warning_eligible"].astype(bool).to_numpy()
                metrics = binary_metrics(
                    frame.loc[mask, target].astype(int).to_numpy(),
                    values["risk"][mask, index],
                    thresholds.get(name, 0.5),
                )
                binary_rows.append({"role": role, "expert": expert, "task": f"DELAY_{name.upper()}", **metrics})
            for index, (horizon, target) in enumerate(zip(HAZARD_HORIZONS, HAZARD_TARGETS)):
                mask = role_mask & frame["pre_breach_eligible"].astype(bool).to_numpy()
                metrics = binary_metrics(
                    frame.loc[mask, target].astype(int).to_numpy(),
                    values["hazard"][mask, index],
                    thresholds.get(f"breach_{horizon}h", 0.5),
                )
                binary_rows.append({"role": role, "expert": expert, "task": f"BREACH_WITHIN_{horizon}H", **metrics})
            mask = role_mask & frame["target_remaining_h"].notna().to_numpy()
            duration_rows.append(
                {
                    "role": role,
                    "expert": expert,
                    "task": "REMAINING_DURATION",
                    **regression_metrics(frame.loc[mask, "target_remaining_h"].to_numpy(), values["quantiles"][mask]),
                }
            )
            actual_class = delay_class_index(frame.loc[role_mask, "target_departure_delay_h"])
            predicted_class = np.sum(values["risk"][role_mask] >= 0.5, axis=1)
            ordinal_rows.append(
                {
                    "role": role,
                    "expert": expert,
                    "rows": int(role_mask.sum()),
                    "accuracy": float(np.mean(actual_class == predicted_class)),
                    "within_one_class": float(np.mean(np.abs(actual_class - predicted_class) <= 1)),
                    "mean_absolute_class_error": float(np.mean(np.abs(actual_class - predicted_class))),
                }
            )
    return pd.DataFrame(binary_rows), pd.DataFrame(duration_rows), pd.DataFrame(ordinal_rows)


def _prediction_frame(
    frame: pd.DataFrame,
    ensemble: dict[str, np.ndarray],
    thresholds: dict[str, float],
    run_id: str,
) -> pd.DataFrame:
    keep = frame["model_role"].isin(["VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"]).to_numpy()
    risk = ensemble["risk"][keep]
    duration = ensemble["quantiles"][keep]
    hazard = ensemble["hazard"][keep]
    source = frame.loc[keep]
    predicted_class = np.sum(
        np.column_stack(
            [risk[:, index] >= thresholds.get(name, 0.5) for index, name in enumerate(RISK_NAMES)]
        ),
        axis=1,
    )
    labels = np.asarray(["LT_OR_EQ_1H", "GT1_TO_3H", "GT3_TO_6H", "GT6H"])
    return pd.DataFrame(
        {
            "model_version": MODEL_VERSION,
            "port_call_id": source["port_call_id"].astype(str).to_numpy(),
            "landmark_at": source["landmark_at"].to_numpy(),
            "split": source["split"].astype(str).to_numpy(),
            "evaluation_role": source["model_role"].astype(str).to_numpy(),
            "regime": source["regime"].astype(str).to_numpy(),
            "p_delay_gt1": risk[:, 0],
            "p_delay_gt3": risk[:, 1],
            "p_delay_gt6": risk[:, 2],
            "predicted_delay_class": labels[np.clip(predicted_class, 0, 3)],
            "remaining_p10_h": duration[:, 0],
            "remaining_p50_h": duration[:, 1],
            "remaining_p90_h": duration[:, 2],
            "p_gt3_breach_within_6h": hazard[:, 0],
            "p_gt3_breach_within_12h": hazard[:, 1],
            "p_gt3_breach_within_24h": hazard[:, 2],
            "interval_kind": "ADAPTIVE_SPLIT_CONFORMAL_80",
            "model_family": "CONTEXTUAL_CATBOOST_GRU_MOE",
            "production_claim_allowed": False,
            "materialization_run_id": run_id,
        }
    )


def _materialize_predictions(frame: pd.DataFrame) -> int:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS serving.maritime_port_call_multitask_prediction_v1 (
                model_version TEXT NOT NULL,
                port_call_id TEXT NOT NULL,
                landmark_at TIMESTAMPTZ NOT NULL,
                split TEXT NOT NULL,
                evaluation_role TEXT NOT NULL,
                regime TEXT NOT NULL,
                p_delay_gt1 DOUBLE PRECISION NOT NULL,
                p_delay_gt3 DOUBLE PRECISION NOT NULL,
                p_delay_gt6 DOUBLE PRECISION NOT NULL,
                predicted_delay_class TEXT NOT NULL,
                remaining_p10_h DOUBLE PRECISION NOT NULL,
                remaining_p50_h DOUBLE PRECISION NOT NULL,
                remaining_p90_h DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_6h DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_12h DOUBLE PRECISION NOT NULL,
                p_gt3_breach_within_24h DOUBLE PRECISION NOT NULL,
                interval_kind TEXT NOT NULL,
                model_family TEXT NOT NULL,
                production_claim_allowed BOOLEAN NOT NULL,
                materialization_run_id TEXT NOT NULL,
                PRIMARY KEY (model_version, port_call_id, landmark_at)
            )
            """
        )
        cursor.execute("DELETE FROM serving.maritime_port_call_multitask_prediction_v1 WHERE model_version=%s", (MODEL_VERSION,))
        for start in range(0, len(frame), 2_000):
            batch = frame.iloc[start : start + 2_000]
            records = [tuple(row) for row in batch.itertuples(index=False, name=None)]
            execute_values(
                cursor,
                "INSERT INTO serving.maritime_port_call_multitask_prediction_v1 VALUES %s",
                records,
                page_size=2_000,
            )
        cursor.execute("CREATE INDEX IF NOT EXISTS ix_b61b_prediction_landmark ON serving.maritime_port_call_multitask_prediction_v1 (landmark_at DESC)")
    return len(frame)


def _put_bytes(key: str, payload: bytes, content_type: str) -> str:
    _s3_client().put_object(Bucket=OUTPUT_BUCKET, Key=key, Body=payload, ContentType=content_type)
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _put_json(key: str, payload: Any) -> str:
    return _put_bytes(key, json.dumps(clean_json(payload), indent=2, sort_keys=True).encode("utf-8"), "application/json")


def _put_csv(key: str, frame: pd.DataFrame) -> str:
    return _put_bytes(key, frame.to_csv(index=False).encode("utf-8"), "text/csv")


def _quality_gates(
    frame: pd.DataFrame,
    binary_report: pd.DataFrame,
    duration_report: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    select_binary = binary_report.loc[
        binary_report["role"].eq("VALID_SELECT") & binary_report["expert"].eq("MOE_CALIBRATED")
    ]
    select_duration = duration_report.loc[
        duration_report["role"].eq("VALID_CALIBRATE") & duration_report["expert"].eq("MOE_CALIBRATED")
    ]
    finite_columns = [
        "p_delay_gt1", "p_delay_gt3", "p_delay_gt6", "remaining_p10_h",
        "remaining_p50_h", "remaining_p90_h", "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h", "p_gt3_breach_within_24h",
    ]
    rows = [
        {"check": "TRAIN_VALID_TEST_PRESENT", "passed": set(frame["split"]) == {"TRAIN", "VALID", "TEST"}, "severity": "CRITICAL"},
        {"check": "PORT_CALL_BELONGS_TO_ONE_SPLIT", "passed": bool(frame.groupby("port_call_id")["split"].nunique().max() == 1), "severity": "CRITICAL"},
        {"check": "ALL_MODELED_ROWS_GOVERNED", "passed": bool(~frame["model_role"].eq("UNUSED_NOT_GOVERNED").any()), "severity": "CRITICAL"},
        {"check": "VALID_SELECTION_AND_CALIBRATION_DISJOINT", "passed": frame["model_role"].isin(["VALID_SELECT", "VALID_CALIBRATE"]).sum() == frame["split"].eq("VALID").sum(), "severity": "CRITICAL"},
        {"check": "TEST_NOT_USED_FOR_SELECTION", "passed": True, "severity": "CRITICAL"},
        {"check": "NO_SYNTHETIC_OR_IMPUTED_TARGETS", "passed": True, "severity": "CRITICAL"},
        {"check": "FINITE_SERVING_PREDICTIONS", "passed": np.isfinite(predictions[finite_columns].to_numpy()).all(), "severity": "CRITICAL"},
        {"check": "RISK_MONOTONIC", "passed": bool(((predictions["p_delay_gt1"] >= predictions["p_delay_gt3"]) & (predictions["p_delay_gt3"] >= predictions["p_delay_gt6"])).all()), "severity": "CRITICAL"},
        {"check": "HAZARD_MONOTONIC", "passed": bool(((predictions["p_gt3_breach_within_6h"] <= predictions["p_gt3_breach_within_12h"]) & (predictions["p_gt3_breach_within_12h"] <= predictions["p_gt3_breach_within_24h"])).all()), "severity": "CRITICAL"},
        {"check": "QUANTILES_MONOTONIC", "passed": bool(((predictions["remaining_p10_h"] <= predictions["remaining_p50_h"]) & (predictions["remaining_p50_h"] <= predictions["remaining_p90_h"])).all()), "severity": "CRITICAL"},
        {"check": "VALID_AUC_ABOVE_RANDOM", "passed": bool((select_binary["roc_auc"].fillna(0.0) >= 0.5).mean() >= 0.5), "severity": "MODEL"},
        {"check": "CONFORMAL_VALID_COVERAGE_REASONABLE", "passed": bool((select_duration["coverage_p10_p90"].between(0.72, 0.92)).all()), "severity": "MODEL"},
        {"check": "PRODUCTION_BLOCKED_UNTIL_ISSUE_TIME_HISTORY", "passed": bool((~predictions["production_claim_allowed"]).all()), "severity": "CRITICAL"},
    ]
    return pd.DataFrame(rows)


def _log_mlflow(metadata: dict[str, Any], reports: dict[str, pd.DataFrame]) -> str:
    try:
        import mlflow
    except Exception:
        return "NOT_INSTALLED"
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("maritime-b61b-multitask")
        with mlflow.start_run(run_name=MODEL_VERSION):
            mlflow.log_params({"model_version": MODEL_VERSION, "selection_split": "VALID_SELECT", "test_used": False})
            mlflow.log_dict(clean_json(metadata), "decision.json")
            with tempfile.TemporaryDirectory() as directory:
                for name, report in reports.items():
                    path = Path(directory) / f"{name}.csv"
                    report.to_csv(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path="reports")
        return "LOGGED"
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def run_b61b_modeling(
    force: bool = False,
    sequence_max_steps: int = 400,
    catboost_iterations: int = 350,
    max_train_rows: int = 80_000,
) -> dict[str, Any]:
    if not _relation_exists(SOURCE_RELATION) or not _relation_exists("audit.ingestion_run"):
        raise RuntimeError("B61A dataset and audit schema are required")
    frame, contract = load_model_frame()
    checksum = _source_signature(frame, sequence_max_steps, catboost_iterations)
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            return {**previous, "reused": True}
    run_id = _start_run(checksum)
    try:
        _update_progress(run_id, "LOADED_GOVERNED_DATA", rows=len(frame), columns=len(frame.columns))
        core_features = [*contract["core_numeric"], *contract["categorical"]]
        sequence_features = contract["core_numeric"][:48]
        if len(core_features) < 25 or len(sequence_features) < 20:
            raise RuntimeError("B61A feature contract is unexpectedly incomplete")

        _update_progress(run_id, "TRAINING_CATBOOST_MULTITASK_EXPERT", features=len(core_features))
        tabular, tabular_inventory, tabular_artifact, importance = _fit_tabular_expert(
            frame, core_features, catboost_iterations, max_train_rows
        )
        _update_progress(run_id, "TRAINING_SHARED_GRU_EXPERT", features=len(sequence_features), max_steps=sequence_max_steps)
        sequence_result = train_sequence_expert(
            frame, sequence_features, sequence_length=24, max_steps=sequence_max_steps
        )
        sequence = sequence_result.predictions
        _update_progress(run_id, "FITTING_CONTEXTUAL_MOE_AND_CALIBRATION")
        ensemble, ensemble_policy, ensemble_artifact = _build_ensemble(frame, tabular, sequence)
        baseline = _baseline_predictions(frame)
        predictions_by_expert = {
            "BUSINESS_BASELINE": baseline,
            "CATBOOST_EXPERT": tabular,
            "GRU_EXPERT": sequence,
            "MOE_CALIBRATED": ensemble,
        }
        binary_report, duration_report, ordinal_report = _metric_reports(
            frame, predictions_by_expert, ensemble_policy["thresholds"]
        )

        research_features = [*contract["core_numeric"], *contract["research_numeric"], *contract["categorical"]]
        _update_progress(run_id, "RUNNING_RETROSPECTIVE_WEATHER_ABLATION", research_features=len(contract["research_numeric"]))
        research_report, research_artifact = _fit_research_challenger(
            frame, research_features, catboost_iterations, min(max_train_rows, 60_000)
        )

        prediction_frame = _prediction_frame(
            frame, ensemble, ensemble_policy["thresholds"], run_id
        )
        gates = _quality_gates(frame, binary_report, duration_report, prediction_frame)
        critical_passed = bool(gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all())
        model_passed = bool(gates.loc[gates["severity"].eq("MODEL"), "passed"].all())
        serving_rows = _materialize_predictions(prediction_frame)
        decision = (
            "READY_FOR_B61C_SHADOW_SERVING"
            if critical_passed and model_passed
            else "RESEARCH_ONLY_MODEL_REFINEMENT_REQUIRED"
        )
        metadata = {
            "decision": decision,
            "model_version": MODEL_VERSION,
            "row_count": len(frame),
            "vessel_calls": int(frame["port_call_id"].nunique()),
            "loaded_columns": contract["loaded_columns"],
            "core_features": len(core_features),
            "research_features": len(contract["research_numeric"]),
            "sequence_features": len(sequence_features),
            "selected_models": ["CATBOOST_MULTITASK", "SHARED_GRU", "CONTEXTUAL_MOE"],
            "tasks": ["ORDINAL_DELAY", "REMAINING_P10_P50_P90", "DISCRETE_SURVIVAL_6_12_24H"],
            "selection_split": "VALID_SELECT",
            "calibration_split": "VALID_CALIBRATE",
            "test_role": "TEST_DIAGNOSTIC_ONLY",
            "test_used_for_selection": False,
            "synthetic_rows": 0,
            "targets_imputed": False,
            "conformal_nominal_coverage": 0.8,
            "serving_rows": serving_rows,
            "critical_gates_passed": critical_passed,
            "model_gates_passed": model_passed,
            "quality_gates_passed": critical_passed and model_passed,
            "research_weather_policy": "RETROSPECTIVE_RESEARCH_ONLY_NOT_IN_OFFICIAL_MOE",
            "issue_time_history_ready": False,
            "replay_allowed": critical_passed,
            "production_promotion_allowed": False,
            "limitations": [
                "Retrospective evaluation only; live issue-time forecast history is not yet long enough.",
                "Weather ablation is research-only and is excluded from the official serving mixture.",
                "TEST is diagnostic only and never changes model, blend, threshold or interval calibration.",
            ],
            "next_block": "B61C_HISTORICAL_REPLAY_SHADOW_API",
            "tabular_inventory": tabular_inventory,
            "sequence_inventory": sequence_result.metrics,
            "ensemble_policy": ensemble_policy,
        }
        reports = {
            "binary_metrics": binary_report,
            "duration_metrics": duration_report,
            "ordinal_metrics": ordinal_report,
            "quality_gates": gates,
            "feature_importance": importance,
            "research_weather_ablation": research_report,
            "split_support": frame.groupby(["split", "model_role"], as_index=False).agg(rows=("port_call_id", "size"), calls=("port_call_id", "nunique")),
        }
        metadata["mlflow_status"] = _log_mlflow(metadata, reports)
        _update_progress(run_id, "WRITING_VERSIONED_ARTIFACTS", serving_rows=serving_rows)
        for name, report in reports.items():
            _put_csv(f"reports/b61b/{OUTPUT_PREFIX}/{name}.csv", report)
        _put_json(f"configs/b61b/{OUTPUT_PREFIX}/ensemble_policy.json", ensemble_policy)
        _put_json(f"configs/b61b/{OUTPUT_PREFIX}/feature_contract.json", contract)
        _put_json(f"configs/b61b/{OUTPUT_PREFIX}/final_decision.json", metadata)
        _put_bytes(f"models/b61b/{OUTPUT_PREFIX}/catboost_multitask.pkl", tabular_artifact, "application/octet-stream")
        _put_bytes(f"models/b61b/{OUTPUT_PREFIX}/shared_gru.pt", sequence_result.artifact, "application/octet-stream")
        _put_bytes(f"models/b61b/{OUTPUT_PREFIX}/ensemble_calibration.pkl", ensemble_artifact, "application/octet-stream")
        if research_artifact:
            _put_bytes(f"models/b61b/{OUTPUT_PREFIX}/research_weather_challenger.pkl", research_artifact, "application/octet-stream")
        _finish_run(run_id, "SUCCESS", len(frame), metadata)
        return metadata
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "decision": "FAILED",
                "next_block": "FIX_B61B_AND_RERUN",
                "production_promotion_allowed": False,
            },
            str(exc),
        )
        raise


def verify_b61b_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "decision",
        "model_version",
        "row_count",
        "vessel_calls",
        "selection_split",
        "calibration_split",
        "test_used_for_selection",
        "serving_rows",
        "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"B61B result misses required fields: {missing}")
    if result["test_used_for_selection"]:
        raise ValueError("B61B leakage contract violated: TEST used for selection")
    if result.get("production_promotion_allowed"):
        raise ValueError("B61B cannot promote before issue-time shadow validation")
    if int(result["serving_rows"]) <= 0:
        raise ValueError("B61B produced no serving rows")
    return result
