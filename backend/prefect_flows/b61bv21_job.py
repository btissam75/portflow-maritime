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
from psycopg2.extras import Json, execute_values

from prefect_flows.b61b_core import (
    HAZARD_HORIZONS,
    HAZARD_TARGETS,
    RISK_NAMES,
    RISK_TARGETS,
    clean_json,
    enforce_hazard_order,
    enforce_quantile_order,
    enforce_risk_order,
    risk_from_class_probabilities,
)
from prefect_flows.b61bv2_core import (
    apply_adaptive_conformal,
    coherent_outputs,
    grouped_bootstrap_ci,
    select_threshold,
    weighted_binary_metrics,
    weighted_regression_metrics,
)
from prefect_flows.b61bv2_job import (
    _db_connection,
    _metric_reports,
    _model_frame,
    _predict_binary_batched,
    _predict_ordinal_batched,
    _predict_regression_batched,
    _relation_exists,
    load_governed_frames,
)
from prefect_flows.b61bv21_core import (
    MODEL_VERSION,
    RARE_RISK_TASKS,
    SOURCE_MODEL_VERSION,
    build_recalibration_evidence,
    calibration_gate_rows,
    evidence_as_dict,
    fit_rank_preserving_platt,
)


SOURCE_NAME = "b61b_v21_recalibration_only"
DATASET_NAME = "maritime_port_call_multitask_predictions_v21"
TARGET_RELATION = "serving.maritime_port_call_multitask_prediction_v21"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=2.1"
SOURCE_PREFIX = "version=2"
SOURCE_POLICY_KEY = f"configs/b61bv2/{SOURCE_PREFIX}/ensemble_policy.json"
SOURCE_DECISION_KEY = f"configs/b61bv2/{SOURCE_PREFIX}/final_decision.json"
SOURCE_CALIBRATION_KEY = f"models/b61bv2/{SOURCE_PREFIX}/hybrid_calibration.pkl"


def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        region_name="us-east-1",
    )


def _get_bytes(key: str) -> bytes:
    return _s3_client().get_object(Bucket=OUTPUT_BUCKET, Key=key)["Body"].read()


def _get_json(key: str) -> dict[str, Any]:
    return json.loads(_get_bytes(key).decode("utf-8"))


def _object_etag(key: str) -> str:
    result = _s3_client().head_object(Bucket=OUTPUT_BUCKET, Key=key)
    return str(result["ETag"]).strip('"')


def _put_bytes(key: str, payload: bytes, content_type: str) -> str:
    _s3_client().put_object(
        Bucket=OUTPUT_BUCKET, Key=key, Body=payload, ContentType=content_type
    )
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _put_json(key: str, payload: Any) -> str:
    return _put_bytes(
        key,
        json.dumps(clean_json(payload), indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )


def _put_csv(key: str, frame: pd.DataFrame) -> str:
    return _put_bytes(key, frame.to_csv(index=False).encode("utf-8"), "text/csv")


def _load_source_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    decision = _get_json(SOURCE_DECISION_KEY)
    policy = _get_json(SOURCE_POLICY_KEY)
    calibration = pickle.loads(_get_bytes(SOURCE_CALIBRATION_KEY))
    if decision.get("model_version") != SOURCE_MODEL_VERSION:
        raise RuntimeError(
            f"Unexpected B61B-v2 source model: {decision.get('model_version')}"
        )
    if calibration.get("model_version") != SOURCE_MODEL_VERSION:
        raise RuntimeError("B61B-v2 calibration artifact version mismatch")
    champions = policy.get("champions") or {}
    required = {
        "RISK_GT1",
        "RISK_GT3",
        "RISK_GT6",
        "HAZARD_6H",
        "HAZARD_12H",
        "HAZARD_24H",
        "REMAINING_DURATION",
    }
    if required.difference(champions):
        raise RuntimeError("B61B-v2 champion policy is incomplete")
    return decision, policy, calibration


def _track_key(track: str) -> str:
    if track == "SEQUENCE_REAL_ONLY":
        raise RuntimeError(
            "B61B-v2.1 recalibration-only supports the selected tabular artifacts; "
            "the current source policy unexpectedly selected the sequence expert"
        )
    return f"models/b61bv2/{SOURCE_PREFIX}/{track.lower()}.pkl"


def _load_track_artifacts(champions: dict[str, str]) -> tuple[dict[str, Any], pd.DataFrame]:
    artifacts: dict[str, Any] = {}
    rows = []
    for track in sorted(set(champions.values())):
        key = _track_key(track)
        payload = _get_bytes(key)
        artifact = pickle.loads(payload)
        if artifact.get("model_version") != SOURCE_MODEL_VERSION:
            raise RuntimeError(f"Source artifact version mismatch for {track}")
        if artifact.get("track") != track:
            raise RuntimeError(f"Source artifact track mismatch for {track}")
        artifacts[track] = artifact
        rows.append(
            {
                "track": track,
                "source_key": key,
                "etag": _object_etag(key),
                "bytes": len(payload),
                "feature_count": len(artifact.get("features", [])),
                "training_executed": False,
            }
        )
    for name, key in (
        ("SOURCE_POLICY", SOURCE_POLICY_KEY),
        ("SOURCE_DECISION", SOURCE_DECISION_KEY),
        ("SOURCE_CALIBRATION", SOURCE_CALIBRATION_KEY),
    ):
        rows.append(
            {
                "track": name,
                "source_key": key,
                "etag": _object_etag(key),
                "bytes": None,
                "feature_count": None,
                "training_executed": False,
            }
        )
    return artifacts, pd.DataFrame(rows)


def _predict_track(real: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, np.ndarray]:
    features = list(artifact["features"])
    models = artifact["models"]
    risk = risk_from_class_probabilities(
        _predict_ordinal_batched(models["ordinal"], real, features)
    )
    duration = enforce_quantile_order(
        _predict_regression_batched(models["quantile"], real, features)
    )
    hazards = []
    for horizon in HAZARD_HORIZONS:
        model = models["hazard"][str(horizon)]
        if isinstance(model, dict) and "constant" in model:
            probability = np.repeat(float(model["constant"]), len(real))
        else:
            probability = _predict_binary_batched(model, real, features)
        hazards.append(probability)
    return coherent_outputs(risk, duration, np.column_stack(hazards))


def _assemble_selected(
    tracks: dict[str, dict[str, np.ndarray]], champions: dict[str, str]
) -> dict[str, np.ndarray]:
    risk = np.column_stack(
        [
            tracks[champions[f"RISK_{name.upper()}"]]["risk"][:, index]
            for index, name in enumerate(RISK_NAMES)
        ]
    )
    hazard = np.column_stack(
        [
            tracks[champions[f"HAZARD_{horizon}H"]]["hazard"][:, index]
            for index, horizon in enumerate(HAZARD_HORIZONS)
        ]
    )
    duration = tracks[champions["REMAINING_DURATION"]]["quantiles"]
    return coherent_outputs(risk, duration, hazard)


def _apply_calibration(
    real: pd.DataFrame,
    selected: dict[str, np.ndarray],
    calibration_artifact: dict[str, Any],
    replacements: dict[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    replacements = replacements or {}
    calibrators = {**calibration_artifact["calibrators"], **replacements}
    risk = enforce_risk_order(
        np.column_stack(
            [
                calibrators[f"risk_{name}"].predict(selected["risk"][:, index])
                for index, name in enumerate(RISK_NAMES)
            ]
        )
    )
    duration = apply_adaptive_conformal(
        real,
        selected["quantiles"],
        calibration_artifact["policy"]["conformal"],
    )
    hazard = enforce_hazard_order(
        np.column_stack(
            [
                calibrators[f"hazard_{horizon}"].predict(
                    selected["hazard"][:, index]
                )
                for index, horizon in enumerate(HAZARD_HORIZONS)
            ]
        )
    )
    return coherent_outputs(risk, duration, hazard)


def _fit_rare_recalibration(
    real: pd.DataFrame,
    selected: dict[str, np.ndarray],
    prior_policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], pd.DataFrame, list[Any]]:
    calibration = real["model_role"].eq("VALID_CALIBRATE").to_numpy()
    eligible = real["early_warning_eligible"].fillna(False).astype(bool).to_numpy()
    weights = real["per_call_sample_weight"].to_numpy(dtype="float64")
    replacements: dict[str, Any] = {}
    thresholds = dict(prior_policy["thresholds"])
    report = []
    evidence = []
    for name in RARE_RISK_TASKS:
        index = RISK_NAMES.index(name)
        target = RISK_TARGETS[index]
        mask = calibration & eligible
        actual = real[target].astype(int).to_numpy()
        raw = selected["risk"][:, index]
        calibrator = fit_rank_preserving_platt(actual[mask], raw[mask], weights[mask])
        calibrated = calibrator.predict(raw)
        false_negative_cost = 3.0 if name == "gt6" else 2.0
        threshold = select_threshold(
            actual[mask], calibrated[mask], weights[mask], false_negative_cost
        )
        thresholds[name] = threshold
        replacements[f"risk_{name}"] = calibrator
        item = build_recalibration_evidence(
            name,
            calibrator.method,
            actual[mask],
            raw[mask],
            calibrated[mask],
            weights[mask],
        )
        evidence.append(item)
        before = weighted_binary_metrics(
            actual[mask], raw[mask], sample_weight=weights[mask]
        )
        after = weighted_binary_metrics(
            actual[mask], calibrated[mask], threshold, weights[mask]
        )
        report.append(
            {
                **evidence_as_dict(item),
                "fit_role": "VALID_CALIBRATE",
                "eligibility_filter": "early_warning_eligible=true",
                "threshold": threshold,
                "false_negative_cost": false_negative_cost,
                "raw_roc_auc": before["roc_auc"],
                "calibrated_roc_auc": after["roc_auc"],
                "raw_average_precision": before["average_precision"],
                "calibrated_average_precision": after["average_precision"],
                "raw_brier": before["brier"],
                "calibrated_brier": after["brier"],
                "raw_ece": before["ece_10"],
                "calibrated_ece": after["ece_10"],
                "predicted_positive_rows": int((calibrated[mask] >= threshold).sum()),
            }
        )
    return replacements, thresholds, pd.DataFrame(report), evidence


def _corrected_bootstrap(
    real: pd.DataFrame,
    ensemble: dict[str, np.ndarray],
    replicates: int,
) -> pd.DataFrame:
    test_mask = real["model_role"].eq("TEST_DIAGNOSTIC_ONLY").to_numpy()
    test = real.loc[test_mask].reset_index(drop=True)
    risk = ensemble["risk"][test_mask]
    duration = ensemble["quantiles"][test_mask]
    weights = test["per_call_sample_weight"].to_numpy(dtype="float64")
    rows = []
    eligible = test["early_warning_eligible"].fillna(False).astype(bool).to_numpy()
    for name in RARE_RISK_TASKS:
        index = RISK_NAMES.index(name)
        target = RISK_TARGETS[index]
        task_frame = test.loc[eligible].reset_index(drop=True)
        actual = test.loc[eligible, target].astype(int).to_numpy()
        probability = risk[eligible, index]
        weight = weights[eligible]

        def metric(positions: np.ndarray) -> float:
            result = weighted_binary_metrics(
                actual[positions], probability[positions], sample_weight=weight[positions]
            )
            return float(result["average_precision"] or 0.0)

        point, lower, upper, used = grouped_bootstrap_ci(
            task_frame, metric, replicates
        )
        rows.append(
            {
                "task": f"DELAY_{name.upper()}",
                "metric": "AVERAGE_PRECISION",
                "point": point,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "replicates": used,
                "bootstrap_unit": "PORT_CALL",
                "eligibility_filter": "early_warning_eligible=true",
                "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            }
        )
    actual_remaining = test["target_remaining_h"].to_numpy(dtype="float64")

    def mae_metric(positions: np.ndarray) -> float:
        return float(
            weighted_regression_metrics(
                actual_remaining[positions], duration[positions], weights[positions]
            )["mae_p50"]
        )

    point, lower, upper, used = grouped_bootstrap_ci(test, mae_metric, replicates)
    rows.append(
        {
            "task": "REMAINING_DURATION",
            "metric": "MAE_P50",
            "point": point,
            "ci_lower_95": lower,
            "ci_upper_95": upper,
            "replicates": used,
            "bootstrap_unit": "PORT_CALL",
            "eligibility_filter": "target_remaining_h_not_null",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
        }
    )
    return pd.DataFrame(rows)


def _prediction_frame(
    real: pd.DataFrame,
    ensemble: dict[str, np.ndarray],
    policy: dict[str, Any],
    run_id: str,
) -> pd.DataFrame:
    keep = real["model_role"].isin(
        ["VALID_SELECT", "VALID_CALIBRATE", "TEST_DIAGNOSTIC_ONLY"]
    ).to_numpy()
    source = real.loc[keep]
    risk = ensemble["risk"][keep]
    duration = ensemble["quantiles"][keep]
    hazard = ensemble["hazard"][keep]
    thresholds = policy["thresholds"]
    predicted_class = np.sum(
        np.column_stack(
            [
                risk[:, index] >= thresholds.get(name, 0.5)
                for index, name in enumerate(RISK_NAMES)
            ]
        ),
        axis=1,
    )
    labels = np.asarray(["LT_OR_EQ_1H", "GT1_TO_3H", "GT3_TO_6H", "GT6H"])
    return pd.DataFrame(
        {
            "model_version": MODEL_VERSION,
            "source_model_version": SOURCE_MODEL_VERSION,
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
            "selected_policy": [policy["champions"]] * int(keep.sum()),
            "calibration_policy": "PLATT_GT3_GT6_REUSE_OTHER_V2_CALIBRATORS",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "production_claim_allowed": False,
            "materialization_run_id": run_id,
        }
    )


def _materialize_predictions(frame: pd.DataFrame) -> int:
    columns = list(frame.columns)
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TARGET_RELATION} (
                model_version TEXT NOT NULL,
                source_model_version TEXT NOT NULL,
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
                selected_policy JSONB NOT NULL,
                calibration_policy TEXT NOT NULL,
                test_role TEXT NOT NULL,
                production_claim_allowed BOOLEAN NOT NULL,
                materialization_run_id TEXT NOT NULL,
                PRIMARY KEY (model_version, port_call_id, landmark_at)
            )
            """
        )
        cursor.execute(
            f"DELETE FROM {TARGET_RELATION} WHERE model_version=%s", (MODEL_VERSION,)
        )
        column_sql = ", ".join(columns)
        for start in range(0, len(frame), 2_000):
            records = []
            for row in frame.iloc[start : start + 2_000].itertuples(index=False):
                values = list(row)
                policy_index = columns.index("selected_policy")
                values[policy_index] = Json(values[policy_index])
                records.append(tuple(values))
            execute_values(
                cursor,
                f"INSERT INTO {TARGET_RELATION} ({column_sql}) VALUES %s",
                records,
                page_size=2_000,
            )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_b61bv21_prediction_landmark "
            f"ON {TARGET_RELATION} (landmark_at DESC)"
        )
    return len(frame)


def _quality_gates(
    real: pd.DataFrame,
    predictions: pd.DataFrame,
    evidence: list[Any],
    binary_report: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    finite = [
        "p_delay_gt1",
        "p_delay_gt3",
        "p_delay_gt6",
        "remaining_p10_h",
        "remaining_p50_h",
        "remaining_p90_h",
        "p_gt3_breach_within_6h",
        "p_gt3_breach_within_12h",
        "p_gt3_breach_within_24h",
    ]
    rows: list[dict[str, Any]] = [
        {
            "check": "SOURCE_MODEL_VERSION_PINNED",
            "passed": True,
            "severity": "CRITICAL",
            "value": SOURCE_MODEL_VERSION,
        },
        {
            "check": "SOURCE_ARTIFACTS_IMMUTABLY_HASHED",
            "passed": bool(manifest["etag"].astype(str).str.len().gt(0).all()),
            "severity": "CRITICAL",
            "value": len(manifest),
        },
        {
            "check": "NO_MODEL_TRAINING_EXECUTED",
            "passed": True,
            "severity": "CRITICAL",
            "value": False,
        },
        {
            "check": "CALIBRATORS_FIT_ON_VALID_CALIBRATE_ONLY",
            "passed": True,
            "severity": "CRITICAL",
            "value": "VALID_CALIBRATE",
        },
        {
            "check": "TEST_NOT_USED_FOR_FIT_OR_THRESHOLDS",
            "passed": True,
            "severity": "CRITICAL",
            "value": False,
        },
        {
            "check": "TEST_DECLARED_REUSED_NOT_CONFIRMATORY",
            "passed": True,
            "severity": "CRITICAL",
            "value": True,
        },
        {
            "check": "PORT_CALL_ONE_TEMPORAL_SPLIT",
            "passed": bool(real.groupby("port_call_id")["split"].nunique().max() == 1),
            "severity": "CRITICAL",
            "value": None,
        },
        {
            "check": "FINITE_OUTPUTS",
            "passed": bool(np.isfinite(predictions[finite].to_numpy()).all()),
            "severity": "CRITICAL",
            "value": None,
        },
        {
            "check": "RISK_MONOTONIC",
            "passed": bool(
                (
                    (predictions["p_delay_gt1"] >= predictions["p_delay_gt3"])
                    & (predictions["p_delay_gt3"] >= predictions["p_delay_gt6"])
                ).all()
            ),
            "severity": "CRITICAL",
            "value": None,
        },
        {
            "check": "HAZARD_MONOTONIC",
            "passed": bool(
                (
                    (
                        predictions["p_gt3_breach_within_6h"]
                        <= predictions["p_gt3_breach_within_12h"]
                    )
                    & (
                        predictions["p_gt3_breach_within_12h"]
                        <= predictions["p_gt3_breach_within_24h"]
                    )
                ).all()
            ),
            "severity": "CRITICAL",
            "value": None,
        },
        {
            "check": "QUANTILES_MONOTONIC",
            "passed": bool(
                (
                    (predictions["remaining_p10_h"] <= predictions["remaining_p50_h"])
                    & (predictions["remaining_p50_h"] <= predictions["remaining_p90_h"])
                ).all()
            ),
            "severity": "CRITICAL",
            "value": None,
        },
    ]
    rows.extend(calibration_gate_rows(evidence))
    test = binary_report.loc[
        binary_report["role"].eq("TEST_DIAGNOSTIC_ONLY")
        & binary_report["expert"].eq("V21_RECALIBRATED")
        & binary_report["task"].isin(["DELAY_GT3", "DELAY_GT6"])
    ]
    for row in test.itertuples(index=False):
        rows.append(
            {
                "check": f"{row.task}_REUSED_TEST_AP_DIAGNOSTIC",
                "passed": True,
                "severity": "DIAGNOSTIC",
                "value": row.average_precision,
            }
        )
    rows.append(
        {
            "check": "FRESH_FORWARD_CONFIRMATORY_WINDOW_REQUIRED",
            "passed": True,
            "severity": "GOVERNANCE",
            "value": True,
        }
    )
    return pd.DataFrame(rows)


def _source_signature(real: pd.DataFrame, manifest: pd.DataFrame) -> str:
    digest = hashlib.sha256(MODEL_VERSION.encode("ascii"))
    digest.update(
        pd.util.hash_pandas_object(
            real[["port_call_id", "landmark_at", "target_departure_delay_h"]],
            index=False,
        ).to_numpy(dtype="uint64").tobytes()
    )
    for row in manifest.sort_values("source_key").itertuples(index=False):
        digest.update(f"{row.source_key}:{row.etag}".encode("utf-8"))
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
        if row is None or not _relation_exists(TARGET_RELATION):
            return None
        cursor.execute(
            f"SELECT COUNT(*) FROM {TARGET_RELATION} WHERE model_version=%s",
            (MODEL_VERSION,),
        )
        return dict(row[0]) if int(cursor.fetchone()[0]) > 0 else None


def _start_run(checksum: str) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "source_model_version": SOURCE_MODEL_VERSION,
        "orchestrator": "PREFECT",
        "training_executed": False,
        "fit_split": "VALID_CALIBRATE",
        "calibration_split_reused": True,
        "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
        "test_used_for_fit_or_thresholds": False,
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


def _log_mlflow(metadata: dict[str, Any], reports: dict[str, pd.DataFrame]) -> str:
    try:
        import mlflow
    except Exception:
        return "NOT_INSTALLED"
    try:
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        mlflow.set_experiment("maritime-b61b-v21-recalibration-only")
        with mlflow.start_run(run_name=MODEL_VERSION):
            mlflow.log_params(
                {
                    "source_model_version": SOURCE_MODEL_VERSION,
                    "training_executed": False,
                    "rare_calibrator": "PLATT",
                    "fit_split": "VALID_CALIBRATE",
                    "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
                }
            )
            mlflow.log_dict(clean_json(metadata), "decision.json")
            with tempfile.TemporaryDirectory() as directory:
                for name, report in reports.items():
                    path = Path(directory) / f"{name}.csv"
                    report.to_csv(path, index=False)
                    mlflow.log_artifact(str(path), artifact_path="reports")
        return "LOGGED"
    except Exception as exc:
        return f"UNAVAILABLE:{type(exc).__name__}"


def run_b61bv21_recalibration(
    force: bool = False,
    bootstrap_replicates: int = 500,
) -> dict[str, Any]:
    for relation in ("features.maritime_port_call_governed_v1", "audit.ingestion_run"):
        if not _relation_exists(relation):
            raise RuntimeError(f"Required relation is missing: {relation}")
    real, _synthetic, source_info = load_governed_frames()
    source_decision, source_policy, source_calibration = _load_source_contract()
    artifacts, manifest = _load_track_artifacts(source_policy["champions"])
    checksum = _source_signature(real, manifest)
    if not force:
        previous = _previous_success(checksum)
        if previous is not None:
            return {**previous, "reused": True}
    run_id = _start_run(checksum)
    try:
        _update_progress(
            run_id,
            "LOADED_IMMUTABLE_B61B_V2_ARTIFACTS",
            rows=len(real),
            calls=real["port_call_id"].nunique(),
            source_artifacts=len(manifest),
        )
        track_predictions = {}
        for track, artifact in artifacts.items():
            _update_progress(run_id, "REPLAYING_FROZEN_EXPERT", track=track)
            track_predictions[track] = _predict_track(real, artifact)
        selected = _assemble_selected(track_predictions, source_policy["champions"])
        prior = _apply_calibration(real, selected, source_calibration)

        _update_progress(run_id, "FITTING_RARE_RISK_PLATT_ON_VALID_CALIBRATE")
        replacements, thresholds, calibration_report, evidence = _fit_rare_recalibration(
            real, selected, source_policy
        )
        recalibrated = _apply_calibration(
            real, selected, source_calibration, replacements
        )
        policy = {
            **source_policy,
            "thresholds": thresholds,
            "source_model_version": SOURCE_MODEL_VERSION,
            "calibration_version": MODEL_VERSION,
            "rare_risk_calibrators": {name: "PLATT" for name in RARE_RISK_TASKS},
            "other_calibrators": "REUSED_FROM_B61B_V2",
            "fit_split": "VALID_CALIBRATE",
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "test_used_for_fit_or_thresholds": False,
            "fresh_forward_confirmation_required": True,
        }
        all_predictions = {
            "RAW_SELECTED": selected,
            "PRIOR_V2_CALIBRATED": prior,
            "V21_RECALIBRATED": recalibrated,
        }
        prior_binary, prior_duration, prior_survival = _metric_reports(
            real,
            {
                "RAW_SELECTED": all_predictions["RAW_SELECTED"],
                "PRIOR_V2_CALIBRATED": all_predictions["PRIOR_V2_CALIBRATED"],
            },
            source_policy["thresholds"],
        )
        new_binary, new_duration, new_survival = _metric_reports(
            real,
            {"V21_RECALIBRATED": all_predictions["V21_RECALIBRATED"]},
            thresholds,
        )
        binary = pd.concat([prior_binary, new_binary], ignore_index=True)
        duration = pd.concat([prior_duration, new_duration], ignore_index=True)
        survival = pd.concat([prior_survival, new_survival], ignore_index=True)
        bootstrap = _corrected_bootstrap(
            real, recalibrated, max(200, int(bootstrap_replicates))
        )
        prediction_frame = _prediction_frame(real, recalibrated, policy, run_id)
        gates = _quality_gates(real, prediction_frame, evidence, binary, manifest)
        critical_passed = bool(
            gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all()
        )
        calibration_passed = bool(
            gates.loc[gates["severity"].eq("CALIBRATION"), "passed"].all()
        )
        replay_allowed = critical_passed and calibration_passed
        serving_rows = _materialize_predictions(prediction_frame)
        recalibration_artifact = pickle.dumps(
            {
                "model_version": MODEL_VERSION,
                "source_model_version": SOURCE_MODEL_VERSION,
                "policy": policy,
                "replacement_calibrators": replacements,
                "source_artifact_etags": manifest[
                    ["source_key", "etag"]
                ].to_dict("records"),
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        decision = (
            "READY_FOR_B61C_HISTORICAL_REPLAY_AND_FRESH_SHADOW_VALIDATION"
            if replay_allowed
            else "RESEARCH_ONLY_B61B_V21_CALIBRATION_REFINEMENT_REQUIRED"
        )
        metadata = {
            "decision": decision,
            "model_version": MODEL_VERSION,
            "source_model_version": SOURCE_MODEL_VERSION,
            "source_decision": source_decision.get("decision"),
            "row_count": int(len(real)),
            "vessel_calls": int(real["port_call_id"].nunique()),
            "serving_rows": int(serving_rows),
            "training_executed": False,
            "recalibrated_tasks": ["RISK_GT3", "RISK_GT6"],
            "reused_components": [
                "CATBOOST_EXPERTS",
                "RISK_GT1_CALIBRATOR",
                "HAZARD_CALIBRATORS",
                "DURATION_CONFORMAL_POLICY",
                "CHAMPION_ROUTING",
            ],
            "fit_split": "VALID_CALIBRATE",
            "calibration_split_reused": True,
            "test_role": "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY",
            "test_used_for_fit_or_thresholds": False,
            "test_used_for_replay_decision": False,
            "corrected_bootstrap_filter": "early_warning_eligible=true",
            "bootstrap_unit": "PORT_CALL",
            "bootstrap_replicates": max(200, int(bootstrap_replicates)),
            "critical_gates_passed": critical_passed,
            "calibration_gates_passed": calibration_passed,
            "replay_allowed": replay_allowed,
            "production_promotion_allowed": False,
            "fresh_forward_confirmation_required": True,
            "source_contract": source_info,
            "limitations": [
                "The previous TEST window was already inspected in B61B-v2 and is diagnostic only.",
                "Formal promotion requires a new forward issue-time shadow window.",
                "This run recalibrates frozen predictions and performs no model training.",
                "The result is predictive, not a causal effect estimate.",
            ],
            "next_block": (
                "B61C_HISTORICAL_REPLAY_AND_FRESH_SHADOW_API"
                if replay_allowed
                else "B61B_V21_REVIEW_FAILED_CALIBRATION_GATES"
            ),
        }
        reports = {
            "recalibration_report": calibration_report,
            "binary_metrics": binary,
            "duration_metrics": duration,
            "survival_metrics": survival,
            "corrected_test_bootstrap": bootstrap,
            "quality_gates": gates,
            "source_artifact_manifest": manifest,
        }
        metadata["mlflow_status"] = _log_mlflow(metadata, reports)
        _update_progress(run_id, "WRITING_VERSIONED_RECALIBRATION_ARTIFACTS")
        for name, report in reports.items():
            _put_csv(f"reports/b61bv21/{OUTPUT_PREFIX}/{name}.csv", report)
        _put_json(f"configs/b61bv21/{OUTPUT_PREFIX}/recalibration_policy.json", policy)
        _put_json(f"configs/b61bv21/{OUTPUT_PREFIX}/final_decision.json", metadata)
        _put_bytes(
            f"models/b61bv21/{OUTPUT_PREFIX}/recalibration_only.pkl",
            recalibration_artifact,
            "application/octet-stream",
        )
        _finish_run(run_id, "SUCCESS", len(real), metadata)
        return metadata
    except Exception as exc:
        _finish_run(
            run_id,
            "FAILED",
            None,
            {
                "decision": "FAILED",
                "training_executed": False,
                "test_used_for_fit_or_thresholds": False,
                "production_promotion_allowed": False,
                "next_block": "FIX_B61B_V21_AND_RERUN",
            },
            str(exc),
        )
        raise


def verify_b61bv21_result(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "decision",
        "model_version",
        "source_model_version",
        "row_count",
        "vessel_calls",
        "serving_rows",
        "training_executed",
        "fit_split",
        "test_role",
        "test_used_for_fit_or_thresholds",
        "fresh_forward_confirmation_required",
        "next_block",
    }
    missing = sorted(required.difference(result))
    if missing:
        raise ValueError(f"B61B-v2.1 result misses required fields: {missing}")
    if result["training_executed"]:
        raise ValueError("B61B-v2.1 recalibration-only unexpectedly trained a model")
    if result["test_used_for_fit_or_thresholds"]:
        raise ValueError("B61B-v2.1 leakage contract violated")
    if result["test_role"] != "REUSED_DIAGNOSTIC_NOT_CONFIRMATORY":
        raise ValueError("B61B-v2.1 must disclose that TEST was previously inspected")
    if result.get("production_promotion_allowed"):
        raise ValueError("B61B-v2.1 cannot promote before fresh forward validation")
    if int(result["serving_rows"]) <= 0:
        raise ValueError("B61B-v2.1 produced no serving rows")
    return result
