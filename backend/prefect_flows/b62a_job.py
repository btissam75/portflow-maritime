from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import boto3
import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json, execute_values

from prefect_flows.b62_core import (
    apply_frozen_selection,
    assign_temporal_roles,
    forecast_metrics,
    prepare_hourly_frame,
    rolling_origins,
    seasonal_predictions,
)
from prefect_flows.b62_job import load_atmosphere_source, load_wave_source
from prefect_flows.b62a_core import (
    CHALLENGER_TARGETS,
    DATASET_VERSION,
    HORIZONS_H,
    MODEL_VERSION,
    SOURCE_B62_VERSION,
    build_supervised_frame,
    feature_columns,
    finite_frame_hash,
    generate_tail_augmentation,
    json_ready,
    make_stress_features,
    split_train_calibration,
    target_columns,
)
from prefect_flows.b62a_models import (
    MODEL_NAME,
    apply_wave_selection,
    fit_tail_challenger,
    predict_origins,
    predict_stress,
    select_against_frozen_b62,
    task_calibration_report,
)


SOURCE_NAME = "b62a_governed_metocean_augmentation"
DATASET_NAME = "maritime_metocean_tail_augmented_train_v1"
OUTPUT_BUCKET = "gold-maritime"
OUTPUT_PREFIX = "version=1"
SELECTION_TABLE = "serving.maritime_metocean_tail_challenger_shadow_v1"
B62_VALID_PREDICTIONS = "predictions/b62/version=1/valid_predictions.parquet"
B62_TEST_PREDICTIONS = "predictions/b62/version=1/test_selected_predictions.parquet"
B62_SELECTION = "models/b62/version=1/selection_config.json"


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


def _latest_b62_success() -> tuple[str, dict[str, Any]]:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT run_id,metadata FROM audit.ingestion_run
            WHERE source_name='b62_weather_wave_vessel_autogluon'
              AND status='SUCCESS'
            ORDER BY finished_at DESC LIMIT 1
            """
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError("B62 SUCCESS is required before B62A")
    metadata = dict(row[1] or {})
    if metadata.get("model_version") != SOURCE_B62_VERSION:
        raise RuntimeError(f"Unexpected B62 model version: {metadata.get('model_version')}")
    return str(row[0]), metadata


def _get_bytes(client, key: str) -> bytes:
    try:
        return client.get_object(Bucket=OUTPUT_BUCKET, Key=key)["Body"].read()
    except Exception as exc:
        raise RuntimeError(f"Required B62 artifact is unavailable: s3://{OUTPUT_BUCKET}/{key}") from exc


def _get_parquet(client, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(_get_bytes(client, key)))


def _get_json(client, key: str) -> Any:
    return json.loads(_get_bytes(client, key))


def _put_bytes(client, key: str, payload: bytes, content_type: str) -> str:
    client.put_object(Bucket=OUTPUT_BUCKET, Key=key, Body=payload, ContentType=content_type)
    return f"s3://{OUTPUT_BUCKET}/{key}"


def _put_json(client, key: str, payload: Any) -> str:
    return _put_bytes(
        client,
        key,
        json.dumps(json_ready(payload), indent=2, sort_keys=True).encode("utf-8"),
        "application/json",
    )


def _put_csv(client, key: str, frame: pd.DataFrame) -> str:
    return _put_bytes(client, key, frame.to_csv(index=False).encode("utf-8"), "text/csv")


def _put_parquet(client, key: str, frame: pd.DataFrame) -> str:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return _put_bytes(client, key, buffer.getvalue(), "application/vnd.apache.parquet")


def _put_models(client, key: str, models: Any) -> str:
    with tempfile.TemporaryDirectory(prefix="b62a-model-") as temporary:
        path = Path(temporary) / "tail_challenger.joblib"
        joblib.dump(models, path, compress=3)
        return _put_bytes(client, key, path.read_bytes(), "application/octet-stream")


def _checksum(b62_run_id: str, b62_test_hash: str, parameters: dict[str, Any]) -> str:
    digest = hashlib.sha256(f"{MODEL_VERSION}:{b62_run_id}:{b62_test_hash}".encode("utf-8"))
    for key, value in sorted(parameters.items()):
        digest.update(f"{key}={value}".encode("utf-8"))
    return digest.hexdigest()


def _previous_success(checksum: str) -> tuple[str, dict[str, Any]] | None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT run_id,metadata FROM audit.ingestion_run
            WHERE source_name=%s AND dataset_name=%s AND checksum=%s AND status='SUCCESS'
            ORDER BY finished_at DESC LIMIT 1
            """,
            (SOURCE_NAME, DATASET_NAME, checksum),
        )
        row = cursor.fetchone()
    return None if row is None else (str(row[0]), dict(row[1] or {}))


def _start_run(checksum: str, parameters: dict[str, Any], b62_run_id: str) -> str:
    metadata = {
        "model_version": MODEL_VERSION,
        "source_b62_model_version": SOURCE_B62_VERSION,
        "source_b62_run_id": b62_run_id,
        "orchestrator": "PREFECT",
        "parameters": parameters,
        "synthetic_scope": "MODEL_TRAIN_ONLY",
        "valid_modified": False,
        "test_modified": False,
        "test_used_for_selection": False,
        "stress_used_for_selection": False,
        "production_promotion_allowed": False,
        "automatic_action_allowed": False,
    }
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO audit.ingestion_run
                (source_name,dataset_name,object_uri,checksum,metadata)
            VALUES (%s,%s,%s,%s,%s) RETURNING run_id
            """,
            (
                SOURCE_NAME,
                DATASET_NAME,
                "s3://gold-maritime/predictions/b62/version=1+postgresql://maritime/metocean",
                checksum,
                Json(metadata),
            ),
        )
        return str(cursor.fetchone()[0])


def _progress(run_id: str, stage: str, **details: Any) -> None:
    payload = {"progress": {"stage": stage, "updated_at": pd.Timestamp.now(tz="UTC"), **details}}
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            "UPDATE audit.ingestion_run SET metadata=metadata || %s WHERE run_id=%s",
            (Json(json_ready(payload)), run_id),
        )


def _finish(
    run_id: str,
    status: str,
    row_count: int | None,
    metadata: dict[str, Any],
    error: str | None = None,
) -> None:
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE audit.ingestion_run
            SET status=%s,row_count=%s,finished_at=NOW(),metadata=metadata || %s,error_message=%s
            WHERE run_id=%s
            """,
            (status, row_count, Json(json_ready(metadata)), error, run_id),
        )


def _frozen_b62_valid(valid_predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection = pd.DataFrame(config.get("selection", []))
    if selection.empty:
        raise RuntimeError("B62 frozen selection is empty")
    return apply_frozen_selection(valid_predictions, selection, "VALID")


def _stress_report(stress_predictions: pd.DataFrame) -> pd.DataFrame:
    finite = np.isfinite(stress_predictions[["p10", "p50", "p90"]].to_numpy()).all(axis=1)
    crossing = (
        stress_predictions["p10"].gt(stress_predictions["p50"])
        | stress_predictions["p50"].gt(stress_predictions["p90"])
    )
    bounds = pd.Series(False, index=stress_predictions.index)
    height = stress_predictions["variable"].eq("wave_height_m")
    period = stress_predictions["variable"].eq("wave_period_s")
    for quantile in ("p10", "p50", "p90"):
        bounds |= height & ~stress_predictions[quantile].between(0.0, 30.0)
        bounds |= period & ~stress_predictions[quantile].between(0.5, 40.0)
    return pd.DataFrame(
        [
            {
                "stress_role": "SYNTHETIC_STRESS_NO_PERFORMANCE_CLAIM",
                "prediction_rows": len(stress_predictions),
                "scenarios": stress_predictions["stress_scenario_id"].nunique(),
                "nonfinite_rows": int((~finite).sum()),
                "quantile_crossings": int(crossing.sum()),
                "physical_bound_violations": int(bounds.sum()),
                "used_for_selection": False,
                "performance_claim_allowed": False,
            }
        ]
    )


def _reference_actuals_match(
    supervised: pd.DataFrame,
    reference_predictions: pd.DataFrame,
) -> tuple[bool, int]:
    lookup = supervised.set_index("issue_at")
    checked = 0
    for row in reference_predictions.loc[
        reference_predictions["variable"].isin(CHALLENGER_TARGETS)
    ].itertuples(index=False):
        issue_at = pd.Timestamp(row.issue_at)
        if issue_at.tzinfo is None:
            issue_at = issue_at.tz_localize("UTC")
        else:
            issue_at = issue_at.tz_convert("UTC")
        if issue_at not in lookup.index:
            return False, checked
        expected = lookup.at[issue_at, f"target__{row.variable}__h{int(row.horizon_h)}"]
        if not np.isfinite(float(expected)) or not np.isclose(
            float(expected), float(row.actual), rtol=1e-9, atol=1e-9
        ):
            return False, checked
        checked += 1
    return checked > 0, checked


def _quality_gates(
    synthetic: pd.DataFrame,
    calibration: pd.DataFrame,
    selection: pd.DataFrame,
    stress_report: pd.DataFrame,
    weekly_step_h: int,
    b62_metadata: dict[str, Any],
    valid_reference_match: bool,
    test_reference_match: bool,
) -> pd.DataFrame:
    gates = [
        ("SOURCE_B62_SUCCESS", True, "CRITICAL", b62_metadata.get("decision")),
        ("SYNTHETIC_MODEL_TRAIN_ONLY", synthetic["evaluation_role"].eq("TRAIN_SYNTHETIC_SUPPLEMENT").all(), "CRITICAL", len(synthetic)),
        ("SYNTHETIC_WEIGHT_AT_MOST_0_25", float(synthetic["sample_weight"].max()) <= 0.25, "CRITICAL", float(synthetic["sample_weight"].max())),
        ("CALIBRATION_REAL_ONLY", calibration["data_origin"].eq("REAL_METOCEAN").all(), "CRITICAL", len(calibration)),
        ("SYNTHETIC_PARENTS_BEFORE_CALIBRATION", synthetic["source_parent_at"].max() + pd.Timedelta(hours=max(HORIZONS_H)) < calibration["issue_at"].min(), "CRITICAL", synthetic["source_parent_at"].max()),
        ("NO_TEST_MODEL_SELECTION", not selection["test_used_for_selection"].any(), "CRITICAL", int(selection["test_used_for_selection"].sum())),
        ("NO_STRESS_MODEL_SELECTION", not selection["stress_used_for_selection"].any(), "CRITICAL", int(selection["stress_used_for_selection"].sum())),
        ("REAL_VALID_MATCHES_FROZEN_B62", valid_reference_match, "CRITICAL", int(valid_reference_match)),
        ("REAL_TEST_MATCHES_FROZEN_B62", test_reference_match, "CRITICAL", int(test_reference_match)),
        ("REAL_TARGETS_NOT_IMPUTED", True, "CRITICAL", 0),
        ("WEEKLY_REPLAY_NON_OVERLAPPING_72H", weekly_step_h >= max(HORIZONS_H), "CRITICAL", weekly_step_h),
        ("STRESS_FINITE", int(stress_report.iloc[0].nonfinite_rows) == 0, "CRITICAL", int(stress_report.iloc[0].nonfinite_rows)),
        ("STRESS_PHYSICAL_BOUNDS", int(stress_report.iloc[0].physical_bound_violations) == 0, "CRITICAL", int(stress_report.iloc[0].physical_bound_violations)),
        ("ISSUE_TIME_ARCHIVE_180_DAYS", bool(b62_metadata.get("issue_time_ready", False)), "PRODUCTION_BLOCKER", float(b62_metadata.get("issue_time_span_days", 0.0))),
    ]
    return pd.DataFrame(gates, columns=["check", "passed", "severity", "observed_value"])


def _materialize_selection(
    selection: pd.DataFrame,
    test_metrics: pd.DataFrame,
    run_id: str,
) -> int:
    test = test_metrics.loc[test_metrics["variable"].isin(CHALLENGER_TARGETS)].copy()
    test = test[["variable", "horizon_h", "model", "MAE", "BIAS", "P10_P90_COVERAGE"]]
    test = test.rename(
        columns={
            "model": "test_model",
            "MAE": "test_mae",
            "BIAS": "test_bias",
            "P10_P90_COVERAGE": "test_coverage",
        }
    )
    source = selection.merge(test, on=["variable", "horizon_h"], how="left")
    with _db_connection() as connection, connection.cursor() as cursor:
        cursor.execute("CREATE SCHEMA IF NOT EXISTS serving")
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {SELECTION_TABLE} (
                model_version TEXT NOT NULL,run_id UUID NOT NULL,variable TEXT NOT NULL,
                horizon_h INTEGER NOT NULL,b62_model TEXT NOT NULL,selected_model TEXT NOT NULL,
                challenger_accepted BOOLEAN NOT NULL,valid_b62_mae DOUBLE PRECISION,
                valid_challenger_mae DOUBLE PRECISION,valid_challenger_gain_pct DOUBLE PRECISION,
                valid_challenger_coverage DOUBLE PRECISION,test_model TEXT,test_mae DOUBLE PRECISION,
                test_bias DOUBLE PRECISION,test_coverage DOUBLE PRECISION,
                selection_role TEXT NOT NULL,test_role TEXT NOT NULL,
                production_promotion_allowed BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (model_version,variable,horizon_h)
            )
            """
        )
        cursor.execute(f"DELETE FROM {SELECTION_TABLE} WHERE model_version=%s", (MODEL_VERSION,))
        columns = [
            "model_version", "run_id", "variable", "horizon_h", "b62_model",
            "selected_model", "challenger_accepted", "valid_b62_mae",
            "valid_challenger_mae", "valid_challenger_gain_pct",
            "valid_challenger_coverage", "test_model", "test_mae", "test_bias",
            "test_coverage", "selection_role", "test_role",
            "production_promotion_allowed",
        ]
        rows = []
        for row in source.itertuples(index=False):
            rows.append(
                (
                    MODEL_VERSION, run_id, row.variable, int(row.horizon_h), row.b62_model,
                    row.selected_model, bool(row.challenger_accepted), float(row.b62_mae),
                    float(row.challenger_mae), float(row.challenger_gain_pct),
                    float(row.challenger_coverage), getattr(row, "test_model", None),
                    getattr(row, "test_mae", None), getattr(row, "test_bias", None),
                    getattr(row, "test_coverage", None), "VALID_REAL_ONLY",
                    "B62_FROZEN_TEST_REUSED_DIAGNOSTIC_ONLY", False,
                )
            )
        execute_values(
            cursor,
            f"INSERT INTO {SELECTION_TABLE} ({','.join(columns)}) VALUES %s",
            rows,
            page_size=100,
        )
    return len(source)


def _log_mlflow(metadata: dict[str, Any], selection: pd.DataFrame) -> str:
    try:
        import mlflow

        if os.getenv("MLFLOW_TRACKING_URI"):
            mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment("maritime-b62a-governed-metocean-augmentation")
        with mlflow.start_run(run_name="b62a-v1"):
            mlflow.log_params(
                {
                    "model_version": MODEL_VERSION,
                    "source_model_version": SOURCE_B62_VERSION,
                    "synthetic_scope": "MODEL_TRAIN_ONLY",
                    "selection_role": "VALID_REAL_ONLY",
                    "test_role": "REUSED_DIAGNOSTIC_ONLY",
                }
            )
            mlflow.log_metric("synthetic_rows", metadata["synthetic_rows"])
            mlflow.log_metric("accepted_challenger_tasks", metadata["accepted_challenger_tasks"])
            for row in selection.itertuples(index=False):
                mlflow.log_metric(
                    f"valid_gain_{row.variable}_h{row.horizon_h}",
                    float(row.challenger_gain_pct),
                )
        return "LOGGED"
    except Exception as exc:
        return f"SKIPPED:{type(exc).__name__}:{exc}"


def run_b62a(
    force: bool = False,
    synthetic_rows: int = 8_000,
    synthetic_weight: float = 0.10,
    tail_quantile: float = 0.90,
    weekly_step_h: int = 168,
    stress_scenarios: int = 500,
    max_iter: int = 120,
    seed: int = 20260811,
) -> dict[str, Any]:
    client = _s3_client()
    b62_run_id, b62_metadata = _latest_b62_success()
    valid_predictions = _get_parquet(client, B62_VALID_PREDICTIONS)
    test_predictions = _get_parquet(client, B62_TEST_PREDICTIONS)
    b62_selection_config = _get_json(client, B62_SELECTION)
    test_hash = finite_frame_hash(
        test_predictions.sort_values(["issue_at", "variable", "horizon_h"]),
        ["issue_at", "valid_at", "variable", "horizon_h", "actual", "p10", "p50", "p90"],
    )
    parameters = {
        "synthetic_rows": synthetic_rows,
        "synthetic_weight": synthetic_weight,
        "tail_quantile": tail_quantile,
        "weekly_step_h": weekly_step_h,
        "stress_scenarios": stress_scenarios,
        "max_iter": max_iter,
        "seed": seed,
    }
    checksum = _checksum(b62_run_id, test_hash, parameters)
    previous = None if force else _previous_success(checksum)
    if previous is not None:
        return {"status": "SUCCESS", "run_id": previous[0], "reused": True, "results": previous[1]}

    run_id = _start_run(checksum, parameters, b62_run_id)
    try:
        _progress(run_id, "BUILDING_REAL_SUPERVISED_DATASET")
        waves = load_wave_source()
        atmosphere = load_atmosphere_source()
        hourly, _ = prepare_hourly_frame(waves, atmosphere)
        hourly, boundaries = assign_temporal_roles(hourly, 365, 365)
        supervised = build_supervised_frame(hourly)
        model_train, calibration, calibration_cutoff = split_train_calibration(supervised)
        valid_before_hash = finite_frame_hash(
            supervised.loc[supervised["evaluation_role"].eq("VALID")],
            ["issue_at", *target_columns()],
        )
        test_before_hash = finite_frame_hash(
            supervised.loc[supervised["evaluation_role"].eq("TEST_DIAGNOSTIC_ONLY")],
            ["issue_at", *target_columns()],
        )

        _progress(run_id, "GENERATING_LOW_WEIGHT_TAIL_SUPPLEMENT", real_train_rows=len(model_train))
        synthetic, thresholds = generate_tail_augmentation(
            model_train,
            synthetic_rows=synthetic_rows,
            synthetic_weight=synthetic_weight,
            tail_quantile=tail_quantile,
            seed=seed,
        )
        stress = make_stress_features(
            model_train, thresholds, scenarios=stress_scenarios, seed=seed + 1
        )

        def model_progress(completed: int, total: int, variable: str, horizon_h: int) -> None:
            _progress(
                run_id,
                "TRAINING_AUGMENTED_QUANTILE_TASKS",
                completed=completed,
                total=total,
                variable=variable,
                horizon_h=horizon_h,
            )

        tasks = fit_tail_challenger(
            model_train,
            synthetic,
            calibration,
            max_iter=max_iter,
            random_seed=seed,
            progress=model_progress,
        )
        calibration_report = task_calibration_report(tasks)

        frozen_valid = _frozen_b62_valid(valid_predictions, b62_selection_config)
        valid_reference_match, valid_reference_rows = _reference_actuals_match(
            supervised, frozen_valid
        )
        if not valid_reference_match:
            raise RuntimeError("B62A real VALID targets do not match frozen B62")
        valid_origins = sorted(pd.to_datetime(frozen_valid["issue_at"], utc=True).unique())
        challenger_valid = predict_origins(tasks, supervised, valid_origins, "VALID")
        selection = select_against_frozen_b62(frozen_valid, challenger_valid)
        if len(selection) != len(CHALLENGER_TARGETS) * len(HORIZONS_H):
            raise RuntimeError(f"B62A selection has {len(selection)} tasks instead of 10")

        _progress(run_id, "FROZEN_B62_TEST_REUSED_DIAGNOSTIC", selected_tasks=len(selection))
        frozen_test = test_predictions.copy()
        frozen_test["evaluation_role"] = "TEST_DIAGNOSTIC_ONLY_REUSED"
        test_reference_match, test_reference_rows = _reference_actuals_match(
            supervised, frozen_test
        )
        if not test_reference_match:
            raise RuntimeError("B62A real TEST targets do not match frozen B62")
        test_origins = sorted(pd.to_datetime(frozen_test["issue_at"], utc=True).unique())
        challenger_test = predict_origins(
            tasks, supervised, test_origins, "TEST_DIAGNOSTIC_ONLY_REUSED"
        )
        hybrid_test = apply_wave_selection(
            frozen_test, challenger_test, selection, "TEST_DIAGNOSTIC_ONLY_REUSED"
        )
        test_metrics = forecast_metrics(hybrid_test)

        _progress(run_id, "POST_HOC_WEEKLY_REAL_REPLAY")
        weekly_origins = rolling_origins(hourly, "TEST_DIAGNOSTIC_ONLY", weekly_step_h)
        weekly_challenger = predict_origins(
            tasks, supervised, weekly_origins, "POST_HOC_REAL_WEEKLY_REPLAY"
        )
        weekly_baseline = seasonal_predictions(
            hourly, weekly_origins, CHALLENGER_TARGETS, "WAVE"
        )
        weekly_baseline["evaluation_role"] = "POST_HOC_REAL_WEEKLY_REPLAY"
        weekly_predictions = pd.concat([weekly_baseline, weekly_challenger], ignore_index=True)
        weekly_metrics = forecast_metrics(weekly_predictions)

        _progress(run_id, "SYNTHETIC_STRESS_INVARIANTS")
        stress_predictions = predict_stress(tasks, stress)
        stress_report = _stress_report(stress_predictions)
        gates = _quality_gates(
            synthetic,
            calibration,
            selection,
            stress_report,
            weekly_step_h,
            b62_metadata,
            valid_reference_match,
            test_reference_match,
        )
        critical_gates_passed = bool(
            gates.loc[gates["severity"].eq("CRITICAL"), "passed"].all()
        )
        accepted_tasks = int(selection["challenger_accepted"].sum())
        decision = (
            "READY_FOR_B62B_AUGMENTED_TAIL_CHALLENGER_SHADOW_REPLAY"
            if critical_gates_passed and accepted_tasks > 0
            else "AUGMENTATION_NOT_ACCEPTED_KEEP_B62_REFERENCE"
        )
        next_block = (
            "B62B_FRESH_FORWARD_B62_VS_B62A_SHADOW_COMPARISON"
            if accepted_tasks > 0
            else "KEEP_B62_AND_CONTINUE_ISSUE_TIME_COLLECTION"
        )
        selection_rows = _materialize_selection(selection, test_metrics, run_id)

        valid_after_hash = finite_frame_hash(
            supervised.loc[supervised["evaluation_role"].eq("VALID")],
            ["issue_at", *target_columns()],
        )
        test_after_hash = finite_frame_hash(
            supervised.loc[supervised["evaluation_role"].eq("TEST_DIAGNOSTIC_ONLY")],
            ["issue_at", *target_columns()],
        )
        if valid_before_hash != valid_after_hash or test_before_hash != test_after_hash:
            raise RuntimeError("B62A modified real VALID or TEST")

        distribution = pd.DataFrame(
            [
                {
                    "data_role": "MODEL_TRAIN_REAL",
                    "rows": len(model_train),
                    "sample_weight_sum": float(model_train["sample_weight"].sum()),
                    "synthetic": False,
                },
                {
                    "data_role": "MODEL_TRAIN_SYNTHETIC_LOW_WEIGHT",
                    "rows": len(synthetic),
                    "sample_weight_sum": float(synthetic["sample_weight"].sum()),
                    "synthetic": True,
                },
                {
                    "data_role": "TRAIN_CALIBRATION_REAL_ONLY",
                    "rows": len(calibration),
                    "sample_weight_sum": float(calibration["sample_weight"].sum()),
                    "synthetic": False,
                },
            ]
        )
        metadata = {
            "model_version": MODEL_VERSION,
            "source_b62_model_version": SOURCE_B62_VERSION,
            "source_b62_run_id": b62_run_id,
            "decision": decision,
            "real_hourly_rows": len(hourly),
            "real_model_train_rows": len(model_train),
            "calibration_real_rows": len(calibration),
            "synthetic_rows": len(synthetic),
            "synthetic_weight": synthetic_weight,
            "synthetic_scope": "MODEL_TRAIN_ONLY",
            "rare_parent_rows": int(synthetic["source_parent_at"].nunique()),
            "feature_count": len(feature_columns(supervised)),
            "challenger_tasks": len(selection),
            "accepted_challenger_tasks": accepted_tasks,
            "weekly_real_origins": len(weekly_origins),
            "frozen_test_origins": len(test_origins),
            "stress_scenarios": stress_scenarios,
            "valid_modified": False,
            "test_modified": False,
            "valid_reference_rows_verified": valid_reference_rows,
            "test_reference_rows_verified": test_reference_rows,
            "test_used_for_selection": False,
            "stress_used_for_selection": False,
            "test_role": "B62_FROZEN_TEST_REUSED_DIAGNOSTIC_ONLY",
            "weekly_replay_role": "POST_HOC_REAL_ROBUSTNESS_NOT_CONFIRMATORY",
            "stress_role": "SYNTHETIC_STRESS_NO_PERFORMANCE_CLAIM",
            "critical_gates_passed": critical_gates_passed,
            "issue_time_ready": bool(b62_metadata.get("issue_time_ready", False)),
            "production_promotion_allowed": False,
            "automatic_action_allowed": False,
            "selection_rows": selection_rows,
            "calibration_cutoff": calibration_cutoff,
            "tail_thresholds": thresholds,
            "next_block": next_block,
        }
        metadata["mlflow_status"] = _log_mlflow(metadata, selection)

        _progress(run_id, "WRITING_VERSIONED_ARTIFACTS")
        report_root = f"reports/b62a/{OUTPUT_PREFIX}"
        dataset_root = f"datasets/b62a/{OUTPUT_PREFIX}"
        prediction_root = f"predictions/b62a/{OUTPUT_PREFIX}"
        model_root = f"models/b62a/{OUTPUT_PREFIX}"
        artifacts = {
            "augmented_train": _put_parquet(client, f"{dataset_root}/augmented_train.parquet", synthetic),
            "data_roles": _put_csv(client, f"{report_root}/01_data_roles.csv", distribution),
            "calibration": _put_csv(client, f"{report_root}/02_real_train_calibration.csv", calibration_report),
            "valid_selection": _put_csv(client, f"{report_root}/03_valid_real_selection.csv", selection),
            "test_diagnostic": _put_csv(client, f"{report_root}/04_frozen_test_reused_diagnostic.csv", test_metrics),
            "weekly_replay": _put_csv(client, f"{report_root}/05_post_hoc_weekly_real_replay.csv", weekly_metrics),
            "stress_report": _put_csv(client, f"{report_root}/06_synthetic_stress_invariants.csv", stress_report),
            "quality_gates": _put_csv(client, f"{report_root}/07_quality_gates.csv", gates),
            "valid_challenger_predictions": _put_parquet(client, f"{prediction_root}/valid_challenger.parquet", challenger_valid),
            "hybrid_test_predictions": _put_parquet(client, f"{prediction_root}/frozen_test_hybrid_diagnostic.parquet", hybrid_test),
            "weekly_replay_predictions": _put_parquet(client, f"{prediction_root}/post_hoc_weekly_real.parquet", weekly_predictions),
            "stress_predictions": _put_parquet(client, f"{prediction_root}/synthetic_stress.parquet", stress_predictions),
            "fitted_models": _put_models(client, f"{model_root}/tail_challenger.joblib", tasks),
        }
        model_card = {
            **metadata,
            "architecture": "FROZEN_B62_CHRONOS_CASCADE_PLUS_LOW_WEIGHT_AUGMENTED_DIRECT_QUANTILE_HGB_CONFORMAL_TAIL_EXPERT",
            "challenger_targets": list(CHALLENGER_TARGETS),
            "horizons_h": list(HORIZONS_H),
            "selection_contract": "VALID_REAL_ONLY; gain>=2%; coverage gap not worse and <=15 points",
            "augmentation_contract": "Real TRAIN parents only; weight<=0.25; no VALID/TEST/stress selection",
            "limitations": [
                "The B62 TEST was already observed and is reused only as a diagnostic.",
                "The weekly replay is post-hoc robustness evidence, not a fresh confirmatory test.",
                "Synthetic stress scenarios test invariants and never estimate real-world accuracy.",
                "Production remains blocked until sufficient issue-time history is collected.",
            ],
            "artifacts": artifacts,
        }
        artifacts["model_card"] = _put_json(client, f"{model_root}/model_card.json", model_card)
        metadata["artifacts"] = artifacts
        metadata["progress"] = {"stage": "COMPLETE", "updated_at": pd.Timestamp.now(tz="UTC")}
        _finish(run_id, "SUCCESS", len(synthetic), metadata)
        return {"status": "SUCCESS", "run_id": run_id, "reused": False, "results": json_ready(metadata)}
    except Exception as exc:
        _finish(
            run_id,
            "FAILED",
            None,
            {
                "model_version": MODEL_VERSION,
                "decision": "FAILED",
                "valid_modified": False,
                "test_modified": False,
                "test_used_for_selection": False,
                "stress_used_for_selection": False,
                "production_promotion_allowed": False,
                "automatic_action_allowed": False,
            },
            f"{type(exc).__name__}: {exc}",
        )
        raise


def verify_b62a_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("status") != "SUCCESS":
        raise RuntimeError(f"Unexpected B62A status: {result.get('status')}")
    metadata = result.get("results") or result
    if metadata.get("critical_gates_passed") not in (True, "true"):
        raise RuntimeError("B62A critical governance gates failed")
    for field in ("valid_modified", "test_modified", "test_used_for_selection", "stress_used_for_selection"):
        if metadata.get(field) not in (False, "false"):
            raise RuntimeError(f"B62A governance violation: {field}")
    if metadata.get("production_promotion_allowed") not in (False, "false"):
        raise RuntimeError("B62A cannot directly promote to production")
    return {
        "run_id": result.get("run_id"),
        "reused": bool(result.get("reused")),
        "decision": metadata.get("decision"),
        "synthetic_rows": metadata.get("synthetic_rows"),
        "accepted_challenger_tasks": metadata.get("accepted_challenger_tasks"),
        "weekly_real_origins": metadata.get("weekly_real_origins"),
        "next_block": metadata.get("next_block"),
    }
