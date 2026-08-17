from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator


router = APIRouter(prefix="/api/v1/model-serving", tags=["PortFlow model serving"])

REQUIRED_UNIT_ROLES = (
    "B36B_MAE",
    "B36B_RMSE",
    "B36C_GE12",
    "B36C_GE24",
    "B36C_GE36",
    "B44_Q50",
    "B44_Q80",
    "B44_Q95",
    "B48_GATE",
    "B48_META",
)

ComputedScalar = str | int | float | bool | None


class ArtifactContract(BaseModel):
    role: str = Field(min_length=2)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    format: Literal["catboost", "joblib", "parquet", "json"]
    task: Literal["regression", "classification", "parameters"]
    prediction_kind: Literal[
        "scalar", "probability_positive", "probability_vector", "not_predictive"
    ] = "scalar"
    output_transform: Literal["identity", "add_b36b"] = "identity"
    feature_names: list[str] = Field(default_factory=list)
    cat_feature_names: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_features(self) -> "ArtifactContract":
        if self.format == "catboost" and not self.feature_names:
            raise ValueError(f"{self.role}: feature_names must be frozen for CatBoost")
        unknown = sorted(set(self.cat_feature_names) - set(self.feature_names))
        if unknown:
            raise ValueError(f"{self.role}: unknown categorical features: {unknown}")
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError(f"{self.role}: duplicate feature names")
        return self


class ModelBundleManifest(BaseModel):
    schema_version: Literal["PORTFLOW_MODEL_BUNDLE_V1"]
    bundle_id: str = Field(min_length=3)
    created_at: datetime
    source_max_time: datetime
    promotion_status: Literal["DEVELOPMENT", "VALIDATED", "PROMOTED", "REJECTED"]
    training_run: str
    git_commit: str | None = None
    artifacts: list[ArtifactContract]
    formulas: dict[str, str]
    scientific_limits: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_roles(self) -> "ModelBundleManifest":
        roles = [artifact.role for artifact in self.artifacts]
        if len(roles) != len(set(roles)):
            raise ValueError("The model bundle contains duplicate artifact roles")
        required_formulas = {
            "B36B": "clip(0.9*B36B_MAE+0.1*B36B_RMSE,0,48)",
            "B48R": "clip(0.9*B36B+0.1*B48_META,0,48)",
        }
        for key, expected in required_formulas.items():
            if self.formulas.get(key) != expected:
                raise ValueError(f"Formula {key} must be frozen as: {expected}")
        return self


class ModelServingStatus(BaseModel):
    configured: bool
    ready: bool
    live_enabled: bool
    live_eligible: bool
    state: str
    bundle_id: str | None = None
    promotion_status: str | None = None
    source_max_time: datetime | None = None
    manifest_path: str | None = None
    validated_artifacts: int = 0
    required_unit_roles: int = len(REQUIRED_UNIT_ROLES)
    issues: list[str] = Field(default_factory=list)


class UnitInferenceRequest(BaseModel):
    unit_id: str = Field(min_length=1, max_length=160)
    snapshot_at: datetime
    source_observed_at: datetime
    features: dict[str, ComputedScalar]

    @model_validator(mode="after")
    def validate_payload_size(self) -> "UnitInferenceRequest":
        if len(self.features) > 2_000:
            raise ValueError("features exceeds the 2,000-field safety limit")
        invalid_names = [name for name in self.features if not name or len(name) > 180]
        if invalid_names:
            raise ValueError("feature names must contain between 1 and 180 characters")
        return self


class UnitInferenceResponse(BaseModel):
    unit_id: str
    calculated_at: datetime
    bundle_id: str
    serving_mode: Literal["LIVE", "HISTORICAL_REPLAY_ONLY"]
    live_eligible: bool
    source_age_hours: float
    b36b_mae_h: float
    b36b_rmse_h: float
    b36b_h: float
    p_ge12: float
    p_ge24: float
    p_ge36: float
    b44_p50_h: float
    b44_p80_h: float
    b44_p95_h: float
    b48_meta_h: float
    b48r_h: float
    gate_weights: list[float]
    quantiles_reordered: bool
    model_roles: list[str]
    warnings: list[str]


class ModelServingError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedArtifact:
    contract: ArtifactContract
    path: Path
    model: Any


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool_env(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _bundle_directory() -> Path | None:
    raw = os.getenv("PORTFLOW_MODEL_BUNDLE_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else None


class ModelRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._signature: tuple[str, int] | None = None
        self._manifest: ModelBundleManifest | None = None
        self._loaded: dict[str, LoadedArtifact] = {}
        self._issues: list[str] = []

    @property
    def manifest(self) -> ModelBundleManifest | None:
        self.refresh()
        return self._manifest

    def _manifest_path(self) -> Path | None:
        directory = _bundle_directory()
        if directory is None:
            return None
        filename = os.getenv("PORTFLOW_MODEL_MANIFEST", "manifest.json")
        return (directory / filename).resolve()

    def refresh(self, force: bool = False) -> None:
        with self._lock:
            manifest_path = self._manifest_path()
            if manifest_path is None:
                self._signature = None
                self._manifest = None
                self._loaded = {}
                self._issues = ["PORTFLOW_MODEL_BUNDLE_DIR is not configured"]
                return
            try:
                signature = (str(manifest_path), manifest_path.stat().st_mtime_ns)
            except FileNotFoundError:
                self._signature = None
                self._manifest = None
                self._loaded = {}
                self._issues = [f"Model manifest not found: {manifest_path}"]
                return
            if not force and signature == self._signature:
                return

            self._signature = signature
            self._manifest = None
            self._loaded = {}
            self._issues = []
            try:
                manifest = ModelBundleManifest.model_validate_json(
                    manifest_path.read_text(encoding="utf-8")
                )
                bundle_dir = manifest_path.parent.resolve()
                artifact_paths: dict[str, Path] = {}
                for artifact in manifest.artifacts:
                    artifact_path = (bundle_dir / artifact.path).resolve()
                    if bundle_dir != artifact_path and bundle_dir not in artifact_path.parents:
                        raise ModelServingError(
                            f"{artifact.role}: artifact path escapes the bundle directory"
                        )
                    if not artifact_path.is_file():
                        raise ModelServingError(
                            f"{artifact.role}: artifact not found: {artifact.path}"
                        )
                    observed_hash = _sha256(artifact_path)
                    if observed_hash.lower() != artifact.sha256.lower():
                        raise ModelServingError(
                            f"{artifact.role}: SHA-256 mismatch ({observed_hash})"
                        )
                    artifact_paths[artifact.role] = artifact_path

                unit_roles = {artifact.role for artifact in manifest.artifacts}
                missing_roles = sorted(set(REQUIRED_UNIT_ROLES) - unit_roles)
                if missing_roles:
                    raise ModelServingError(
                        f"Missing unit inference roles: {', '.join(missing_roles)}"
                    )

                loaded: dict[str, LoadedArtifact] = {}
                for artifact in manifest.artifacts:
                    model: Any = None
                    if artifact.format == "catboost":
                        try:
                            from catboost import CatBoost
                        except ImportError as exc:
                            raise ModelServingError(
                                "catboost is required to load the registered model bundle"
                            ) from exc
                        model = CatBoost()
                        model.load_model(str(artifact_paths[artifact.role]))
                        model_features = list(model.feature_names_ or [])
                        if model_features and model_features != artifact.feature_names:
                            raise ModelServingError(
                                f"{artifact.role}: checkpoint feature order differs from manifest"
                            )
                    loaded[artifact.role] = LoadedArtifact(
                        contract=artifact,
                        path=artifact_paths[artifact.role],
                        model=model,
                    )
                self._manifest = manifest
                self._loaded = loaded
            except Exception as exc:  # status endpoint must remain available
                self._manifest = None
                self._loaded = {}
                self._issues = [str(exc)]

    def status(self) -> ModelServingStatus:
        self.refresh()
        path = self._manifest_path()
        manifest = self._manifest
        configured = path is not None
        ready = manifest is not None and not self._issues
        live_enabled = _bool_env("PORTFLOW_MODEL_LIVE_ENABLED")
        live_eligible = bool(
            ready and live_enabled and manifest and manifest.promotion_status == "PROMOTED"
        )
        if not configured:
            state = "NOT_CONFIGURED"
        elif not ready:
            state = "INVALID_BUNDLE"
        elif live_eligible:
            state = "READY_FOR_GOVERNED_LIVE_INFERENCE"
        else:
            state = "READY_FOR_HISTORICAL_REPLAY_ONLY"
        return ModelServingStatus(
            configured=configured,
            ready=ready,
            live_enabled=live_enabled,
            live_eligible=live_eligible,
            state=state,
            bundle_id=manifest.bundle_id if manifest else None,
            promotion_status=manifest.promotion_status if manifest else None,
            source_max_time=manifest.source_max_time if manifest else None,
            manifest_path=path.name if path else None,
            validated_artifacts=len(self._loaded),
            issues=list(self._issues),
        )

    def _predict(self, role: str, features: dict[str, ComputedScalar]) -> float | list[float]:
        artifact = self._loaded.get(role)
        if artifact is None or artifact.model is None:
            raise ModelServingError(f"Model role is not loaded: {role}")
        contract = artifact.contract
        missing = [name for name in contract.feature_names if name not in features]
        if missing:
            preview = ", ".join(missing[:12])
            suffix = "..." if len(missing) > 12 else ""
            raise ModelServingError(f"{role}: missing features: {preview}{suffix}")

        values: list[Any] = []
        categorical_indices: list[int] = []
        categorical = set(contract.cat_feature_names)
        for index, name in enumerate(contract.feature_names):
            value = features[name]
            if name in categorical:
                categorical_indices.append(index)
                values.append("__MISSING__" if value is None else str(value))
                continue
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ModelServingError(f"{role}: {name} must be a finite numeric value")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ModelServingError(f"{role}: {name} must be finite")
            values.append(numeric)

        from catboost import Pool

        pool = Pool(
            data=[values],
            feature_names=contract.feature_names,
            cat_features=categorical_indices,
        )
        if contract.prediction_kind in {"probability_positive", "probability_vector"}:
            raw = artifact.model.predict(pool, prediction_type="Probability")[0]
            vector = [float(value) for value in raw]
            if contract.prediction_kind == "probability_vector":
                return vector
            return vector[-1]
        raw = artifact.model.predict(pool)[0]
        return float(raw)

    def predict_unit(self, request: UnitInferenceRequest) -> UnitInferenceResponse:
        self.refresh()
        if self._manifest is None or self._issues:
            raise ModelServingError("Model bundle is not ready: " + "; ".join(self._issues))

        context = dict(request.features)
        b36_mae = float(self._predict("B36B_MAE", context))
        b36_rmse = float(self._predict("B36B_RMSE", context))
        b36b = min(48.0, max(0.0, 0.9 * b36_mae + 0.1 * b36_rmse))
        context.update(
            {
                "PRED_B36B_MAE": b36_mae,
                "PRED_B36B_RMSE": b36_rmse,
                "PRED_B36B_FINAL": b36b,
                "PRED_B36B_BASE": b36b,
            }
        )

        p_ge12 = float(self._predict("B36C_GE12", context))
        p_ge24 = float(self._predict("B36C_GE24", context))
        p_ge36 = float(self._predict("B36C_GE36", context))
        context.update({"P_GE12": p_ge12, "P_GE24": p_ge24, "P_GE36": p_ge36})

        raw_residuals: list[float] = []
        raw_quantiles: list[float] = []
        for role in ("B44_Q50", "B44_Q80", "B44_Q95"):
            artifact = self._loaded[role]
            value = float(self._predict(role, context))
            raw_residuals.append(value)
            if artifact.contract.output_transform == "add_b36b":
                value += b36b
            raw_quantiles.append(min(48.0, max(0.0, value)))
        ordered_quantiles = sorted(raw_quantiles)
        quantiles_reordered = any(
            abs(left - right) > 1e-9
            for left, right in zip(raw_quantiles, ordered_quantiles, strict=True)
        )
        q50, q80, q95 = ordered_quantiles
        context.update(
            {
                "RESID_Q50_RAW": raw_residuals[0],
                "RESID_Q80_RAW": raw_residuals[1],
                "RESID_Q95_RAW": raw_residuals[2],
                "ETA_Q50_RAW": raw_quantiles[0],
                "ETA_Q80_RAW": raw_quantiles[1],
                "ETA_Q95_RAW": raw_quantiles[2],
                "ETA_Q50_RAW_MONO": q50,
                "ETA_Q80_RAW_MONO": q80,
                "ETA_Q95_RAW_MONO": q95,
                "ETA_P50_B44": q50,
                "ETA_P80_B44": q80,
                "ETA_P95_B44": q95,
                "UNCERTAINTY_WIDTH_P95_P50": q95 - q50,
                "UNCERTAINTY_WIDTH_P80_P50": q80 - q50,
                "B44_P50_MINUS_B36B": q50 - b36b,
                "B44_P80_MINUS_B36B": q80 - b36b,
                "B44_P95_MINUS_B36B": q95 - b36b,
                "RISK_MAX": max(p_ge12, p_ge24, p_ge36),
                "RISK_LONG_SUM": p_ge24 + p_ge36,
                "IS_HIGH_RISK_GE24": float(p_ge24 >= 0.5),
                "IS_HIGH_RISK_GE36": float(p_ge36 >= 0.5),
            }
        )
        gate_raw = self._predict("B48_GATE", context)
        gate_weights = gate_raw if isinstance(gate_raw, list) else [float(gate_raw)]
        for index, value in enumerate(gate_weights):
            context[f"GATE_WEIGHT_{index}"] = value
        gate_aliases = (
            "GATE_W_EXP_B36B",
            "GATE_W_EXP_B44_P50",
            "GATE_W_EXP_B44_P80",
            "GATE_W_EXP_B44_P95",
        )
        for alias, value in zip(gate_aliases, gate_weights):
            context[alias] = value
        if len(gate_weights) == 4:
            context["PRED_B48_GATE_BLEND"] = sum(
                weight * expert
                for weight, expert in zip(gate_weights, (b36b, q50, q80, q95), strict=True)
            )
        b48_meta = min(48.0, max(0.0, float(self._predict("B48_META", context))))
        b48r = min(48.0, max(0.0, 0.9 * b36b + 0.1 * b48_meta))

        snapshot_at = _utc(request.snapshot_at)
        source_at = _utc(request.source_observed_at)
        source_age_hours = max(0.0, (snapshot_at - source_at).total_seconds() / 3600.0)
        freshness_limit = float(os.getenv("PORTFLOW_SOURCE_FRESHNESS_LIMIT_HOURS", "2"))
        status = self.status()
        live_eligible = status.live_eligible and source_age_hours <= freshness_limit
        warnings: list[str] = []
        if source_at > snapshot_at:
            warnings.append("source_observed_at is after snapshot_at; live serving is refused")
            live_eligible = False
        if source_age_hours > freshness_limit:
            warnings.append(
                f"source age {source_age_hours:.2f} h exceeds the {freshness_limit:.2f} h limit"
            )
        if quantiles_reordered:
            warnings.append("B44 quantiles crossed and were monotonically reordered")

        return UnitInferenceResponse(
            unit_id=request.unit_id,
            calculated_at=datetime.now(timezone.utc),
            bundle_id=self._manifest.bundle_id,
            serving_mode="LIVE" if live_eligible else "HISTORICAL_REPLAY_ONLY",
            live_eligible=live_eligible,
            source_age_hours=source_age_hours,
            b36b_mae_h=b36_mae,
            b36b_rmse_h=b36_rmse,
            b36b_h=b36b,
            p_ge12=p_ge12,
            p_ge24=p_ge24,
            p_ge36=p_ge36,
            b44_p50_h=q50,
            b44_p80_h=q80,
            b44_p95_h=q95,
            b48_meta_h=b48_meta,
            b48r_h=b48r,
            gate_weights=gate_weights,
            quantiles_reordered=quantiles_reordered,
            model_roles=list(REQUIRED_UNIT_ROLES),
            warnings=warnings,
        )


registry = ModelRegistry()


@router.get("/status", response_model=ModelServingStatus)
def model_serving_status() -> ModelServingStatus:
    return registry.status()


@router.post("/reload", response_model=ModelServingStatus)
def reload_model_bundle() -> ModelServingStatus:
    registry.refresh(force=True)
    return registry.status()


@router.post("/unit/remaining-time", response_model=UnitInferenceResponse)
def predict_unit_remaining_time(payload: UnitInferenceRequest) -> UnitInferenceResponse:
    try:
        return registry.predict_unit(payload)
    except ModelServingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def public_model_status() -> dict[str, Any]:
    """Small dependency-safe status used by the Control Tower snapshot."""
    return registry.status().model_dump(mode="json")
