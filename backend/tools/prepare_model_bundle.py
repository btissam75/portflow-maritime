from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_ROLES = {
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
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_catboost_contract(path: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    try:
        from catboost import CatBoost
    except ImportError as exc:
        raise RuntimeError("Install catboost==1.2.8 before preparing the bundle") from exc
    model = CatBoost()
    model.load_model(str(path))
    feature_names = list(model.feature_names_ or [])
    if not feature_names:
        raise RuntimeError(f"No feature names stored in checkpoint: {path}")
    cat_indices = list(model.get_cat_feature_indices())
    cat_names = [feature_names[index] for index in cat_indices]
    parameters = model.get_all_params()
    metadata = {
        "tree_count": int(model.tree_count_),
        "loss_function": parameters.get("loss_function"),
        "random_seed": parameters.get("random_seed"),
        "cat_feature_indices": cat_indices,
        "parameters": parameters,
    }
    return feature_names, cat_names, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a self-describing, checksummed PortFlow model bundle."
    )
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    artifact_specs = spec.get("artifacts", [])
    roles = {item.get("role") for item in artifact_specs}
    missing = sorted(REQUIRED_ROLES - roles)
    if missing:
        raise RuntimeError("Missing required roles: " + ", ".join(missing))
    if len(roles) != len(artifact_specs):
        raise RuntimeError("Artifact roles must be unique")

    output = args.output.resolve()
    model_dir = output / "models"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Output directory must be empty: {output}")
    model_dir.mkdir(parents=True, exist_ok=True)

    artifacts: list[dict[str, Any]] = []
    for item in artifact_specs:
        source = Path(item["source_path"]).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = model_dir / f"{item['role']}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        feature_names, cat_names, metadata = extract_catboost_contract(destination)
        artifacts.append(
            {
                "role": item["role"],
                "path": destination.relative_to(output).as_posix(),
                "sha256": sha256(destination),
                "format": "catboost",
                "task": item["task"],
                "prediction_kind": item["prediction_kind"],
                "output_transform": item.get("output_transform", "identity"),
                "feature_names": feature_names,
                "cat_feature_names": item.get("cat_feature_names", cat_names),
                "metadata": metadata,
            }
        )
        print(f"{item['role']}: {destination.name} ({len(feature_names)} features)")

    manifest = {
        "schema_version": "PORTFLOW_MODEL_BUNDLE_V1",
        "bundle_id": spec["bundle_id"],
        "created_at": spec.get("created_at") or datetime.now(timezone.utc).isoformat(),
        "source_max_time": spec["source_max_time"],
        "promotion_status": spec.get("promotion_status", "VALIDATED"),
        "training_run": spec["training_run"],
        "git_commit": spec.get("git_commit"),
        "artifacts": artifacts,
        "formulas": {
            "B36B": "clip(0.9*B36B_MAE+0.1*B36B_RMSE,0,48)",
            "B48R": "clip(0.9*B36B+0.1*B48_META,0,48)",
        },
        "scientific_limits": spec.get("scientific_limits", []),
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Bundle ready: {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
