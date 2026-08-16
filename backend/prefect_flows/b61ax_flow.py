from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61ax_core import (
    DEFAULT_EXTERNAL_LICENSE,
    DEFAULT_EXTERNAL_NAME,
    DEFAULT_EXTERNAL_URL,
    RANDOM_SEED,
)
from prefect_flows.b61ax_job import run_b61ax_augmentation, verify_b61ax_result


@task(
    name="b61ax-build-governed-rare-tail-augmentation",
    retries=1,
    retry_delay_seconds=60,
    timeout_seconds=10_800,
    tags={"maritime", "external-data", "evt", "synthetic", "train-only"},
)
def build_task(
    force: bool = False,
    force_download: bool = False,
    external_url: str = DEFAULT_EXTERNAL_URL,
    external_source_name: str = DEFAULT_EXTERNAL_NAME,
    external_source_license: str = DEFAULT_EXTERNAL_LICENSE,
    synthetic_calls: int = 2_500,
    synthetic_call_weight: float = 0.20,
    max_delay_h: float = 240.0,
    max_download_mb: int = 600,
    seed: int = RANDOM_SEED,
) -> dict:
    return run_b61ax_augmentation(
        force=force,
        force_download=force_download,
        external_url=external_url,
        external_source_name=external_source_name,
        external_source_license=external_source_license,
        synthetic_calls=synthetic_calls,
        synthetic_call_weight=synthetic_call_weight,
        max_delay_h=max_delay_h,
        max_download_mb=max_download_mb,
        seed=seed,
    )


@task(name="b61ax-verify-governance-contract", tags={"quality-gate", "anti-leakage"})
def verify_task(result: dict) -> dict:
    return verify_b61ax_result(result)


@flow(
    name="b61ax-governed-rare-tail-augmentation",
    description=(
        "Use a traceable external real port-call source as an EVT tail prior, "
        "then generate low-weight counterfactual TRAIN-only landmarks from real "
        "local TRAIN trajectories without modifying B61A, VALID or TEST."
    ),
    log_prints=True,
    timeout_seconds=11_400,
)
def b61ax_governed_rare_tail_augmentation_flow(
    force: bool = False,
    force_download: bool = False,
    external_url: str = DEFAULT_EXTERNAL_URL,
    external_source_name: str = DEFAULT_EXTERNAL_NAME,
    external_source_license: str = DEFAULT_EXTERNAL_LICENSE,
    synthetic_calls: int = 2_500,
    synthetic_call_weight: float = 0.20,
    max_delay_h: float = 240.0,
    max_download_mb: int = 600,
    seed: int = RANDOM_SEED,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61A-X: external EVT prior, %s TRAIN-only calls, weight=%s",
        synthetic_calls,
        synthetic_call_weight,
    )
    result = build_task(
        force=force,
        force_download=force_download,
        external_url=external_url,
        external_source_name=external_source_name,
        external_source_license=external_source_license,
        synthetic_calls=synthetic_calls,
        synthetic_call_weight=synthetic_call_weight,
        max_delay_h=max_delay_h,
        max_download_mb=max_download_mb,
        seed=seed,
    )
    verified = verify_task(result)
    logger.info(
        "B61A-X decision=%s synthetic_rows=%s next=%s",
        verified["decision"],
        verified["synthetic_rows"],
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61ax_governed_rare_tail_augmentation_flow()
