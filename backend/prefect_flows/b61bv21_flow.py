from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61bv21_job import (
    run_b61bv21_recalibration,
    verify_b61bv21_result,
)


@task(
    name="b61bv21-recalibrate-frozen-rare-risk-experts",
    retries=0,
    timeout_seconds=7_200,
    tags={"maritime", "recalibration", "platt", "anti-leakage"},
)
def recalibrate_task(force: bool = False, bootstrap_replicates: int = 500) -> dict:
    return run_b61bv21_recalibration(
        force=force,
        bootstrap_replicates=bootstrap_replicates,
    )


@task(
    name="b61bv21-verify-recalibration-and-test-disclosure",
    tags={"quality-gate", "governance", "anti-leakage"},
)
def verify_task(result: dict) -> dict:
    return verify_b61bv21_result(result)


@flow(
    name="b61b-v21-maritime-recalibration-only",
    description=(
        "Reuse immutable B61B-v2 experts, fit rank-preserving Platt maps for "
        "GT3/GT6 on VALID_CALIBRATE, correct the eligible port-call bootstrap, "
        "and disclose the previously inspected TEST as non-confirmatory."
    ),
    log_prints=True,
    timeout_seconds=7_500,
)
def b61bv21_maritime_recalibration_only_flow(
    force: bool = False,
    bootstrap_replicates: int = 500,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61B-v2.1 recalibration-only: no model fit; bootstrap=%s",
        bootstrap_replicates,
    )
    result = recalibrate_task(
        force=force,
        bootstrap_replicates=bootstrap_replicates,
    )
    verified = verify_task(result)
    logger.info(
        "B61B-v2.1 decision=%s replay_allowed=%s next=%s",
        verified["decision"],
        verified.get("replay_allowed"),
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61bv21_maritime_recalibration_only_flow()
