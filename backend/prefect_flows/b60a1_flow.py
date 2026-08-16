from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b60a1_job import run_b60a1_feature_audit, verify_b60a1_result


@task(
    name="b60a1-audit-correlations-and-representations",
    retries=1,
    retry_delay_seconds=30,
    timeout_seconds=7_200,
    tags={"time-series", "correlation", "pca", "anti-leakage"},
)
def audit_task(force: bool = False) -> dict:
    return run_b60a1_feature_audit(force=force)


@task(name="b60a1-verify-representation-contract", tags={"quality-gate"})
def verify_task(result: dict) -> dict:
    return verify_b60a1_result(result)


@flow(
    name="b60a1-feature-representation-audit",
    description=(
        "TRAIN-only correlation, redundancy, temporal stability and semantic-block "
        "PCA audit for B60A arrival and wave feature sets."
    ),
    log_prints=True,
    timeout_seconds=7_800,
)
def b60a1_feature_representation_audit_flow(force: bool = False) -> dict:
    logger = get_run_logger()
    logger.info("Starting B60A.1; TEST remains diagnostic only")
    result = audit_task(force=force)
    verified = verify_task(result)
    logger.info(
        "B60A.1 decision=%s representations=%s next=%s",
        verified.get("decision"),
        verified.get("representation_count"),
        verified.get("next_block"),
    )
    return verified


if __name__ == "__main__":
    b60a1_feature_representation_audit_flow()
