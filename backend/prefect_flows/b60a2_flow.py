from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b60a2_job import (
    run_b60a2_predictive_signal_audit,
    verify_b60a2_result,
)


@task(name="audit-predictive-signal", retries=0, timeout_seconds=14_400)
def audit_task(force: bool = False) -> dict:
    return run_b60a2_predictive_signal_audit(force=force)


@task(name="verify-predictive-signal-contract")
def verify_task(result: dict) -> dict:
    return verify_b60a2_result(result)


@flow(
    name="b60a2-predictive-signal-audit",
    log_prints=True,
)
def b60a2_predictive_signal_audit_flow(force: bool = False) -> dict:
    logger = get_run_logger()
    logger.info("Starting leakage-safe B60A.2 predictive signal audit")
    result = audit_task(force=force)
    verified = verify_task(result)
    logger.info("B60A.2 decision: %s", verified["decision"])
    return verified


if __name__ == "__main__":
    b60a2_predictive_signal_audit_flow()
