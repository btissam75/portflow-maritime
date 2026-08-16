from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b59a_job import run_b59a_data_audit, verify_b59a_result


@task(
    name="b59a-audit-dynamic-port-call-data",
    retries=1,
    retry_delay_seconds=15,
    timeout_seconds=1_800,
    tags={"port-call", "data-quality", "point-in-time", "read-only"},
)
def audit_data_task(force: bool = False) -> dict:
    return run_b59a_data_audit(force=force)


@task(name="b59a-verify-data-contract", tags={"quality-gate", "read-only"})
def verify_task(result: dict) -> dict:
    return verify_b59a_result(result)


@flow(
    name="b59a-dynamic-port-call-data-audit",
    description=(
        "Read-only audit of port-call target quality, dynamic landmark yield, "
        "point-in-time availability and temporal split safety."
    ),
    log_prints=True,
    timeout_seconds=2_100,
)
def b59a_dynamic_port_call_data_audit_flow(force: bool = False) -> dict:
    logger = get_run_logger()
    logger.info("Starting B59A in read-only source mode")
    result = audit_data_task(force=force)
    verified = verify_task(result)
    logger.info(
        "B59A decision=%s next=%s",
        verified.get("decision"),
        verified.get("next_block"),
    )
    return verified


if __name__ == "__main__":
    b59a_dynamic_port_call_data_audit_flow()
