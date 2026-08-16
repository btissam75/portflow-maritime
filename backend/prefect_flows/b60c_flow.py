from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b60c_job import run_b60c_dataset_build, verify_b60c_result


@task(
    name="b60c-build-operational-port-call-landmarks",
    retries=1,
    retry_delay_seconds=30,
    timeout_seconds=7_200,
    tags={"maritime", "port-call", "landmarking", "point-in-time"},
)
def build_dataset_task(force: bool = False) -> dict:
    return run_b60c_dataset_build(force=force)


@task(name="b60c-verify-dataset-contract", tags={"quality-gate", "anti-leakage"})
def verify_dataset_task(result: dict) -> dict:
    return verify_b60c_result(result)


@flow(
    name="b60c-operational-port-call-dataset",
    description=(
        "Build real vessel port-call decision landmarks for early delay warning, "
        "remaining-time regression and survival analysis, with separate non-trainable "
        "counterfactual stress scenarios."
    ),
    log_prints=True,
    timeout_seconds=7_800,
)
def b60c_operational_port_call_dataset_flow(force: bool = False) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B60C: real vessel-call landmarks; synthetic scenarios stay separate"
    )
    result = build_dataset_task(force=force)
    verified = verify_dataset_task(result)
    logger.info(
        "B60C decision=%s calls=%s rows=%s next=%s",
        verified.get("decision"),
        verified.get("eligible_calls"),
        verified.get("row_count"),
        verified.get("next_block"),
    )
    return verified


if __name__ == "__main__":
    b60c_operational_port_call_dataset_flow()
