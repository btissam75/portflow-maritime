from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b60a_job import run_b60a_dataset_build, verify_b60a_result


@task(
    name="b60a-build-multitask-hourly-dataset",
    retries=1,
    retry_delay_seconds=30,
    timeout_seconds=3_600,
    tags={"maritime", "dataset", "time-series", "point-in-time"},
)
def build_dataset_task(force: bool = False) -> dict:
    return run_b60a_dataset_build(force=force)


@task(name="b60a-verify-dataset-contract", tags={"quality-gate", "anti-leakage"})
def verify_dataset_task(result: dict) -> dict:
    return verify_b60a_result(result)


@flow(
    name="b60a-maritime-multitask-hourly-dataset",
    description=(
        "Build a versioned hourly dataset for arrival-count, wave and temporal "
        "point-process benchmarks with task-specific temporal splits."
    ),
    log_prints=True,
    timeout_seconds=4_200,
)
def b60a_maritime_multitask_hourly_dataset_flow(force: bool = False) -> dict:
    logger = get_run_logger()
    logger.info("Starting B60A without synthetic targets or source mutation")
    result = build_dataset_task(force=force)
    verified = verify_dataset_task(result)
    logger.info(
        "B60A decision=%s rows=%s next=%s",
        verified.get("decision"),
        verified.get("row_count"),
        verified.get("next_block"),
    )
    return verified


if __name__ == "__main__":
    b60a_maritime_multitask_hourly_dataset_flow()
