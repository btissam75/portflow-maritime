from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b60b_job import run_b60b_benchmark, verify_b60b_result


@task(name="run-b60b-advanced-timeseries-benchmark", retries=0, timeout_seconds=28_800)
def benchmark_task(force: bool = False, sequence_max_steps: int = 250) -> dict:
    return run_b60b_benchmark(
        force=force,
        sequence_max_steps=sequence_max_steps,
    )


@task(name="verify-b60b-contract")
def verify_task(result: dict) -> dict:
    return verify_b60b_result(result)


@flow(name="b60b-advanced-timeseries-benchmark", log_prints=True)
def b60b_advanced_timeseries_benchmark_flow(
    force: bool = False,
    sequence_max_steps: int = 250,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B60B target-specific benchmark with max_steps=%s",
        sequence_max_steps,
    )
    result = benchmark_task(
        force=force,
        sequence_max_steps=sequence_max_steps,
    )
    verified = verify_task(result)
    logger.info("B60B decision: %s", verified["decision"])
    return verified


if __name__ == "__main__":
    b60b_advanced_timeseries_benchmark_flow()
