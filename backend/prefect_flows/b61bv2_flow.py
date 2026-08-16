from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61bv2_job import run_b61bv2_modeling, verify_b61bv2_result


@task(
    name="b61bv2-train-governed-maritime-rare-event-hybrid",
    retries=0,
    timeout_seconds=28_800,
    tags={
        "maritime",
        "multitask",
        "survival",
        "rare-event",
        "mixture-of-experts",
        "conformal",
    },
)
def train_task(
    force: bool = False,
    sequence_max_steps: int = 300,
    catboost_iterations: int = 320,
    max_train_rows: int = 80_000,
    bootstrap_replicates: int = 300,
) -> dict:
    return run_b61bv2_modeling(
        force=force,
        sequence_max_steps=sequence_max_steps,
        catboost_iterations=catboost_iterations,
        max_train_rows=max_train_rows,
        bootstrap_replicates=bootstrap_replicates,
    )


@task(name="b61bv2-verify-governance-and-test-contract", tags={"quality-gate", "anti-leakage"})
def verify_task(result: dict) -> dict:
    return verify_b61bv2_result(result)


@flow(
    name="b61b-v2-maritime-rare-event-hybrid",
    description=(
        "Benchmark real reference, cost-sensitive, governed EVT-tail and real-only "
        "sequence experts. Select each task on VALID_SELECT, calibrate on "
        "VALID_CALIBRATE, then open unchanged real TEST once for diagnostics."
    ),
    log_prints=True,
    timeout_seconds=29_400,
)
def b61bv2_maritime_rare_event_hybrid_flow(
    force: bool = False,
    sequence_max_steps: int = 300,
    catboost_iterations: int = 320,
    max_train_rows: int = 80_000,
    bootstrap_replicates: int = 300,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61B-v2: sequence_steps=%s catboost_iterations=%s "
        "max_train_rows=%s bootstrap=%s",
        sequence_max_steps,
        catboost_iterations,
        max_train_rows,
        bootstrap_replicates,
    )
    result = train_task(
        force=force,
        sequence_max_steps=sequence_max_steps,
        catboost_iterations=catboost_iterations,
        max_train_rows=max_train_rows,
        bootstrap_replicates=bootstrap_replicates,
    )
    verified = verify_task(result)
    logger.info(
        "B61B-v2 decision=%s selected=%s synthetic_tasks=%s next=%s",
        verified["decision"],
        verified["selected_models"],
        verified["synthetic_selected_tasks"],
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61bv2_maritime_rare_event_hybrid_flow()
