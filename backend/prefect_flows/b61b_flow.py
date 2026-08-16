from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61b_job import run_b61b_modeling, verify_b61b_result


@task(
    name="b61b-train-multitask-temporal-survival-moe",
    retries=0,
    timeout_seconds=21_600,
    tags={"maritime", "multitask", "survival", "gru", "mixture-of-experts"},
)
def train_task(
    force: bool = False,
    sequence_max_steps: int = 400,
    catboost_iterations: int = 350,
    max_train_rows: int = 80_000,
) -> dict:
    return run_b61b_modeling(
        force=force,
        sequence_max_steps=sequence_max_steps,
        catboost_iterations=catboost_iterations,
        max_train_rows=max_train_rows,
    )


@task(name="b61b-verify-model-governance-contract", tags={"quality-gate", "anti-leakage"})
def verify_task(result: dict) -> dict:
    return verify_b61b_result(result)


@flow(
    name="b61b-multitask-temporal-survival-moe",
    description=(
        "Train governed port-call delay, remaining-duration and discrete-survival "
        "experts, select a contextual CatBoost/GRU mixture on VALID_SELECT, "
        "calibrate it on VALID_CALIBRATE and reserve TEST for diagnostics."
    ),
    log_prints=True,
    timeout_seconds=22_200,
)
def b61b_multitask_temporal_survival_moe_flow(
    force: bool = False,
    sequence_max_steps: int = 400,
    catboost_iterations: int = 350,
    max_train_rows: int = 80_000,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61B with sequence_steps=%s catboost_iterations=%s max_train_rows=%s",
        sequence_max_steps,
        catboost_iterations,
        max_train_rows,
    )
    result = train_task(
        force=force,
        sequence_max_steps=sequence_max_steps,
        catboost_iterations=catboost_iterations,
        max_train_rows=max_train_rows,
    )
    verified = verify_task(result)
    logger.info(
        "B61B decision=%s serving_rows=%s next=%s",
        verified["decision"],
        verified["serving_rows"],
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61b_multitask_temporal_survival_moe_flow()
