from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b62a_job import run_b62a, verify_b62a_result


@task(name="b62a-build-governed-metocean-tail-challenger", timeout_seconds=43_200)
def build_task(
    force: bool,
    synthetic_rows: int,
    synthetic_weight: float,
    tail_quantile: float,
    weekly_step_h: int,
    stress_scenarios: int,
    max_iter: int,
    seed: int,
) -> dict:
    return run_b62a(
        force=force,
        synthetic_rows=synthetic_rows,
        synthetic_weight=synthetic_weight,
        tail_quantile=tail_quantile,
        weekly_step_h=weekly_step_h,
        stress_scenarios=stress_scenarios,
        max_iter=max_iter,
        seed=seed,
    )


@task(name="b62a-verify-scientific-governance-contract")
def verify_task(result: dict) -> dict:
    return verify_b62a_result(result)


@flow(
    name="b62a-governed-metocean-augmentation",
    description=(
        "Build a real-TRAIN-parented, low-weight metocean tail supplement, fit a "
        "direct quantile wave challenger, calibrate on real TRAIN, select on real "
        "VALID and keep frozen TEST, post-hoc weekly replay and synthetic stress separate."
    ),
    log_prints=True,
    timeout_seconds=43_200,
)
def b62a_governed_metocean_augmentation_flow(
    force: bool = False,
    synthetic_rows: int = 8_000,
    synthetic_weight: float = 0.10,
    tail_quantile: float = 0.90,
    weekly_step_h: int = 168,
    stress_scenarios: int = 500,
    max_iter: int = 120,
    seed: int = 20260811,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B62A: synthetic=%s weight=%s tail_q=%s weekly=%sh stress=%s",
        synthetic_rows,
        synthetic_weight,
        tail_quantile,
        weekly_step_h,
        stress_scenarios,
    )
    result = build_task(
        force,
        synthetic_rows,
        synthetic_weight,
        tail_quantile,
        weekly_step_h,
        stress_scenarios,
        max_iter,
        seed,
    )
    verified = verify_task(result)
    logger.info(
        "B62A decision=%s accepted=%s weekly_real_origins=%s next=%s",
        verified["decision"],
        verified["accepted_challenger_tasks"],
        verified["weekly_real_origins"],
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b62a_governed_metocean_augmentation_flow()
