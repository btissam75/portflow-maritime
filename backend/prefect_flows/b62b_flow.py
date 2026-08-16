from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b62b_job import run_b62b, verify_b62b_result


@task(name="b62b-build-vintage-forecast-shadow-validation", timeout_seconds=43_200)
def build_task(**parameters) -> dict:
    return run_b62b(**parameters)


@task(name="b62b-verify-vintage-governance-and-promotion-contract")
def verify_task(result: dict) -> dict:
    return verify_b62b_result(result)


@flow(
    name="b62b-vintage-forecast-shadow-validation",
    description=(
        "Backfill authentic fixed-lead weather forecasts, compare frozen B62 and "
        "B62A with a real-target weather-to-wave challenger, consume a frozen TEST "
        "once, and require fresh issue-time evidence before a limited manual pilot."
    ),
    log_prints=True,
    timeout_seconds=43_200,
)
def b62b_vintage_forecast_shadow_validation_flow(
    force: bool = False,
    force_download: bool = False,
    backfill_days: int = 900,
    valid_days: int = 180,
    test_days: int = 180,
    calibration_days: int = 90,
    chunk_days: int = 90,
    min_fresh_origins: int = 60,
    min_fresh_days: int = 30,
    bootstrap_iterations: int = 500,
    min_gain_pct: float = 5.0,
    max_iter: int = 160,
    seed: int = 20260811,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B62B: archive=%sd VALID=%sd TEST=%sd fresh=%s/%sd bootstrap=%s",
        backfill_days,
        valid_days,
        test_days,
        min_fresh_origins,
        min_fresh_days,
        bootstrap_iterations,
    )
    result = build_task(
        force=force,
        force_download=force_download,
        backfill_days=backfill_days,
        valid_days=valid_days,
        test_days=test_days,
        calibration_days=calibration_days,
        chunk_days=chunk_days,
        min_fresh_origins=min_fresh_origins,
        min_fresh_days=min_fresh_days,
        bootstrap_iterations=bootstrap_iterations,
        min_gain_pct=min_gain_pct,
        max_iter=max_iter,
        seed=seed,
    )
    verified = verify_task(result)
    logger.info(
        "B62B decision=%s selected=%s archive_confirmed=%s fresh_confirmed=%s next=%s",
        verified["decision"],
        verified["selected_model"],
        verified["archive_confirmed"],
        verified["fresh_confirmed"],
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b62b_vintage_forecast_shadow_validation_flow()
