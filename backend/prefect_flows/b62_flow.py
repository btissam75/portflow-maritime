from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b62_core import verify_b62_result
from prefect_flows.b62_job import run_b62


@task(name="b62-run-weather-wave-vessel-autogluon", retries=0)
def execute(
    force: bool,
    validation_days: int,
    test_days: int,
    backtest_step_h: int,
    preset: str,
) -> dict:
    return run_b62(
        force=force,
        validation_days=validation_days,
        test_days=test_days,
        backtest_step_h=backtest_step_h,
        preset=preset,
    )


@task(name="b62-verify-scientific-and-governance-contract")
def verify(result: dict) -> dict:
    return verify_b62_result(result)


@flow(
    name="b62-weather-wave-vessel-autogluon",
    description=(
        "Forecast weather with AutoGluon Chronos-2, feed those forecasts into a "
        "wave model, then combine metocean severity with the frozen B61E vessel "
        "watchlist. VALID selects models; TEST is diagnostic only."
    ),
    log_prints=True,
    timeout_seconds=43_200,
)
def b62_weather_wave_vessel_autogluon_flow(
    force: bool = False,
    validation_days: int = 365,
    test_days: int = 365,
    backtest_step_h: int = 168,
    preset: str = "chronos2_small",
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B62: preset=%s VALID=%sd TEST=%sd origins_step=%sh",
        preset,
        validation_days,
        test_days,
        backtest_step_h,
    )
    result = execute(
        force,
        validation_days,
        test_days,
        backtest_step_h,
        preset,
    )
    verified = verify(result)
    logger.info(
        "B62 decision=%s chronos_tasks=%s issue_time_ready=%s "
        "forecast_rows=%s impact_rows=%s next=%s",
        verified["decision"],
        verified["selected_chronos_tasks"],
        verified["issue_time_ready"],
        verified["serving_forecast_rows"],
        verified["serving_impact_rows"],
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b62_weather_wave_vessel_autogluon_flow()
