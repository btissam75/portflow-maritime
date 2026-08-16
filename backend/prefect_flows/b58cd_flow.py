from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger, task

from prefect_flows.b58cd_collector import run_b58cd_issue_time_collection


@task(
    name="collect-live-issue-time-weather-forecast",
    retries=2,
    retry_delay_seconds=90,
    timeout_seconds=900,
    tags={"weather", "marine", "issue-time", "external-data"},
    log_prints=True,
)
def collect_issue_time_forecast_task() -> dict[str, Any]:
    return run_b58cd_issue_time_collection()


@flow(
    name="b58cd-issue-time-weather-forecast-collection",
    description=(
        "Archives live weather and marine forecast snapshots with real "
        "request, availability and valid timestamps."
    ),
    log_prints=True,
    timeout_seconds=1_200,
)
def b58cd_issue_time_weather_forecast_collection_flow() -> dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting B58C-D live issue-time weather collection")
    result = collect_issue_time_forecast_task()
    logger.info(
        "B58C-D decision=%s rows=%s next=%s",
        result.get("decision"),
        result.get("forecast_rows"),
        result.get("next_block"),
    )
    return result


if __name__ == "__main__":
    b58cd_issue_time_weather_forecast_collection_flow()
