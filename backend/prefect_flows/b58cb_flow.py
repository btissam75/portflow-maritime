from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger, task

from prefect_flows.b58cb_external_weather import (
    run_b58cb_external_enrichment,
    verify_b58cb,
)


@task(
    name="collect-and-audit-external-weather",
    retries=1,
    retry_delay_seconds=60,
    timeout_seconds=7_200,
    tags={"external-data", "weather", "reanalysis"},
)
def collect_external_weather_task(
    force_download: bool,
    materialize_timescale: bool,
) -> dict[str, Any]:
    return run_b58cb_external_enrichment(
        force_download=force_download,
        materialize_timescale=materialize_timescale,
    )


@task(name="verify-b58cb-contract", tags={"quality-gate"})
def verify_external_weather_task(result: dict[str, Any]) -> dict[str, Any]:
    return verify_b58cb(result)


@flow(
    name="b58cb-external-weather-enrichment",
    description=(
        "Collects retrospective ERA5 atmosphere, ERA5-Ocean and archived "
        "visibility data with strict provenance and no Core mutation."
    ),
    log_prints=True,
    timeout_seconds=7_800,
)
def b58cb_external_weather_enrichment_flow(
    force_download: bool = False,
    materialize_timescale: bool = True,
) -> dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting B58C-B external weather enrichment")
    result = collect_external_weather_task(
        force_download=force_download,
        materialize_timescale=materialize_timescale,
    )
    verified = verify_external_weather_task(result)
    logger.info(
        "B58C-B decision=%s rows=%s next=%s",
        verified.get("decision"),
        verified.get("rows"),
        verified.get("next_block"),
    )
    return verified


if __name__ == "__main__":
    b58cb_external_weather_enrichment_flow()
