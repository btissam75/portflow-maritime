from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b60ch_job import (
    run_b60ch_event_intelligence,
    verify_b60ch_result,
)


@task(
    name="b60ch-build-historical-event-intelligence",
    retries=1,
    retry_delay_seconds=30,
    timeout_seconds=3_600,
    tags={"maritime", "events", "calendar", "point-in-time"},
)
def build_event_intelligence_task(force: bool = False) -> dict:
    return run_b60ch_event_intelligence(force=force)


@task(name="b60ch-verify-scientific-contract", tags={"quality-gate", "anti-leakage"})
def verify_event_intelligence_task(result: dict) -> dict:
    return verify_b60ch_result(result)


@flow(
    name="b60ch-historical-event-intelligence",
    description=(
        "Build a sourced 2020-2025 Morocco/Tanger Med event registry, point-in-time "
        "B60C landmark features, matched association reports, and non-trainable "
        "latent-period investigation candidates."
    ),
    log_prints=True,
    timeout_seconds=4_200,
)
def b60ch_historical_event_intelligence_flow(force: bool = False) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B60C-H: sourced event context; retrospective events stay research-only"
    )
    result = build_event_intelligence_task(force=force)
    verified = verify_event_intelligence_task(result)
    logger.info(
        "B60C-H decision=%s events=%s context_rows=%s next=%s",
        verified.get("decision"),
        verified.get("registry_events"),
        verified.get("context_rows"),
        verified.get("next_block"),
    )
    return verified


if __name__ == "__main__":
    b60ch_historical_event_intelligence_flow()
