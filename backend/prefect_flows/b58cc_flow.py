from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b58cc_ablation import run_b58cc_weather_ablation


@task(
    name="b58cc-run-observed-masking-and-ablation",
    retries=0,
    timeout_seconds=14_400,
    tags={"weather", "model-research", "temporal-validation"},
    log_prints=True,
)
def run_ablation_task(force: bool = False) -> dict:
    return run_b58cc_weather_ablation(force=force)


@flow(
    name="b58cc-weather-feature-ablation",
    description=(
        "Compares fixed wave-forecast models across retrospective external "
        "weather feature tracks with purged temporal validation."
    ),
    log_prints=True,
    timeout_seconds=14_700,
)
def b58cc_weather_feature_ablation_flow(force: bool = False) -> dict:
    logger = get_run_logger()
    logger.info("Starting B58C-C temporal weather feature ablation")
    result = run_ablation_task(force=force)
    logger.info(
        "B58C-C completed with decision=%s",
        result.get("results", {}).get("decision"),
    )
    return result


if __name__ == "__main__":
    b58cc_weather_feature_ablation_flow()
