from __future__ import annotations

from typing import Any

from prefect import flow, get_run_logger, task

from prefect_flows.b58ca_job import run_missingness_audit, verify_audit_result


@task(
    name="audit-weather-missingness",
    description="Read-only B58C-A weather missingness and repairability audit.",
    retries=1,
    retry_delay_seconds=20,
    timeout_seconds=1_800,
    tags={"data-quality", "weather", "read-only"},
)
def audit_weather_missingness_task(
    output_bucket: str,
    output_prefix: str,
    force: bool,
) -> dict[str, Any]:
    return run_missingness_audit(
        output_bucket=output_bucket,
        output_prefix=output_prefix,
        force=force,
    )


@task(
    name="verify-b58ca-safety-and-artifacts",
    retries=1,
    retry_delay_seconds=10,
    tags={"quality-gate", "read-only"},
)
def verify_b58ca_task(result: dict[str, Any]) -> dict[str, Any]:
    return verify_audit_result(result)


@flow(
    name="b58ca-weather-missingness-audit",
    description=(
        "Prefect-native audit that distinguishes sparse gaps, structural absence, "
        "and complete weather variables before any imputation or synthesis."
    ),
    log_prints=True,
    timeout_seconds=2_400,
)
def b58ca_weather_missingness_audit_flow(
    output_bucket: str = "gold-maritime",
    output_prefix: str = "version=1",
    force: bool = False,
) -> dict[str, Any]:
    logger = get_run_logger()
    logger.info("Starting B58C-A in strict read-only mode")
    result = audit_weather_missingness_task(
        output_bucket=output_bucket,
        output_prefix=output_prefix,
        force=force,
    )
    verified = verify_b58ca_task(result)
    logger.info(
        "B58C-A decision=%s next_block=%s",
        verified.get("decision"),
        verified.get("next_block"),
    )
    return verified


if __name__ == "__main__":
    b58ca_weather_missingness_audit_flow()
