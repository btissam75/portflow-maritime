from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61a_job import run_b61a_governed_completion, verify_b61a_result


@task(
    name="b61a-build-governed-port-call-enrichment",
    retries=1,
    retry_delay_seconds=30,
    timeout_seconds=7_200,
    tags={"maritime", "governance", "feature-engineering", "anti-leakage"},
)
def build_governed_dataset_task(force: bool = False) -> dict:
    return run_b61a_governed_completion(force=force)


@task(name="b61a-verify-governed-contract", tags={"quality-gate", "anti-leakage"})
def verify_governed_dataset_task(result: dict) -> dict:
    return verify_b61a_result(result)


@flow(
    name="b61a-governed-data-completion",
    description=(
        "Enrich real B60C port-call landmarks with governed event, physical, "
        "operational and issue-time forecast features without generating targets "
        "or mixing synthetic scenarios into model evaluation."
    ),
    log_prints=True,
    timeout_seconds=7_800,
)
def b61a_governed_data_completion_flow(force: bool = False) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61A: governed completion; targets and real TEST remain immutable"
    )
    result = build_governed_dataset_task(force=force)
    verified = verify_governed_dataset_task(result)
    logger.info(
        "B61A decision=%s rows=%s issue_time_ready=%s next=%s",
        verified.get("decision"),
        verified.get("row_count"),
        verified.get("issue_time_history_ready"),
        verified.get("next_block"),
    )
    return verified


if __name__ == "__main__":
    b61a_governed_data_completion_flow()
