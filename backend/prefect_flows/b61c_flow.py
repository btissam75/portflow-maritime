from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61c_job import run_b61c_replay, verify_b61c_result


@task(
    name="b61c-replay-temporal-hazards-and-select-dynamic-policy",
    retries=0,
    timeout_seconds=7_200,
    tags={"maritime", "historical-replay", "decision-policy", "temporal-hazard"},
)
def replay_task(
    force: bool = False,
    gt3_budgets: str = "1,2,3,5",
    gt6_budgets: str = "0.25,0.5,1",
    bucket_hours: int = 6,
) -> dict:
    return run_b61c_replay(
        force=force,
        gt3_budgets=gt3_budgets,
        gt6_budgets=gt6_budgets,
        bucket_hours=bucket_hours,
    )


@task(
    name="b61c-verify-shadow-decision-governance",
    tags={"quality-gate", "anti-leakage", "shadow-only"},
)
def verify_task(result: dict) -> dict:
    return verify_b61c_result(result)


@flow(
    name="b61c-historical-replay-shadow-decision-api",
    description=(
        "Replay frozen B61B-v2.1 temporal hazards, select a capacity-constrained "
        "dynamic policy on VALID only, materialize a hysteretic state machine, "
        "and serve read-only shadow decisions."
    ),
    log_prints=True,
    timeout_seconds=7_500,
)
def b61c_historical_replay_shadow_decision_flow(
    force: bool = False,
    gt3_budgets: str = "1,2,3,5",
    gt6_budgets: str = "0.25,0.5,1",
    bucket_hours: int = 6,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61C replay: GT3 budgets=%s GT6 budgets=%s bucket=%sh",
        gt3_budgets,
        gt6_budgets,
        bucket_hours,
    )
    result = replay_task(
        force=force,
        gt3_budgets=gt3_budgets,
        gt6_budgets=gt6_budgets,
        bucket_hours=bucket_hours,
    )
    verified = verify_task(result)
    logger.info(
        "B61C decision=%s policy=%s shadow=%s dynamic_alerts=%s",
        verified["decision"],
        verified["selected_policy_id"],
        verified["shadow_api_allowed"],
        verified.get("dynamic_alert_shadow_allowed"),
    )
    return verified


if __name__ == "__main__":
    b61c_historical_replay_shadow_decision_flow()
