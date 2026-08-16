from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61d_v12_core import verify_b61d_v12_result
from prefect_flows.b61d_v12_job import run_b61d_v12


@task(name="b61dv12-replay-state-conditional-policy", retries=0)
def execute(
    force: bool,
    policy_modes: str,
    probability_thresholds: str,
    alert_budgets: str,
    hold_windows: str,
    bucket_hours: int,
    bootstrap_iterations: int,
) -> dict:
    return run_b61d_v12(
        force=force,
        policy_modes=policy_modes,
        probability_thresholds=probability_thresholds,
        alert_budgets=alert_budgets,
        hold_windows=hold_windows,
        bucket_hours=bucket_hours,
        bootstrap_iterations=bootstrap_iterations,
    )


@task(name="b61dv12-verify-selection-and-shadow-governance")
def verify(result: dict) -> dict:
    return verify_b61d_v12_result(result)


@flow(
    name="b61d-v12-state-conditional-policy",
    description=(
        "Frozen anchored-HSMM state policy with B61C secondary ranking, "
        "capacity, hysteresis, constrained VALID selection and clustered bootstrap."
    ),
    log_prints=True,
    timeout_seconds=7_200,
)
def b61d_v12_state_conditional_policy_flow(
    force: bool = False,
    policy_modes: str = (
        "STATE_STRICT,POSTERIOR_STRICT,CRITICAL_WITH_CONGESTED_BACKSTOP"
    ),
    probability_thresholds: str = "0.2,0.3,0.4,0.5",
    alert_budgets: str = "1,2,3",
    hold_windows: str = "0,1,2",
    bucket_hours: int = 6,
    bootstrap_iterations: int = 500,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61D-v1.2: modes=%s thresholds=%s budgets=%s holds=%s bucket=%sh bootstrap=%s",
        policy_modes, probability_thresholds, alert_budgets, hold_windows,
        bucket_hours, bootstrap_iterations,
    )
    result = execute(
        force, policy_modes, probability_thresholds, alert_budgets,
        hold_windows, bucket_hours, bootstrap_iterations,
    )
    verified = verify(result)
    logger.info(
        "B61D-v1.2 decision=%s candidate=%s constraints=%s next=%s",
        verified["decision"], verified["selected_candidate_id"],
        verified["constraints_passed"], verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61d_v12_state_conditional_policy_flow()
