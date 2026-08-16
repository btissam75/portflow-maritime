from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61d_v131_core import verify_b61d_v131_result
from prefect_flows.b61d_v131_job import run_b61d_v131


@task(name="b61dv131-recalibrate-separate-stage-contracts", retries=0)
def execute(
    force: bool,
    early_modes: str,
    early_min_scores: str,
    early_top_ks: str,
    critical_top_ks: str,
    hold_windows: str,
    bucket_hours: int,
    bootstrap_iterations: int,
) -> dict:
    return run_b61d_v131(
        force=force,
        early_modes=early_modes,
        early_min_scores=early_min_scores,
        early_top_ks=early_top_ks,
        critical_top_ks=critical_top_ks,
        hold_windows=hold_windows,
        bucket_hours=bucket_hours,
        bootstrap_iterations=bootstrap_iterations,
    )


@task(name="b61dv131-verify-early-only-and-critical-contracts")
def verify(result: dict) -> dict:
    return verify_b61d_v131_result(result)


@flow(
    name="b61d-v131-contract-recalibration",
    description=(
        "Recalibration-only audit of EARLY_WARNING and CRITICAL_ACTION as "
        "separate contracts replicated on both validation periods."
    ),
    log_prints=True,
    timeout_seconds=7_200,
)
def b61d_v131_contract_recalibration_flow(
    force: bool = False,
    early_modes: str = "PRESSURE_CONGESTED,TRANSITION_AWARE,NON_FLUID",
    early_min_scores: str = "0.05,0.10,0.15,0.20",
    early_top_ks: str = "1,2",
    critical_top_ks: str = "1,2",
    hold_windows: str = "0,1",
    bucket_hours: int = 6,
    bootstrap_iterations: int = 500,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61D-v1.3.1 recalibration-only: modes=%s scores=%s "
        "early_k=%s critical_k=%s holds=%s bootstrap=%s",
        early_modes, early_min_scores, early_top_ks, critical_top_ks,
        hold_windows, bootstrap_iterations,
    )
    result = execute(
        force, early_modes, early_min_scores, early_top_ks, critical_top_ks,
        hold_windows, bucket_hours, bootstrap_iterations,
    )
    verified = verify(result)
    logger.info(
        "B61D-v1.3.1 decision=%s early=%s critical=%s candidate=%s next=%s",
        verified["decision"], verified["early_stage_validated"],
        verified["critical_stage_validated"], verified["selected_candidate_id"],
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61d_v131_contract_recalibration_flow()
