from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61e_core import verify_b61e_result
from prefect_flows.b61e_job import run_b61e


@task(name="b61e-build-capacity-aware-temporal-ranking", retries=0)
def execute(
    force: bool,
    score_names: str,
    top_ks: str,
    bucket_hours: int,
    bootstrap_iterations: int,
) -> dict:
    return run_b61e(
        force=force,
        score_names=score_names,
        top_ks=top_ks,
        bucket_hours=bucket_hours,
        bootstrap_iterations=bootstrap_iterations,
    )


@task(name="b61e-verify-ranking-governance-contract")
def verify(result: dict) -> dict:
    return verify_b61e_result(result)


@flow(
    name="b61e-capacity-aware-temporal-ranking",
    description=(
        "Rank all active port calls for GT3 breach risk within 24h. Score selection "
        "uses VALID_SELECT, capacity calibration uses VALID_CALIBRATE, and TEST is diagnostic."
    ),
    log_prints=True,
    timeout_seconds=7_200,
)
def b61e_capacity_aware_temporal_ranking_flow(
    force: bool = False,
    score_names: str = "HAZARD_24H,P_GT3,TEMPORAL_HAZARD_MOE",
    top_ks: str = "1,2",
    bucket_hours: int = 6,
    bootstrap_iterations: int = 500,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61E: scores=%s top_k=%s bucket=%sh bootstrap=%s",
        score_names, top_ks, bucket_hours, bootstrap_iterations,
    )
    result = execute(
        force, score_names, top_ks, bucket_hours, bootstrap_iterations
    )
    verified = verify(result)
    logger.info(
        "B61E decision=%s policy=%s score=%s top_k=%s contracts=%s next=%s",
        verified["decision"], verified["selected_candidate_id"],
        verified["selected_score"], verified["selected_top_k"],
        verified["contracts_passed"], verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61e_capacity_aware_temporal_ranking_flow()
