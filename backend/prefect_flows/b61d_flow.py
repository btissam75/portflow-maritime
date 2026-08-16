from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61d_job import run_b61d, verify_b61d_result


@task(
    name="b61d-fit-decode-and-benchmark-contextual-hsmm",
    retries=0,
    timeout_seconds=10_800,
    tags={"maritime", "hsmm", "explicit-duration", "lead-time"},
)
def hsmm_task(
    force: bool = False,
    hsmm_weights: str = "0.25,0.5,0.75,1",
    gt3_budgets: str = "1,2,3",
    gt6_budgets: str = "0.5,1",
    bucket_hours: int = 6,
) -> dict:
    return run_b61d(
        force=force,
        hsmm_weights=hsmm_weights,
        gt3_budgets=gt3_budgets,
        gt6_budgets=gt6_budgets,
        bucket_hours=bucket_hours,
    )


@task(
    name="b61d-verify-hsmm-governance-and-challenger-contract",
    tags={"quality-gate", "anti-leakage", "shadow-only"},
)
def verify_task(result: dict) -> dict:
    return verify_b61d_result(result)


@flow(
    name="b61d-contextual-hsmm",
    description=(
        "Fit a chronological contextual explicit-duration HSMM, decode hidden "
        "maritime regimes, optimize lead-aware shadow policies on VALID, and "
        "retain B61C automatically when the challenger is not non-inferior."
    ),
    log_prints=True,
    timeout_seconds=11_100,
)
def b61d_contextual_hsmm_flow(
    force: bool = False,
    hsmm_weights: str = "0.25,0.5,0.75,1",
    gt3_budgets: str = "1,2,3",
    gt6_budgets: str = "0.5,1",
    bucket_hours: int = 6,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61D HSMM: weights=%s GT3=%s GT6=%s bucket=%sh",
        hsmm_weights,
        gt3_budgets,
        gt6_budgets,
        bucket_hours,
    )
    result = hsmm_task(
        force=force,
        hsmm_weights=hsmm_weights,
        gt3_budgets=gt3_budgets,
        gt6_budgets=gt6_budgets,
        bucket_hours=bucket_hours,
    )
    verified = verify_task(result)
    logger.info(
        "B61D decision=%s candidate=%s accepted=%s next=%s",
        verified["decision"],
        verified["selected_candidate_id"],
        verified["challenger_accepted"],
        verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61d_contextual_hsmm_flow()
