from __future__ import annotations

from prefect import flow, get_run_logger, task

from prefect_flows.b61d_v11_job import run_b61d_v11, verify_b61d_v11_result


@task(name="b61dv11-fit-decode-and-benchmark-anchored-hsmm", retries=0)
def execute(
    force: bool,
    hsmm_weights: str,
    gt3_budgets: str,
    gt6_budgets: str,
    bucket_hours: int,
) -> dict:
    return run_b61d_v11(
        force=force,
        hsmm_weights=hsmm_weights,
        gt3_budgets=gt3_budgets,
        gt6_budgets=gt6_budgets,
        bucket_hours=bucket_hours,
    )


@task(name="b61dv11-verify-anchored-governance-contract")
def verify(result: dict) -> dict:
    return verify_b61d_v11_result(result)


@flow(
    name="b61d-v11-anchored-hsmm",
    description=(
        "Pre-breach anchored contextual explicit-duration HSMM with an exact "
        "B61C weight-zero control and governed shadow comparison."
    ),
    log_prints=True,
    timeout_seconds=10_800,
)
def b61d_v11_anchored_hsmm_flow(
    force: bool = False,
    hsmm_weights: str = "0.1,0.2,0.3,0.4",
    gt3_budgets: str = "1,2,3",
    gt6_budgets: str = "0.5,1",
    bucket_hours: int = 6,
) -> dict:
    logger = get_run_logger()
    logger.info(
        "Starting B61D-v1.1: weights=%s GT3=%s GT6=%s bucket=%sh",
        hsmm_weights, gt3_budgets, gt6_budgets, bucket_hours,
    )
    result = execute(
        force, hsmm_weights, gt3_budgets, gt6_budgets, bucket_hours
    )
    verified = verify(result)
    logger.info(
        "B61D-v1.1 decision=%s candidate=%s accepted=%s next=%s",
        verified["decision"], verified["selected_candidate_id"],
        verified["challenger_accepted"], verified["next_block"],
    )
    return verified


if __name__ == "__main__":
    b61d_v11_anchored_hsmm_flow()
