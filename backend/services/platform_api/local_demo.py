from __future__ import annotations

import math
import os
from datetime import datetime, timedelta, timezone
from typing import Any


DEMO_POLICY_VERSION = "local-demo-capacity-shadow-v1"
DEMO_METOCEAN_VERSION = "local-demo-metocean-shadow-v1"
VESSELS = (
    ("PF-1001", "Atlas Horizon", "TC1", "Container", "CONTENEUR"),
    ("PF-1002", "Tangier Star", "TC2", "Container", "CONTENEUR"),
    ("PF-1003", "Mistral Bridge", "TC3", "Ro-Ro", "ROULIER"),
    ("PF-1004", "Ocean Cedar", "TC4", "Container", "CONTENEUR"),
    ("PF-1005", "Strait Pioneer", "TC1", "Ro-Ro", "ROULIER"),
    ("PF-1006", "Maghreb Trader", "TC2", "General cargo", "DIVERS"),
    ("PF-1007", "Blue Rif", "TC3", "Container", "CONTENEUR"),
    ("PF-1008", "Cap Spartel", "TC4", "Ro-Ro", "ROULIER"),
    ("PF-1009", "North Gate", "TC1", "Container", "CONTENEUR"),
    ("PF-1010", "Alboran Link", "TC2", "General cargo", "DIVERS"),
    ("PF-1011", "Med Express", "TC3", "Container", "CONTENEUR"),
    ("PF-1012", "Hercules Bay", "TC4", "Ro-Ro", "ROULIER"),
    ("PF-1013", "Port Cedar", "TC1", "Container", "CONTENEUR"),
    ("PF-1014", "Gibraltar Wave", "TC2", "Container", "CONTENEUR"),
    ("PF-1015", "Rif Connector", "TC3", "Ro-Ro", "ROULIER"),
    ("PF-1016", "Marhaba One", "TC4", "General cargo", "DIVERS"),
    ("PF-1017", "Atlantic Crown", "TC1", "Container", "CONTENEUR"),
    ("PF-1018", "Zenith Strait", "TC2", "Container", "CONTENEUR"),
)


def enabled() -> bool:
    return os.getenv("SMART_PORT_LOCAL_DEMO_MODE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _utc_hour() -> datetime:
    configured = os.getenv("SMART_PORT_LOCAL_DEMO_ANCHOR", "").strip()
    if configured:
        value = datetime.fromisoformat(configured.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)


def _six_hour_anchor() -> datetime:
    value = _utc_hour()
    return value.replace(hour=value.hour - value.hour % 6)


def capacity_snapshot_times() -> list[datetime]:
    anchor = _six_hour_anchor()
    return [anchor - timedelta(hours=6 * offset) for offset in range(20, -1, -1)]


def resolve_capacity_time(requested: datetime | None) -> datetime | None:
    times = capacity_snapshot_times()
    if requested is None:
        return times[-1]
    if requested.tzinfo is None:
        requested = requested.replace(tzinfo=timezone.utc)
    requested = requested.astimezone(timezone.utc)
    eligible = [item for item in times if item <= requested]
    return eligible[-1] if eligible else None


def capacity_decisions(resolved: datetime, role: str) -> list[dict[str, Any]]:
    phase = int(resolved.timestamp() // 21_600)
    rows: list[dict[str, Any]] = []
    for index, (call_id, vessel, terminal, vessel_type, cargo) in enumerate(VESSELS):
        wave = 0.5 + 0.5 * math.sin((phase + index * 1.7) / 3.2)
        trend = 0.10 * math.sin((phase + index) / 2.1)
        risk = max(0.04, min(0.96, 0.18 + 0.62 * wave + trend))
        p_delay = max(0.02, min(0.98, risk * 0.92 + 0.03))
        remaining_p50 = max(1.2, 6.0 + 29.0 * risk + (index % 4) * 1.3)
        spread = 2.8 + risk * 8.2
        rows.append(
            {
                "port_call_id": call_id,
                "vessel_name": vessel,
                "port_code": "MAPTM",
                "terminal_code": terminal,
                "vessel_type": vessel_type,
                "cargo_group": cargo,
                "landmark_at": resolved - timedelta(hours=1 + index % 8),
                "decision_at": resolved,
                "evaluation_role": role,
                "risk_score": round(risk, 6),
                "active_calls": len(VESSELS),
                "capacity": 6,
                "watchlist_selected": False,
                "action_tier": "MONITOR",
                "reason_code": "DEMO_TEMPORAL_RISK",
                "p_delay_gt3": round(p_delay, 6),
                "hazard_6h": round(max(0.01, p_delay * 0.32), 6),
                "hazard_12h": round(max(0.02, p_delay * 0.61), 6),
                "hazard_24h": round(max(0.03, p_delay * 0.88), 6),
                "remaining_p10_h": round(max(0.0, remaining_p50 - spread), 3),
                "remaining_p50_h": round(remaining_p50, 3),
                "remaining_p90_h": round(remaining_p50 + spread, 3),
                "hsmm_state": ("APPROACH", "WAITING", "BERTH_WINDOW")[index % 3],
                "hsmm_state_confidence": round(0.64 + 0.28 * (1.0 - risk / 2.0), 6),
                "production_claim_allowed": False,
                "automatic_action_allowed": False,
            }
        )
    rows.sort(key=lambda item: (-item["risk_score"], item["port_call_id"]))
    for rank, row in enumerate(rows, start=1):
        row["rank_in_window"] = rank
        row["watchlist_selected"] = rank <= row["capacity"]
        row["action_tier"] = (
            "REVIEW_NOW"
            if row["risk_score"] >= 0.65
            else "WATCH"
            if row["risk_score"] >= 0.40
            else "ROUTINE"
        )
        row["reason_code"] = (
            "DEMO_HIGH_DELAY_AND_CAPACITY"
            if row["watchlist_selected"]
            else "DEMO_TEMPORAL_RISK"
        )
    return rows


def capacity_timeline(port_call_id: str) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = []
    for snapshot_at in capacity_snapshot_times():
        timeline.extend(
            row
            for row in capacity_decisions(snapshot_at, "VALID_SELECT")
            if row["port_call_id"] == port_call_id
        )
    return timeline


def metocean_forecast(track: str) -> list[dict[str, Any]]:
    issue_at = _utc_hour()
    rows: list[dict[str, Any]] = []
    for horizon in range(1, 73):
        valid_at = issue_at + timedelta(hours=horizon)
        daily = 2.0 * math.pi * ((valid_at.hour - 6) / 24.0)
        synoptic = math.sin(horizon / 7.5)
        values = {
            "temperature_2m": 20.8 + 3.7 * math.sin(daily) + 0.7 * synoptic,
            "wave_height_m": 1.05 + 0.42 * math.sin(horizon / 6.2) + 0.16 * math.sin(horizon / 2.7),
            "wind_speed_ms": 7.2 + 2.1 * math.sin(horizon / 5.4) + 0.8 * math.cos(daily),
            "pressure_hpa": 1014.0 + 4.5 * math.cos(horizon / 13.0),
        }
        widths = {
            "temperature_2m": 0.8 + horizon * 0.018,
            "wave_height_m": 0.16 + horizon * 0.006,
            "wind_speed_ms": 0.9 + horizon * 0.025,
            "pressure_hpa": 1.4 + horizon * 0.025,
        }
        for variable, raw_value in values.items():
            p50 = max(0.0, raw_value) if variable != "temperature_2m" else raw_value
            width = widths[variable]
            rows.append(
                {
                    "track": track,
                    "issue_at": issue_at,
                    "valid_at": valid_at,
                    "horizon_h": horizon,
                    "variable": variable,
                    "p10": round(max(0.0, p50 - width), 4),
                    "p50": round(p50, 4),
                    "p90": round(p50 + width, 4),
                    "source_model": "LOCAL_DEMO_SEASONAL_SHADOW",
                    "uncertainty_status": "DEMO_SYNTHETIC_INTERVALS",
                    "operationally_available": False,
                    "production_claim_allowed": False,
                }
            )
    return rows


def vessel_impacts() -> list[dict[str, Any]]:
    issue_at = _utc_hour()
    forecast = metocean_forecast("ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT")
    wave_by_horizon = {
        int(row["horizon_h"]): float(row["p50"])
        for row in forecast
        if row["variable"] == "wave_height_m"
    }
    rows: list[dict[str, Any]] = []
    for index, (call_id, vessel, terminal, vessel_type, cargo) in enumerate(VESSELS[:12]):
        horizon = 3 + index * 3
        base = 0.24 + (index % 5) * 0.105
        severity = min(1.0, wave_by_horizon[horizon] / 2.2)
        exposure = min(1.0, 0.46 + 0.08 * (index % 6))
        score = min(0.98, 0.5 * base + 0.3 * severity + 0.2 * exposure)
        rows.append(
            {
                "port_call_id": call_id,
                "vessel_name": vessel,
                "port_code": "MAPTM",
                "terminal_code": terminal,
                "vessel_type": vessel_type,
                "cargo_group": cargo,
                "source_decision_at": issue_at - timedelta(hours=1),
                "forecast_issue_at": issue_at,
                "valid_at": issue_at + timedelta(hours=horizon),
                "horizon_h": horizon,
                "base_temporal_risk": round(base, 6),
                "metocean_severity": round(severity, 6),
                "vessel_exposure": round(exposure, 6),
                "combined_priority_score": round(score, 6),
                "metocean_tier": "HIGH" if severity >= 0.7 else "MODERATE",
                "priority_tier": "REVIEW" if score >= 0.6 else "MONITOR",
                "forecast_track": "ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT",
                "score_semantics": "LOCAL_DEMO_HUMAN_REVIEW_ONLY",
                "automatic_action_allowed": False,
                "production_claim_allowed": False,
            }
        )
    return sorted(rows, key=lambda item: -item["combined_priority_score"])


def metocean_selections() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variables = ("temperature_2m", "wave_height_m", "wind_speed_ms", "pressure_hpa")
    for variable_index, variable in enumerate(variables):
        for horizon in (6, 12, 24, 48, 72):
            reference = 0.42 + variable_index * 0.21 + horizon * 0.004
            gain = 3.0 + ((horizon + variable_index) % 5) * 1.1
            challenger = reference * (1.0 - gain / 100.0)
            accepted = horizon in {24, 48}
            rows.append(
                {
                    "variable": variable,
                    "horizon_h": horizon,
                    "b62_model": "DEMO_REFERENCE",
                    "selected_model": "DEMO_TAIL_CHALLENGER" if accepted else "DEMO_REFERENCE",
                    "challenger_accepted": accepted,
                    "valid_b62_mae": round(reference, 4),
                    "valid_challenger_mae": round(challenger, 4),
                    "valid_challenger_gain_pct": round(gain, 3),
                    "valid_challenger_coverage": 0.82,
                    "test_model": "NOT_CONSUMED_IN_SELECTION",
                    "test_mae": None,
                    "test_bias": None,
                    "test_coverage": None,
                    "selection_role": "DEMO_VALID_SELECTION",
                    "test_role": "NOT_CONSUMED",
                    "production_promotion_allowed": False,
                }
            )
    return rows


def metocean_metrics() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for role_index, role in enumerate(
        ("VALID_SELECTION", "ARCHIVE_CONFIRMATORY", "FRESH_FORWARD_CONFIRMATORY")
    ):
        for model_index, model in enumerate(("DEMO_REFERENCE", "DEMO_TAIL_CHALLENGER")):
            mae = 1.18 - 0.08 * model_index + 0.06 * role_index
            rows.append(
                {
                    "evaluation_role": role,
                    "model": model,
                    "rows": 720 - role_index * 120,
                    "origins": 30 - role_index * 6,
                    "mae": round(mae, 4),
                    "rmse": round(mae * 1.31, 4),
                    "bias": round((-0.06 + 0.04 * role_index) * (1 - model_index * 0.4), 4),
                    "coverage": round(0.84 - 0.02 * role_index + 0.01 * model_index, 4),
                    "mean_interval_width": round(2.9 + role_index * 0.25, 4),
                    "quantile_crossings": 0,
                }
            )
    return rows
