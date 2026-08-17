from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field


router = APIRouter(prefix="/api/v1/control-tower", tags=["PortFlow Control Tower"])

STAGES = (
    ("ZRE", "ZRE", 180, 2.4, 5.8),
    ("COULOIR", "Couloir", 145, 1.6, 4.2),
    ("PARK", "Park", 210, 5.1, 12.8),
    ("SCAN", "Scan", 120, 2.8, 7.9),
    ("PV", "PV", 72, 4.4, 11.6),
    ("SAS", "SAS", 96, 1.3, 3.7),
    ("TERMINAL", "Terminal", 240, 3.6, 8.8),
)
ROUTES = ("DIRECT", "PV", "REVUE")
CAUSES = (
    "Séjour Park supérieur au profil habituel",
    "Charge Scan en hausse sur les prochaines heures",
    "Contrôle PV probable avant accès au SAS",
    "Événement attendu non reçu dans le délai normal",
    "Fenêtre terminale proche avec marge réduite",
    "Progression régulière, surveillance préventive",
)
ASSIGNEES = ("Équipe ZRE", "Superviseur flux", "Chef de quart", "Cellule contrôle", None)
DECISION_STATUSES = (
    "À analyser",
    "Décidée",
    "Affectée",
    "En cours",
    "Vérifiée",
    "Clôturée",
)
VESSEL_NAMES = (
    "Atlas Horizon",
    "Tangier Star",
    "Mistral Bridge",
    "Ocean Cedar",
    "Strait Pioneer",
    "Maghreb Trader",
)


class DecisionCreate(BaseModel):
    title: str = Field(min_length=4, max_length=160)
    description: str = Field(default="", max_length=1000)
    assignee: str = Field(default="Chef de quart", max_length=100)
    due_at: datetime | None = None
    alert_id: str | None = None
    unit_ids: list[str] = Field(default_factory=list)


class DecisionPatch(BaseModel):
    status: Literal[
        "À analyser", "Décidée", "Affectée", "En cours", "Vérifiée", "Clôturée"
    ] | None = None
    assignee: str | None = None
    comment: str | None = None
    outcome: str | None = None


class SimulationRequest(BaseModel):
    stage: str = "SCAN"
    capacity_boost: int = Field(default=18, ge=0, le=100)
    duration_h: int = Field(default=4, ge=1, le=24)
    arrival_change_pct: int = Field(default=0, ge=-40, le=80)
    route_policy: Literal["CURRENT", "DIRECT", "PV"] = "CURRENT"


_decision_lock = Lock()
_decisions: dict[str, dict[str, Any]] = {}
_audit_extra: list[dict[str, Any]] = []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _phase(now: datetime) -> int:
    return int(now.timestamp() // 300)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _risk_tier(score: float) -> str:
    if score >= 0.76:
        return "CRITIQUE"
    if score >= 0.48:
        return "VIGILANCE"
    return "NORMAL"


def _build_units(now: datetime) -> list[dict[str, Any]]:
    phase = _phase(now)
    rows: list[dict[str, Any]] = []
    for index in range(64):
        stage_index = (index * 5 + phase // 3) % len(STAGES)
        stage, stage_label, _, median_dwell, _ = STAGES[stage_index]
        pulse = 0.5 + 0.5 * math.sin((phase + index * 1.91) / 4.7)
        route = ROUTES[(index + phase // 9) % len(ROUTES)]
        dwell_h = median_dwell * (0.62 + 2.05 * pulse) + (2.2 if route == "REVUE" else 0)
        remaining_steps = max(1, len(STAGES) - stage_index)
        eta_p50 = 1.4 + remaining_steps * 2.05 + dwell_h * 0.58 + (3.8 if route == "PV" else 0)
        spread = 1.7 + 5.8 * pulse + (2.1 if route == "REVUE" else 0)
        ge12 = _clamp((eta_p50 - 8.0) / 18.0 + pulse * 0.18)
        ge24 = _clamp((eta_p50 - 15.0) / 22.0 + pulse * 0.12)
        ge36 = _clamp((eta_p50 - 27.0) / 23.0 + pulse * 0.09)
        confidence = _clamp(0.94 - spread / 24.0 - (0.09 if index % 13 == 0 else 0), 0.45, 0.96)
        urgency = _clamp(0.22 + ge24 * 0.63 + ge36 * 0.25)
        impact = _clamp(0.36 + (index % 7) * 0.085)
        risk = _clamp(0.42 * ge24 + 0.38 * ge36 + 0.20 * pulse)
        priority = _clamp((urgency * impact * risk * confidence) ** 0.25)
        age_minutes = 2 + (index * 7 + phase) % 58
        quality = "GPS actuelle" if age_minutes <= 8 else "Dernière zone métier" if age_minutes <= 30 else "Position ancienne"
        rows.append(
            {
                "unit_id": f"TMU-26-{12000 + index:05d}",
                "stage": stage,
                "stage_label": stage_label,
                "dwell_h": round(dwell_h, 1),
                "eta_p10_h": round(max(0.2, eta_p50 - spread * 0.65), 1),
                "eta_p50_h": round(eta_p50, 1),
                "eta_p80_h": round(eta_p50 + spread * 0.55, 1),
                "eta_p90_h": round(eta_p50 + spread, 1),
                "ge12": round(ge12, 3),
                "ge24": round(ge24, 3),
                "ge36": round(ge36, 3),
                "route": route,
                "confidence": round(confidence, 3),
                "cause": CAUSES[(index + stage_index) % len(CAUSES)],
                "urgency": round(urgency, 3),
                "impact": round(impact, 3),
                "priority": round(priority, 3),
                "tier": _risk_tier(priority),
                "assignee": ASSIGNEES[(index + phase // 11) % len(ASSIGNEES)],
                "status": ("À analyser", "Affectée", "En cours", "Surveillance")[index % 4],
                "last_event_at": now - timedelta(minutes=age_minutes),
                "location_quality": quality,
                "location_age_minutes": age_minutes,
                "location": {
                    "zone": stage_label,
                    "x": 8 + stage_index * 13 + (index % 4) * 1.8,
                    "y": 28 + ((index * 17) % 46),
                    "precision": "EXACTE" if quality == "GPS actuelle" else "ZONE",
                },
            }
        )
    return sorted(rows, key=lambda item: (-item["priority"], item["unit_id"]))


def _build_stages(now: datetime) -> list[dict[str, Any]]:
    phase = _phase(now)
    rows = []
    for index, (code, label, capacity, median_dwell, p90_dwell) in enumerate(STAGES):
        wave = 0.5 + 0.5 * math.sin((phase + index * 3.3) / 5.2)
        units = round(capacity * (0.55 + wave * 0.49))
        occupancy = units / capacity
        rows.append(
            {
                "code": code,
                "label": label,
                "order": index,
                "units": units,
                "capacity": capacity,
                "occupancy_pct": round(occupancy * 100, 1),
                "dwell_median_h": round(median_dwell * (0.9 + wave * 0.34), 1),
                "dwell_p90_h": round(p90_dwell * (0.88 + wave * 0.42), 1),
                "blocked": max(0, round(units * max(0, occupancy - 0.72) * 0.23)),
                "trend": "HAUSSE" if wave > 0.62 else "BAISSE" if wave < 0.35 else "STABLE",
                "forecast": {
                    f"h{horizon}": round(
                        units * (1 + 0.06 * math.sin((phase + horizon + index) / 2.8))
                        + horizon * (0.7 + index * 0.11)
                    )
                    for horizon in (1, 3, 6, 12, 24)
                },
            }
        )
    return rows


def _build_forecast(now: datetime, stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    phase = _phase(now)
    current_backlog = sum(stage["units"] for stage in stages)
    rows = []
    for horizon in range(1, 25):
        arrivals = 48 + 17 * math.sin((phase + horizon) / 3.4) + 8 * math.cos(horizon / 2.1)
        departures = 46 + 13 * math.sin((phase + horizon - 2) / 4.1)
        backlog = max(80, current_backlog + sum(
            2 + 4 * math.sin((phase + step) / 3.4) for step in range(1, horizon + 1)
        ))
        width = 22 + horizon * 1.8
        rows.append(
            {
                "horizon_h": horizon,
                "valid_at": now + timedelta(hours=horizon),
                "arrivals": round(max(0, arrivals)),
                "departures": round(max(0, departures)),
                "backlog_p10": round(max(0, backlog - width)),
                "backlog_p50": round(backlog),
                "backlog_p90": round(backlog + width),
                "normal_capacity": 515,
                "reinforced_capacity": 590,
            }
        )
    return rows


def _build_alerts(now: datetime, units: list[dict[str, Any]], stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hot_stage = max(stages, key=lambda item: item["occupancy_pct"])
    critical = [unit for unit in units if unit["tier"] == "CRITIQUE"]
    return [
        {
            "alert_id": "ALT-260816-01",
            "severity": "CRITIQUE" if hot_stage["occupancy_pct"] >= 95 else "VIGILANCE",
            "title": f"Risque de saturation {hot_stage['label']}",
            "message": f"La charge pourrait dépasser la capacité entre H+3 et H+6. {min(19, len(critical) + 8)} unités sont directement exposées.",
            "probability": 0.82,
            "impact": "Allongement probable de l’ETA prudente et report vers les zones amont.",
            "cause": f"Occupation actuelle {hot_stage['occupancy_pct']:.0f}% et arrivées en hausse.",
            "recommendation": f"Renforcer temporairement la capacité {hot_stage['label']} avant H+2.",
            "deadline_at": now + timedelta(hours=2),
            "confidence": 0.86,
            "unit_ids": [unit["unit_id"] for unit in critical[:12]],
            "status": "À traiter",
        },
        {
            "alert_id": "ALT-260816-02",
            "severity": "VIGILANCE",
            "title": "Unités sans progression récente",
            "message": "Onze unités n’ont pas produit l’événement attendu dans leur fenêtre habituelle.",
            "probability": 0.74,
            "impact": "Risque d’accumulation invisible dans le Park et le Scan.",
            "cause": "Délai inter-événements supérieur au P90 historique.",
            "recommendation": "Lancer une vérification ciblée des unités concernées.",
            "deadline_at": now + timedelta(minutes=50),
            "confidence": 0.79,
            "unit_ids": [unit["unit_id"] for unit in units[10:21]],
            "status": "En analyse",
        },
        {
            "alert_id": "ALT-260816-03",
            "severity": "INFORMATION",
            "title": "Fenêtre navire avancée",
            "message": "Atlas Horizon pourrait se présenter 1 h 35 plus tôt que la fenêtre initiale.",
            "probability": 0.68,
            "impact": "Quatorze unités associées disposent d’une marge terminale réduite.",
            "cause": "Vitesse d’approche supérieure au profil du voyage.",
            "recommendation": "Surveiller les unités associées et confirmer la fenêtre d’accostage.",
            "deadline_at": now + timedelta(hours=3),
            "confidence": 0.72,
            "unit_ids": [unit["unit_id"] for unit in units[21:35]],
            "status": "Surveillance",
        },
    ]


def _ensure_decisions(now: datetime) -> None:
    with _decision_lock:
        if _decisions:
            return
        seeds = (
            ("DEC-001", "Renforcer la capacité Scan", "Affectée", "Chef de quart", 2),
            ("DEC-002", "Vérifier les unités sans mouvement", "En cours", "Cellule contrôle", 1),
            ("DEC-003", "Préparer la fenêtre Atlas Horizon", "À analyser", "Superviseur flux", 4),
        )
        for decision_id, title, status, assignee, due_h in seeds:
            _decisions[decision_id] = {
                "decision_id": decision_id,
                "title": title,
                "description": "Décision préparée depuis une alerte consolidée.",
                "status": status,
                "assignee": assignee,
                "created_at": now - timedelta(hours=3),
                "updated_at": now - timedelta(minutes=35),
                "due_at": now + timedelta(hours=due_h),
                "alert_id": f"ALT-260816-0{len(_decisions) + 1}",
                "unit_ids": [],
                "comments": [],
                "outcome": None,
                "expected_effect": "Réduire le backlog et protéger les ETA prudentes.",
            }


def _build_vessels(now: datetime) -> list[dict[str, Any]]:
    rows = []
    for index, name in enumerate(VESSEL_NAMES):
        longitude = -7.0 + index * 0.37
        latitude = 35.5 + (index % 3) * 0.25
        speed = 8.2 + index * 1.15
        eta_h = 3.5 + index * 3.2
        rows.append(
            {
                "vessel_id": f"VSL-{8100 + index}",
                "name": name,
                "imo": f"IMO 98{4100 + index}",
                "mmsi": f"24200{310 + index}",
                "longitude": longitude,
                "latitude": latitude,
                "heading": 72 + index * 9,
                "speed_kn": round(speed, 1),
                "status": "EN ROUTE" if index < 4 else "ATTENTE",
                "announced_eta": now + timedelta(hours=eta_h + (1.5 if index % 2 else 0)),
                "predicted_eta": now + timedelta(hours=eta_h),
                "eta_delta_minutes": -90 if index == 0 else 35 * (index - 2),
                "distance_nm": round(18 + index * 14.5, 1),
                "terminal": f"TC{1 + index % 4}",
                "berth_window": f"H+{round(eta_h)} à H+{round(eta_h + 3)}",
                "ais_age_minutes": 2 + index * 4,
                "ais_quality": "BONNE" if index < 4 else "MOYENNE",
                "associated_units": 28 + index * 13,
                "units_ready": 18 + index * 8,
                "congestion_risk": round(_clamp(0.31 + index * 0.11), 2),
            }
        )
    return rows


def _snapshot() -> dict[str, Any]:
    now = _now()
    units = _build_units(now)
    stages = _build_stages(now)
    forecast = _build_forecast(now, stages)
    alerts = _build_alerts(now, units, stages)
    _ensure_decisions(now)
    active_total = sum(stage["units"] for stage in stages)
    critical_count = sum(1 for unit in units if unit["tier"] == "CRITIQUE")
    eta_values = sorted(unit["eta_p50_h"] for unit in units)
    metrics = {
        "active_units": active_total,
        "at_risk_units": critical_count + sum(1 for unit in units if unit["tier"] == "VIGILANCE"),
        "median_eta_h": eta_values[len(eta_values) // 2],
        "ge12_units": sum(1 for unit in units if unit["ge12"] >= 0.5),
        "ge24_units": sum(1 for unit in units if unit["ge24"] >= 0.5),
        "ge36_units": sum(1 for unit in units if unit["ge36"] >= 0.5),
        "open_alerts": sum(1 for alert in alerts if alert["status"] != "Clôturée"),
        "pending_decisions": sum(
            1 for item in _decisions.values() if item["status"] not in {"Vérifiée", "Clôturée"}
        ),
    }
    sources = [
        {"source": "Événements métier", "status": "À JOUR", "age_minutes": 2, "completeness_pct": 99.3, "detail": "Dernier événement consolidé"},
        {"source": "Prédictions ETA", "status": "SHADOW", "age_minutes": 8, "completeness_pct": 98.7, "detail": "Contrat prêt, moteur à raccorder"},
        {"source": "Localisation unités", "status": "PARTIEL", "age_minutes": 14, "completeness_pct": 87.4, "detail": "GPS ou dernière zone métier"},
        {"source": "Positions navires", "status": "EXERCICE", "age_minutes": 4, "completeness_pct": 94.8, "detail": "Contrat AIS prêt"},
    ]
    audit = [
        {"event_id": f"AUD-{index:03d}", "at": now - timedelta(minutes=index * 17), "actor": ("Système", "Chef de quart", "Superviseur flux")[index % 3], "action": ("Prévision recalculée", "Alerte consolidée", "Décision mise à jour", "Snapshot qualité publié")[index % 4], "object": (units[index % len(units)]["unit_id"], alerts[index % len(alerts)]["alert_id"], f"DEC-00{1 + index % 3}")[index % 3], "immutable": True}
        for index in range(12)
    ] + list(_audit_extra)
    return {
        "contract_version": "control-tower-mvp-v1",
        "mode": "EXERCISE",
        "serving_status": "CONTRACT_READY_MODEL_NOT_CONNECTED",
        "generated_at": now,
        "refresh_after_seconds": 30,
        "metrics": metrics,
        "stages": stages,
        "units": units,
        "forecast": forecast,
        "alerts": alerts,
        "decisions": sorted(_decisions.values(), key=lambda item: item["due_at"]),
        "vessels": _build_vessels(now),
        "sources": sources,
        "audit": sorted(audit, key=lambda item: item["at"], reverse=True),
        "recommendations": [
            {"recommendation_id": "REC-001", "title": "Renforcer Scan pendant 4 h", "expected_gain_h": 2.8, "beneficiary_units": 47, "confidence": 0.84, "secondary_risk": "Déplacement temporaire de la file vers le SAS", "evidence": ["Charge H+3", "Dwell P90", "Unités GE24"]},
            {"recommendation_id": "REC-002", "title": "Prioriser 14 unités Atlas Horizon", "expected_gain_h": 1.9, "beneficiary_units": 14, "confidence": 0.77, "secondary_risk": "Retard marginal pour 6 unités non prioritaires", "evidence": ["ETA maritime", "Fenêtre terminale", "Marge unité-navire"]},
        ],
        "permissions": ["VIEW", "DECIDE", "ASSIGN", "EXPORT", "SIMULATE"],
    }


@router.get("/snapshot")
def snapshot() -> dict[str, Any]:
    return _snapshot()


@router.get("/units/{unit_id}")
def unit_detail(unit_id: str) -> dict[str, Any]:
    snap = _snapshot()
    unit = next((item for item in snap["units"] if item["unit_id"] == unit_id), None)
    if unit is None:
        raise HTTPException(status_code=404, detail="Unité introuvable")
    now = snap["generated_at"]
    stage_index = next(index for index, stage in enumerate(STAGES) if stage[0] == unit["stage"])
    timeline = [
        {"at": now - timedelta(hours=(stage_index - index + 1) * 3.2), "stage": stage[0], "label": stage[1], "duration_h": round(1.2 + index * 0.7, 1), "reliability": "FIABLE"}
        for index, stage in enumerate(STAGES[: stage_index + 1])
    ]
    eta_history = [
        {"issued_at": now - timedelta(hours=12 - offset * 2), "p50_h": round(unit["eta_p50_h"] + 4.2 - offset * 0.7, 1), "p90_h": round(unit["eta_p90_h"] + 4.8 - offset * 0.8, 1)}
        for offset in range(7)
    ]
    return {
        **unit,
        "timeline": timeline,
        "eta_history": eta_history,
        "explanation": f"L’ETA prudente atteint {unit['eta_p90_h']:.1f} h principalement en raison de : {unit['cause'].lower()}. La route {unit['route']} reste la plus probable avec une confiance de {unit['confidence'] * 100:.0f} %.",
        "previous_alerts": [alert for alert in snap["alerts"] if unit_id in alert["unit_ids"]],
        "previous_decisions": [item for item in snap["decisions"] if unit_id in item["unit_ids"]],
        "prediction": {"calculated_at": now - timedelta(minutes=8), "freshness_minutes": 8, "model": "Moteur ETA à raccorder", "fallback": False, "experimental": True},
    }


@router.post("/decisions")
def create_decision(payload: DecisionCreate) -> dict[str, Any]:
    now = _now()
    with _decision_lock:
        decision_id = f"DEC-{len(_decisions) + 1:03d}"
        row = {
            "decision_id": decision_id,
            "title": payload.title,
            "description": payload.description,
            "status": "À analyser",
            "assignee": payload.assignee,
            "created_at": now,
            "updated_at": now,
            "due_at": payload.due_at or now + timedelta(hours=4),
            "alert_id": payload.alert_id,
            "unit_ids": payload.unit_ids,
            "comments": [],
            "outcome": None,
            "expected_effect": "À confirmer par le responsable affecté.",
        }
        _decisions[decision_id] = row
        _audit_extra.append({"event_id": f"AUD-X-{len(_audit_extra) + 1:03d}", "at": now, "actor": "Utilisateur local", "action": "Décision créée", "object": decision_id, "immutable": True})
    return row


@router.patch("/decisions/{decision_id}")
def update_decision(decision_id: str, payload: DecisionPatch) -> dict[str, Any]:
    now = _now()
    with _decision_lock:
        row = _decisions.get(decision_id)
        if row is None:
            raise HTTPException(status_code=404, detail="Décision introuvable")
        if payload.status is not None:
            row["status"] = payload.status
        if payload.assignee is not None:
            row["assignee"] = payload.assignee
        if payload.comment:
            row["comments"].append({"at": now, "author": "Utilisateur local", "text": payload.comment})
        if payload.outcome is not None:
            row["outcome"] = payload.outcome
        row["updated_at"] = now
        _audit_extra.append({"event_id": f"AUD-X-{len(_audit_extra) + 1:03d}", "at": now, "actor": "Utilisateur local", "action": "Décision mise à jour", "object": decision_id, "immutable": True})
    return row


@router.post("/simulations")
def simulate(payload: SimulationRequest) -> dict[str, Any]:
    if payload.stage not in {stage[0] for stage in STAGES}:
        raise HTTPException(status_code=422, detail="Étape inconnue")
    demand_factor = 1 + payload.arrival_change_pct / 100
    route_gain = 0.08 if payload.route_policy == "DIRECT" else -0.05 if payload.route_policy == "PV" else 0
    effective_boost = payload.capacity_boost * payload.duration_h / 4
    before = {"max_backlog": round(546 * demand_factor), "ge24_units": round(87 * demand_factor), "mean_eta_p90_h": round(23.4 * demand_factor, 1), "recovery_h": 11.0}
    after = {
        "max_backlog": max(80, round(before["max_backlog"] - effective_boost * 3.9 - route_gain * 180)),
        "ge24_units": max(0, round(before["ge24_units"] - effective_boost * 0.72 - route_gain * 55)),
        "mean_eta_p90_h": max(4.0, round(before["mean_eta_p90_h"] - effective_boost * 0.095 - route_gain * 8, 1)),
        "recovery_h": max(1.0, round(before["recovery_h"] - effective_boost * 0.08 - route_gain * 9, 1)),
    }
    return {
        "simulation_id": f"SIM-{int(_now().timestamp())}",
        "created_at": _now(),
        "inputs": payload,
        "before": before,
        "after": after,
        "confidence": 0.71,
        "status": "EXERCISE_ENGINE",
        "recommendation": f"Le renforcement de {payload.stage} réduit le backlog maximal de {before['max_backlog'] - after['max_backlog']} unités dans ce scénario.",
        "automatic_action_allowed": False,
    }


@router.get("/reports/shift")
def shift_report() -> dict[str, Any]:
    snap = _snapshot()
    return {
        "report_type": "RAPPORT_DE_QUART",
        "generated_at": snap["generated_at"],
        "mode": snap["mode"],
        "summary": snap["metrics"],
        "critical_alerts": [item for item in snap["alerts"] if item["severity"] == "CRITIQUE"],
        "open_decisions": [item for item in snap["decisions"] if item["status"] != "Clôturée"],
        "data_sources": snap["sources"],
        "notice": "Contrat d’export prêt. Les exports PDF/Excel réels seront branchés au moteur de reporting.",
    }
