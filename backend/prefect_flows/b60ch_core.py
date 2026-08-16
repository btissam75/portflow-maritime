from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


REGISTRY_VERSION = "b60ch-morocco-maritime-event-history-v1"
CONTEXT_VERSION = "b60ch-port-call-event-context-v1"
SOURCE_DATASET_VERSION = "b60c-operational-port-call-landmark-v1"

MAROC_HOLIDAY_SOURCE = "https://maroc.ma/fr/le-maroc/fetes-nationales-et-religieuses"
HABOUS_SOURCE = "https://www.habous.gov.ma/"
TANGER_MED_DOCS = "https://www.tangermed.ma/fr/documentation/"


@dataclass
class EventIntelligenceResult:
    registry: pd.DataFrame
    context: pd.DataFrame
    reports: dict[str, pd.DataFrame]
    decision: dict[str, Any]
    feature_columns: list[str]


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _as_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _event_row(
    event_id: str,
    event_name: str,
    event_family: str,
    event_type: str,
    start_date: str,
    end_date_inclusive: str,
    *,
    calendar_role: str,
    knowledge_policy: str,
    known_at: str | pd.Timestamp,
    source_name: str,
    source_uri: str,
    confidence: str = "HIGH",
    affected_flow: str = "ALL_MARITIME",
    source_resolution: str = "EXACT_DATE",
    notes: str = "",
) -> dict[str, Any]:
    start_at = _utc(start_date).normalize()
    end_at = _utc(end_date_inclusive).normalize() + pd.Timedelta(days=1)
    return {
        "registry_version": REGISTRY_VERSION,
        "event_id": event_id,
        "event_name": event_name,
        "event_family": event_family,
        "event_type": event_type,
        "start_at": start_at,
        "end_at": end_at,
        "known_at": _utc(known_at),
        "calendar_role": calendar_role,
        "predictive_feature_allowed": calendar_role == "PREDICTIVE_KNOWN_CALENDAR",
        "retrospective_only": calendar_role != "PREDICTIVE_KNOWN_CALENDAR",
        "knowledge_policy": knowledge_policy,
        "source_name": source_name,
        "source_uri": source_uri,
        "source_resolution": source_resolution,
        "confidence": confidence,
        "affected_flow": affected_flow,
        "notes": notes,
        "synthetic_event": False,
    }


def build_event_registry() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    fixed_holidays = (
        (1, 1, "NEW_YEAR", "New Year", 2020),
        (1, 11, "INDEPENDENCE_MANIFESTO", "Independence Manifesto", 2020),
        (1, 14, "AMAZIGH_NEW_YEAR", "Amazigh New Year", 2024),
        (5, 1, "LABOUR_DAY", "Labour Day", 2020),
        (7, 30, "THRONE_DAY", "Throne Day", 2020),
        (8, 14, "OUED_ED_DAHAB", "Oued Ed-Dahab Day", 2020),
        (8, 20, "REVOLUTION_DAY", "Revolution Day", 2020),
        (8, 21, "YOUTH_DAY", "Youth Day", 2020),
        (11, 6, "GREEN_MARCH", "Green March", 2020),
        (11, 18, "INDEPENDENCE_DAY", "Independence Day", 2020),
    )
    for year in range(2020, 2026):
        for month, day, code, name, effective_year in fixed_holidays:
            if year < effective_year:
                continue
            date = pd.Timestamp(year=year, month=month, day=day)
            known_at = "2000-01-01"
            policy = "STATUTORY_FIXED_DATE"
            if code == "AMAZIGH_NEW_YEAR":
                known_at = "2023-05-03"
                policy = "STATUTORY_FROM_2024"
            rows.append(
                _event_row(
                    f"MA_FIXED_{code}_{year}",
                    f"{name} {year}",
                    "NATIONAL_HOLIDAY",
                    "PUBLIC_HOLIDAY",
                    str(date.date()),
                    str(date.date()),
                    calendar_role="PREDICTIVE_KNOWN_CALENDAR",
                    knowledge_policy=policy,
                    known_at=known_at,
                    source_name="Kingdom of Morocco",
                    source_uri=MAROC_HOLIDAY_SOURCE,
                )
            )

    lunar_dates = {
        2020: {
            "ramadan": ("2020-04-25", "2020-05-23"),
            "fitr": "2020-05-24",
            "adha": "2020-07-31",
            "hijri": "2020-08-21",
            "mawlid": "2020-10-29",
        },
        2021: {
            "ramadan": ("2021-04-14", "2021-05-12"),
            "fitr": "2021-05-13",
            "adha": "2021-07-21",
            "hijri": "2021-08-10",
            "mawlid": "2021-10-19",
        },
        2022: {
            "ramadan": ("2022-04-03", "2022-05-01"),
            "fitr": "2022-05-02",
            "adha": "2022-07-10",
            "hijri": "2022-07-30",
            "mawlid": "2022-10-09",
        },
        2023: {
            "ramadan": ("2023-03-23", "2023-04-21"),
            "fitr": "2023-04-22",
            "adha": "2023-06-29",
            "hijri": "2023-07-19",
            "mawlid": "2023-09-28",
        },
        2024: {
            "ramadan": ("2024-03-12", "2024-04-09"),
            "fitr": "2024-04-10",
            "adha": "2024-06-17",
            "hijri": "2024-07-08",
            "mawlid": "2024-09-16",
        },
        2025: {
            "ramadan": ("2025-03-02", "2025-03-30"),
            "fitr": "2025-03-31",
            "adha": "2025-06-07",
            "hijri": "2025-06-27",
            "mawlid": "2025-09-05",
        },
    }
    specific_sources = {
        "RAMADAN_2020": "https://www.habous.gov.ma/fr/observation-du-croissant-lunaire-aff-isla/6365-1er-ramadan-1441-correspond-au-samedi-25-avril-2020.html",
        "FITR_2020": "https://www.habous.gov.ma/%D9%85%D8%B1%D8%A7%D9%82%D8%A8%D8%A9-%D8%A7%D9%84%D8%A3%D9%87%D9%84%D8%A9-3/14672-%D9%81%D8%A7%D8%AA%D8%AD-%D8%B4%D9%88%D8%A7%D9%84-1441-%D8%A7%D9%84%D8%A3%D8%AD%D8%AF-24-%D9%85%D8%A7%D9%8A-2020.html",
        "RAMADAN_2021": "https://www.habous.gov.ma/fr/observation-du-croissant-lunaire-aff-isla/6430-1er-ramadan-1442-est-le-mercredi-14-avril-2021.html",
        "FITR_2022": "https://www.habous.gov.ma/%D9%85%D8%B1%D8%A7%D9%82%D8%A8%D8%A9-%D8%A7%D9%84%D8%A3%D9%87%D9%84%D8%A9-4/17082-%D9%81%D8%A7%D8%AA%D8%AD-%D8%B4%D9%88%D8%A7%D9%84-1443%D9%80-%D8%A7%D9%84%D8%A5%D8%AB%D9%86%D9%8A%D9%86-02-%D9%85%D8%A7%D9%8A-2022%D9%85.html",
        "RAMADAN_2024": "https://www.habous.gov.ma/fr/derniere-actualite/6490-mardi-premier-jour-du-mois-de-ramadan-au-maroc-minist%C3%A8re.html",
        "RAMADAN_2025": "https://www.habous.gov.ma/%D9%85%D8%B1%D8%A7%D9%82%D8%A8%D8%A9-%D8%A7%D9%84%D8%A3%D9%87%D9%84%D8%A9-4/20763-%D9%81%D8%A7%D8%AA%D8%AD-%D8%B4%D9%87%D8%B1-%D8%B1%D9%85%D8%B6%D8%A7%D9%86-%D8%A7%D9%84%D9%85%D8%B9%D8%B8%D9%85-%D8%A7%D9%84%D8%A3%D8%AD%D8%AF-02-%D9%85%D8%A7%D8%B1%D8%B3-2025%D9%85.html",
        "ADHA_2020": "https://www.habous.gov.ma/%D9%85%D8%B1%D8%A7%D9%82%D8%A8%D8%A9-%D8%A7%D9%84%D8%A3%D9%87%D9%84%D8%A9-4/14764-%D9%81%D8%A7%D8%AA%D8%AD-%D8%B0%D9%8A-%D8%A7%D9%84%D8%AD%D8%AC%D8%A9-1441-%D8%A7%D9%84%D8%A3%D8%B1%D8%A8%D8%B9%D8%A7%D8%A1-22-%D9%8A%D9%88%D9%84%D9%8A%D9%88-2020.html",
    }
    for year, dates in lunar_dates.items():
        ramadan_start, ramadan_end = dates["ramadan"]
        start_known = _utc(ramadan_start) - pd.Timedelta(hours=12)
        rows.append(
            _event_row(
                f"MA_RAMADAN_{year}",
                f"Ramadan {year}",
                "RAMADAN",
                "RELIGIOUS_PERIOD",
                ramadan_start,
                ramadan_end,
                calendar_role="PREDICTIVE_KNOWN_CALENDAR",
                knowledge_policy="OFFICIAL_LUNAR_ANNOUNCEMENT",
                known_at=start_known,
                source_name="Moroccan Ministry of Habous and Islamic Affairs",
                source_uri=specific_sources.get(f"RAMADAN_{year}", HABOUS_SOURCE),
            )
        )
        religious = (
            ("fitr", "EID_FITR", "Eid al-Fitr", 2),
            ("adha", "EID_ADHA", "Eid al-Adha", 2),
            ("hijri", "HIJRI_NEW_YEAR", "Hijri New Year", 1),
            ("mawlid", "MAWLID", "Mawlid", 2),
        )
        for key, family, name, duration_days in religious:
            start = _utc(dates[key]).normalize()
            end_inclusive = start + pd.Timedelta(days=duration_days - 1)
            rows.append(
                _event_row(
                    f"MA_{family}_{year}",
                    f"{name} {year}",
                    family,
                    "RELIGIOUS_HOLIDAY",
                    str(start.date()),
                    str(end_inclusive.date()),
                    calendar_role="PREDICTIVE_KNOWN_CALENDAR",
                    knowledge_policy="OFFICIAL_LUNAR_ANNOUNCEMENT",
                    known_at=start - pd.Timedelta(hours=12),
                    source_name="Moroccan Ministry of Habous and Islamic Affairs",
                    source_uri=specific_sources.get(f"{family.replace('EID_', '')}_{year}", HABOUS_SOURCE),
                )
            )

    for year in range(2019, 2025):
        start = pd.Timestamp(year=year, month=12, day=15)
        end = pd.Timestamp(year=year + 1, month=1, day=10)
        rows.append(
            _event_row(
                f"CAL_YEAR_END_{year}_{year + 1}",
                f"Year-end logistics window {year}/{year + 1}",
                "YEAR_END",
                "DETERMINISTIC_SEASONAL_WINDOW",
                str(start.date()),
                str(end.date()),
                calendar_role="PREDICTIVE_KNOWN_CALENDAR",
                knowledge_policy="DETERMINISTIC_GREGORIAN_CALENDAR",
                known_at="2000-01-01",
                source_name="B60C-H deterministic calendar contract",
                source_uri=MAROC_HOLIDAY_SOURCE,
                confidence="HIGH",
                notes="Hypothesis window; effect must be estimated from data.",
            )
        )

    marhaba = (
        (
            2021,
            "2021-06-15",
            "2021-09-15",
            "https://www.tangermed.ma/en/23190-2/",
            "EXACT_OFFICIAL_WINDOW",
        ),
        (
            2022,
            "2022-06-15",
            "2022-09-15",
            "https://www.tangermed.ma/wp-content/uploads/2023/12/RAPPORT-ANNUEL-VF-2022.pdf",
            "DOCUMENTED_CAMPAIGN_APPROXIMATE_WINDOW",
        ),
        (
            2023,
            "2023-06-05",
            "2023-09-15",
            "https://www.tangermed.ma/en/1616647-passengers-and-412635-vehicles-transited-through-tanger-med-port-during-the-marhaba-2023-cam-paign/",
            "EXACT_OFFICIAL_WINDOW",
        ),
        (
            2024,
            "2024-06-05",
            "2024-09-15",
            "https://www.tangermed.ma/fr/1-734-160-passagers-et-457-624-vehicules-ont-transite-par-le-port-tanger-med-durant-la-campagne-marhaba-2024/",
            "EXACT_OFFICIAL_WINDOW",
        ),
    )
    for year, start, end, uri, resolution in marhaba:
        rows.append(
            _event_row(
                f"TANGER_MED_MARHABA_{year}",
                f"Marhaba campaign {year}",
                "MARHABA",
                "PORT_OPERATING_CAMPAIGN",
                start,
                end,
                calendar_role="RETROSPECTIVE_EXPLANATORY",
                knowledge_policy="SOURCE_PUBLICATION_TIME_NOT_CAPTURED",
                known_at=_utc(end) + pd.Timedelta(days=1),
                source_name="Tanger Med",
                source_uri=uri,
                confidence="MEDIUM" if year == 2022 else "HIGH",
                affected_flow="PASSENGER_AND_VEHICLE",
                source_resolution=resolution,
                notes="Research-only until historical announcement availability is proven.",
            )
        )

    rows.extend(
        [
            _event_row(
                "TANGER_MED_COVID_PASSENGER_SUSPENSION_2020",
                "COVID passenger suspension and partial restart",
                "COVID_DISRUPTION",
                "EXTERNAL_SHOCK",
                "2020-03-15",
                "2020-06-14",
                calendar_role="RETROSPECTIVE_EXPLANATORY",
                knowledge_policy="RETROSPECTIVE_MONTH_RESOLUTION",
                known_at="2020-06-30",
                source_name="Tanger Med 2020 port activity report",
                source_uri="https://www.tangermed.ma/fr/chiffres-definitifs-du-bilan-portuaire-arretes-au-31-12-2020/",
                confidence="MEDIUM",
                affected_flow="PASSENGER",
                source_resolution="OFFICIAL_MONTH_LEVEL",
                notes="Commercial port activity continued; this is not a total port closure.",
            ),
            _event_row(
                "TANGER_MED_COVID_AUTOMOTIVE_SLOWDOWN_2020",
                "COVID automotive export slowdown",
                "COVID_DISRUPTION",
                "EXTERNAL_SHOCK",
                "2020-03-01",
                "2020-05-31",
                calendar_role="RETROSPECTIVE_EXPLANATORY",
                knowledge_policy="RETROSPECTIVE_MONTH_RESOLUTION",
                known_at="2020-06-30",
                source_name="Tanger Med 2020 port activity report",
                source_uri="https://www.tangermed.ma/fr/chiffres-definitifs-du-bilan-portuaire-arretes-au-31-12-2020/",
                confidence="MEDIUM",
                affected_flow="AUTOMOTIVE_EXPORT",
                source_resolution="OFFICIAL_MONTH_LEVEL",
            ),
            _event_row(
                "TANGER_MED_TC3_RAMP_2021",
                "TC3 startup and capacity ramp",
                "CAPACITY_TRANSITION",
                "STRUCTURAL_CHANGE",
                "2021-01-01",
                "2021-12-31",
                calendar_role="RETROSPECTIVE_EXPLANATORY",
                knowledge_policy="RETROSPECTIVE_ANNUAL_RESOLUTION",
                known_at="2022-01-01",
                source_name="Tanger Med 2021 activity report",
                source_uri="https://www.tangermed.ma/fr/bilan-de-lactivite-portuaire-en-2021/",
                confidence="MEDIUM",
                affected_flow="CONTAINER",
                source_resolution="OFFICIAL_YEAR_LEVEL",
            ),
        ]
    )

    registry = pd.DataFrame(rows).sort_values(["start_at", "event_id"]).reset_index(drop=True)
    registry["duration_days"] = (
        registry["end_at"] - registry["start_at"]
    ).dt.total_seconds() / 86400.0
    return registry


def _required_landmark_columns(frame: pd.DataFrame) -> None:
    required = {
        "dataset_version",
        "port_call_id",
        "landmark_at",
        "split",
        "actual_ata",
        "target_departure_delay_h",
        "target_delay_gt_3h",
        "target_delay_gt_6h",
        "target_total_stay_h",
        "data_origin",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"B60C landmark columns are missing: {missing}")


def build_event_context(
    landmarks: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    _required_landmark_columns(landmarks)
    source = landmarks.copy()
    source["landmark_at"] = _as_utc(source["landmark_at"])
    times = source["landmark_at"]
    size = len(source)

    context = source[["dataset_version", "port_call_id", "landmark_at", "split"]].copy()
    context.insert(0, "event_context_version", CONTEXT_VERSION)
    features: list[str] = []
    predictive = registry.loc[registry["predictive_feature_allowed"]].copy()
    retrospective = registry.loc[registry["retrospective_only"]].copy()

    family_aliases = {
        "NATIONAL_HOLIDAY": "national_holiday",
        "RAMADAN": "ramadan",
        "EID_FITR": "eid_fitr",
        "EID_ADHA": "eid_adha",
        "HIJRI_NEW_YEAR": "hijri_new_year",
        "MAWLID": "mawlid",
        "YEAR_END": "year_end",
    }
    horizons = ((0, "now"), (6, "6h"), (12, "12h"), (24, "24h"))
    for horizon_h, suffix in horizons:
        target = times + pd.Timedelta(hours=horizon_h)
        any_active = np.zeros(size, dtype="int8")
        holiday_active = np.zeros(size, dtype="int8")
        family_values = {
            alias: np.zeros(size, dtype="int8") for alias in family_aliases.values()
        }
        for event in predictive.itertuples(index=False):
            known = times.ge(event.known_at)
            active = known & target.ge(event.start_at) & target.lt(event.end_at)
            mask = active.to_numpy()
            any_active[mask] = 1
            if event.event_type in ("PUBLIC_HOLIDAY", "RELIGIOUS_HOLIDAY"):
                holiday_active[mask] = 1
            alias = family_aliases.get(event.event_family)
            if alias:
                family_values[alias][mask] = 1
        any_name = f"known_event_any_{suffix}"
        holiday_name = f"known_holiday_any_{suffix}"
        context[any_name] = any_active
        context[holiday_name] = holiday_active
        features.extend([any_name, holiday_name])
        for alias, values in family_values.items():
            name = f"known_{alias}_{suffix}"
            context[name] = values
            features.append(name)

    event_count = np.zeros(size, dtype="int16")
    pre7 = np.zeros(size, dtype="int8")
    post7 = np.zeros(size, dtype="int8")
    days_to_next = np.full(size, np.nan)
    days_since_previous = np.full(size, np.nan)
    ramadan_day = np.zeros(size, dtype="int8")
    ramadan_progress = np.zeros(size, dtype="float64")
    ramadan_last10 = np.zeros(size, dtype="int8")
    for event in predictive.itertuples(index=False):
        known = times.ge(event.known_at)
        active = known & times.ge(event.start_at) & times.lt(event.end_at)
        event_count += active.to_numpy(dtype="int16")
        pre = known & times.ge(event.start_at - pd.Timedelta(days=7)) & times.lt(event.start_at)
        post = known & times.ge(event.end_at) & times.lt(event.end_at + pd.Timedelta(days=7))
        pre7[pre.to_numpy()] = 1
        post7[post.to_numpy()] = 1

        to_start = (event.start_at - times).dt.total_seconds().to_numpy() / 86400.0
        candidate_next = known.to_numpy() & (to_start >= 0.0) & (to_start <= 90.0)
        days_to_next[candidate_next] = np.fmin(
            np.nan_to_num(days_to_next[candidate_next], nan=np.inf),
            to_start[candidate_next],
        )
        since_end = (times - event.end_at).dt.total_seconds().to_numpy() / 86400.0
        candidate_previous = known.to_numpy() & (since_end >= 0.0) & (since_end <= 90.0)
        days_since_previous[candidate_previous] = np.fmin(
            np.nan_to_num(days_since_previous[candidate_previous], nan=np.inf),
            since_end[candidate_previous],
        )
        if event.event_family == "RAMADAN":
            active_mask = active.to_numpy()
            day = np.floor(
                (times - event.start_at).dt.total_seconds().to_numpy() / 86400.0
            ) + 1
            duration = max(1.0, float(event.duration_days))
            ramadan_day[active_mask] = day[active_mask].astype("int8")
            ramadan_progress[active_mask] = day[active_mask] / duration
            ramadan_last10[active_mask & (day > duration - 10)] = 1

    context["known_event_count_now"] = event_count
    context["known_event_pre_7d"] = pre7
    context["known_event_post_7d"] = post7
    context["known_days_to_next_event_90d"] = np.nan_to_num(days_to_next, nan=91.0)
    context["known_days_since_event_90d"] = np.nan_to_num(days_since_previous, nan=91.0)
    context["known_ramadan_day"] = ramadan_day
    context["known_ramadan_progress"] = ramadan_progress
    context["known_ramadan_last10"] = ramadan_last10
    features.extend(
        [
            "known_event_count_now",
            "known_event_pre_7d",
            "known_event_post_7d",
            "known_days_to_next_event_90d",
            "known_days_since_event_90d",
            "known_ramadan_day",
            "known_ramadan_progress",
            "known_ramadan_last10",
        ]
    )

    normalized = times.dt.normalize()
    days_in_month = times.dt.days_in_month
    context["calendar_month_end_last3d"] = times.dt.day.ge(days_in_month - 2).astype("int8")
    context["calendar_quarter_end_last7d"] = (
        times.dt.month.isin([3, 6, 9, 12]) & times.dt.day.ge(days_in_month - 6)
    ).astype("int8")
    context["calendar_year_end_last21d"] = (
        times.dt.month.eq(12) & times.dt.day.ge(11)
    ).astype("int8")
    context["calendar_week_of_year_sin"] = np.sin(
        2.0 * np.pi * times.dt.isocalendar().week.astype(float) / 52.18
    )
    context["calendar_week_of_year_cos"] = np.cos(
        2.0 * np.pi * times.dt.isocalendar().week.astype(float) / 52.18
    )
    features.extend(
        [
            "calendar_month_end_last3d",
            "calendar_quarter_end_last7d",
            "calendar_year_end_last21d",
            "calendar_week_of_year_sin",
            "calendar_week_of_year_cos",
        ]
    )

    research_any = np.zeros(size, dtype="int8")
    research_covid = np.zeros(size, dtype="int8")
    research_marhaba = np.zeros(size, dtype="int8")
    research_capacity = np.zeros(size, dtype="int8")
    for event in retrospective.itertuples(index=False):
        active = normalized.ge(event.start_at) & normalized.lt(event.end_at)
        mask = active.to_numpy()
        research_any[mask] = 1
        if event.event_family == "COVID_DISRUPTION":
            research_covid[mask] = 1
        elif event.event_family == "MARHABA":
            research_marhaba[mask] = 1
        elif event.event_family == "CAPACITY_TRANSITION":
            research_capacity[mask] = 1
    context["research_event_any_now"] = research_any
    context["research_covid_disruption_now"] = research_covid
    context["research_marhaba_now"] = research_marhaba
    context["research_capacity_transition_now"] = research_capacity
    research_features = [
        "research_event_any_now",
        "research_covid_disruption_now",
        "research_marhaba_now",
        "research_capacity_transition_now",
    ]
    features.extend(research_features)
    context["predictive_feature_count"] = len(features) - len(research_features)
    context["research_feature_count"] = len(research_features)
    context["synthetic_feature_rows"] = 0
    return context, features


def _call_level(landmarks: pd.DataFrame) -> pd.DataFrame:
    source = landmarks.copy()
    source["landmark_at"] = _as_utc(source["landmark_at"])
    source["actual_ata"] = _as_utc(source["actual_ata"])
    calls = (
        source.sort_values(["port_call_id", "landmark_at"])
        .drop_duplicates("port_call_id", keep="first")
        .reset_index(drop=True)
    )
    calls["event_time"] = calls["actual_ata"].fillna(calls["landmark_at"])
    calls["year"] = calls["event_time"].dt.year
    calls["month"] = calls["event_time"].dt.month
    calls["dow"] = calls["event_time"].dt.dayofweek
    calls["match_cell"] = (
        calls["year"].astype(str)
        + ":"
        + calls["month"].astype(str)
        + ":"
        + calls["dow"].astype(str)
    )
    return calls


def build_call_event_exposure(
    calls: pd.DataFrame,
    registry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    times = _as_utc(calls["event_time"])
    for family, events in registry.groupby("event_family", sort=True):
        phase = np.full(len(calls), "NONE", dtype=object)
        event_ids = np.full(len(calls), "", dtype=object)
        for event in events.itertuples(index=False):
            during = times.ge(event.start_at) & times.lt(event.end_at)
            pre = times.ge(event.start_at - pd.Timedelta(days=7)) & times.lt(event.start_at)
            post = times.ge(event.end_at) & times.lt(event.end_at + pd.Timedelta(days=7))
            phase[(post & (phase == "NONE")).to_numpy()] = "POST_7D"
            phase[(pre & (phase == "NONE")).to_numpy()] = "PRE_7D"
            phase[during.to_numpy()] = "DURING"
            event_ids[(pre | during | post).to_numpy()] = event.event_id
        part = calls[
            [
                "port_call_id",
                "split",
                "event_time",
                "match_cell",
                "target_delay_gt_3h",
                "target_delay_gt_6h",
                "target_departure_delay_h",
                "target_total_stay_h",
            ]
        ].copy()
        part["event_family"] = family
        part["phase"] = phase
        part["event_id"] = event_ids
        rows.append(part)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _normal_pvalue(effect: float, treated: np.ndarray, control: np.ndarray) -> float:
    if len(treated) < 2 or len(control) < 2:
        return np.nan
    treated_var = float(np.var(treated, ddof=1))
    control_var = float(np.var(control, ddof=1))
    standard_error = math.sqrt(treated_var / len(treated) + control_var / len(control))
    if standard_error <= 0.0:
        return 1.0 if abs(effect) < 1e-12 else 0.0
    return math.erfc(abs(effect / standard_error) / math.sqrt(2.0))


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype="float64")
    valid = values.dropna().sort_values()
    if valid.empty:
        return result
    count = len(valid)
    adjusted = np.minimum(1.0, valid.to_numpy() * count / np.arange(1, count + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result.loc[valid.index] = adjusted
    return result


def estimate_event_associations(exposure: pd.DataFrame) -> pd.DataFrame:
    if exposure.empty:
        return pd.DataFrame()
    outcomes = {
        "MAJOR_DELAY_RATE": "target_delay_gt_3h",
        "CRITICAL_DELAY_RATE": "target_delay_gt_6h",
        "DEPARTURE_DELAY_H": "target_departure_delay_h",
        "TOTAL_STAY_H": "target_total_stay_h",
    }
    split_roles = {
        "TRAIN": "TRAIN_DISCOVERY",
        "VALID": "VALID_CONFIRMATION",
        "TEST": "TEST_DIAGNOSTIC_ONLY",
    }
    rows: list[dict[str, Any]] = []
    for (split, family), group in exposure.groupby(["split", "event_family"], sort=True):
        for phase in ("PRE_7D", "DURING", "POST_7D"):
            treated = group.loc[group["phase"].eq(phase)]
            if treated.empty:
                continue
            cells = set(treated["match_cell"])
            control = group.loc[group["phase"].eq("NONE") & group["match_cell"].isin(cells)]
            for outcome, column in outcomes.items():
                treated_values = pd.to_numeric(treated[column], errors="coerce").dropna().to_numpy(float)
                control_values = pd.to_numeric(control[column], errors="coerce").dropna().to_numpy(float)
                estimable = len(treated_values) >= 15 and len(control_values) >= 30
                treated_mean = float(np.mean(treated_values)) if len(treated_values) else np.nan
                control_mean = float(np.mean(control_values)) if len(control_values) else np.nan
                effect = treated_mean - control_mean
                rows.append(
                    {
                        "split": split,
                        "analysis_role": split_roles.get(split, "EXCLUDED"),
                        "event_family": family,
                        "phase": phase,
                        "outcome": outcome,
                        "treated_calls": len(treated_values),
                        "matched_control_calls": len(control_values),
                        "treated_mean": treated_mean,
                        "control_mean": control_mean,
                        "absolute_effect": effect,
                        "relative_effect_pct": (
                            100.0 * effect / abs(control_mean)
                            if math.isfinite(control_mean) and abs(control_mean) > 1e-12
                            else np.nan
                        ),
                        "p_value": (
                            _normal_pvalue(effect, treated_values, control_values)
                            if estimable
                            else np.nan
                        ),
                        "estimable": estimable,
                        "causal_claim_allowed": False,
                        "selection_allowed": split == "TRAIN",
                    }
                )
    report = pd.DataFrame(rows)
    if report.empty:
        return report
    report["q_value_bh"] = report.groupby("split", group_keys=False)["p_value"].apply(
        _benjamini_hochberg
    )
    report["fdr_10pct_signal"] = report["q_value_bh"].le(0.10).fillna(False)

    train = report.loc[report["split"].eq("TRAIN")].set_index(
        ["event_family", "phase", "outcome"]
    )["absolute_effect"]
    confirmation = []
    for row in report.itertuples(index=False):
        key = (row.event_family, row.phase, row.outcome)
        train_effect = train.get(key, np.nan)
        confirmation.append(
            bool(
                row.split in ("VALID", "TEST")
                and math.isfinite(float(train_effect))
                and math.isfinite(float(row.absolute_effect))
                and np.sign(train_effect) == np.sign(row.absolute_effect)
            )
        )
    report["train_direction_confirmed"] = confirmation
    return report.sort_values(
        ["analysis_role", "event_family", "phase", "outcome"]
    ).reset_index(drop=True)


def build_daily_panel(calls: pd.DataFrame) -> pd.DataFrame:
    source = calls.copy()
    source["date"] = _as_utc(source["event_time"]).dt.normalize()
    grouped = source.groupby("date", as_index=True).agg(
        arrivals=("port_call_id", "nunique"),
        major_delay_calls=("target_delay_gt_3h", "sum"),
        critical_delay_calls=("target_delay_gt_6h", "sum"),
        major_delay_rate=("target_delay_gt_3h", "mean"),
        critical_delay_rate=("target_delay_gt_6h", "mean"),
        median_departure_delay_h=("target_departure_delay_h", "median"),
        median_total_stay_h=("target_total_stay_h", "median"),
    )
    index = pd.date_range(source["date"].min(), source["date"].max(), freq="D", tz="UTC")
    panel = grouped.reindex(index).rename_axis("date").reset_index()
    panel["arrivals"] = panel["arrivals"].fillna(0.0)
    panel["split"] = np.select(
        [
            panel["date"].lt(pd.Timestamp("2024-01-01", tz="UTC")),
            panel["date"].lt(pd.Timestamp("2025-01-01", tz="UTC")),
        ],
        ["TRAIN", "VALID"],
        default="TEST",
    )
    panel["dow"] = panel["date"].dt.dayofweek
    return panel


def _cluster_anomaly_days(panel: pd.DataFrame, split: str) -> list[dict[str, Any]]:
    subset = panel.loc[panel["split"].eq(split) & panel["latent_anomaly"]].copy()
    if subset.empty:
        return []
    subset["cluster"] = subset["date"].diff().dt.days.ne(1).cumsum()
    rows = []
    for _, group in subset.groupby("cluster"):
        metrics = sorted(
            {
                str(column)[: -len("_robust_z")]
                for column in group.columns
                if column.endswith("_robust_z")
                and column != "max_abs_robust_z"
                and pd.to_numeric(group[column], errors="coerce").abs().max() >= 3.0
            }
        )
        rows.append(
            {
                "candidate_id": f"LATENT_{split}_{group['date'].min():%Y%m%d}_{group['date'].max():%Y%m%d}",
                "split": split,
                "analysis_role": "TRAIN_DISCOVERY" if split == "TRAIN" else f"{split}_DIAGNOSTIC_ONLY",
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "days": len(group),
                "max_abs_robust_z": float(group["max_abs_robust_z"].max()),
                "max_metric_count": int(group["anomaly_metric_count"].max()),
                "trigger_metrics": "|".join(metrics),
                "documented_event_overlap_days": int(
                    group["documented_event_overlap"].sum()
                ),
                "novel_candidate": bool(
                    not group["documented_event_overlap"].any()
                ),
                "registry_insertion_allowed": False,
                "model_feature_allowed": False,
                "interpretation": "INVESTIGATION_CANDIDATE_NOT_AN_EVENT_LABEL",
            }
        )
    return rows


def detect_latent_periods(
    daily_panel: pd.DataFrame,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = daily_panel.copy()
    metrics = [
        "arrivals",
        "major_delay_rate",
        "critical_delay_rate",
        "median_departure_delay_h",
        "median_total_stay_h",
    ]
    train = panel["split"].eq("TRAIN")
    z_columns = []
    scale_rows = []
    median_daily_arrivals = max(
        1.0, float(pd.to_numeric(panel.loc[train, "arrivals"], errors="coerce").median())
    )
    train_major_rate = float(
        pd.to_numeric(panel.loc[train, "major_delay_calls"], errors="coerce").sum()
        / max(1.0, pd.to_numeric(panel.loc[train, "arrivals"], errors="coerce").sum())
    )
    train_critical_rate = float(
        pd.to_numeric(panel.loc[train, "critical_delay_calls"], errors="coerce").sum()
        / max(1.0, pd.to_numeric(panel.loc[train, "arrivals"], errors="coerce").sum())
    )
    fixed_floors = {
        "arrivals": 1.0,
        "major_delay_rate": max(
            0.01,
            math.sqrt(max(1e-6, train_major_rate * (1.0 - train_major_rate)) / median_daily_arrivals),
        ),
        "critical_delay_rate": max(
            0.005,
            math.sqrt(max(1e-6, train_critical_rate * (1.0 - train_critical_rate)) / median_daily_arrivals),
        ),
        "median_departure_delay_h": 0.25,
        "median_total_stay_h": 0.25,
    }
    for metric in metrics:
        values = pd.to_numeric(panel[metric], errors="coerce")
        dow_median = panel.loc[train].groupby("dow")[metric].median()
        seasonal = panel["dow"].map(dow_median)
        trailing = values.shift(1).rolling(28, min_periods=14).median()
        baseline = trailing.fillna(seasonal)
        residual = values - baseline
        train_residual = residual.loc[train].dropna()
        center = float(train_residual.median()) if len(train_residual) else 0.0
        mad = float((train_residual - center).abs().median()) if len(train_residual) else 1.0
        quantile_scale = (
            float(train_residual.quantile(0.90) - train_residual.quantile(0.10)) / 2.563
            if len(train_residual)
            else 0.0
        )
        scale = max(fixed_floors[metric], 1.4826 * mad, quantile_scale)
        name = f"{metric}_robust_z"
        panel[name] = ((residual - center) / scale).clip(-20.0, 20.0)
        z_columns.append(name)
        scale_rows.append(
            {
                "metric": metric,
                "train_center": center,
                "mad_scale": 1.4826 * mad,
                "quantile_scale": quantile_scale,
                "statistical_floor": fixed_floors[metric],
                "effective_scale": scale,
                "z_clip": 20.0,
            }
        )
    panel["anomaly_metric_count"] = panel[z_columns].abs().ge(3.0).sum(axis=1)
    panel["max_abs_robust_z"] = panel[z_columns].abs().max(axis=1)
    panel["latent_anomaly"] = panel["anomaly_metric_count"].ge(2) | (
        panel["max_abs_robust_z"].ge(6.0) & panel["arrivals"].ge(20)
    )
    panel["documented_event_overlap"] = False
    for event in registry.itertuples(index=False):
        overlap = panel["date"].ge(event.start_at) & panel["date"].lt(event.end_at)
        panel.loc[overlap, "documented_event_overlap"] = True
    candidates = []
    for split in ("TRAIN", "VALID", "TEST"):
        candidates.extend(_cluster_anomaly_days(panel, split))
    return panel, pd.DataFrame(candidates), pd.DataFrame(scale_rows)


def _event_support(calls: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    times = _as_utc(calls["event_time"])
    for event in registry.itertuples(index=False):
        active = times.ge(event.start_at) & times.lt(event.end_at)
        for split in ("TRAIN", "VALID", "TEST"):
            mask = active & calls["split"].eq(split)
            rows.append(
                {
                    "event_id": event.event_id,
                    "event_family": event.event_family,
                    "calendar_role": event.calendar_role,
                    "split": split,
                    "calls_during": int(mask.sum()),
                    "major_delay_calls": int(
                        pd.to_numeric(calls.loc[mask, "target_delay_gt_3h"], errors="coerce").fillna(0).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def _registry_source_report(registry: pd.DataFrame) -> pd.DataFrame:
    return (
        registry.groupby(
            ["source_name", "calendar_role", "source_resolution", "confidence"],
            as_index=False,
        )
        .agg(
            events=("event_id", "nunique"),
            first_event=("start_at", "min"),
            last_event=("end_at", "max"),
            families=("event_family", lambda values: "|".join(sorted(set(values)))),
        )
    )


def build_historical_event_intelligence(
    landmarks: pd.DataFrame,
    registry: pd.DataFrame | None = None,
) -> EventIntelligenceResult:
    _required_landmark_columns(landmarks)
    source = landmarks.copy()
    source["landmark_at"] = _as_utc(source["landmark_at"])
    source["actual_ata"] = _as_utc(source["actual_ata"])
    if not source["dataset_version"].eq(SOURCE_DATASET_VERSION).all():
        raise ValueError("B60C-H received an unexpected source dataset version")
    events = build_event_registry() if registry is None else registry.copy()
    for column in ("start_at", "end_at", "known_at"):
        events[column] = _as_utc(events[column])

    context, feature_columns = build_event_context(source, events)
    calls = _call_level(source)
    exposure = build_call_event_exposure(calls, events)
    associations = estimate_event_associations(exposure)
    daily = build_daily_panel(calls)
    daily_with_anomalies, latent_candidates, anomaly_scales = detect_latent_periods(
        daily, events
    )
    support = _event_support(calls, events)

    duplicate_registry = int(events.duplicated("event_id", keep=False).sum())
    invalid_dates = int(events["start_at"].ge(events["end_at"]).sum())
    missing_sources = int(events["source_uri"].astype(str).str.strip().eq("").sum())
    predictive = events.loc[events["predictive_feature_allowed"]]
    predictive_known_too_late = int(predictive["known_at"].ge(predictive["end_at"]).sum())
    retrospective_marked_predictive = int(
        (events["retrospective_only"] & events["predictive_feature_allowed"]).sum()
    )
    duplicate_context = int(
        context.duplicated(["event_context_version", "port_call_id", "landmark_at"], keep=False).sum()
    )
    target_context_columns = [column for column in context if column.startswith("target_")]
    synthetic_events = int(events["synthetic_event"].fillna(False).sum())
    synthetic_source_rows = int(source["data_origin"].ne("REAL_RETROSPECTIVE").sum())
    lunar_years = set(
        events.loc[events["event_family"].eq("RAMADAN"), "start_at"].dt.year
    )
    required_years = set(range(2020, 2026))
    families_by_split = (
        support.loc[support["calls_during"].gt(0)]
        .groupby("split")["event_family"]
        .nunique()
        .to_dict()
    )
    test_selection_rows = int(
        associations.loc[
            associations.get("split", pd.Series(dtype=str)).eq("TEST"),
            "selection_allowed",
        ].sum()
    ) if not associations.empty else 0
    unstable_anomaly_z = int(
        daily_with_anomalies["max_abs_robust_z"].gt(20.0 + 1e-9).sum()
    )

    gates = [
        ("UNIQUE_EVENT_IDS", duplicate_registry == 0, duplicate_registry),
        ("VALID_EVENT_INTERVALS", invalid_dates == 0, invalid_dates),
        ("EVERY_EVENT_HAS_SOURCE", missing_sources == 0, missing_sources),
        ("PREDICTIVE_EVENT_KNOWN_BEFORE_INTERVAL_END", predictive_known_too_late == 0, predictive_known_too_late),
        ("RETROSPECTIVE_EVENTS_NOT_PREDICTIVE", retrospective_marked_predictive == 0, retrospective_marked_predictive),
        ("RAMADAN_2020_2025_COMPLETE", lunar_years == required_years, len(lunar_years)),
        ("REGISTRY_STOPS_AT_2025", int(events["start_at"].dt.year.max()) <= 2025, int(events["start_at"].dt.year.max())),
        ("NO_PREMATURE_UNITY_DAY_BACKFILL", not events["event_id"].str.contains("UNITY", case=False).any(), 0),
        ("CONTEXT_ROW_PARITY", len(context) == len(source), len(context)),
        ("UNIQUE_CONTEXT_GRAIN", duplicate_context == 0, duplicate_context),
        ("NO_TARGET_IN_EVENT_CONTEXT", len(target_context_columns) == 0, len(target_context_columns)),
        ("REAL_B60C_SOURCE_ONLY", synthetic_source_rows == 0, synthetic_source_rows),
        ("NO_SYNTHETIC_EVENT_LABELS", synthetic_events == 0, synthetic_events),
        ("TRAIN_EVENT_FAMILY_SUPPORT", int(families_by_split.get("TRAIN", 0)) >= 7, int(families_by_split.get("TRAIN", 0))),
        ("VALID_EVENT_FAMILY_SUPPORT", int(families_by_split.get("VALID", 0)) >= 7, int(families_by_split.get("VALID", 0))),
        ("TEST_EVENT_FAMILY_SUPPORT", int(families_by_split.get("TEST", 0)) >= 2, int(families_by_split.get("TEST", 0))),
        ("ASSOCIATION_AUDIT_CREATED", len(associations) > 0, len(associations)),
        ("TEST_NEVER_USED_FOR_SELECTION", test_selection_rows == 0, test_selection_rows),
        ("LATENT_PERIODS_NOT_MODEL_FEATURES", not any("latent" in column for column in feature_columns), 0),
        ("ANOMALY_SCALES_REGULARIZED", unstable_anomaly_z == 0, unstable_anomaly_z),
        ("CAUSAL_CLAIMS_DISABLED", bool(associations.empty or ~associations["causal_claim_allowed"].any()), 0),
    ]
    quality_gates = pd.DataFrame(gates, columns=["gate", "passed", "observed"])
    quality_gates["severity"] = "CRITICAL"
    gates_passed = bool(quality_gates["passed"].all())

    family_summary = (
        events.groupby(["event_family", "calendar_role"], as_index=False)
        .agg(
            events=("event_id", "nunique"),
            first_start=("start_at", "min"),
            last_end=("end_at", "max"),
            confidence=("confidence", lambda values: "|".join(sorted(set(values)))),
        )
    )
    feature_registry = pd.DataFrame(
        [
            {
                "feature": column,
                "role": "RESEARCH_ONLY" if column.startswith("research_") else "PREDICTIVE_POINT_IN_TIME",
                "target_derived": False,
                "available_at_policy": (
                    "RETROSPECTIVE_EXPLANATORY_NOT_PROMOTABLE"
                    if column.startswith("research_")
                    else "EVENT_KNOWN_AT_OR_BEFORE_LANDMARK"
                ),
            }
            for column in feature_columns
        ]
    )
    reports = {
        "01_registry_source_inventory.csv": _registry_source_report(events),
        "02_event_family_inventory.csv": family_summary,
        "03_event_support_by_split.csv": support,
        "04_call_event_associations.csv": associations,
        "05_daily_operational_panel.csv": daily_with_anomalies,
        "06_latent_period_candidates.csv": latent_candidates,
        "06b_anomaly_scale_registry.csv": anomaly_scales,
        "07_event_feature_registry.csv": feature_registry,
        "08_quality_gates.csv": quality_gates,
    }
    significant_train = int(
        associations.loc[
            associations["split"].eq("TRAIN")
            & associations["estimable"]
            & associations["fdr_10pct_signal"]
        ].shape[0]
    ) if not associations.empty else 0
    valid_confirmed = int(
        associations.loc[
            associations["split"].eq("VALID")
            & associations["estimable"]
            & associations["train_direction_confirmed"]
        ].shape[0]
    ) if not associations.empty else 0
    valid_fdr_confirmed = int(
        associations.loc[
            associations["split"].eq("VALID")
            & associations["estimable"]
            & associations["train_direction_confirmed"]
            & associations["fdr_10pct_signal"]
        ].shape[0]
    ) if not associations.empty else 0
    decision_name = (
        "READY_FOR_B60C_V2_EVENT_ENRICHMENT"
        if gates_passed
        else "BLOCKED_EVENT_CONTRACT_REPAIR_REQUIRED"
    )
    decision = clean_json(
        {
            "status": "SUCCESS",
            "decision": decision_name,
            "registry_version": REGISTRY_VERSION,
            "context_version": CONTEXT_VERSION,
            "source_dataset_version": SOURCE_DATASET_VERSION,
            "association_algorithm": "YEAR_MONTH_DOW_MATCHED_V2",
            "latent_detection_algorithm": "REGULARIZED_ROBUST_Z_V2",
            "source_landmarks": len(source),
            "source_calls": calls["port_call_id"].nunique(),
            "registry_events": len(events),
            "event_families": events["event_family"].nunique(),
            "predictive_calendar_events": int(events["predictive_feature_allowed"].sum()),
            "retrospective_events": int(events["retrospective_only"].sum()),
            "event_context_rows": len(context),
            "event_feature_count": len(feature_columns),
            "predictive_feature_count": sum(not name.startswith("research_") for name in feature_columns),
            "research_feature_count": sum(name.startswith("research_") for name in feature_columns),
            "association_rows": len(associations),
            "significant_train_associations_fdr10": significant_train,
            "valid_direction_confirmations": valid_confirmed,
            "valid_fdr_confirmed_associations": valid_fdr_confirmed,
            "latent_period_candidates": len(latent_candidates),
            "latent_candidates_inserted_as_events": 0,
            "synthetic_event_rows": synthetic_events,
            "synthetic_target_rows": 0,
            "test_used_for_selection": False,
            "causal_claim_allowed": False,
            "quality_gates_passed": gates_passed,
            "production_promotion_allowed": False,
            "scientific_scope": "ASSOCIATION_AND_PREDICTIVE_CONTEXT_NOT_CAUSAL_EFFECT",
            "next_block": "B60C_V2_EVENT_ENRICHED_LANDMARK_DATASET",
        }
    )
    return EventIntelligenceResult(
        registry=events,
        context=context,
        reports=reports,
        decision=decision,
        feature_columns=feature_columns,
    )
