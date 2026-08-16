from __future__ import annotations

import csv
import io
import json
import os
import re
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("/tmp/b55a_source_audit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INVENTORY_PATH = OUTPUT_DIR / "01_minio_source_inventory.csv"
COLUMN_STATS_PATH = OUTPUT_DIR / "02_recoverable_column_statistics.csv"
SUMMARY_PATH = OUTPUT_DIR / "03_b55a_source_audit_summary.json"

GOLD_BUCKET = "gold-maritime"
GOLD_KEY = (
    "datasets/b54f/version=1/"
    "port_call_one_row_model_ready_no_split_v1.parquet"
)

MAX_OBJECTS_TO_INSPECT = 60
MAX_PARQUET_DOWNLOAD_BYTES = 500 * 1024 * 1024
MAX_STATS_BYTES = 150 * 1024 * 1024
CSV_HEADER_BYTES = 2 * 1024 * 1024


PRIORITY_COLUMNS = {
    "TYPE_UNITE": {
        "TYPEUNITE",
        "UNITTYPE",
        "TYPEUNIT",
    },
    "SS_TYPE_UNITE": {
        "SSTYPEUNITE",
        "SOUSTYPEUNITE",
        "UNITSUBTYPE",
        "SUBTYPEUNITE",
    },
    "DECLARANT": {
        "DECLARANT",
        "DECLARANTID",
        "DECLARANTCODE",
    },
    "VIDE_PLEIN": {
        "VIDEPLEIN",
        "FULLEMPTY",
        "EMPTYFULL",
        "LOADSTATUS",
    },
    "NATURE_MARCHANDISE": {
        "NATUREMARCHANDISE",
        "CARGONATURE",
        "GOODSNATURE",
        "COMMODITY",
        "COMMODITYTYPE",
    },
    "MATIERE_DANGER": {
        "MATIEREDANGER",
        "MATIEREDANGEREUSE",
        "DANGEROUSGOODS",
        "HAZARDOUSMATERIAL",
        "HAZMAT",
    },
    "POIDS": {
        "POIDS",
        "WEIGHT",
        "GROSSWEIGHT",
        "NETWEIGHT",
    },
    "IS_GROUPAGE": {
        "ISGROUPAGE",
        "GROUPAGE",
        "CONSOLIDATED",
    },
    "COULOIR": {
        "COULOIR",
        "CORRIDOR",
        "LANE",
        "TRADELANE",
    },
    "TERMINAL": {
        "TERMINAL",
        "TERMINALCODE",
        "TERMINALNAME",
    },
    "CARGO_TYPE": {
        "CARGOTYPE",
        "TYPECARGO",
        "GOODSTYPE",
    },
    "VESSEL_TYPE": {
        "VESSELTYPE",
        "SHIPTYPE",
        "TYPENAVIRE",
        "NAVIRETYPE",
    },
    "VESSEL_SUBTYPE": {
        "VESSELSUBTYPE",
        "SHIPSUBTYPE",
        "SSTYPENAVIRE",
    },
    "MMSI": {
        "MMSI",
    },
    "IMO": {
        "IMO",
        "IMONUMBER",
        "NUMEROIMO",
    },
    "PORT_CODE": {
        "PORTCODE",
        "UNLOCODE",
        "CODEPORT",
    },
    "ORIGIN_PORT": {
        "ORIGINPORT",
        "PORTORIGINE",
        "PREVIOUSPORT",
        "LASTPORT",
    },
    "DESTINATION_PORT": {
        "DESTINATIONPORT",
        "PORTDESTINATION",
        "NEXTPORT",
    },
    "ETA": {
        "ETA",
        "PLANNEDETA",
        "DATEETA",
        "EXPECTEDARRIVAL",
    },
    "ATA": {
        "ATA",
        "ACTUALATA",
        "ACTUALARRIVAL",
        "DATEARRIVEE",
    },
    "ETD": {
        "ETD",
        "PLANNEDETD",
        "EXPECTEDDEPARTURE",
    },
    "ATD": {
        "ATD",
        "ACTUALATD",
        "ACTUALDEPARTURE",
        "DATEDEPART",
    },
}

JOIN_KEYS = {
    "PORT_CALL_ID": {
        "PORTCALLID",
        "CALLID",
        "ESCALEID",
        "NUMEROESCALE",
    },
    "SOURCE_RECORD_ID": {
        "SOURCERECORDID",
        "RECORDID",
        "ROWID",
    },
    "IMO": {
        "IMO",
        "IMONUMBER",
        "NUMEROIMO",
    },
    "MMSI": {
        "MMSI",
    },
    "PLANNED_ETA": {
        "PLANNEDETA",
        "ETA",
        "DATEETA",
        "EXPECTEDARRIVAL",
    },
    "ACTUAL_ATA": {
        "ACTUALATA",
        "ATA",
        "DATEARRIVEE",
        "ACTUALARRIVAL",
    },
}

OBJECT_KEYWORDS = {
    "tir": 15,
    "port_call": 12,
    "port-call": 12,
    "portcall": 12,
    "maritime": 8,
    "silver": 7,
    "bronze": 6,
    "dataset": 5,
    "data1": 12,
    "data6": 10,
    "data7": 10,
    "classique": 8,
    "dynamic": 6,
    "dynamique": 6,
    "container": 5,
    "cargo": 5,
    "vessel": 4,
}


# =============================================================================
# UTILITAIRES
# =============================================================================

def normalize_name(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def build_s3_client():
    endpoint = (
        os.getenv("S3_ENDPOINT_URL")
        or os.getenv("MINIO_ENDPOINT_URL")
        or os.getenv("MINIO_ENDPOINT")
        or "http://minio:9000"
    )

    access_key = (
        os.getenv("AWS_ACCESS_KEY_ID")
        or os.getenv("MINIO_ACCESS_KEY")
        or os.getenv("MINIO_ROOT_USER")
    )

    secret_key = (
        os.getenv("AWS_SECRET_ACCESS_KEY")
        or os.getenv("MINIO_SECRET_KEY")
        or os.getenv("MINIO_ROOT_PASSWORD")
    )

    arguments = {
        "endpoint_url": endpoint,
    }

    if access_key and secret_key:
        arguments["aws_access_key_id"] = access_key
        arguments["aws_secret_access_key"] = secret_key

    return boto3.client("s3", **arguments)


def list_all_objects(client) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    for bucket_info in client.list_buckets().get("Buckets", []):
        bucket = bucket_info["Name"]
        continuation_token = None

        while True:
            arguments: dict[str, Any] = {
                "Bucket": bucket,
                "MaxKeys": 1000,
            }

            if continuation_token:
                arguments["ContinuationToken"] = continuation_token

            response = client.list_objects_v2(**arguments)

            for item in response.get("Contents", []):
                results.append(
                    {
                        "bucket": bucket,
                        "key": item["Key"],
                        "size": int(item["Size"]),
                        "last_modified": item["LastModified"],
                    }
                )

            if not response.get("IsTruncated"):
                break

            continuation_token = response["NextContinuationToken"]

    return results


def object_score(item: dict[str, Any]) -> int:
    key = item["key"].lower()
    score = 0

    for keyword, weight in OBJECT_KEYWORDS.items():
        if keyword in key:
            score += weight

    extension = Path(key).suffix.lower()

    if extension == ".parquet":
        score += 8
    elif extension in {".csv", ".txt"}:
        score += 5
    elif extension in {".json", ".jsonl"}:
        score += 2

    if "report" in key or "prediction" in key or "model" in key:
        score -= 8

    if "b54f" in key:
        score += 3

    return score


def identify_columns(
    columns: list[str],
    mapping: dict[str, set[str]],
) -> dict[str, str]:
    normalized_to_original: dict[str, str] = {}

    for column in columns:
        normalized_to_original.setdefault(
            normalize_name(column),
            str(column),
        )

    found: dict[str, str] = {}

    for canonical, aliases in mapping.items():
        for alias in aliases:
            if alias in normalized_to_original:
                found[canonical] = normalized_to_original[alias]
                break

    return found


def read_csv_columns(client, bucket: str, key: str) -> list[str]:
    response = client.get_object(
        Bucket=bucket,
        Key=key,
        Range=f"bytes=0-{CSV_HEADER_BYTES - 1}",
    )

    raw = response["Body"].read()

    text = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    if not text:
        return []

    first_lines = "\n".join(text.splitlines()[:20])

    try:
        dialect = csv.Sniffer().sniff(
            first_lines,
            delimiters=",;\t|",
        )
        separator = dialect.delimiter
    except csv.Error:
        separator = ","

    reader = csv.reader(io.StringIO(text), delimiter=separator)
    header = next(reader, [])

    return [
        str(column).strip()
        for column in header
        if str(column).strip()
    ]


def inspect_parquet(
    client,
    bucket: str,
    key: str,
    size: int,
) -> tuple[list[str], int | None, str | None, str | None]:
    if size > MAX_PARQUET_DOWNLOAD_BYTES:
        return (
            [],
            None,
            None,
            (
                "SKIPPED_TOO_LARGE:"
                f"{size / (1024 ** 2):.1f}MB"
            ),
        )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".parquet",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name

        client.download_file(
            bucket,
            key,
            temporary_path,
        )

        parquet_file = pq.ParquetFile(temporary_path)
        columns = parquet_file.schema.names
        row_count = (
            parquet_file.metadata.num_rows
            if parquet_file.metadata is not None
            else None
        )

        return columns, row_count, temporary_path, None

    except Exception as exc:
        if temporary_path and Path(temporary_path).exists():
            Path(temporary_path).unlink(missing_ok=True)

        return [], None, None, f"{type(exc).__name__}: {exc}"


def safe_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def calculate_column_stats(
    parquet_path: str,
    bucket: str,
    key: str,
    found_columns: dict[str, str],
    size: int,
) -> list[dict[str, Any]]:
    if size > MAX_STATS_BYTES or not found_columns:
        return []

    selected_columns = sorted(set(found_columns.values()))

    try:
        table = pq.read_table(
            parquet_path,
            columns=selected_columns,
        )
        frame = table.to_pandas()
    except Exception:
        return []

    stats: list[dict[str, Any]] = []

    for canonical, actual_column in found_columns.items():
        series = frame[actual_column]

        non_null = int(series.notna().sum())
        total = int(len(series))
        non_null_pct = (
            100.0 * non_null / total
            if total
            else None
        )

        try:
            unique_count = int(series.nunique(dropna=True))
        except Exception:
            unique_count = None

        top_values = []

        try:
            counts = series.dropna().astype(str).value_counts().head(5)

            top_values = [
                {
                    "value": str(index),
                    "count": int(value),
                }
                for index, value in counts.items()
            ]
        except Exception:
            pass

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        numeric_non_null = int(numeric.notna().sum())

        stats.append(
            {
                "bucket": bucket,
                "key": key,
                "canonical_column": canonical,
                "actual_column": actual_column,
                "dtype": str(series.dtype),
                "rows": total,
                "non_null_rows": non_null,
                "non_null_pct": non_null_pct,
                "unique_count": unique_count,
                "numeric_rows": numeric_non_null,
                "numeric_min": (
                    safe_scalar(numeric.min())
                    if numeric_non_null
                    else None
                ),
                "numeric_max": (
                    safe_scalar(numeric.max())
                    if numeric_non_null
                    else None
                ),
                "numeric_mean": (
                    safe_scalar(numeric.mean())
                    if numeric_non_null
                    else None
                ),
                "top_values_json": json.dumps(
                    top_values,
                    ensure_ascii=False,
                ),
            }
        )

    return stats


# =============================================================================
# EXÉCUTION
# =============================================================================

client = build_s3_client()

all_objects = list_all_objects(client)

supported_extensions = {
    ".parquet",
    ".csv",
    ".txt",
}

candidates = [
    item
    for item in all_objects
    if Path(item["key"]).suffix.lower() in supported_extensions
]

for item in candidates:
    item["score"] = object_score(item)

candidates.sort(
    key=lambda item: (
        item["score"],
        item["last_modified"],
    ),
    reverse=True,
)

candidates = candidates[:MAX_OBJECTS_TO_INSPECT]

inventory_rows: list[dict[str, Any]] = []
statistics_rows: list[dict[str, Any]] = []

print("=" * 120)
print("B55A — EXPLORATION DES SOURCES MINIO")
print("=" * 120)
print(f"Objets MinIO totaux       : {len(all_objects)}")
print(f"Candidats inspectés       : {len(candidates)}")
print()

for position, item in enumerate(candidates, start=1):
    bucket = item["bucket"]
    key = item["key"]
    size = item["size"]
    extension = Path(key).suffix.lower()

    print(
        f"[{position:02d}/{len(candidates):02d}] "
        f"s3://{bucket}/{key} "
        f"({size / (1024 ** 2):.1f} MB)"
    )

    columns: list[str] = []
    row_count = None
    parquet_path = None
    status = "OK"
    error = None

    try:
        if extension == ".parquet":
            columns, row_count, parquet_path, error = inspect_parquet(
                client,
                bucket,
                key,
                size,
            )

            if error:
                status = error

        elif extension in {".csv", ".txt"}:
            columns = read_csv_columns(
                client,
                bucket,
                key,
            )

            if not columns:
                status = "HEADER_NOT_READABLE"

    except Exception as exc:
        status = "ERROR"
        error = f"{type(exc).__name__}: {exc}"

    found_business = identify_columns(
        columns,
        PRIORITY_COLUMNS,
    )

    found_join_keys = identify_columns(
        columns,
        JOIN_KEYS,
    )

    missing_priority = sorted(
        set(PRIORITY_COLUMNS) - set(found_business)
    )

    if parquet_path:
        statistics_rows.extend(
            calculate_column_stats(
                parquet_path=parquet_path,
                bucket=bucket,
                key=key,
                found_columns=found_business,
                size=size,
            )
        )

        Path(parquet_path).unlink(missing_ok=True)

    inventory_rows.append(
        {
            "score": item["score"],
            "bucket": bucket,
            "key": key,
            "uri": f"s3://{bucket}/{key}",
            "extension": extension,
            "size_mb": round(size / (1024 ** 2), 3),
            "last_modified_utc": item["last_modified"].isoformat(),
            "row_count": row_count,
            "column_count": len(columns),
            "found_business_count": len(found_business),
            "found_business_columns": "|".join(
                f"{canonical}={actual}"
                for canonical, actual in sorted(found_business.items())
            ),
            "found_join_key_count": len(found_join_keys),
            "found_join_keys": "|".join(
                f"{canonical}={actual}"
                for canonical, actual in sorted(found_join_keys.items())
            ),
            "missing_priority_columns": "|".join(missing_priority),
            "status": status,
            "error": error,
        }
    )


# =============================================================================
# AUDIT DU GOLD COURANT
# =============================================================================

gold_columns: list[str] = []
gold_rows = None
gold_last_modified = None
gold_error = None

try:
    gold_metadata = client.head_object(
        Bucket=GOLD_BUCKET,
        Key=GOLD_KEY,
    )

    gold_last_modified = gold_metadata["LastModified"].isoformat()

    columns, row_count, temporary_path, error = inspect_parquet(
        client,
        GOLD_BUCKET,
        GOLD_KEY,
        int(gold_metadata["ContentLength"]),
    )

    gold_columns = columns
    gold_rows = row_count
    gold_error = error

    if temporary_path:
        Path(temporary_path).unlink(missing_ok=True)

except Exception as exc:
    gold_error = f"{type(exc).__name__}: {exc}"

gold_business = identify_columns(
    gold_columns,
    PRIORITY_COLUMNS,
)

gold_join_keys = identify_columns(
    gold_columns,
    JOIN_KEYS,
)

gold_missing_priority = sorted(
    set(PRIORITY_COLUMNS) - set(gold_business)
)


# =============================================================================
# CLASSEMENT DES MEILLEURES SOURCES
# =============================================================================

useful_rows = [
    row
    for row in inventory_rows
    if row["found_business_count"] > 0
]

useful_rows.sort(
    key=lambda row: (
        row["found_business_count"],
        row["found_join_key_count"],
        row["score"],
    ),
    reverse=True,
)

best_sources = useful_rows[:15]

recoverability = Counter()

for row in useful_rows:
    assignments = row["found_business_columns"].split("|")

    for assignment in assignments:
        if "=" in assignment:
            canonical = assignment.split("=", 1)[0]
            recoverability[canonical] += 1


# =============================================================================
# ÉCRITURE DES RAPPORTS
# =============================================================================

inventory_frame = pd.DataFrame(inventory_rows)

if not inventory_frame.empty:
    inventory_frame.to_csv(
        INVENTORY_PATH,
        index=False,
        encoding="utf-8",
    )
else:
    pd.DataFrame(
        columns=[
            "score",
            "bucket",
            "key",
            "uri",
            "status",
        ]
    ).to_csv(
        INVENTORY_PATH,
        index=False,
        encoding="utf-8",
    )

statistics_frame = pd.DataFrame(statistics_rows)

if not statistics_frame.empty:
    statistics_frame.to_csv(
        COLUMN_STATS_PATH,
        index=False,
        encoding="utf-8",
    )
else:
    pd.DataFrame(
        columns=[
            "bucket",
            "key",
            "canonical_column",
            "actual_column",
            "non_null_pct",
        ]
    ).to_csv(
        COLUMN_STATS_PATH,
        index=False,
        encoding="utf-8",
    )

summary = {
    "audit_version": "b55a-source-feature-recovery-audit-v1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "read_only_audit": True,
    "minio_object_count": len(all_objects),
    "inspected_object_count": len(inventory_rows),
    "useful_source_count": len(useful_rows),
    "priority_feature_count": len(PRIORITY_COLUMNS),
    "gold": {
        "uri": f"s3://{GOLD_BUCKET}/{GOLD_KEY}",
        "last_modified_utc": gold_last_modified,
        "rows": gold_rows,
        "columns": len(gold_columns),
        "business_columns_already_present": gold_business,
        "join_keys_present": gold_join_keys,
        "missing_priority_columns": gold_missing_priority,
        "error": gold_error,
    },
    "recoverability_source_counts": dict(
        sorted(
            recoverability.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ),
    "best_sources": best_sources,
    "decision_rules": {
        "direct_join_candidate": (
            "Object contains PORT_CALL_ID or SOURCE_RECORD_ID."
        ),
        "composite_join_candidate": (
            "Object contains IMO plus ETA/ATA timestamp."
        ),
        "unsafe_without_review": (
            "Object has business columns but no stable call identifier "
            "or timestamp."
        ),
    },
    "recommended_next_step": (
        "Select the best source, validate call-level join coverage, "
        "then create leakage-safe call aggregates."
    ),
}

SUMMARY_PATH.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)


# =============================================================================
# SORTIE CONSOLE
# =============================================================================

print("\n" + "=" * 120)
print("GOLD ACTUEL")
print("=" * 120)
print(f"URI                       : s3://{GOLD_BUCKET}/{GOLD_KEY}")
print(f"LastModified              : {gold_last_modified}")
print(f"Lignes                    : {gold_rows}")
print(f"Colonnes                  : {len(gold_columns)}")
print(
    "Variables métier présentes :",
    sorted(gold_business),
)
print(
    "Variables métier absentes  :",
    gold_missing_priority,
)

print("\n" + "=" * 120)
print("MEILLEURES SOURCES TROUVÉES")
print("=" * 120)

if not best_sources:
    print("Aucune source métier exploitable détectée automatiquement.")
else:
    for index, row in enumerate(best_sources, start=1):
        print(f"\n#{index}")
        print("URI              :", row["uri"])
        print("Lignes           :", row["row_count"])
        print("Colonnes         :", row["column_count"])
        print("Variables métier :", row["found_business_columns"])
        print("Clés de jointure :", row["found_join_keys"])
        print("Statut            :", row["status"])

print("\n" + "=" * 120)
print("RÉCUPÉRABILITÉ DES VARIABLES")
print("=" * 120)

for column, source_count in sorted(
    recoverability.items(),
    key=lambda item: (-item[1], item[0]),
):
    print(f"{column:25} : {source_count} source(s)")

print("\n" + "=" * 120)
print("RAPPORTS")
print("=" * 120)
print(INVENTORY_PATH)
print(COLUMN_STATS_PATH)
print(SUMMARY_PATH)

if not useful_rows:
    raise SystemExit(
        "Audit terminé, mais aucune source métier exploitable "
        "n'a été trouvée automatiquement."
    )