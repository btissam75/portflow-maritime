from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import os
import random
import signal
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import boto3
import psycopg2
from psycopg2.extras import execute_values
import websockets


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
LOGGER = logging.getLogger("aisstream-collector")

AIS_URL = os.getenv("AISSTREAM_URL", "wss://stream.aisstream.io/v0/stream")
AIS_API_KEY = os.environ["AISSTREAM_API_KEY"]
LAT_MIN = float(os.getenv("AIS_LAT_MIN", "35.65"))
LAT_MAX = float(os.getenv("AIS_LAT_MAX", "36.25"))
LON_MIN = float(os.getenv("AIS_LON_MIN", "-6.10"))
LON_MAX = float(os.getenv("AIS_LON_MAX", "-5.00"))
FLUSH_SECONDS = int(os.getenv("AIS_FLUSH_SECONDS", "60"))
MAX_BATCH_MESSAGES = int(os.getenv("AIS_MAX_BATCH_MESSAGES", "1000"))
BRONZE_BUCKET = os.getenv("AIS_BRONZE_BUCKET", "bronze-maritime")
COLLECTOR_VERSION = os.getenv("AIS_COLLECTOR_VERSION", "b56d1-aisstream-v1")
RETRY_INITIAL_SECONDS = int(os.getenv("AIS_RETRY_INITIAL_SECONDS", "15"))
RETRY_MAX_SECONDS = int(os.getenv("AIS_RETRY_MAX_SECONDS", "300"))
RETRY_503_MIN_SECONDS = int(os.getenv("AIS_RETRY_503_MIN_SECONDS", "60"))
STABLE_CONNECTION_SECONDS = int(os.getenv("AIS_STABLE_CONNECTION_SECONDS", "120"))

POSITION_TYPES = {
    "PositionReport",
    "StandardClassBPositionReport",
    "ExtendedClassBPositionReport",
}
SUBSCRIBED_TYPES = sorted(POSITION_TYPES | {"ShipStaticData", "StaticDataReport"})


@dataclass
class VesselStaticState:
    imo: int | None = None
    name: str | None = None
    destination: str | None = None
    reported_eta: datetime | None = None
    updated_at: datetime | None = None


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).replace("@", " ").strip()
    return text or None


def clean_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def clean_float(value: Any, invalid: set[float] | None = None) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if invalid and parsed in invalid:
        return None
    return parsed


def parse_metadata_time(value: Any, fallback: datetime) -> datetime:
    if not value:
        return fallback
    text = str(value).strip()
    formats = (
        "%Y-%m-%d %H:%M:%S.%f %z UTC",
        "%Y-%m-%d %H:%M:%S %z UTC",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S%z",
    )
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    LOGGER.warning("Could not parse AIS metadata time %r; using receipt time", text)
    return fallback


def infer_reported_eta(eta: Any, available_at: datetime) -> datetime | None:
    if not isinstance(eta, dict):
        return None
    try:
        month = int(eta.get("Month") or 0)
        day = int(eta.get("Day") or 0)
        hour = int(eta.get("Hour") or 0)
        minute = int(eta.get("Minute") or 0)
        if not (1 <= month <= 12 and 1 <= day <= 31):
            return None
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        candidate = datetime(
            available_at.year, month, day, hour, minute, tzinfo=timezone.utc
        )
        if candidate < available_at - timedelta(days=1):
            candidate = candidate.replace(year=candidate.year + 1)
        return candidate
    except (TypeError, ValueError):
        return None


def body_for(message: dict[str, Any], message_type: str) -> dict[str, Any]:
    body = message.get("Message", {}).get(message_type, {})
    return body if isinstance(body, dict) else {}


def update_static_cache(
    message: dict[str, Any],
    message_type: str,
    available_at: datetime,
    cache: dict[int, VesselStaticState],
) -> None:
    metadata = message.get("MetaData") or message.get("Metadata") or {}
    body = body_for(message, message_type)
    mmsi = clean_int(metadata.get("MMSI") or body.get("UserID"))
    if mmsi is None:
        return

    current = cache.get(mmsi, VesselStaticState())
    if message_type == "ShipStaticData":
        current.imo = clean_int(body.get("ImoNumber")) or current.imo
        current.name = clean_text(body.get("Name")) or current.name
        current.destination = clean_text(body.get("Destination")) or current.destination
        current.reported_eta = infer_reported_eta(body.get("Eta"), available_at)
    elif message_type == "StaticDataReport":
        report_a = body.get("ReportA") or {}
        current.name = clean_text(report_a.get("Name")) or current.name

    current.name = clean_text(metadata.get("ShipName")) or current.name
    current.updated_at = available_at
    cache[mmsi] = current


def normalize_position(
    message: dict[str, Any],
    message_type: str,
    available_at: datetime,
    cache: dict[int, VesselStaticState],
) -> tuple[Any, ...] | None:
    metadata = message.get("MetaData") or message.get("Metadata") or {}
    body = body_for(message, message_type)
    mmsi = clean_int(metadata.get("MMSI") or body.get("UserID"))
    if mmsi is None:
        return None

    latitude = clean_float(body.get("Latitude", metadata.get("latitude")))
    longitude = clean_float(body.get("Longitude", metadata.get("longitude")))
    if latitude is None or longitude is None:
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None

    observed_at = parse_metadata_time(metadata.get("time_utc"), available_at)
    static = cache.get(mmsi, VesselStaticState())
    navigation_status = clean_text(body.get("NavigationalStatus"))

    return (
        observed_at,
        mmsi,
        static.imo,
        latitude,
        longitude,
        clean_float(body.get("Sog"), {102.3, 102.4}),
        clean_float(body.get("Cog"), {360.0}),
        clean_float(body.get("TrueHeading"), {511.0}),
        navigation_status,
        static.destination,
        static.reported_eta,
    )


def database_connection():
    return psycopg2.connect(
        host=os.getenv("SMART_PORT_DB_HOST", "timescaledb"),
        port=int(os.getenv("SMART_PORT_DB_PORT", "5432")),
        dbname=os.environ["SMART_PORT_DB_NAME"],
        user=os.environ["SMART_PORT_DB_USER"],
        password=os.environ["SMART_PORT_DB_PASSWORD"],
        connect_timeout=15,
    )


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["SMART_PORT_S3_ENDPOINT"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def ensure_schema() -> None:
    sql = """
    CREATE SCHEMA IF NOT EXISTS lineage;
    CREATE TABLE IF NOT EXISTS lineage.ais_message_receipt (
        message_sha256 text PRIMARY KEY,
        message_type text NOT NULL,
        mmsi bigint,
        event_time timestamptz,
        available_at timestamptz NOT NULL,
        ingestion_run_id uuid REFERENCES audit.ingestion_run(run_id),
        bronze_uri text NOT NULL,
        collector_version text NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ais_message_receipt_available_idx
        ON lineage.ais_message_receipt (available_at DESC);
    CREATE INDEX IF NOT EXISTS ais_message_receipt_mmsi_idx
        ON lineage.ais_message_receipt (mmsi, event_time DESC);
    DROP TRIGGER IF EXISTS ais_message_receipt_append_only_guard
        ON lineage.ais_message_receipt;
    CREATE TRIGGER ais_message_receipt_append_only_guard
    BEFORE UPDATE OR DELETE ON lineage.ais_message_receipt
    FOR EACH ROW EXECUTE FUNCTION lineage.reject_append_only_mutation();
    """
    with database_connection() as conn, conn.cursor() as cur:
        cur.execute(sql)


def persist_batch(messages: list[dict[str, Any]], positions: list[tuple[Any, ...]]) -> None:
    if not messages:
        return

    run_id = uuid4()
    started_at = utc_now()
    raw_lines = b"\n".join(
        json.dumps(item, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        for item in messages
    ) + b"\n"
    compressed = gzip.compress(raw_lines, mtime=0)
    checksum = hashlib.sha256(compressed).hexdigest()
    key = (
        "ais/aisstream/tanger_med/"
        f"year={started_at:%Y}/month={started_at:%m}/day={started_at:%d}/"
        f"hour={started_at:%H}/batch_{started_at:%Y%m%dT%H%M%S}_{run_id}.jsonl.gz"
    )
    bronze_uri = f"s3://{BRONZE_BUCKET}/{key}"

    s3_client().put_object(
        Bucket=BRONZE_BUCKET,
        Key=key,
        Body=compressed,
        ContentType="application/x-ndjson",
        ContentEncoding="gzip",
        Metadata={
            "collector-version": COLLECTOR_VERSION,
            "sha256": checksum,
            "received-count": str(len(messages)),
        },
    )

    try:
        with database_connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO audit.ingestion_run (
                    run_id, source_name, dataset_name, started_at,
                    status, object_uri, checksum, metadata
                ) VALUES (
                    %s, 'aisstream', 'tanger_med_live_ais', %s,
                    'RUNNING', %s, %s, %s::jsonb
                )
                """,
                (
                    str(run_id),
                    started_at,
                    bronze_uri,
                    checksum,
                    json.dumps(
                        {
                            "collector_version": COLLECTOR_VERSION,
                            "bounding_box": [LAT_MIN, LON_MIN, LAT_MAX, LON_MAX],
                            "raw_messages": len(messages),
                        }
                    ),
                ),
            )

            inserted_positions = 0
            if positions:
                values = [position + (str(run_id),) for position in positions]
                execute_values(
                    cur,
                    """
                    INSERT INTO core.vessel_position (
                        observed_at, mmsi, imo, latitude, longitude,
                        speed_over_ground_kn, course_over_ground_deg,
                        heading_deg, navigation_status, destination,
                        reported_eta, ingestion_run_id
                    ) VALUES %s
                    ON CONFLICT (observed_at, mmsi) DO NOTHING
                    """,
                    values,
                    page_size=500,
                )
                inserted_positions = cur.rowcount

            receipt_rows = []
            for item in messages:
                raw = item["raw"]
                message = item["message"]
                metadata = message.get("MetaData") or message.get("Metadata") or {}
                body = body_for(message, item["message_type"])
                mmsi = clean_int(metadata.get("MMSI") or body.get("UserID"))
                event_time = parse_metadata_time(
                    metadata.get("time_utc"), item["available_at"]
                )
                receipt_rows.append(
                    (
                        hashlib.sha256(
                            (
                                raw
                                + "|available_at="
                                + item["available_at"].isoformat()
                            ).encode("utf-8")
                        ).hexdigest(),
                        item["message_type"],
                        mmsi,
                        event_time,
                        item["available_at"],
                        str(run_id),
                        bronze_uri,
                        COLLECTOR_VERSION,
                    )
                )
            execute_values(
                cur,
                """
                INSERT INTO lineage.ais_message_receipt (
                    message_sha256, message_type, mmsi, event_time,
                    available_at, ingestion_run_id, bronze_uri, collector_version
                ) VALUES %s
                ON CONFLICT (message_sha256) DO NOTHING
                """,
                receipt_rows,
                page_size=500,
            )

            cur.execute(
                """
                UPDATE audit.ingestion_run
                   SET status='SUCCESS', finished_at=clock_timestamp(), row_count=%s,
                       metadata=metadata || %s::jsonb
                 WHERE run_id=%s
                """,
                (
                    inserted_positions,
                    json.dumps(
                        {
                            "normalized_positions": len(positions),
                            "inserted_positions": inserted_positions,
                        }
                    ),
                    str(run_id),
                ),
            )
        LOGGER.info(
            "Persisted batch raw=%s normalized=%s inserted=%s uri=%s",
            len(messages),
            len(positions),
            inserted_positions,
            bronze_uri,
        )
    except Exception:
        LOGGER.exception("Batch database persistence failed; raw Bronze is preserved at %s", bronze_uri)
        raise


async def collect_forever() -> None:
    ensure_schema()
    static_cache: dict[int, VesselStaticState] = {}
    retry_seconds = RETRY_INITIAL_SECONDS

    while True:
        messages: list[dict[str, Any]] = []
        positions: list[tuple[Any, ...]] = []
        loop = asyncio.get_running_loop()
        last_flush = loop.time()
        connection_started: float | None = None
        received_on_connection = 0
        try:
            LOGGER.info(
                "Connecting to AISStream bbox=[%s,%s]-[%s,%s]",
                LAT_MIN,
                LON_MIN,
                LAT_MAX,
                LON_MAX,
            )
            async with websockets.connect(
                AIS_URL,
                open_timeout=30,
                ping_interval=30,
                ping_timeout=30,
                close_timeout=10,
                max_size=2**20,
            ) as websocket:
                connection_started = loop.time()
                await websocket.send(
                    json.dumps(
                        {
                            "APIKey": AIS_API_KEY,
                            "BoundingBoxes": [
                                [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]]
                            ],
                            "FilterMessageTypes": SUBSCRIBED_TYPES,
                        }
                    )
                )
                LOGGER.info(
                    "AISStream subscription sent; waiting for the first AIS message"
                )

                while True:
                    timeout = max(
                        1,
                        FLUSH_SECONDS
                        - (loop.time() - last_flush),
                    )
                    try:
                        raw = await asyncio.wait_for(websocket.recv(), timeout=timeout)
                    except asyncio.TimeoutError:
                        if messages:
                            await asyncio.to_thread(persist_batch, messages, positions)
                            messages, positions = [], []
                        last_flush = loop.time()
                        continue

                    available_at = utc_now()
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        LOGGER.warning("Ignoring a non-JSON AISStream frame")
                        continue
                    if "error" in parsed:
                        raise RuntimeError(f"AISStream error: {parsed['error']}")
                    message_type = parsed.get("MessageType")
                    if message_type not in SUBSCRIBED_TYPES:
                        continue

                    received_on_connection += 1
                    if received_on_connection == 1:
                        LOGGER.info(
                            "First AIS message received after %.1fs; type=%s",
                            loop.time() - connection_started,
                            message_type,
                        )

                    update_static_cache(parsed, message_type, available_at, static_cache)
                    position = None
                    if message_type in POSITION_TYPES:
                        position = normalize_position(
                            parsed, message_type, available_at, static_cache
                        )
                        if position is not None:
                            positions.append(position)

                    messages.append(
                        {
                            "message_type": message_type,
                            "available_at": available_at,
                            "message": parsed,
                            "raw": raw,
                        }
                    )
                    if len(messages) >= MAX_BATCH_MESSAGES:
                        await asyncio.to_thread(persist_batch, messages, positions)
                        messages, positions = [], []
                        last_flush = loop.time()
        except asyncio.CancelledError:
            if messages:
                await asyncio.to_thread(persist_batch, messages, positions)
            raise
        except Exception as exc:
            connection_age = (
                loop.time() - connection_started
                if connection_started is not None
                else 0.0
            )
            if messages:
                try:
                    await asyncio.to_thread(persist_batch, messages, positions)
                except Exception:
                    LOGGER.exception("Could not flush buffered AIS messages")

            # A short-lived accepted socket isn't a stable recovery. Keeping the
            # accumulated backoff prevents reconnect storms against the beta API.
            if connection_age >= STABLE_CONNECTION_SECONDS:
                retry_seconds = RETRY_INITIAL_SECONDS
            if "503" in str(exc):
                retry_seconds = max(retry_seconds, RETRY_503_MIN_SECONDS)

            jitter = random.uniform(0, max(1.0, retry_seconds * 0.20))
            delay = retry_seconds + jitter
            LOGGER.error(
                "AISStream connection failed after %.1fs and %s messages: %s; "
                "retrying in %.1fs",
                connection_age,
                received_on_connection,
                exc,
                delay,
            )
            await asyncio.sleep(delay)
            retry_seconds = min(retry_seconds * 2, RETRY_MAX_SECONDS)


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(collect_forever())

    def stop() -> None:
        if not task.done():
            task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop)
        except NotImplementedError:
            pass

    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        LOGGER.info("AIS collector stopped")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
