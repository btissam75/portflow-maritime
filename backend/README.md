# Smart Port Maritime Backend

This directory contains the data, machine-learning, orchestration and API layers of the PortFlow
Maritime monorepo. The canonical setup, architecture, governance and operations reference is
[`../PLATFORM_ENGINEERING_GUIDE.md`](../PLATFORM_ENGINEERING_GUIDE.md).

Backend foundation for the maritime delay, remaining-time, capacity and metocean platform:

```text
Copernicus / ECMWF / AIS / port calls
        -> Airflow ingestion
        -> MinIO S3 data lake
        -> xarray maritime feature builder
        -> TimescaleDB
        -> forecast models and MLflow
        -> FastAPI / React / Grafana (next phases)
```

## Phase 1 services

| Service | Role | Local URL |
|---|---|---|
| TimescaleDB | Time-series and port-call database | `localhost:5432` |
| MinIO API | S3-compatible data lake | `http://localhost:9000` |
| MinIO Console | Data lake administration | `http://localhost:9001` |
| MLflow | Experiments, metrics and model registry | `http://localhost:5000` |
| Grafana | Operational and data monitoring | `http://localhost:3001` |
| Airflow | Ingestion and feature workflow orchestration | `http://localhost:8080` |
| Feature Builder | NetCDF/xarray transformation API | `http://localhost:8090` |

The first startup creates these S3 buckets:

- `bronze-maritime`: immutable source payloads from Copernicus, ECMWF, AIS and port systems.
- `silver-maritime`: standardized and quality-controlled datasets.
- `gold-maritime`: model-ready features and analytical datasets.
- `mlflow-artifacts`: trained models, plots and evaluation artifacts.

TimescaleDB is initialized with schemas for ingestion audit, maritime observations,
AIS vessel positions, port calls, dynamic snapshots and delay predictions.

## Start

From PowerShell in the project directory:

```powershell
Copy-Item .env.example .env
docker compose config
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

The first run downloads and builds several images, so it can take a few minutes.
MinIO is built from its pinned official source tag because its community edition
is source-only; no unverified third-party storage image is used.

## Verify

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-stack.ps1
docker compose ps
```

## Start Airflow phase

The base stack must already be running. Airflow uses a resource-conscious
`LocalExecutor` for the local workstation and a separate `airflow` database in
the existing TimescaleDB server.

The Airflow orchestration image intentionally contains only the S3 and
PostgreSQL clients. Heavy xarray/netCDF computation belongs to a separate
feature-builder image so the scheduler and API server remain small and stable.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-airflow.ps1
```

The startup script retries the Docker Hub base-image pull four times and retries
the custom build three times. This protects the initial installation from short
registry TLS or low-bandwidth timeouts.

Open `http://localhost:8080`. Local authentication is deliberately disabled by
the Airflow simple auth manager. Replace it with a production auth manager
before exposing the service outside the workstation.

The first DAG, `maritime_bronze_to_silver_manifest`, inventories Copernicus
NetCDF/GRIB objects under `bronze-maritime/copernicus/`, writes a versioned JSON
manifest to `silver-maritime`, and records the run in `audit.ingestion_run`.

## Start the NetCDF Feature Builder

The xarray runtime is isolated from Airflow so a large NetCDF file cannot make
the scheduler or API server unstable. The local service is limited to one worker,
2 CPUs and 3 GB of RAM by default.

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start-features.ps1
```

Verify it with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check-features.ps1
```

Airflow then exposes a second DAG, `maritime_netcdf_to_timescale_features`.
For each Copernicus NetCDF object it:

1. extracts the nearest valid Tanger Med sea point with xarray;
2. standardizes wave variables and physical quality flags;
3. builds rolling 3/6/12/24-hour wave features;
4. writes versioned Zstandard Parquet under
   `silver-maritime/features/copernicus/tanger_med/`;
5. upserts operational observations into `core.maritime_observation`;
6. records success, checksum, row count and lineage in `audit.ingestion_run`.

Stop only the Feature Builder with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-features.ps1
```

Stop only Airflow with:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop-airflow.ps1
```

MinIO and Grafana credentials are stored in `.env`. Change every local password
before sharing or deploying the stack.

## Stop without deleting data

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\stop.ps1
```

Named Docker volumes are deliberately preserved.

## Next implementation phase

1. Upload or ingest Copernicus/ECMWF files into `bronze-maritime/copernicus/`.
2. Add ECMWF wind/visibility ingestion beside the Copernicus wave features.
3. Add AIS route exposure and materialize vessel snapshots in TimescaleDB.
4. Train delay and ETA models tracked by MLflow.
5. Expose predictions through FastAPI and the React operational dashboard.
