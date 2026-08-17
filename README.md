# PortFlow Maritime

Full-stack maritime supervision and decision-support platform for Tanger Med. The repository is
now a monorepo: the React control tower lives at the root and the data/ML/API platform lives in
[`backend/`](./backend/).

## Functional pages

- `/weather`: live metocean situation, weather and wave forecasts, map and vessel-impact analysis
- `/capacity`: capacity watchlist, temporal risk ranking and decision support

The visual rules, typography, color tokens and component conventions are documented in
[`DESIGN_SYSTEM.md`](./DESIGN_SYSTEM.md).

Detailed handoff documentation:

- [`FRONTEND_HANDOFF.md`](./FRONTEND_HANDOFF.md): pages, interactions and frontend limitations
- [`TECHNICAL_ARCHITECTURE.md`](./TECHNICAL_ARCHITECTURE.md): Prefect, TimescaleDB, MinIO,
  FastAPI, modeling, governance and deployment
- [`PLATFORM_ENGINEERING_GUIDE.md`](./PLATFORM_ENGINEERING_GUIDE.md): canonical end-to-end
  installation, architecture, data, modeling, API, operations, security and handoff guide

## Stack

- React 18 and TypeScript
- Vite
- Material UI
- Apache ECharts
- Nginx and Docker for publication
- FastAPI and Pydantic
- Prefect and Airflow
- TimescaleDB, MinIO, MLflow and Grafana
- CatBoost, PyTorch/GRU, NeuralForecast, AutoGluon/Chronos and conformal calibration

## Full platform

```powershell
Copy-Item .\backend\.env.example .\backend\.env
# Replace every CHANGE_ME value before startup.
Set-Location .\backend
docker compose up -d --build
docker compose -f compose.prefect.yaml up -d --build
docker compose -f compose.prefect.yaml --profile tools run --rm prefect-init
docker compose -f compose.platform.yaml up -d --build
```

Platform URLs:

- frontend: `http://localhost:8088/weather`
- API docs: `http://localhost:8092/docs`
- Prefect: `http://localhost:4200`
- MinIO: `http://localhost:9001`
- MLflow: `http://localhost:5000`

Read [`PLATFORM_ENGINEERING_GUIDE.md`](./PLATFORM_ENGINEERING_GUIDE.md) before running scientific
pipelines or changing data/model contracts.

## Local UI demonstration without Docker

The Git repository intentionally excludes port data, trained models and persistent TimescaleDB/
MinIO volumes. The local demo mode exercises the Weather and Capacity API contracts without
claiming that generated values are real or production-ready. It is disabled by default.

Terminal 1 — local FastAPI shadow data:

```powershell
powershell -ExecutionPolicy Bypass -File ".\backend\scripts\start-local-demo.ps1"
```

Terminal 2 — React application:

```powershell
$env:VITE_API_BASE_URL = "http://localhost:8092"
pnpm dev
```

Open `http://localhost:3000/weather` or `http://localhost:3000/capacity`. Every simulated result is
visibly labelled `DEMO`; production promotion and automatic actions remain forbidden. To use real
outputs, leave `SMART_PORT_LOCAL_DEMO_MODE` disabled and materialize the governed B61E/B62 tables
through the platform flows described in the engineering guide.

## Local development

```powershell
pnpm install --frozen-lockfile
$env:VITE_API_BASE_URL = "http://localhost:8092"
pnpm dev
```

Vite serves the application locally, normally at `http://localhost:3000`.

## Build

```powershell
pnpm run build
```

## Docker publication

With Docker Desktop available:

```powershell
powershell -ExecutionPolicy Bypass -File ".\deploy-live-metocean-ui.ps1"
```

Published routes:

- `http://localhost:8088/weather`
- `http://localhost:8088/capacity`

The frontend expects the maritime FastAPI service at `http://localhost:8092` unless
`VITE_API_BASE_URL` is overridden.

## Repository hygiene

Local environment files, dependencies, generated bundles, Docker inspection directories and
rollback archives are intentionally excluded from Git. Never commit API credentials or `.env`
files; provide non-secret examples through `.env.example` when configuration evolves.

The interface began from the free Dabang React/Material UI template by ThemeWagon. Its original
attribution is retained in the application footer.
