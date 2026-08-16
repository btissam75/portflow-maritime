# PortFlow Maritime

React control tower for maritime supervision and decision support at Tanger Med.

## Functional pages

- `/control-tower`: operational replay, probabilistic arrival forecasts and model diagnostics
- `/weather`: live metocean situation, weather and wave forecasts, map and vessel-impact analysis
- `/capacity`: capacity watchlist, temporal risk ranking and decision support

The visual rules, typography, color tokens and component conventions are documented in
[`DESIGN_SYSTEM.md`](./DESIGN_SYSTEM.md).

## Stack

- React 18 and TypeScript
- Vite
- Material UI
- Apache ECharts
- Nginx and Docker for publication

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

- `http://localhost:8088/control-tower`
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
