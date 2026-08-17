# PortFlow Maritime — Passation frontend

## 1. Périmètre actif

Le frontend contient volontairement deux pages métier :

| Route       | Page               | Responsabilité                                           |
| ----------- | ------------------ | -------------------------------------------------------- |
| `/weather`  | `WeatherPage.tsx`  | Météo, état de mer, projection et exposition des escales |
| `/capacity` | `CapacityPage.tsx` | Vigilance temporelle, capacité de revue et trajectoires  |

La racine `/` et toute route inconnue redirigent vers `/weather`. L’ancienne route
`/control-tower` n’est plus publiée et redirige également vers `/weather`.

```mermaid
flowchart LR
    ROUTER[React Router] --> LAYOUT[MainLayout]
    LAYOUT --> WEATHER[/weather]
    LAYOUT --> CAPACITY[/capacity]
    WEATHER --> LIVE[Open-Meteo]
    WEATHER --> B62[FastAPI B62/B62A/B62B]
    CAPACITY --> B61E[FastAPI B61E]
```

## 2. Structure à conserver

- `src/routes/` : routes et redirections.
- `src/layouts/main-layout/` : navigation, barre supérieure et zone de contenu.
- `src/theme/` et `DESIGN_SYSTEM.md` : tokens, couleurs et règles visuelles.
- `src/services/http.ts` : client GET commun, timeout et annulation.
- `src/services/liveMetoceanApi.ts` : observations externes Open-Meteo.
- `src/services/metoceanApi.ts` : contrats B62, B62A et B62B.
- `src/services/capacityApi.ts` : contrats B61E, replay et cache de session.
- `src/types/` : types des réponses API actives.
- `src/components/sections/maritime/` : cinq composants utilisés par les deux pages.

Il n’existe plus de provider replay global. Chaque page charge uniquement les sources dont elle a
besoin, ce qui évite les appels réseau parasites et les faux états d’indisponibilité.

## 3. Page Météo

La page distingue les observations externes Open-Meteo des prévisions scientifiques gouvernées
B62/B62A/B62B.

Fonctions disponibles :

- conditions atmosphériques et marines ;
- carte interactive du détroit et des approches de Tanger Med ;
- horizons 12, 24 et 72 heures ;
- affichage Celsius/Fahrenheit ;
- réglage du seuil de vague ;
- trajectoires P10/P50/P90 ;
- classement indicatif de l’exposition des navires ;
- statut du challenger et validation fresh-forward ;
- actualisation manuelle et automatique ;
- mode dégradé explicite lorsqu’une source manque.

Les observations externes peuvent être réelles tandis que la couche interne est en `SHADOW` ou en
`DEMO`. L’interface ne doit jamais fusionner ces droits d’usage.

## 4. Page Escales & capacité

Fonctions disponibles :

- statut B61E et snapshot `VALID_SELECT` ;
- recherche par navire, terminal ou type ;
- filtres Toutes, Vigilance et Critiques ;
- classement par risque et capacité de revue ;
- replay de six snapshots espacés de six heures ;
- lecture/pause et navigation précédent/suivant ;
- temps restant probabiliste P10/P50/P90 ;
- hazards à 6, 12 et 24 heures ;
- trajectoire décisionnelle de l’escale sélectionnée ;
- brouillon local de revue humaine ;
- cache de session et état stale explicite.

Le brouillon ne transmet aucune action au système portuaire. Les réponses `SHADOW` ou `DEMO`
gardent `production_claim_allowed=false` et `automatic_action_allowed=false`.

## 5. Contrats API

### B61E

- `GET /api/v1/maritime/capacity-ranking/status`
- `GET /api/v1/maritime/capacity-ranking/snapshot`
- `GET /api/v1/maritime/capacity-ranking/port-calls/{id}/timeline`

### B62

- `GET /api/v1/maritime/metocean-cascade/status`
- `GET /api/v1/maritime/metocean-cascade/forecast`
- `GET /api/v1/maritime/metocean-cascade/vessel-impact`

### B62A et B62B

- `GET /api/v1/maritime/metocean-augmentation/status`
- `GET /api/v1/maritime/metocean-augmentation/selection`
- `GET /api/v1/maritime/metocean-vintage-validation/status`
- `GET /api/v1/maritime/metocean-vintage-validation/metrics`

L’URL du backend vient de `VITE_API_BASE_URL`. Aucun composant ne doit coder en dur une autre base
d’API.

## 6. Modes d’exécution

En mode réel, FastAPI lit les tables gouvernées TimescaleDB matérialisées par les flows B61E/B62.
Les données, modèles et volumes persistants ne sont pas versionnés dans Git.

Le mode local démarre FastAPI sans Docker avec des données dynamiques explicitement simulées :

```powershell
powershell -ExecutionPolicy Bypass -File ".\backend\scripts\start-local-demo.ps1"
```

`SMART_PORT_LOCAL_DEMO_MODE=true` n’est positionnée que dans ce processus. Le mode est désactivé
par défaut et toutes les cartes simulées portent un avertissement visible.

## 7. Développement local

Terminal API :

```powershell
powershell -ExecutionPolicy Bypass -File ".\backend\scripts\start-local-demo.ps1"
```

Terminal frontend :

```powershell
$env:VITE_API_BASE_URL = "http://localhost:8092"
pnpm install --frozen-lockfile
pnpm dev
```

Contrôles obligatoires avant livraison :

```powershell
pnpm lint
pnpm build
```

## 8. Règles pour les futures interfaces

Chaque nouveau module du projet principal doit :

1. déclarer sa route dans `src/routes/paths.ts` et `src/routes/router.tsx` ;
2. ajouter une seule entrée claire dans `NavigationRail.tsx` ;
3. créer des types à partir du contrat OpenAPI ;
4. isoler les appels réseau dans `src/services/` ;
5. distinguer chargement, partiel, indisponible, cache et stale ;
6. afficher la fraîcheur et le droit d’usage des données ;
7. interdire les zéros fictifs et les décisions automatiquement validées ;
8. tester desktop, mobile, clavier, erreurs réseau, lint et build.

Cette base à deux pages sert désormais de socle propre pour construire progressivement les
interfaces du projet principal.
