# Raccordement des modèles PortFlow au backend

Le backend ne considère jamais un fichier `.cbm` comme « prêt » uniquement parce qu'il existe.
Un bundle exploitable doit contenir les dix rôles B36B/B36C/B44/B48, l'ordre exact des variables,
les variables catégorielles, les paramètres enregistrés et le SHA-256 de chaque artefact.

## 1. Construire le bundle depuis Colab ou Google Drive

1. Copier `model_bundle/bundle-spec.template.json` et corriger chaque `source_path` si nécessaire.
2. Installer la version utilisée par les checkpoints : `pip install catboost==1.2.8`.
3. Exécuter :

```bash
python backend/tools/prepare_model_bundle.py \
  --spec backend/model_bundle/bundle-spec.json \
  --output /content/drive/MyDrive/TIR/PORTFLOW_MODEL_BUNDLE_B48R
```

Le script extrait directement des `.cbm` les noms de variables et leur ordre, les indices
catégoriels, le nombre d'arbres, la loss, la seed et les paramètres. Il copie les modèles sous
des noms de rôles stables et calcule leurs hashes. Il refuse un bundle incomplet.

## 2. Installer le bundle sur le poste backend

Copier le dossier produit dans `backend/model_bundle/runtime`. Ce répertoire et les `.cbm` sont
ignorés par Git. Ne jamais publier les poids, données métier ou secrets dans le dépôt public.

Dans `backend/.env` :

```dotenv
PORTFLOW_MODEL_BUNDLE_HOST_PATH=./model_bundle/runtime
PORTFLOW_MODEL_MANIFEST=manifest.json
PORTFLOW_MODEL_LIVE_ENABLED=false
PORTFLOW_SOURCE_FRESHNESS_LIMIT_HOURS=2
```

Le conteneur monte le bundle en lecture seule. Reconstruire puis démarrer l'API :

```powershell
cd backend
docker compose -f compose.yaml -f compose.platform.yaml build platform-api
docker compose -f compose.yaml -f compose.platform.yaml up -d platform-api
```

## 3. Vérifier avant toute prédiction

```powershell
Invoke-RestMethod http://localhost:8092/api/v1/model-serving/status |
  ConvertTo-Json -Depth 8
```

`ready=true` signifie que les dix rôles, les hashes et les contrats CatBoost sont valides.
Cela ne signifie pas automatiquement que le live est autorisé. Un bundle `VALIDATED` reste en
rejeu historique. Il faut un statut `PROMOTED`, `PORTFLOW_MODEL_LIVE_ENABLED=true` et une source
plus fraîche que la limite pour obtenir `serving_mode=LIVE`.

## 4. Appeler le moteur B48R

L'endpoint `POST /api/v1/model-serving/unit/remaining-time` reçoit une photographie causale :

```json
{
  "unit_id": "TMU-18402",
  "snapshot_at": "2026-04-22T08:00:00Z",
  "source_observed_at": "2026-04-22T07:58:00Z",
  "features": {
    "FEATURE_1": 12.4,
    "CATEGORICAL_FEATURE": "SCAN_EXPORT"
  }
}
```

Le dictionnaire doit contenir l'union des variables brutes exigées par les modèles. Le moteur :

1. calcule B36B MAE et RMSE ;
2. applique le blend figé `0.9/0.1` ;
3. calcule les risques GE12/GE24/GE36 ;
4. calcule et ordonne les quantiles B44 ;
5. évalue le gate et le meta-stacker B48 ;
6. applique la formule B48R figée ;
7. publie le bundle, le mode, la fraîcheur et les avertissements.

Une variable absente, un type incorrect, un hash différent ou un ordre de features incompatible
provoque un refus explicite : aucune prédiction silencieusement dégradée n'est servie.

## 5. Ce qui reste distinct d'un checkpoint CatBoost

- C2 Time Series : déployer ses pipelines `.joblib`, son routeur cible/horizon et sa calibration
  résiduelle dans un bundle de prévision portuaire séparé.
- Semi-Markov / Monte-Carlo : déployer les matrices de transition, distributions de séjour,
  contrats de route et snapshots de paramètres. Ce moteur n'est pas un unique checkpoint neural.
- Petri : déployer le graphe métier versionné et ses invariants de conservation.
- C5A TCN : ne pas le mettre en production ; l'expérience n'a pas obtenu le gate de promotion.

Le raccordement complet à la Control Tower nécessite enfin un **feature service** qui construit ces
variables depuis le journal événementiel C0-R, avec `snapshot_at` et sans aucune information future.
Le endpoint de modèle est prêt pour ce contrat, mais ne doit pas inventer ces variables depuis le
frontend.
