# PortFlow Control Tower - Système de design

## 1. Direction choisie

PortFlow n'utilise pas un template Streamlit ni un dashboard administratif générique.
La direction retenue est une **Control Tower maritime opérationnelle** conçue pour un
superviseur portuaire qui doit comprendre une situation, identifier une priorité et
prendre une décision rapidement.

La plateforme repose sur trois idées :

1. **Situation avant décoration** : la carte, les séries temporelles, les alertes et
   les navires occupent les zones principales.
2. **Décision avant reporting** : chaque page doit aboutir à un état, un risque, une
   recommandation ou une action humaine traçable.
3. **Densité maîtrisée** : l'interface reste compacte et professionnelle, mais les
   informations sont regroupées par tâche plutôt que répétées dans une grille de cartes.

## 2. Pourquoi cette direction

| Choix | Justification |
|---|---|
| Navigation latérale compacte | Libère la largeur pour les cartes, tableaux et courbes. |
| Barre système persistante | Affiche la zone active, l'heure de Tanger et l'état de la plateforme. |
| Fond graphite | Réduit la fatigue visuelle dans un poste de supervision et valorise les états colorés. |
| Surfaces peu arrondies | Donne une apparence technique et évite l'effet application grand public. |
| Accent turquoise | Signale le fonctionnement normal, la sélection et l'action disponible. |
| Ambre et rouge réservés | Indiquent uniquement la vigilance et la priorité. |
| Typographie monospace | Réservée aux heures, probabilités, identifiants et valeurs techniques. |
| Animations courtes | Montrent une transition, une sélection ou une actualisation sans distraire. |

## 3. Stack de présentation

- **React + TypeScript** : structure des pages et interactions.
- **Material UI** : composants accessibles et système responsive.
- **ECharts** : cartes, séries temporelles, bandes d'incertitude et visualisations métier.
- **Iconify** : iconographie cohérente.
- **CSS/MUI keyframes** : animations fonctionnelles sans dépendance supplémentaire.

## 4. Fichiers responsables du design

### Référence fonctionnelle

- `DESIGN_SYSTEM.md` : source de vérité pour les choix et les règles visuelles.

### Thème global

- `src/theme/theme.ts` : assemblage du thème Material UI.
- `src/theme/palette.ts` : palette MUI générale.
- `src/theme/typography.ts` : familles et niveaux typographiques.
- `src/theme/components/CssBaseline.tsx` : fond global, sélection de texte,
  animations et comportement réduit pour l'accessibilité.

### Structure commune

- `src/layouts/main-layout/index.tsx` : dimensions du contenu et transition entre pages.
- `src/layouts/main-layout/NavigationRail.tsx` : navigation desktop et mobile.
- `src/layouts/main-layout/topbar/Topbar.tsx` : contexte, horloge et état système.

### Pages métier

- `src/components/sections/maritime/MetoceanAnalyticsDashboard.tsx` : météo et mer.
- `src/components/sections/maritime/MetoceanSituationMap.tsx` : situation cartographique.
- `src/pages/maritime/CapacityPage.tsx` : watchlist, risque et capacité des escales.
- `src/components/sections/maritime/CapacityTimelineChart.tsx` : évolution temporelle.

Le shell et le thème sont communs. Une page métier ne doit pas recréer une nouvelle
navigation, une nouvelle palette ou une nouvelle identité.

## 5. Langage visuel

### Couleurs principales

| Rôle | Valeur indicative | Usage |
|---|---|---|
| Fond | `#090B0E` | Fond de l'application. |
| Surface | `#101317` | Outils, tableaux et panneaux. |
| Bordure | `#262C33` | Séparation sans ombre excessive. |
| Texte | `#F4F6F5` | Titres et informations importantes. |
| Texte secondaire | `#86909A` | Métadonnées et contexte. |
| Normal/sélection | `#35E3C0` | État sain, focus et sélection. |
| Information | `#4C8DFF` | Prévision et information neutre. |
| Vigilance | `#FFB35C` | Surveillance et incertitude. |
| Critique | `#FF6B72` | Priorité et risque élevé. |

Une couleur d'alerte ne doit jamais être utilisée comme décoration.

### Typographie

- **Source Serif 4 / Georgia** : grands titres et titres de section uniquement. Le serif exprime l'autorité institutionnelle sans entrer dans les contrôles compacts.
- **IBM Plex Sans / Segoe UI** : navigation, commandes, texte métier et annotations.
- **JetBrains Mono / Consolas** : nombres, heures, probabilités, identifiants et codes.
- Aucun serif sous 18 px dans les surfaces opérationnelles; les KPI et statuts restent sans-serif ou monospace.

### Couleur et profondeur

- **Bleu institutionnel** (`#4C8DFF`) pour l'identité, la sélection et les états stables.
- Variations bleues discrètes sur les KPI; ambre et rouge restent réservés aux alertes métier.
- Aucune ombre portée sur les cartes, la navigation ou les graphiques. La hiérarchie repose sur les bordures, les surfaces et l'espacement.
- Une métrique décisionnelle ne doit apparaître qu'une fois dans chaque niveau de lecture.
- Pas de texte géant à l'intérieur des panneaux.
- Pas d'espacement négatif entre les lettres.

### Formes et espacements

- Rayon maximal habituel : `8px`.
- Boutons icônes : dimensions stables de `34px` à `42px`.
- Une section principale peut être ouverte; un outil autonome peut être encadré.
- Pas de carte imbriquée dans une autre carte.
- Les séparateurs structurent les données avant les ombres.

## 6. Animations autorisées

| Animation | Fonction |
|---|---|
| `portflowRouteIn` | Transition courte lors du changement de page. |
| `portflowStatusRing` | Montre qu'un système est actif. |
| `portflowRowIn` | Introduit progressivement une watchlist. |
| `portflowScan` | Signale une liste analysée ou actualisée. |
| `portflowCardIn` | Met à jour le détail après sélection d'une escale. |

Toutes les animations doivent respecter `prefers-reduced-motion`. Les mouvements
continus purement décoratifs, les orbes et les effets de fond sont interdits.

## 7. Méthode de construction d'une page

Chaque nouvelle page est construite et validée séparément :

1. **Question utilisateur** : écrire la décision que la page doit permettre.
2. **Contrat de données** : identifier les API, états vides, erreurs et horodatages.
3. **Hiérarchie** : situation générale, objet prioritaire, preuve, action.
4. **Composition** : choisir carte, tableau ou courbe selon la nature des données.
5. **Interactions** : sélection, filtrage, comparaison et actualisation.
6. **Responsive** : vérifier desktop, tablette et mobile sans chevauchement.
7. **États complets** : chargement, indisponibilité, absence de données et succès.
8. **Validation visuelle** : vérifier les routes publiées puis obtenir l'accord utilisateur.

Une page n'est pas validée parce qu'elle contient beaucoup de widgets. Elle est validée
quand un opérateur peut répondre rapidement à : **Que se passe-t-il ? Pourquoi ? Que
dois-je examiner ou décider ?**

## 8. Critères d'acceptation

- Le rôle de la page est compréhensible dans le premier écran.
- Les dates utilisent `Africa/Casablanca` et indiquent la fraîcheur des données.
- Les prévisions possèdent une courbe ou une représentation temporelle.
- Les risques montrent leur niveau et leur horizon.
- Les données historiques ne sont pas présentées comme temps réel.
- Toute recommandation précise si une validation humaine est requise.
- Aucun nom interne de modèle ou de bloc expérimental n'est exposé à l'opérateur.
- La page reste exploitable au clavier et avec les animations réduites.

## 9. Workflow technique

Après modification :

```powershell
Set-Location "C:\Users\HP\Documents\Codex\2026-06-28\utilise-google-drive-slack-github-ou-2\maritime-platform"

powershell -ExecutionPolicy Bypass -File ".\deploy-live-metocean-ui.ps1"
```

Pour vérifier un bundle déjà publié sans reconstruire :

```powershell
powershell -ExecutionPolicy Bypass -File ".\deploy-live-metocean-ui.ps1" -VerifyOnly
```

Routes actuellement validées :

- `/weather` : météo, état de mer et prévisions.
- `/capacity` : vigilance des escales, capacité et trajectoire de risque.

## 10. Intégration du cahier « Data in Motion »

Le cahier avancé est une direction, pas une bibliothèque à recopier littéralement.
Les éléments suivants sont retenus et implémentés :

- transitions de route et de sélection courtes ;
- état système animé et compatible avec `prefers-reduced-motion` ;
- watchlist accessible au clavier ;
- jauge de risque liée à la valeur et non décorative ;
- trajectoire temporelle avec P10, P50, P90 et bande d'incertitude ;
- tooltips ECharts contrastés ;
- distinction entre risque critique et revue planifiée par capacité ;
- composition desktop à deux colonnes et navigation mobile dédiée.

Les éléments suivants restent conditionnels :

- **Mapbox/Deck.gl/AIS** : seulement après disponibilité de positions AIS fraîches,
  de trajectoires et de géofences réelles ;
- **vue 3D** : seulement avec une géométrie portuaire fiable et un cas d'usage
  opérationnel mesurable ;
- **WebSocket** : seulement après définition d'un flux serveur et d'un contrat de
  reconnexion ;
- **virtualisation** : à introduire lorsque la watchlist visible dépasse plusieurs
  centaines de lignes ;
- **undo/redo** : à introduire avec le module de décisions, son journal d'audit et
  ses règles d'autorisation.

Une fonctionnalité visuelle ne doit jamais simuler une donnée qui n'existe pas.
