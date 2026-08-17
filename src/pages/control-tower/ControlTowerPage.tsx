import {
  Alert,
  Box,
  Button,
  ButtonBase,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  InputAdornment,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Slider,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import IconifyIcon from 'components/base/IconifyIcon';
import ControlTowerForecastChart from 'components/control-tower/ControlTowerForecastChart';
import ControlTowerProcessBoard from 'components/control-tower/ControlTowerProcessBoard';
import InternalPortMap from 'components/control-tower/InternalPortMap';
import MaritimeApproachMap from 'components/control-tower/MaritimeApproachMap';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import paths from 'routes/paths';
import { controlTowerApi } from 'services/controlTowerApi';
import { portflowPalette as pf } from 'theme/portflowPalette';
import type {
  ControlTowerSnapshot,
  ControlTowerView,
  SimulationPayload,
  SimulationResult,
  TowerAlert,
  TowerDecision,
  TowerStage,
  TowerUnit,
  TowerUnitDetail,
} from 'types/controlTower';

const viewCopy: Record<
  ControlTowerView,
  { eyebrow: string; title: string; description: string; icon: string }
> = {
  overview: {
    eyebrow: 'TANGER MED · POSTE DÉCISIONNEL',
    title: 'Operational Control Tower',
    description:
      'Situation portuaire, prévisions probabilistes et décisions recommandées sur une seule surface.',
    icon: 'lucide:layout-dashboard',
  },
  units: {
    eyebrow: 'VIGILANCE OPÉRATIONNELLE',
    title: 'Unités à examiner',
    description: 'Une file explicable, ordonnée par impact attendu et échéance métier.',
    icon: 'lucide:container',
  },
  process: {
    eyebrow: 'FLUX MÉTIER',
    title: 'Trajectoire portuaire',
    description: 'Surveiller la charge et la progression de ZRE jusqu’à l’embarquement.',
    icon: 'lucide:git-branch',
  },
  forecast: {
    eyebrow: 'ANTICIPATION',
    title: 'Prévisions multi-horizons',
    description: 'Comparer les arrivées, le backlog probable et les capacités à H+1…H+24.',
    icon: 'lucide:chart-no-axes-combined',
  },
  alerts: {
    eyebrow: 'VIGILANCE',
    title: 'Alertes consolidées',
    description: 'Comprendre la probabilité, l’impact, la cause et l’action recommandée.',
    icon: 'lucide:shield-alert',
  },
  decisions: {
    eyebrow: 'PILOTAGE',
    title: 'Cycle de vie des décisions',
    description: 'Décider, affecter, exécuter, vérifier et clôturer avec une trace complète.',
    icon: 'lucide:list-checks',
  },
  vessels: {
    eyebrow: 'APPROCHES MARITIMES',
    title: 'Navires et fenêtres d’escale',
    description: 'Relier l’approche des navires aux unités, à la capacité et au risque terminal.',
    icon: 'lucide:ship',
  },
  simulation: {
    eyebrow: 'WHAT-IF',
    title: 'Laboratoire de scénarios',
    description: 'Mesurer l’effet opérationnel d’un renfort de capacité ou d’une règle de route.',
    icon: 'lucide:flask-conical',
  },
  quality: {
    eyebrow: 'FIABILITÉ',
    title: 'Qualité et fraîcheur des données',
    description: 'Savoir précisément quelles informations soutiennent la décision.',
    icon: 'lucide:database-zap',
  },
  audit: {
    eyebrow: 'GOUVERNANCE',
    title: 'Journal d’audit',
    description: 'Retracer les prévisions, alertes et actions sans modifier l’historique.',
    icon: 'lucide:scroll-text',
  },
  reports: {
    eyebrow: 'TRANSMISSION',
    title: 'Rapports et exports',
    description: 'Préparer un passage de quart lisible, reproductible et partageable.',
    icon: 'lucide:file-chart-column',
  },
};

const tierTone = {
  CRITIQUE: pf.functional.red,
  VIGILANCE: pf.functional.amber,
  NORMAL: pf.functional.green,
} as const;

const alertTone = {
  CRITIQUE: pf.functional.red,
  VIGILANCE: pf.functional.amber,
  INFORMATION: pf.functional.blue,
} as const;

const decisionStatuses = ['À analyser', 'Décidée', 'Affectée', 'En cours', 'Vérifiée', 'Clôturée'];
const horizons = [1, 3, 6, 12, 24] as const;

const formatDateTime = (value: string) =>
  new Intl.DateTimeFormat('fr-FR', {
    day: '2-digit',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Africa/Casablanca',
  }).format(new Date(value));

const relativeMinutes = (value: string) =>
  Math.max(0, Math.round((Date.now() - new Date(value).getTime()) / 60_000));

const downloadJson = (filename: string, payload: unknown) => {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }),
  );
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};

const panelSx = {
  p: { xs: 1.4, sm: 1.8 },
  color: pf.text.primary,
  bgcolor: '#081F2A',
  backgroundImage: 'linear-gradient(145deg, rgba(85,214,194,.025), transparent 48%)',
  border: '1px solid #174453',
  borderRadius: '14px',
  boxShadow: '0 18px 44px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.018)',
};

const SectionTitle = ({
  icon,
  title,
  caption,
  action,
}: {
  icon: string;
  title: string;
  caption?: string;
  action?: React.ReactNode;
}) => (
  <Stack direction="row" alignItems="center" gap={1} mb={1.4}>
    <Box
      sx={{
        width: 32,
        height: 32,
        display: 'grid',
        placeItems: 'center',
        color: pf.functional.cyan,
        bgcolor: pf.functional.cyanSoft,
        border: `1px solid ${pf.functional.cyan}2f`,
        borderRadius: '9px',
      }}
    >
      <IconifyIcon icon={icon} sx={{ fontSize: 16 }} />
    </Box>
    <Box minWidth={0}>
      <Typography sx={{ color: pf.text.primary, fontSize: 13, fontWeight: 800 }}>
        {title}
      </Typography>
      {caption && (
        <Typography sx={{ color: pf.text.tertiary, fontSize: 9.5 }}>{caption}</Typography>
      )}
    </Box>
    {action && <Box ml="auto">{action}</Box>}
  </Stack>
);

const MetricCard = ({
  label,
  value,
  unit,
  icon,
  color,
  note,
}: {
  label: string;
  value: string | number;
  unit?: string;
  icon: string;
  color: string;
  note: string;
}) => (
  <Paper
    sx={{
      ...panelSx,
      position: 'relative',
      minHeight: 126,
      overflow: 'hidden',
      transition: 'transform 180ms ease, border-color 180ms ease',
      '&:hover': { transform: 'translateY(-4px)', borderColor: `${color}70` },
    }}
  >
    <Box
      sx={{
        position: 'absolute',
        width: 100,
        height: 100,
        right: -28,
        top: -30,
        borderRadius: '50%',
        bgcolor: `${color}0e`,
        boxShadow: `0 0 40px ${color}10`,
      }}
    />
    <Stack direction="row" justifyContent="space-between">
      <Typography
        sx={{ color: pf.text.tertiary, fontSize: 8.5, fontWeight: 900, letterSpacing: '.1em' }}
      >
        {label.toUpperCase()}
      </Typography>
      <IconifyIcon icon={icon} sx={{ color, fontSize: 17 }} />
    </Stack>
    <Stack direction="row" alignItems="baseline" gap={0.55} mt={1.2}>
      <Typography sx={{ color, fontSize: 29, fontWeight: 850, letterSpacing: '-.04em' }}>
        {value}
      </Typography>
      {unit && <Typography sx={{ color: pf.text.secondary, fontSize: 10 }}>{unit}</Typography>}
    </Stack>
    <Typography sx={{ color: pf.text.tertiary, fontSize: 8.5, mt: 0.7 }}>{note}</Typography>
  </Paper>
);

const ProbabilityBar = ({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: string;
}) => (
  <Box>
    <Stack direction="row" justifyContent="space-between" mb={0.3}>
      <Typography sx={{ color: pf.text.tertiary, fontSize: 8.5 }}>{label}</Typography>
      <Typography sx={{ color, fontSize: 8.5, fontWeight: 850 }}>
        {Math.round(value * 100)}%
      </Typography>
    </Stack>
    <LinearProgress
      variant="determinate"
      value={value * 100}
      sx={{
        height: 4,
        borderRadius: 4,
        bgcolor: pf.background.secondary,
        '& .MuiLinearProgress-bar': { bgcolor: color, borderRadius: 4 },
      }}
    />
  </Box>
);

const UnitRow = ({
  unit,
  selected,
  onSelect,
}: {
  unit: TowerUnit;
  selected: boolean;
  onSelect: (id: string) => void;
}) => {
  const color = tierTone[unit.tier];
  return (
    <ButtonBase
      onClick={() => onSelect(unit.unit_id)}
      sx={{
        width: 1,
        display: 'grid',
        gridTemplateColumns: { xs: '1fr auto', lg: '1.1fr .65fr .6fr .7fr .8fr' },
        gap: 1,
        alignItems: 'center',
        p: 1.1,
        textAlign: 'left',
        color: pf.text.primary,
        bgcolor: selected ? `${color}10` : 'transparent',
        border: `1px solid ${selected ? `${color}70` : pf.structure.borderSoft}`,
        borderRadius: '11px',
        transition: 'all 150ms ease',
        '&:hover': { bgcolor: pf.background.panelHover, transform: 'translateX(3px)' },
      }}
    >
      <Stack direction="row" alignItems="center" gap={0.9} minWidth={0}>
        <Box
          sx={{
            width: 3,
            height: 34,
            bgcolor: color,
            borderRadius: 3,
            boxShadow: `0 0 9px ${color}`,
          }}
        />
        <Box minWidth={0}>
          <Typography noWrap sx={{ fontSize: 11, fontWeight: 800 }}>
            {unit.unit_id}
          </Typography>
          <Typography noWrap sx={{ color: pf.text.tertiary, fontSize: 8.5 }}>
            {unit.cause}
          </Typography>
        </Box>
      </Stack>
      <Box sx={{ display: { xs: 'none', lg: 'block' } }}>
        <Typography sx={{ color: pf.text.primary, fontSize: 10.5, fontWeight: 700 }}>
          {unit.stage_label}
        </Typography>
        <Typography sx={{ color: pf.text.tertiary, fontSize: 8 }}>
          {unit.dwell_h} h sur zone
        </Typography>
      </Box>
      <Box sx={{ display: { xs: 'none', lg: 'block' } }}>
        <Typography sx={{ color: pf.functional.purple, fontSize: 12, fontWeight: 850 }}>
          {unit.eta_p50_h} h
        </Typography>
        <Typography sx={{ color: pf.text.tertiary, fontSize: 8 }}>
          P90 {unit.eta_p90_h} h
        </Typography>
      </Box>
      <Box sx={{ display: { xs: 'none', lg: 'block' } }}>
        <Typography
          sx={{
            color: unit.ge24 >= 0.5 ? pf.functional.red : pf.text.secondary,
            fontSize: 10.5,
            fontWeight: 800,
          }}
        >
          {Math.round(unit.ge24 * 100)}%
        </Typography>
        <Typography sx={{ color: pf.text.tertiary, fontSize: 8 }}>risque ≥ 24 h</Typography>
      </Box>
      <Chip
        label={unit.tier}
        size="small"
        sx={{
          justifySelf: 'end',
          height: 21,
          color,
          bgcolor: `${color}12`,
          border: `1px solid ${color}35`,
          fontSize: 7.5,
          fontWeight: 900,
        }}
      />
    </ButtonBase>
  );
};

const UnitQueueCard = ({
  unit,
  rank,
  selected,
  onSelect,
}: {
  unit: TowerUnit;
  rank: number;
  selected: boolean;
  onSelect: (id: string) => void;
}) => {
  const color = tierTone[unit.tier];
  const sequence = ['ZRE', 'COULOIR', 'PARK', 'SCAN', 'PV', 'SAS', 'TERMINAL'];
  const stageIndex = sequence.indexOf(unit.stage);
  const nextStage =
    stageIndex >= 0 && stageIndex < sequence.length - 1 ? sequence[stageIndex + 1] : 'EMBARQUEMENT';
  return (
    <ButtonBase
      onClick={() => onSelect(unit.unit_id)}
      sx={{
        width: 1,
        display: 'block',
        p: 1.35,
        textAlign: 'left',
        color: pf.text.primary,
        bgcolor: selected ? `${color}12` : '#081E29',
        border: `1px solid ${selected ? color : '#174453'}`,
        borderRadius: '11px',
        boxShadow: selected ? `0 0 30px ${color}0d` : 'none',
        transition: 'transform 160ms ease, border-color 160ms ease, background 160ms ease',
        '&:hover': { transform: 'translateY(-2px)', borderColor: `${color}AA`, bgcolor: '#0A2632' },
      }}
    >
      <Stack direction="row" alignItems="center">
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: '50%',
            bgcolor: color,
            boxShadow: `0 0 10px ${color}`,
          }}
        />
        <Typography sx={{ ml: 0.65, color: '#86A4AF', fontSize: 8, fontWeight: 900 }}>
          PRIORITÉ {String(rank).padStart(2, '0')}
        </Typography>
        <Chip
          label={unit.tier}
          size="small"
          sx={{
            ml: 'auto',
            height: 18,
            color,
            bgcolor: `${color}10`,
            border: `1px solid ${color}35`,
            fontSize: 6.5,
            fontWeight: 900,
          }}
        />
      </Stack>
      <Stack
        direction={{ xs: 'column', sm: 'row' }}
        alignItems={{ sm: 'center' }}
        gap={0.65}
        mt={0.85}
      >
        <Typography sx={{ color: pf.text.primary, fontSize: 12.5, fontWeight: 850 }}>
          {unit.unit_id}
        </Typography>
        <Chip
          icon={<IconifyIcon icon="lucide:route" />}
          label={`${unit.stage} → ${nextStage}`}
          sx={{
            height: 21,
            color: pf.functional.cyan,
            bgcolor: pf.functional.cyanSoft,
            fontSize: 7.5,
            '& .MuiChip-icon': { color: pf.functional.cyan, fontSize: 11 },
          }}
        />
        <Typography sx={{ ml: { sm: 'auto' }, color: '#86A4AF', fontSize: 8 }}>
          Confiance {Math.round(unit.confidence * 100)} %
        </Typography>
      </Stack>
      <Typography sx={{ color: '#8BA8B3', fontSize: 9.5, lineHeight: 1.45, mt: 0.55 }}>
        {unit.cause}
      </Typography>
      <Divider sx={{ my: 1, borderColor: '#174453' }} />
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2,1fr)', sm: 'repeat(4,1fr)' },
          gap: 0.75,
        }}
      >
        {[
          ['ETA centrale', `${unit.eta_p50_h} h`, pf.functional.purple],
          ['ETA prudente', `${unit.eta_p90_h} h`, color],
          [
            'Risque ≥ 24 h',
            `${Math.round(unit.ge24 * 100)} %`,
            unit.ge24 >= 0.5 ? pf.functional.red : pf.text.primary,
          ],
          ['Séjour en zone', `${unit.dwell_h} h`, pf.text.primary],
        ].map(([label, value, tone]) => (
          <Box key={label}>
            <Typography sx={{ color: '#648895', fontSize: 7 }}>{label.toUpperCase()}</Typography>
            <Typography sx={{ color: tone, fontSize: 10.5, fontWeight: 800, mt: 0.15 }}>
              {value}
            </Typography>
          </Box>
        ))}
      </Box>
      <Stack direction="row" alignItems="center" mt={1}>
        <IconifyIcon icon="lucide:map-pin" sx={{ color: '#648895', fontSize: 12 }} />
        <Typography sx={{ ml: 0.4, color: '#7898A4', fontSize: 7.5 }}>
          {unit.location.zone} · {unit.location_quality.toLowerCase()} · {unit.location_age_minutes}{' '}
          min
        </Typography>
        <Typography sx={{ ml: 'auto', color: pf.functional.cyan, fontSize: 8, fontWeight: 800 }}>
          EXAMINER →
        </Typography>
      </Stack>
    </ButtonBase>
  );
};

const UnitDetailPanel = ({
  detail,
  loading,
  onDecision,
}: {
  detail: TowerUnitDetail | null;
  loading: boolean;
  onDecision: (unit: TowerUnitDetail) => void;
}) => {
  if (loading)
    return (
      <Paper sx={{ ...panelSx, minHeight: 460, display: 'grid', placeItems: 'center' }}>
        <CircularProgress size={28} sx={{ color: pf.functional.cyan }} />
      </Paper>
    );
  if (!detail)
    return (
      <Paper
        sx={{
          ...panelSx,
          minHeight: 460,
          display: 'grid',
          placeItems: 'center',
          textAlign: 'center',
        }}
      >
        <Box>
          <IconifyIcon
            icon="lucide:mouse-pointer-click"
            sx={{ color: pf.text.tertiary, fontSize: 34 }}
          />
          <Typography sx={{ color: pf.text.secondary, fontSize: 12, mt: 1 }}>
            Sélectionnez une unité pour ouvrir sa fiche.
          </Typography>
        </Box>
      </Paper>
    );
  const color = tierTone[detail.tier];
  return (
    <Paper sx={{ ...panelSx, position: { xl: 'sticky' }, top: 84 }}>
      <Stack direction="row" alignItems="flex-start" gap={1}>
        <Box>
          <Typography sx={{ color: color, fontSize: 8.5, fontWeight: 900 }}>
            {detail.tier} · {detail.route}
          </Typography>
          <Typography sx={{ color: pf.text.primary, fontSize: 17, fontWeight: 850 }}>
            {detail.unit_id}
          </Typography>
          <Typography sx={{ color: pf.text.tertiary, fontSize: 9 }}>
            {detail.stage_label} · position {detail.location.precision.toLowerCase()} ·{' '}
            {detail.location_age_minutes} min
          </Typography>
        </Box>
        <Button
          onClick={() => onDecision(detail)}
          sx={{
            ml: 'auto',
            color: pf.background.primary,
            bgcolor: pf.functional.cyan,
            borderRadius: '9px',
            fontSize: 8.5,
            '&:hover': { bgcolor: pf.functional.green },
          }}
        >
          Créer une décision
        </Button>
      </Stack>
      <Divider sx={{ my: 1.4, borderColor: pf.structure.border }} />
      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 0.7 }}>
        {(
          [
            ['P10', detail.eta_p10_h],
            ['P50', detail.eta_p50_h],
            ['P80', detail.eta_p80_h],
            ['P90', detail.eta_p90_h],
          ] as const
        ).map(([label, value]) => (
          <Box key={label} sx={{ p: 0.75, bgcolor: pf.background.secondary, borderRadius: '8px' }}>
            <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>{label}</Typography>
            <Typography
              sx={{
                color: label === 'P90' ? pf.functional.purple : pf.text.primary,
                fontSize: 13,
                fontWeight: 850,
              }}
            >
              {value} h
            </Typography>
          </Box>
        ))}
      </Box>
      <Stack gap={0.8} mt={1.35}>
        <ProbabilityBar label="Dépassement 12 h" value={detail.ge12} color={pf.functional.amber} />
        <ProbabilityBar label="Dépassement 24 h" value={detail.ge24} color={pf.functional.red} />
        <ProbabilityBar label="Dépassement 36 h" value={detail.ge36} color={pf.functional.purple} />
      </Stack>
      <Box
        sx={{
          mt: 1.4,
          p: 1,
          bgcolor: `${color}0d`,
          border: `1px solid ${color}28`,
          borderRadius: '10px',
        }}
      >
        <Typography sx={{ color, fontSize: 8.5, fontWeight: 900 }}>
          POURQUOI CETTE ESTIMATION ?
        </Typography>
        <Typography sx={{ color: pf.text.secondary, fontSize: 9.5, lineHeight: 1.55, mt: 0.4 }}>
          {detail.explanation}
        </Typography>
      </Box>
      <Typography sx={{ color: pf.text.tertiary, fontSize: 8, fontWeight: 900, mt: 1.5, mb: 0.8 }}>
        TRAJECTOIRE OBSERVÉE
      </Typography>
      <Stack gap={0.55}>
        {detail.timeline.map((event, index) => (
          <Stack key={`${event.stage}-${event.at}`} direction="row" alignItems="center" gap={0.8}>
            <Box
              sx={{
                width: 17,
                height: 17,
                display: 'grid',
                placeItems: 'center',
                borderRadius: '50%',
                color:
                  index === detail.timeline.length - 1 ? pf.background.primary : pf.functional.cyan,
                bgcolor:
                  index === detail.timeline.length - 1
                    ? pf.functional.cyan
                    : pf.functional.cyanSoft,
                border: `1px solid ${pf.functional.cyan}45`,
                fontSize: 7,
              }}
            >
              {index + 1}
            </Box>
            <Box flex={1}>
              <Typography sx={{ color: pf.text.primary, fontSize: 9.5, fontWeight: 700 }}>
                {event.label}
              </Typography>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>
                {formatDateTime(event.at)} · {event.duration_h} h
              </Typography>
            </Box>
            <Chip
              label={event.reliability}
              size="small"
              sx={{
                height: 17,
                color: pf.functional.green,
                bgcolor: pf.functional.greenSoft,
                fontSize: 6.5,
              }}
            />
          </Stack>
        ))}
      </Stack>
      <Typography sx={{ color: pf.text.tertiary, fontSize: 8, mt: 1.3 }}>
        Calcul {relativeMinutes(detail.prediction.calculated_at)} min · confiance{' '}
        {Math.round(detail.confidence * 100)}% · moteur en mode préparation
      </Typography>
    </Paper>
  );
};

const OverviewView = ({
  snapshot,
  selectedStage,
  setSelectedStage,
  selectedUnit,
  setSelectedUnit,
  onOpenDecision,
}: {
  snapshot: ControlTowerSnapshot;
  selectedStage: string;
  setSelectedStage: (value: string) => void;
  selectedUnit: string;
  setSelectedUnit: (value: string) => void;
  onOpenDecision: (alert: TowerAlert) => void;
}) => {
  const [twinView, setTwinView] = useState<'PORT' | 'APPROACH' | 'FLOW'>('PORT');
  const [evidence, setEvidence] = useState('');
  const zre = snapshot.stages.find((stage) => stage.code === 'ZRE') ?? snapshot.stages[0];
  const terminal = snapshot.stages.find((stage) => stage.code === 'TERMINAL') ?? snapshot.stages[0];
  const h6 = snapshot.forecast.find((point) => point.horizon_h === 6) ?? snapshot.forecast[0];
  const executiveMetrics = [
    {
      label: 'Arrivées ZRE · H6',
      value: zre.forecast.h6,
      unit: 'unités',
      note: `P10 ${Math.max(0, zre.forecast.h6 - 19)} · P90 ${zre.forecast.h6 + 27}`,
      icon: 'lucide:log-in',
      color: pf.text.primary,
    },
    {
      label: 'Charge portuaire',
      value: snapshot.metrics.active_units.toLocaleString('fr-FR'),
      unit: 'unités',
      note: `+${Math.max(1, snapshot.metrics.at_risk_units)} sous vigilance`,
      icon: 'lucide:boxes',
      color: pf.text.primary,
    },
    {
      label: 'Occupation terminal',
      value: Math.round(terminal.occupancy_pct),
      unit: '%',
      note: 'Seuil d’attention · 80 %',
      icon: 'lucide:warehouse',
      color: terminal.occupancy_pct >= 80 ? pf.functional.amber : pf.functional.green,
    },
    {
      label: 'Embarquements · H6',
      value: h6.departures,
      unit: 'unités',
      note: `Intervalle estimé ${Math.max(0, h6.departures - 18)}–${h6.departures + 21}`,
      icon: 'lucide:ship',
      color: pf.text.primary,
    },
    {
      label: 'Décisions à examiner',
      value: snapshot.metrics.pending_decisions,
      unit: '',
      note: `${snapshot.alerts.filter((alert) => alert.severity === 'CRITIQUE').length || 1} priorité opérationnelle`,
      icon: 'lucide:list-checks',
      color: pf.functional.amber,
    },
  ];

  return (
    <Stack gap={1.6}>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', sm: 'repeat(2,1fr)', lg: 'repeat(3,1fr)' },
          gap: 1,
        }}
      >
        {executiveMetrics.map((metric, index) => (
          <Paper
            key={metric.label}
            sx={{
              ...panelSx,
              minHeight: 118,
              p: 1.6,
              gridColumn: { lg: index >= 3 ? 'span 1' : 'auto' },
              transition: 'transform 180ms ease, border-color 180ms ease',
              '&:hover': { transform: 'translateY(-3px)', borderColor: '#2D6678' },
            }}
          >
            <Stack direction="row" alignItems="center">
              <Typography sx={{ color: '#89A8B4', fontSize: 9 }}>{metric.label}</Typography>
              <IconifyIcon icon={metric.icon} sx={{ ml: 'auto', color: '#5D8B9C', fontSize: 18 }} />
            </Stack>
            <Stack direction="row" alignItems="baseline" gap={0.6} mt={1.1}>
              <Typography
                sx={{
                  color: metric.color,
                  fontSize: 27,
                  fontWeight: 850,
                  letterSpacing: '-.035em',
                }}
              >
                {metric.value}
              </Typography>
              {metric.unit && (
                <Typography sx={{ color: '#89A8B4', fontSize: 9 }}>{metric.unit}</Typography>
              )}
            </Stack>
            <Typography
              sx={{ color: index === 1 ? pf.functional.cyan : '#6F929F', fontSize: 8.5, mt: 0.6 }}
            >
              {metric.note}
            </Typography>
          </Paper>
        ))}
      </Box>

      <Paper sx={{ ...panelSx, p: 0, overflow: 'hidden' }}>
        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          alignItems={{ sm: 'center' }}
          gap={1}
          sx={{ px: 1.8, py: 1.25, borderBottom: '1px solid #174453' }}
        >
          <Stack direction="row" alignItems="center" gap={0.8}>
            <IconifyIcon
              icon="lucide:map-pinned"
              sx={{ color: pf.functional.cyan, fontSize: 18 }}
            />
            <Typography sx={{ color: pf.text.primary, fontSize: 13, fontWeight: 850 }}>
              Jumeau numérique opérationnel
            </Typography>
            <Typography sx={{ color: '#6F929F', fontSize: 8.5 }}>· vue schématique</Typography>
          </Stack>
          <Stack
            direction="row"
            gap={0.35}
            ml={{ sm: 'auto' }}
            sx={{ p: 0.35, bgcolor: '#061721', border: '1px solid #174453', borderRadius: '9px' }}
          >
            {(
              [
                ['PORT', 'Port'],
                ['APPROACH', 'Approche'],
                ['FLOW', 'Flux'],
              ] as const
            ).map(([value, label]) => (
              <Button
                key={value}
                onClick={() => setTwinView(value)}
                sx={{
                  minWidth: 64,
                  minHeight: 30,
                  color: twinView === value ? '#06141D' : '#6F929F',
                  bgcolor: twinView === value ? '#5BA7BC' : 'transparent',
                  borderRadius: '7px',
                  fontSize: 8.5,
                }}
              >
                {label}
              </Button>
            ))}
          </Stack>
        </Stack>
        {twinView === 'PORT' && (
          <InternalPortMap
            stages={snapshot.stages}
            units={snapshot.units}
            vessels={snapshot.vessels}
            selectedStage={selectedStage}
            selectedUnit={selectedUnit}
            onStageSelect={setSelectedStage}
            onUnitSelect={setSelectedUnit}
            riskOnly={false}
            onRiskOnlyChange={() => undefined}
          />
        )}
        {twinView === 'APPROACH' && (
          <MaritimeApproachMap
            vessels={snapshot.vessels}
            selected={snapshot.vessels[0]?.vessel_id}
            onSelect={() => undefined}
          />
        )}
        {twinView === 'FLOW' && (
          <Box sx={{ p: 1.4 }}>
            <ControlTowerProcessBoard
              stages={snapshot.stages}
              selected={selectedStage}
              onSelect={setSelectedStage}
              horizon={6}
            />
            <ControlTowerForecastChart data={snapshot.forecast} height={290} />
          </Box>
        )}
      </Paper>

      <Paper sx={{ ...panelSx, p: 0, overflow: 'hidden' }}>
        <Stack
          direction="row"
          alignItems="center"
          sx={{ px: 1.8, py: 1.25, borderBottom: '1px solid #174453' }}
        >
          <IconifyIcon icon="lucide:lightbulb" sx={{ color: pf.functional.cyan, fontSize: 18 }} />
          <Typography sx={{ ml: 0.8, color: pf.text.primary, fontSize: 13, fontWeight: 850 }}>
            File de décisions
          </Typography>
          <Typography sx={{ ml: 'auto', color: '#6F929F', fontSize: 8 }}>
            classées par impact
          </Typography>
        </Stack>
        <Stack gap={1} p={1.5}>
          {snapshot.alerts.map((alert, index) => {
            const recommendation =
              snapshot.recommendations[index % snapshot.recommendations.length];
            const color =
              index === 0 ? pf.functional.amber : index === 1 ? '#58B9F6' : pf.functional.cyan;
            const expanded = evidence === alert.alert_id;
            return (
              <Box
                key={alert.alert_id}
                sx={{
                  p: 1.45,
                  bgcolor: '#081E29',
                  border: `1px solid ${index === 0 ? `${color}90` : '#174453'}`,
                  borderRadius: '11px',
                  boxShadow: index === 0 ? `0 0 28px ${color}0b` : 'none',
                }}
              >
                <Stack direction="row" alignItems="center">
                  <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: color }} />
                  <Typography sx={{ ml: 0.65, color: '#87A5B1', fontSize: 8, fontWeight: 900 }}>
                    PRIORITÉ {String(index + 1).padStart(2, '0')}
                  </Typography>
                  <Typography sx={{ ml: 'auto', color: '#87A5B1', fontSize: 8 }}>
                    Confiance {Math.round(alert.confidence * 100)} %
                  </Typography>
                </Stack>
                <Typography sx={{ color: pf.text.primary, fontSize: 13, fontWeight: 850, mt: 1 }}>
                  {alert.recommendation}
                </Typography>
                <Typography sx={{ color: '#83A2AE', fontSize: 9.5, mt: 0.35 }}>
                  {alert.cause}
                </Typography>
                <Divider sx={{ my: 1.05, borderColor: '#174453' }} />
                <Stack direction={{ xs: 'column', sm: 'row' }} gap={2.2}>
                  <Box>
                    <Typography sx={{ color: '#648895', fontSize: 7.5 }}>
                      UNITÉS CONCERNÉES
                    </Typography>
                    <Typography sx={{ color: pf.functional.green, fontSize: 11, fontWeight: 800 }}>
                      {alert.unit_ids.length}
                    </Typography>
                  </Box>
                  <Box>
                    <Typography sx={{ color: '#648895', fontSize: 7.5 }}>
                      GAIN ETA ATTENDU
                    </Typography>
                    <Typography sx={{ color: pf.functional.green, fontSize: 11, fontWeight: 800 }}>
                      −{recommendation?.expected_gain_h ?? 1.4} h
                    </Typography>
                  </Box>
                  <Box>
                    <Typography sx={{ color: '#648895', fontSize: 7.5 }}>AGIR AVANT</Typography>
                    <Typography sx={{ color, fontSize: 11, fontWeight: 800 }}>
                      {formatDateTime(alert.deadline_at)}
                    </Typography>
                  </Box>
                </Stack>
                {expanded && (
                  <Box sx={{ mt: 1, p: 1, bgcolor: '#061721', borderRadius: '8px' }}>
                    <Typography sx={{ color: pf.functional.cyan, fontSize: 8, fontWeight: 900 }}>
                      PREUVES DISPONIBLES
                    </Typography>
                    <Typography sx={{ color: '#8BA8B3', fontSize: 8.5, mt: 0.35 }}>
                      {alert.message} Impact : {alert.impact}
                    </Typography>
                  </Box>
                )}
                <Stack direction="row" gap={0.7} mt={1.15}>
                  <Button
                    component={Link}
                    to={paths.simulation}
                    startIcon={<IconifyIcon icon="lucide:play" />}
                    sx={{
                      color: '#06141D',
                      bgcolor: pf.functional.cyan,
                      borderRadius: '8px',
                      fontSize: 8.5,
                      '&:hover': { bgcolor: '#72E6DC' },
                    }}
                  >
                    Simuler
                  </Button>
                  <Button
                    onClick={() => setEvidence(expanded ? '' : alert.alert_id)}
                    sx={{
                      color: '#A8C0C9',
                      border: '1px solid #245566',
                      borderRadius: '8px',
                      fontSize: 8.5,
                    }}
                  >
                    {expanded ? 'Masquer' : 'Voir les preuves'}
                  </Button>
                  <Button
                    onClick={() => onOpenDecision(alert)}
                    sx={{ ml: { sm: 'auto' }, color, fontSize: 8.5 }}
                  >
                    Ouvrir une décision
                  </Button>
                </Stack>
              </Box>
            );
          })}
        </Stack>
      </Paper>
    </Stack>
  );
};

const AlertsView = ({
  alerts,
  onDecision,
}: {
  alerts: TowerAlert[];
  onDecision: (alert: TowerAlert) => void;
}) => (
  <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: 'repeat(3,1fr)' }, gap: 1.2 }}>
    {alerts.map((alert, index) => {
      const color = alertTone[alert.severity];
      return (
        <Paper
          key={alert.alert_id}
          sx={{
            ...panelSx,
            position: 'relative',
            overflow: 'hidden',
            animation: 'portflowSlideIn 350ms ease both',
            animationDelay: `${index * 80}ms`,
          }}
        >
          <Box sx={{ position: 'absolute', inset: '0 auto 0 0', width: 3, bgcolor: color }} />
          <Stack direction="row" alignItems="center">
            <Chip
              label={alert.severity}
              size="small"
              sx={{ height: 21, color, bgcolor: `${color}12`, fontSize: 7.5, fontWeight: 900 }}
            />
            <Typography sx={{ ml: 'auto', color: pf.text.tertiary, fontSize: 8 }}>
              échéance {formatDateTime(alert.deadline_at)}
            </Typography>
          </Stack>
          <Typography sx={{ color: pf.text.primary, fontSize: 16, fontWeight: 850, mt: 1.2 }}>
            {alert.title}
          </Typography>
          <Typography sx={{ color: pf.text.secondary, fontSize: 10, lineHeight: 1.55, mt: 0.55 }}>
            {alert.message}
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.8, mt: 1.2 }}>
            <Box sx={{ p: 0.9, bgcolor: pf.background.secondary, borderRadius: '9px' }}>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>PROBABILITÉ</Typography>
              <Typography sx={{ color, fontSize: 18, fontWeight: 850 }}>
                {Math.round(alert.probability * 100)}%
              </Typography>
            </Box>
            <Box sx={{ p: 0.9, bgcolor: pf.background.secondary, borderRadius: '9px' }}>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>
                UNITÉS EXPOSÉES
              </Typography>
              <Typography sx={{ color: pf.text.primary, fontSize: 18, fontWeight: 850 }}>
                {alert.unit_ids.length}
              </Typography>
            </Box>
          </Box>
          <Stack gap={0.8} mt={1.2}>
            <Box>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5, fontWeight: 900 }}>
                CAUSE PRINCIPALE
              </Typography>
              <Typography sx={{ color: pf.text.secondary, fontSize: 9.5 }}>
                {alert.cause}
              </Typography>
            </Box>
            <Box>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5, fontWeight: 900 }}>
                IMPACT ATTENDU
              </Typography>
              <Typography sx={{ color: pf.text.secondary, fontSize: 9.5 }}>
                {alert.impact}
              </Typography>
            </Box>
            <Box
              sx={{
                p: 1,
                bgcolor: `${pf.functional.cyan}0b`,
                border: `1px solid ${pf.functional.cyan}25`,
                borderRadius: '10px',
              }}
            >
              <Typography sx={{ color: pf.functional.cyan, fontSize: 7.5, fontWeight: 900 }}>
                ACTION RECOMMANDÉE
              </Typography>
              <Typography sx={{ color: pf.text.primary, fontSize: 10, fontWeight: 700, mt: 0.25 }}>
                {alert.recommendation}
              </Typography>
            </Box>
          </Stack>
          <Button
            fullWidth
            onClick={() => onDecision(alert)}
            startIcon={<IconifyIcon icon="lucide:list-plus" />}
            sx={{
              mt: 1.3,
              color: pf.background.primary,
              bgcolor: color,
              borderRadius: '9px',
              fontSize: 9,
              '&:hover': { bgcolor: color, filter: 'brightness(1.08)' },
            }}
          >
            Ouvrir une décision
          </Button>
        </Paper>
      );
    })}
  </Box>
);

const DecisionsView = ({
  decisions,
  busyId,
  onAdvance,
}: {
  decisions: TowerDecision[];
  busyId: string;
  onAdvance: (decision: TowerDecision) => void;
}) => (
  <Box
    sx={{
      display: 'grid',
      gridTemplateColumns: { xs: '1fr', md: 'repeat(3,1fr)', xl: 'repeat(6,1fr)' },
      gap: 0.8,
      alignItems: 'start',
    }}
  >
    {decisionStatuses.map((status, statusIndex) => {
      const rows = decisions.filter((item) => item.status === status);
      const color = [
        pf.functional.amber,
        pf.functional.blue,
        pf.functional.purple,
        pf.functional.cyan,
        pf.functional.green,
        pf.text.tertiary,
      ][statusIndex];
      return (
        <Paper key={status} sx={{ ...panelSx, p: 1, minHeight: 270 }}>
          <Stack direction="row" alignItems="center" mb={1}>
            <Box sx={{ width: 7, height: 7, borderRadius: '50%', bgcolor: color }} />
            <Typography sx={{ ml: 0.65, color: pf.text.primary, fontSize: 9, fontWeight: 850 }}>
              {status.toUpperCase()}
            </Typography>
            <Chip
              label={rows.length}
              size="small"
              sx={{ ml: 'auto', height: 17, color, bgcolor: `${color}12`, fontSize: 7 }}
            />
          </Stack>
          <Stack gap={0.7}>
            {rows.map((decision) => (
              <Box
                key={decision.decision_id}
                sx={{
                  p: 0.9,
                  bgcolor: pf.background.secondary,
                  border: `1px solid ${pf.structure.borderSoft}`,
                  borderRadius: '10px',
                }}
              >
                <Typography sx={{ color: pf.text.primary, fontSize: 10, fontWeight: 800 }}>
                  {decision.title}
                </Typography>
                <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5, mt: 0.35 }}>
                  {decision.decision_id} · {decision.assignee}
                </Typography>
                <Typography sx={{ color: pf.text.secondary, fontSize: 8, mt: 0.55 }}>
                  Avant {formatDateTime(decision.due_at)}
                </Typography>
                {statusIndex < decisionStatuses.length - 1 && (
                  <Button
                    disabled={busyId === decision.decision_id}
                    onClick={() => onAdvance(decision)}
                    endIcon={
                      busyId === decision.decision_id ? (
                        <CircularProgress size={10} />
                      ) : (
                        <IconifyIcon icon="lucide:arrow-right" />
                      )
                    }
                    sx={{
                      width: 1,
                      mt: 0.8,
                      minHeight: 25,
                      color,
                      bgcolor: `${color}10`,
                      borderRadius: '7px',
                      fontSize: 7.5,
                    }}
                  >
                    {decisionStatuses[statusIndex + 1]}
                  </Button>
                )}
              </Box>
            ))}
          </Stack>
        </Paper>
      );
    })}
  </Box>
);

const SimulationView = ({ stages }: { stages: TowerStage[] }) => {
  const [payload, setPayload] = useState<SimulationPayload>({
    stage: 'SCAN',
    capacity_boost: 18,
    duration_h: 4,
    arrival_change_pct: 0,
    route_policy: 'CURRENT',
  });
  const [result, setResult] = useState<SimulationResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const run = async () => {
    setLoading(true);
    setError('');
    try {
      setResult(await controlTowerApi.simulate(payload));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Simulation indisponible');
    } finally {
      setLoading(false);
    }
  };
  const comparisons: Array<[keyof SimulationResult['before'], string, string]> = [
    ['max_backlog', 'Backlog maximal', 'unités'],
    ['ge24_units', 'Unités ≥ 24 h', 'unités'],
    ['mean_eta_p90_h', 'ETA prudente moyenne', 'h'],
    ['recovery_h', 'Retour à la normale', 'h'],
  ];
  return (
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '.72fr 1.28fr' }, gap: 1.3 }}>
      <Paper sx={panelSx}>
        <SectionTitle
          icon="lucide:sliders-horizontal"
          title="Hypothèses du scénario"
          caption="Aucune action n’est exécutée automatiquement"
        />
        <Stack gap={1.45}>
          <FormControl size="small">
            <Typography sx={{ color: pf.text.tertiary, fontSize: 8, mb: 0.5 }}>
              ZONE RENFORCÉE
            </Typography>
            <Select
              value={payload.stage}
              onChange={(event) => setPayload({ ...payload, stage: event.target.value })}
              sx={{ color: pf.text.primary, bgcolor: pf.background.secondary }}
            >
              {stages.map((stage) => (
                <MenuItem key={stage.code} value={stage.code}>
                  {stage.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          <Box>
            <Stack direction="row" justifyContent="space-between">
              <Typography sx={{ color: pf.text.secondary, fontSize: 9 }}>
                Capacité additionnelle
              </Typography>
              <Typography sx={{ color: pf.functional.cyan, fontSize: 10, fontWeight: 850 }}>
                +{payload.capacity_boost} unités/h
              </Typography>
            </Stack>
            <Slider
              value={payload.capacity_boost}
              onChange={(_, value) => setPayload({ ...payload, capacity_boost: value as number })}
              min={0}
              max={60}
              sx={{ color: pf.functional.cyan }}
            />
          </Box>
          <Box>
            <Stack direction="row" justifyContent="space-between">
              <Typography sx={{ color: pf.text.secondary, fontSize: 9 }}>
                Durée du renfort
              </Typography>
              <Typography sx={{ color: pf.functional.purple, fontSize: 10, fontWeight: 850 }}>
                {payload.duration_h} h
              </Typography>
            </Stack>
            <Slider
              value={payload.duration_h}
              onChange={(_, value) => setPayload({ ...payload, duration_h: value as number })}
              min={1}
              max={12}
              sx={{ color: pf.functional.purple }}
            />
          </Box>
          <Box>
            <Stack direction="row" justifyContent="space-between">
              <Typography sx={{ color: pf.text.secondary, fontSize: 9 }}>
                Variation des arrivées
              </Typography>
              <Typography sx={{ color: pf.functional.amber, fontSize: 10, fontWeight: 850 }}>
                {payload.arrival_change_pct > 0 ? '+' : ''}
                {payload.arrival_change_pct}%
              </Typography>
            </Stack>
            <Slider
              value={payload.arrival_change_pct}
              onChange={(_, value) =>
                setPayload({ ...payload, arrival_change_pct: value as number })
              }
              min={-30}
              max={60}
              sx={{ color: pf.functional.amber }}
            />
          </Box>
          <FormControl size="small">
            <Typography sx={{ color: pf.text.tertiary, fontSize: 8, mb: 0.5 }}>
              POLITIQUE DE ROUTE
            </Typography>
            <Select
              value={payload.route_policy}
              onChange={(event) =>
                setPayload({
                  ...payload,
                  route_policy: event.target.value as SimulationPayload['route_policy'],
                })
              }
              sx={{ color: pf.text.primary, bgcolor: pf.background.secondary }}
            >
              <MenuItem value="CURRENT">Règle actuelle</MenuItem>
              <MenuItem value="DIRECT">Favoriser la route directe</MenuItem>
              <MenuItem value="PV">Renforcer le passage PV</MenuItem>
            </Select>
          </FormControl>
          <Button
            onClick={run}
            disabled={loading}
            startIcon={
              loading ? <CircularProgress size={13} /> : <IconifyIcon icon="lucide:play" />
            }
            sx={{ color: pf.background.primary, bgcolor: pf.functional.cyan, borderRadius: '10px' }}
          >
            Simuler le scénario
          </Button>
          {error && <Alert severity="error">{error}</Alert>}
        </Stack>
      </Paper>
      <Paper sx={panelSx}>
        <SectionTitle
          icon="lucide:git-compare-arrows"
          title="Comparaison avant / après"
          caption="Projection indicative du moteur what-if"
        />
        {!result ? (
          <Box sx={{ minHeight: 380, display: 'grid', placeItems: 'center', textAlign: 'center' }}>
            <Box>
              <IconifyIcon
                icon="lucide:flask-conical"
                sx={{ color: pf.text.tertiary, fontSize: 42 }}
              />
              <Typography sx={{ color: pf.text.secondary, fontSize: 12, mt: 1 }}>
                Configurez puis exécutez un scénario.
              </Typography>
            </Box>
          </Box>
        ) : (
          <Stack gap={1.2}>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: 'repeat(2,1fr)', md: 'repeat(4,1fr)' },
                gap: 0.8,
              }}
            >
              {comparisons.map(([key, label, unit]) => {
                const gain = result.before[key] - result.after[key];
                return (
                  <Box
                    key={key}
                    sx={{
                      p: 1.15,
                      bgcolor: pf.background.secondary,
                      borderRadius: '11px',
                      border: `1px solid ${gain > 0 ? `${pf.functional.green}35` : pf.structure.borderSoft}`,
                    }}
                  >
                    <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>
                      {label.toUpperCase()}
                    </Typography>
                    <Stack direction="row" alignItems="baseline" gap={0.4} mt={0.65}>
                      <Typography
                        sx={{
                          color: pf.text.tertiary,
                          fontSize: 12,
                          textDecoration: 'line-through',
                        }}
                      >
                        {result.before[key]}
                      </Typography>
                      <IconifyIcon
                        icon="lucide:arrow-right"
                        sx={{ color: pf.text.tertiary, fontSize: 11 }}
                      />
                      <Typography
                        sx={{
                          color: gain > 0 ? pf.functional.green : pf.functional.amber,
                          fontSize: 21,
                          fontWeight: 850,
                        }}
                      >
                        {result.after[key]}
                      </Typography>
                      <Typography sx={{ color: pf.text.tertiary, fontSize: 8 }}>{unit}</Typography>
                    </Stack>
                    <Typography sx={{ color: pf.functional.green, fontSize: 8, mt: 0.4 }}>
                      gain {gain.toFixed(1)} {unit}
                    </Typography>
                  </Box>
                );
              })}
            </Box>
            <Box
              sx={{
                p: 1.25,
                bgcolor: pf.functional.greenSoft,
                border: `1px solid ${pf.functional.green}30`,
                borderRadius: '12px',
              }}
            >
              <Typography sx={{ color: pf.functional.green, fontSize: 8, fontWeight: 900 }}>
                LECTURE DU SCÉNARIO · CONFIANCE {Math.round(result.confidence * 100)}%
              </Typography>
              <Typography sx={{ color: pf.text.primary, fontSize: 11, fontWeight: 700, mt: 0.4 }}>
                {result.recommendation}
              </Typography>
            </Box>
            <Alert severity="info" icon={<IconifyIcon icon="lucide:user-check" />}>
              Ce résultat prépare la décision. Il ne déclenche jamais une action opérationnelle.
            </Alert>
          </Stack>
        )}
      </Paper>
    </Box>
  );
};

const ControlTowerPage = ({ view }: { view: ControlTowerView }) => {
  const [snapshot, setSnapshot] = useState<ControlTowerSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState('');
  const [selectedStage, setSelectedStage] = useState('');
  const [selectedUnit, setSelectedUnit] = useState('');
  const [unitDetail, setUnitDetail] = useState<TowerUnitDetail | null>(null);
  const [unitLoading, setUnitLoading] = useState(false);
  const [selectedVessel, setSelectedVessel] = useState('');
  const [query, setQuery] = useState('');
  const [tier, setTier] = useState<'TOUS' | TowerUnit['tier']>('TOUS');
  const [riskOnly, setRiskOnly] = useState(false);
  const [horizon, setHorizon] = useState<(typeof horizons)[number]>(6);
  const [operationMessage, setOperationMessage] = useState('');
  const [busyDecision, setBusyDecision] = useState('');

  const refresh = useCallback(async (silent = false) => {
    if (!silent) setRefreshing(true);
    setError('');
    try {
      setSnapshot(await controlTowerApi.getSnapshot());
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'La Control Tower est indisponible.');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    void refresh(true);
    const timer = window.setInterval(() => void refresh(true), 30_000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!selectedUnit) {
      setUnitDetail(null);
      return;
    }
    const controller = new AbortController();
    setUnitLoading(true);
    controlTowerApi
      .getUnit(selectedUnit, controller.signal)
      .then(setUnitDetail)
      .catch((reason) => {
        if (reason instanceof DOMException && reason.name === 'AbortError') return;
        setOperationMessage(reason instanceof Error ? reason.message : 'Fiche indisponible');
      })
      .finally(() => setUnitLoading(false));
    return () => controller.abort();
  }, [selectedUnit]);

  const filteredUnits = useMemo(() => {
    if (!snapshot) return [];
    const normalized = query.trim().toLocaleLowerCase('fr-FR');
    return snapshot.units.filter(
      (unit) =>
        (tier === 'TOUS' || unit.tier === tier) &&
        (!selectedStage || unit.stage === selectedStage) &&
        (!normalized ||
          `${unit.unit_id} ${unit.stage_label} ${unit.cause} ${unit.route}`
            .toLocaleLowerCase('fr-FR')
            .includes(normalized)),
    );
  }, [query, selectedStage, snapshot, tier]);

  const createDecision = async (source: TowerAlert | TowerUnitDetail) => {
    try {
      const alert = 'alert_id' in source;
      const decision = await controlTowerApi.createDecision({
        title: alert ? source.recommendation : `Revue prioritaire ${source.unit_id}`,
        description: alert ? `${source.title} — ${source.message}` : source.explanation,
        alert_id: alert ? source.alert_id : undefined,
        unit_ids: alert ? source.unit_ids : [source.unit_id],
      });
      setOperationMessage(`${decision.decision_id} créée et ajoutée au pilotage.`);
      await refresh(true);
    } catch (reason) {
      setOperationMessage(
        reason instanceof Error ? reason.message : 'Impossible de créer la décision.',
      );
    }
  };

  const advanceDecision = async (decision: TowerDecision) => {
    const index = decisionStatuses.indexOf(decision.status);
    if (index < 0 || index >= decisionStatuses.length - 1) return;
    setBusyDecision(decision.decision_id);
    try {
      await controlTowerApi.updateDecision(decision.decision_id, {
        status: decisionStatuses[index + 1],
        comment: `Passage vers ${decisionStatuses[index + 1]} depuis la Control Tower.`,
      });
      setOperationMessage(
        `${decision.decision_id} passe à l’étape « ${decisionStatuses[index + 1]} ».`,
      );
      await refresh(true);
    } catch (reason) {
      setOperationMessage(reason instanceof Error ? reason.message : 'Mise à jour impossible.');
    } finally {
      setBusyDecision('');
    }
  };

  const page = viewCopy[view];

  if (loading)
    return (
      <Box sx={{ minHeight: '70vh', display: 'grid', placeItems: 'center' }}>
        <Stack alignItems="center" gap={1.2}>
          <CircularProgress sx={{ color: pf.functional.cyan }} />
          <Typography sx={{ color: pf.text.secondary, fontSize: 11 }}>
            Construction de la situation opérationnelle…
          </Typography>
        </Stack>
      </Box>
    );
  if (!snapshot)
    return (
      <Alert severity="error" action={<Button onClick={() => refresh()}>Réessayer</Button>}>
        {error || 'Aucune situation disponible.'}
      </Alert>
    );

  const selectedStageData =
    snapshot.stages.find((stage) => stage.code === selectedStage) ?? snapshot.stages[0];
  const selectedVesselData =
    snapshot.vessels.find((vessel) => vessel.vessel_id === selectedVessel) ?? snapshot.vessels[0];

  return (
    <Stack gap={1.6}>
      <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ sm: 'center' }} gap={1.1}>
        <Box
          sx={{
            width: 42,
            height: 42,
            display: 'grid',
            placeItems: 'center',
            color: pf.functional.cyan,
            bgcolor: pf.functional.cyanSoft,
            border: `1px solid ${pf.functional.cyan}38`,
            borderRadius: '12px',
            boxShadow: `0 0 24px ${pf.functional.cyan}0d`,
          }}
        >
          <IconifyIcon icon={page.icon} sx={{ fontSize: 21 }} />
        </Box>
        <Box>
          <Typography
            sx={{
              color: pf.functional.cyan,
              fontSize: 8.5,
              fontWeight: 900,
              letterSpacing: '.13em',
            }}
          >
            {page.eyebrow}
          </Typography>
          <Typography
            component="h1"
            sx={{
              color: pf.text.primary,
              fontSize: { xs: 22, md: 27 },
              lineHeight: 1.15,
              fontWeight: 850,
              letterSpacing: '-.025em',
            }}
          >
            {page.title}
          </Typography>
          <Typography sx={{ color: pf.text.secondary, fontSize: 10.5, mt: 0.25 }}>
            {page.description}
          </Typography>
        </Box>
        <Stack direction="row" alignItems="center" gap={0.65} ml={{ sm: 'auto' }}>
          <Chip
            icon={<IconifyIcon icon="lucide:flask-conical" />}
            label="DONNÉES D’EXERCICE"
            size="small"
            sx={{
              height: 26,
              color: pf.functional.amber,
              bgcolor: pf.functional.amberSoft,
              border: `1px solid ${pf.functional.amber}35`,
              fontSize: 7.5,
              fontWeight: 900,
              '& .MuiChip-icon': { color: pf.functional.amber },
            }}
          />
          <Chip
            icon={<IconifyIcon icon="lucide:radio" />}
            label={`ACTUALISÉE · ${relativeMinutes(snapshot.generated_at)} MIN`}
            size="small"
            sx={{
              height: 26,
              color: pf.functional.green,
              bgcolor: pf.functional.greenSoft,
              border: `1px solid ${pf.functional.green}30`,
              fontSize: 7.5,
              '& .MuiChip-icon': { color: pf.functional.green },
            }}
          />
          <Button
            onClick={() => refresh()}
            disabled={refreshing}
            aria-label="Actualiser la situation"
            sx={{
              minWidth: 34,
              width: 34,
              height: 34,
              p: 0,
              color: pf.functional.cyan,
              border: `1px solid ${pf.structure.border}`,
              borderRadius: '9px',
            }}
          >
            <IconifyIcon
              icon="lucide:refresh-cw"
              sx={{ animation: refreshing ? 'spin 1s linear infinite' : 'none' }}
            />
          </Button>
        </Stack>
      </Stack>

      {error && (
        <Alert severity="warning" onClose={() => setError('')}>
          {error} La dernière situation connue reste affichée.
        </Alert>
      )}
      {operationMessage && (
        <Alert
          severity="success"
          onClose={() => setOperationMessage('')}
          sx={{
            bgcolor: pf.functional.greenSoft,
            color: pf.text.primary,
            border: `1px solid ${pf.functional.green}30`,
          }}
        >
          {operationMessage}
        </Alert>
      )}

      {view === 'overview' && (
        <OverviewView
          snapshot={snapshot}
          selectedStage={selectedStage}
          setSelectedStage={setSelectedStage}
          selectedUnit={selectedUnit}
          setSelectedUnit={setSelectedUnit}
          onOpenDecision={createDecision}
        />
      )}

      {view === 'units' && (
        <>
          <Paper sx={panelSx}>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: 'repeat(3,1fr)' },
                gap: 0.8,
                mb: 1.2,
              }}
            >
              {[
                {
                  label: 'À examiner maintenant',
                  value: snapshot.units.filter((unit) => unit.tier !== 'NORMAL').length,
                  color: pf.functional.amber,
                  icon: 'lucide:list-checks',
                },
                {
                  label: 'Risque de durée ≥ 24 h',
                  value: snapshot.units.filter((unit) => unit.ge24 >= 0.5).length,
                  color: pf.functional.red,
                  icon: 'lucide:timer-alert',
                },
                {
                  label: 'Position à confirmer',
                  value: snapshot.units.filter((unit) => unit.location_quality !== 'GPS actuelle')
                    .length,
                  color: pf.functional.blue,
                  icon: 'lucide:map-pin-check-inside',
                },
              ].map((metric) => (
                <Box
                  key={metric.label}
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 1,
                    p: 1,
                    bgcolor: '#061A24',
                    border: '1px solid #174453',
                    borderRadius: '10px',
                  }}
                >
                  <Box
                    sx={{
                      width: 32,
                      height: 32,
                      display: 'grid',
                      placeItems: 'center',
                      color: metric.color,
                      bgcolor: `${metric.color}12`,
                      border: `1px solid ${metric.color}35`,
                      borderRadius: '8px',
                    }}
                  >
                    <IconifyIcon icon={metric.icon} sx={{ fontSize: 16 }} />
                  </Box>
                  <Box>
                    <Typography sx={{ color: metric.color, fontSize: 17, fontWeight: 900 }}>
                      {metric.value}
                    </Typography>
                    <Typography sx={{ color: '#7F9EA9', fontSize: 8.5 }}>{metric.label}</Typography>
                  </Box>
                </Box>
              ))}
            </Box>
            <Stack direction={{ xs: 'column', md: 'row' }} gap={0.8}>
              <TextField
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Identifiant, étape, cause ou route…"
                size="small"
                sx={{ flex: 1 }}
                InputProps={{
                  startAdornment: (
                    <InputAdornment position="start">
                      <IconifyIcon icon="lucide:search" sx={{ color: pf.text.tertiary }} />
                    </InputAdornment>
                  ),
                }}
              />
              {(['TOUS', 'CRITIQUE', 'VIGILANCE', 'NORMAL'] as const).map((value) => (
                <Button
                  key={value}
                  onClick={() => setTier(value)}
                  sx={{
                    color:
                      tier === value
                        ? pf.background.primary
                        : value === 'TOUS'
                          ? pf.text.secondary
                          : tierTone[value],
                    bgcolor:
                      tier === value
                        ? value === 'TOUS'
                          ? pf.functional.cyan
                          : tierTone[value]
                        : 'transparent',
                    border: `1px solid ${value === 'TOUS' ? pf.structure.border : `${tierTone[value]}35`}`,
                    borderRadius: '9px',
                    fontSize: 8,
                  }}
                >
                  {value}
                </Button>
              ))}
            </Stack>
          </Paper>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', xl: '1.45fr .75fr' },
              gap: 1.2,
              alignItems: 'start',
            }}
          >
            <Paper sx={panelSx}>
              <SectionTitle
                icon="lucide:list-filter"
                title={`${filteredUnits.length} unités à examiner`}
                caption="Impact × urgence × risque de durée × confiance"
              />
              <Stack gap={0.8}>
                {filteredUnits.map((unit, index) => (
                  <UnitQueueCard
                    key={unit.unit_id}
                    unit={unit}
                    rank={index + 1}
                    selected={selectedUnit === unit.unit_id}
                    onSelect={setSelectedUnit}
                  />
                ))}
              </Stack>
            </Paper>
            <UnitDetailPanel
              detail={unitDetail}
              loading={unitLoading}
              onDecision={createDecision}
            />
          </Box>
        </>
      )}

      {view === 'process' && (
        <Stack gap={1.2}>
          <Paper sx={panelSx}>
            <Stack direction="row" alignItems="center">
              <SectionTitle
                icon="lucide:route"
                title="Chaîne opérationnelle"
                caption="PV est un passage conditionnel ; la route directe rejoint le SAS"
              />
              <Stack direction="row" gap={0.35} ml="auto">
                {horizons.map((value) => (
                  <Button
                    key={value}
                    onClick={() => setHorizon(value)}
                    sx={{
                      minWidth: 38,
                      color: horizon === value ? pf.background.primary : pf.text.secondary,
                      bgcolor: horizon === value ? pf.functional.cyan : 'transparent',
                      borderRadius: '8px',
                      fontSize: 8,
                    }}
                  >
                    H+{value}
                  </Button>
                ))}
              </Stack>
            </Stack>
            <ControlTowerProcessBoard
              stages={snapshot.stages}
              selected={selectedStage}
              onSelect={setSelectedStage}
              horizon={horizon}
            />
          </Paper>
          <Box
            sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '1.5fr .5fr' }, gap: 1.2 }}
          >
            <InternalPortMap
              stages={snapshot.stages}
              units={snapshot.units}
              vessels={snapshot.vessels}
              selectedStage={selectedStage}
              selectedUnit={selectedUnit}
              onStageSelect={setSelectedStage}
              onUnitSelect={setSelectedUnit}
              riskOnly={riskOnly}
              onRiskOnlyChange={setRiskOnly}
            />
            <Paper sx={panelSx}>
              <SectionTitle
                icon="lucide:map-pin"
                title={selectedStageData.label}
                caption="Lecture de la zone sélectionnée"
              />
              <MetricCard
                label="Occupation"
                value={selectedStageData.occupancy_pct}
                unit="%"
                icon="lucide:gauge"
                color={
                  selectedStageData.occupancy_pct >= 95 ? pf.functional.red : pf.functional.amber
                }
                note={`${selectedStageData.units} / ${selectedStageData.capacity} unités`}
              />
              <Stack gap={0.8} mt={1}>
                <ProbabilityBar
                  label={`Charge prévue H+${horizon}`}
                  value={Math.min(
                    1,
                    selectedStageData.forecast[`h${horizon}`] / selectedStageData.capacity,
                  )}
                  color={pf.functional.purple}
                />
                <Typography sx={{ color: pf.text.secondary, fontSize: 9 }}>
                  Séjour médian {selectedStageData.dwell_median_h} h · prudent{' '}
                  {selectedStageData.dwell_p90_h} h
                </Typography>
                <Typography sx={{ color: pf.text.secondary, fontSize: 9 }}>
                  {selectedStageData.blocked} unités potentiellement bloquées · tendance{' '}
                  {selectedStageData.trend.toLowerCase()}
                </Typography>
              </Stack>
              <Divider sx={{ my: 1.2, borderColor: pf.structure.border }} />
              <Typography sx={{ color: pf.text.tertiary, fontSize: 8, fontWeight: 900, mb: 0.7 }}>
                UNITÉS LES PLUS PRIORITAIRES
              </Typography>
              <Stack gap={0.55}>
                {snapshot.units
                  .filter((unit) => unit.stage === selectedStageData.code)
                  .slice(0, 5)
                  .map((unit) => (
                    <UnitRow
                      key={unit.unit_id}
                      unit={unit}
                      selected={selectedUnit === unit.unit_id}
                      onSelect={setSelectedUnit}
                    />
                  ))}
              </Stack>
            </Paper>
          </Box>
        </Stack>
      )}

      {view === 'forecast' && (
        <Stack gap={1.2}>
          <Paper sx={panelSx}>
            <SectionTitle
              icon="lucide:chart-spline"
              title="Projection consolidée à 24 heures"
              caption="Arrivées, sorties, intervalle prudent et deux scénarios de capacité"
            />
            <ControlTowerForecastChart data={snapshot.forecast} height={430} />
          </Paper>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: 'repeat(2,1fr)', md: 'repeat(5,1fr)' },
              gap: 0.8,
            }}
          >
            {horizons.map((value) => {
              const point = snapshot.forecast[value - 1];
              const gap = point.backlog_p90 - point.normal_capacity;
              return (
                <ButtonBase
                  key={value}
                  onClick={() => setHorizon(value)}
                  sx={{
                    ...panelSx,
                    display: 'block',
                    textAlign: 'left',
                    borderColor: horizon === value ? pf.functional.purple : pf.structure.border,
                  }}
                >
                  <Typography sx={{ color: pf.functional.purple, fontSize: 8, fontWeight: 900 }}>
                    HORIZON H+{value}
                  </Typography>
                  <Typography
                    sx={{ color: pf.text.primary, fontSize: 23, fontWeight: 850, mt: 0.4 }}
                  >
                    {point.backlog_p50}
                  </Typography>
                  <Typography sx={{ color: pf.text.tertiary, fontSize: 8 }}>
                    backlog central · P90 {point.backlog_p90}
                  </Typography>
                  <Typography
                    sx={{
                      color: gap > 0 ? pf.functional.red : pf.functional.green,
                      fontSize: 8.5,
                      fontWeight: 800,
                      mt: 0.55,
                    }}
                  >
                    {gap > 0 ? `+${gap} au-dessus` : `${Math.abs(gap)} sous`} capacité normale
                  </Typography>
                </ButtonBase>
              );
            })}
          </Box>
          <Paper sx={panelSx}>
            <SectionTitle
              icon="lucide:warehouse"
              title={`Charge par étape à H+${horizon}`}
              caption="Cliquez sur un horizon pour recalculer la lecture"
            />
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', sm: 'repeat(2,1fr)', lg: 'repeat(4,1fr)' },
                gap: 0.8,
              }}
            >
              {snapshot.stages.map((stage) => {
                const projected = stage.forecast[`h${horizon}`];
                const pct = (projected / stage.capacity) * 100;
                const color =
                  pct >= 95
                    ? pf.functional.red
                    : pct >= 82
                      ? pf.functional.amber
                      : pf.functional.green;
                return (
                  <Box
                    key={stage.code}
                    sx={{
                      p: 1.1,
                      bgcolor: pf.background.secondary,
                      borderRadius: '10px',
                      border: `1px solid ${color}25`,
                    }}
                  >
                    <Stack direction="row">
                      <Typography sx={{ color: pf.text.primary, fontSize: 10, fontWeight: 800 }}>
                        {stage.label}
                      </Typography>
                      <Typography sx={{ ml: 'auto', color, fontSize: 10, fontWeight: 850 }}>
                        {pct.toFixed(0)}%
                      </Typography>
                    </Stack>
                    <LinearProgress
                      variant="determinate"
                      value={Math.min(100, pct)}
                      sx={{
                        mt: 0.75,
                        height: 5,
                        borderRadius: 4,
                        bgcolor: pf.background.primary,
                        '& .MuiLinearProgress-bar': { bgcolor: color },
                      }}
                    />
                    <Typography sx={{ color: pf.text.tertiary, fontSize: 8, mt: 0.5 }}>
                      {projected} unités prévues / {stage.capacity}
                    </Typography>
                  </Box>
                );
              })}
            </Box>
          </Paper>
        </Stack>
      )}

      {view === 'alerts' && <AlertsView alerts={snapshot.alerts} onDecision={createDecision} />}
      {view === 'decisions' && (
        <DecisionsView
          decisions={snapshot.decisions}
          busyId={busyDecision}
          onAdvance={advanceDecision}
        />
      )}

      {view === 'vessels' && (
        <Box
          sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: '1.5fr .5fr' }, gap: 1.2 }}
        >
          <MaritimeApproachMap
            vessels={snapshot.vessels}
            selected={selectedVesselData.vessel_id}
            onSelect={setSelectedVessel}
          />
          <Stack gap={1}>
            <Paper sx={panelSx}>
              <SectionTitle
                icon="lucide:ship"
                title={selectedVesselData.name}
                caption={`${selectedVesselData.imo} · ${selectedVesselData.status}`}
              />
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 0.7 }}>
                {(
                  [
                    ['ETA recalculée', formatDateTime(selectedVesselData.predicted_eta)],
                    [
                      'Écart annoncé',
                      `${selectedVesselData.eta_delta_minutes > 0 ? '+' : ''}${selectedVesselData.eta_delta_minutes} min`,
                    ],
                    ['Distance', `${selectedVesselData.distance_nm} NM`],
                    ['Vitesse', `${selectedVesselData.speed_kn} nd`],
                    ['Terminal', selectedVesselData.terminal],
                    ['Fenêtre', selectedVesselData.berth_window],
                  ] as const
                ).map(([label, value]) => (
                  <Box
                    key={label}
                    sx={{ p: 0.8, bgcolor: pf.background.secondary, borderRadius: '9px' }}
                  >
                    <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>
                      {label.toUpperCase()}
                    </Typography>
                    <Typography
                      sx={{ color: pf.text.primary, fontSize: 11, fontWeight: 800, mt: 0.2 }}
                    >
                      {value}
                    </Typography>
                  </Box>
                ))}
              </Box>
              <ProbabilityBar
                label="Risque de congestion à l’arrivée"
                value={selectedVesselData.congestion_risk}
                color={
                  selectedVesselData.congestion_risk >= 0.7
                    ? pf.functional.red
                    : pf.functional.amber
                }
              />
              <Typography sx={{ color: pf.text.secondary, fontSize: 9.5, mt: 1 }}>
                {selectedVesselData.units_ready} / {selectedVesselData.associated_units} unités
                associées déjà prêtes.
              </Typography>
              <Typography sx={{ color: pf.text.tertiary, fontSize: 8, mt: 0.5 }}>
                AIS {selectedVesselData.ais_quality.toLowerCase()} · dernière position il y a{' '}
                {selectedVesselData.ais_age_minutes} min
              </Typography>
            </Paper>
            <Paper sx={panelSx}>
              <SectionTitle icon="lucide:list" title="Prochaines approches" />
              <Stack gap={0.5}>
                {snapshot.vessels.map((vessel) => (
                  <ButtonBase
                    key={vessel.vessel_id}
                    onClick={() => setSelectedVessel(vessel.vessel_id)}
                    sx={{
                      p: 0.8,
                      display: 'flex',
                      textAlign: 'left',
                      borderRadius: '9px',
                      bgcolor:
                        vessel.vessel_id === selectedVesselData.vessel_id
                          ? pf.functional.cyanSoft
                          : pf.background.secondary,
                    }}
                  >
                    <Box>
                      <Typography sx={{ color: pf.text.primary, fontSize: 9.5, fontWeight: 750 }}>
                        {vessel.name}
                      </Typography>
                      <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>
                        {vessel.distance_nm} NM · {vessel.terminal}
                      </Typography>
                    </Box>
                    <Typography
                      sx={{
                        ml: 'auto',
                        color:
                          vessel.congestion_risk >= 0.7 ? pf.functional.red : pf.functional.green,
                        fontSize: 9,
                        fontWeight: 850,
                      }}
                    >
                      {Math.round(vessel.congestion_risk * 100)}%
                    </Typography>
                  </ButtonBase>
                ))}
              </Stack>
            </Paper>
          </Stack>
        </Box>
      )}

      {view === 'simulation' && <SimulationView stages={snapshot.stages} />}

      {view === 'quality' && (
        <Stack gap={1.2}>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(2,1fr)', xl: 'repeat(4,1fr)' },
              gap: 1,
            }}
          >
            {snapshot.sources.map((source) => {
              const color =
                source.status === 'À JOUR'
                  ? pf.functional.green
                  : source.status === 'PARTIEL'
                    ? pf.functional.amber
                    : source.status === 'SHADOW'
                      ? pf.functional.purple
                      : pf.functional.blue;
              return (
                <Paper key={source.source} sx={panelSx}>
                  <Stack direction="row" alignItems="center">
                    <IconifyIcon icon="lucide:database" sx={{ color, fontSize: 18 }} />
                    <Chip
                      label={source.status}
                      size="small"
                      sx={{ ml: 'auto', height: 20, color, bgcolor: `${color}12`, fontSize: 7 }}
                    />
                  </Stack>
                  <Typography sx={{ color: pf.text.primary, fontSize: 13, fontWeight: 800, mt: 1 }}>
                    {source.source}
                  </Typography>
                  <Typography sx={{ color: pf.text.tertiary, fontSize: 8.5, mt: 0.2 }}>
                    {source.detail}
                  </Typography>
                  <Stack direction="row" alignItems="baseline" gap={0.4} mt={1}>
                    <Typography sx={{ color, fontSize: 24, fontWeight: 850 }}>
                      {source.completeness_pct}
                    </Typography>
                    <Typography sx={{ color: pf.text.tertiary, fontSize: 8 }}>
                      % complétude
                    </Typography>
                  </Stack>
                  <LinearProgress
                    variant="determinate"
                    value={source.completeness_pct}
                    sx={{
                      height: 5,
                      borderRadius: 3,
                      bgcolor: pf.background.secondary,
                      '& .MuiLinearProgress-bar': { bgcolor: color },
                    }}
                  />
                  <Typography sx={{ color: pf.text.tertiary, fontSize: 8, mt: 0.7 }}>
                    Fraîcheur : {source.age_minutes} min
                  </Typography>
                </Paper>
              );
            })}
          </Box>
          <Paper sx={panelSx}>
            <SectionTitle
              icon="lucide:shield-check"
              title="Contrat de confiance opérationnel"
              caption="Les limites sont visibles au même endroit que les indicateurs"
            />
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: { xs: '1fr', md: 'repeat(3,1fr)' },
                gap: 0.8,
              }}
            >
              {[
                [
                  'Aucune position inventée',
                  'Une zone métier est distinguée d’une coordonnée GPS exacte.',
                  'lucide:map-pin-check',
                ],
                [
                  'Prévisions sous contrôle',
                  'Le moteur réel sera activé uniquement après validation de fraîcheur et de performance.',
                  'lucide:brain-circuit',
                ],
                [
                  'Décision humaine conservée',
                  'Recommandations et simulations ne déclenchent aucune action automatique.',
                  'lucide:user-check',
                ],
              ].map(([title, text, icon]) => (
                <Box
                  key={title}
                  sx={{ p: 1.15, bgcolor: pf.background.secondary, borderRadius: '11px' }}
                >
                  <IconifyIcon icon={icon} sx={{ color: pf.functional.cyan, fontSize: 20 }} />
                  <Typography
                    sx={{ color: pf.text.primary, fontSize: 10.5, fontWeight: 800, mt: 0.65 }}
                  >
                    {title}
                  </Typography>
                  <Typography
                    sx={{ color: pf.text.secondary, fontSize: 8.5, lineHeight: 1.5, mt: 0.3 }}
                  >
                    {text}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        </Stack>
      )}

      {view === 'audit' && (
        <Paper sx={panelSx}>
          <SectionTitle
            icon="lucide:history"
            title="Chronologie immuable"
            caption={`${snapshot.audit.length} événements dans la vue courante`}
          />
          <Stack>
            {snapshot.audit.map((event, index) => (
              <Stack
                key={event.event_id}
                direction="row"
                gap={1.1}
                sx={{ position: 'relative', pb: index === snapshot.audit.length - 1 ? 0 : 1.2 }}
              >
                <Box
                  sx={{
                    position: 'relative',
                    zIndex: 1,
                    width: 30,
                    height: 30,
                    display: 'grid',
                    placeItems: 'center',
                    color: pf.functional.cyan,
                    bgcolor: pf.background.panelRaised,
                    border: `1px solid ${pf.functional.cyan}45`,
                    borderRadius: '50%',
                    flexShrink: 0,
                  }}
                >
                  <IconifyIcon
                    icon={
                      event.action.includes('Décision')
                        ? 'lucide:list-checks'
                        : event.action.includes('Alerte')
                          ? 'lucide:shield-alert'
                          : 'lucide:activity'
                    }
                    sx={{ fontSize: 13 }}
                  />
                </Box>
                {index < snapshot.audit.length - 1 && (
                  <Box
                    sx={{
                      position: 'absolute',
                      left: 14.5,
                      top: 30,
                      bottom: 0,
                      width: 1,
                      bgcolor: pf.structure.border,
                    }}
                  />
                )}
                <Box
                  flex={1}
                  sx={{ p: 0.85, bgcolor: pf.background.secondary, borderRadius: '9px' }}
                >
                  <Stack direction="row">
                    <Typography sx={{ color: pf.text.primary, fontSize: 10, fontWeight: 800 }}>
                      {event.action}
                    </Typography>
                    <Chip
                      icon={<IconifyIcon icon="lucide:lock-keyhole" />}
                      label="IMMUABLE"
                      size="small"
                      sx={{
                        ml: 'auto',
                        height: 17,
                        color: pf.functional.green,
                        bgcolor: pf.functional.greenSoft,
                        fontSize: 6,
                        '& .MuiChip-icon': { color: pf.functional.green, fontSize: 10 },
                      }}
                    />
                  </Stack>
                  <Typography sx={{ color: pf.text.secondary, fontSize: 8.5, mt: 0.25 }}>
                    {event.actor} · objet {event.object}
                  </Typography>
                  <Typography sx={{ color: pf.text.tertiary, fontSize: 7.5, mt: 0.25 }}>
                    {formatDateTime(event.at)} · {event.event_id}
                  </Typography>
                </Box>
              </Stack>
            ))}
          </Stack>
        </Paper>
      )}

      {view === 'reports' && (
        <Stack gap={1.2}>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(3,1fr)' },
              gap: 1,
            }}
          >
            {[
              [
                'Rapport de quart',
                'Situation, alertes, décisions et qualité des sources.',
                'lucide:clipboard-list',
              ],
              [
                'File priorisée',
                'Unités, ETA prudentes, risques et responsables.',
                'lucide:container',
              ],
              [
                'Journal de gouvernance',
                'Trace immuable des événements et interventions.',
                'lucide:scroll-text',
              ],
            ].map(([title, description, icon], index) => (
              <Paper key={title} sx={panelSx}>
                <IconifyIcon
                  icon={icon}
                  sx={{
                    color: [pf.functional.cyan, pf.functional.purple, pf.functional.green][index],
                    fontSize: 25,
                  }}
                />
                <Typography sx={{ color: pf.text.primary, fontSize: 14, fontWeight: 850, mt: 1 }}>
                  {title}
                </Typography>
                <Typography
                  sx={{ color: pf.text.secondary, fontSize: 9, lineHeight: 1.5, mt: 0.35 }}
                >
                  {description}
                </Typography>
                <Button
                  onClick={async () => {
                    if (index === 0)
                      downloadJson(
                        'portflow-rapport-quart.json',
                        await controlTowerApi.getShiftReport(),
                      );
                    else if (index === 1) downloadJson('portflow-file-unites.json', snapshot.units);
                    else downloadJson('portflow-journal-audit.json', snapshot.audit);
                  }}
                  startIcon={<IconifyIcon icon="lucide:download" />}
                  sx={{
                    mt: 1.2,
                    color: pf.functional.cyan,
                    border: `1px solid ${pf.functional.cyan}30`,
                    borderRadius: '9px',
                    fontSize: 8.5,
                  }}
                >
                  Exporter JSON
                </Button>
              </Paper>
            ))}
          </Box>
          <Alert severity="info" icon={<IconifyIcon icon="lucide:info" />}>
            Le contrat d’export est opérationnel. Les gabarits PDF et Excel officiels seront
            raccordés au moteur de reporting lors de l’intégration.
          </Alert>
        </Stack>
      )}
    </Stack>
  );
};

export default ControlTowerPage;
