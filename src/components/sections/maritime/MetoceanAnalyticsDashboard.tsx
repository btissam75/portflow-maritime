import {
  Alert,
  Box,
  Button,
  Chip,
  Paper,
  Skeleton,
  Slider,
  Stack,
  Typography,
} from '@mui/material';
import { useEffect, useMemo, useState } from 'react';
import IconifyIcon from 'components/base/IconifyIcon';
import {
  KpiSparkline,
  MarineAnalyticsChart,
  MetoceanProjectionChart,
} from 'components/sections/maritime/MetoceanAnalyticsCharts';
import type { ForecastMode, ProjectionFocus } from 'components/sections/maritime/MetoceanAnalyticsCharts';
import MetoceanSituationMap from 'components/sections/maritime/MetoceanSituationMap';
import type { LiveMetoceanData } from 'types/liveMetocean';
import type { MetoceanDashboardData, MetoceanForecastPoint } from 'types/metocean';
import { portflowPalette as pf } from 'theme/portflowPalette';

interface MetoceanAnalyticsDashboardProps {
  data: MetoceanDashboardData | null;
  liveData: LiveMetoceanData | null;
  loading: boolean;
  liveLoading: boolean;
  error: string | null;
  lastUpdatedAt: string | null;
  autoRefresh: boolean;
  onAutoRefreshChange: (value: boolean) => void;
  onRefresh: () => void;
}

const tokens = {
  bg: pf.background.primary,
  surface: pf.background.panel,
  surfaceStrong: pf.background.panelRaised,
  border: pf.structure.border,
  teal: pf.functional.cyan,
  violet: pf.functional.purple,
  amber: pf.functional.amber,
  red: pf.functional.red,
  text: pf.text.primary,
  muted: pf.text.secondary,
};

const monoFont = 'Inter, "Segoe UI", Arial, sans-serif';
const displayFont = monoFont;

const glassCard = {
  position: 'relative',
  overflow: 'hidden',
  bgcolor: tokens.surface,
  border: `1px solid ${tokens.border}`,
  borderRadius: '8px',
  boxShadow: '0 10px 30px rgba(0,0,0,0.18)',
} as const;

const formatNumber = (input: number | null | undefined, digits = 1) =>
  input == null || !Number.isFinite(input) ? '—' : input.toFixed(digits);

const cardinal = (degrees: number | null | undefined) => {
  if (degrees == null || !Number.isFinite(degrees)) return '—';
  const labels = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO'];
  return labels[Math.round((((degrees % 360) + 360) % 360) / 45) % 8];
};

const weatherDescription = (code: number | null | undefined) => {
  if (code == null) return 'Condition indisponible';
  if (code === 0) return 'Ciel dégagé';
  if (code <= 3) return 'Nuages variables';
  if (code <= 48) return 'Brume ou brouillard';
  if (code <= 67) return 'Pluie';
  if (code <= 82) return 'Averses';
  return 'Risque orageux';
};

const seaState = (height: number | null | undefined) => {
  if (height == null) return { label: 'Indisponible', severity: 'NO DATA', color: tokens.muted };
  if (height < 0.5) return { label: 'Ridée', severity: 'FAIBLE', color: tokens.teal };
  if (height < 1.25) return { label: 'Peu agitée', severity: 'MODÉRÉ', color: pf.functional.blue };
  if (height < 2.5) return { label: 'Agitée', severity: 'VIGILANCE', color: tokens.amber };
  return { label: 'Forte', severity: 'CRITIQUE', color: tokens.red };
};

const trend = (values: Array<number | null>) => {
  const usable = values.filter((item): item is number => item != null && Number.isFinite(item));
  if (usable.length < 2) return null;
  return usable[usable.length - 1] - usable[0];
};

const outlookValues = (points: MetoceanForecastPoint[], variable: string) =>
  points
    .filter((point) => point.variable === variable)
    .sort((left, right) => left.horizon_h - right.horizon_h)
    .map((point) => point.p50);

const valueAtHorizon = (
  times: string[] | undefined,
  values: Array<number | null> | undefined,
  horizonHours: number,
) => {
  if (!times?.length || !values?.length) return null;
  const target = Date.now() + horizonHours * 60 * 60 * 1000;
  let bestIndex = -1;
  let bestDistance = Number.POSITIVE_INFINITY;
  times.forEach((time, index) => {
    const distance = Math.abs(new Date(time).getTime() - target);
    if (Number.isFinite(distance) && distance < bestDistance) {
      bestDistance = distance;
      bestIndex = index;
    }
  });
  return bestIndex >= 0 ? values[bestIndex] ?? null : null;
};

const valuesWithinHorizon = (
  times: string[] | undefined,
  values: Array<number | null> | undefined,
  horizonHours: number,
) => {
  if (!times?.length || !values?.length) return [];
  const now = Date.now();
  const end = now + horizonHours * 60 * 60 * 1000;
  return times.flatMap((time, index) => {
    const stamp = new Date(time).getTime();
    const value = values[index];
    return Number.isFinite(stamp)
      && stamp >= now
      && stamp <= end
      && value != null
      && Number.isFinite(value)
      ? [{ time, stamp, value }]
      : [];
  });
};

const range = (values: number[]) => {
  if (!values.length) return { min: null, max: null };
  return { min: Math.min(...values), max: Math.max(...values) };
};

const formatForecastMoment = (value: string | null) => {
  if (!value) return 'Aucun dépassement prévu';
  return new Intl.DateTimeFormat('fr-FR', {
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Africa/Casablanca',
  }).format(new Date(value));
};

const AnimatedNumber = ({
  value,
  digits = 1,
}: {
  value: number | null | undefined;
  digits?: number;
}) => {
  const [displayed, setDisplayed] = useState(0);

  useEffect(() => {
    if (value == null || !Number.isFinite(value)) return undefined;
    const start = performance.now();
    const duration = 650;
    let frame = 0;
    const update = (now: number) => {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplayed(value * eased);
      if (progress < 1) frame = window.requestAnimationFrame(update);
    };
    frame = window.requestAnimationFrame(update);
    return () => window.cancelAnimationFrame(frame);
  }, [value]);

  return <>{value == null ? '—' : displayed.toFixed(digits)}</>;
};

const KpiCard = ({
  label,
  value,
  unit,
  detail,
  icon,
  color,
  sparkline,
  delta,
  loading,
}: {
  label: string;
  value: number | null | undefined;
  unit: string;
  detail: string;
  icon: string;
  color: string;
  sparkline: Array<number | null>;
  delta: number | null;
  loading: boolean;
}) => (
  <Paper
    sx={{
      ...glassCard,
      p: 2,
      minHeight: 164,
      height: 164,
      bgcolor: tokens.surface,
      borderColor: tokens.border,
      overflow: 'hidden',
      animation: 'portflowCardIn 620ms cubic-bezier(.2,.8,.2,1) both',
    }}
  >
    <Stack height="100%" justifyContent="space-between" gap={0.8}>
      <Stack direction="row" justifyContent="space-between" alignItems="flex-start">
        <Box
          sx={{
            width: 38,
            height: 38,
            display: 'grid',
            placeItems: 'center',
            color,
            bgcolor: `${color}12`,
            border: `1px solid ${color}25`,
            borderRadius: '6px',
          }}
        >
          <IconifyIcon icon={icon} sx={{ fontSize: 21 }} />
        </Box>
        <Stack alignItems="flex-end" gap={0.3}>
          {delta != null && !loading && (
            <Typography sx={{ color: delta > 0 ? tokens.amber : tokens.teal, fontFamily: monoFont, fontSize: 10, fontWeight: 700 }}>
              {delta >= 0 ? '↗' : '↘'} {Math.abs(delta).toFixed(1)}
            </Typography>
          )}
          <Typography sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 8.5 }}>24 H</Typography>
        </Stack>
      </Stack>
      <Box>
        <Typography sx={{ color: tokens.muted, fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}>
          {label}
        </Typography>
        <Stack direction="row" alignItems="baseline" gap={0.6} mt={0.25}>
          <Typography
            aria-live="polite"
            sx={{ color: tokens.text, fontFamily: monoFont, fontSize: 29, fontWeight: 600, lineHeight: 1.15 }}
          >
            {loading ? '···' : <AnimatedNumber value={value} digits={value != null && Math.abs(value) >= 100 ? 0 : 1} />}
          </Typography>
          <Typography sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 12 }}>{unit}</Typography>
        </Stack>
        <Typography noWrap sx={{ color: tokens.muted, fontSize: 10.5, mt: 0.35 }}>{detail}</Typography>
      </Box>
      <Box sx={{ width: 1, height: 42, borderTop: `1px solid ${color}32`, pt: 0.3 }}>
        {loading ? <Skeleton width="100%" height={42} sx={{ bgcolor: 'rgba(255,255,255,0.06)' }} /> : <KpiSparkline data={sparkline} color={color} />}
      </Box>
    </Stack>
  </Paper>
);

const ProjectionResultCard = ({
  label,
  value,
  detail,
  support,
  icon,
  color,
  active = true,
}: {
  label: string;
  value: string;
  detail: string;
  support: string;
  icon: string;
  color: string;
  active?: boolean;
}) => (
  <Paper
    sx={{
      ...glassCard,
      minHeight: 166,
      p: 2,
      bgcolor: tokens.surface,
      borderColor: active ? `${color}66` : tokens.border,
      borderLeft: `3px solid ${active ? color : tokens.border}`,
    }}
  >
    <Stack height={1} justifyContent="space-between" gap={1.2}>
      <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
        <Typography sx={{ color: tokens.muted, fontSize: 10, fontWeight: 800, textTransform: 'uppercase' }}>
          {label}
        </Typography>
        <Box sx={{ width: 34, height: 34, borderRadius: '6px', display: 'grid', placeItems: 'center', color, bgcolor: `${color}18` }}>
          <IconifyIcon icon={icon} sx={{ fontSize: 20 }} />
        </Box>
      </Stack>
      <Box>
        <Typography aria-live="polite" sx={{ color: tokens.text, fontFamily: monoFont, fontSize: 23, fontWeight: 700, lineHeight: 1.2 }}>
          {value}
        </Typography>
        <Typography sx={{ color: pf.text.secondary, fontSize: 11.5, mt: 0.7 }}>{detail}</Typography>
        <Typography sx={{ color: tokens.muted, fontSize: 10, mt: 0.45 }}>{support}</Typography>
      </Box>
    </Stack>
  </Paper>
);

const SectionHeader = ({
  eyebrow,
  title,
  meta,
}: {
  eyebrow: string;
  title: string;
  meta: string;
}) => (
  <Stack direction="row" justifyContent="space-between" alignItems="flex-start" gap={1}>
    <Box>
      <Typography sx={{ color: tokens.teal, fontFamily: monoFont, fontSize: 9, fontWeight: 800, textTransform: 'uppercase' }}>
        {eyebrow}
      </Typography>
      <Typography sx={{ color: tokens.text, fontFamily: displayFont, fontSize: 16, lineHeight: '22px', fontWeight: 650, animation: 'portflowTitleIn 320ms cubic-bezier(.2,.8,.2,1) both' }}>
        {title}
      </Typography>
    </Box>
    <Typography sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 10, textAlign: 'right' }}>
      {meta}
    </Typography>
  </Stack>
);

const SegmentedMode = ({ mode, onChange }: { mode: ForecastMode; onChange: (mode: ForecastMode) => void }) => (
  <Box
    role="group"
    aria-label="Période de prévision"
    sx={{
      display: 'grid',
      gridTemplateColumns: '1fr 1fr',
      p: '3px',
      bgcolor: 'rgba(255,255,255,0.045)',
      border: `1px solid ${tokens.border}`,
      borderRadius: '6px',
      minWidth: { xs: 1, sm: 230 },
    }}
  >
    {([
      ['NEXT_24H', 'PROCHAINES 24 H'],
      ['OUTLOOK_72H', 'TENDANCE 72 H'],
    ] as const).map(([value, label]) => (
      <Button
        key={value}
        size="small"
        onClick={() => onChange(value)}
        sx={{
          minWidth: 0,
          px: 1.25,
          py: 0.75,
          borderRadius: '4px',
          color: mode === value ? tokens.bg : tokens.muted,
          bgcolor: mode === value ? tokens.teal : 'transparent',
          fontFamily: monoFont,
          fontSize: 10,
          '&:hover': { bgcolor: mode === value ? '#54E3DD' : tokens.surfaceStrong },
        }}
      >
        {label}
      </Button>
    ))}
  </Box>
);

const DirectionCompass = ({ degrees, color }: { degrees?: number | null; color: string }) => (
  <Box
    sx={{
      width: 108,
      height: 108,
      borderRadius: '50%',
      border: '1px solid rgba(255,255,255,0.14)',
      position: 'relative',
      display: 'grid',
      placeItems: 'center',
      bgcolor: 'rgba(255,255,255,0.018)',
      boxShadow: 'inset 0 0 30px rgba(0,0,0,0.35)',
      flexShrink: 0,
    }}
  >
    {['N', 'E', 'S', 'O'].map((label, index) => (
      <Typography
        key={label}
        sx={{
          position: 'absolute',
          color: index === 0 ? tokens.text : tokens.muted,
          fontFamily: monoFont,
          fontSize: 9,
          ...(index === 0 && { top: 7 }),
          ...(index === 1 && { right: 8 }),
          ...(index === 2 && { bottom: 6 }),
          ...(index === 3 && { left: 8 }),
        }}
      >
        {label}
      </Typography>
    ))}
    <IconifyIcon
      icon="material-symbols:navigation-rounded"
      sx={{
        color,
        fontSize: 39,
        transform: `rotate(${degrees ?? 0}deg)`,
        filter: `drop-shadow(0 0 8px ${color}88)`,
        transition: 'transform 650ms cubic-bezier(.16,1,.3,1)',
      }}
    />
  </Box>
);

const MetoceanAnalyticsDashboard = ({
  data,
  liveData,
  loading,
  liveLoading,
  error,
  lastUpdatedAt,
  autoRefresh,
  onAutoRefreshChange,
  onRefresh,
}: MetoceanAnalyticsDashboardProps) => {
  const [mode, setMode] = useState<ForecastMode>('NEXT_24H');
  const [projectionFocus, setProjectionFocus] = useState<ProjectionFocus>('COMBINED');
  const [projectionHorizon, setProjectionHorizon] = useState<12 | 24 | 72>(24);
  const [temperatureUnit, setTemperatureUnit] = useState<'C' | 'F'>('C');
  const [waveThreshold, setWaveThreshold] = useState(1.5);
  const [clock, setClock] = useState(() => new Date());
  const atmosphere = liveData?.atmosphere?.current;
  const atmosphereHourly = liveData?.atmosphere?.hourly;
  const marine = liveData?.marine?.current;
  const marineHourly = liveData?.marine?.hourly;
  const forecasts = data?.forecast ?? [];
  const state = seaState(marine?.wave_height);
  const unavailable = [...(liveData?.unavailable ?? []), ...(data?.unavailable ?? [])];

  const temperatureSpark = atmosphereHourly?.temperature_2m ?? [];
  const windSpark = atmosphereHourly?.wind_speed_10m ?? [];
  const waveSpark = marineHourly?.wave_height ?? [];
  const pressureSpark = atmosphereHourly?.surface_pressure ?? [];
  const maxWave = waveSpark.filter((item): item is number => item != null).reduce<number | null>(
    (maximum, item) => (maximum == null || item > maximum ? item : maximum),
    null,
  );

  const issueAt = useMemo(() => {
    const timestamps = forecasts.map((point) => point.issue_at).filter(Boolean).sort();
    return timestamps.length ? timestamps[timestamps.length - 1] : null;
  }, [forecasts]);

  const outlookTemperature = outlookValues(forecasts, 'temperature_2m');
  const outlookWave = outlookValues(forecasts, 'wave_height_m');
  const outlookWind = outlookValues(forecasts, 'wind_speed_ms');
  const outlookPressure = outlookValues(forecasts, 'pressure_hpa');
  const selectedLoading = mode === 'NEXT_24H' ? liveLoading : loading;
  const hasSelectedData = mode === 'NEXT_24H' ? Boolean(atmosphereHourly || marineHourly) : forecasts.length > 0;
  const projectedTemperatureC = valueAtHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.temperature_2m,
    projectionHorizon,
  );
  const projectedTemperature = projectedTemperatureC == null
    ? null
    : temperatureUnit === 'F'
      ? (projectedTemperatureC * 9) / 5 + 32
      : projectedTemperatureC;
  const projectedWave = valueAtHorizon(marineHourly?.time, marineHourly?.wave_height, projectionHorizon);
  const temperatureWindowC = valuesWithinHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.temperature_2m,
    projectionHorizon,
  );
  const temperatureWindow = temperatureWindowC.map((point) => ({
    ...point,
    value: temperatureUnit === 'F' ? (point.value * 9) / 5 + 32 : point.value,
  }));
  const waveWindow = valuesWithinHorizon(marineHourly?.time, marineHourly?.wave_height, projectionHorizon);
  const gustWindow = valuesWithinHorizon(atmosphereHourly?.time, atmosphereHourly?.wind_gusts_10m, projectionHorizon);
  const precipitationWindow = valuesWithinHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.precipitation_probability,
    projectionHorizon,
  );
  const temperatureRange = range(temperatureWindow.map((point) => point.value));
  const waveRange = range(waveWindow.map((point) => point.value));
  const gustRange = range(gustWindow.map((point) => point.value));
  const precipitationRange = range(precipitationWindow.map((point) => point.value));
  const thresholdPoints = waveWindow.filter((point) => point.value >= waveThreshold);
  const thresholdWindows = thresholdPoints.length;
  const firstThresholdAt = thresholdPoints[0]?.time ?? null;
  const currentTemperature = atmosphere?.temperature_2m == null
    ? null
    : temperatureUnit === 'F'
      ? (atmosphere.temperature_2m * 9) / 5 + 32
      : atmosphere.temperature_2m;
  const projectedTemperatureDelta = projectedTemperature == null || currentTemperature == null
    ? null
    : projectedTemperature - currentTemperature;
  const operationalImpact = (() => {
    if (waveRange.max == null && gustRange.max == null && precipitationRange.max == null) {
      return {
        label: 'Données insuffisantes',
        detail: 'La recommandation sera calculée dès réception des séries météo-marines.',
        color: tokens.muted,
        icon: 'material-symbols:database-off-rounded',
      };
    }
    if (
      (waveRange.max != null && waveRange.max >= Math.max(2.5, waveThreshold + 0.5))
      || (gustRange.max != null && gustRange.max >= 20)
      || (precipitationRange.max != null && precipitationRange.max >= 80)
    ) {
      return {
        label: 'Exposition forte',
        detail: 'Revoir les créneaux sensibles et confirmer les moyens disponibles.',
        color: tokens.red,
        icon: 'material-symbols:crisis-alert-rounded',
      };
    }
    if (
      thresholdWindows > 0
      || (gustRange.max != null && gustRange.max >= 14)
      || (precipitationRange.max != null && precipitationRange.max >= 60)
    ) {
      return {
        label: 'Vigilance requise',
        detail: 'Surveiller la fenêtre signalée avant de confirmer les opérations.',
        color: tokens.amber,
        icon: 'material-symbols:warning-rounded',
      };
    }
    return {
      label: 'Fenêtre favorable',
      detail: 'Aucune contrainte météo-marine majeure détectée sur cet horizon.',
      color: pf.functional.blue,
      icon: 'material-symbols:check-circle-rounded',
    };
  })();

  useEffect(() => {
    const interval = window.setInterval(() => setClock(new Date()), 60_000);
    return () => window.clearInterval(interval);
  }, []);

  const todayLabel = new Intl.DateTimeFormat('fr-FR', {
    weekday: 'long',
    day: '2-digit',
    month: 'long',
    year: 'numeric',
    timeZone: 'Africa/Casablanca',
  }).format(clock);
  const currentTime = new Intl.DateTimeFormat('fr-FR', {
    hour: '2-digit',
    minute: '2-digit',
    timeZone: 'Africa/Casablanca',
  }).format(clock);

  const activity = [
    {
      icon: 'material-symbols:satellite-alt-rounded',
      color: tokens.teal,
      title: 'Conditions actualisées',
      detail: liveData?.fetchedAt ? new Date(liveData.fetchedAt).toLocaleTimeString('fr-FR') : 'En attente',
    },
    {
      icon: 'material-symbols:calendar-clock-rounded',
      color: tokens.violet,
      title: 'Tendance 72 h disponible',
      detail: issueAt ? `Calculée ${new Date(issueAt).toLocaleString('fr-FR')}` : 'Mise à jour en attente',
    },
    {
      icon: state.color === tokens.red ? 'material-symbols:warning-rounded' : 'material-symbols:verified-rounded',
      color: state.color,
      title: `État de mer ${state.label.toLowerCase()}`,
      detail: maxWave == null ? 'Pic 24 h indisponible' : `Pic prévu ${maxWave.toFixed(2)} m sur 24 h`,
    },
    {
      icon: forecasts.length ? 'material-symbols:event-available-rounded' : 'material-symbols:event-busy-rounded',
      color: forecasts.length ? tokens.teal : tokens.amber,
      title: forecasts.length ? 'Prévisions longue échéance prêtes' : 'Prévisions longue échéance indisponibles',
      detail: forecasts.length ? 'Horizon disponible jusqu’à 72 h' : 'Nouvelle tentative à la prochaine actualisation',
    },
  ];

  return (
    <Box sx={{ color: tokens.text }}>
      <Stack gap={2}>
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          justifyContent="space-between"
          alignItems={{ xs: 'stretch', md: 'flex-end' }}
          gap={1.5}
        >
          <Box sx={{ position: 'relative', pb: 1.35 }}>
            <Stack direction="row" alignItems="center" gap={1} mb={0.65}>
              <Box sx={{ width: 7, height: 7, borderRadius: '50%', bgcolor: tokens.teal, animation: 'portflowPulse 2s infinite' }} />
              <Typography sx={{ color: tokens.teal, fontFamily: monoFont, fontSize: 10, fontWeight: 700 }}>
                TANGER MED · SUPERVISION MARITIME
              </Typography>
            </Stack>
            <Typography
              component="h1"
              sx={{
                color: tokens.text,
                fontFamily: displayFont,
                fontWeight: 700,
                fontSize: { xs: 24, md: 28 },
                lineHeight: '34px',
                animation: 'portflowTitleIn 620ms cubic-bezier(.2,.8,.2,1) both',
              }}
            >
              Météo & état de mer
            </Typography>
            <Box aria-hidden="true" sx={{ width: 72, height: 2, mt: 0.65, bgcolor: tokens.teal, transformOrigin: 'left center', animation: 'portflowTitleRule 520ms 120ms cubic-bezier(.2,.8,.2,1) both' }} />
            <Typography sx={{ color: tokens.muted, fontSize: 13, mt: 0.65 }}>
              Conditions actuelles, prévisions et vigilance sur une seule vue.
            </Typography>
            <Typography sx={{ color: pf.text.secondary, fontFamily: monoFont, fontSize: 10, mt: 0.7, textTransform: 'capitalize' }}>
              Aujourd’hui · {todayLabel} · {currentTime} à Tanger
            </Typography>
          </Box>
          <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ xs: 'stretch', sm: 'center' }} gap={1}>
            <SegmentedMode mode={mode} onChange={setMode} />
            <Button
              aria-label="Actualiser les données metocean"
              onClick={onRefresh}
              disabled={loading || liveLoading}
              sx={{
                minWidth: 40,
                height: 40,
                borderRadius: '10px',
                border: `1px solid ${tokens.border}`,
                color: tokens.text,
                bgcolor: tokens.surface,
                '&:hover': { bgcolor: tokens.surfaceStrong },
              }}
            >
              <IconifyIcon icon="material-symbols:refresh-rounded" sx={{ fontSize: 19 }} />
            </Button>
          </Stack>
        </Stack>

        {(error || unavailable.length > 0) && (
          <Alert
            severity={error ? 'error' : 'warning'}
            sx={{ bgcolor: 'rgba(255,107,107,0.08)', color: tokens.text, border: `1px solid ${tokens.border}` }}
          >
            {error ?? 'Certaines données sont momentanément indisponibles. Les informations disponibles restent affichées.'}
          </Alert>
        )}

        <Paper sx={{ ...glassCard, overflow: 'hidden' }}>
          <Box sx={{ px: { xs: 1.5, md: 2.2 }, pt: 1.8, pb: 1.2, borderBottom: `1px solid ${tokens.border}` }}>
            <SectionHeader eyebrow="Zone opérationnelle" title="Détroit de Gibraltar et approches de Tanger Med" meta="CARTE INTERACTIVE" />
          </Box>
          <MetoceanSituationMap atmosphere={atmosphere} marine={marine} height={450} />
        </Paper>

        <Paper component="section" aria-label="Réglages de projection" sx={{ ...glassCard, p: { xs: 1.5, md: 2.2 } }}>
          <SectionHeader eyebrow="Analyse prospective" title="Ajuster l’analyse" meta={`RÉSULTAT SUR ${projectionHorizon} H`} />
          <Typography sx={{ color: tokens.muted, fontSize: 11.5, mt: 0.55 }}>
            Choisissez une période et une variable. Tous les résultats situés juste dessous sont recalculés immédiatement.
          </Typography>
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '0.8fr 1.5fr', xl: '0.75fr 1.4fr 0.55fr 1.15fr' },
              gap: 1.2,
              mt: 2,
            }}
          >
            <Box sx={{ p: 1.4, border: `1px solid ${tokens.border}`, borderRadius: '6px', bgcolor: pf.background.secondary }}>
              <Typography sx={{ color: pf.functional.cyan, fontFamily: monoFont, fontSize: 9, fontWeight: 800 }}>1 · PÉRIODE À ANALYSER</Typography>
              <Typography sx={{ color: tokens.muted, fontSize: 10, mt: 0.35, mb: 1 }}>Jusqu’où regarder dans le futur</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 0.6 }}>
                {([12, 24, 72] as const).map((hours) => (
                  <Button
                    key={hours}
                    onClick={() => setProjectionHorizon(hours)}
                    sx={{
                      minWidth: 0,
                      borderRadius: '4px',
                      border: `1px solid ${projectionHorizon === hours ? pf.functional.cyan : tokens.border}`,
                      bgcolor: projectionHorizon === hours ? tokens.teal : tokens.surface,
                      color: projectionHorizon === hours ? pf.background.primary : tokens.muted,
                      fontFamily: monoFont,
                      fontSize: 11,
                    }}
                  >
                    {hours} h
                  </Button>
                ))}
              </Box>
            </Box>

            <Box sx={{ p: 1.4, border: `1px solid ${tokens.border}`, borderRadius: '6px', bgcolor: pf.background.secondary }}>
              <Typography sx={{ color: pf.functional.cyan, fontFamily: monoFont, fontSize: 9, fontWeight: 800 }}>2 · INFORMATION À AFFICHER</Typography>
              <Typography sx={{ color: tokens.muted, fontSize: 10, mt: 0.35, mb: 1 }}>Le graphique détaillé s’adapte à ce choix</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(3,1fr)' }, gap: 0.6 }}>
                {([
                  ['COMBINED', 'Vue complète', 'Température + vagues', 'material-symbols:dashboard-rounded'],
                  ['TEMPERATURE', 'Température', 'Air + ressenti', 'material-symbols:device-thermostat-rounded'],
                  ['WAVES', 'État de mer', 'Vagues + seuil', 'material-symbols:waves-rounded'],
                ] as const).map(([value, label, detail, icon]) => (
                  <Button
                    key={value}
                    onClick={() => setProjectionFocus(value)}
                    startIcon={<IconifyIcon icon={icon} sx={{ fontSize: 17 }} />}
                    sx={{
                      justifyContent: 'flex-start',
                      textAlign: 'left',
                      minWidth: 0,
                      px: 1,
                      py: 0.85,
                      borderRadius: '5px',
                      border: `1px solid ${projectionFocus === value ? pf.functional.blue : tokens.border}`,
                      bgcolor: projectionFocus === value ? pf.functional.blueSoft : tokens.surface,
                      color: tokens.text,
                    }}
                  >
                    <Box minWidth={0}>
                      <Typography noWrap sx={{ fontSize: 10.5, fontWeight: 700 }}>{label}</Typography>
                      <Typography noWrap sx={{ color: tokens.muted, fontSize: 8.5 }}>{detail}</Typography>
                    </Box>
                  </Button>
                ))}
              </Box>
            </Box>

            <Box sx={{ p: 1.4, border: `1px solid ${tokens.border}`, borderRadius: '6px', bgcolor: pf.background.secondary }}>
              <Typography sx={{ color: pf.functional.cyan, fontFamily: monoFont, fontSize: 9, fontWeight: 800 }}>3 · UNITÉ</Typography>
              <Typography sx={{ color: tokens.muted, fontSize: 10, mt: 0.35, mb: 1 }}>Température affichée</Typography>
              <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 0.6 }}>
                {(['C', 'F'] as const).map((unit) => (
                  <Button
                    key={unit}
                    onClick={() => setTemperatureUnit(unit)}
                    sx={{
                      minWidth: 0,
                      height: 38,
                      borderRadius: '4px',
                      border: `1px solid ${temperatureUnit === unit ? pf.functional.cyan : tokens.border}`,
                      color: temperatureUnit === unit ? pf.background.primary : tokens.muted,
                      bgcolor: temperatureUnit === unit ? pf.functional.cyan : tokens.surface,
                    }}
                  >
                    °{unit}
                  </Button>
                ))}
              </Box>
            </Box>

            <Box sx={{ p: 1.4, border: `1px solid ${tokens.border}`, borderRadius: '6px', bgcolor: pf.background.secondary }}>
              <Stack direction="row" justifyContent="space-between" alignItems="flex-start" gap={1}>
                <Box>
                  <Typography sx={{ color: pf.functional.cyan, fontFamily: monoFont, fontSize: 9, fontWeight: 800 }}>4 · SEUIL DE VIGILANCE</Typography>
                  <Typography sx={{ color: tokens.muted, fontSize: 10, mt: 0.35 }}>Signaler les vagues supérieures à</Typography>
                </Box>
                <Typography sx={{ color: tokens.text, fontFamily: monoFont, fontSize: 16, fontWeight: 700 }}>{waveThreshold.toFixed(1)} m</Typography>
              </Stack>
              <Slider
                value={waveThreshold}
                min={0.5}
                max={3.5}
                step={0.1}
                marks={[{ value: 0.5, label: '0,5 m' }, { value: 2, label: '2 m' }, { value: 3.5, label: '3,5 m' }]}
                onChange={(_, value) => setWaveThreshold(value as number)}
                aria-label="Seuil de vigilance de hauteur de vague"
                sx={{ color: pf.functional.blue, mt: 0.35, '& .MuiSlider-markLabel': { color: tokens.muted, fontSize: 8.5 } }}
              />
            </Box>
          </Box>
        </Paper>

        <Box component="section" aria-label="Résultat de la projection">
          <SectionHeader eyebrow="Résultat de l’analyse" title={`Ce qui est prévu sur les prochaines ${projectionHorizon} heures`} meta="MISE À JOUR IMMÉDIATE" />
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', sm: 'repeat(2,minmax(0,1fr))', xl: 'repeat(4,minmax(0,1fr))' },
              gap: 1.2,
              mt: 1.2,
            }}
          >
            <ProjectionResultCard
              label={`Température à +${projectionHorizon} h`}
              value={`${formatNumber(projectedTemperature)} °${temperatureUnit}`}
              detail={`Plage prévue : ${formatNumber(temperatureRange.min)} à ${formatNumber(temperatureRange.max)} °${temperatureUnit}`}
              support={projectedTemperatureDelta == null ? 'Écart avec maintenant indisponible' : `${projectedTemperatureDelta >= 0 ? '+' : ''}${projectedTemperatureDelta.toFixed(1)}° par rapport à maintenant`}
              icon="material-symbols:device-thermostat-rounded"
              color="#36D6CF"
              active={projectionFocus !== 'WAVES'}
            />
            <ProjectionResultCard
              label={`État de mer sur ${projectionHorizon} h`}
              value={`Pic ${formatNumber(waveRange.max, 2)} m`}
              detail={`À +${projectionHorizon} h : ${formatNumber(projectedWave, 2)} m · seuil ${waveThreshold.toFixed(1)} m`}
              support={thresholdWindows > 0 ? `${thresholdWindows} heure(s) au-dessus du seuil · dès ${formatForecastMoment(firstThresholdAt)}` : 'Aucun dépassement du seuil sélectionné'}
              icon="material-symbols:waves-rounded"
              color={thresholdWindows > 0 ? tokens.amber : pf.functional.blue}
              active={projectionFocus !== 'TEMPERATURE'}
            />
            <ProjectionResultCard
              label="Vent et précipitations"
              value={`Rafales ${formatNumber(gustRange.max)} m/s`}
              detail={`Probabilité de pluie maximale : ${formatNumber(precipitationRange.max, 0)} %`}
              support={`Valeurs maximales calculées sur la fenêtre de ${projectionHorizon} h`}
              icon="material-symbols:air-rounded"
              color="#49A7FF"
            />
            <ProjectionResultCard
              label="Impact opérationnel indicatif"
              value={operationalImpact.label}
              detail={operationalImpact.detail}
              support="Aide à la décision : aucune action automatique"
              icon={operationalImpact.icon}
              color={operationalImpact.color}
            />
          </Box>
        </Box>

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, minmax(0,1fr))', xl: 'repeat(4, minmax(0,1fr))' },
            gap: 2,
          }}
        >
          <KpiCard
            label="Température air"
            value={mode === 'NEXT_24H' ? atmosphere?.temperature_2m : outlookTemperature[0]}
            unit="°C"
            detail={mode === 'NEXT_24H' ? weatherDescription(atmosphere?.weather_code) : 'Prévision au début de période'}
            icon="material-symbols:device-thermostat-rounded"
            color="#FFBD4A"
            sparkline={mode === 'NEXT_24H' ? temperatureSpark : outlookTemperature}
            delta={trend(mode === 'NEXT_24H' ? temperatureSpark : outlookTemperature)}
            loading={selectedLoading}
          />
          <KpiCard
            label="Hauteur de vague"
            value={mode === 'NEXT_24H' ? marine?.wave_height : outlookWave[0]}
            unit="m"
            detail={mode === 'NEXT_24H' ? `Mer ${state.label.toLowerCase()}` : 'Prévision au début de période'}
            icon="material-symbols:waves-rounded"
            color="#49A7FF"
            sparkline={mode === 'NEXT_24H' ? waveSpark : outlookWave}
            delta={trend(mode === 'NEXT_24H' ? waveSpark : outlookWave)}
            loading={selectedLoading}
          />
          <KpiCard
            label="Vent moyen"
            value={mode === 'NEXT_24H' ? atmosphere?.wind_speed_10m : outlookWind[0]}
            unit="m/s"
            detail={mode === 'NEXT_24H' ? `${cardinal(atmosphere?.wind_direction_10m)} · rafales ${formatNumber(atmosphere?.wind_gusts_10m)} m/s` : 'Prévision au début de période'}
            icon="material-symbols:air-rounded"
            color="#36D6CF"
            sparkline={mode === 'NEXT_24H' ? windSpark : outlookWind}
            delta={trend(mode === 'NEXT_24H' ? windSpark : outlookWind)}
            loading={selectedLoading}
          />
          <KpiCard
            label="Pression"
            value={mode === 'NEXT_24H' ? atmosphere?.surface_pressure : outlookPressure[0]}
            unit="hPa"
            detail={mode === 'NEXT_24H' ? `${formatNumber(atmosphere?.relative_humidity_2m, 0)} % humidité` : 'Prévision au début de période'}
            icon="material-symbols:speed-rounded"
            color="#5DE0A3"
            sparkline={mode === 'NEXT_24H' ? pressureSpark : outlookPressure}
            delta={trend(mode === 'NEXT_24H' ? pressureSpark : outlookPressure)}
            loading={selectedLoading}
          />
        </Box>

        <Paper sx={{ ...glassCard, p: { xs: 1.5, md: 2.2 } }}>
          <SectionHeader
            eyebrow="Historique et prévision"
            title={projectionFocus === 'TEMPERATURE' ? 'Trajectoire de température' : projectionFocus === 'WAVES' ? 'Trajectoire de l’état de mer' : 'Trajectoire météo-marine'}
            meta={`24 H OBSERVÉES · ${projectionHorizon} H PRÉVUES`}
          />
          <Typography sx={{ color: tokens.muted, fontSize: 10.5, mt: 0.45 }}>
            Trait plein : observations passées · trait discontinu : prévisions futures · la ligne horizontale indique votre seuil de vague.
          </Typography>
          {liveLoading ? (
            <Skeleton variant="rounded" height={455} sx={{ mt: 1, bgcolor: pf.background.panelRaised }} />
          ) : atmosphereHourly || marineHourly ? (
            <MetoceanProjectionChart
              liveAtmosphere={atmosphereHourly}
              liveMarine={marineHourly}
              horizonHours={projectionHorizon}
              focus={projectionFocus}
              temperatureUnit={temperatureUnit}
              waveThreshold={waveThreshold}
              height={455}
            />
          ) : (
            <Box sx={{ height: 455, display: 'grid', placeItems: 'center', color: tokens.muted }}>Série indisponible</Box>
          )}
        </Paper>

        <Paper sx={{ ...glassCard, p: { xs: 1.5, md: 2.2 } }}>
          <SectionHeader eyebrow="Lecture actuelle" title="Détail de l’état de mer" meta={state.severity} />
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '220px minmax(0,1fr)' },
              alignItems: 'center',
              gap: 2.2,
              mt: 1.6,
            }}
          >
            <Stack direction="row" justifyContent={{ xs: 'center', md: 'space-between' }} alignItems="center" gap={2}>
              <DirectionCompass degrees={marine?.wave_direction} color={state.color} />
              <Box textAlign="right">
                <Typography sx={{ color: state.color, fontFamily: monoFont, fontSize: 34, fontWeight: 600 }}>
                  {formatNumber(marine?.wave_height, 2)}
                  <Box component="span" sx={{ color: tokens.muted, fontSize: 12, ml: 0.5 }}>m</Box>
                </Typography>
                <Typography sx={{ color: tokens.text, fontSize: 14, fontWeight: 700 }}>{state.label}</Typography>
                <Typography sx={{ color: tokens.muted, fontSize: 11 }}>
                  {cardinal(marine?.wave_direction)} · {formatNumber(marine?.wave_direction, 0)}°
                </Typography>
              </Box>
            </Stack>
            <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2,minmax(0,1fr))', lg: 'repeat(4,minmax(0,1fr))' }, borderTop: `1px solid ${tokens.border}`, borderLeft: `1px solid ${tokens.border}` }}>
              {[
                ['Période dominante', `${formatNumber(marine?.wave_period)} s`],
                ['Houle', `${formatNumber(marine?.swell_wave_height, 2)} m · ${cardinal(marine?.swell_wave_direction)}`],
                ['Température mer', `${formatNumber(marine?.sea_surface_temperature)} °C`],
                ['Courant', `${formatNumber(marine?.ocean_current_velocity, 2)} km/h · ${cardinal(marine?.ocean_current_direction)}`],
              ].map(([label, value]) => (
                <Box key={label} sx={{ minHeight: 82, p: 1.3, borderRight: `1px solid ${tokens.border}`, borderBottom: `1px solid ${tokens.border}` }}>
                  <Typography sx={{ color: tokens.muted, fontSize: 10 }}>{label}</Typography>
                  <Typography sx={{ color: tokens.text, fontFamily: monoFont, fontSize: 12, mt: 0.6 }}>{value}</Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </Paper>

        <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', xl: 'minmax(0,1.55fr) minmax(340px,1fr)' }, gap: 2 }}>
          <Paper sx={{ ...glassCard, p: { xs: 1.5, md: 2 } }}>
            <SectionHeader
              eyebrow="Prévisions marines"
              title="Vague totale, houle et période"
              meta={mode === 'NEXT_24H' ? 'HEURE PAR HEURE · 24 H' : 'TENDANCE · JUSQU’À 72 H'}
            />
            {selectedLoading ? (
              <Skeleton variant="rounded" height={315} sx={{ mt: 1, bgcolor: 'rgba(255,255,255,0.055)' }} />
            ) : hasSelectedData ? (
              <MarineAnalyticsChart live={marineHourly} forecast={forecasts} mode={mode} height={315} />
            ) : (
              <Box sx={{ height: 315, display: 'grid', placeItems: 'center', color: tokens.muted }}>Série indisponible</Box>
            )}
          </Paper>

          <Paper sx={{ ...glassCard, p: 2 }}>
            <SectionHeader eyebrow="Mises à jour" title="Activité récente" meta="TEMPS RÉEL" />
            <Stack mt={1.6}>
              {activity.map((item, index) => (
                <Stack
                  key={item.title}
                  direction="row"
                  gap={1.2}
                  py={1.2}
                  sx={{ borderBottom: index < activity.length - 1 ? `1px solid ${tokens.border}` : 'none' }}
                >
                  <Box
                    sx={{
                      width: 32,
                      height: 32,
                      borderRadius: '9px',
                      display: 'grid',
                      placeItems: 'center',
                      color: item.color,
                      bgcolor: `${item.color}12`,
                      border: `1px solid ${item.color}20`,
                      flexShrink: 0,
                    }}
                  >
                    <IconifyIcon icon={item.icon} sx={{ fontSize: 17 }} />
                  </Box>
                  <Box minWidth={0}>
                    <Typography noWrap sx={{ color: tokens.text, fontSize: 12, fontWeight: 700 }}>{item.title}</Typography>
                    <Typography noWrap sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 10, mt: 0.3 }}>{item.detail}</Typography>
                  </Box>
                </Stack>
              ))}
            </Stack>
            <Stack direction="row" justifyContent="space-between" alignItems="center" mt={1.5} pt={1.4} borderTop={`1px solid ${tokens.border}`}>
              <Stack direction="row" gap={0.75} alignItems="center">
                <Box sx={{ width: 7, height: 7, borderRadius: '50%', bgcolor: autoRefresh ? tokens.teal : tokens.muted }} />
                <Typography sx={{ color: tokens.muted, fontSize: 10 }}>Auto 5 min</Typography>
              </Stack>
              <Button
                size="small"
                onClick={() => onAutoRefreshChange(!autoRefresh)}
                sx={{ color: autoRefresh ? tokens.teal : tokens.muted, fontFamily: monoFont, fontSize: 10 }}
              >
                {autoRefresh ? 'ACTIF' : 'PAUSE'}
              </Button>
            </Stack>
          </Paper>
        </Box>

        <Paper sx={{ ...glassCard, px: 2, py: 1.35 }}>
          <Stack direction={{ xs: 'column', md: 'row' }} justifyContent="space-between" gap={1}>
            <Stack direction="row" gap={1} alignItems="center" flexWrap="wrap">
              <Chip
                size="small"
                label="SUPERVISION ACTIVE"
                sx={{ color: tokens.teal, borderColor: tokens.border }}
                variant="outlined"
              />
              <Typography sx={{ color: tokens.muted, fontSize: 10 }}>
                Prévisions disponibles jusqu’à 72 h · Aucune action automatique
              </Typography>
            </Stack>
            <Typography sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 10 }}>
              SYNCHRO {lastUpdatedAt ? new Date(lastUpdatedAt).toLocaleTimeString('fr-FR') : '—'}
            </Typography>
          </Stack>
        </Paper>

        <Typography sx={{ color: tokens.muted, fontSize: 10, lineHeight: 1.6 }}>
          Conditions et prévisions indicatives, actualisées automatiquement. Les données côtières ne remplacent
          pas les sources nautiques réglementaires.
        </Typography>
      </Stack>
    </Box>
  );
};

export default MetoceanAnalyticsDashboard;
