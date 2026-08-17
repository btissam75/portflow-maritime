import { Alert, Box, Button, Chip, Paper, Skeleton, Stack, Typography } from '@mui/material';
import { useEffect, useMemo, useState } from 'react';
import IconifyIcon from 'components/base/IconifyIcon';
import {
  KpiSparkline,
  MarineAnalyticsChart,
  MetoceanProjectionChart,
} from 'components/sections/maritime/MetoceanAnalyticsCharts';
import type {
  ForecastMode,
  ProjectionFocus,
} from 'components/sections/maritime/MetoceanAnalyticsCharts';
import MetoceanSituationMap from 'components/sections/maritime/MetoceanSituationMap';
import { useSearchParams } from 'react-router-dom';
import type { LiveMetoceanData } from 'types/liveMetocean';
import type { MetoceanDashboardData, MetoceanForecastPoint } from 'types/metocean';
import { portflowPalette as pf } from 'theme/portflowPalette';

interface MetoceanAnalyticsDashboardProps {
  data: MetoceanDashboardData | null;
  liveData: LiveMetoceanData | null;
  loading: boolean;
  liveLoading: boolean;
  outlookError: string | null;
  liveError: string | null;
  outlookUpdatedAt: string | null;
  liveUpdatedAt: string | null;
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
  bgcolor: 'rgba(21,25,30,.88)',
  border: `1px solid ${tokens.border}`,
  borderRadius: '18px',
  boxShadow: '0 22px 60px rgba(0,0,0,.24), inset 0 1px 0 rgba(255,255,255,.025)',
  backdropFilter: 'blur(18px)',
} as const;

type SourceState = 'LOADING' | 'AVAILABLE' | 'PARTIAL' | 'SHADOW' | 'UNAVAILABLE';

const sourceStatePresentation: Record<SourceState, { label: string; color: string }> = {
  LOADING: { label: 'Connexion', color: pf.functional.blue },
  AVAILABLE: { label: 'Disponible', color: pf.functional.green },
  PARTIAL: { label: 'Partiel', color: pf.functional.amber },
  SHADOW: { label: 'En validation', color: pf.functional.purple },
  UNAVAILABLE: { label: 'Indisponible', color: pf.functional.red },
};

const SourceStatusCard = ({
  title,
  detail,
  state,
  updatedAt,
}: {
  title: string;
  detail: string;
  state: SourceState;
  updatedAt: string | null;
}) => {
  const presentation = sourceStatePresentation[state];
  return (
    <Stack
      component="section"
      aria-label={`${title} : ${presentation.label}`}
      direction="row"
      alignItems="center"
      gap={1}
      sx={{
        minWidth: 0,
        minHeight: 58,
        px: 1.25,
        py: 0.85,
        bgcolor: tokens.surface,
        border: `1px solid ${tokens.border}`,
        borderRadius: '8px',
      }}
    >
      <Box
        aria-hidden="true"
        sx={{
          width: 9,
          height: 9,
          borderRadius: '50%',
          bgcolor: presentation.color,
          boxShadow: `0 0 0 4px ${presentation.color}18`,
          flexShrink: 0,
        }}
      />
      <Box minWidth={0} flex={1}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" gap={1}>
          <Typography noWrap sx={{ color: tokens.text, fontSize: 11.5, fontWeight: 700 }}>
            {title}
          </Typography>
          <Typography
            sx={{ color: presentation.color, fontFamily: monoFont, fontSize: 8.5, fontWeight: 800 }}
          >
            {presentation.label.toUpperCase()}
          </Typography>
        </Stack>
        <Typography noWrap sx={{ color: tokens.muted, fontSize: 9.5, mt: 0.2 }}>
          {detail} ·{' '}
          {updatedAt
            ? `synchro ${new Date(updatedAt).toLocaleTimeString('fr-FR')}`
            : 'non synchronisé'}
        </Typography>
      </Box>
    </Stack>
  );
};

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
    return Number.isFinite(stamp) &&
      stamp >= now &&
      stamp <= end &&
      value != null &&
      Number.isFinite(value)
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
      background: `radial-gradient(circle at 88% 8%, ${color}20 0%, transparent 33%), linear-gradient(145deg, rgba(27,32,38,.97), rgba(13,15,18,.98))`,
      borderColor: `${color}30`,
      overflow: 'hidden',
      animation: 'portflowCardIn 620ms cubic-bezier(.2,.8,.2,1) both',
      transition: 'transform 240ms ease, border-color 240ms ease, box-shadow 240ms ease',
      '&::after': {
        content: '""',
        position: 'absolute',
        width: 120,
        height: 120,
        right: -54,
        bottom: -72,
        borderRadius: '50%',
        bgcolor: `${color}10`,
        filter: 'blur(2px)',
        pointerEvents: 'none',
      },
      '&:hover': {
        transform: 'translateY(-4px)',
        borderColor: `${color}68`,
        boxShadow: `0 24px 55px rgba(0,0,0,.32), 0 0 32px ${color}12`,
      },
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
            borderRadius: '12px',
            boxShadow: `0 9px 24px ${color}16`,
          }}
        >
          <IconifyIcon icon={icon} sx={{ fontSize: 21 }} />
        </Box>
        <Stack alignItems="flex-end" gap={0.3}>
          {delta != null && !loading && (
            <Typography
              sx={{
                color: delta > 0 ? tokens.amber : tokens.teal,
                fontFamily: monoFont,
                fontSize: 10,
                fontWeight: 700,
              }}
            >
              {delta >= 0 ? '↗' : '↘'} {Math.abs(delta).toFixed(1)}
            </Typography>
          )}
          <Typography sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 8.5 }}>
            24 H
          </Typography>
        </Stack>
      </Stack>
      <Box>
        <Typography
          sx={{ color: tokens.muted, fontSize: 11, fontWeight: 700, textTransform: 'uppercase' }}
        >
          {label}
        </Typography>
        <Stack direction="row" alignItems="baseline" gap={0.6} mt={0.25}>
          <Typography
            aria-live="polite"
            sx={{
              color: tokens.text,
              fontFamily: monoFont,
              fontSize: 29,
              fontWeight: 600,
              lineHeight: 1.15,
            }}
          >
            {loading ? (
              '···'
            ) : (
              <AnimatedNumber
                value={value}
                digits={value != null && Math.abs(value) >= 100 ? 0 : 1}
              />
            )}
          </Typography>
          <Typography sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 12 }}>
            {unit}
          </Typography>
        </Stack>
        <Typography noWrap sx={{ color: tokens.muted, fontSize: 10.5, mt: 0.35 }}>
          {detail}
        </Typography>
      </Box>
      <Box sx={{ width: 1, height: 42, borderTop: `1px solid ${color}32`, pt: 0.3 }}>
        {loading ? (
          <Skeleton width="100%" height={42} sx={{ bgcolor: 'rgba(255,255,255,0.06)' }} />
        ) : (
          <KpiSparkline data={sparkline} color={color} />
        )}
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
        <Typography
          sx={{ color: tokens.muted, fontSize: 10, fontWeight: 800, textTransform: 'uppercase' }}
        >
          {label}
        </Typography>
        <Box
          sx={{
            width: 34,
            height: 34,
            borderRadius: '6px',
            display: 'grid',
            placeItems: 'center',
            color,
            bgcolor: `${color}18`,
          }}
        >
          <IconifyIcon icon={icon} sx={{ fontSize: 20 }} />
        </Box>
      </Stack>
      <Box>
        <Typography
          aria-live="polite"
          sx={{
            color: tokens.text,
            fontFamily: monoFont,
            fontSize: 23,
            fontWeight: 700,
            lineHeight: 1.2,
          }}
        >
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
      <Typography
        sx={{
          color: tokens.teal,
          fontFamily: monoFont,
          fontSize: 9,
          fontWeight: 800,
          textTransform: 'uppercase',
        }}
      >
        {eyebrow}
      </Typography>
      <Typography
        sx={{
          color: tokens.text,
          fontFamily: displayFont,
          fontSize: 16,
          lineHeight: '22px',
          fontWeight: 650,
          animation: 'portflowTitleIn 320ms cubic-bezier(.2,.8,.2,1) both',
        }}
      >
        {title}
      </Typography>
    </Box>
    <Typography
      sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 10, textAlign: 'right' }}
    >
      {meta}
    </Typography>
  </Stack>
);

const weatherIcon = (code: number | null | undefined) => {
  if (code == null) return 'lucide:cloud';
  if (code === 0) return 'lucide:sun';
  if (code <= 3) return 'lucide:cloud-sun';
  if (code <= 48) return 'lucide:cloud-fog';
  if (code <= 82) return 'lucide:cloud-rain';
  return 'lucide:cloud-lightning';
};

const TemporalWeatherCard = ({
  kind,
  title,
  subtitle,
  weatherCode,
  temperature,
  wind,
  windDirection,
  wave,
  swell,
  period,
}: {
  kind: 'OBSERVED' | 'FORECAST';
  title: string;
  subtitle: string;
  weatherCode: number | null | undefined;
  temperature: number | null | undefined;
  wind: number | null | undefined;
  windDirection: number | null | undefined;
  wave: number | null | undefined;
  swell: number | null | undefined;
  period: number | null | undefined;
}) => {
  const observed = kind === 'OBSERVED';
  const color = observed ? '#54E3B2' : '#9A91FF';
  const sunColor = weatherCode === 0 ? '#FFD35A' : color;
  const measurements = [
    ['Vent', `${cardinal(windDirection)} · ${formatNumber(wind)} m/s`, 'lucide:wind', '#FFC44D'],
    ['Vague totale', `${formatNumber(wave, 2)} m`, 'lucide:waves', '#35D7F2'],
    ['Houle', `${formatNumber(swell, 2)} m`, 'lucide:activity', '#54E3B2'],
    ['Période', `${formatNumber(period)} s`, 'lucide:timer', '#C6A4FF'],
  ] as const;

  return (
    <Paper
      component="article"
      sx={{
        position: 'relative',
        overflow: 'hidden',
        minHeight: 174,
        p: { xs: 1.5, md: 1.8 },
        borderRadius: '18px',
        border: `1px solid ${color}3D`,
        background: `radial-gradient(circle at 92% 8%, ${sunColor}22, transparent 30%), linear-gradient(145deg, ${color}0D, rgba(17,20,24,.97) 52%)`,
        boxShadow: `0 22px 55px rgba(0,0,0,.23), inset 0 1px 0 ${color}18`,
        transition: 'transform 240ms ease, border-color 240ms ease',
        '&:hover': { transform: 'translateY(-3px)', borderColor: `${color}70` },
      }}
    >
      <Stack direction="row" justifyContent="space-between" alignItems="flex-start" gap={1}>
        <Box>
          <Stack direction="row" alignItems="center" gap={0.65}>
            <Box
              sx={{
                width: 7,
                height: 7,
                borderRadius: '50%',
                bgcolor: color,
                boxShadow: `0 0 13px ${color}`,
              }}
            />
            <Typography sx={{ color, fontSize: 9, fontWeight: 900, letterSpacing: '.08em' }}>
              {observed ? 'MESURÉ' : 'PRÉVU'}
            </Typography>
          </Stack>
          <Typography sx={{ color: tokens.text, fontSize: 15, fontWeight: 800, mt: 0.35 }}>
            {title}
          </Typography>
          <Typography sx={{ color: tokens.muted, fontSize: 9.5, mt: 0.15 }}>{subtitle}</Typography>
        </Box>
        <Stack direction="row" alignItems="center" gap={0.75}>
          <IconifyIcon
            icon={weatherIcon(weatherCode)}
            sx={{ color: sunColor, fontSize: 29, filter: `drop-shadow(0 0 14px ${sunColor}88)` }}
          />
          <Typography sx={{ color: tokens.text, fontSize: 26, fontWeight: 800 }}>
            {formatNumber(temperature)}°
          </Typography>
        </Stack>
      </Stack>
      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2,minmax(0,1fr))', sm: 'repeat(4,minmax(0,1fr))' },
          gap: 0.7,
          mt: 1.45,
        }}
      >
        {measurements.map(([label, value, icon, metricColor]) => (
          <Stack
            key={label}
            direction="row"
            alignItems="center"
            gap={0.65}
            sx={{
              minWidth: 0,
              p: 0.75,
              borderRadius: '10px',
              bgcolor: 'rgba(8,9,11,.48)',
              border: '1px solid rgba(213,206,190,.1)',
            }}
          >
            <IconifyIcon icon={icon} sx={{ color: metricColor, fontSize: 15, flexShrink: 0 }} />
            <Box minWidth={0}>
              <Typography noWrap sx={{ color: pf.text.tertiary, fontSize: 7.5 }}>
                {label.toUpperCase()}
              </Typography>
              <Typography noWrap sx={{ color: tokens.text, fontSize: 9.5, fontWeight: 750 }}>
                {value}
              </Typography>
            </Box>
          </Stack>
        ))}
      </Box>
    </Paper>
  );
};

const SegmentedMode = ({
  mode,
  onChange,
}: {
  mode: ForecastMode;
  onChange: (mode: ForecastMode) => void;
}) => (
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
    {(
      [
        ['NEXT_24H', 'PROCHAINES 24 H'],
        ['OUTLOOK_72H', 'TENDANCE 72 H'],
      ] as const
    ).map(([value, label]) => (
      <Button
        key={value}
        size="small"
        aria-pressed={mode === value}
        onClick={() => onChange(value)}
        sx={{
          minWidth: 0,
          minHeight: 44,
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
  outlookError,
  liveError,
  outlookUpdatedAt,
  liveUpdatedAt,
  autoRefresh,
  onAutoRefreshChange,
  onRefresh,
}: MetoceanAnalyticsDashboardProps) => {
  const [searchParams] = useSearchParams();
  const [mode, setMode] = useState<ForecastMode>('NEXT_24H');
  const [projectionFocus] = useState<ProjectionFocus>('COMBINED');
  const [projectionHorizon, setProjectionHorizon] = useState<12 | 24 | 72>(24);
  const [temperatureUnit] = useState<'C' | 'F'>('C');
  const waveThreshold = 1.5;
  const [mapHour, setMapHour] = useState(0);
  const [clock, setClock] = useState(() => new Date());
  const atmosphere = liveData?.atmosphere?.current;
  const atmosphereHourly = liveData?.atmosphere?.hourly;
  const marine = liveData?.marine?.current;
  const marineHourly = liveData?.marine?.hourly;
  const comparisonHour = mapHour > 0 ? mapHour : 24;
  const comparisonTemperature = valueAtHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.temperature_2m,
    comparisonHour,
  );
  const comparisonWeatherCode = valueAtHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.weather_code,
    comparisonHour,
  );
  const comparisonWind = valueAtHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.wind_speed_10m,
    comparisonHour,
  );
  const comparisonWindDirection = valueAtHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.wind_direction_10m,
    comparisonHour,
  );
  const comparisonWave = valueAtHorizon(
    marineHourly?.time,
    marineHourly?.wave_height,
    comparisonHour,
  );
  const comparisonSwell = valueAtHorizon(
    marineHourly?.time,
    marineHourly?.swell_wave_height,
    comparisonHour,
  );
  const comparisonPeriod = valueAtHorizon(
    marineHourly?.time,
    marineHourly?.wave_period,
    comparisonHour,
  );
  const forecasts = data?.forecast ?? [];
  const state = seaState(marine?.wave_height);
  const unavailable = [
    ...new Set([...(liveData?.unavailable ?? []), ...(data?.unavailable ?? [])]),
  ];

  const temperatureSpark = atmosphereHourly?.temperature_2m ?? [];
  const windSpark = atmosphereHourly?.wind_speed_10m ?? [];
  const waveSpark = marineHourly?.wave_height ?? [];
  const pressureSpark = atmosphereHourly?.surface_pressure ?? [];
  const maxWave = waveSpark
    .filter((item): item is number => item != null)
    .reduce<
      number | null
    >((maximum, item) => (maximum == null || item > maximum ? item : maximum), null);

  const issueAt = useMemo(() => {
    const timestamps = forecasts
      .map((point) => point.issue_at)
      .filter(Boolean)
      .sort();
    return timestamps.length ? timestamps[timestamps.length - 1] : null;
  }, [forecasts]);
  const liveSourceCount = Number(Boolean(atmosphere)) + Number(Boolean(marine));
  const liveSourceState: SourceState =
    liveLoading && !liveData
      ? 'LOADING'
      : liveSourceCount === 2
        ? 'AVAILABLE'
        : liveSourceCount === 1
          ? 'PARTIAL'
          : 'UNAVAILABLE';
  const governedForecastState: SourceState =
    loading && !data
      ? 'LOADING'
      : forecasts.length > 0
        ? data?.status?.production_promotion_allowed
          ? 'AVAILABLE'
          : 'SHADOW'
        : data?.status
          ? 'PARTIAL'
          : 'UNAVAILABLE';
  const forecastTrackLabel =
    data?.status?.audit_status === 'DEMO'
      ? 'Tendance d’exercice'
      : data?.forecastTrack === 'ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT'
        ? 'Prévisions disponibles jusqu’à 72 h'
        : data?.forecastTrack === 'RESEARCH_REANALYSIS_SHADOW'
          ? 'Tendance en cours de validation'
          : 'Prévision portuaire';
  const hasUsableData = liveSourceCount > 0 || forecasts.length > 0;
  const errorMessages = [liveError, outlookError].filter((message): message is string =>
    Boolean(message),
  );
  const topImpacts = useMemo(
    () =>
      [...(data?.impacts ?? [])]
        .sort((left, right) => right.combined_priority_score - left.combined_priority_score)
        .slice(0, 3),
    [data?.impacts],
  );
  const modelOperational = Boolean(
    data?.status?.production_promotion_allowed &&
      data?.validation?.fresh_confirmed &&
      data?.validation?.critical_gates_passed,
  );
  const updateTimestamps = [liveUpdatedAt, outlookUpdatedAt]
    .filter((value): value is string => Boolean(value))
    .sort();
  const lastUpdatedAt = updateTimestamps[updateTimestamps.length - 1] ?? null;

  const outlookTemperature = outlookValues(forecasts, 'temperature_2m');
  const outlookWave = outlookValues(forecasts, 'wave_height_m');
  const outlookWind = outlookValues(forecasts, 'wind_speed_ms');
  const outlookPressure = outlookValues(forecasts, 'pressure_hpa');
  const selectedLoading = mode === 'NEXT_24H' ? liveLoading : loading;
  const hasSelectedData =
    mode === 'NEXT_24H' ? Boolean(atmosphereHourly || marineHourly) : forecasts.length > 0;
  const projectedTemperatureC = valueAtHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.temperature_2m,
    projectionHorizon,
  );
  const projectedTemperature =
    projectedTemperatureC == null
      ? null
      : temperatureUnit === 'F'
        ? (projectedTemperatureC * 9) / 5 + 32
        : projectedTemperatureC;
  const projectedWave = valueAtHorizon(
    marineHourly?.time,
    marineHourly?.wave_height,
    projectionHorizon,
  );
  const temperatureWindowC = valuesWithinHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.temperature_2m,
    projectionHorizon,
  );
  const temperatureWindow = temperatureWindowC.map((point) => ({
    ...point,
    value: temperatureUnit === 'F' ? (point.value * 9) / 5 + 32 : point.value,
  }));
  const waveWindow = valuesWithinHorizon(
    marineHourly?.time,
    marineHourly?.wave_height,
    projectionHorizon,
  );
  const gustWindow = valuesWithinHorizon(
    atmosphereHourly?.time,
    atmosphereHourly?.wind_gusts_10m,
    projectionHorizon,
  );
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
  const currentTemperature =
    atmosphere?.temperature_2m == null
      ? null
      : temperatureUnit === 'F'
        ? (atmosphere.temperature_2m * 9) / 5 + 32
        : atmosphere.temperature_2m;
  const projectedTemperatureDelta =
    projectedTemperature == null || currentTemperature == null
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
      (waveRange.max != null && waveRange.max >= Math.max(2.5, waveThreshold + 0.5)) ||
      (gustRange.max != null && gustRange.max >= 20) ||
      (precipitationRange.max != null && precipitationRange.max >= 80)
    ) {
      return {
        label: 'Exposition forte',
        detail: 'Revoir les créneaux sensibles et confirmer les moyens disponibles.',
        color: tokens.red,
        icon: 'material-symbols:crisis-alert-rounded',
      };
    }
    if (
      thresholdWindows > 0 ||
      (gustRange.max != null && gustRange.max >= 14) ||
      (precipitationRange.max != null && precipitationRange.max >= 60)
    ) {
      return {
        label: 'Vigilance requise',
        detail: 'Surveiller la fenêtre signalée avant de confirmer les opérations.',
        color: tokens.amber,
        icon: 'material-symbols:warning-rounded',
      };
    }
    return {
      label: 'Conditions favorables',
      detail: 'Les conditions prévues restent compatibles avec les opérations habituelles.',
      color: pf.functional.blue,
      icon: 'material-symbols:check-circle-rounded',
    };
  })();

  useEffect(() => {
    const interval = window.setInterval(() => setClock(new Date()), 60_000);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    if (searchParams.get('horizon') !== '72') return;
    setMode('OUTLOOK_72H');
    setProjectionHorizon(72);
    setMapHour(72);
  }, [searchParams]);

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
      detail: liveData?.fetchedAt
        ? new Date(liveData.fetchedAt).toLocaleTimeString('fr-FR')
        : 'En attente',
    },
    {
      icon: 'material-symbols:calendar-clock-rounded',
      color: tokens.violet,
      title: 'Tendance 72 h disponible',
      detail: issueAt
        ? `Calculée ${new Date(issueAt).toLocaleString('fr-FR')}`
        : 'Mise à jour en attente',
    },
    {
      icon:
        state.color === tokens.red
          ? 'material-symbols:warning-rounded'
          : 'material-symbols:verified-rounded',
      color: state.color,
      title: `État de mer ${state.label.toLowerCase()}`,
      detail:
        maxWave == null ? 'Pic 24 h indisponible' : `Pic prévu ${maxWave.toFixed(2)} m sur 24 h`,
    },
    {
      icon: forecasts.length
        ? 'material-symbols:event-available-rounded'
        : 'material-symbols:event-busy-rounded',
      color: forecasts.length ? tokens.teal : tokens.amber,
      title: forecasts.length
        ? 'Prévisions longue échéance prêtes'
        : 'Prévisions longue échéance indisponibles',
      detail: forecasts.length
        ? 'Horizon disponible jusqu’à 72 h'
        : 'Nouvelle tentative à la prochaine actualisation',
    },
  ];

  return (
    <Box
      sx={{
        color: tokens.text,
        position: 'relative',
        isolation: 'isolate',
        '&::before': {
          content: '""',
          position: 'fixed',
          inset: '78px 0 0 80px',
          pointerEvents: 'none',
          zIndex: -1,
          background:
            atmosphere?.weather_code === 0
              ? 'radial-gradient(circle at 13% 5%, rgba(255,211,90,.13), transparent 25%), radial-gradient(circle at 82% 19%, rgba(85,214,194,.07), transparent 31%)'
              : 'radial-gradient(circle at 11% 6%, rgba(141,145,255,.09), transparent 25%), radial-gradient(circle at 82% 19%, rgba(242,184,75,.055), transparent 31%)',
        },
      }}
    >
      <Stack gap={2.25}>
        <Stack
          direction={{ xs: 'column', md: 'row' }}
          justifyContent="space-between"
          alignItems={{ xs: 'stretch', md: 'flex-end' }}
          gap={1.5}
        >
          <Box sx={{ position: 'relative', pb: 1.35 }}>
            <Stack direction="row" alignItems="center" gap={1} mb={0.65}>
              <Box
                sx={{
                  width: 7,
                  height: 7,
                  borderRadius: '50%',
                  bgcolor: tokens.teal,
                  animation: 'portflowPulse 2s infinite',
                }}
              />
              <Typography
                sx={{ color: tokens.teal, fontFamily: monoFont, fontSize: 10, fontWeight: 700 }}
              >
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
            <Box
              aria-hidden="true"
              sx={{
                width: 72,
                height: 2,
                mt: 0.65,
                bgcolor: tokens.teal,
                transformOrigin: 'left center',
                animation: 'portflowTitleRule 520ms 120ms cubic-bezier(.2,.8,.2,1) both',
              }}
            />
            <Typography sx={{ color: tokens.muted, fontSize: 13, mt: 0.65 }}>
              Conditions actuelles, prévisions et vigilance sur une seule vue.
            </Typography>
            <Typography
              sx={{
                color: pf.text.secondary,
                fontFamily: monoFont,
                fontSize: 10,
                mt: 0.7,
                textTransform: 'capitalize',
              }}
            >
              Aujourd’hui · {todayLabel} · {currentTime} à Tanger
            </Typography>
          </Box>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            alignItems={{ xs: 'stretch', sm: 'center' }}
            gap={1}
          >
            <SegmentedMode mode={mode} onChange={setMode} />
            <Button
              aria-label="Actualiser les données metocean"
              onClick={onRefresh}
              disabled={loading || liveLoading}
              sx={{
                minWidth: 40,
                minHeight: 44,
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

        <Box
          aria-label="État des sources météo-marines"
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', md: 'repeat(2,minmax(0,1fr))' },
            gap: 1,
          }}
        >
          <SourceStatusCard
            title="Observations météo-marines"
            detail={
              liveSourceCount === 2
                ? 'Atmosphère et état de mer'
                : `${liveSourceCount}/2 source(s) disponible(s)`
            }
            state={liveSourceState}
            updatedAt={liveUpdatedAt}
          />
          <SourceStatusCard
            title="Prévisions à venir"
            detail={forecastTrackLabel}
            state={governedForecastState}
            updatedAt={outlookUpdatedAt}
          />
        </Box>

        {(errorMessages.length > 0 || unavailable.length > 0) && (
          <Alert
            severity={hasUsableData ? 'warning' : 'error'}
            sx={{
              bgcolor: 'rgba(255,107,107,0.08)',
              color: tokens.text,
              border: `1px solid ${tokens.border}`,
            }}
          >
            <Typography sx={{ fontSize: 11.5, fontWeight: 700 }}>
              {hasUsableData
                ? 'Mode dégradé : les données disponibles restent affichées.'
                : 'Aucune source météo-marine n’est disponible.'}
            </Typography>
            <Typography sx={{ color: tokens.muted, fontSize: 10, mt: 0.25 }}>
              {[
                ...errorMessages,
                ...(unavailable.length ? [`Sources manquantes : ${unavailable.join(', ')}`] : []),
              ].join(' ')}
            </Typography>
          </Alert>
        )}

        <Box component="section" aria-label="Comparaison entre observation et prévision">
          <SectionHeader
            eyebrow="Lecture temporelle"
            title="Ce qui est mesuré / ce qui est prévu"
            meta={`COMPARAISON À H+${comparisonHour}`}
          />
          <Typography sx={{ color: tokens.muted, fontSize: 10.5, mt: 0.45, mb: 1.15 }}>
            Les mesures disponibles maintenant restent séparées de la projection sélectionnée sur la
            frise.
          </Typography>
          <Box
            sx={{
              position: 'relative',
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: 'repeat(2,minmax(0,1fr))' },
              gap: 1.15,
            }}
          >
            <TemporalWeatherCard
              kind="OBSERVED"
              title="Situation observée"
              subtitle="Dernière mesure disponible à Tanger Med"
              weatherCode={atmosphere?.weather_code}
              temperature={atmosphere?.temperature_2m}
              wind={atmosphere?.wind_speed_10m}
              windDirection={atmosphere?.wind_direction_10m}
              wave={marine?.wave_height}
              swell={marine?.swell_wave_height}
              period={marine?.wave_period}
            />
            <TemporalWeatherCard
              kind="FORECAST"
              title={`Projection dans ${comparisonHour} h`}
              subtitle="Valeurs futures issues de la tendance horaire"
              weatherCode={comparisonWeatherCode}
              temperature={comparisonTemperature}
              wind={comparisonWind}
              windDirection={comparisonWindDirection}
              wave={comparisonWave}
              swell={comparisonSwell}
              period={comparisonPeriod}
            />
            <Box
              aria-hidden="true"
              sx={{
                display: { xs: 'none', md: 'grid' },
                placeItems: 'center',
                position: 'absolute',
                top: '50%',
                left: '50%',
                width: 34,
                height: 34,
                borderRadius: '50%',
                color: '#EAF7F8',
                bgcolor: '#15191E',
                border: `1px solid ${tokens.border}`,
                boxShadow: '0 10px 24px rgba(0,0,0,.3)',
                transform: 'translate(-50%,-50%)',
                zIndex: 2,
              }}
            >
              <IconifyIcon icon="lucide:arrow-right" sx={{ fontSize: 16 }} />
            </Box>
          </Box>
        </Box>

        <Paper sx={{ ...glassCard, overflow: 'hidden' }}>
          <Box
            sx={{
              px: { xs: 1.5, md: 2.2 },
              pt: 1.8,
              pb: 1.2,
              borderBottom: `1px solid ${tokens.border}`,
            }}
          >
            <SectionHeader
              eyebrow={
                mapHour === 0
                  ? 'Zone opérationnelle · Observation'
                  : 'Zone opérationnelle · Prévision'
              }
              title="Détroit de Gibraltar et approches de Tanger Med"
              meta={mapHour === 0 ? 'SITUATION MESURÉE' : `PROJECTION H+${mapHour}`}
            />
          </Box>
          <MetoceanSituationMap
            atmosphere={atmosphere}
            atmosphereHourly={atmosphereHourly}
            marine={marine}
            marineHourly={marineHourly}
            impacts={data?.impacts}
            forecastHour={mapHour}
            height={590}
          />
        </Paper>

        <Box component="section" aria-label="Résultat de la projection">
          <SectionHeader
            eyebrow="Résultat de l’analyse"
            title={`Ce qui est prévu sur les prochaines ${projectionHorizon} heures`}
            meta="MISE À JOUR IMMÉDIATE"
          />
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: {
                xs: '1fr',
                sm: 'repeat(2,minmax(0,1fr))',
                xl: 'repeat(4,minmax(0,1fr))',
              },
              gap: 1.2,
              mt: 1.2,
            }}
          >
            <ProjectionResultCard
              label={`Température dans ${projectionHorizon} h`}
              value={`${formatNumber(projectedTemperature)} °${temperatureUnit}`}
              detail={`Entre ${formatNumber(temperatureRange.min)} et ${formatNumber(temperatureRange.max)} °${temperatureUnit} sur la période`}
              support={
                projectedTemperatureDelta == null
                  ? 'Comparaison avec la température actuelle indisponible'
                  : `${projectedTemperatureDelta >= 0 ? '+' : ''}${projectedTemperatureDelta.toFixed(1)} °${temperatureUnit} par rapport à maintenant`
              }
              icon="material-symbols:device-thermostat-rounded"
              color="#36D6CF"
              active={projectionFocus !== 'WAVES'}
            />
            <ProjectionResultCard
              label={`État de mer sur ${projectionHorizon} h`}
              value={`Pic ${formatNumber(waveRange.max, 2)} m`}
              detail={`À +${projectionHorizon} h : ${formatNumber(projectedWave, 2)} m · seuil ${waveThreshold.toFixed(1)} m`}
              support={
                thresholdWindows > 0
                  ? `${thresholdWindows} heure(s) au-dessus du seuil · dès ${formatForecastMoment(firstThresholdAt)}`
                  : 'Aucun dépassement du seuil sélectionné'
              }
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
              label="Conséquence pour les opérations"
              value={operationalImpact.label}
              detail={operationalImpact.detail}
              support="L’équipe d’exploitation confirme la décision finale"
              icon={operationalImpact.icon}
              color={operationalImpact.color}
            />
          </Box>
        </Box>

        <Box
          component="section"
          aria-label="Impact sur les escales et gouvernance"
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', xl: 'minmax(0,1.35fr) minmax(320px,.65fr)' },
            gap: 1.2,
          }}
        >
          <Paper sx={{ ...glassCard, p: { xs: 1.5, md: 2 } }}>
            <SectionHeader
              eyebrow="Exposition des escales"
              title="Navires à examiner selon la météo"
              meta={topImpacts.length ? `${topImpacts.length} PRIORITÉS` : 'AUCUN SIGNAL'}
            />
            <Typography sx={{ color: tokens.muted, fontSize: 10.5, mt: 0.4 }}>
              Le classement aide à identifier les navires à examiner en premier selon les conditions
              prévues.
            </Typography>
            <Stack mt={1.25} gap={0.7}>
              {topImpacts.length ? (
                topImpacts.map((impact, index) => {
                  const score = Math.max(0, Math.min(1, impact.combined_priority_score));
                  const color =
                    score >= 0.65 ? tokens.red : score >= 0.4 ? tokens.amber : tokens.teal;
                  return (
                    <Stack
                      key={`${impact.port_call_id}-${impact.valid_at}`}
                      direction="row"
                      alignItems="center"
                      gap={1}
                      sx={{
                        minHeight: 54,
                        px: 1,
                        py: 0.75,
                        bgcolor: pf.background.secondary,
                        border: `1px solid ${tokens.border}`,
                        borderRadius: '6px',
                      }}
                    >
                      <Typography
                        sx={{
                          width: 24,
                          color,
                          fontFamily: monoFont,
                          fontSize: 10,
                          fontWeight: 800,
                        }}
                      >
                        {String(index + 1).padStart(2, '0')}
                      </Typography>
                      <Box minWidth={0} flex={1}>
                        <Typography
                          noWrap
                          sx={{ color: tokens.text, fontSize: 11.5, fontWeight: 700 }}
                        >
                          {impact.vessel_name || 'Navire non identifié'}
                        </Typography>
                        <Typography noWrap sx={{ color: tokens.muted, fontSize: 9.5, mt: 0.2 }}>
                          {impact.terminal_code || impact.port_code || 'Terminal non renseigné'} ·
                          dans {impact.horizon_h} h ·{' '}
                          {{ LOW: 'FAIBLE', MODERATE: 'MODÉRÉE', HIGH: 'FORTE' }[
                            impact.metocean_tier
                          ] ?? impact.metocean_tier}
                        </Typography>
                      </Box>
                      <Box textAlign="right">
                        <Typography
                          sx={{ color, fontFamily: monoFont, fontSize: 14, fontWeight: 800 }}
                        >
                          {Math.round(score * 100)}
                        </Typography>
                        <Typography sx={{ color: tokens.muted, fontSize: 8 }}>
                          PRIORITÉ / 100
                        </Typography>
                      </Box>
                    </Stack>
                  );
                })
              ) : (
                <Box
                  sx={{
                    minHeight: 112,
                    display: 'grid',
                    placeItems: 'center',
                    px: 2,
                    color: tokens.muted,
                    textAlign: 'center',
                    fontSize: 10.5,
                    border: `1px dashed ${tokens.border}`,
                    borderRadius: '6px',
                  }}
                >
                  Aucun score d’exposition navire n’est disponible. Les conditions générales restent
                  consultables ci-dessus.
                </Box>
              )}
            </Stack>
          </Paper>

          <Paper sx={{ ...glassCard, p: { xs: 1.5, md: 2 } }}>
            <SectionHeader
              eyebrow="Gouvernance"
              title="Fiabilité des prévisions"
              meta={modelOperational ? 'PRÊT' : 'EN VALIDATION'}
            />
            <Stack mt={1.25} gap={0.9}>
              {[
                ['Prévisions disponibles', data?.status?.issue_time_ready === true],
                ['Contrôles qualité', data?.status?.critical_gates_passed === true],
                ['Vérification sur données récentes', data?.validation?.fresh_confirmed === true],
              ].map(([label, passed]) => (
                <Stack
                  key={String(label)}
                  direction="row"
                  justifyContent="space-between"
                  alignItems="center"
                  gap={1}
                  sx={{ py: 0.65, borderBottom: `1px solid ${tokens.border}` }}
                >
                  <Typography sx={{ color: tokens.muted, fontSize: 10.5 }}>{label}</Typography>
                  <Typography
                    sx={{
                      color: passed ? pf.functional.green : tokens.amber,
                      fontFamily: monoFont,
                      fontSize: 9,
                      fontWeight: 800,
                    }}
                  >
                    {passed ? 'VALIDÉ' : 'NON CONFIRMÉ'}
                  </Typography>
                </Stack>
              ))}
              <Alert
                severity={modelOperational ? 'success' : 'info'}
                sx={{
                  mt: 0.4,
                  bgcolor: modelOperational ? pf.functional.greenSoft : pf.functional.blueSoft,
                  color: tokens.text,
                  border: `1px solid ${tokens.border}`,
                }}
              >
                <Typography sx={{ fontSize: 10.5 }}>
                  {modelOperational
                    ? 'Ces prévisions peuvent aider à préparer les opérations. L’équipe d’exploitation confirme la décision finale.'
                    : 'Utilisez ces tendances pour anticiper la situation. Une confirmation opérationnelle reste nécessaire.'}
                </Typography>
              </Alert>
            </Stack>
          </Paper>
        </Box>

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: {
              xs: '1fr',
              sm: 'repeat(2, minmax(0,1fr))',
              xl: 'repeat(4, minmax(0,1fr))',
            },
            gap: 2,
          }}
        >
          <KpiCard
            label="Température air"
            value={mode === 'NEXT_24H' ? atmosphere?.temperature_2m : outlookTemperature[0]}
            unit="°C"
            detail={
              mode === 'NEXT_24H'
                ? weatherDescription(atmosphere?.weather_code)
                : 'Prévision au début de période'
            }
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
            detail={
              mode === 'NEXT_24H'
                ? `Mer ${state.label.toLowerCase()}`
                : 'Prévision au début de période'
            }
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
            detail={
              mode === 'NEXT_24H'
                ? `${cardinal(atmosphere?.wind_direction_10m)} · rafales ${formatNumber(atmosphere?.wind_gusts_10m)} m/s`
                : 'Prévision au début de période'
            }
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
            detail={
              mode === 'NEXT_24H'
                ? `${formatNumber(atmosphere?.relative_humidity_2m, 0)} % humidité`
                : 'Prévision au début de période'
            }
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
            title={
              projectionFocus === 'TEMPERATURE'
                ? 'Trajectoire de température'
                : projectionFocus === 'WAVES'
                  ? 'Trajectoire de l’état de mer'
                  : 'Trajectoire météo-marine'
            }
            meta={`24 H OBSERVÉES · ${projectionHorizon} H PRÉVUES`}
          />
          <Typography sx={{ color: tokens.muted, fontSize: 10.5, mt: 0.45 }}>
            Trait plein : observations passées · trait discontinu : prévisions futures · la ligne
            horizontale indique votre seuil de vague.
          </Typography>
          {liveLoading ? (
            <Skeleton
              variant="rounded"
              height={455}
              sx={{ mt: 1, bgcolor: pf.background.panelRaised }}
            />
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
            <Box sx={{ height: 455, display: 'grid', placeItems: 'center', color: tokens.muted }}>
              Série indisponible
            </Box>
          )}
        </Paper>

        <Paper sx={{ ...glassCard, p: { xs: 1.5, md: 2.2 } }}>
          <SectionHeader
            eyebrow="Lecture actuelle"
            title="Détail de l’état de mer"
            meta={state.severity}
          />
          <Box
            sx={{
              display: 'grid',
              gridTemplateColumns: { xs: '1fr', md: '220px minmax(0,1fr)' },
              alignItems: 'center',
              gap: 2.2,
              mt: 1.6,
            }}
          >
            <Stack
              direction="row"
              justifyContent={{ xs: 'center', md: 'space-between' }}
              alignItems="center"
              gap={2}
            >
              <DirectionCompass degrees={marine?.wave_direction} color={state.color} />
              <Box textAlign="right">
                <Typography
                  sx={{ color: state.color, fontFamily: monoFont, fontSize: 34, fontWeight: 600 }}
                >
                  {formatNumber(marine?.wave_height, 2)}
                  <Box component="span" sx={{ color: tokens.muted, fontSize: 12, ml: 0.5 }}>
                    m
                  </Box>
                </Typography>
                <Typography sx={{ color: tokens.text, fontSize: 14, fontWeight: 700 }}>
                  {state.label}
                </Typography>
                <Typography sx={{ color: tokens.muted, fontSize: 11 }}>
                  {cardinal(marine?.wave_direction)} · {formatNumber(marine?.wave_direction, 0)}°
                </Typography>
              </Box>
            </Stack>
            <Box
              sx={{
                display: 'grid',
                gridTemplateColumns: {
                  xs: 'repeat(2,minmax(0,1fr))',
                  lg: 'repeat(4,minmax(0,1fr))',
                },
                borderTop: `1px solid ${tokens.border}`,
                borderLeft: `1px solid ${tokens.border}`,
              }}
            >
              {[
                ['Période dominante', `${formatNumber(marine?.wave_period)} s`],
                [
                  'Houle',
                  `${formatNumber(marine?.swell_wave_height, 2)} m · ${cardinal(marine?.swell_wave_direction)}`,
                ],
                ['Température mer', `${formatNumber(marine?.sea_surface_temperature)} °C`],
                [
                  'Courant',
                  `${formatNumber(marine?.ocean_current_velocity, 2)} km/h · ${cardinal(marine?.ocean_current_direction)}`,
                ],
              ].map(([label, value]) => (
                <Box
                  key={label}
                  sx={{
                    minHeight: 82,
                    p: 1.3,
                    borderRight: `1px solid ${tokens.border}`,
                    borderBottom: `1px solid ${tokens.border}`,
                  }}
                >
                  <Typography sx={{ color: tokens.muted, fontSize: 10 }}>{label}</Typography>
                  <Typography
                    sx={{ color: tokens.text, fontFamily: monoFont, fontSize: 12, mt: 0.6 }}
                  >
                    {value}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Box>
        </Paper>

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', xl: 'minmax(0,1.55fr) minmax(340px,1fr)' },
            gap: 2,
          }}
        >
          <Paper sx={{ ...glassCard, p: { xs: 1.5, md: 2 } }}>
            <SectionHeader
              eyebrow="Prévisions marines"
              title="Vague totale, houle et période"
              meta={mode === 'NEXT_24H' ? 'HEURE PAR HEURE · 24 H' : 'TENDANCE · JUSQU’À 72 H'}
            />
            {selectedLoading ? (
              <Skeleton
                variant="rounded"
                height={315}
                sx={{ mt: 1, bgcolor: 'rgba(255,255,255,0.055)' }}
              />
            ) : hasSelectedData ? (
              <MarineAnalyticsChart
                live={marineHourly}
                forecast={forecasts}
                mode={mode}
                height={315}
              />
            ) : (
              <Box sx={{ height: 315, display: 'grid', placeItems: 'center', color: tokens.muted }}>
                Série indisponible
              </Box>
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
                  sx={{
                    borderBottom:
                      index < activity.length - 1 ? `1px solid ${tokens.border}` : 'none',
                  }}
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
                    <Typography noWrap sx={{ color: tokens.text, fontSize: 12, fontWeight: 700 }}>
                      {item.title}
                    </Typography>
                    <Typography
                      noWrap
                      sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 10, mt: 0.3 }}
                    >
                      {item.detail}
                    </Typography>
                  </Box>
                </Stack>
              ))}
            </Stack>
            <Stack
              direction="row"
              justifyContent="space-between"
              alignItems="center"
              mt={1.5}
              pt={1.4}
              borderTop={`1px solid ${tokens.border}`}
            >
              <Stack direction="row" gap={0.75} alignItems="center">
                <Box
                  sx={{
                    width: 7,
                    height: 7,
                    borderRadius: '50%',
                    bgcolor: autoRefresh ? tokens.teal : tokens.muted,
                  }}
                />
                <Typography sx={{ color: tokens.muted, fontSize: 10 }}>Auto 5 min</Typography>
              </Stack>
              <Button
                size="small"
                aria-pressed={autoRefresh}
                onClick={() => onAutoRefreshChange(!autoRefresh)}
                sx={{
                  minHeight: 44,
                  color: autoRefresh ? tokens.teal : tokens.muted,
                  fontFamily: monoFont,
                  fontSize: 10,
                }}
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
                Observations et tendances jusqu’à 72 h · Décision confirmée par l’exploitation
              </Typography>
            </Stack>
            <Typography sx={{ color: tokens.muted, fontFamily: monoFont, fontSize: 10 }}>
              SYNCHRO {lastUpdatedAt ? new Date(lastUpdatedAt).toLocaleTimeString('fr-FR') : '—'}
            </Typography>
          </Stack>
        </Paper>

        <Typography sx={{ color: tokens.muted, fontSize: 10, lineHeight: 1.6 }}>
          Conditions et prévisions indicatives, actualisées automatiquement. Les données côtières ne
          remplacent pas les sources nautiques réglementaires.
        </Typography>
      </Stack>
    </Box>
  );
};

export default MetoceanAnalyticsDashboard;
