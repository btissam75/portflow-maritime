import {
  Alert,
  Box,
  Button,
  InputAdornment,
  Paper,
  Skeleton,
  Stack,
  TextField,
  Typography,
} from '@mui/material';
import type { KeyboardEvent } from 'react';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import IconifyIcon from 'components/base/IconifyIcon';
import CapacityEtaBandChart from 'components/sections/maritime/CapacityEtaBandChart';
import CapacityOperationsMap from 'components/sections/maritime/CapacityOperationsMap';
import CapacityReplayRail from 'components/sections/maritime/CapacityReplayRail';
import CapacityTimelineChart from 'components/sections/maritime/CapacityTimelineChart';
import { useSearchParams } from 'react-router-dom';
import { capacityApi } from 'services/capacityApi';
import type { CapacityEvaluationRole } from 'services/capacityApi';
import type { CapacityDashboardData, CapacityDecision } from 'types/capacity';
import { portflowPalette as pf } from 'theme/portflowPalette';

const colors = {
  bg: pf.background.primary,
  surface: pf.background.panel,
  surfaceRaised: pf.background.panelRaised,
  border: pf.structure.border,
  borderStrong: pf.functional.cyan,
  text: pf.text.primary,
  muted: pf.text.secondary,
  cyan: pf.functional.cyan,
  blue: pf.functional.blue,
  primary: pf.functional.cyan,
  amber: pf.functional.amber,
  red: pf.functional.red,
  green: pf.functional.green,
};
const mono = 'Inter, "Segoe UI", Arial, sans-serif';
const display = mono;
const heading = display;
const REPLAY_ROLE: CapacityEvaluationRole = 'VALID_SELECT';
const REPLAY_INTERVAL_MS = 8_000;
const REVIEW_DRAFT_KEY = 'portflow.capacity.review-drafts.v1';
const toolSurface = {
  bgcolor: colors.surface,
  border: `1px solid ${colors.border}`,
  borderRadius: '8px',
  boxShadow: 'none',
} as const;

const localTime = (value: string | null | undefined, withDate = false) =>
  value
    ? new Intl.DateTimeFormat('fr-FR', {
        ...(withDate ? { day: '2-digit', month: 'short' } : {}),
        hour: '2-digit',
        minute: '2-digit',
        timeZone: 'Africa/Casablanca',
      }).format(new Date(value))
    : '—';
const percent = (value: number | null | undefined) => {
  if (value == null || !Number.isFinite(value)) return '—';
  const numeric = Math.max(0, value * 100);
  if (numeric > 0 && numeric < 0.1) return '< 0,1 %';
  return `${numeric.toLocaleString('fr-FR', { maximumFractionDigits: numeric < 10 ? 1 : 0 })} %`;
};

const readReviewDrafts = () => {
  if (typeof window === 'undefined') return new Set<string>();
  try {
    const parsed = JSON.parse(window.sessionStorage.getItem(REVIEW_DRAFT_KEY) ?? '[]') as unknown;
    return new Set(
      Array.isArray(parsed)
        ? parsed.filter((value): value is string => typeof value === 'string')
        : [],
    );
  } catch {
    return new Set<string>();
  }
};

const writeReviewDrafts = (drafts: Set<string>) => {
  try {
    window.sessionStorage.setItem(REVIEW_DRAFT_KEY, JSON.stringify([...drafts]));
  } catch {
    // La préparation locale reste utilisable si le stockage de session est désactivé.
  }
};
const tier = (item: CapacityDecision) => {
  if (item.risk_score >= 0.65) return { label: 'Critique', color: colors.red };
  if (item.watchlist_selected || item.risk_score >= 0.4)
    return { label: 'Vigilance', color: colors.amber };
  return { label: 'Normal', color: colors.green };
};

const operationalStateLabel = (state: string | null | undefined) =>
  ({
    APPROACH: 'En approche',
    WAITING: 'En attente',
    BERTH_WINDOW: 'Créneau d’accostage',
  })[state ?? ''] ??
  state?.replace(/_/g, ' ') ??
  'Indisponible';

type FilterMode = 'ALL' | 'PRIORITY' | 'WATCH';
type RiskGroup = {
  key: 'CRITICAL' | 'WATCH' | 'NORMAL';
  label: string;
  color: string;
  items: CapacityDecision[];
};

const OperationStat = ({
  icon,
  label,
  value,
  detail,
  color,
}: {
  icon: string;
  label: string;
  value: string;
  detail: string;
  color: string;
}) => (
  <Stack
    direction="row"
    alignItems="center"
    gap={1}
    sx={{
      minWidth: 0,
      minHeight: 82,
      px: { xs: 1.2, sm: 1.6 },
      py: 1.15,
      bgcolor: colors.surface,
      border: `1px solid ${colors.border}`,
      borderRadius: '8px',
      animation: 'portflowCardIn 360ms cubic-bezier(.2,.8,.2,1) both',
      transition: 'border-color 180ms ease, background-color 180ms ease',
    }}
  >
    <Box
      sx={{
        width: 36,
        height: 36,
        display: 'grid',
        placeItems: 'center',
        color,
        bgcolor: `${color}18`,
        border: `1px solid ${color}48`,
        borderRadius: '7px',
        flex: '0 0 auto',
      }}
    >
      <IconifyIcon icon={icon} sx={{ fontSize: 20 }} />
    </Box>
    <Box minWidth={0}>
      <Typography
        sx={{
          color: pf.text.secondary,
          fontFamily: heading,
          fontSize: 10.5,
          fontWeight: 600,
          textTransform: 'uppercase',
        }}
      >
        {label}
      </Typography>
      <Stack direction="row" alignItems="baseline" gap={0.7}>
        <Typography
          sx={{
            color,
            fontFamily: mono,
            fontSize: { xs: 23, md: 27 },
            lineHeight: 1.25,
            fontWeight: 700,
          }}
        >
          {value}
        </Typography>
        <Typography
          noWrap
          sx={{ display: { xs: 'none', xl: 'block' }, color: colors.muted, fontSize: 10 }}
        >
          {detail}
        </Typography>
      </Stack>
    </Box>
  </Stack>
);

const OperationalHorizon = ({ decision }: { decision: CapacityDecision | undefined }) => {
  const steps = [
    {
      label: 'Score actuel',
      sublabel: 'Priorité de revue',
      value: decision?.risk_score,
      color: colors.cyan,
    },
    {
      label: 'Retard > 3 h',
      sublabel: 'Probabilité estimée',
      value: decision?.p_delay_gt3,
      color: colors.blue,
    },
    {
      label: 'Horizon 6 h',
      sublabel: 'Risque à court terme',
      value: decision?.hazard_6h,
      color: colors.amber,
    },
    {
      label: 'Horizon 12 h',
      sublabel: 'Risque à moyen terme',
      value: decision?.hazard_12h,
      color: colors.red,
    },
  ];

  return (
    <Paper
      component="section"
      aria-label="Signaux de risque de l'escale sélectionnée"
      sx={{ ...toolSurface, overflow: 'hidden' }}
    >
      <Stack
        direction={{ xs: 'column', md: 'row' }}
        justifyContent="space-between"
        alignItems={{ xs: 'stretch', md: 'center' }}
        gap={1}
        sx={{ px: { xs: 1.5, md: 2 }, py: 1.05, borderBottom: `1px solid ${colors.border}` }}
      >
        <Stack direction="row" alignItems="center" gap={1} minWidth={0}>
          <Box
            sx={{
              width: 32,
              height: 32,
              display: 'grid',
              placeItems: 'center',
              border: `1px solid ${colors.cyan}66`,
              color: colors.cyan,
              borderRadius: '6px',
              flex: '0 0 auto',
            }}
          >
            <IconifyIcon icon="material-symbols:timeline-rounded" sx={{ fontSize: 18 }} />
          </Box>
          <Box minWidth={0}>
            <Typography
              sx={{ color: colors.cyan, fontFamily: mono, fontSize: 9.5, fontWeight: 700 }}
            >
              SIGNAUX DE RISQUE
            </Typography>
            <Typography
              noWrap
              sx={{ color: colors.text, fontFamily: heading, fontSize: 15.5, fontWeight: 600 }}
            >
              {decision?.vessel_name ?? 'Aucune escale sélectionnée'}
            </Typography>
          </Box>
        </Stack>
        <Typography sx={{ color: colors.muted, fontSize: 11 }}>
          Indicateurs distincts, à interpréter ensemble
        </Typography>
      </Stack>

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: 'repeat(2,minmax(0,1fr))', md: 'repeat(4,minmax(0,1fr))' },
        }}
      >
        {steps.map((step, index) => {
          const value = step.value == null ? null : Math.max(0, Math.min(1, step.value));
          const signalColor =
            value == null
              ? colors.muted
              : value >= 0.65
                ? colors.red
                : value >= 0.4
                  ? colors.amber
                  : step.color;
          return (
            <Box
              key={step.label}
              sx={{
                minWidth: 0,
                px: { xs: 1.3, md: 1.7 },
                py: 1.25,
                borderRight: {
                  xs: index % 2 === 0 ? `1px solid ${colors.border}` : 0,
                  md: index < 3 ? `1px solid ${colors.border}` : 0,
                },
                borderBottom: { xs: index < 2 ? `1px solid ${colors.border}` : 0, md: 0 },
              }}
            >
              <Stack direction="row" justifyContent="space-between" alignItems="baseline" gap={1}>
                <Box minWidth={0}>
                  <Typography
                    sx={{ color: colors.text, fontFamily: mono, fontSize: 10.5, fontWeight: 700 }}
                  >
                    {step.label}
                  </Typography>
                  <Typography noWrap sx={{ color: colors.muted, fontSize: 9.5 }}>
                    {step.sublabel}
                  </Typography>
                </Box>
                <Typography
                  sx={{ color: signalColor, fontFamily: mono, fontSize: 15, fontWeight: 700 }}
                >
                  {percent(value)}
                </Typography>
              </Stack>
              <Box
                sx={{
                  height: 4,
                  mt: 1,
                  bgcolor: pf.background.secondary,
                  overflow: 'hidden',
                  borderRadius: '2px',
                }}
              >
                <Box
                  sx={{
                    width: value == null ? 0 : `${Math.max(2, value * 100)}%`,
                    height: 1,
                    bgcolor: signalColor,
                    transformOrigin: 'left center',
                    animation: 'portflowSignalIn 650ms cubic-bezier(.2,.8,.2,1) both',
                    animationDelay: `${index * 70}ms`,
                  }}
                />
              </Box>
            </Box>
          );
        })}
      </Box>
    </Paper>
  );
};

const RiskDial = ({ value, color }: { value: number | undefined; color: string }) => {
  const degrees = value == null ? 0 : Math.max(0, Math.min(360, value * 360));
  const score =
    value == null ? '—' : value > 0 && value < 0.001 ? '<0,1' : Math.round(value * 100).toString();
  return (
    <Box
      sx={{
        position: 'relative',
        width: 112,
        height: 112,
        display: 'grid',
        placeItems: 'center',
        borderRadius: '50%',
        background: `conic-gradient(${color} 0deg ${degrees}deg, ${pf.background.secondary} ${degrees}deg 360deg)`,
        '&::before': {
          content: '""',
          position: 'absolute',
          inset: 9,
          borderRadius: '50%',
          bgcolor: colors.surface,
          border: `1px solid ${colors.border}`,
        },
      }}
    >
      <Stack alignItems="center" sx={{ position: 'relative' }}>
        <Typography sx={{ color, fontFamily: mono, fontSize: 22, fontWeight: 700 }}>
          {score}
        </Typography>
        <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 8 }}>
          RISQUE / 100
        </Typography>
      </Stack>
    </Box>
  );
};

const CapacityPage = () => {
  const [searchParams] = useSearchParams();
  const initialCache = useRef(capacityApi.getCachedDashboardEntry(REPLAY_ROLE));
  const [data, setData] = useState<CapacityDashboardData | null>(
    () => initialCache.current?.value ?? null,
  );
  const hasDashboardData = useRef(data != null);
  const [replaySnapshots, setReplaySnapshots] = useState<
    NonNullable<CapacityDashboardData['snapshot']>[]
  >([]);
  const [replayIndex, setReplayIndex] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(true);
  const [replaySpeed, setReplaySpeed] = useState<1 | 2 | 4>(1);
  const [replayLoading, setReplayLoading] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<CapacityDecision[]>([]);
  const [filter, setFilter] = useState<FilterMode>(() =>
    searchParams.get('filter') === 'critical' ? 'PRIORITY' : 'ALL',
  );
  const [loading, setLoading] = useState(data == null);
  const [timelineLoading, setTimelineLoading] = useState(false);
  const [timelineError, setTimelineError] = useState<string | null>(null);
  const [reviewDrafts, setReviewDrafts] = useState<Set<string>>(readReviewDrafts);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [sourceState, setSourceState] = useState<
    'NETWORK' | 'PARTIAL' | 'CACHE' | 'STALE' | 'UNAVAILABLE'
  >(initialCache.current ? (initialCache.current.stale ? 'STALE' : 'CACHE') : 'UNAVAILABLE');
  const [updatedAt, setUpdatedAt] = useState<string | null>(initialCache.current?.cachedAt ?? null);

  const load = useCallback(async (signal?: AbortSignal) => {
    setLoading(!hasDashboardData.current);
    setError(null);
    try {
      const response = await capacityApi.getDashboard(signal, REPLAY_ROLE);
      setData(response);
      hasDashboardData.current = true;
      setSourceState(response.unavailable.length ? 'PARTIAL' : 'NETWORK');
      setUpdatedAt(response.fetchedAt);
      const initial = response.snapshot?.decisions[0]?.port_call_id ?? null;
      setSelectedId((current) => current ?? initial);
      if (response.snapshot) {
        setReplayLoading(true);
        void capacityApi
          .getReplaySnapshots(response.snapshot.resolved_at, REPLAY_ROLE, signal)
          .then((frames) => {
            if (signal?.aborted || !frames.length) return;
            setReplaySnapshots(frames);
            setReplayIndex(frames.length - 1);
          })
          .finally(() => {
            if (!signal?.aborted) setReplayLoading(false);
          });
      }
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === 'AbortError') return;
      setSourceState(hasDashboardData.current ? 'STALE' : 'UNAVAILABLE');
      setError(
        reason instanceof Error ? reason.message : 'La vigilance des escales est indisponible.',
      );
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    const interval = window.setInterval(() => void load(), 5 * 60_000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [load]);

  useEffect(() => {
    if (searchParams.get('filter') === 'critical') setFilter('PRIORITY');
  }, [searchParams]);

  useEffect(() => {
    if (!replayPlaying || replaySnapshots.length < 2) return undefined;
    const timer = window.setInterval(
      () => setReplayIndex((current) => (current + 1) % replaySnapshots.length),
      REPLAY_INTERVAL_MS / replaySpeed,
    );
    return () => window.clearInterval(timer);
  }, [replayPlaying, replaySnapshots.length, replaySpeed]);

  useEffect(() => {
    const frame = replaySnapshots[replayIndex];
    if (!frame) return;
    setData((current) => (current ? { ...current, snapshot: frame } : current));
    setSelectedId((current) =>
      frame.decisions.some((item) => item.port_call_id === current)
        ? current
        : frame.decisions[0]?.port_call_id ?? null,
    );
  }, [replayIndex, replaySnapshots]);

  useEffect(() => {
    if (!selectedId) return undefined;
    const controller = new AbortController();
    const cachedTimeline = capacityApi.getCachedTimelineEntry(selectedId);
    setTimeline(cachedTimeline?.value ?? []);
    setTimelineError(cachedTimeline?.stale ? 'Historique affiché depuis un cache ancien.' : null);
    setTimelineLoading(!cachedTimeline?.value.length);
    capacityApi
      .getTimeline(selectedId, controller.signal)
      .then((response) => {
        setTimeline(response);
        setTimelineError(null);
      })
      .catch((reason: unknown) => {
        if (controller.signal.aborted) return;
        if (!cachedTimeline?.value.length) setTimeline([]);
        setTimelineError(
          cachedTimeline?.value.length
            ? 'La trajectoire en direct est indisponible. La dernière version en cache reste affichée.'
            : reason instanceof Error
              ? reason.message
              : 'La trajectoire de cette escale est indisponible.',
        );
      })
      .finally(() => {
        if (!controller.signal.aborted) setTimelineLoading(false);
      });
    return () => controller.abort();
  }, [selectedId]);

  const decisions = useMemo(
    () =>
      [...(data?.snapshot?.decisions ?? [])].sort((left, right) => {
        if (left.watchlist_selected !== right.watchlist_selected)
          return left.watchlist_selected ? -1 : 1;
        return right.risk_score - left.risk_score;
      }),
    [data],
  );
  const selected = decisions.find((item) => item.port_call_id === selectedId) ?? decisions[0];
  const visibleDecisions = useMemo(() => {
    const normalizedQuery = query.trim().toLocaleLowerCase('fr-FR');
    return decisions.filter((item) => {
      if (filter === 'PRIORITY' && item.risk_score < 0.65) return false;
      if (filter === 'WATCH' && item.risk_score < 0.4) return false;
      if (!normalizedQuery) return true;
      return [
        item.vessel_name,
        item.terminal_code,
        item.port_code,
        item.vessel_type,
        item.cargo_group,
      ]
        .filter(Boolean)
        .some((value) => value?.toLocaleLowerCase('fr-FR').includes(normalizedQuery));
    });
  }, [decisions, filter, query]);
  const groupedDecisions = useMemo<RiskGroup[]>(() => {
    const groups: RiskGroup[] = [
      {
        key: 'CRITICAL',
        label: 'Critique',
        color: colors.red,
        items: visibleDecisions.filter((item) => item.risk_score >= 0.65),
      },
      {
        key: 'WATCH',
        label: 'Vigilance',
        color: colors.amber,
        items: visibleDecisions.filter(
          (item) => item.risk_score < 0.65 && (item.watchlist_selected || item.risk_score >= 0.4),
        ),
      },
      {
        key: 'NORMAL',
        label: 'Normal',
        color: colors.green,
        items: visibleDecisions.filter((item) => item.risk_score < 0.4 && !item.watchlist_selected),
      },
    ];
    return groups.filter((group) => group.items.length > 0);
  }, [visibleDecisions]);

  useEffect(() => {
    if (!visibleDecisions.length) return;
    if (!visibleDecisions.some((item) => item.port_call_id === selectedId)) {
      setSelectedId(visibleDecisions[0].port_call_id);
    }
  }, [selectedId, visibleDecisions]);

  const snapshot = data?.snapshot;
  const loadPct = snapshot?.active_calls
    ? Math.round((snapshot.selected_calls / snapshot.active_calls) * 100)
    : 0;
  const scheduledReviews = decisions.filter((item) => item.watchlist_selected).length;
  const selectedStatus = selected ? tier(selected) : { label: 'Indisponible', color: colors.muted };
  const intervalWidth = selected
    ? Math.max(0, selected.remaining_p90_h - selected.remaining_p10_h)
    : 0;
  const reviewPrepared = selected ? reviewDrafts.has(selected.port_call_id) : false;
  const sourcePresentation = {
    NETWORK: { label: 'Données à jour', color: colors.green },
    PARTIAL: { label: 'Données partielles', color: colors.amber },
    CACHE: { label: 'Dernières données disponibles', color: colors.blue },
    STALE: { label: 'Données anciennes', color: colors.amber },
    UNAVAILABLE: { label: 'Indisponible', color: colors.red },
  }[sourceState];
  const hasSnapshot = Boolean(snapshot);
  const servingMode = 'HISTORIQUE DISPONIBLE';

  const toggleReviewDraft = () => {
    if (!selected) return;
    setReviewDrafts((current) => {
      const next = new Set(current);
      if (next.has(selected.port_call_id)) next.delete(selected.port_call_id);
      else next.add(selected.port_call_id);
      writeReviewDrafts(next);
      return next;
    });
  };

  const handleDecisionKeyDown = (
    event: KeyboardEvent<HTMLButtonElement>,
    item: CapacityDecision,
  ) => {
    if (event.key === 'Escape') {
      event.currentTarget.blur();
      return;
    }
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    event.preventDefault();
    const current = visibleDecisions.findIndex(
      (decision) => decision.port_call_id === item.port_call_id,
    );
    const nextIndex =
      event.key === 'ArrowDown'
        ? Math.min(visibleDecisions.length - 1, current + 1)
        : Math.max(0, current - 1);
    const next = visibleDecisions[nextIndex];
    if (!next) return;
    setSelectedId(next.port_call_id);
    window.requestAnimationFrame(() => {
      document
        .querySelector<HTMLButtonElement>(`[data-port-call-id="${CSS.escape(next.port_call_id)}"]`)
        ?.focus();
    });
  };

  return (
    <Box sx={{ width: 1, maxWidth: 1840, mx: 'auto', color: colors.text }}>
      <Stack gap={1.5}>
        <Stack
          direction={{ xs: 'column', lg: 'row' }}
          justifyContent="space-between"
          alignItems={{ xs: 'stretch', lg: 'flex-end' }}
          gap={1.5}
        >
          <Box sx={{ position: 'relative', pb: 1.4 }}>
            <Stack direction="row" alignItems="center" gap={0.8} mb={0.55}>
              <Typography
                sx={{ color: colors.cyan, fontFamily: mono, fontSize: 10, fontWeight: 700 }}
              >
                OPS / ESCALES
              </Typography>
              <Box sx={{ width: 26, height: 1, bgcolor: colors.borderStrong }} />
              <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 9.5 }}>
                SUPERVISION TEMPORELLE
              </Typography>
            </Stack>
            <Typography
              component="h1"
              sx={{
                color: colors.text,
                fontFamily: display,
                fontSize: { xs: 24, md: 28 },
                lineHeight: '34px',
                fontWeight: 700,
                animation: 'portflowTitleIn 320ms cubic-bezier(.2,.8,.2,1) both',
              }}
            >
              Vigilance des escales
            </Typography>
            <Box
              aria-hidden="true"
              sx={{
                width: 72,
                height: 2,
                mt: 0.65,
                bgcolor: colors.cyan,
                transformOrigin: 'left center',
                animation: 'portflowTitleRule 520ms 120ms cubic-bezier(.2,.8,.2,1) both',
              }}
            />
            <Typography sx={{ color: pf.text.secondary, fontSize: 13, lineHeight: 1.5, mt: 0.5 }}>
              Prioriser les revues humaines selon le risque temporel et la capacité disponible.
            </Typography>
          </Box>

          <Stack direction="row" alignItems="center" gap={0.75} flexWrap="wrap">
            <Box
              sx={{
                px: 1.1,
                py: 0.65,
                border: `1px solid ${sourcePresentation.color}66`,
                borderRadius: '6px',
                bgcolor: `${sourcePresentation.color}12`,
              }}
            >
              <Stack direction="row" alignItems="center" gap={0.65}>
                <Box
                  sx={{
                    width: 7,
                    height: 7,
                    borderRadius: '50%',
                    bgcolor: sourcePresentation.color,
                    animation: sourceState === 'NETWORK' ? 'portflowPulse 2s infinite' : 'none',
                  }}
                />
                <Box>
                  <Typography
                    sx={{
                      color: sourcePresentation.color,
                      fontFamily: mono,
                      fontSize: 8,
                      fontWeight: 700,
                    }}
                  >
                    {sourcePresentation.label.toUpperCase()}
                  </Typography>
                  <Typography sx={{ color: colors.text, fontFamily: mono, fontSize: 9.5 }}>
                    {replayLoading
                      ? 'PRÉPARATION…'
                      : replaySnapshots.length
                        ? `HISTORIQUE ${replayIndex + 1}/${replaySnapshots.length}`
                        : servingMode}
                  </Typography>
                </Box>
              </Stack>
            </Box>
            <Stack
              direction="row"
              gap={0.35}
              sx={{
                p: 0.35,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                bgcolor: colors.surface,
              }}
            >
              <Button
                aria-label="Cycle historique précédent"
                title="Cycle précédent"
                disabled={replaySnapshots.length < 2}
                onClick={() => {
                  setReplayPlaying(false);
                  setReplayIndex(
                    (current) => (current - 1 + replaySnapshots.length) % replaySnapshots.length,
                  );
                }}
                sx={{
                  minWidth: 40,
                  width: 40,
                  height: 40,
                  color: colors.text,
                  borderRadius: '4px',
                }}
              >
                <IconifyIcon icon="material-symbols:skip-previous-rounded" sx={{ fontSize: 18 }} />
              </Button>
              <Button
                aria-label={
                  replayPlaying
                    ? 'Mettre la lecture de l’historique en pause'
                    : 'Reprendre la lecture de l’historique'
                }
                title={replayPlaying ? 'Pause' : 'Lecture'}
                disabled={replaySnapshots.length < 2}
                onClick={() => setReplayPlaying((current) => !current)}
                sx={{
                  minWidth: 40,
                  width: 40,
                  height: 40,
                  color: pf.background.primary,
                  bgcolor: colors.primary,
                  borderRadius: '4px',
                  '&:hover': { bgcolor: '#54E3DD' },
                  '&.Mui-disabled': { bgcolor: colors.surfaceRaised, color: colors.muted },
                }}
              >
                <IconifyIcon
                  icon={
                    replayPlaying
                      ? 'material-symbols:pause-rounded'
                      : 'material-symbols:play-arrow-rounded'
                  }
                  sx={{ fontSize: 18 }}
                />
              </Button>
              <Button
                aria-label="Cycle historique suivant"
                title="Cycle suivant"
                disabled={replaySnapshots.length < 2}
                onClick={() => {
                  setReplayPlaying(false);
                  setReplayIndex((current) => (current + 1) % replaySnapshots.length);
                }}
                sx={{
                  minWidth: 40,
                  width: 40,
                  height: 40,
                  color: colors.text,
                  borderRadius: '4px',
                }}
              >
                <IconifyIcon icon="material-symbols:skip-next-rounded" sx={{ fontSize: 18 }} />
              </Button>
            </Stack>
            <Box
              sx={{
                px: 1.1,
                py: 0.65,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                bgcolor: colors.surface,
              }}
            >
              <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 8.5 }}>
                INSTANT ANALYSÉ
              </Typography>
              <Typography sx={{ color: colors.text, fontFamily: mono, fontSize: 11 }}>
                {localTime(snapshot?.resolved_at, true)}
              </Typography>
            </Box>
            <Box
              sx={{
                px: 1.1,
                py: 0.65,
                border: `1px solid ${colors.border}`,
                borderRadius: '6px',
                bgcolor: colors.surface,
              }}
            >
              <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 8.5 }}>
                SYNCHRONISATION
              </Typography>
              <Typography sx={{ color: colors.cyan, fontFamily: mono, fontSize: 11 }}>
                {localTime(updatedAt, true)}
              </Typography>
            </Box>
            <Button
              onClick={() => void load()}
              disabled={loading}
              aria-label="Actualiser la vigilance"
              title="Actualiser"
              sx={{
                minWidth: 44,
                width: 44,
                height: 44,
                borderRadius: '6px',
                border: `1px solid ${colors.border}`,
                color: colors.text,
                bgcolor: colors.surface,
              }}
            >
              <IconifyIcon icon="material-symbols:refresh-rounded" sx={{ fontSize: 18 }} />
            </Button>
          </Stack>
        </Stack>

        {error && (
          <Alert
            severity={hasSnapshot ? 'warning' : 'error'}
            action={
              <Button
                color="inherit"
                size="small"
                onClick={() => void load()}
                disabled={loading}
                sx={{ minHeight: 40 }}
              >
                Réessayer
              </Button>
            }
            sx={{
              bgcolor: hasSnapshot ? pf.functional.amberSoft : pf.functional.redSoft,
              border: `1px solid ${hasSnapshot ? pf.functional.amber : pf.functional.red}55`,
              color: colors.text,
            }}
          >
            <Typography sx={{ fontSize: 11.5, fontWeight: 700 }}>
              {hasSnapshot
                ? 'Connexion interrompue : la dernière situation connue reste affichée.'
                : error}
            </Typography>
            {hasSnapshot && (
              <Typography sx={{ color: colors.muted, fontSize: 10, mt: 0.2 }}>{error}</Typography>
            )}
          </Alert>
        )}
        {data?.unavailable.length ? (
          <Alert
            severity="warning"
            sx={{
              bgcolor: pf.functional.amberSoft,
              border: `1px solid ${pf.functional.amber}55`,
              color: colors.text,
            }}
          >
            Mode partiel · indisponible : {data.unavailable.join(', ')}.
          </Alert>
        ) : null}
        {data?.status?.audit_status === 'DEMO' ? (
          <Alert
            severity="warning"
            sx={{
              bgcolor: 'rgba(242,184,75,.085)',
              border: `1px solid ${pf.functional.amber}42`,
              color: colors.text,
            }}
          >
            <Typography sx={{ fontSize: 11.5, fontWeight: 700 }}>Données d’exercice</Typography>
            <Typography sx={{ color: colors.muted, fontSize: 10, mt: 0.2 }}>
              Les escales affichées servent à tester le poste de supervision. Elles ne représentent
              pas le planning portuaire réel.
            </Typography>
          </Alert>
        ) : null}
        {loading && !data && (
          <Alert
            severity="info"
            sx={{
              bgcolor: pf.functional.purpleSoft,
              border: `1px solid ${pf.functional.purple}42`,
              color: colors.text,
            }}
          >
            Connexion à l’historique décisionnel. Le premier chargement peut prendre quelques
            secondes.
          </Alert>
        )}

        <Box
          aria-label="Synthèse opérationnelle"
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: 'repeat(2,minmax(0,1fr))', sm: 'repeat(4,minmax(0,1fr))' },
            gap: 1,
          }}
        >
          <OperationStat
            icon="material-symbols:directions-boat-rounded"
            label="Escales actives"
            value={loading && !data ? '···' : snapshot ? String(snapshot.active_calls) : '—'}
            detail="dans la fenêtre"
            color={hasSnapshot ? colors.primary : colors.muted}
          />
          <OperationStat
            icon="material-symbols:fact-check-outline-rounded"
            label="Revues planifiées"
            value={
              loading && !data
                ? '···'
                : snapshot
                  ? String(snapshot.selected_calls ?? scheduledReviews)
                  : '—'
            }
            detail={snapshot ? `${loadPct} % de la charge` : 'donnée indisponible'}
            color={hasSnapshot ? colors.cyan : colors.muted}
          />
          <OperationStat
            icon="material-symbols:work-history-rounded"
            label="Capacité de revue"
            value={loading && !data ? '···' : snapshot ? String(snapshot.capacity) : '—'}
            detail="dossiers / fenêtre"
            color={hasSnapshot ? colors.amber : colors.muted}
          />
          <OperationStat
            icon="material-symbols:schedule-rounded"
            label="Cycle de décision"
            value={data?.status?.bucket_hours != null ? `${data.status.bucket_hours} h` : '—'}
            detail="recalcul opérationnel"
            color={data?.status ? colors.green : colors.muted}
          />
        </Box>

        <Paper sx={{ ...toolSurface, overflow: 'hidden' }}>
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            justifyContent="space-between"
            alignItems={{ xs: 'flex-start', sm: 'center' }}
            gap={0.7}
            sx={{ px: { xs: 1.4, md: 1.8 }, py: 1.1, borderBottom: `1px solid ${colors.border}` }}
          >
            <Box>
              <Typography
                sx={{ color: colors.cyan, fontFamily: mono, fontSize: 9, fontWeight: 750 }}
              >
                JUMEAU OPÉRATIONNEL
              </Typography>
              <Typography sx={{ color: colors.text, fontSize: 17, fontWeight: 700 }}>
                Approches, attente et accès terminal
              </Typography>
            </Box>
            <Typography sx={{ color: colors.muted, fontSize: 9.5 }}>
              La carte, la file et la fiche utilisent la même escale sélectionnée
            </Typography>
          </Stack>
          <CapacityOperationsMap
            decisions={decisions}
            selectedId={selected?.port_call_id ?? null}
            onSelect={setSelectedId}
            height={440}
          />
        </Paper>

        <CapacityReplayRail
          snapshots={replaySnapshots}
          index={replayIndex}
          playing={replayPlaying}
          speed={replaySpeed}
          onIndexChange={(index) => {
            setReplayPlaying(false);
            setReplayIndex(index);
          }}
          onPlayingChange={setReplayPlaying}
          onSpeedChange={setReplaySpeed}
        />

        <OperationalHorizon decision={selected} />

        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: 'minmax(0,1fr)', lg: 'minmax(0,3fr) minmax(390px,2fr)' },
            gap: 1.5,
            alignItems: 'start',
          }}
        >
          <Paper sx={{ ...toolSurface, overflow: 'hidden' }}>
            <Stack
              direction={{ xs: 'column', sm: 'row' }}
              justifyContent="space-between"
              alignItems={{ xs: 'stretch', sm: 'center' }}
              gap={1}
              px={1.5}
              py={1.2}
              borderBottom={`1px solid ${colors.border}`}
            >
              <Stack direction="row" alignItems="center" gap={0.9}>
                <Box sx={{ width: 3, height: 28, bgcolor: colors.cyan }} />
                <Box>
                  <Typography
                    sx={{
                      color: colors.text,
                      fontFamily: heading,
                      fontSize: { xs: 19, md: 21 },
                      lineHeight: 1.15,
                      fontWeight: 700,
                      animation: 'portflowTitleIn 500ms cubic-bezier(.2,.8,.2,1) both',
                    }}
                  >
                    File de vigilance
                  </Typography>
                  <Typography sx={{ color: colors.muted, fontSize: 10.5 }}>
                    {visibleDecisions.length} escales · flèches haut/bas pour naviguer
                  </Typography>
                </Box>
              </Stack>
              <Stack
                direction={{ xs: 'column', md: 'row' }}
                gap={0.7}
                alignItems={{ xs: 'stretch', md: 'center' }}
              >
                <TextField
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Navire, terminal, type…"
                  aria-label="Rechercher une escale"
                  size="small"
                  InputProps={{
                    startAdornment: (
                      <InputAdornment position="start">
                        <IconifyIcon
                          icon="material-symbols:search-rounded"
                          sx={{ color: colors.muted, fontSize: 17 }}
                        />
                      </InputAdornment>
                    ),
                  }}
                  sx={{
                    minWidth: { md: 210 },
                    '& .MuiOutlinedInput-root': {
                      height: 40,
                      color: colors.text,
                      bgcolor: colors.bg,
                      fontSize: 10.5,
                    },
                    '& .MuiOutlinedInput-notchedOutline': { borderColor: colors.border },
                  }}
                />
                <Stack
                  direction="row"
                  gap={0.35}
                  sx={{
                    bgcolor: colors.bg,
                    border: `1px solid ${colors.border}`,
                    p: 0.35,
                    borderRadius: '6px',
                  }}
                >
                  {(
                    [
                      ['ALL', 'Toutes'],
                      ['WATCH', 'Vigilance'],
                      ['PRIORITY', 'Critiques'],
                    ] as Array<[FilterMode, string]>
                  ).map(([value, label]) => (
                    <Button
                      key={value}
                      aria-pressed={filter === value}
                      onClick={() => setFilter(value)}
                      size="small"
                      sx={{
                        minWidth: 0,
                        minHeight: 40,
                        px: 1,
                        borderRadius: '4px',
                        color: filter === value ? pf.background.primary : colors.muted,
                        bgcolor: filter === value ? colors.primary : 'transparent',
                        fontSize: 10,
                        '&:hover': { bgcolor: filter === value ? '#54E3DD' : colors.surfaceRaised },
                      }}
                    >
                      {label}
                    </Button>
                  ))}
                </Stack>
              </Stack>
            </Stack>

            <Box
              sx={{
                maxHeight: { xs: 540, lg: 720 },
                overflowY: 'auto',
                scrollbarColor: `${colors.borderStrong} ${colors.bg}`,
              }}
            >
              {loading ? (
                Array.from({ length: 6 }).map((_, index) => (
                  <Skeleton key={index} height={82} sx={{ bgcolor: colors.surfaceRaised, mx: 1 }} />
                ))
              ) : groupedDecisions.length ? (
                groupedDecisions.map((group) => (
                  <Box
                    key={group.key}
                    component="section"
                    aria-label={`${group.label}, ${group.items.length} escales`}
                  >
                    <Stack
                      direction="row"
                      alignItems="center"
                      gap={0.8}
                      sx={{
                        position: 'sticky',
                        top: 0,
                        zIndex: 2,
                        px: 1.5,
                        py: 0.65,
                        bgcolor: pf.background.navigation,
                        borderBottom: `1px solid ${colors.border}`,
                      }}
                    >
                      <Box
                        sx={{ width: 7, height: 7, borderRadius: '50%', bgcolor: group.color }}
                      />
                      <Typography
                        sx={{
                          color: group.color,
                          fontFamily: heading,
                          fontSize: 10,
                          fontWeight: 600,
                          textTransform: 'uppercase',
                        }}
                      >
                        {group.label}
                      </Typography>
                      <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 8 }}>
                        {group.items.length}
                      </Typography>
                    </Stack>
                    {group.items.map((item, index) => {
                      const status = tier(item);
                      const active = item.port_call_id === selected?.port_call_id;
                      return (
                        <Box
                          key={item.port_call_id}
                          component="button"
                          type="button"
                          data-port-call-id={item.port_call_id}
                          onClick={() => setSelectedId(item.port_call_id)}
                          onKeyDown={(event) => handleDecisionKeyDown(event, item)}
                          sx={{
                            position: 'relative',
                            display: 'grid',
                            gridTemplateColumns: {
                              xs: 'minmax(0,1fr) auto',
                              md: 'minmax(210px,2.2fr) minmax(95px,.9fr) minmax(92px,.8fr) 62px',
                            },
                            alignItems: 'center',
                            gap: { xs: 0.75, md: 1.1 },
                            width: 1,
                            minHeight: 80,
                            px: 1.5,
                            py: 1,
                            color: 'inherit',
                            textAlign: 'left',
                            bgcolor: active ? pf.functional.cyanSoft : 'transparent',
                            border: 0,
                            borderBottom: `1px solid ${colors.border}`,
                            cursor: 'pointer',
                            contentVisibility: 'auto',
                            containIntrinsicSize: '80px',
                            animation: 'portflowRowIn 300ms cubic-bezier(.2,.8,.2,1) both',
                            animationDelay: `${Math.min(index, 8) * 22}ms`,
                            '&::before': {
                              content: '""',
                              position: 'absolute',
                              inset: '0 auto 0 0',
                              width: active ? 4 : 2,
                              bgcolor: status.color,
                              transition: 'width 160ms ease',
                            },
                            '&:hover': {
                              bgcolor: active ? pf.functional.cyanSoft : pf.background.panelHover,
                            },
                            '&:focus-visible': {
                              outline: `2px solid ${colors.blue}`,
                              outlineOffset: -2,
                            },
                          }}
                        >
                          <Box minWidth={0}>
                            <Stack direction="row" alignItems="center" gap={0.65}>
                              <Typography
                                noWrap
                                sx={{
                                  color: colors.text,
                                  fontFamily: heading,
                                  fontSize: 14,
                                  fontWeight: 600,
                                }}
                              >
                                {item.vessel_name || 'Navire non identifié'}
                              </Typography>
                              {item.watchlist_selected && (
                                <IconifyIcon
                                  icon="material-symbols:bookmark-rounded"
                                  sx={{ color: colors.cyan, fontSize: 14 }}
                                />
                              )}
                            </Stack>
                            <Typography
                              noWrap
                              sx={{
                                color: colors.muted,
                                fontFamily: mono,
                                fontSize: 8.5,
                                mt: 0.35,
                              }}
                            >
                              {item.terminal_code || item.port_code || 'Terminal non renseigné'} ·{' '}
                              {item.vessel_type || item.cargo_group || 'Type non renseigné'}
                            </Typography>
                          </Box>
                          <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                            <Typography
                              sx={{ color: colors.muted, fontSize: 9, textTransform: 'uppercase' }}
                            >
                              Temps estimé
                            </Typography>
                            <Typography
                              sx={{
                                color: colors.text,
                                fontFamily: mono,
                                fontSize: 12,
                                fontWeight: 700,
                                mt: 0.25,
                              }}
                            >
                              {item.remaining_p50_h.toFixed(1)} h
                            </Typography>
                            <Typography
                              sx={{ color: colors.muted, fontFamily: mono, fontSize: 7.5 }}
                            >
                              Fourchette {item.remaining_p10_h.toFixed(0)} à{' '}
                              {item.remaining_p90_h.toFixed(0)} h
                            </Typography>
                          </Box>
                          <Box sx={{ display: { xs: 'none', md: 'block' } }}>
                            <Typography
                              sx={{ color: colors.muted, fontSize: 9, textTransform: 'uppercase' }}
                            >
                              Retard &gt; 3 h
                            </Typography>
                            <Typography
                              sx={{
                                color: status.color,
                                fontFamily: mono,
                                fontSize: 12,
                                fontWeight: 700,
                                mt: 0.25,
                              }}
                            >
                              {percent(item.p_delay_gt3)}
                            </Typography>
                          </Box>
                          <Box sx={{ justifySelf: 'end', minWidth: 54 }}>
                            <Typography
                              sx={{
                                color: status.color,
                                fontFamily: mono,
                                fontSize: 14,
                                fontWeight: 700,
                                textAlign: 'right',
                              }}
                            >
                              {percent(item.risk_score).replace(' %', '')}
                            </Typography>
                            <Typography
                              sx={{ color: colors.muted, fontSize: 7.5, textAlign: 'right' }}
                            >
                              RISQUE
                            </Typography>
                          </Box>
                        </Box>
                      );
                    })}
                  </Box>
                ))
              ) : (
                <Box
                  sx={{
                    minHeight: 280,
                    display: 'grid',
                    placeItems: 'center',
                    color: colors.muted,
                    fontSize: 11,
                  }}
                >
                  Aucune escale pour ce filtre
                </Box>
              )}
            </Box>
          </Paper>

          <Paper
            key={selected?.port_call_id ?? 'empty'}
            aria-live="polite"
            sx={{
              ...toolSurface,
              p: { xs: 1.4, md: 1.7 },
              position: { lg: 'sticky' },
              top: { lg: 78 },
              animation: 'portflowCardIn 360ms cubic-bezier(.2,.8,.2,1) both',
            }}
          >
            <Stack
              direction="row"
              justifyContent="space-between"
              alignItems="flex-start"
              gap={1}
              pb={1.3}
              borderBottom={`1px solid ${colors.border}`}
            >
              <Box minWidth={0}>
                <Typography
                  sx={{ color: colors.cyan, fontFamily: mono, fontSize: 9.5, fontWeight: 700 }}
                >
                  FICHE ESCALE
                </Typography>
                <Typography
                  noWrap
                  sx={{
                    color: colors.text,
                    fontFamily: display,
                    fontSize: 22,
                    lineHeight: 1.2,
                    fontWeight: 700,
                    mt: 0.3,
                    animation: 'portflowTitleIn 500ms cubic-bezier(.2,.8,.2,1) both',
                  }}
                >
                  {selected?.vessel_name ?? 'Aucune escale'}
                </Typography>
                <Stack direction="row" alignItems="center" gap={0.55} mt={0.35}>
                  <IconifyIcon
                    icon="material-symbols:location-on-outline-rounded"
                    sx={{ color: colors.muted, fontSize: 13 }}
                  />
                  <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 9 }}>
                    {selected?.terminal_code ?? selected?.port_code ?? 'Terminal non renseigné'}
                  </Typography>
                </Stack>
              </Box>
              <Box
                sx={{
                  px: 0.8,
                  py: 0.45,
                  borderRadius: '4px',
                  color: selectedStatus.color,
                  bgcolor: `${selectedStatus.color}12`,
                  border: `1px solid ${selectedStatus.color}45`,
                  fontFamily: mono,
                  fontSize: 8,
                  fontWeight: 700,
                }}
              >
                {selectedStatus.label.toUpperCase()}
              </Box>
            </Stack>

            <Stack direction="row" alignItems="center" gap={1.8} py={1.6}>
              <RiskDial value={selected?.risk_score} color={selectedStatus.color} />
              <Box flex={1} minWidth={0}>
                <Typography
                  sx={{
                    color: colors.muted,
                    fontFamily: heading,
                    fontSize: 10,
                    fontWeight: 600,
                    textTransform: 'uppercase',
                  }}
                >
                  Probabilité de retard &gt; 3 h
                </Typography>
                <Typography
                  sx={{
                    color: selectedStatus.color,
                    fontFamily: mono,
                    fontSize: 22,
                    fontWeight: 700,
                    mt: 0.35,
                  }}
                >
                  {percent(selected?.p_delay_gt3)}
                </Typography>
                <Typography sx={{ color: colors.muted, fontSize: 9, lineHeight: 1.45, mt: 0.55 }}>
                  Ce signal aide à ordonner les revues. L’opérateur confirme la décision.
                </Typography>
              </Box>
            </Stack>

            <Box sx={{ py: 1.3, borderBlock: `1px solid ${colors.border}` }}>
              <Stack direction="row" justifyContent="space-between" alignItems="baseline" mb={0.4}>
                <Box>
                  <Typography
                    sx={{
                      color: colors.muted,
                      fontFamily: heading,
                      fontSize: 10,
                      fontWeight: 600,
                      textTransform: 'uppercase',
                    }}
                  >
                    Temps restant estimé
                  </Typography>
                  <Typography
                    sx={{
                      color: colors.text,
                      fontFamily: mono,
                      fontSize: 21,
                      fontWeight: 700,
                      mt: 0.2,
                    }}
                  >
                    {selected ? `${selected.remaining_p50_h.toFixed(1)} h` : '—'}
                  </Typography>
                </Box>
                <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 8 }}>
                  FOURCHETTE {selected ? `${intervalWidth.toFixed(1)} h` : '—'}
                </Typography>
              </Stack>
              {timelineLoading ? (
                <Skeleton
                  variant="rectangular"
                  height={205}
                  sx={{ bgcolor: colors.surfaceRaised, borderRadius: '6px' }}
                />
              ) : (
                <CapacityEtaBandChart data={timeline} fallback={selected} height={205} />
              )}
            </Box>

            <Stack
              direction="row"
              justifyContent="space-between"
              alignItems="center"
              py={1.2}
              borderBottom={`1px solid ${colors.border}`}
            >
              <Box>
                <Typography
                  sx={{
                    color: colors.muted,
                    fontFamily: heading,
                    fontSize: 10,
                    fontWeight: 600,
                    textTransform: 'uppercase',
                  }}
                >
                  État opérationnel
                </Typography>
                <Typography
                  sx={{
                    color: selected ? colors.cyan : colors.muted,
                    fontFamily: mono,
                    fontSize: 10.5,
                    fontWeight: 700,
                    mt: 0.35,
                  }}
                >
                  {operationalStateLabel(selected?.hsmm_state)}
                </Typography>
              </Box>
              {selected?.hsmm_state_confidence != null && (
                <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 8.5 }}>
                  CONFIANCE {percent(selected.hsmm_state_confidence)}
                </Typography>
              )}
            </Stack>

            <Box
              sx={{
                mt: 1.3,
                p: 1.15,
                borderLeft: `3px solid ${!selected ? colors.muted : selected.watchlist_selected ? selectedStatus.color : colors.green}`,
                bgcolor: !selected
                  ? colors.surfaceRaised
                  : selected.watchlist_selected
                    ? `${selectedStatus.color}0C`
                    : pf.functional.greenSoft,
              }}
            >
              <Stack direction="row" alignItems="center" gap={0.65}>
                <IconifyIcon
                  icon={
                    !selected
                      ? 'material-symbols:database-off-rounded'
                      : selected.watchlist_selected
                        ? 'material-symbols:fact-check-outline-rounded'
                        : 'material-symbols:verified-rounded'
                  }
                  sx={{
                    color: !selected
                      ? colors.muted
                      : selected.watchlist_selected
                        ? selectedStatus.color
                        : colors.green,
                    fontSize: 16,
                  }}
                />
                <Typography
                  sx={{
                    color: !selected
                      ? colors.muted
                      : selected.watchlist_selected
                        ? selectedStatus.color
                        : colors.green,
                    fontFamily: heading,
                    fontSize: 9.5,
                    fontWeight: 700,
                    textTransform: 'uppercase',
                  }}
                >
                  {!selected
                    ? 'Décision indisponible'
                    : selected.watchlist_selected
                      ? 'Revue humaine recommandée'
                      : 'Maintenir la surveillance'}
                </Typography>
              </Stack>
              <Typography sx={{ color: pf.text.secondary, fontSize: 10, lineHeight: 1.5, mt: 0.5 }}>
                {!selected
                  ? 'Aucune recommandation ne peut être formulée sans situation disponible.'
                  : selected.watchlist_selected
                    ? 'Examiner cette escale dans la fenêtre actuelle et documenter la décision opérateur.'
                    : 'Aucune intervention immédiate. Réévaluer pendant le prochain cycle.'}
              </Typography>
            </Box>

            <Button
              fullWidth
              disabled={!selected}
              aria-pressed={reviewPrepared}
              onClick={toggleReviewDraft}
              startIcon={
                <IconifyIcon
                  icon={
                    reviewPrepared
                      ? 'material-symbols:remove-done-rounded'
                      : 'material-symbols:assignment-add-outline-rounded'
                  }
                />
              }
              sx={{
                mt: 1.2,
                minHeight: 44,
                bgcolor: reviewPrepared ? `${colors.green}22` : colors.primary,
                color: reviewPrepared ? colors.green : pf.background.primary,
                border: `1px solid ${reviewPrepared ? colors.green : colors.primary}`,
                borderRadius: '6px',
                fontSize: 11,
                '&:hover': { bgcolor: reviewPrepared ? `${colors.green}2E` : '#54E3DD' },
              }}
            >
              {reviewPrepared ? 'Retirer du brouillon de revue' : 'Ajouter au brouillon de revue'}
            </Button>
            <Typography sx={{ color: colors.muted, fontSize: 8.5, textAlign: 'center', mt: 0.55 }}>
              {reviewDrafts.size} brouillon(s) dans cette session · validation de l’opérateur
              requise avant transmission.
            </Typography>
          </Paper>
        </Box>

        <Box component="section">
          <Stack
            direction={{ xs: 'column', sm: 'row' }}
            justifyContent="space-between"
            alignItems={{ xs: 'flex-start', sm: 'flex-end' }}
            gap={0.6}
            mb={1}
          >
            <Box>
              <Typography
                sx={{ color: colors.cyan, fontFamily: mono, fontSize: 9.5, fontWeight: 700 }}
              >
                HISTORIQUE / {selected?.vessel_name?.toUpperCase() ?? 'AUCUNE ESCALE'}
              </Typography>
              <Typography
                sx={{
                  color: colors.text,
                  fontFamily: heading,
                  fontSize: { xs: 21, md: 24 },
                  lineHeight: 1.15,
                  fontWeight: 700,
                  mt: 0.25,
                  animation: 'portflowTitleIn 520ms cubic-bezier(.2,.8,.2,1) both',
                }}
              >
                Évolution de la vigilance
              </Typography>
            </Box>
            <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 8.5 }}>
              DÉCISION CONFIRMÉE PAR L’OPÉRATEUR
            </Typography>
          </Stack>
          <Paper sx={{ ...toolSurface, p: { xs: 1, sm: 1.4 } }}>
            {timelineError && (
              <Alert
                severity="warning"
                sx={{
                  mb: 1,
                  bgcolor: pf.functional.amberSoft,
                  color: colors.text,
                  border: `1px solid ${colors.amber}55`,
                }}
              >
                {timelineError}
              </Alert>
            )}
            {timelineLoading ? (
              <Skeleton
                variant="rectangular"
                height={270}
                sx={{ bgcolor: colors.surfaceRaised, borderRadius: '6px' }}
              />
            ) : timeline.length > 1 ? (
              <CapacityTimelineChart data={timeline} height={270} />
            ) : (
              <Box
                sx={{
                  height: 270,
                  display: 'grid',
                  placeItems: 'center',
                  color: colors.muted,
                  fontSize: 11,
                }}
              >
                Historique insuffisant pour cette escale
              </Box>
            )}
          </Paper>
        </Box>

        <Stack
          direction={{ xs: 'column', sm: 'row' }}
          justifyContent="space-between"
          gap={0.4}
          px={0.25}
        >
          <Typography sx={{ color: colors.muted, fontSize: 8.5 }}>
            Les priorités proposées restent soumises à la validation de l’opérateur.
          </Typography>
          <Typography sx={{ color: colors.muted, fontFamily: mono, fontSize: 8.5 }}>
            DERNIÈRE SYNCHRO {localTime(updatedAt)}
          </Typography>
        </Stack>
      </Stack>
    </Box>
  );
};

export default CapacityPage;
