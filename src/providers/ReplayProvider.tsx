import { createContext, useCallback, useContext, useEffect, useMemo, useState } from 'react';
import type { PropsWithChildren } from 'react';
import { replayApi } from 'services/replayApi';
import type {
  DataHealth,
  ErrorHeatmapCell,
  ForecastPoint,
  HorizonMetric,
  ModelGovernance,
  OperationalSummary,
  PerformancePoint,
  PortCallItem,
  ReplayRange,
  ReplaySnapshot,
  SourceStatus,
  WeatherPoint,
} from 'types/replay';

const HOUR_MS = 60 * 60 * 1000;

export interface OperationalAlert {
  id: string;
  severity: 'critical' | 'warning' | 'info' | 'success';
  title: string;
  detail: string;
  source: string;
}

interface ReplayContextValue {
  range: ReplayRange | null;
  sourceStatus: SourceStatus | null;
  snapshot: ReplaySnapshot | null;
  timeline: ForecastPoint[];
  metrics: HorizonMetric[];
  summary: OperationalSummary | null;
  portCalls: PortCallItem[];
  weather: WeatherPoint[];
  dataHealth: DataHealth | null;
  modelGovernance: ModelGovernance | null;
  performanceHistory: PerformancePoint[];
  errorHeatmap: ErrorHeatmapCell[];
  alerts: OperationalAlert[];
  asOf: string | null;
  horizon: number;
  speed: number;
  playing: boolean;
  loading: boolean;
  refreshing: boolean;
  error: string | null;
  selectedForecast: ForecastPoint | null;
  selectedMetric: HorizonMetric | null;
  setHorizon: (value: number) => void;
  setSpeed: (value: number) => void;
  setPlaying: (value: boolean | ((current: boolean) => boolean)) => void;
  seek: (value: string) => void;
  resetReplay: () => void;
  jumpToLatest: () => void;
  refresh: () => Promise<void>;
}

const ReplayContext = createContext<ReplayContextValue | null>(null);

const buildAlerts = (
  sourceStatus: SourceStatus | null,
  summary: OperationalSummary | null,
  selectedForecast: ForecastPoint | null,
  selectedMetric: HorizonMetric | null,
): OperationalAlert[] => {
  const alerts: OperationalAlert[] = [];

  if (!sourceStatus?.live_serving_allowed) {
    alerts.push({
      id: 'source-live',
      severity: 'warning',
      title: 'Flux live suspendu',
      detail: 'Les écrans rejouent des décisions historiques validées.',
      source: 'Gouvernance',
    });
  }
  if (summary && summary.ais_positions_72h === 0) {
    alerts.push({
      id: 'ais-missing',
      severity: 'critical',
      title: 'Couche AIS indisponible',
      detail: 'Aucune position navire capturée dans la fenêtre de 72 heures.',
      source: 'AIS',
    });
  }
  if (summary && summary.overdue_calls > 0) {
    alerts.push({
      id: 'overdue-calls',
      severity: summary.overdue_calls >= 3 ? 'critical' : 'warning',
      title: `${summary.overdue_calls} escale${summary.overdue_calls > 1 ? 's' : ''} en dépassement`,
      detail: 'ETA planifiée dépassée sans arrivée observée à cet instant.',
      source: 'Escales',
    });
  }
  if ((summary?.wave_height_m ?? 0) >= 2.5) {
    alerts.push({
      id: 'sea-state',
      severity: 'warning',
      title: 'État de mer défavorable',
      detail: `Hauteur de vague observée ${summary?.wave_height_m?.toFixed(1)} m.`,
      source: 'Météo',
    });
  }
  if (selectedForecast && selectedForecast.p90 - selectedForecast.p10 >= 8) {
    alerts.push({
      id: 'wide-interval',
      severity: 'info',
      title: 'Incertitude prévisionnelle élevée',
      detail: `L’intervalle P10-P90 couvre ${(selectedForecast.p90 - selectedForecast.p10).toFixed(1)} arrivées.`,
      source: 'Modèle',
    });
  }
  if (selectedMetric && selectedMetric.coverage_p10_p90 < 0.72) {
    alerts.push({
      id: 'coverage',
      severity: 'warning',
      title: 'Couverture probabiliste sous la cible',
      detail: `Couverture actuelle ${(selectedMetric.coverage_p10_p90 * 100).toFixed(1)} %, cible 80 %.`,
      source: 'Modèle',
    });
  }
  if (alerts.length === 0) {
    alerts.push({
      id: 'nominal',
      severity: 'success',
      title: 'Situation nominale',
      detail: 'Aucun seuil opérationnel n’est dépassé.',
      source: 'Cockpit',
    });
  }
  return alerts;
};

export const ReplayProvider = ({ children }: PropsWithChildren) => {
  const [range, setRange] = useState<ReplayRange | null>(null);
  const [sourceStatus, setSourceStatus] = useState<SourceStatus | null>(null);
  const [snapshot, setSnapshot] = useState<ReplaySnapshot | null>(null);
  const [timeline, setTimeline] = useState<ForecastPoint[]>([]);
  const [metrics, setMetrics] = useState<HorizonMetric[]>([]);
  const [summary, setSummary] = useState<OperationalSummary | null>(null);
  const [portCalls, setPortCalls] = useState<PortCallItem[]>([]);
  const [weather, setWeather] = useState<WeatherPoint[]>([]);
  const [dataHealth, setDataHealth] = useState<DataHealth | null>(null);
  const [modelGovernance, setModelGovernance] = useState<ModelGovernance | null>(null);
  const [performanceHistory, setPerformanceHistory] = useState<PerformancePoint[]>([]);
  const [errorHeatmap, setErrorHeatmap] = useState<ErrorHeatmapCell[]>([]);
  const [asOf, setAsOf] = useState<string | null>(null);
  const [horizon, setHorizon] = useState(24);
  const [speed, setSpeed] = useState(6);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const bootstrap = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [nextRange, nextSourceStatus, nextModelGovernance] = await Promise.all([
        replayApi.getRange(),
        replayApi.getSourceStatus(),
        replayApi.getModelGovernance(),
      ]);
      setRange(nextRange);
      setSourceStatus(nextSourceStatus);
      setModelGovernance(nextModelGovernance);
      setAsOf((current) => current ?? nextRange.last_as_of_time);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : 'API indisponible');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  useEffect(() => {
    if (!asOf) return;
    const controller = new AbortController();
    let active = true;
    setRefreshing(true);
    setError(null);
    Promise.all([
      replayApi.getSnapshot(asOf, controller.signal),
      replayApi.getTimeline(asOf, horizon, 168, controller.signal),
      replayApi.getMetrics(asOf, 30, controller.signal),
      replayApi.getOperationalSummary(asOf, controller.signal),
      replayApi.getPortCalls(asOf, controller.signal),
      replayApi.getWeather(asOf, 168, controller.signal),
      replayApi.getDataHealth(asOf, controller.signal),
      replayApi.getPerformanceHistory(asOf, horizon, 60, controller.signal),
      replayApi.getErrorHeatmap(asOf, horizon, 120, controller.signal),
    ])
      .then(
        ([
          nextSnapshot,
          nextTimeline,
          nextMetrics,
          nextSummary,
          nextPortCalls,
          nextWeather,
          nextDataHealth,
          nextPerformanceHistory,
          nextErrorHeatmap,
        ]) => {
          if (!active) return;
          setSnapshot(nextSnapshot);
          setTimeline(nextTimeline);
          setMetrics(nextMetrics);
          setSummary(nextSummary);
          setPortCalls(nextPortCalls);
          setWeather(nextWeather);
          setDataHealth(nextDataHealth);
          setPerformanceHistory(nextPerformanceHistory);
          setErrorHeatmap(nextErrorHeatmap);
        },
      )
      .catch((requestError: unknown) => {
        if (!active) return;
        if (requestError instanceof DOMException && requestError.name === 'AbortError') return;
        setError(requestError instanceof Error ? requestError.message : 'API indisponible');
      })
      .finally(() => {
        if (active) setRefreshing(false);
      });
    return () => {
      active = false;
      controller.abort();
    };
  }, [asOf, horizon]);

  useEffect(() => {
    if (!playing || !range) return;
    const timer = window.setInterval(() => {
      setAsOf((currentValue) => {
        if (!currentValue) return range.first_as_of_time;
        const next = new Date(currentValue).getTime() + speed * HOUR_MS;
        const maximum = new Date(range.last_as_of_time).getTime();
        if (next >= maximum) {
          setPlaying(false);
          return range.last_as_of_time;
        }
        return new Date(next).toISOString();
      });
    }, 1800);
    return () => window.clearInterval(timer);
  }, [playing, range, speed]);

  const selectedForecast = useMemo(
    () => snapshot?.forecasts.find((item) => item.horizon_h === horizon) ?? null,
    [horizon, snapshot],
  );
  const selectedMetric = useMemo(
    () => metrics.find((item) => item.horizon_h === horizon) ?? null,
    [horizon, metrics],
  );
  const alerts = useMemo(
    () => buildAlerts(sourceStatus, summary, selectedForecast, selectedMetric),
    [selectedForecast, selectedMetric, sourceStatus, summary],
  );

  const seek = useCallback((value: string) => {
    setAsOf(value);
    setPlaying(false);
  }, []);
  const resetReplay = useCallback(() => {
    if (range) setAsOf(range.first_as_of_time);
    setPlaying(false);
  }, [range]);
  const jumpToLatest = useCallback(() => {
    if (range) setAsOf(range.last_as_of_time);
    setPlaying(false);
  }, [range]);

  const value = useMemo<ReplayContextValue>(
    () => ({
      range,
      sourceStatus,
      snapshot,
      timeline,
      metrics,
      summary,
      portCalls,
      weather,
      dataHealth,
      modelGovernance,
      performanceHistory,
      errorHeatmap,
      alerts,
      asOf,
      horizon,
      speed,
      playing,
      loading,
      refreshing,
      error,
      selectedForecast,
      selectedMetric,
      setHorizon,
      setSpeed,
      setPlaying,
      seek,
      resetReplay,
      jumpToLatest,
      refresh: bootstrap,
    }),
    [
      alerts,
      asOf,
      bootstrap,
      dataHealth,
      errorHeatmap,
      error,
      horizon,
      jumpToLatest,
      loading,
      metrics,
      modelGovernance,
      playing,
      portCalls,
      performanceHistory,
      range,
      refreshing,
      resetReplay,
      seek,
      selectedForecast,
      selectedMetric,
      snapshot,
      sourceStatus,
      speed,
      summary,
      timeline,
      weather,
    ],
  );

  return <ReplayContext.Provider value={value}>{children}</ReplayContext.Provider>;
};

export const useReplay = () => {
  const context = useContext(ReplayContext);
  if (!context) throw new Error('useReplay must be used inside ReplayProvider');
  return context;
};
