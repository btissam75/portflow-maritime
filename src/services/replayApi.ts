import type {
  DataHealth,
  ErrorHeatmapCell,
  ForecastPoint,
  HorizonMetric,
  ModelGovernance,
  OperationalSummary,
  PerformancePoint,
  PortCallItem,
  ReplayBundle,
  ReplayRange,
  ReplaySnapshot,
  SourceStatus,
  WeatherPoint,
} from 'types/replay';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(
  /\/$/,
  '',
);
const REPLAY_ROOT = `${API_BASE_URL}/api/v1/maritime/replay`;
const OPERATIONS_ROOT = `${API_BASE_URL}/api/v1/maritime/operations`;

async function getJson<T>(
  path: string,
  signal?: AbortSignal,
  root = REPLAY_ROOT,
): Promise<T> {
  const response = await fetch(`${root}${path}`, {
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `API request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const replayApi = {
  getRange: (signal?: AbortSignal) => getJson<ReplayRange>('/range', signal),

  getSourceStatus: (signal?: AbortSignal) =>
    getJson<SourceStatus>('/source-status', signal),

  getSnapshot: (asOf: string, signal?: AbortSignal) =>
    getJson<ReplaySnapshot>(`/snapshot?as_of=${encodeURIComponent(asOf)}`, signal),

  getTimeline: (
    asOf: string,
    horizonH: number,
    hours = 168,
    signal?: AbortSignal,
  ) =>
    getJson<ForecastPoint[]>(
      `/timeline?end=${encodeURIComponent(asOf)}&horizon_h=${horizonH}&hours=${hours}`,
      signal,
    ),

  getMetrics: (asOf: string, days = 30, signal?: AbortSignal) =>
    getJson<HorizonMetric[]>(
      `/metrics?end=${encodeURIComponent(asOf)}&days=${days}`,
      signal,
    ),

  getModelGovernance: (signal?: AbortSignal) =>
    getJson<ModelGovernance>('/model-governance', signal),

  getPerformanceHistory: (
    asOf: string,
    horizonH: number,
    days = 60,
    signal?: AbortSignal,
  ) =>
    getJson<PerformancePoint[]>(
      `/performance-history?end=${encodeURIComponent(asOf)}&horizon_h=${horizonH}&days=${days}`,
      signal,
    ),

  getErrorHeatmap: (
    asOf: string,
    horizonH: number,
    days = 120,
    signal?: AbortSignal,
  ) =>
    getJson<ErrorHeatmapCell[]>(
      `/error-heatmap?end=${encodeURIComponent(asOf)}&horizon_h=${horizonH}&days=${days}`,
      signal,
    ),

  getPortCalls: (asOf: string, signal?: AbortSignal) =>
    getJson<PortCallItem[]>(
      `/port-calls?as_of=${encodeURIComponent(asOf)}&before_h=24&after_h=72&limit=400`,
      signal,
      OPERATIONS_ROOT,
    ),

  getWeather: (asOf: string, hours = 168, signal?: AbortSignal) =>
    getJson<WeatherPoint[]>(
      `/weather?as_of=${encodeURIComponent(asOf)}&hours=${hours}`,
      signal,
      OPERATIONS_ROOT,
    ),

  getOperationalSummary: (asOf: string, signal?: AbortSignal) =>
    getJson<OperationalSummary>(
      `/summary?as_of=${encodeURIComponent(asOf)}`,
      signal,
      OPERATIONS_ROOT,
    ),

  getDataHealth: (asOf: string, signal?: AbortSignal) =>
    getJson<DataHealth>(
      `/data-health?as_of=${encodeURIComponent(asOf)}`,
      signal,
      OPERATIONS_ROOT,
    ),

  async getBundle(
    asOf: string,
    horizonH: number,
    signal?: AbortSignal,
  ): Promise<ReplayBundle> {
    const [range, sourceStatus, snapshot, timeline, metrics] = await Promise.all([
      this.getRange(signal),
      this.getSourceStatus(signal),
      this.getSnapshot(asOf, signal),
      this.getTimeline(asOf, horizonH, 168, signal),
      this.getMetrics(asOf, 30, signal),
    ]);
    return { range, sourceStatus, snapshot, timeline, metrics };
  },
};
