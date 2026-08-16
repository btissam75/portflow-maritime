import type {
  CapacityDashboardData,
  CapacityDecision,
  CapacityRankingStatus,
  CapacitySnapshot,
} from 'types/capacity';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(
  /\/$/,
  '',
);
const ROOT = `${API_BASE_URL}/api/v1/maritime/capacity-ranking`;
const DASHBOARD_CACHE_PREFIX = 'portflow.capacity.dashboard.v2.';
const TIMELINE_CACHE_PREFIX = 'portflow.capacity.timeline.v1.';
export type CapacityEvaluationRole = 'VALID_SELECT' | 'VALID_CALIBRATE' | 'TEST_DIAGNOSTIC_ONLY';

interface CachedValue<T> {
  cachedAt: string;
  value: T;
}

function readSessionCache<T>(key: string): T | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.sessionStorage.getItem(key);
    if (!raw) return null;
    return (JSON.parse(raw) as CachedValue<T>).value;
  } catch {
    return null;
  }
}

function writeSessionCache<T>(key: string, value: T): void {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.setItem(key, JSON.stringify({ cachedAt: new Date().toISOString(), value }));
  } catch {
    // The API remains usable when private browsing disables session storage.
  }
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { headers: { Accept: 'application/json' }, signal });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `API request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const capacityApi = {
  getCachedDashboard(evaluationRole: CapacityEvaluationRole = 'VALID_SELECT'): CapacityDashboardData | null {
    return readSessionCache<CapacityDashboardData>(`${DASHBOARD_CACHE_PREFIX}${evaluationRole}`);
  },

  getCachedTimeline(portCallId: string): CapacityDecision[] {
    return readSessionCache<CapacityDecision[]>(`${TIMELINE_CACHE_PREFIX}${portCallId}`) ?? [];
  },

  async getDashboard(signal?: AbortSignal, evaluationRole: CapacityEvaluationRole = 'VALID_SELECT'): Promise<CapacityDashboardData> {
    const [statusResult, snapshotResult] = await Promise.allSettled([
      getJson<CapacityRankingStatus>(`${ROOT}/status`, signal),
      getJson<CapacitySnapshot>(
        `${ROOT}/snapshot?evaluation_role=${evaluationRole}&selected_only=false&limit=250`,
        signal,
      ),
    ]);
    const unavailable: string[] = [];
    const status = statusResult.status === 'fulfilled' ? statusResult.value : null;
    const snapshot = snapshotResult.status === 'fulfilled' ? snapshotResult.value : null;
    if (!status) unavailable.push('état de la vigilance');
    if (!snapshot) unavailable.push('liste des escales');
    if (!status && !snapshot) throw new Error('La vigilance des escales est indisponible.');
    const dashboard = { status, snapshot, unavailable };
    writeSessionCache(`${DASHBOARD_CACHE_PREFIX}${evaluationRole}`, dashboard);
    return dashboard;
  },

  getSnapshot(at: string, evaluationRole: CapacityEvaluationRole, signal?: AbortSignal): Promise<CapacitySnapshot> {
    return getJson<CapacitySnapshot>(
      `${ROOT}/snapshot?evaluation_role=${evaluationRole}&selected_only=false&limit=250&at=${encodeURIComponent(at)}`,
      signal,
    );
  },

  async getReplaySnapshots(
    anchorAt: string,
    evaluationRole: CapacityEvaluationRole,
    signal?: AbortSignal,
    frameCount = 6,
  ): Promise<CapacitySnapshot[]> {
    const anchor = new Date(anchorAt).getTime();
    const requests = Array.from({ length: frameCount }, (_, index) => {
      const at = new Date(anchor - index * 6 * 60 * 60 * 1000).toISOString();
      return getJson<CapacitySnapshot>(
        `${ROOT}/snapshot?evaluation_role=${evaluationRole}&selected_only=false&limit=250&at=${encodeURIComponent(at)}`,
        signal,
      );
    });
    const results = await Promise.allSettled(requests);
    const unique = new Map<string, CapacitySnapshot>();
    results.forEach((result) => {
      if (result.status === 'fulfilled' && result.value.decisions.length) {
        unique.set(result.value.resolved_at, result.value);
      }
    });
    return [...unique.values()].sort(
      (left, right) => new Date(left.resolved_at).getTime() - new Date(right.resolved_at).getTime(),
    );
  },

  async getTimeline(portCallId: string, signal?: AbortSignal): Promise<CapacityDecision[]> {
    const timeline = await getJson<CapacityDecision[]>(
      `${ROOT}/port-calls/${encodeURIComponent(portCallId)}/timeline?limit=250`,
      signal,
    );
    writeSessionCache(`${TIMELINE_CACHE_PREFIX}${portCallId}`, timeline);
    return timeline;
  },
};
