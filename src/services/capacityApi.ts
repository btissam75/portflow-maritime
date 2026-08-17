import type {
  CapacityDashboardData,
  CapacityDecision,
  CapacityRankingStatus,
  CapacitySnapshot,
} from 'types/capacity';
import { getJson } from 'services/http';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const ROOT = `${API_BASE_URL}/api/v1/maritime/capacity-ranking`;
const DASHBOARD_CACHE_PREFIX = 'portflow.capacity.dashboard.v2.';
const TIMELINE_CACHE_PREFIX = 'portflow.capacity.timeline.v1.';
const DASHBOARD_CACHE_TTL_MS = 15 * 60_000;
const TIMELINE_CACHE_TTL_MS = 30 * 60_000;
export type CapacityEvaluationRole = 'VALID_SELECT' | 'VALID_CALIBRATE' | 'TEST_DIAGNOSTIC_ONLY';

interface CachedValue<T> {
  cachedAt: string;
  value: T;
}

export interface CapacityCacheEntry<T> {
  value: T;
  cachedAt: string;
  ageMs: number;
  stale: boolean;
}

function readSessionCache<T>(key: string, ttlMs: number): CapacityCacheEntry<T> | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.sessionStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CachedValue<T>;
    const cachedAtMs = new Date(parsed.cachedAt).getTime();
    if (!parsed.value || !Number.isFinite(cachedAtMs)) return null;
    const ageMs = Math.max(0, Date.now() - cachedAtMs);
    return { value: parsed.value, cachedAt: parsed.cachedAt, ageMs, stale: ageMs > ttlMs };
  } catch {
    return null;
  }
}

function writeSessionCache<T>(key: string, value: T): void {
  if (typeof window === 'undefined') return;
  try {
    window.sessionStorage.setItem(
      key,
      JSON.stringify({ cachedAt: new Date().toISOString(), value }),
    );
  } catch {
    // The API remains usable when private browsing disables session storage.
  }
}

export const capacityApi = {
  getCachedDashboardEntry(
    evaluationRole: CapacityEvaluationRole = 'VALID_SELECT',
  ): CapacityCacheEntry<CapacityDashboardData> | null {
    return readSessionCache<CapacityDashboardData>(
      `${DASHBOARD_CACHE_PREFIX}${evaluationRole}`,
      DASHBOARD_CACHE_TTL_MS,
    );
  },

  getCachedDashboard(
    evaluationRole: CapacityEvaluationRole = 'VALID_SELECT',
  ): CapacityDashboardData | null {
    return (
      readSessionCache<CapacityDashboardData>(
        `${DASHBOARD_CACHE_PREFIX}${evaluationRole}`,
        DASHBOARD_CACHE_TTL_MS,
      )?.value ?? null
    );
  },

  getCachedTimelineEntry(portCallId: string): CapacityCacheEntry<CapacityDecision[]> | null {
    return readSessionCache<CapacityDecision[]>(
      `${TIMELINE_CACHE_PREFIX}${portCallId}`,
      TIMELINE_CACHE_TTL_MS,
    );
  },

  getCachedTimeline(portCallId: string): CapacityDecision[] {
    return (
      readSessionCache<CapacityDecision[]>(
        `${TIMELINE_CACHE_PREFIX}${portCallId}`,
        TIMELINE_CACHE_TTL_MS,
      )?.value ?? []
    );
  },

  async getDashboard(
    signal?: AbortSignal,
    evaluationRole: CapacityEvaluationRole = 'VALID_SELECT',
  ): Promise<CapacityDashboardData> {
    const [statusResult, snapshotResult] = await Promise.allSettled([
      getJson<CapacityRankingStatus>(`${ROOT}/status`, {
        signal,
        sourceLabel: 'Le statut de vigilance',
      }),
      getJson<CapacitySnapshot>(
        `${ROOT}/snapshot?evaluation_role=${evaluationRole}&selected_only=false&limit=250`,
        { signal, sourceLabel: 'La liste des escales' },
      ),
    ]);
    if (signal?.aborted) throw new DOMException('Requête annulée', 'AbortError');
    const unavailable: string[] = [];
    const status = statusResult.status === 'fulfilled' ? statusResult.value : null;
    const snapshot = snapshotResult.status === 'fulfilled' ? snapshotResult.value : null;
    if (!status) unavailable.push('état de la vigilance');
    if (!snapshot) unavailable.push('liste des escales');
    if (!status && !snapshot) throw new Error('La vigilance des escales est indisponible.');
    const dashboard = { status, snapshot, unavailable, fetchedAt: new Date().toISOString() };
    writeSessionCache(`${DASHBOARD_CACHE_PREFIX}${evaluationRole}`, dashboard);
    return dashboard;
  },

  getSnapshot(
    at: string,
    evaluationRole: CapacityEvaluationRole,
    signal?: AbortSignal,
  ): Promise<CapacitySnapshot> {
    return getJson<CapacitySnapshot>(
      `${ROOT}/snapshot?evaluation_role=${evaluationRole}&selected_only=false&limit=250&at=${encodeURIComponent(at)}`,
      { signal, sourceLabel: 'Le snapshot historique' },
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
        { signal, sourceLabel: 'Le replay historique' },
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
      { signal, sourceLabel: 'La trajectoire de l’escale' },
    );
    if (!Array.isArray(timeline)) throw new Error('La trajectoire reçue est invalide.');
    writeSessionCache(`${TIMELINE_CACHE_PREFIX}${portCallId}`, timeline);
    return timeline;
  },
};
