import type {
  MetoceanAugmentationStatus,
  MetoceanDashboardData,
  MetoceanForecastPoint,
  MetoceanMetric,
  MetoceanStatus,
  MetoceanTaskSelection,
  MetoceanValidationStatus,
  MetoceanVesselImpact,
} from 'types/metocean';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(
  /\/$/,
  '',
);

const B62_ROOT = `${API_BASE_URL}/api/v1/maritime/metocean-cascade`;
const B62A_ROOT = `${API_BASE_URL}/api/v1/maritime/metocean-augmentation`;
const B62B_ROOT = `${API_BASE_URL}/api/v1/maritime/metocean-vintage-validation`;

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, {
    headers: { Accept: 'application/json' },
    signal,
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `API request failed with status ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function getForecast(signal?: AbortSignal) {
  const tracks: MetoceanForecastPoint['track'][] = [
    'ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT',
    'RESEARCH_REANALYSIS_SHADOW',
  ];
  let lastError: unknown;
  for (const track of tracks) {
    try {
      const forecast = await getJson<MetoceanForecastPoint[]>(
        `${B62_ROOT}/forecast?track=${track}&limit=2000`,
        signal,
      );
      return { forecast, track };
    } catch (error) {
      lastError = error;
    }
  }
  throw lastError instanceof Error ? lastError : new Error('Prévision B62 indisponible');
}

const settledValue = <T>(result: PromiseSettledResult<T>, unavailable: string[], label: string) => {
  if (result.status === 'fulfilled') return result.value;
  unavailable.push(label);
  return null;
};

export const metoceanApi = {
  async getDashboard(signal?: AbortSignal): Promise<MetoceanDashboardData> {
    const results = await Promise.allSettled([
      getJson<MetoceanStatus>(`${B62_ROOT}/status`, signal),
      getForecast(signal),
      getJson<MetoceanVesselImpact[]>(`${B62_ROOT}/vessel-impact?limit=500`, signal),
      getJson<MetoceanAugmentationStatus>(`${B62A_ROOT}/status`, signal),
      getJson<MetoceanTaskSelection[]>(`${B62A_ROOT}/selection`, signal),
      getJson<MetoceanValidationStatus>(`${B62B_ROOT}/status`, signal),
      getJson<MetoceanMetric[]>(`${B62B_ROOT}/metrics`, signal),
    ] as const);

    const unavailable: string[] = [];
    const status = settledValue(results[0], unavailable, 'statut B62');
    const forecastResult = settledValue(results[1], unavailable, 'prévisions B62');
    const impacts = settledValue(results[2], unavailable, 'impacts navires');
    const augmentation = settledValue(results[3], unavailable, 'statut B62A');
    const selections = settledValue(results[4], unavailable, 'sélection B62A');
    const validation = settledValue(results[5], unavailable, 'validation B62B');
    const metrics = settledValue(results[6], unavailable, 'métriques B62B');

    return {
      status,
      forecast: forecastResult?.forecast ?? [],
      impacts: impacts ?? [],
      augmentation,
      selections: selections ?? [],
      validation,
      metrics: metrics ?? [],
      forecastTrack: forecastResult?.track ?? null,
      unavailable,
    };
  },
};
