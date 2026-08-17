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
import { getJson } from 'services/http';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');

const B62_ROOT = `${API_BASE_URL}/api/v1/maritime/metocean-cascade`;
const B62A_ROOT = `${API_BASE_URL}/api/v1/maritime/metocean-augmentation`;
const B62B_ROOT = `${API_BASE_URL}/api/v1/maritime/metocean-vintage-validation`;

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
        { signal, sourceLabel: 'La prévision météo-marine gouvernée' },
      );
      return { forecast, track };
    } catch (error) {
      if (signal?.aborted) throw new DOMException('Requête annulée', 'AbortError');
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
      getJson<MetoceanStatus>(`${B62_ROOT}/status`, {
        signal,
        sourceLabel: 'Le statut du modèle météo-marin',
      }),
      getForecast(signal),
      getJson<MetoceanVesselImpact[]>(`${B62_ROOT}/vessel-impact?limit=500`, {
        signal,
        sourceLabel: 'Les impacts sur les navires',
      }),
      getJson<MetoceanAugmentationStatus>(`${B62A_ROOT}/status`, {
        signal,
        sourceLabel: 'Le statut du challenger météo-marin',
      }),
      getJson<MetoceanTaskSelection[]>(`${B62A_ROOT}/selection`, {
        signal,
        sourceLabel: 'La sélection des modèles météo-marins',
      }),
      getJson<MetoceanValidationStatus>(`${B62B_ROOT}/status`, {
        signal,
        sourceLabel: 'La validation temporelle météo-marine',
      }),
      getJson<MetoceanMetric[]>(`${B62B_ROOT}/metrics`, {
        signal,
        sourceLabel: 'Les métriques météo-marines',
      }),
    ] as const);

    if (signal?.aborted) throw new DOMException('Requête annulée', 'AbortError');

    const unavailable: string[] = [];
    const status = settledValue(results[0], unavailable, 'statut de prévision scientifique');
    const forecastResult = settledValue(results[1], unavailable, 'prévision scientifique');
    const impacts = settledValue(results[2], unavailable, 'impacts navires');
    const augmentation = settledValue(results[3], unavailable, 'statut du challenger');
    const selections = settledValue(results[4], unavailable, 'sélection du challenger');
    const validation = settledValue(results[5], unavailable, 'validation fresh-forward');
    const metrics = settledValue(results[6], unavailable, 'métriques de validation');

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
