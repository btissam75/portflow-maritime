import { getJson, requestJson } from 'services/http';
import type {
  ControlTowerSnapshot,
  SimulationPayload,
  SimulationResult,
  TowerDecision,
  TowerUnitDetail,
} from 'types/controlTower';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const ROOT = `${API_BASE_URL}/api/v1/control-tower`;

export const controlTowerApi = {
  getSnapshot(signal?: AbortSignal) {
    return getJson<ControlTowerSnapshot>(`${ROOT}/snapshot`, {
      signal,
      timeoutMs: 15_000,
      sourceLabel: 'La Control Tower',
    });
  },

  getUnit(unitId: string, signal?: AbortSignal) {
    return getJson<TowerUnitDetail>(`${ROOT}/units/${encodeURIComponent(unitId)}`, {
      signal,
      sourceLabel: 'La fiche unité',
    });
  },

  createDecision(payload: {
    title: string;
    description?: string;
    assignee?: string;
    alert_id?: string;
    unit_ids?: string[];
  }) {
    return requestJson<TowerDecision>(`${ROOT}/decisions`, {
      method: 'POST',
      body: payload,
      sourceLabel: 'La création de décision',
    });
  },

  updateDecision(decisionId: string, payload: { status?: string; comment?: string }) {
    return requestJson<TowerDecision>(`${ROOT}/decisions/${encodeURIComponent(decisionId)}`, {
      method: 'PATCH',
      body: payload,
      sourceLabel: 'La mise à jour de décision',
    });
  },

  simulate(payload: SimulationPayload) {
    return requestJson<SimulationResult>(`${ROOT}/simulations`, {
      method: 'POST',
      body: payload,
      timeoutMs: 20_000,
      sourceLabel: 'Le moteur de simulation',
    });
  },

  getShiftReport() {
    return getJson<Record<string, unknown>>(`${ROOT}/reports/shift`, {
      sourceLabel: 'Le rapport de quart',
    });
  },
};
