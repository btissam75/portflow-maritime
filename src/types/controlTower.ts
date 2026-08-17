export type ControlTowerView =
  | 'overview'
  | 'units'
  | 'process'
  | 'forecast'
  | 'alerts'
  | 'decisions'
  | 'vessels'
  | 'simulation'
  | 'quality'
  | 'audit'
  | 'reports';

export interface TowerMetrics {
  active_units: number;
  at_risk_units: number;
  median_eta_h: number;
  ge12_units: number;
  ge24_units: number;
  ge36_units: number;
  open_alerts: number;
  pending_decisions: number;
}

export interface StageForecast {
  h1: number;
  h3: number;
  h6: number;
  h12: number;
  h24: number;
}

export interface TowerStage {
  code: string;
  label: string;
  order: number;
  units: number;
  capacity: number;
  occupancy_pct: number;
  dwell_median_h: number;
  dwell_p90_h: number;
  blocked: number;
  trend: 'HAUSSE' | 'BAISSE' | 'STABLE';
  forecast: StageForecast;
}

export interface TowerUnit {
  unit_id: string;
  stage: string;
  stage_label: string;
  dwell_h: number;
  eta_p10_h: number;
  eta_p50_h: number;
  eta_p80_h: number;
  eta_p90_h: number;
  ge12: number;
  ge24: number;
  ge36: number;
  route: 'DIRECT' | 'PV' | 'REVUE';
  confidence: number;
  cause: string;
  urgency: number;
  impact: number;
  priority: number;
  tier: 'CRITIQUE' | 'VIGILANCE' | 'NORMAL';
  assignee: string | null;
  status: string;
  last_event_at: string;
  location_quality: string;
  location_age_minutes: number;
  location: { zone: string; x: number; y: number; precision: 'EXACTE' | 'ZONE' };
}

export interface TowerForecastPoint {
  horizon_h: number;
  valid_at: string;
  arrivals: number;
  departures: number;
  backlog_p10: number;
  backlog_p50: number;
  backlog_p90: number;
  normal_capacity: number;
  reinforced_capacity: number;
}

export interface TowerAlert {
  alert_id: string;
  severity: 'CRITIQUE' | 'VIGILANCE' | 'INFORMATION';
  title: string;
  message: string;
  probability: number;
  impact: string;
  cause: string;
  recommendation: string;
  deadline_at: string;
  confidence: number;
  unit_ids: string[];
  status: string;
}

export interface TowerDecision {
  decision_id: string;
  title: string;
  description: string;
  status: string;
  assignee: string;
  created_at: string;
  updated_at: string;
  due_at: string;
  alert_id: string | null;
  unit_ids: string[];
  comments: Array<{ at: string; author: string; text: string }>;
  outcome: string | null;
  expected_effect: string;
}

export interface TowerVessel {
  vessel_id: string;
  name: string;
  imo: string;
  mmsi: string;
  longitude: number;
  latitude: number;
  heading: number;
  speed_kn: number;
  status: string;
  announced_eta: string;
  predicted_eta: string;
  eta_delta_minutes: number;
  distance_nm: number;
  terminal: string;
  berth_window: string;
  ais_age_minutes: number;
  ais_quality: string;
  associated_units: number;
  units_ready: number;
  congestion_risk: number;
}

export interface TowerSource {
  source: string;
  status: string;
  age_minutes: number;
  completeness_pct: number;
  detail: string;
}

export interface TowerAuditEvent {
  event_id: string;
  at: string;
  actor: string;
  action: string;
  object: string;
  immutable: boolean;
}

export interface TowerRecommendation {
  recommendation_id: string;
  title: string;
  expected_gain_h: number;
  beneficiary_units: number;
  confidence: number;
  secondary_risk: string;
  evidence: string[];
}

export interface ControlTowerSnapshot {
  contract_version: string;
  mode: 'EXERCISE' | 'LIVE';
  serving_status: string;
  generated_at: string;
  refresh_after_seconds: number;
  metrics: TowerMetrics;
  stages: TowerStage[];
  units: TowerUnit[];
  forecast: TowerForecastPoint[];
  alerts: TowerAlert[];
  decisions: TowerDecision[];
  vessels: TowerVessel[];
  sources: TowerSource[];
  audit: TowerAuditEvent[];
  recommendations: TowerRecommendation[];
  permissions: string[];
}

export interface TowerUnitDetail extends TowerUnit {
  timeline: Array<{
    at: string;
    stage: string;
    label: string;
    duration_h: number;
    reliability: string;
  }>;
  eta_history: Array<{ issued_at: string; p50_h: number; p90_h: number }>;
  explanation: string;
  previous_alerts: TowerAlert[];
  previous_decisions: TowerDecision[];
  prediction: {
    calculated_at: string;
    freshness_minutes: number;
    model: string;
    fallback: boolean;
    experimental: boolean;
  };
}

export interface SimulationPayload {
  stage: string;
  capacity_boost: number;
  duration_h: number;
  arrival_change_pct: number;
  route_policy: 'CURRENT' | 'DIRECT' | 'PV';
}

export interface SimulationResult {
  simulation_id: string;
  created_at: string;
  inputs: SimulationPayload;
  before: Record<'max_backlog' | 'ge24_units' | 'mean_eta_p90_h' | 'recovery_h', number>;
  after: Record<'max_backlog' | 'ge24_units' | 'mean_eta_p90_h' | 'recovery_h', number>;
  confidence: number;
  status: string;
  recommendation: string;
  automatic_action_allowed: boolean;
}
