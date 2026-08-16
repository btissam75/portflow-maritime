export interface ReplayRange {
  first_as_of_time: string;
  last_as_of_time: string;
  timestamps: number;
  serving_rows: number;
  horizons_h: number[];
  forecast_version: string;
  source_mode: 'HISTORICAL_REPLAY';
}

export interface SourceStatus {
  audit_status: string;
  decision: string;
  source_status: string;
  source_break: string | null;
  latest_eligible: string | null;
  historical_replay_allowed: boolean;
  live_serving_allowed: boolean;
  training_executed: boolean;
  selection_used_test: boolean;
  finished_at: string | null;
}

export interface ForecastPoint {
  as_of_time: string;
  target_time: string;
  horizon_h: number;
  selected_model: string;
  actual_arrivals: number | null;
  point_prediction: number;
  p10: number;
  p50: number;
  p90: number;
  absolute_error: number | null;
  source_mode: 'HISTORICAL_REPLAY';
}

export interface ReplaySnapshot {
  requested_as_of: string | null;
  resolved_as_of: string;
  forecasts: ForecastPoint[];
  source_mode: 'HISTORICAL_REPLAY';
  live: false;
}

export interface HorizonMetric {
  horizon_h: number;
  observations: number;
  mae: number;
  rmse: number;
  wape_pct: number;
  bias: number;
  coverage_p10_p90: number;
  mean_interval_width: number;
}

export type ModelGateStatus = 'PASS' | 'WATCH' | 'FAIL';

export interface HorizonGovernance {
  horizon_h: number;
  selected_policy: string;
  window_days: number | null;
  gamma: number | null;
  coverage_30d: number;
  mae_30d: number;
  interval_width_30d: number;
  gate_status: ModelGateStatus;
}

export interface ModelGovernance {
  model_version: string;
  point_source: string;
  mode: 'HISTORICAL_REPLAY';
  calibration_decision: string;
  shadow_decision: string;
  replay_allowed: boolean;
  live_allowed: boolean;
  integrity_passed: boolean;
  point_fidelity_passed: boolean;
  coherence_passed: boolean;
  recent30_gates_passed: boolean;
  formal_promotion_allowed: boolean;
  promotion_blocker: string | null;
  prospective_forecasts: number;
  paired_forecasts: number;
  last_audit_at: string | null;
  horizons: HorizonGovernance[];
}

export interface PerformancePoint {
  period_start: string;
  horizon_h: number;
  observations: number;
  mae: number;
  bias: number;
  coverage_p10_p90: number;
  mean_interval_width: number;
}

export interface ErrorHeatmapCell {
  day_of_week: number;
  hour_of_day: number;
  observations: number;
  mae: number;
  bias: number;
  coverage_p10_p90: number;
}

export interface ReplayBundle {
  range: ReplayRange;
  sourceStatus: SourceStatus;
  snapshot: ReplaySnapshot;
  timeline: ForecastPoint[];
  metrics: HorizonMetric[];
}

export type PortCallStatus = 'EXPECTED' | 'OVERDUE' | 'ARRIVED' | 'BERTHED' | 'DEPARTED';

export interface PortCallItem {
  port_call_id: string;
  port_code: string;
  terminal_code: string | null;
  imo: number | null;
  vessel_name: string;
  voyage_id: string | null;
  planned_eta: string;
  actual_ata: string | null;
  planned_etd: string | null;
  actual_atd: string | null;
  cargo_type: string | null;
  vessel_type: string | null;
  status: PortCallStatus;
  arrival_delay_h: number | null;
}

export interface WeatherPoint {
  observed_at: string;
  latitude: number;
  longitude: number;
  wave_height_m: number | null;
  wave_period_s: number | null;
  wave_direction_deg: number | null;
  wind_speed_ms: number | null;
  wind_direction_deg: number | null;
  surface_current_ms: number | null;
  visibility_m: number | null;
  pressure_hpa: number | null;
  quality_flag: number;
}

export interface OperationalSummary {
  resolved_as_of: string;
  expected_next_24h: number;
  expected_next_72h: number;
  arrived_previous_24h: number;
  overdue_calls: number;
  vessels_in_port: number;
  active_call_window: number;
  wave_height_m: number | null;
  wave_period_s: number | null;
  wind_speed_ms: number | null;
  weather_observed_at: string | null;
  ais_positions_72h: number;
  ais_vessels_72h: number;
  ais_last_observed_at: string | null;
}

export type DataHealthStatus = 'READY' | 'STALE' | 'MISSING';

export interface DataHealthSource {
  source: string;
  label: string;
  status: DataHealthStatus;
  rows: number;
  latest_event_time: string | null;
  age_hours: number | null;
  detail: string;
}

export interface DataHealth {
  resolved_as_of: string;
  sources: DataHealthSource[];
}
