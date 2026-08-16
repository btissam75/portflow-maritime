export interface MetoceanStatus {
  audit_status: string;
  decision: string;
  model_version: string;
  rows: number;
  selected_chronos_tasks: number;
  issue_time_ready: boolean;
  issue_time_span_days: number;
  critical_gates_passed: boolean;
  test_used_for_selection: boolean;
  production_promotion_allowed: boolean;
  automatic_action_allowed: boolean;
  serving_forecast_rows: number;
  serving_impact_rows: number;
  next_block: string | null;
  finished_at: string | null;
}

export interface MetoceanForecastPoint {
  track: 'RESEARCH_REANALYSIS_SHADOW' | 'ISSUE_TIME_PROVIDER_OPERATIONAL_INPUT';
  issue_at: string;
  valid_at: string;
  horizon_h: number;
  variable: string;
  p10: number | null;
  p50: number;
  p90: number | null;
  source_model: string;
  uncertainty_status: string;
  operationally_available: boolean;
  production_claim_allowed: boolean;
}

export interface MetoceanVesselImpact {
  port_call_id: string;
  vessel_name: string | null;
  port_code: string | null;
  terminal_code: string | null;
  vessel_type: string | null;
  cargo_group: string | null;
  source_decision_at: string;
  forecast_issue_at: string;
  valid_at: string;
  horizon_h: number;
  base_temporal_risk: number;
  metocean_severity: number;
  vessel_exposure: number;
  combined_priority_score: number;
  metocean_tier: string;
  priority_tier: string;
  forecast_track: string;
  score_semantics: string;
  automatic_action_allowed: boolean;
  production_claim_allowed: boolean;
}

export interface MetoceanAugmentationStatus {
  audit_status: string;
  decision: string;
  model_version: string;
  synthetic_rows: number;
  synthetic_weight: number;
  accepted_challenger_tasks: number;
  challenger_tasks: number;
  weekly_real_origins: number;
  frozen_test_origins: number;
  stress_scenarios: number;
  critical_gates_passed: boolean;
  valid_modified: boolean;
  test_modified: boolean;
  test_used_for_selection: boolean;
  production_promotion_allowed: boolean;
  automatic_action_allowed: boolean;
  next_block: string | null;
  finished_at: string | null;
}

export interface MetoceanTaskSelection {
  variable: string;
  horizon_h: number;
  b62_model: string;
  selected_model: string;
  challenger_accepted: boolean;
  valid_b62_mae: number | null;
  valid_challenger_mae: number | null;
  valid_challenger_gain_pct: number | null;
  valid_challenger_coverage: number | null;
  test_model: string | null;
  test_mae: number | null;
  test_bias: number | null;
  test_coverage: number | null;
  selection_role: string;
  test_role: string;
  production_promotion_allowed: boolean;
}

export interface MetoceanValidationStatus {
  audit_status: string;
  decision: string;
  model_version: string;
  rows: number;
  archive_origins: number;
  valid_origins: number;
  test_origins: number;
  fresh_origins: number;
  fresh_span_days: number;
  selected_model: string | null;
  reference_model: string | null;
  valid_accepted: boolean;
  archive_confirmed: boolean;
  fresh_confirmed: boolean;
  critical_gates_passed: boolean;
  production_promotion_allowed: boolean;
  limited_pilot_allowed: boolean;
  automatic_action_allowed: boolean;
  test_role: string;
  next_block: string | null;
  finished_at: string | null;
}

export interface MetoceanMetric {
  evaluation_role: string;
  model: string;
  rows: number;
  origins: number;
  mae: number | null;
  rmse: number | null;
  bias: number | null;
  coverage: number | null;
  mean_interval_width: number | null;
  quantile_crossings: number;
}

export interface MetoceanDashboardData {
  status: MetoceanStatus | null;
  forecast: MetoceanForecastPoint[];
  impacts: MetoceanVesselImpact[];
  augmentation: MetoceanAugmentationStatus | null;
  selections: MetoceanTaskSelection[];
  validation: MetoceanValidationStatus | null;
  metrics: MetoceanMetric[];
  forecastTrack: MetoceanForecastPoint['track'] | null;
  unavailable: string[];
}
