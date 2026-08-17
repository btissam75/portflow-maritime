export interface CapacityRankingStatus {
  audit_status: string;
  decision: string;
  policy_version: string;
  selected_candidate_id: string | null;
  selected_score: string | null;
  selected_top_k: number | null;
  bucket_hours: number | null;
  contracts_passed: boolean;
  integrity_passed: boolean;
  shadow_api_allowed: boolean;
  production_promotion_allowed: boolean;
  automatic_action_allowed: boolean;
  serving_rows: number;
  next_block: string | null;
  finished_at: string | null;
}

export interface CapacityDecision {
  port_call_id: string;
  vessel_name: string | null;
  port_code: string | null;
  terminal_code: string | null;
  vessel_type: string | null;
  cargo_group: string | null;
  landmark_at: string;
  decision_at: string;
  evaluation_role: string;
  risk_score: number;
  rank_in_window: number;
  active_calls: number;
  capacity: number;
  watchlist_selected: boolean;
  action_tier: string;
  reason_code: string;
  p_delay_gt3: number;
  hazard_6h: number;
  hazard_12h: number;
  hazard_24h: number;
  remaining_p10_h: number;
  remaining_p50_h: number;
  remaining_p90_h: number;
  hsmm_state: string | null;
  hsmm_state_confidence: number | null;
  production_claim_allowed: boolean;
  automatic_action_allowed: boolean;
}

export interface CapacitySnapshot {
  requested_at: string | null;
  resolved_at: string;
  evaluation_role: string;
  active_calls: number;
  capacity: number;
  selected_calls: number;
  decisions: CapacityDecision[];
}

export interface CapacityDashboardData {
  status: CapacityRankingStatus | null;
  snapshot: CapacitySnapshot | null;
  unavailable: string[];
  fetchedAt: string;
}
