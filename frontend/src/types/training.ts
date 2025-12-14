/**
 * TypeScript types for training visualization data.
 * Matches the Pydantic models in visualization/models.py
 */

export interface OHLCBar {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface TradeMarker {
  id?: number;
  timestamp: number;
  price: number;
  entry?: number;
  exit?: number | null;
  direction: -1 | 0 | 1; // -1=Short, 0=Flat, 1=Long
  size: number;
  pnl?: number;
  is_entry: boolean;
  status?: string;
  duration?: number;
  close_reason?: 'exit' | 'stop_loss' | 'take_profit';
}

export interface AnalystState {
  epoch: number;
  train_loss: number;
  val_loss: number;
  train_acc: number;
  val_acc: number;
  direction_acc: number;
  grad_norm: number;
  learning_rate: number;
  epochs_without_improvement?: number;
  attention_weights: [number, number]; // [1h, 4h]
  p_down: number;
  p_up: number;
  p_neutral?: number;
  confidence: number;
  edge: number;
  uncertainty: number;
  encoder_15m_norm: number;
  encoder_1h_norm: number;
  encoder_4h_norm: number;
  context_vector_sample: number[];
}

export interface AgentState {
  timestep: number;
  episode: number;
  episode_reward: number;
  episode_pnl: number;
  episode_trades: number;
  win_rate: number;
  avg_reward?: number;
  best_reward?: number;
  action_probs: [number, number, number]; // [flat, long, short]
  size_probs: [number, number, number, number]; // [0.25x, 0.5x, 0.75x, 1.0x]
  value_estimate: number;
  advantage: number;
  entropy: number;
  learning_rate?: number;
  last_action_direction: 0 | 1 | 2;
  last_action_size: 0 | 1 | 2 | 3;
}

export interface MarketState {
  price: number;
  current_price: number;
  ohlc?: OHLCBar;
  position: -1 | 0 | 1;
  position_size: number;
  entry_price: number;
  unrealized_pnl: number;
  total_pnl: number;
  n_trades: number;
  sl_level: number;
  tp_level: number;
  atr: number;
  chop: number;
  adx: number;
  regime: 0 | 1 | 2; // 0=Bullish, 1=Ranging, 2=Bearish
  sma_distance: number;
}

export interface RewardComponents {
  pnl_delta: number;
  transaction_cost: number;
  direction_bonus: number;
  confidence_bonus: number;
  fomo_penalty: number;
  chop_penalty: number;
  total: number;
}

export interface SystemStatus {
  phase: 'idle' | 'analyst_training' | 'agent_training' | 'backtest';
  memory_used_mb: number;
  memory_total_mb: number;
  memory_percent?: number;
  mps_memory_used?: number;
  ram_used?: number;
  cpu_percent?: number;
  steps_per_second: number;
  steps_per_sec?: number;
  episodes_per_hour: number;
  elapsed_seconds: number;
  latency_ms?: number;
  messages_per_sec?: number;
  device: string;
}

export interface TrainingSnapshot {
  timestamp: number;
  message_type: 'snapshot' | 'trade' | 'episode_end' | 'analyst_epoch' | 'agent_step' | 'full_state';
  analyst?: AnalystState;
  agent?: AgentState;
  market?: MarketState;
  reward?: RewardComponents;
  system?: SystemStatus;
  trade?: TradeMarker;
  price_history?: OHLCBar[];
  trade_history?: TradeMarker[];
  loss_history?: Array<{ epoch: number; train_loss: number; val_loss: number }>;
  reward_history?: number[];
  clear_chart?: boolean;  // When true, frontend should clear price/trade history
}

export interface TrainingStore {
  // Connection state
  connected: boolean;
  lastUpdate: number;

  // Current state
  analyst: AnalystState | null;
  agent: AgentState | null;
  market: MarketState | null;
  reward: RewardComponents | null;
  system: SystemStatus | null;

  // History
  priceHistory: OHLCBar[];
  tradeHistory: TradeMarker[];
  lossHistory: Array<{ epoch: number; train_loss: number; val_loss: number }>;
  rewardHistory: number[];
  pnlHistory: number[];

  // Actions
  setConnected: (connected: boolean) => void;
  updateFromSnapshot: (snapshot: TrainingSnapshot) => void;
  reset: () => void;
}
