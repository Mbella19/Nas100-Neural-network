"""
Configuration settings for the Hybrid NAS100 Trading System.

This module contains all hyperparameters, paths, and constants.
Optimized for Apple M2 Silicon with 8GB RAM constraints.
"""

import torch
import gc
from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple
from pathlib import Path


def get_device() -> torch.device:
    """Get the optimal device for computation."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def clear_memory():
    """Clear GPU/MPS memory cache and run garbage collection."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


@dataclass
class PathConfig:
    """Path configurations for data and model storage."""

    # Base directory (relative to project root)
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Training data directory (external location)
    training_data_dir: Path = field(
        default_factory=lambda: Path("/Users/gervaciusjr/Desktop/Oanda data")
    )

    # Component stock data directory
    components_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "stocks"
    )

    @property
    def data_raw(self) -> Path:
        return self.training_data_dir

    @property
    def data_processed(self) -> Path:
        return self.base_dir / "data" / "processed"

    @property
    def models_analyst(self) -> Path:
        return self.base_dir / "models" / "analyst"

    @property
    def models_agent(self) -> Path:
        return self.base_dir / "models" / "agent"

    @property
    def component_sequences(self) -> Path:
        """Path to preprocessed component sequence data."""
        return self.data_processed / "component_sequences.npz"

    def ensure_dirs(self):
        """Create all necessary directories."""
        for path in [self.data_raw, self.data_processed,
                     self.models_analyst, self.models_agent]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Data file names
    raw_file: str = "NAS100_USD_1min_data.csv"
    processed_file: str = "nas100_processed.parquet"
    datetime_format: str = "ISO8601"  # Auto-detect or ISO format

    # Timeframes (use lowercase 'h' for pandas 2.0+ compatibility)
    # UPDATED: Changed from 15m/1h/4h to 5m/15m/45m for faster trading analysis
    timeframes: Dict[str, str] = field(default_factory=lambda: {
        '5m': '5min',
        '15m': '15min',
        '45m': '45min'
    })

    # Lookback windows (number of candles)
    # v9 FIX: INCREASED lookbacks to provide proper context WITHOUT overlapping prediction window
    # Rule: lookback > prediction horizon to avoid temporal confusion
    # Subsample ratios: 15m = 3x base (5m), 45m = 9x base (5m)
    lookback_windows: Dict[str, int] = field(default_factory=lambda: {
        '5m': 48,    # 4 Hours - 2x prediction horizon (proper context)
        '15m': 16,   # 4 Hours - captures trading session
        '45m': 6     # 4.5 Hours - captures trend
    })

    # Train/validation/test splits
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Memory-efficient chunk size for processing
    chunk_size: int = 100_000


@dataclass
class AnalystConfig:
    """Market Analyst configuration (supports both Transformer and TCN)."""
    # v13: Architecture selection - TCN is more stable for binary classification
    architecture: str = "tcn"   # "transformer" or "tcn"

    # Shared architecture settings
    d_model: int = 32           # Hidden dimension (reduced for smaller Analyst)
    nhead: int = 4              # Transformer only: attention heads (64/4 = 16 dim/head)
    num_layers: int = 2         # Transformer only: encoder layers
    dim_feedforward: int = 128  # Transformer only: FFN hidden dim (4x d_model)
    dropout: float = 0.3        # v14: Standard regularization for small model
    context_dim: int = 32       # Output context vector dimension (Matched to d_model)

    # Input noise regularization (training only)
    # Adds small Gaussian noise to input features during Analyst training to reduce overfitting.
    # 0.0 disables. Typical useful range: 0.002–0.01 on normalized features.
    input_noise_std: float = 0.0

    # TCN-specific settings (v13)
    tcn_num_blocks: int = 4     # Number of residual blocks (dilations: 1, 2, 4, 8) to cover full 48-bar lookback
    tcn_kernel_size: int = 3    # Convolution kernel size

    # Cross-Asset Attention settings (v15)
    # Enables learning from component stock temporal patterns (AAPL, MSFT, NVDA, etc.)
    use_cross_asset_attention: bool = True  # Enable cross-asset attention module
    d_component: int = 32                   # Dimension of component embeddings
    component_seq_len: int = 12             # Lookback window for component sequences
    n_components: int = 6                   # Number of component stocks
    component_input_dim: int = 4            # Features per timestep (OHLC returns)

    batch_size: int = 128       # Keep at 128
    learning_rate: float = 1e-4 # REDUCED from 3e-4 - more stable convergence
    weight_decay: float = 1e-4  # FIXED: Was 1e-2 (100x too high!) - standard value
    max_epochs: int = 100
    patience: int = 20  # Increased from 15 - more time to find recall balance

    cache_clear_interval: int = 50

    # v9 FIX: TARGET DEFINITION - reduced horizon and smoothing for more predictable signal
    future_window: int = 24      # 2 Hours (24 * 5m) - shorter = more predictable
    smooth_window: int = 12      # 1 Hour (12 * 5m) - less smoothing = preserves signal

    # Binary classification mode
    num_classes: int = 2        # Binary: 0=Down, 1=Up
    use_binary_target: bool = True  # Use binary direction target
    min_move_atr_threshold: float = 7.0  # v9 FIX: Was 0.3 - lower = 4x more training data

    # Auxiliary losses (multi-task learning)
    # v14: RE-ENABLED - easier tasks (regime ~70%, volatility ~65%) provide
    # stronger gradients to shared encoder, reducing overfitting and improving direction
    use_auxiliary_losses: bool = True   # v14: Enabled for multi-task learning
    aux_volatility_weight: float = 0.2  # v14: Volatility prediction (MSE)
    aux_regime_weight: float = 0.4      # v14: INCREASED - Regime is easier, stronger gradients

    # Gradient accumulation for smoother updates (effective batch = batch_size * steps)
    gradient_accumulation_steps: int = 2  # Effective batch size = 128 * 2 = 256

    # Multi-horizon prediction (addresses Target Mismatch)
    # DISABLED: Gradient conflicts between horizons caused recall oscillation
    use_multi_horizon: bool = False  # Was True - disabled to focus on single target
    multi_horizon_weights: Dict[str, float] = field(default_factory=lambda: {
        '1h': 0.0,   # 1-hour horizon weight - disabled
        '2h': 0.0,   # 2-hour horizon weight - disabled
        '4h': 1.0    # 4-hour horizon weight (primary target)
    })

    # Legacy 3-class config (kept for compatibility)
    class_std_thresholds: Tuple[float, float] = (-0.15, 0.15)

    # Input Lookback Windows (Must match DataConfig)
    # v9 FIX: INCREASED lookbacks to provide proper context WITHOUT overlapping prediction window
    # UPDATED: Changed from 15m/1h/4h to 5m/15m/45m
    lookback_5m: int = 48       # 4 Hours - 6x prediction horizon (proper context)
    lookback_15m: int = 16      # 4 Hours - captures trading session
    lookback_45m: int = 6       # 4.5 Hours - captures trend


@dataclass
class InstrumentConfig:
    """Instrument-specific parameters for NAS100 (Oanda CFD)."""
    name: str = "NAS100"
    pip_value: float = 1.0           # 1 point = 1.0 price movement (NOT 0.1 tick size)
    lot_size: float = 1.0            # CFD lot ($1 per point per lot)
    point_multiplier: float = 1.0    # PnL: points × pip_value × lot_size × multiplier = $1/point
    min_body_points: float = 2.0     # Pattern detection: min body size (2 points)
    min_range_points: float = 5.0    # Pattern detection: min range size (5 points)


@dataclass
class TradingConfig:
    """Trading environment configuration for NAS100."""
    spread_pips: float = 4.5    # NAS100 typical spread
    slippage_pips: float = 2.5  # NAS100 typical slippage

    # Confidence filtering: Only take trades when agent probability >= threshold
    min_action_confidence: float = 0.0  # Filter low-confidence trades (0.0 = disabled)

    # NEW: Enforce Analyst Alignment (Action Masking)
    # If True, Agent can ONLY trade in direction of Analyst (or Flat)
    # DISABLED: Soft masking breaks PPO gradients - agent samples action X, gets
    # masked to Flat, but PPO updates as if X led to the reward. This causes
    # frozen action distributions and no learning.
    enforce_analyst_alignment: bool = False
    
    # NEW: Risk-Based Sizing (Not Fixed Lots)
    risk_multipliers: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    
    # NEW: ATR-Based Stops (Not Fixed Pips)
    sl_atr_multiplier: float = 2.0
    tp_atr_multiplier: float = 6.0
    
    # Risk Limits
    max_position_size: float = 5.0
    
    # Reward Params (calibrated for NAS100)
    # NAS100 has ~100-200 point daily range vs EURUSD ~50-100 pip range
    # reward_scaling = 0.01 means 100 points = 1.0 reward (similar magnitude to EURUSD)
    fomo_penalty: float = -0.5    # Moderate penalty for missing high-momentum moves
    chop_penalty: float = 0.0     # Disabled (can cause over-penalization in legitimate ranging trades)
    fomo_threshold_atr: float = 6.0  # Trigger on >4x ATR moves
    chop_threshold: float = 80.0     # Only extreme chop triggers penalty
    reward_scaling: float = 0.01    # 1.0 reward per 100 points (NAS100 calibration)

    # Trade entry bonus: Offsets entry cost to encourage exploration
    # NAS100 spread ~2.5 points × 0.01 = 0.025 reward cost, so bonus = 0.03
    trade_entry_bonus: float = 0.003  # REMOVED BONUS: Agent pays full cost (was 0.03)
    
    # These are mostly unused now but keep for compatibility if needed
    use_stop_loss: bool = True
    use_take_profit: bool = True
    
    # Environment settings
    max_steps_per_episode: int = 500    # Reduced to ~1 week (was 2000) for rapid regime cycling
    initial_balance: float = 10000.0
    
    # Validation
    noise_level: float = 0.01  # Reduced to 2% to encourage more activity (was 5%)

    # NEW: "Full Eyes" Agent Features
    agent_lookback_window: int = 12   # Increased to 12 as requested (60 mins of 5m bars)
    include_structure_features: bool = True  # Agent sees BOS/CHoCH
    include_attention_features: bool = True  # Agent sees which stocks drive the signal


@dataclass
class AgentConfig:
    """PPO Sniper Agent configuration."""

    # PPO hyperparameters (from CLAUDE.md spec)
    # FIX v15: Previous ent_coef (0.01) caused rapid policy collapse to flat.
    # For a 12-action discrete space, higher entropy is needed to maintain exploration.
    learning_rate: float = 1e-4  # Higher initial LR for faster learning (was 1e-4)
    n_steps: int = 2048         # Timesteps per update
    batch_size: int = 256       # Minibatch size
    n_epochs: int = 10          # Reduced to prevent overfitting (was 20)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01        # HIGH entropy to force exploration (was 0.01)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Training
    total_timesteps: int = 20_000_000

    # Policy network
    # FIX v15: [64, 64] may bottleneck for 49-dim input with 12-action output
    policy_type: str = "MlpPolicy"
    net_arch: List[int] = field(default_factory=lambda: [256, 256])


@dataclass
class FeatureConfig:
    """Feature engineering configuration for NAS100."""

    # Price action patterns (thresholds in price units, not pips)
    # NAS100: 1 point = 0.1 price units, so 2.0 = 20 points, 5.0 = 50 points
    pinbar_wick_ratio: float = 2.0    # Wick must be > 2x body
    doji_body_ratio: float = 0.1      # Body < 10% of range
    min_body_points: float = 2.0      # Min body size in points (NAS100)
    min_range_points: float = 5.0     # Min candle range in points (NAS100)

    # Market structure
    fractal_window: int = 5           # Williams fractal window
    sr_lookback: int = 100            # S/R level lookback

    # Trend indicators
    sma_period: int = 200
    ema_fast: int = 12
    ema_slow: int = 26

    # Regime indicators
    chop_period: int = 14
    adx_period: int = 14
    atr_period: int = 14

    # Volatility sizing reference (NAS100 calibration)
    risk_pips_target: float = 50.0    # Reference risk ~50 points (was 15 for EURUSD)


@dataclass
class SMCConfig:
    """Smart Money Concepts (SMC) microstructure feature configuration (OHLC-only)."""

    # Base timeframe used for bar-based hyperparameters below (minutes per bar).
    # The pipeline is configured around a 5-minute decision timeframe.
    base_timeframe_minutes: int = 5

    # Inclusion toggles (agent-only)
    include_higher_timeframes: bool = True   # Add 15m/45m blocks to the agent observation
    include_m1_features: bool = True         # Compute 1m SMC and resample to 5m grid

    # Pivot/swing detection (odd window; 5 means 2 bars each side)
    swing_length: int = 5

    # Fair Value Gaps (FVG)
    fvg_min_size_atr: float = 0.25
    fvg_max_age_bars: int = 240
    fvg_max_active: int = 50

    # Order Blocks (OB)
    ob_lookback: int = 20
    ob_max_age_bars: int = 300
    ob_max_active: int = 50

    # Liquidity pools (swing-high/low clustering)
    liquidity_lookback_bars: int = 300
    liquidity_range_pct: float = 0.001  # 0.1% of price (NAS100-friendly)
    liquidity_range_atr: float = 0.50   # also allow ATR-based clustering

    # Displacement candles
    displacement_lookback: int = 20
    displacement_multiplier: float = 1.5
    displacement_min_body_ratio: float = 0.70
    body_wick_ratio_clip: float = 20.0
    momentum_z_clip: float = 5.0

    # Buying/Selling pressure proxies (OHLC-only)
    cumulative_delta_window: int = 50
    buying_pressure_windows: Tuple[int, int, int] = (1, 3, 9)  # 5m/15m/45m on 5m bars

    # Session/Killzone features (assumes index is UTC-naive like the pipeline)
    london_killzone_hours: Tuple[int, int] = (2, 5)  # [02:00, 05:00)
    ny_killzone_hours: Tuple[int, int] = (7, 10)     # [07:00, 10:00)

    # Previous period levels
    weekly_resample_rule: str = "W-FRI"

    def for_timeframe_minutes(self, timeframe_minutes: int) -> "SMCConfig":
        """
        Return a derived config scaled for a specific bar size.

        Bar-based fields (lookbacks/ages/windows expressed in BASE bars) are scaled to
        preserve approximately the same wall-clock horizons across timeframes:
          scaled_bars ≈ base_bars * (base_minutes / timeframe_minutes)

        Notes:
        - Thresholds expressed in ATR/price units are not scaled.
        - `swing_length` is forced to an odd integer >= 3 for fractal confirmation.
        - Buying pressure windows are expressed in minutes (5/15/45) and converted to bars.
        """
        if timeframe_minutes <= 0:
            raise ValueError("timeframe_minutes must be positive")

        base_minutes = int(self.base_timeframe_minutes) if self.base_timeframe_minutes else 5
        scale = float(base_minutes) / float(timeframe_minutes)

        def _scale_int(value: int, *, min_value: int = 1) -> int:
            return max(min_value, int(round(float(value) * scale)))

        def _scale_odd(value: int, *, min_value: int = 3) -> int:
            scaled = _scale_int(value, min_value=min_value)
            if scaled % 2 == 0:
                scaled += 1
            return max(min_value, scaled)

        # Convert absolute-minute windows into bars for this timeframe.
        bp_5m = max(1, int(round(5.0 / float(timeframe_minutes))))
        bp_15m = max(1, int(round(15.0 / float(timeframe_minutes))))
        bp_45m = max(1, int(round(45.0 / float(timeframe_minutes))))

        return replace(
            self,
            swing_length=_scale_odd(int(self.swing_length), min_value=3),
            fvg_max_age_bars=_scale_int(int(self.fvg_max_age_bars), min_value=1),
            ob_lookback=_scale_int(int(self.ob_lookback), min_value=1),
            ob_max_age_bars=_scale_int(int(self.ob_max_age_bars), min_value=1),
            liquidity_lookback_bars=_scale_int(int(self.liquidity_lookback_bars), min_value=1),
            displacement_lookback=_scale_int(int(self.displacement_lookback), min_value=2),
            cumulative_delta_window=_scale_int(int(self.cumulative_delta_window), min_value=5),
            buying_pressure_windows=(bp_5m, bp_15m, bp_45m),
        )


@dataclass
class Config:
    """Master configuration combining all sub-configs."""

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    analyst: AnalystConfig = field(default_factory=AnalystConfig)
    instrument: InstrumentConfig = field(default_factory=InstrumentConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    smc: SMCConfig = field(default_factory=SMCConfig)

    # Global settings
    seed: int = 42
    dtype: torch.dtype = torch.float32  # NEVER use float64 on M2
    device: torch.device = field(default_factory=get_device)

    def __post_init__(self):
        """Ensure directories exist and set random seeds."""
        self.paths.ensure_dirs()
        torch.manual_seed(self.seed)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)


# Global configuration instance
config = Config()
