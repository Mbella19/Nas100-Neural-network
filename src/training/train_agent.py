"""
Training script for the PPO Sniper Agent.

Trains the RL agent using a frozen Market Analyst to provide
context vectors for decision making.

Memory-optimized for Apple M2 Silicon.

Features:
- Comprehensive logging of training progress
- Detailed reward and action statistics
- Training visualizations (reward curves, action distributions)
- Episode-level tracking and analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import gc
import json

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from ..models.analyst import load_analyst, MarketAnalyst
from ..environments.trading_env import TradingEnv
from ..agents.sniper_agent import SniperAgent, create_agent
from ..utils.logging_config import setup_logging, get_logger
from ..utils.metrics import calculate_trading_metrics, TradingMetrics
from ..data.features import (
    compute_regime_labels,
    add_market_sessions,
    detect_fractals,
    detect_structure_breaks
)
from ..data.microstructure import add_smc_microstructure_features
from ..data.feature_names import (
    SMC_FEATURE_COLUMNS,
    AGENT_TF_BASE_FEATURE_COLUMNS,
    AGENT_MARKET_FEATURE_COLUMNS,
)
from config.settings import SMCConfig

logger = get_logger(__name__)


class AgentTrainingLogger(BaseCallback):
    """
    Custom callback for detailed agent training logging.

    Tracks:
    - Episode rewards
    - Action distributions
    - PnL statistics
    - Win rate evolution
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_freq: int = 1000,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.log_dir = Path(log_dir) if log_dir else None
        self.log_freq = log_freq

        # Tracking variables
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_pnls: List[float] = []
        self.episode_trades: List[int] = []
        self.episode_win_rates: List[float] = []
        self.action_counts = {0: 0, 1: 0, 2: 0}  # Flat, Long, Short
        self.size_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Position sizes: 0.25, 0.5, 0.75, 1.0
        self.timestep_rewards: List[float] = []

        # Current episode tracking
        self.current_ep_reward = 0
        self.current_ep_length = 0
        self.current_ep_actions = []

        # Training start time
        self.start_time = None

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _on_training_start(self):
        self.start_time = datetime.now()
        logger.info("=" * 70)
        logger.info("PPO AGENT TRAINING STARTED")
        logger.info("=" * 70)
        logger.info(f"Total timesteps: {self.model.num_timesteps if hasattr(self.model, 'num_timesteps') else 'N/A'}")
        logger.info(f"Log directory: {self.log_dir}")
        logger.info("=" * 70)

    def _on_step(self) -> bool:
        # Track rewards
        if len(self.locals.get('rewards', [])) > 0:
            reward = self.locals['rewards'][0]
            self.timestep_rewards.append(reward)
            self.current_ep_reward += reward
            self.current_ep_length += 1

        # Track actions (direction and size for MultiDiscrete)
        if len(self.locals.get('actions', [])) > 0:
            action = self.locals['actions'][0]
            if isinstance(action, np.ndarray):
                if len(action) >= 2:
                    # MultiDiscrete action: [direction, size]
                    direction = int(action[0])
                    size = int(action[1])
                    self.action_counts[direction] = self.action_counts.get(direction, 0) + 1
                    self.size_counts[size] = self.size_counts.get(size, 0) + 1
                    self.current_ep_actions.append(direction)
                else:
                    # Single action (fallback)
                    direction = int(action[0]) if len(action) == 1 else int(action)
                    self.action_counts[direction] = self.action_counts.get(direction, 0) + 1
                    self.current_ep_actions.append(direction)
            elif isinstance(action, (int, np.integer)):
                self.action_counts[int(action)] = self.action_counts.get(int(action), 0) + 1
                self.current_ep_actions.append(int(action))

        # Check for episode done
        dones = self.locals.get('dones', [False])
        if any(dones):
            self.episode_rewards.append(self.current_ep_reward)
            self.episode_lengths.append(self.current_ep_length)

            # Get episode info if available
            infos = self.locals.get('infos', [{}])
            if len(infos) > 0 and infos[0]:
                info = infos[0]
                self.episode_pnls.append(info.get('total_pnl', 0.0))
                # CRITICAL FIX: Use 'n_trades' (what env actually sets) instead of 'total_trades'
                self.episode_trades.append(info.get('n_trades', 0))
                # Calculate win rate from trades if available
                n_trades = info.get('n_trades', 0)
                win_rate = 0.0
                if n_trades > 0 and 'trades' in info:
                    wins = sum(1 for t in info['trades'] if t.get('pnl', 0) > 0)
                    win_rate = wins / n_trades
                self.episode_win_rates.append(win_rate)

            # Log episode summary
            n_episodes = len(self.episode_rewards)
            if n_episodes % 10 == 0:
                self._log_episode_summary(n_episodes)

            # Reset current episode tracking
            self.current_ep_reward = 0
            self.current_ep_length = 0
            self.current_ep_actions = []

        # Periodic detailed logging
        if self.n_calls % self.log_freq == 0:
            self._log_training_progress()

        return True

    def _log_episode_summary(self, n_episodes: int):
        """Log summary for recent episodes."""
        recent_rewards = self.episode_rewards[-10:]
        recent_pnls = self.episode_pnls[-10:] if self.episode_pnls else [0]

        logger.info("-" * 50)
        logger.info(f"Episode {n_episodes} Summary:")
        logger.info(f"  Recent Avg Reward: {np.mean(recent_rewards):.2f}")
        logger.info(f"  Recent Avg PnL: {np.mean(recent_pnls):.2f} pips")
        logger.info(f"  Episode Length: {self.episode_lengths[-1]}")

        if self.episode_win_rates:
            logger.info(f"  Win Rate: {self.episode_win_rates[-1]*100:.1f}%")

    def _log_training_progress(self):
        """Log overall training progress."""
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return

        action_pcts = {k: v/total_actions*100 for k, v in self.action_counts.items()}

        logger.info("-" * 50)
        logger.info(f"Training Progress @ {self.n_calls} steps:")
        logger.info(f"  Episodes completed: {len(self.episode_rewards)}")
        logger.info(f"  Action Distribution: Flat={action_pcts.get(0, 0):.1f}%, "
                   f"Long={action_pcts.get(1, 0):.1f}%, Short={action_pcts.get(2, 0):.1f}%")

        if self.episode_rewards:
            logger.info(f"  Avg Episode Reward: {np.mean(self.episode_rewards):.2f}")
            logger.info(f"  Max Episode Reward: {np.max(self.episode_rewards):.2f}")
            logger.info(f"  Min Episode Reward: {np.min(self.episode_rewards):.2f}")

        if self.episode_pnls:
            logger.info(f"  Avg PnL: {np.mean(self.episode_pnls):.2f} pips")

        # Memory info
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def _on_training_end(self):
        """Log final training summary and create visualizations."""
        total_time = (datetime.now() - self.start_time).total_seconds()

        logger.info("=" * 70)
        logger.info("PPO AGENT TRAINING COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total Training Time: {total_time/60:.1f} minutes")
        logger.info(f"Total Episodes: {len(self.episode_rewards)}")
        logger.info(f"Total Timesteps: {self.n_calls}")

        if self.episode_rewards:
            logger.info(f"Final Avg Reward (last 100): {np.mean(self.episode_rewards[-100:]):.2f}")

        if self.episode_pnls:
            logger.info(f"Final Avg PnL (last 100): {np.mean(self.episode_pnls[-100:]):.2f} pips")

        logger.info("=" * 70)

        # Save metrics
        if self.log_dir:
            self._save_metrics()
            self._create_visualizations()

    def _save_metrics(self):
        """Save training metrics to JSON."""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': self.action_counts,
            'total_timesteps': self.n_calls,
            'total_episodes': len(self.episode_rewards)
        }

        metrics_path = self.log_dir / 'agent_training_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Metrics saved to: {metrics_path}")

    def _create_visualizations(self):
        """Create training visualizations."""
        if len(self.episode_rewards) < 10:
            logger.warning("Not enough episodes for visualizations")
            return

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        # Episode rewards
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, alpha=0.3, color='blue')
        # Smoothed
        window = min(50, len(self.episode_rewards) // 5 + 1)
        smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(self.episode_rewards)), smoothed,
               color='red', linewidth=2, label=f'Smoothed (w={window})')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # PnL per episode
        ax = axes[0, 1]
        if self.episode_pnls:
            ax.plot(self.episode_pnls, alpha=0.3, color='green')
            if len(self.episode_pnls) >= window:
                smoothed_pnl = np.convolve(self.episode_pnls, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.episode_pnls)), smoothed_pnl,
                       color='darkgreen', linewidth=2)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('PnL (pips)')
        ax.set_title('Episode PnL')
        ax.grid(True, alpha=0.3)

        # Win rate evolution
        ax = axes[0, 2]
        if self.episode_win_rates:
            ax.plot(self.episode_win_rates, alpha=0.5, color='purple')
            ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='50%')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate Evolution')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Action distribution
        ax = axes[1, 0]
        action_names = ['Flat', 'Long', 'Short']
        action_vals = [self.action_counts.get(i, 0) for i in range(3)]
        colors = ['gray', 'green', 'red']
        bars = ax.bar(action_names, action_vals, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Action Distribution')
        # Add percentage labels
        total = sum(action_vals)
        if total > 0:
            for bar, val in zip(bars, action_vals):
                pct = val / total * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{pct:.1f}%', ha='center', va='bottom')

        # Episode length distribution
        ax = axes[1, 1]
        ax.hist(self.episode_lengths, bins=30, color='blue', alpha=0.7)
        ax.axvline(x=np.mean(self.episode_lengths), color='red',
                  linestyle='--', label=f'Mean: {np.mean(self.episode_lengths):.0f}')
        ax.set_xlabel('Episode Length')
        ax.set_ylabel('Count')
        ax.set_title('Episode Length Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Cumulative PnL
        ax = axes[1, 2]
        if self.episode_pnls:
            cumulative_pnl = np.cumsum(self.episode_pnls)
            ax.plot(cumulative_pnl, color='green', linewidth=2)
            ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, alpha=0.3)
            ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cumulative PnL (pips)')
        ax.set_title('Cumulative PnL')
        ax.grid(True, alpha=0.3)

        plt.suptitle('PPO Agent Training Summary', fontsize=14, fontweight='bold')
        plt.tight_layout()

        save_path = self.log_dir / 'agent_training_summary.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Training visualization saved to: {save_path}")

    def get_metrics(self) -> Dict:
        """Get all tracked metrics."""
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_pnls': self.episode_pnls,
            'episode_trades': self.episode_trades,
            'episode_win_rates': self.episode_win_rates,
            'action_counts': self.action_counts
        }


def prepare_env_data(
    df_5m,
    df_15m,
    df_45m,
    feature_cols: list,
    lookback_5m: int = 48,
    lookback_15m: int = 16,
    lookback_45m: int = 6,
    component_sequences: Optional[np.ndarray] = None,
    smc_cfg: Optional[SMCConfig] = None,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Prepare windowed data for the trading environment.
    
    FIXED: 15m and 45m data now correctly subsampled from the aligned 5m index.
    Since df_15m and df_45m are aligned to the 5m index via forward-fill,
    we subsample every 3rd bar for 15m and every 9th bar for 45m.

    Returns:
        Tuple of (data_5m, data_15m, data_45m, close_prices, market_features, component_data, returns)
    """
    # Subsampling ratios: how many 5m bars per higher TF bar
    subsample_15m = 3   # 3 x 5m = 15m
    subsample_45m = 9   # 9 x 5m = 45m
    
    # Calculate valid range accounting for subsampling.
    # Need enough bars for: 5m lookback, (15m lookback-1)*3+1, (45m lookback-1)*9+1.
    # This matches `src/training/precompute_analyst.py` and avoids dropping extra valid samples.
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(df_5m) - start_idx

    logger.info(f"Preparing {n_samples} samples for environment")
    logger.info(f"  5m: {lookback_5m} bars = {lookback_5m * 5 / 60:.1f} hours")
    logger.info(f"  15m: {lookback_15m} bars = {lookback_15m * 15 / 60:.1f} hours (using {(lookback_15m - 1) * subsample_15m + 1} aligned indices)")
    logger.info(f"  45m: {lookback_45m} bars = {lookback_45m * 45 / 60:.1f} hours (using {(lookback_45m - 1) * subsample_45m + 1} aligned indices)")

    # Prepare windowed data
    data_5m = np.zeros((n_samples, lookback_5m, len(feature_cols)), dtype=np.float32)
    data_15m = np.zeros((n_samples, lookback_15m, len(feature_cols)), dtype=np.float32)
    data_45m = np.zeros((n_samples, lookback_45m, len(feature_cols)), dtype=np.float32)

    features_5m = df_5m[feature_cols].values.astype(np.float32)
    features_15m = df_15m[feature_cols].values.astype(np.float32)
    features_45m = df_45m[feature_cols].values.astype(np.float32)

    for i in range(n_samples):
        actual_idx = start_idx + i
        # 5m: direct indexing (include current candle)
        data_5m[i] = features_5m[actual_idx - lookback_5m + 1:actual_idx + 1]

        # FIXED: 15m - subsample every 3rd bar from aligned data, INCLUDING current candle
        # range() is exclusive at end, so we use actual_idx + 1 to include current
        idx_range_15m = list(range(
            actual_idx - (lookback_15m - 1) * subsample_15m,
            actual_idx + 1,
            subsample_15m
        ))
        data_15m[i] = features_15m[idx_range_15m]

        # FIXED: 45m - subsample every 9th bar from aligned data, INCLUDING current candle
        idx_range_45m = list(range(
            actual_idx - (lookback_45m - 1) * subsample_45m,
            actual_idx + 1,
            subsample_45m
        ))
        data_45m[i] = features_45m[idx_range_45m]

    # Close prices
    close_prices = df_5m['close'].values[start_idx:start_idx + n_samples].astype(np.float32)
    
    # Returns (for "Full Eyes" agent peripheral vision)
    # Using 'returns' column if available, else derive from prices
    if 'returns' in df_5m.columns:
        returns = df_5m['returns'].values[start_idx:start_idx + n_samples].astype(np.float32)
    else:
        # Calculate returns on the fly
        ret = df_5m['close'].pct_change().fillna(0).values
        returns = ret[start_idx:start_idx + n_samples].astype(np.float32)

    # ------------------------------------------------------------------
    # Agent-only microstructure (SMC) + multi-timeframe visibility
    # ------------------------------------------------------------------
    if smc_cfg is None:
        smc_cfg = SMCConfig()
    include_higher_tfs = bool(getattr(smc_cfg, "include_higher_timeframes", True))

    # 5m SMC (always computed/available in the main pipeline; compute as fallback)
    try:
        missing_5m = [c for c in SMC_FEATURE_COLUMNS if c not in df_5m.columns]
        if missing_5m:
            logger.info(f"Computing SMC microstructure features for 5m (missing {len(missing_5m)})...")
            add_smc_microstructure_features(df_5m, smc_cfg.for_timeframe_minutes(5), inplace=True)
    except Exception as e:
        logger.warning(f"Failed to compute 5m SMC features; filling zeros: {e}")

    if include_higher_tfs:
        # 15m/45m SMC: these SHOULD be computed on native TF before alignment in the pipeline.
        # If missing, we fall back to computing on the aligned series (less ideal, but keeps training runnable).
        for tf_name, df_tf in (("15m", df_15m), ("45m", df_45m)):
            try:
                missing_tf = [c for c in SMC_FEATURE_COLUMNS if c not in df_tf.columns]
                if missing_tf:
                    logger.info(f"Computing SMC microstructure features for {tf_name} (missing {len(missing_tf)})...")
                    tf_minutes = 15 if tf_name == "15m" else 45
                    add_smc_microstructure_features(df_tf, smc_cfg.for_timeframe_minutes(tf_minutes), inplace=True)
            except Exception as e:
                logger.warning(f"Failed to compute {tf_name} SMC features; filling zeros: {e}")

    # Copy 15m/45m base+SMC features onto df_5m with suffixes so the env observes them.
    if include_higher_tfs:
        tf_copy_cols = list(AGENT_TF_BASE_FEATURE_COLUMNS) + list(SMC_FEATURE_COLUMNS)
        for tf_name, df_tf in (("15m", df_15m), ("45m", df_45m)):
            # Only add columns that are not already present on the 5m frame.
            cols_to_add = [c for c in tf_copy_cols if f"{c}_{tf_name}" not in df_5m.columns]
            if not cols_to_add:
                continue
            for col in cols_to_add:
                if col not in df_tf.columns:
                    df_tf[col] = 0.0
            df_5m = df_5m.join(df_tf[cols_to_add].add_suffix(f"_{tf_name}"), how="left")

    # Market features for reward shaping (includes S/R for breakout vs chase detection)
    market_cols = list(AGENT_MARKET_FEATURE_COLUMNS)
    for col in market_cols:
        if col not in df_5m.columns:
            df_5m[col] = 0.0

    # Ensure no NaNs reach the environment (NaNs can crash SB3 / destabilize PPO).
    df_5m[market_cols] = df_5m[market_cols].fillna(0.0).astype(np.float32)

    market_features = df_5m[market_cols].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Component data (optional, already windowed per index)
    component_data = None
    if component_sequences is not None:
        try:
            if component_sequences.shape[0] == len(df_5m):
                component_data = component_sequences[start_idx:start_idx + n_samples].astype(np.float32)
            else:
                logger.warning(
                    f"Component sequences length mismatch ({component_sequences.shape[0]} vs {len(df_5m)}); "
                    "ignoring component data."
                )
        except Exception as e:
            logger.warning(f"Failed to trim component sequences: {e}")
            component_data = None

            component_data = None
            
    return data_5m, data_15m, data_45m, close_prices, market_features, component_data, returns


def create_trading_env(
    data_15m: np.ndarray,
    data_1h: np.ndarray,
    data_4h: np.ndarray,
    close_prices: np.ndarray,
    market_features: np.ndarray,
    component_data: Optional[np.ndarray] = None,
    analyst_model: Optional[MarketAnalyst] = None,
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    market_feat_mean: Optional[np.ndarray] = None,  # Pre-computed from training
    market_feat_std: Optional[np.ndarray] = None,   # Pre-computed from training
    regime_labels: Optional[np.ndarray] = None,     # Regime labels for balanced sampling
    use_regime_sampling: bool = True,               # Enable regime-balanced episode starts
    precomputed_analyst_cache: Optional[dict] = None,  # Pre-computed Analyst outputs
    ohlc_data: Optional[np.ndarray] = None,         # Real OHLC for visualization
    timestamps: Optional[np.ndarray] = None,        # Real timestamps for visualization
    returns: Optional[np.ndarray] = None,           # Returns for Full Eyes
) -> TradingEnv:
    """
    Create the trading environment.

    Args:
        data_*: Prepared window data
        close_prices: Close prices for PnL
        market_features: Features for reward shaping
        component_data: Optional component windows aligned to data_* samples
        analyst_model: Frozen Market Analyst
        config: TradingConfig
        device: Torch device

    Returns:
        TradingEnv instance
    """
    # Default configuration (matches config/settings.py v16 fixes)
    # FIX v16: After fixing mixed PnL units and inverted entry_price_norm,
    # these defaults now produce correct reward signals
    spread_pips = 0.2       # Razor/Raw spread
    slippage_pips = 0.5     # Includes commission + slippage
    fomo_penalty = -0.5     # Moderate penalty for missing high-momentum moves
    chop_penalty = 0.0      # Disabled
    fomo_threshold_atr = 4.0  # Trigger on >1.5x ATR moves
    chop_threshold = 80.0   # Only extreme chop triggers penalty
    max_steps = 500
    reward_scaling = 0.1    # 1.0 reward per 1 pip (safe after fixing unit bugs)
    context_dim = 64
    trade_entry_bonus = 0.01  # Moderate bonus (~half of entry cost)
    noise_level = 0.0         # Default: no observation noise unless configured

    # Risk Management defaults
    sl_atr_multiplier = 1.0
    tp_atr_multiplier = 3.0
    use_stop_loss = True
    use_take_profit = True

    # Volatility Sizing
    volatility_sizing = True
    risk_pips_target = 15.0
    
    # Analyst Alignment - Force agent to trade only in analyst's predicted direction
    # DISABLED: Soft masking breaks PPO gradients - agent samples action X, gets
    # masked to Flat, but PPO updates as if X led to the reward. This causes
    # frozen action distributions and no learning.
    enforce_analyst_alignment = False

    if config is not None:
        # FIX: Access config.trading for trading parameters (not config directly)
        trading_cfg = getattr(config, 'trading', config)
        spread_pips = getattr(trading_cfg, 'spread_pips', spread_pips)
        slippage_pips = getattr(trading_cfg, 'slippage_pips', slippage_pips)
        fomo_penalty = getattr(trading_cfg, 'fomo_penalty', fomo_penalty)
        chop_penalty = getattr(trading_cfg, 'chop_penalty', chop_penalty)
        fomo_threshold_atr = getattr(trading_cfg, 'fomo_threshold_atr', fomo_threshold_atr)
        chop_threshold = getattr(trading_cfg, 'chop_threshold', chop_threshold)
        max_steps = getattr(trading_cfg, 'max_steps_per_episode', max_steps)
        reward_scaling = getattr(trading_cfg, 'reward_scaling', reward_scaling)
        # Risk Management
        sl_atr_multiplier = getattr(trading_cfg, 'sl_atr_multiplier', sl_atr_multiplier)
        tp_atr_multiplier = getattr(trading_cfg, 'tp_atr_multiplier', tp_atr_multiplier)
        use_stop_loss = getattr(trading_cfg, 'use_stop_loss', use_stop_loss)
        use_take_profit = getattr(trading_cfg, 'use_take_profit', use_take_profit)
        # Analyst Alignment
        enforce_analyst_alignment = getattr(trading_cfg, 'enforce_analyst_alignment', enforce_analyst_alignment)
        # v15: Trade entry bonus
        trade_entry_bonus = getattr(trading_cfg, 'trade_entry_bonus', trade_entry_bonus)
        noise_level = getattr(trading_cfg, 'noise_level', noise_level)

        # Log config values to verify they're applied
        logger.info(f"Config applied: fomo_penalty={fomo_penalty}, reward_scaling={reward_scaling}, "
                    f"slippage_pips={slippage_pips}, trade_entry_bonus={trade_entry_bonus}, "
                    f"enforce_analyst_alignment={enforce_analyst_alignment}, noise_level={noise_level}")

    if analyst_model is not None:
        context_dim = analyst_model.context_dim
        # Get num_classes from analyst (binary=2, multi-class=3)
        num_classes = getattr(analyst_model, 'num_classes', 2)
    else:
        num_classes = 2  # Default to binary

    env = TradingEnv(
        # Map legacy arg names to TradingEnv signature
        data_5m=data_15m,
        data_15m=data_1h,
        data_45m=data_4h,
        close_prices=close_prices,
        market_features=market_features,
        component_data=component_data,
        analyst_model=analyst_model,
        context_dim=context_dim,
        spread_pips=spread_pips,
        slippage_pips=slippage_pips,
        fomo_penalty=fomo_penalty,
        chop_penalty=chop_penalty,
        fomo_threshold_atr=fomo_threshold_atr,
        chop_threshold=chop_threshold,
        max_steps=max_steps,
        reward_scaling=reward_scaling,  # v15 FIX: Use local variable, not config directly
        trade_entry_bonus=trade_entry_bonus,  # v15: Exploration bonus
        device=device,
        noise_level=noise_level,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        # Risk Management
        sl_atr_multiplier=sl_atr_multiplier,
        tp_atr_multiplier=tp_atr_multiplier,
        use_stop_loss=use_stop_loss,
        use_take_profit=use_take_profit,
        # Regime-balanced sampling
        regime_labels=regime_labels,
        use_regime_sampling=use_regime_sampling,
        # Volatility Sizing
        volatility_sizing=volatility_sizing,
        risk_pips_target=risk_pips_target,
        # Classification mode
        num_classes=num_classes,
        # Analyst Alignment
        enforce_analyst_alignment=enforce_analyst_alignment,
        # Pre-computed Analyst cache
        precomputed_analyst_cache=precomputed_analyst_cache,
        # Visualization data
        ohlc_data=ohlc_data,
        timestamps=timestamps,
        # Full Eyes Features
        returns=returns,
        agent_lookback_window=getattr(trading_cfg, 'agent_lookback_window', 6) if config is not None else 6,
        include_attention_features=getattr(trading_cfg, 'include_attention_features', False) if config is not None else False,
    )

    return env


def train_agent(
    df_15m,
    df_1h,
    df_4h,
    feature_cols: list,
    analyst_path: str,
    save_path: str,
    component_sequences: Optional[np.ndarray] = None,
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    total_timesteps: int = 500_000,
    resume_path: Optional[str] = None
) -> Tuple[SniperAgent, Dict]:
    """
    Main function to train the PPO Sniper Agent.

    Args:
        df_15m: 15-minute DataFrame with features
        df_1h: 1-hour DataFrame with features
        df_4h: 4-hour DataFrame with features
        feature_cols: Feature columns used
        analyst_path: Path to trained analyst model
        save_path: Path to save agent
        config: Configuration object
        device: Torch device
        component_sequences: Optional precomputed component windows aligned to df_15m index
        total_timesteps: Total training timesteps

    Returns:
        Tuple of (trained agent, training info)
    """
    # Setup logging for this run
    log_dir = Path(save_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(log_dir), name=__name__)

    # Device selection
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    logger.info(f"Training agent on device: {device}")

    # Load frozen analyst
    logger.info(f"Loading analyst from {analyst_path}")
    # Use the same features as Analyst training (11 base + 7 extra = 18 features)
    model_features = [
        'returns', 'volatility', 'pinbar', 'engulfing', 'doji',
        'ema_trend', 'ema_crossover', 'regime', 'sma_distance',
        'dist_to_resistance', 'dist_to_support',
        # Extra features added during training
        'session_asian', 'session_london', 'session_ny',
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
        'top6_momentum', 'top6_dispersion'
    ]
    
    # TCNAnalyst expects true timeframe keys in this system
    feature_dims = {
        '5m': len(model_features),
        '15m': len(model_features),
        '45m': len(model_features)
    }
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)
    logger.info("Analyst loaded and frozen")

    # Log analyst info
    logger.info(f"Analyst context_dim: {analyst.context_dim}")
    logger.info(f"Analyst parameters: {sum(p.numel() for p in analyst.parameters()):,} (frozen)")

    # Prepare data
    logger.info("Preparing environment data...")
    
    # Add Market Sessions
    logger.info("Adding market session features...")
    df_15m = add_market_sessions(df_15m)
    df_1h = add_market_sessions(df_1h)
    df_4h = add_market_sessions(df_4h)

    # Add Structure Features (BOS/CHoCH)
    logger.info("Adding structure features (BOS/CHoCH)...")
    for df in [df_15m, df_1h, df_4h]:
        f_high, f_low = detect_fractals(df)
        struct_df = detect_structure_breaks(df, f_high, f_low)
        for col in struct_df.columns:
            df[col] = struct_df[col]

    # NOTE: Do NOT append session/structure columns to `feature_cols` here.
    # `feature_cols` must match what the Analyst checkpoint was trained on.
    # Session/structure columns can still be used by the agent via `market_features`
    # (see `prepare_env_data()`).

    # NOTE: We now operate on 5m/15m/45m timeframes.
    # train_agent() keeps legacy arg names (df_15m/df_1h/df_4h) for compatibility,
    # but the mapping in the pipeline is:
    #   df_15m -> 5m base
    #   df_1h  -> 15m mid
    #   df_4h  -> 45m high
    lookback_5m = config.analyst.lookback_5m
    lookback_15m = config.analyst.lookback_15m
    lookback_45m = config.analyst.lookback_45m

    seq_path = getattr(config.paths, 'component_sequences', None) if config is not None else None

    # Load component sequences from disk if not provided
    if component_sequences is None and config is not None:
        if getattr(config.analyst, 'use_cross_asset_attention', False) and seq_path is not None and Path(seq_path).exists():
            try:
                from ..data.components import load_component_sequences
                component_sequences, _ = load_component_sequences(seq_path)
            except Exception as e:
                logger.warning(f"Failed to load component sequences for agent training: {e}")
                component_sequences = None

    # Validate/rebuild component sequences if feature dimensions changed
    if (
        config is not None and
        getattr(config.analyst, 'use_cross_asset_attention', False) and
        component_sequences is not None
    ):
        expected_seq_len = getattr(config.analyst, 'component_seq_len', 12)
        expected_input_dim = getattr(config.analyst, 'component_input_dim', 4)
        if (
            component_sequences.ndim != 4 or
            component_sequences.shape[0] != len(df_15m) or
            component_sequences.shape[2] != expected_seq_len or
            component_sequences.shape[3] != expected_input_dim
        ):
            logger.info("Existing component sequences mismatch; recomputing...")
            try:
                from ..data.components import prepare_component_sequences, save_component_sequences

                resample_rule = config.data.timeframes.get('5m', '5min')
                component_sequences = prepare_component_sequences(
                    config.paths.components_dir,
                    df_15m.index,
                    resample_rule=resample_rule,
                    seq_len=expected_seq_len
                )
                if component_sequences is not None and seq_path is not None:
                    save_component_sequences(component_sequences, seq_path)
            except Exception as e:
                logger.warning(f"Failed to recompute component sequences: {e}")
                component_sequences = None

    data_15m, data_1h, data_4h, close_prices, market_features, component_data, returns = prepare_env_data(
        df_15m, df_1h, df_4h, feature_cols,
        lookback_5m, lookback_15m, lookback_45m,
        component_sequences=component_sequences,
        smc_cfg=getattr(config, "smc", None) if config is not None else None,
    )

    logger.info(f"Data shapes: 15m={data_15m.shape}, 1h={data_1h.shape}, 4h={data_4h.shape}")
    logger.info(f"Price range: {close_prices.min():.5f} - {close_prices.max():.5f}")

    # Extract real OHLC data for visualization
    # Calculate start_idx to match prepare_env_data (5m base with 15m/45m subsampling)
    subsample_15m = 3
    subsample_45m = 9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,
        (lookback_45m - 1) * subsample_45m + 1,
    )
    n_samples = len(close_prices)
    
    ohlc_data = None
    timestamps = None
    if all(col in df_15m.columns for col in ['open', 'high', 'low', 'close']):
        ohlc_data = df_15m[['open', 'high', 'low', 'close']].values[start_idx:start_idx + n_samples].astype(np.float32)
        logger.info(f"OHLC data extracted: {ohlc_data.shape}, range: {ohlc_data[:, 3].min():.5f} - {ohlc_data[:, 3].max():.5f}")
        logger.info(f"First 5 OHLC rows:\n{ohlc_data[:5]}") # DEBUG: Check if values are raw or normalized
    
    if df_15m.index.dtype == 'datetime64[ns]' or hasattr(df_15m.index, 'to_pydatetime'):
        try:
            timestamps = (df_15m.index[start_idx:start_idx + n_samples].astype('int64') // 10**9).values
            logger.info(f"Timestamps extracted: {len(timestamps)}")
        except Exception as e:
            logger.warning(f"Failed to extract timestamps: {e}")

    # Split into train/eval
    split_idx = int(0.85 * len(close_prices))
    logger.info(f"Train samples: {split_idx}, Eval samples: {len(close_prices) - split_idx}")

    train_data = (
        data_15m[:split_idx],
        data_1h[:split_idx],
        data_4h[:split_idx],
        close_prices[:split_idx],
        market_features[:split_idx],
        component_data[:split_idx] if component_data is not None else None,
        returns[:split_idx] if returns is not None else None,
    )

    eval_data = (
        data_15m[split_idx:],
        data_1h[split_idx:],
        data_4h[split_idx:],
        close_prices[split_idx:],
        market_features[split_idx:],
        component_data[split_idx:] if component_data is not None else None,
        returns[split_idx:] if returns is not None else None,
    )

    # Split OHLC data for train/eval
    train_ohlc = ohlc_data[:split_idx] if ohlc_data is not None else None
    eval_ohlc = ohlc_data[split_idx:] if ohlc_data is not None else None
    train_timestamps = timestamps[:split_idx] if timestamps is not None else None
    eval_timestamps = timestamps[split_idx:] if timestamps is not None else None

    # Compute market feature normalization stats from TRAINING data only
    # FIXED: This prevents look-ahead bias by using only training statistics
    train_market_features = train_data[4]  # market_features from train_data tuple
    market_feat_mean = train_market_features.mean(axis=0).astype(np.float32)
    market_feat_std = train_market_features.std(axis=0).astype(np.float32)
    market_feat_std = np.where(market_feat_std > 1e-8, market_feat_std, 1.0).astype(np.float32)
    
    logger.info("Market feature normalization stats (from training data):")
    logger.info(f"  Mean: {market_feat_mean}")
    logger.info(f"  Std:  {market_feat_std}")

    # Compute regime labels for balanced sampling (training data only)
    # This ensures agent learns from BULLISH, BEARISH, and RANGING markets equally
    logger.info("Computing regime labels for balanced sampling...")
    # Align regime labels to the same trimmed segment used by the environment.
    env_df_5m = df_15m.iloc[start_idx:start_idx + n_samples]
    train_regime_labels = compute_regime_labels(env_df_5m.iloc[:split_idx], lookback=20)
    regime_counts = {
        'Bullish': (train_regime_labels == 0).sum(),
        'Ranging': (train_regime_labels == 1).sum(),
        'Bearish': (train_regime_labels == 2).sum()
    }
    logger.info(f"Regime distribution: {regime_counts}")

    # Try to load pre-computed Analyst cache for sequential context
    analyst_cache_path = Path(save_path).parent.parent / 'data' / 'processed' / 'analyst_cache.npz'
    train_analyst_cache = None
    eval_analyst_cache = None
    
    if analyst_cache_path.exists():
        logger.info(f"Loading pre-computed Analyst cache from {analyst_cache_path}")
        try:
            from .precompute_analyst import load_cached_analyst_outputs
            full_cache = load_cached_analyst_outputs(str(analyst_cache_path))
            
            # Split cache to match train/eval split
            cache_split_idx = split_idx
            train_analyst_cache = {
                'contexts': full_cache['contexts'][:cache_split_idx],
                'probs': full_cache['probs'][:cache_split_idx],
                # DISABLED ACTIVATIONS TO SAVE MEMORY (OOM Protection)
                'activations_15m': None, 
                'activations_1h': None,
                'activations_4h': None,
            }
            eval_analyst_cache = {
                'contexts': full_cache['contexts'][cache_split_idx:],
                'probs': full_cache['probs'][cache_split_idx:],
                # DISABLED ACTIVATIONS TO SAVE MEMORY (OOM Protection)
                'activations_15m': None,
                'activations_1h': None,
                'activations_4h': None,
            }
            logger.info(f"Using sequential Analyst context: train={len(train_analyst_cache['contexts'])}, eval={len(eval_analyst_cache['contexts'])}")
        except Exception as e:
            logger.warning(f"Failed to load Analyst cache: {e}")
            logger.info("Falling back to standard precomputation")
    else:
        logger.info("No pre-computed Analyst cache found. Using standard precomputation.")
        logger.info(f"To enable sequential context, run: python src/training/precompute_analyst.py")

    # Create training environment
    # ENABLE regime sampling to balance training across market regimes.
    # NOTE: We always pass REAL timestamps (no synthetic timeline).
    use_regime_sampling_train = True  # ENABLED: Critical for balanced directional learning
    if use_regime_sampling_train:
        logger.info("Regime sampling ENABLED: Agent will see 33% Bullish, 33% Ranging, 33% Bearish")

    train_env = create_trading_env(
        data_15m=train_data[0],
        data_1h=train_data[1],
        data_4h=train_data[2],
        close_prices=train_data[3],
        market_features=train_data[4],
        component_data=train_data[5],
        returns=train_data[6], # Pass returns
        analyst_model=analyst,
        config=config,
        device=device,
        market_feat_mean=market_feat_mean,
        market_feat_std=market_feat_std,
        regime_labels=train_regime_labels,
        use_regime_sampling=use_regime_sampling_train,
        precomputed_analyst_cache=train_analyst_cache,
        ohlc_data=train_ohlc,
        timestamps=train_timestamps,
    )

    logger.info("Creating evaluation environment...")
    eval_env = create_trading_env(
        data_15m=eval_data[0],
        data_1h=eval_data[1],
        data_4h=eval_data[2],
        close_prices=eval_data[3],
        market_features=eval_data[4],
        component_data=eval_data[5],
        returns=eval_data[6], # Pass returns
        analyst_model=analyst,
        config=config,
        device=device,
        market_feat_mean=market_feat_mean,  # Use TRAINING stats for eval too
        market_feat_std=market_feat_std,
        regime_labels=None,  # Eval uses random sampling (no regime bias)
        use_regime_sampling=False,
        precomputed_analyst_cache=eval_analyst_cache,  # Use sequential context if available
        ohlc_data=eval_ohlc,            # Real OHLC for visualization
        timestamps=eval_timestamps,      # Real timestamps for visualization
    )

    # Log environment info
    logger.info(f"Observation space: {train_env.observation_space}")
    logger.info(f"Action space: {train_env.action_space}")

    # Wrap environments
    train_env = Monitor(train_env)
    eval_env = Monitor(eval_env)

    # Create agent
    reset_timesteps = True
    remaining_timesteps = total_timesteps

    if resume_path:
        resume_p = Path(str(resume_path)).expanduser()
        if not resume_p.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found (or not a file): {resume_p}\n"
                "Expected a SB3 .zip checkpoint like: models/checkpoints/sniper_model_7400000_steps.zip"
            )

        logger.info(f"Resuming PPO agent from checkpoint: {resume_p}")
        try:
            agent = SniperAgent.load(str(resume_p), env=train_env, device=device)
        except Exception as e:
            logger.error(
                "Failed to resume PPO from checkpoint. Most common causes:\n"
                "- Observation space changed (e.g. added/removed features, changed `agent_lookback_window`, "
                "`include_attention_features`, or market feature columns)\n"
                "- Action space changed\n"
                f"Checkpoint: {resume_p}\n"
                f"Env obs shape: {getattr(train_env.observation_space, 'shape', None)}\n"
                f"Env action space: {train_env.action_space}\n"
                f"Error: {e}"
            )
            raise
        
        # Force-update Exploration Rate (Entropy Coefficient) from current config
        # This allows "shock therapy" (increasing exploration) on resumed models
        if hasattr(config.agent, 'ent_coef'):
            current_ent_coef = getattr(agent.model, 'ent_coef', None)
            new_ent_coef = config.agent.ent_coef
            agent.model.ent_coef = new_ent_coef
            logger.info(f"Exploration Rate (Entropy) UPDATED: {current_ent_coef} -> {new_ent_coef}")
        
        # Calculate remaining steps
        current_timesteps = agent.model.num_timesteps
        # Ensure we run for at least 10k steps if completed or close to completion,
        # otherwise run until total_timesteps is reached.
        # If user wants to EXTEND training, they should increase total_timesteps in config.
        remaining_timesteps = max(10000, total_timesteps - current_timesteps)
        reset_timesteps = False
        
        logger.info(f"Resumed from step {current_timesteps:,}. Target: {total_timesteps:,}.")
        logger.info(f"Remaining timesteps: {remaining_timesteps:,}")
        logger.info("Learning Rate Schedule will CONTINUE from current step (NOT reset).")
        
    else:
        logger.info("Creating PPO agent...")
        agent = create_agent(train_env, config.agent)
        remaining_timesteps = total_timesteps
        reset_timesteps = True

    # Create training logger callback
    training_callback = AgentTrainingLogger(
        log_dir=str(log_dir),
        log_freq=5000,
        verbose=1
    )

    # Train
    logger.info(f"Starting training for {remaining_timesteps:,} timesteps...")
    logger.info("-" * 70)

    training_info = agent.train(
        total_timesteps=remaining_timesteps,
        eval_env=eval_env,
        eval_freq=10_000,
        save_path=save_path,
        callback=training_callback,
        reset_num_timesteps=reset_timesteps
    )

    # Get metrics from callback
    training_info['callback_metrics'] = training_callback.get_metrics()

    # Final evaluation
    logger.info("=" * 70)
    logger.info("Running final evaluation...")
    eval_results = agent.evaluate(eval_env, n_episodes=20)
    training_info['final_eval'] = eval_results

    logger.info("-" * 70)
    logger.info("FINAL EVALUATION RESULTS:")
    logger.info(f"  Mean Reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    logger.info(f"  Mean PnL: {eval_results['mean_pnl']:.2f} pips")
    logger.info(f"  Win Rate: {eval_results.get('win_rate', 0)*100:.1f}%")
    logger.info(f"  Mean Trades per Episode: {eval_results.get('mean_trades', 0):.1f}")
    logger.info("-" * 70)

    # Save training summary
    summary = {
        'total_timesteps': total_timesteps,
        'total_episodes': len(training_callback.episode_rewards),
        'final_mean_reward': eval_results['mean_reward'],
        'final_mean_pnl': eval_results['mean_pnl'],
        'final_win_rate': eval_results.get('win_rate', 0),
        'action_distribution': training_callback.action_counts,
        'avg_episode_length': float(np.mean(training_callback.episode_lengths)) if training_callback.episode_lengths else 0
    }

    summary_path = log_dir / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Training summary saved to: {summary_path}")

    # Cleanup
    train_env.close()
    eval_env.close()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return agent, training_info


def load_and_evaluate(
    agent_path: str,
    df_15m,
    df_1h,
    df_4h,
    feature_cols: list,
    analyst_path: str,
    device: Optional[torch.device] = None,
    n_episodes: int = 50
) -> Dict:
    """
    Load a trained agent and evaluate it.

    Args:
        agent_path: Path to saved agent
        df_*: DataFrames
        feature_cols: Feature columns
        analyst_path: Path to analyst model
        device: Torch device
        n_episodes: Number of evaluation episodes

    Returns:
        Evaluation results
    """
    if device is None:
        device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

    logger.info(f"Loading agent from {agent_path}")
    logger.info(f"Evaluating on {n_episodes} episodes")

    # Load analyst
    feature_dims = {'5m': len(feature_cols), '15m': len(feature_cols), '45m': len(feature_cols)}
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)

    # Prepare data (use last portion as test)
    data_15m, data_1h, data_4h, close_prices, market_features, component_data, returns = prepare_env_data(
        df_15m, df_1h, df_4h, feature_cols
    )

    # Use last 15% as test
    test_start = int(0.85 * len(close_prices))
    test_data = (
        data_15m[test_start:],
        data_1h[test_start:],
        data_4h[test_start:],
        close_prices[test_start:],
        market_features[test_start:],
        component_data[test_start:] if component_data is not None else None,
    )
    test_returns = returns[test_start:] if returns is not None else None

    # Create test environment
    test_env = create_trading_env(*test_data, analyst_model=analyst, device=device, returns=test_returns)
    test_env = Monitor(test_env)

    # Load agent
    agent = SniperAgent.load(agent_path, test_env, device='cpu')  # SB3 more stable on CPU

    # Evaluate
    logger.info("Running evaluation...")
    results = agent.evaluate(test_env, n_episodes=n_episodes)

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS:")
    logger.info(f"  Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    logger.info(f"  Mean PnL: {results['mean_pnl']:.2f} pips")
    logger.info(f"  Win Rate: {results.get('win_rate', 0)*100:.1f}%")
    logger.info("=" * 70)

    test_env.close()
    return results


if __name__ == '__main__':
    print("Use this module via: python -m src.training.train_agent")
    print("Or import and call train_agent() function")
