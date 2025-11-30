"""
Training script for the PPO Sniper Agent.

Trains the RL agent using a frozen Market Analyst to provide
context vectors for decision making.

Memory-optimized for Apple M2 Silicon.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
import gc

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from ..models.analyst import load_analyst, MarketAnalyst
from ..environments.trading_env import TradingEnv
from ..agents.sniper_agent import SniperAgent, create_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_env_data(
    df_15m,
    df_1h,
    df_4h,
    feature_cols: list,
    lookback_15m: int = 48,
    lookback_1h: int = 24,
    lookback_4h: int = 12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare windowed data for the trading environment.

    Returns:
        Tuple of (data_15m, data_1h, data_4h, close_prices, market_features)
    """
    # Calculate valid range
    start_idx = max(lookback_15m, lookback_1h * 4, lookback_4h * 16)
    n_samples = len(df_15m) - start_idx - 1

    logger.info(f"Preparing {n_samples} samples for environment")

    # Prepare windowed data
    data_15m = np.zeros((n_samples, lookback_15m, len(feature_cols)), dtype=np.float32)
    data_1h = np.zeros((n_samples, lookback_1h, len(feature_cols)), dtype=np.float32)
    data_4h = np.zeros((n_samples, lookback_4h, len(feature_cols)), dtype=np.float32)

    features_15m = df_15m[feature_cols].values.astype(np.float32)
    features_1h = df_1h[feature_cols].values.astype(np.float32)
    features_4h = df_4h[feature_cols].values.astype(np.float32)

    for i in range(n_samples):
        actual_idx = start_idx + i
        data_15m[i] = features_15m[actual_idx - lookback_15m:actual_idx]
        data_1h[i] = features_1h[actual_idx - lookback_1h:actual_idx]
        data_4h[i] = features_4h[actual_idx - lookback_4h:actual_idx]

    # Close prices
    close_prices = df_15m['close'].values[start_idx:start_idx + n_samples].astype(np.float32)

    # Market features for reward shaping
    market_cols = ['atr', 'chop', 'adx', 'regime', 'sma_distance']
    available_cols = [c for c in market_cols if c in df_15m.columns]

    if len(available_cols) > 0:
        market_features = df_15m[available_cols].values[start_idx:start_idx + n_samples].astype(np.float32)
    else:
        # Create dummy features if not available
        market_features = np.zeros((n_samples, 5), dtype=np.float32)
        market_features[:, 0] = 0.001  # Default ATR
        market_features[:, 1] = 50.0   # Default CHOP
        market_features[:, 2] = 20.0   # Default ADX

    return data_15m, data_1h, data_4h, close_prices, market_features


def create_trading_env(
    data_15m: np.ndarray,
    data_1h: np.ndarray,
    data_4h: np.ndarray,
    close_prices: np.ndarray,
    market_features: np.ndarray,
    analyst_model: Optional[MarketAnalyst] = None,
    config: Optional[object] = None,
    device: Optional[torch.device] = None
) -> TradingEnv:
    """
    Create the trading environment.

    Args:
        data_*: Prepared window data
        close_prices: Close prices for PnL
        market_features: Features for reward shaping
        analyst_model: Frozen Market Analyst
        config: TradingConfig
        device: Torch device

    Returns:
        TradingEnv instance
    """
    # Default configuration
    spread_pips = 1.5
    fomo_penalty = -0.5
    chop_penalty = -0.3
    fomo_threshold_atr = 2.0
    chop_threshold = 60.0
    max_steps = 2000
    context_dim = 64

    if config is not None:
        spread_pips = getattr(config, 'spread_pips', spread_pips)
        fomo_penalty = getattr(config, 'fomo_penalty', fomo_penalty)
        chop_penalty = getattr(config, 'chop_penalty', chop_penalty)
        fomo_threshold_atr = getattr(config, 'fomo_threshold_atr', fomo_threshold_atr)
        chop_threshold = getattr(config, 'chop_threshold', chop_threshold)
        max_steps = getattr(config, 'max_steps_per_episode', max_steps)

    if analyst_model is not None:
        context_dim = analyst_model.context_dim

    env = TradingEnv(
        data_15m=data_15m,
        data_1h=data_1h,
        data_4h=data_4h,
        close_prices=close_prices,
        market_features=market_features,
        analyst_model=analyst_model,
        context_dim=context_dim,
        spread_pips=spread_pips,
        fomo_penalty=fomo_penalty,
        chop_penalty=chop_penalty,
        fomo_threshold_atr=fomo_threshold_atr,
        chop_threshold=chop_threshold,
        max_steps=max_steps,
        device=device
    )

    return env


def train_agent(
    df_15m,
    df_1h,
    df_4h,
    feature_cols: list,
    analyst_path: str,
    save_path: str,
    config: Optional[object] = None,
    device: Optional[torch.device] = None,
    total_timesteps: int = 500_000
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
        total_timesteps: Total training timesteps

    Returns:
        Tuple of (trained agent, training info)
    """
    # Device selection
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    logger.info(f"Training agent on device: {device}")

    # Load frozen analyst
    logger.info(f"Loading analyst from {analyst_path}")
    feature_dims = {
        '15m': len(feature_cols),
        '1h': len(feature_cols),
        '4h': len(feature_cols)
    }
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)
    logger.info("Analyst loaded and frozen")

    # Prepare data
    logger.info("Preparing environment data...")
    lookback_15m = 48
    lookback_1h = 24
    lookback_4h = 12

    data_15m, data_1h, data_4h, close_prices, market_features = prepare_env_data(
        df_15m, df_1h, df_4h, feature_cols,
        lookback_15m, lookback_1h, lookback_4h
    )

    # Split into train/eval
    split_idx = int(0.85 * len(close_prices))

    train_data = (
        data_15m[:split_idx],
        data_1h[:split_idx],
        data_4h[:split_idx],
        close_prices[:split_idx],
        market_features[:split_idx]
    )

    eval_data = (
        data_15m[split_idx:],
        data_1h[split_idx:],
        data_4h[split_idx:],
        close_prices[split_idx:],
        market_features[split_idx:]
    )

    # Create environments
    logger.info("Creating training environment...")
    train_env = create_trading_env(
        *train_data,
        analyst_model=analyst,
        config=config,
        device=device
    )

    logger.info("Creating evaluation environment...")
    eval_env = create_trading_env(
        *eval_data,
        analyst_model=analyst,
        config=config,
        device=device
    )

    # Wrap environments
    train_env = Monitor(train_env)
    eval_env = Monitor(eval_env)

    # Create agent
    logger.info("Creating PPO agent...")
    agent = create_agent(train_env, config)

    # Train
    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    training_info = agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=10_000,
        save_path=save_path
    )

    # Final evaluation
    logger.info("Running final evaluation...")
    eval_results = agent.evaluate(eval_env, n_episodes=20)
    training_info['final_eval'] = eval_results

    logger.info(f"Training complete. Final mean reward: {eval_results['mean_reward']:.2f}")
    logger.info(f"Mean PnL: {eval_results['mean_pnl']:.2f} pips")

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

    # Load analyst
    feature_dims = {'15m': len(feature_cols), '1h': len(feature_cols), '4h': len(feature_cols)}
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)

    # Prepare data (use last portion as test)
    data_15m, data_1h, data_4h, close_prices, market_features = prepare_env_data(
        df_15m, df_1h, df_4h, feature_cols
    )

    # Use last 15% as test
    test_start = int(0.85 * len(close_prices))
    test_data = (
        data_15m[test_start:],
        data_1h[test_start:],
        data_4h[test_start:],
        close_prices[test_start:],
        market_features[test_start:]
    )

    # Create test environment
    test_env = create_trading_env(*test_data, analyst_model=analyst, device=device)
    test_env = Monitor(test_env)

    # Load agent
    agent = SniperAgent.load(agent_path, test_env, device='cpu')  # SB3 more stable on CPU

    # Evaluate
    results = agent.evaluate(test_env, n_episodes=n_episodes)

    test_env.close()
    return results


if __name__ == '__main__':
    print("Use this module via: python -m src.training.train_agent")
    print("Or import and call train_agent() function")
