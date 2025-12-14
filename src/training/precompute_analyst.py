#!/usr/bin/env python3
"""
Pre-compute Analyst outputs for sequential context.

This script runs the Analyst model through the ENTIRE dataset sequentially,
ensuring it sees the full historical context at each timestep. The outputs
(context vectors and probabilities) are cached to disk for use during PPO training.

This fixes the "train-test mismatch" issue where the Analyst only saw
small random windows during PPO training, leading to unreliable predictions.

Usage:
    python src/training/precompute_analyst.py --analyst-path models/analyst/best.pt

The cached outputs are saved to: data/processed/analyst_cache.npz
"""

import sys
from pathlib import Path
import argparse
import logging
import numpy as np
import torch
import pandas as pd
import gc
from tqdm import tqdm
from typing import Optional, Tuple, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, get_device, clear_memory
from src.models.analyst import load_analyst
from src.data.features import get_feature_columns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def precompute_analyst_outputs(
    analyst_path: str,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame,
    feature_cols: list,
    component_sequences: Optional[np.ndarray] = None,
    lookback_5m: int = 48,
    lookback_15m: int = 16,
    lookback_45m: int = 6,
    device: torch.device = None,
    batch_size: int = 64,
    save_path: str = None,
) -> dict:
    """
    Pre-compute Analyst outputs by running SEQUENTIALLY through all data.
    
    This ensures the Analyst sees continuous historical context, not random fragments.
    
    Args:
        analyst_path: Path to trained Analyst model
        df_5m, df_15m, df_45m: Multi-timeframe DataFrames
        feature_cols: Feature columns to use
        component_sequences: Optional precomputed component windows aligned to df_5m index
        lookback_*: Lookback windows per timeframe
        device: Torch device
        batch_size: Batch size for inference
        save_path: Where to save the cached outputs
        
    Returns:
        Dict with 'contexts' and 'probs' arrays
    """
    if device is None:
        device = get_device()
    
    # Load Analyst model (TCN uses true timeframe keys)
    feature_dims = {
        '5m': len(feature_cols),
        '15m': len(feature_cols),
        '45m': len(feature_cols)
    }
    analyst = load_analyst(analyst_path, feature_dims, device, freeze=True)
    analyst.eval()
    
    # Extract features as numpy arrays
    features_5m = df_5m[feature_cols].values.astype(np.float32)
    features_15m = df_15m[feature_cols].values.astype(np.float32)
    features_45m = df_45m[feature_cols].values.astype(np.float32)

    # Get close prices for reference
    close_prices = df_5m['close'].values.astype(np.float32)

    # FIXED: Subsampling factors for multi-timeframe alignment
    # 15m data is aligned to 5m index, so we subsample every 3rd bar
    # 45m data is aligned to 5m index, so we subsample every 9th bar
    subsample_15m = 3
    subsample_45m = 9

    # FIXED: Calculate valid start index accounting for subsampling
    # Need enough bars for: 5m lookback, 15m lookback*3, 45m lookback*9
    start_idx = max(
        lookback_5m,
        (lookback_15m - 1) * subsample_15m + 1,  # For 16 15m candles: need 46 bars
        (lookback_45m - 1) * subsample_45m + 1   # For 6 45m candles: need 46 bars
    )
    n_samples = len(features_5m) - start_idx

    logger.info(f"Pre-computing Analyst outputs for {n_samples:,} timesteps...")
    logger.info(f"Start index: {start_idx} (after lookback warmup with subsampling)")
    logger.info(f"Subsampling: 15m every {subsample_15m} bars, 45m every {subsample_45m} bars")

    # Prepare windowed data (pre-window for each index)
    # This creates [n_samples, lookback, features] arrays
    logger.info("Creating windowed data arrays...")

    data_5m = np.zeros((n_samples, lookback_5m, len(feature_cols)), dtype=np.float32)
    data_15m = np.zeros((n_samples, lookback_15m, len(feature_cols)), dtype=np.float32)
    data_45m = np.zeros((n_samples, lookback_45m, len(feature_cols)), dtype=np.float32)

    for i in range(n_samples):
        idx = start_idx + i

        # 5m: direct slicing (include current candle)
        data_5m[i] = features_5m[idx - lookback_5m + 1:idx + 1]

        # FIXED: 15m - subsample every 3rd bar, including current candle
        idx_range_15m = list(range(
            idx - (lookback_15m - 1) * subsample_15m,
            idx + 1,
            subsample_15m
        ))
        data_15m[i] = features_15m[idx_range_15m]

        # FIXED: 45m - subsample every 9th bar, including current candle
        idx_range_45m = list(range(
            idx - (lookback_45m - 1) * subsample_45m,
            idx + 1,
            subsample_45m
        ))
        data_45m[i] = features_45m[idx_range_45m]

    # Component data (optional, already windowed per index)
    component_data = None
    use_component_data = False
    if component_sequences is not None:
        try:
            expected_input_dim = getattr(analyst, 'component_input_dim', None)
            if component_sequences.ndim != 4:
                logger.warning(
                    f"Component sequences rank mismatch (ndim={component_sequences.ndim}); "
                    "precomputing without components."
                )
            elif component_sequences.shape[0] != len(df_5m):
                logger.warning(
                    f"Component sequences length mismatch ({component_sequences.shape[0]} vs {len(df_5m)}); "
                    "precomputing without components."
                )
            elif expected_input_dim is not None and component_sequences.shape[3] != expected_input_dim:
                logger.warning(
                    f"Component sequences feature-dim mismatch ({component_sequences.shape[3]} vs {expected_input_dim}); "
                    "precomputing without components."
                )
            else:
                component_data = component_sequences[start_idx:start_idx + n_samples].astype(np.float32)
                use_component_data = getattr(analyst, 'cross_asset_module', None) is not None
                logger.info(f"Component data enabled for precompute: {component_data.shape}")
        except Exception as e:
            logger.warning(f"Failed to prepare component data: {e}")
            component_data = None
            use_component_data = False
    
    logger.info(f"Windowed data shapes: 5m={data_5m.shape}, 15m={data_15m.shape}, 45m={data_45m.shape}")
    
    # Run Analyst in batches (SEQUENTIALLY - order matters for cumulative context)
    all_contexts = []
    all_probs = []
    all_activations = {'5m': [], '15m': [], '45m': []}
    
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_i = batch_idx * batch_size
            end_i = min(start_i + batch_size, n_samples)
            
            # Get batch data
            batch_5m = torch.tensor(data_5m[start_i:end_i], device=device)
            batch_15m = torch.tensor(data_15m[start_i:end_i], device=device)
            batch_45m = torch.tensor(data_45m[start_i:end_i], device=device)
            batch_components = None
            if use_component_data and component_data is not None:
                batch_components = torch.tensor(component_data[start_i:end_i], device=device)
            
            # Get Analyst outputs (including activations for visualization)
            if hasattr(analyst, 'get_activations'):
                if use_component_data and batch_components is not None:
                    context, activations = analyst.get_activations(
                        batch_5m, batch_15m, batch_45m, component_data=batch_components
                    )
                else:
                    context, activations = analyst.get_activations(batch_5m, batch_15m, batch_45m)
                
                # Get probs separately
                if use_component_data and batch_components is not None:
                    res = analyst.get_probabilities(
                        batch_5m, batch_15m, batch_45m, component_data=batch_components
                    )
                else:
                    res = analyst.get_probabilities(batch_5m, batch_15m, batch_45m)

                if isinstance(res, (tuple, list)) and len(res) == 3:
                    _, probs, _ = res
                else:
                    _, probs = res
                
                for k in all_activations:
                    all_activations[k].append(activations[k].cpu().numpy())
            elif hasattr(analyst, 'get_probabilities'):
                if use_component_data and batch_components is not None:
                    result = analyst.get_probabilities(
                        batch_5m, batch_15m, batch_45m, component_data=batch_components
                    )
                else:
                    result = analyst.get_probabilities(batch_5m, batch_15m, batch_45m)
                if len(result) == 3:
                    context, probs, _ = result
                else:
                    context, probs = result
            else:
                if use_component_data and batch_components is not None:
                    context = analyst.get_context(
                        batch_5m, batch_15m, batch_45m, component_data=batch_components
                    )
                else:
                    context = analyst.get_context(batch_5m, batch_15m, batch_45m)
                # Dummy probs
                probs = torch.ones(len(context), 3, device=device) / 3
            
            all_contexts.append(context.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
            # Memory cleanup
            del batch_5m, batch_15m, batch_45m, batch_components, context, probs
            if batch_idx % 50 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
    
    # Combine all batches
    contexts = np.vstack(all_contexts).astype(np.float32)
    probs = np.vstack(all_probs).astype(np.float32)
    
    # Combine activations
    activations_5m = np.vstack(all_activations['5m']).astype(np.float32) if all_activations['5m'] else None
    activations_15m = np.vstack(all_activations['15m']).astype(np.float32) if all_activations['15m'] else None
    activations_45m = np.vstack(all_activations['45m']).astype(np.float32) if all_activations['45m'] else None
    
    logger.info(f"Final shapes: contexts={contexts.shape}, probs={probs.shape}")
    
    # Create output dict
    output = {
        'contexts': contexts,
        'probs': probs,
        'close_prices': close_prices[start_idx:],
        'start_idx': start_idx,
        'lookback_5m': lookback_5m,
        'lookback_15m': lookback_15m,
        'lookback_45m': lookback_45m,
        'n_samples': n_samples,
        'feature_cols': feature_cols,
        'use_cross_asset_attention': use_component_data,
    }
    
    # Save to disk
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            contexts=contexts,
            probs=probs,
            activations_5m=activations_5m,
            activations_15m=activations_15m,
            activations_45m=activations_45m,
            close_prices=close_prices[start_idx:],
            start_idx=start_idx,
            lookback_5m=lookback_5m,
            lookback_15m=lookback_15m,
            lookback_45m=lookback_45m,
            use_cross_asset_attention=use_component_data,
        )
        logger.info(f"Saved cached Analyst outputs to: {save_path}")
    
    return output


def load_cached_analyst_outputs(cache_path: str) -> dict:
    """
    Load pre-computed Analyst outputs from disk.
    
    Args:
        cache_path: Path to the .npz file
        
    Returns:
        Dict with 'contexts', 'probs', and metadata
    """
    data = np.load(cache_path, allow_pickle=True)
    return {
        'contexts': data['contexts'],
        'probs': data['probs'],
        'activations_5m': data.get('activations_5m'),
        'activations_15m': data.get('activations_15m'),
        'activations_45m': data.get('activations_45m'),
        'close_prices': data['close_prices'],
        'start_idx': int(data['start_idx']),
        'lookback_5m': int(data['lookback_5m']),
        'lookback_15m': int(data['lookback_15m']),
        'lookback_45m': int(data['lookback_45m']),
    }


def main():
    """Main entry point for pre-computation script."""
    parser = argparse.ArgumentParser(description='Pre-compute Analyst outputs for PPO training')
    parser.add_argument('--analyst-path', type=str, default=None,
                       help='Path to trained Analyst model (default: models/analyst/best.pt)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for cached data (default: data/processed/analyst_cache.npz)')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for inference')
    args = parser.parse_args()
    
    config = Config()
    device = get_device()
    
    # Set default paths
    analyst_path = args.analyst_path or str(config.paths.models_analyst / 'best.pt')
    output_path = args.output or str(config.paths.data_processed / 'analyst_cache.npz')
    
    # Check if analyst model exists
    if not Path(analyst_path).exists():
        logger.error(f"Analyst model not found: {analyst_path}")
        logger.info("Please train the Analyst first, or specify --analyst-path")
        sys.exit(1)
    
    # Load normalized data
    logger.info("Loading normalized data...")
    try:
        df_5m = pd.read_parquet(config.paths.data_processed / 'features_5m_normalized.parquet')
        df_15m = pd.read_parquet(config.paths.data_processed / 'features_15m_normalized.parquet')
        df_45m = pd.read_parquet(config.paths.data_processed / 'features_45m_normalized.parquet')
    except FileNotFoundError:
        logger.error("Normalized data not found. Run the pipeline first to generate it.")
        sys.exit(1)
    
    # Feature columns - MUST match what the saved Analyst checkpoint expects.
    # IMPORTANT: Order matters (must match training-time feature order).
    feature_cols = [
        # Base features
        'returns', 'volatility',
        'pinbar', 'engulfing', 'doji',
        'ema_trend', 'ema_crossover',
        'regime', 'sma_distance',
        'dist_to_resistance', 'dist_to_support',
        # Market structure
        'bos_bullish', 'bos_bearish', 'choch_bullish', 'choch_bearish',
        # Top 6 NAS100 components (legacy scalars)
        'top6_momentum', 'top6_dispersion',
        # Market sessions (kept raw; added after normalization in training)
        'session_asian', 'session_london', 'session_ny',
    ]
    
    # Check which features are available and add missing ones
    from src.data.features import add_market_sessions, detect_fractals, detect_structure_breaks
    
    # Add market sessions if missing
    if 'session_asian' not in df_5m.columns:
        logger.info("Adding market session features...")
        df_5m = add_market_sessions(df_5m)
        df_15m = add_market_sessions(df_15m)
        df_45m = add_market_sessions(df_45m)
    
    # Add structure breaks if missing
    if 'bos_bullish' not in df_5m.columns:
        logger.info("Adding structure break features...")
        for df in [df_5m, df_15m, df_45m]:
            f_high, f_low = detect_fractals(df)
            struct_df = detect_structure_breaks(df, f_high, f_low)
            for col in struct_df.columns:
                df[col] = struct_df[col]
    
    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df_5m.columns]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Load component sequences if cross-asset attention is enabled
    component_sequences = None
    if config.analyst.use_cross_asset_attention and config.paths.component_sequences.exists():
        try:
            from src.data.components import (
                load_component_sequences,
                prepare_component_sequences,
                save_component_sequences,
            )
            component_sequences, _ = load_component_sequences(config.paths.component_sequences)
            expected_seq_len = config.analyst.component_seq_len
            expected_input_dim = config.analyst.component_input_dim
            if (
                component_sequences is None or
                component_sequences.ndim != 4 or
                component_sequences.shape[0] != len(df_5m) or
                component_sequences.shape[2] != expected_seq_len or
                component_sequences.shape[3] != expected_input_dim
            ):
                logger.info("Existing component sequences mismatch; recomputing...")
                component_sequences = prepare_component_sequences(
                    config.paths.components_dir,
                    df_5m.index,
                    resample_rule=config.data.timeframes['5m'],
                    seq_len=expected_seq_len
                )
                if component_sequences is not None:
                    save_component_sequences(component_sequences, config.paths.component_sequences)

            if component_sequences is not None:
                logger.info(f"Loaded component sequences for precompute: {component_sequences.shape}")
        except Exception as e:
            logger.warning(f"Failed to load component sequences, continuing without: {e}")
    
    # Pre-compute
    output = precompute_analyst_outputs(
        analyst_path=analyst_path,
        df_5m=df_5m,
        df_15m=df_15m,
        df_45m=df_45m,
        feature_cols=feature_cols,
        component_sequences=component_sequences,
        device=device,
        batch_size=args.batch_size,
        save_path=output_path,
    )
    
    logger.info("=" * 60)
    logger.info("PRE-COMPUTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Cached {output['n_samples']:,} timesteps")
    logger.info(f"Context shape: {output['contexts'].shape}")
    logger.info(f"Probs shape: {output['probs'].shape}")
    logger.info(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
