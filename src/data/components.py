"""
Component stock integration for NAS100 analysis.

Handles loading, aligning, and feature engineering for the top 6 weighted stocks:
AAPL, MSFT, NVDA, AMZN, GOOG/L, AVGO.

This module provides two levels of integration:
1. Scalar features (legacy): top6_momentum, top6_dispersion - simple aggregates
2. Sequence data (new): Full temporal sequences for cross-asset attention module
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

DATETIME_COLUMNS = ('datetime', 'timestamp', 'time', 'date')

# Ordered list of primary components for sequence data
# This order is used consistently throughout the system
PRIMARY_COMPONENTS = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOG', 'AVGO']

# Approximate weights in NAS100 (as of late 2024/2025)
# Used to construct the "Top 6 Index" signal
COMPONENT_WEIGHTS = {
    'AAPL': 0.087,
    'MSFT': 0.085,
    'NVDA': 0.082,
    'AMZN': 0.052,
    'GOOG': 0.025,  # Class C
    'GOOGL': 0.025, # Class A (treated same as GOOG)
    'AVGO': 0.044,
    'META': 0.048,  # Included if available
    'TSLA': 0.035,  # Included if available
}

def load_component_data(
    data_dir: Path,
    mappings: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load component stock CSVs from directory.
    
    Args:
        data_dir: Directory containing component CSVs
        mappings: Dict mapping Ticker -> Filename (optional)
                  If None, tries to auto-detect based on weights keys.
                  
    Returns:
        Dict of Ticker -> DataFrame (1m OHLC)
    """
    if mappings is None:
        # Auto-detect files
        files = list(data_dir.glob("*.csv"))
        mappings = {}
        for f in files:
            # Simple heuristic: filename starts with Ticker
            name_parts = f.name.split('_')[0].upper()
            if name_parts in COMPONENT_WEIGHTS:
                mappings[name_parts] = f.name
                
    loaded_data = {}

    def _select_usecols(path: Path, required_cols: List[str]) -> Tuple[str, List[str]]:
        header = pd.read_csv(path, nrows=0)
        cols = list(header.columns)
        if not cols:
            raise ValueError(f"No columns found in CSV: {path}")

        lower_map = {str(c).strip().lower(): c for c in cols}

        dt_col = None
        for candidate in DATETIME_COLUMNS:
            if candidate in lower_map:
                dt_col = lower_map[candidate]
                break
        if dt_col is None:
            dt_col = cols[0]

        missing = [c for c in required_cols if c not in lower_map]
        if missing:
            raise ValueError(f"Missing required columns: {set(missing)}")

        usecols = [dt_col] + [lower_map[c] for c in required_cols]
        return str(dt_col), list(dict.fromkeys(usecols))
    
    for ticker, filename in mappings.items():
        file_path = data_dir / filename
        if not file_path.exists():
            logger.warning(f"Component file not found: {file_path}")
            continue
            
        try:
            dt_col, usecols = _select_usecols(file_path, required_cols=['close'])
            df = pd.read_csv(file_path, usecols=usecols)
            
            # Normalize columns
            df.columns = df.columns.str.lower()
            
            dt_col = dt_col.strip().lower()
            if dt_col not in df.columns:
                # Defensive fallback
                dt_col = next((c for c in df.columns if c in DATETIME_COLUMNS), None)
                if dt_col is None:
                    continue
                
            # Parse datetime
            df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
            df[dt_col] = df[dt_col].dt.tz_localize(None) # Remove timezone for alignment
            
            # Set index
            df.set_index(dt_col, inplace=True)
            df.sort_index(inplace=True)
            
            # Keep only Close prices for correlation/momentum
            if 'close' in df.columns:
                loaded_data[ticker] = df[['close']].astype(np.float32)
                
            logger.info(f"Loaded {ticker}: {len(df):,} rows")
            
        except Exception as e:
            logger.error(f"Error loading {ticker}: {e}")
            
    return loaded_data

def merge_component_features(
    main_df: pd.DataFrame,
    components_dir: str | Path,
    resample_rule: str = '5min'  # Default base timeframe
) -> pd.DataFrame:
    """
    Merge component features into the main dataframe.
    
    Calculates:
    - top6_momentum: Weighted average return of components
    - top6_dispersion: Std dev of component returns
    - top6_rel_strength: Component momentum - Index momentum
    
    Args:
        main_df: Main NAS100 DataFrame (already resampled to 5m, 15m or 45m)
        components_dir: Directory with component CSVs
        resample_rule: Timeframe rule to resample components to '5min', '15min', etc.
        
    Returns:
        DataFrame with added component features
    """
    components_dir = Path(components_dir)
    if not components_dir.exists():
        logger.warning(f"Components directory not found: {components_dir}")
        return main_df
        
    logger.info(f"Merging component features from {components_dir}...")
    
    # 1. Load raw 1m component data
    # (We load 1m first then resample to match main_df to ensure alignment)
    components = load_component_data(components_dir)
    
    if not components:
        logger.warning("No valid component data loaded.")
        return main_df
        
    # 2. Resample and Calculate Returns
    # We need to align everything to the main_df index
    aligned_returns = pd.DataFrame(index=main_df.index)
    weights = {}
    
    for ticker, df in components.items():
        # Resample to target timeframe
        # We use 'last' for close price
        # Make sure mapping '5min' -> '5min', '15min' -> '15min' works
        # The key is to match the index of main_df
        
        # Reindex to main_df to ensure precise alignment
        # (ffill to propagate last known price if missing in that specific bin)
        # Note: This assumes main_df and component df are both sorted datetime indices
        
        # Better approach: Resample component df to the rule, then reindex
        # Match main resampling (label='right', closed='left') to avoid alignment drift
        resampled = df['close'].resample(resample_rule, label='right', closed='left').last().dropna()
        
        # ALIGNMENT FIX:
        # 1. Reindex to main_df's timeline.
        # 2. Use 'ffill' with a limit (e.g., 12 bars = 1 hour for 5m data) to bridge small gaps.
        # 3. If a component is missing data (e.g. IPOs later, or halted), it will be NaN.
        # 4. We later fill final NaN returns with 0.0 (neutral impact).
        aligned_close = resampled.reindex(main_df.index, method='ffill', limit=12) # 1 hour for 5m
        
        # Compute Percentage Change (Returns)
        ret = aligned_close.pct_change(fill_method=None)
        
        # Use only if we have at least some valid overlap
        if ret.count() > 0:
            aligned_returns[ticker] = ret
            weights[ticker] = COMPONENT_WEIGHTS.get(ticker, 0.01) # Default low weight
            
    if aligned_returns.empty:
        return main_df
        
    # 3. Compute Composite Features
    
    # Normalize weights to sum to 1.0 (for the subsets we have)
    total_weight = sum(weights.values())
    if total_weight > 0:
        norm_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        norm_weights = {k: 1.0 / len(weights) for k in weights}
        
    # Weighted Momentum (Index of the components)
    weighted_moms = []
    for ticker, weight in norm_weights.items():
        if ticker in aligned_returns.columns:
            weighted_moms.append(aligned_returns[ticker] * weight)
            
    if weighted_moms:
        # Sum of weighted returns
        top6_mom = pd.concat(weighted_moms, axis=1).sum(axis=1)
        
        # Dispersion (Standard deviation across the component returns at each timestep)
        # Higher dispersion = components fighting (choppy/conflicted)
        # Lower dispersion = components aligned (strong trend)
        top6_disp = aligned_returns.std(axis=1)
        
        # Add to result
        result = main_df.copy()
        
        # Fill NaNs with 0 (no signal)
        result['top6_momentum'] = top6_mom.fillna(0).astype(np.float32)
        result['top6_dispersion'] = top6_disp.fillna(0).astype(np.float32)
        
        logger.info(f"Added columns: top6_momentum, top6_dispersion using {list(weights.keys())}")
        return result

    return main_df


def load_component_ohlcv(
    data_dir: Path,
    mappings: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load OHLC data for component stocks (not just close).

    Used for preparing sequence data for the cross-asset attention module.

    Args:
        data_dir: Directory containing component CSVs
        mappings: Dict mapping Ticker -> Filename (optional)

    Returns:
        Dict of Ticker -> DataFrame with columns [open, high, low, close]
    """
    if mappings is None:
        files = list(data_dir.glob("*.csv"))
        mappings = {}
        for f in files:
            name_parts = f.name.split('_')[0].upper()
            if name_parts in COMPONENT_WEIGHTS:
                mappings[name_parts] = f.name

    loaded_data = {}

    for ticker, filename in mappings.items():
        file_path = data_dir / filename
        if not file_path.exists():
            logger.warning(f"Component file not found: {file_path}")
            continue

        try:
            # Read only datetime + OHLC columns (ignore any other columns)
            header = pd.read_csv(file_path, nrows=0)
            cols = list(header.columns)
            if not cols:
                continue

            lower_map = {str(c).strip().lower(): c for c in cols}
            dt_col = None
            for candidate in DATETIME_COLUMNS:
                if candidate in lower_map:
                    dt_col = lower_map[candidate]
                    break
            if dt_col is None:
                dt_col = cols[0]

            required_cols = ['open', 'high', 'low', 'close']
            if not all(c in lower_map for c in required_cols):
                continue

            usecols = [dt_col] + [lower_map[c] for c in required_cols]
            usecols = list(dict.fromkeys(usecols))
            df = pd.read_csv(file_path, usecols=usecols)
            df.columns = df.columns.str.lower()

            dt_col = str(dt_col).strip().lower()
            if dt_col not in df.columns:
                dt_col = next((c for c in df.columns if c in DATETIME_COLUMNS), None)
                if dt_col is None:
                    continue

            df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
            df[dt_col] = df[dt_col].dt.tz_localize(None)

            df.set_index(dt_col, inplace=True)
            df.sort_index(inplace=True)

            # Keep OHLCV columns
            ohlc_cols = ['open', 'high', 'low', 'close']
            available_cols = [c for c in ohlc_cols if c in df.columns]

            if 'close' in available_cols:
                loaded_data[ticker] = df[available_cols].astype(np.float32)
                logger.info(f"Loaded {ticker} OHLC: {len(df):,} rows, columns: {available_cols}")

        except Exception as e:
            logger.error(f"Error loading {ticker} OHLCV: {e}")

    return loaded_data


def compute_component_returns(
    df: pd.DataFrame,
    columns: List[str] = ['open', 'high', 'low', 'close']
) -> pd.DataFrame:
    """
    Compute percentage returns for OHLC columns.

    For price columns (OHLC), uses standard percentage change.

    Args:
        df: DataFrame with OHLCV columns
        columns: Columns to compute returns for

    Returns:
        DataFrame with {col}_ret columns
    """
    result = pd.DataFrame(index=df.index)

    available = [c for c in columns if c in df.columns]

    for col in available:
        # Standard percentage returns for prices
        result[f'{col}_ret'] = df[col].pct_change(fill_method=None).fillna(0).astype(np.float32)

    # Clip extreme values to prevent gradient explosion
    for col in result.columns:
        result[col] = result[col].clip(-0.5, 0.5)

    return result


def prepare_component_sequences(
    components_dir: str | Path,
    main_index: pd.DatetimeIndex,
    resample_rule: str = '5min',
    seq_len: int = 12,
    component_order: List[str] = None
) -> np.ndarray:
    """
    Prepare windowed component sequences for cross-asset attention module.

    Creates a 4D array of component OHLCV returns aligned to the main DataFrame index.
    Each sample gets a lookback window of seq_len bars for all 6 components.

    Args:
        components_dir: Directory containing component CSVs
        main_index: DatetimeIndex to align component data to
        resample_rule: Timeframe for resampling ('5min', '15min', etc.)
        seq_len: Lookback window length
        component_order: Order of components (default: PRIMARY_COMPONENTS)

    Returns:
        np.ndarray of shape [n_samples, n_components, seq_len, n_features]
        where n_features = 4 (open_ret, high_ret, low_ret, close_ret)

    Note:
        Returns array starts from index seq_len-1 to allow full lookback.
        Caller must align this with their data starting from the same offset.
    """
    components_dir = Path(components_dir)
    if not components_dir.exists():
        logger.warning(f"Components directory not found: {components_dir}")
        return None

    if component_order is None:
        component_order = PRIMARY_COMPONENTS

    logger.info(f"Preparing component sequences: {len(main_index):,} samples, seq_len={seq_len}")

    # 1. Load full OHLCV data
    raw_components = load_component_ohlcv(components_dir)

    if not raw_components:
        logger.warning("No component data loaded for sequences.")
        return None

    # 2. Resample and align each component to main index
    aligned_returns = {}

    for ticker in component_order:
        if ticker not in raw_components:
            logger.warning(f"Component {ticker} not available, will use zeros")
            aligned_returns[ticker] = None
            continue

        df = raw_components[ticker]

        # Resample to target timeframe
        # For OHLC: use appropriate aggregation
        ohlc_agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
        }

        # Filter to available columns
        available_agg = {k: v for k, v in ohlc_agg.items() if k in df.columns}

        resampled = df.resample(resample_rule, label='right', closed='left').agg(available_agg).dropna()

        # Align to main index with forward fill (limit to prevent stale data)
        aligned = resampled.reindex(main_index, method='ffill', limit=12)

        # Compute returns
        returns = compute_component_returns(aligned)
        aligned_returns[ticker] = returns

        logger.info(f"  {ticker}: {returns.notna().sum().iloc[0]:,} valid samples")

    # 3. Build sequence array
    n_samples = len(main_index)
    n_components = len(component_order)
    n_features = 4  # open_ret, high_ret, low_ret, close_ret

    # Output array (will be trimmed by caller based on valid start index)
    sequences = np.zeros((n_samples, n_components, seq_len, n_features), dtype=np.float32)

    # Feature column order
    feature_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret']

    for comp_idx, ticker in enumerate(component_order):
        ret_df = aligned_returns[ticker]

        if ret_df is None:
            # No data for this component - leave as zeros
            continue

        # Ensure all feature columns exist (fill missing with zeros)
        for col in feature_cols:
            if col not in ret_df.columns:
                ret_df[col] = 0.0

        # Convert to numpy for faster windowing
        ret_array = ret_df[feature_cols].values  # [n_samples, n_features]

        # Create windowed sequences
        for i in range(seq_len - 1, n_samples):
            sequences[i, comp_idx, :, :] = ret_array[i - seq_len + 1:i + 1, :]

    logger.info(f"Component sequences shape: {sequences.shape}")
    logger.info(f"  Valid samples start at index {seq_len - 1}")

    return sequences


def save_component_sequences(
    sequences: np.ndarray,
    save_path: str | Path,
    component_order: List[str] = None
) -> None:
    """
    Save component sequences to disk.

    Args:
        sequences: Array of shape [n_samples, n_components, seq_len, n_features]
        save_path: Path to save file (will use .npz format)
        component_order: List of component tickers for metadata
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if component_order is None:
        component_order = PRIMARY_COMPONENTS

    np.savez_compressed(
        save_path,
        sequences=sequences,
        component_order=np.array(component_order),
        feature_order=np.array(['open_ret', 'high_ret', 'low_ret', 'close_ret'])
    )

    logger.info(f"Saved component sequences to {save_path} ({sequences.nbytes / 1e6:.1f} MB)")


def load_component_sequences(load_path: str | Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load component sequences from disk.

    Args:
        load_path: Path to .npz file

    Returns:
        Tuple of (sequences array, component order list)
    """
    load_path = Path(load_path)

    if not load_path.exists():
        raise FileNotFoundError(f"Component sequences file not found: {load_path}")

    data = np.load(load_path)
    sequences = data['sequences']
    component_order = data['component_order'].tolist()

    logger.info(f"Loaded component sequences: {sequences.shape} from {load_path}")

    return sequences, component_order
