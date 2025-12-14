"""
Multi-timeframe resampling module.

Resamples 1-minute OHLC data to 5m, 15m, and 45m timeframes
with proper gap handling via forward-fill on complete datetime index.

CRITICAL: Uses label='right' + closed='left' to prevent look-ahead bias!
- closed='left': Bins are [10:00, 11:00) - includes 10:00, excludes 11:00
- label='right': Row labeled 11:00 contains 10:00-10:59 data

At 10:05 (5m candle), we can only know the COMPLETED 10:00-10:05 candle (labeled 10:05 with right),
not the in-progress candle. This prevents future data leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_complete_index(
    start: datetime,
    end: datetime,
    freq: str = '1min'
) -> pd.DatetimeIndex:
    """
    Create a complete datetime index without gaps.

    Args:
        start: Start datetime
        end: End datetime
        freq: Frequency string (e.g., '1min', '15min', '1H', '4H')

    Returns:
        Complete DatetimeIndex
    """
    return pd.date_range(start=start, end=end, freq=freq)


def resample_ohlcv(
    df: pd.DataFrame,
    freq: str,
    fill_gaps: bool = True
) -> pd.DataFrame:
    """
    Resample OHLC data to a higher timeframe.

    CRITICAL: Uses label='right' to prevent look-ahead bias!
    - The timestamp represents when the candle COMPLETED, not when it started.
    - This ensures that at time T, we only have access to candles that
      completed at or before time T (no future data leakage).

    Example with 1H resample:
    - Data from 10:00-10:59 is labeled at 11:00 (when it completed)
    - At 10:30, forward-fill gives us the 10:00 candle (9:00-9:59 data) - CORRECT
    - Without label='right', at 10:30 we'd get 10:00 candle with 10:00-10:59 data - WRONG

    Args:
        df: DataFrame with datetime index and OHLCV columns
        freq: Target frequency (e.g., '15min', '1H', '4H')
        fill_gaps: Whether to forward-fill gaps

    Returns:
        Resampled DataFrame
    """
    # OHLC resampling rules
    ohlc_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
    }

    # Select only columns that exist
    rules = {col: rule for col, rule in ohlc_rules.items() if col in df.columns}

    # Resample with label='right' + closed='left' to prevent look-ahead bias
    # closed='left': interval is [10:00, 11:00) - includes 10:00, excludes 11:00
    # label='right': timestamp is END of period (when candle completes)
    resampled = df.resample(freq, label='right', closed='left').agg(rules)

    if fill_gaps:
        # Create complete index and reindex
        complete_idx = create_complete_index(
            resampled.index.min(),
            resampled.index.max(),
            freq
        )
        resampled = resampled.reindex(complete_idx)

        # Forward-fill missing values
        resampled = resampled.ffill()
    else:
        # If NOT filling gaps, drop the empty bins created by resample()
        # These are the weekend gaps we want to remove
        resampled = resampled.dropna()

    # Ensure float32
    for col in resampled.columns:
        resampled[col] = resampled[col].astype(np.float32)

    logger.info(f"Resampled to {freq}: {len(resampled):,} rows")

    return resampled


def resample_all_timeframes(
    df_1m: pd.DataFrame,
    timeframes: Optional[Dict[str, str]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Resample 1-minute data to all required timeframes.

    Args:
        df_1m: 1-minute OHLCV DataFrame
        timeframes: Dict mapping names to frequencies
                   Default: {'5m': '5min', '15m': '15min', '45m': '45min'}

    Returns:
        Dictionary of resampled DataFrames
    """
    if timeframes is None:
        timeframes = {
            '5m': '5min',
            '15m': '15min',
            '45m': '45min'
        }

    result = {}
    for name, freq in timeframes.items():
        logger.info(f"Resampling to {name} ({freq})...")
        # Disable gap filling to remove weekend flat lines
        result[name] = resample_ohlcv(df_1m, freq, fill_gaps=False)

    return result


def align_timeframes(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_45m: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align multiple timeframe DataFrames to have matching indices.

    The alignment is done by forward-filling higher timeframe data
    onto the 5m index (the fastest timeframe).

    CRITICAL: Prevents look-ahead bias via label='right' resampling!
    - Higher timeframe candles are labeled at their COMPLETION time
    - Forward-fill ensures at time T, we only see candles that completed at/before T

    Example at 10:10 (5m candle):
    - 15m candles labeled: 10:00 (9:45-10:00), 10:15 (10:00-10:15)
    - Forward-fill at 10:10 gives us the 10:00 candle (9:45-10:00 data) ✓ CORRECT
    - Without label='right', we'd get 10:15 candle with 10:00-10:15 data ✗ WRONG

    Args:
        df_5m: 5-minute DataFrame (base timeframe)
        df_15m: 15-minute DataFrame
        df_45m: 45-minute DataFrame

    Returns:
        Tuple of aligned DataFrames (5m, 15m, 45m)
    """
    # Find common date range
    start = max(df_5m.index.min(), df_15m.index.min(), df_45m.index.min())
    end = min(df_5m.index.max(), df_15m.index.max(), df_45m.index.max())

    # Filter to common range
    df_5m = df_5m.loc[start:end].copy()
    df_15m = df_15m.loc[start:end].copy()
    df_45m = df_45m.loc[start:end].copy()

    # Align 15m and 45m to 5m index by reindexing and forward-filling
    # This ensures each 5m candle has corresponding higher TF context
    df_15m_aligned = df_15m.reindex(df_5m.index, method='ffill')
    df_45m_aligned = df_45m.reindex(df_5m.index, method='ffill')

    # Drop any rows with NaN (at the start before first higher TF candle)
    valid_mask = ~(df_15m_aligned.isna().any(axis=1) | df_45m_aligned.isna().any(axis=1))
    df_5m = df_5m.loc[valid_mask].copy()
    df_15m_aligned = df_15m_aligned.loc[valid_mask].copy()
    df_45m_aligned = df_45m_aligned.loc[valid_mask].copy()

    logger.info(f"Aligned timeframes: {len(df_5m):,} rows from {start} to {end}")

    return df_5m, df_15m_aligned, df_45m_aligned


def create_multi_timeframe_dataset(
    df_1m: pd.DataFrame,
    timeframes: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete pipeline: resample and align all timeframes.

    Args:
        df_1m: 1-minute OHLCV DataFrame
        timeframes: Optional custom timeframe mapping

    Returns:
        Tuple of aligned (5m, 15m, 45m) DataFrames
    """
    # Resample to all timeframes
    resampled = resample_all_timeframes(df_1m, timeframes)

    # Align timeframes
    df_5m, df_15m, df_45m = align_timeframes(
        resampled['5m'],
        resampled['15m'],
        resampled['45m']
    )

    return df_5m, df_15m, df_45m


def get_lookback_data(
    df: pd.DataFrame,
    idx: int,
    lookback: int
) -> pd.DataFrame:
    """
    Get lookback window of data ending at idx.

    Args:
        df: DataFrame
        idx: Current index position
        lookback: Number of rows to look back

    Returns:
        DataFrame slice with lookback rows
    """
    start_idx = max(0, idx - lookback + 1)
    return df.iloc[start_idx:idx + 1]
