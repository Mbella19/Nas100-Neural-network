"""
Feature engineering module for the hybrid trading system.

Implements:
- Price Action Patterns: Pinbar, Engulfing, Doji
- Market Structure: Fractal S/R, Distance to S/R
- Trend Filters: SMA distance, EMA crossovers
- Regime Detection: Choppiness Index, ADX
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Price Action Patterns
# =============================================================================

def detect_pinbar(
    df: pd.DataFrame,
    wick_ratio: float = 2.0
) -> pd.Series:
    """
    Detect pinbar candles (rejection candles with long wicks).

    A bullish pinbar has a lower wick > 2x body.
    A bearish pinbar has an upper wick > 2x body.

    Args:
        df: OHLCV DataFrame
        wick_ratio: Minimum wick-to-body ratio

    Returns:
        Series with values: 1 (bullish), -1 (bearish), 0 (none)
    """
    body = abs(df['close'] - df['open'])
    upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - df['low']

    # Avoid division by zero
    body = body.replace(0, 1e-10)

    bullish_pinbar = (lower_wick / body > wick_ratio) & (lower_wick > upper_wick)
    bearish_pinbar = (upper_wick / body > wick_ratio) & (upper_wick > lower_wick)

    result = pd.Series(0, index=df.index, dtype=np.float32)
    result[bullish_pinbar] = 1
    result[bearish_pinbar] = -1

    return result


def detect_engulfing(df: pd.DataFrame) -> pd.Series:
    """
    Detect engulfing patterns.

    Bullish engulfing: Current green candle fully engulfs previous red candle.
    Bearish engulfing: Current red candle fully engulfs previous green candle.

    Returns:
        Series with values: 1 (bullish), -1 (bearish), 0 (none)
    """
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    prev_body_high = df[['open', 'close']].shift(1).max(axis=1)
    prev_body_low = df[['open', 'close']].shift(1).min(axis=1)

    curr_body_high = df[['open', 'close']].max(axis=1)
    curr_body_low = df[['open', 'close']].min(axis=1)

    # Previous candle direction
    prev_bearish = prev_close < prev_open
    prev_bullish = prev_close > prev_open

    # Current candle direction
    curr_bullish = df['close'] > df['open']
    curr_bearish = df['close'] < df['open']

    # Engulfing conditions
    bullish_engulfing = (
        prev_bearish &
        curr_bullish &
        (curr_body_low < prev_body_low) &
        (curr_body_high > prev_body_high)
    )

    bearish_engulfing = (
        prev_bullish &
        curr_bearish &
        (curr_body_high > prev_body_high) &
        (curr_body_low < prev_body_low)
    )

    result = pd.Series(0, index=df.index, dtype=np.float32)
    result[bullish_engulfing] = 1
    result[bearish_engulfing] = -1

    return result


def detect_doji(
    df: pd.DataFrame,
    body_ratio: float = 0.1
) -> pd.Series:
    """
    Detect doji candles (indecision candles with tiny bodies).

    Args:
        df: OHLCV DataFrame
        body_ratio: Maximum body-to-range ratio for doji

    Returns:
        Series with 1 (doji) or 0 (not doji)
    """
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']

    # Avoid division by zero
    total_range = total_range.replace(0, 1e-10)

    is_doji = (body / total_range) < body_ratio

    return is_doji.astype(np.float32)


# =============================================================================
# Market Structure
# =============================================================================

def detect_fractals(
    df: pd.DataFrame,
    n: int = 5
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect Williams Fractals for Support/Resistance levels.

    A fractal high is a high that is higher than n bars before and after.
    A fractal low is a low that is lower than n bars before and after.

    Args:
        df: OHLCV DataFrame
        n: Number of bars on each side

    Returns:
        Tuple of (fractal_highs, fractal_lows) as boolean Series
    """
    half_n = n // 2

    fractal_highs = pd.Series(False, index=df.index)
    fractal_lows = pd.Series(False, index=df.index)

    for i in range(half_n, len(df) - half_n):
        # Check if current high is highest in window
        window_high = df['high'].iloc[i - half_n:i + half_n + 1]
        if df['high'].iloc[i] == window_high.max():
            fractal_highs.iloc[i] = True

        # Check if current low is lowest in window
        window_low = df['low'].iloc[i - half_n:i + half_n + 1]
        if df['low'].iloc[i] == window_low.min():
            fractal_lows.iloc[i] = True

    return fractal_highs, fractal_lows


def get_sr_levels(
    df: pd.DataFrame,
    fractal_window: int = 5,
    lookback: int = 100
) -> Tuple[List[float], List[float]]:
    """
    Get current Support and Resistance levels from recent fractals.

    Returns:
        Tuple of (resistance_levels, support_levels)
    """
    fractal_highs, fractal_lows = detect_fractals(df.tail(lookback), fractal_window)

    resistance = df.loc[fractal_highs[fractal_highs].index, 'high'].tolist()
    support = df.loc[fractal_lows[fractal_lows].index, 'low'].tolist()

    return resistance, support


def distance_to_nearest_sr(
    price: pd.Series,
    df: pd.DataFrame,
    atr: pd.Series,
    fractal_window: int = 5,
    lookback: int = 100
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate ATR-normalized distance to nearest S/R levels.

    Returns:
        Tuple of (distance_to_resistance, distance_to_support)
    """
    dist_to_r = pd.Series(np.nan, index=price.index, dtype=np.float32)
    dist_to_s = pd.Series(np.nan, index=price.index, dtype=np.float32)

    fractal_highs, fractal_lows = detect_fractals(df, fractal_window)

    for i in range(lookback, len(price)):
        # Get recent fractals
        recent_highs = df.loc[fractal_highs.iloc[max(0, i-lookback):i], 'high']
        recent_lows = df.loc[fractal_lows.iloc[max(0, i-lookback):i], 'low']

        current_price = price.iloc[i]
        current_atr = atr.iloc[i]

        if current_atr > 0:
            # Distance to nearest resistance (above current price)
            resistances = recent_highs[recent_highs > current_price]
            if len(resistances) > 0:
                dist_to_r.iloc[i] = (resistances.min() - current_price) / current_atr

            # Distance to nearest support (below current price)
            supports = recent_lows[recent_lows < current_price]
            if len(supports) > 0:
                dist_to_s.iloc[i] = (current_price - supports.max()) / current_atr

    return dist_to_r.fillna(0), dist_to_s.fillna(0)


# =============================================================================
# Trend Filters
# =============================================================================

def sma(df: pd.DataFrame, period: int = 200) -> pd.Series:
    """Simple Moving Average."""
    return df['close'].rolling(window=period).mean().astype(np.float32)


def ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return df['close'].ewm(span=period, adjust=False).mean().astype(np.float32)


def sma_distance(
    df: pd.DataFrame,
    atr: pd.Series,
    period: int = 200
) -> pd.Series:
    """
    Calculate ATR-normalized distance from SMA(200).

    Positive = price above SMA (bullish)
    Negative = price below SMA (bearish)
    """
    sma_val = sma(df, period)
    distance = (df['close'] - sma_val) / atr.replace(0, 1e-10)
    return distance.astype(np.float32)


def ema_crossover(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26
) -> pd.Series:
    """
    EMA crossover signal.

    Returns: 1 (bullish cross), -1 (bearish cross), 0 (no cross)
    """
    ema_fast = ema(df, fast)
    ema_slow = ema(df, slow)

    # Current and previous relationship
    above_now = ema_fast > ema_slow
    above_prev = ema_fast.shift(1) > ema_slow.shift(1)

    result = pd.Series(0, index=df.index, dtype=np.float32)
    result[above_now & ~above_prev] = 1   # Bullish crossover
    result[~above_now & above_prev] = -1  # Bearish crossover

    return result


def ema_trend(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26
) -> pd.Series:
    """
    EMA trend direction (not just crossover).

    Returns: 1 (fast > slow), -1 (fast < slow)
    """
    ema_fast = ema(df, fast)
    ema_slow = ema(df, slow)

    result = pd.Series(0, index=df.index, dtype=np.float32)
    result[ema_fast > ema_slow] = 1
    result[ema_fast < ema_slow] = -1

    return result


# =============================================================================
# Regime Detection
# =============================================================================

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()

    return atr_val.astype(np.float32)


def choppiness_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Choppiness Index (CHOP).

    Values > 61.8 indicate ranging/choppy market.
    Values < 38.2 indicate trending market.
    """
    atr_val = atr(df, 1)  # True Range for each bar
    atr_sum = atr_val.rolling(window=period).sum()

    high_max = df['high'].rolling(window=period).max()
    low_min = df['low'].rolling(window=period).min()

    chop = 100 * np.log10(atr_sum / (high_max - low_min + 1e-10)) / np.log10(period)

    return chop.astype(np.float32)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index (ADX).

    Values > 25 indicate trending market.
    Values < 20 indicate ranging market.
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # +DM and -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

    # True Range
    atr_val = atr(df, period)

    # Smoothed +DI and -DI
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, 1e-10))

    # DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_val = dx.ewm(span=period, adjust=False).mean()

    return adx_val.astype(np.float32)


def market_regime(
    df: pd.DataFrame,
    chop_period: int = 14,
    adx_period: int = 14
) -> pd.Series:
    """
    Classify market regime based on CHOP and ADX.

    Returns:
        1: Trending (tradeable)
        0: Neutral
        -1: Ranging/Choppy (avoid)
    """
    chop = choppiness_index(df, chop_period)
    adx_val = adx(df, adx_period)

    result = pd.Series(0, index=df.index, dtype=np.float32)

    # Trending: low chop + high ADX
    trending = (chop < 38.2) | (adx_val > 25)
    # Ranging: high chop + low ADX
    ranging = (chop > 61.8) & (adx_val < 20)

    result[trending] = 1
    result[ranging] = -1

    return result


# =============================================================================
# Complete Feature Engineering
# =============================================================================

def engineer_all_features(
    df: pd.DataFrame,
    config: Optional[dict] = None
) -> pd.DataFrame:
    """
    Apply all feature engineering to OHLCV DataFrame.

    Args:
        df: OHLCV DataFrame
        config: Optional configuration dict

    Returns:
        DataFrame with all features added
    """
    if config is None:
        config = {
            'pinbar_wick_ratio': 2.0,
            'doji_body_ratio': 0.1,
            'fractal_window': 5,
            'sr_lookback': 100,
            'sma_period': 200,
            'ema_fast': 12,
            'ema_slow': 26,
            'chop_period': 14,
            'adx_period': 14,
            'atr_period': 14
        }

    result = df.copy()
    logger.info("Engineering features...")

    # ATR (needed for other features)
    result['atr'] = atr(df, config['atr_period'])

    # Price Action Patterns
    result['pinbar'] = detect_pinbar(df, config['pinbar_wick_ratio'])
    result['engulfing'] = detect_engulfing(df)
    result['doji'] = detect_doji(df, config['doji_body_ratio'])

    # Trend Filters
    result['sma_distance'] = sma_distance(df, result['atr'], config['sma_period'])
    result['ema_crossover'] = ema_crossover(df, config['ema_fast'], config['ema_slow'])
    result['ema_trend'] = ema_trend(df, config['ema_fast'], config['ema_slow'])

    # Regime Detection
    result['chop'] = choppiness_index(df, config['chop_period'])
    result['adx'] = adx(df, config['adx_period'])
    result['regime'] = market_regime(df, config['chop_period'], config['adx_period'])

    # Returns (for normalization and targets)
    result['returns'] = df['close'].pct_change().astype(np.float32)

    # Volatility
    result['volatility'] = result['returns'].rolling(20).std().astype(np.float32)

    # Drop NaN rows (from lookback calculations)
    initial_len = len(result)
    result = result.dropna()
    logger.info(f"Features complete. Dropped {initial_len - len(result)} NaN rows. Final: {len(result):,} rows")

    # Ensure all float32
    for col in result.columns:
        if result[col].dtype == np.float64:
            result[col] = result[col].astype(np.float32)

    return result


def create_smoothed_target(
    df: pd.DataFrame,
    future_window: int = 12,
    smooth_window: int = 12
) -> pd.Series:
    """
    Create the smoothed future return target for Analyst training.

    Target = (smoothed future close / current close) - 1

    This teaches the model sustained momentum, not noise.

    Args:
        df: DataFrame with 'close' column
        future_window: How many candles ahead
        smooth_window: Rolling window for smoothing

    Returns:
        Series of target values
    """
    future_smoothed = df['close'].shift(-future_window).rolling(smooth_window).mean()
    target = (future_smoothed / df['close']) - 1

    return target.astype(np.float32)


def get_feature_columns(include_ohlcv: bool = False) -> List[str]:
    """Get list of feature column names."""
    features = [
        'atr', 'pinbar', 'engulfing', 'doji',
        'sma_distance', 'ema_crossover', 'ema_trend',
        'chop', 'adx', 'regime', 'returns', 'volatility'
    ]
    if include_ohlcv:
        features = ['open', 'high', 'low', 'close', 'volume'] + features
    return features
