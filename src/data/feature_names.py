"""
Centralized feature name lists (strings only).

This module is intentionally lightweight (no pandas/numpy imports) so it can be
used by both training code and the live MT5 bridge without heavy dependencies.
"""

from __future__ import annotations

from typing import Sequence, Tuple


SMC_FEATURE_COLUMNS: Tuple[str, ...] = (
    # Fair Value Gaps (FVG)
    "fvg_direction",
    "fvg_size",
    "dist_to_fvg",
    "fvg_count_up",
    "fvg_count_down",
    # Order Blocks (OB)
    "ob_direction",
    "ob_strength",
    "dist_to_ob",
    "price_in_ob",
    # Liquidity pools / sweeps + prev day/week levels
    "sweep_signal",
    "liquidity_above",
    "liquidity_below",
    "pdh_dist",
    "pdl_dist",
    "pwh_dist",
    "pwl_dist",
    # Premium/Discount + OTE
    "premium_discount",
    "in_ote_zone",
    "ote_distance",
    "swing_retracement",
    # Displacement candles
    "displacement",
    "displacement_direction",
    "body_wick_ratio",
    "momentum_score",
    # Buying/Selling pressure proxies (OHLC-only)
    "close_position",
    "buying_pressure_15m",
    "buying_pressure_45m",
    "delta_proxy",
    "cumulative_delta",
    # Session/Killzones + session levels
    "in_killzone",
    "killzone_type",
    "asian_high_dist",
    "asian_low_dist",
)


AGENT_TF_BASE_FEATURE_COLUMNS: Tuple[str, ...] = (
    # Reward-critical first (TradingEnv assumes atr=0, chop=1)
    "atr",
    "chop",
    "adx",
    "regime",
    "sma_distance",
    "dist_to_support",
    "dist_to_resistance",
    "session_asian",
    "session_london",
    "session_ny",
    "bos_bullish",
    "bos_bearish",
    "choch_bullish",
    "choch_bearish",
    # Extra model features (agent visibility across timeframes)
    "returns",
    "volatility",
    "pinbar",
    "engulfing",
    "doji",
    "ema_trend",
    "ema_crossover",
    "top6_momentum",
    "top6_dispersion",
)


def with_suffix(cols: Sequence[str], suffix: str) -> Tuple[str, ...]:
    return tuple(f"{c}_{suffix}" for c in cols)


AGENT_MARKET_FEATURE_COLUMNS: Tuple[str, ...] = (
    # 5m (base, unsuffixed)
    *AGENT_TF_BASE_FEATURE_COLUMNS,
    *SMC_FEATURE_COLUMNS,
    # 15m
    *with_suffix(AGENT_TF_BASE_FEATURE_COLUMNS, "15m"),
    *with_suffix(SMC_FEATURE_COLUMNS, "15m"),
    # 45m
    *with_suffix(AGENT_TF_BASE_FEATURE_COLUMNS, "45m"),
    *with_suffix(SMC_FEATURE_COLUMNS, "45m"),
    # 1m (resampled to 5m grid)
    *with_suffix(SMC_FEATURE_COLUMNS, "1m"),
)

