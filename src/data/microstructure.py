"""
Smart Money Concepts (SMC) microstructure features (OHLC-only).

These features are designed for the RL agent observation/reward shaping only.
They should NOT be added to the Analyst model input feature set.

All computations avoid look-ahead bias:
- Pivot (swing) points are confirmed with a delay (fractal-style).
- Session highs/lows use running values during the session and only use the
  final session range after the session has completed.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from config.settings import SMCConfig
from .feature_names import SMC_FEATURE_COLUMNS


def _require_ohlc(df: pd.DataFrame) -> None:
    missing = [c for c in ("open", "high", "low", "close") if c not in df.columns]
    if missing:
        raise ValueError(f"SMC features require columns: {missing}")


def _get_atr(df: pd.DataFrame) -> np.ndarray:
    """
    ATR is expected to be present from the main feature pipeline.
    If missing, compute a simple ATR(14) to keep SMC features usable standalone.
    """
    if "atr" in df.columns:
        return df["atr"].values.astype(np.float32, copy=False)

    high = df["high"].values.astype(np.float32, copy=False)
    low = df["low"].values.astype(np.float32, copy=False)
    close_prev = df["close"].shift(1).values.astype(np.float32, copy=False)

    tr1 = high - low
    tr2 = np.abs(high - close_prev)
    tr3 = np.abs(low - close_prev)
    tr = np.nanmax(np.stack([tr1, tr2, tr3], axis=1), axis=1)
    return pd.Series(tr, index=df.index).rolling(14, min_periods=1).mean().values.astype(np.float32)


def _safe_div(numer: np.ndarray, denom: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom_safe = np.where(np.abs(denom) > eps, denom, eps).astype(np.float32)
    return (numer / denom_safe).astype(np.float32)


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(arr)
        .rolling(window=window, min_periods=1)
        .mean()
        .values.astype(np.float32)
    )


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(arr)
        .rolling(window=window, min_periods=2)
        .std()
        .fillna(0.0)
        .values.astype(np.float32)
    )


def _detect_pivots_confirmed(
    high: pd.Series,
    low: pd.Series,
    length: int,
    atol: float = 1e-6,
) -> Tuple[pd.Series, pd.Series]:
    """
    Detect pivot highs/lows with confirmation delay (fractal-style).

    At time t, a pivot at t - half is confirmed by comparing it to the last
    `length` bars ending at t. This prevents look-ahead bias.
    """
    if length < 3 or length % 2 == 0:
        raise ValueError("swing_length must be an odd integer >= 3")

    half = length // 2

    rolling_max = high.rolling(window=length, min_periods=length).max()
    rolling_min = low.rolling(window=length, min_periods=length).min()

    cand_high = high.shift(half)
    cand_low = low.shift(half)

    edge_left_high = high.shift(length - 1)  # first element in rolling window
    edge_right_high = high  # last element in rolling window (current bar)
    edge_left_low = low.shift(length - 1)
    edge_right_low = low

    is_max = (cand_high - rolling_max).abs() <= atol
    is_min = (cand_low - rolling_min).abs() <= atol

    # Reduce duplicate pivots on flat highs/lows by requiring strict edge separation.
    piv_high = is_max & (cand_high > edge_left_high) & (cand_high > edge_right_high)
    piv_low = is_min & (cand_low < edge_left_low) & (cand_low < edge_right_low)

    return piv_high.fillna(False), piv_low.fillna(False)


def _previous_period_levels(
    df: pd.DataFrame,
    period: str,
) -> Tuple[pd.Series, pd.Series]:
    if not isinstance(df.index, pd.DatetimeIndex):
        nan = pd.Series(np.nan, index=df.index, dtype=np.float32)
        return nan, nan

    key = df.index.to_period(period)
    highs = df["high"].groupby(key).max().shift(1)
    lows = df["low"].groupby(key).min().shift(1)
    prev_high = pd.Series(key, index=df.index).map(highs).astype(np.float32)
    prev_low = pd.Series(key, index=df.index).map(lows).astype(np.float32)
    return prev_high, prev_low


def _asian_session_levels(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    Asian session high/low (00:00-09:00 UTC).

    - During the Asian session: use running high/low (no look-ahead).
    - After the session ends: carry the final session high/low forward.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        nan = pd.Series(np.nan, index=df.index, dtype=np.float32)
        return nan, nan

    hours = df.index.hour
    asian = (hours >= 0) & (hours < 9)
    day = df.index.normalize()

    high = df["high"].astype(np.float32)
    low = df["low"].astype(np.float32)

    asian_high_running = high.where(asian).groupby(day).cummax()
    asian_low_running = low.where(asian).groupby(day).cummin()

    asian_high_final = high.where(asian).groupby(day).transform("max")
    asian_low_final = low.where(asian).groupby(day).transform("min")

    asian_high = asian_high_running.copy()
    asian_low = asian_low_running.copy()
    asian_high.loc[~asian] = asian_high_final.loc[~asian]
    asian_low.loc[~asian] = asian_low_final.loc[~asian]

    return asian_high.astype(np.float32), asian_low.astype(np.float32)


def _killzones(
    df: pd.DataFrame,
    cfg: SMCConfig,
) -> Tuple[pd.Series, pd.Series]:
    if not isinstance(df.index, pd.DatetimeIndex):
        zeros = pd.Series(0.0, index=df.index, dtype=np.float32)
        return zeros, zeros

    hours = df.index.hour
    l0, l1 = cfg.london_killzone_hours
    n0, n1 = cfg.ny_killzone_hours

    in_london = (hours >= l0) & (hours < l1)
    in_ny = (hours >= n0) & (hours < n1)

    in_kz = (in_london | in_ny).astype(np.float32)
    kz_type = (
        in_london.astype(np.int32) * 1
        + in_ny.astype(np.int32) * 2
    ).astype(np.float32)

    return pd.Series(in_kz, index=df.index, dtype=np.float32), pd.Series(kz_type, index=df.index, dtype=np.float32)


def add_smc_microstructure_features(
    df: pd.DataFrame,
    cfg: Optional[SMCConfig] = None,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add OHLC-only SMC microstructure features for the RL agent.

    This function is safe to call on already-normalized DataFrames, as it uses
    only raw OHLC + raw ATR (kept unnormalized by the pipeline) for calculations.
    """
    if cfg is None:
        cfg = SMCConfig()

    _require_ohlc(df)
    out = df if inplace else df.copy()

    open_ = out["open"].values.astype(np.float32, copy=False)
    high = out["high"].values.astype(np.float32, copy=False)
    low = out["low"].values.astype(np.float32, copy=False)
    close = out["close"].values.astype(np.float32, copy=False)
    atr = _get_atr(out)
    atr_safe = np.where(atr > 1e-8, atr, 1.0).astype(np.float32)

    n = len(out)
    if n < 10:
        for col in SMC_FEATURE_COLUMNS:
            out[col] = np.zeros(n, dtype=np.float32)
        return out

    # ---------------------------------------------------------------------
    # Pivots / swing structure (shared dependency)
    # ---------------------------------------------------------------------
    piv_high, piv_low = _detect_pivots_confirmed(
        out["high"].astype(np.float32),
        out["low"].astype(np.float32),
        length=int(cfg.swing_length),
    )
    piv_high_arr = piv_high.values
    piv_low_arr = piv_low.values
    half = int(cfg.swing_length) // 2

    # Pivot levels known at confirmation time (value from the pivot bar).
    piv_high_level = pd.Series(np.where(piv_high, out["high"].shift(half), np.nan), index=out.index, dtype=np.float32)
    piv_low_level = pd.Series(np.where(piv_low, out["low"].shift(half), np.nan), index=out.index, dtype=np.float32)

    piv_high_pos = pd.Series(np.where(piv_high, np.arange(n) - half, np.nan), index=out.index, dtype=np.float32)
    piv_low_pos = pd.Series(np.where(piv_low, np.arange(n) - half, np.nan), index=out.index, dtype=np.float32)

    last_high = piv_high_level.ffill()
    last_low = piv_low_level.ffill()
    last_high_pos = piv_high_pos.ffill()
    last_low_pos = piv_low_pos.ffill()

    # Swing direction: 1 if last swing point is a high, -1 if last is a low.
    swing_dir = np.where(
        (last_high_pos.notna() & last_low_pos.notna()) & (last_high_pos.values > last_low_pos.values),
        1.0,
        np.where(
            (last_high_pos.notna() & last_low_pos.notna()) & (last_high_pos.values < last_low_pos.values),
            -1.0,
            0.0,
        ),
    ).astype(np.float32)

    swing_high = last_high.values.astype(np.float32)
    swing_low = last_low.values.astype(np.float32)
    swing_range = (swing_high - swing_low).astype(np.float32)
    swing_range_safe = np.where(swing_range > 1e-8, swing_range, np.nan).astype(np.float32)

    # ---------------------------------------------------------------------
    # Fair Value Gaps (FVG) - tracked as active zones until first touch
    # ---------------------------------------------------------------------
    fvg_direction = np.zeros(n, dtype=np.float32)
    fvg_size = np.zeros(n, dtype=np.float32)
    dist_to_fvg = np.zeros(n, dtype=np.float32)
    fvg_count_up = np.zeros(n, dtype=np.float32)
    fvg_count_down = np.zeros(n, dtype=np.float32)

    active_fvgs: List[Tuple[int, float, float, int, float]] = []
    max_age = int(cfg.fvg_max_age_bars)
    max_active = int(cfg.fvg_max_active)
    min_size_atr = float(cfg.fvg_min_size_atr)

    for i in range(n):
        # 1) remove mitigated / expired
        if active_fvgs:
            new_active: List[Tuple[int, float, float, int, float]] = []
            lo = float(low[i])
            hi = float(high[i])
            for d, z_lo, z_hi, created, size_atr in active_fvgs:
                if i - created > max_age:
                    continue
                # Mitigation = any overlap with the zone.
                if lo <= z_hi and hi >= z_lo:
                    continue
                new_active.append((d, z_lo, z_hi, created, size_atr))
            active_fvgs = new_active[-max_active:]

        # 2) add new gap at i (uses only current/past bars)
        if i >= 2:
            # Bullish FVG: current low > high two bars ago
            if low[i] > high[i - 2]:
                z_lo = float(high[i - 2])
                z_hi = float(low[i])
                size_atr = float((z_hi - z_lo) / atr_safe[i])
                if size_atr >= min_size_atr:
                    active_fvgs.append((1, z_lo, z_hi, i, size_atr))
            # Bearish FVG: current high < low two bars ago
            elif high[i] < low[i - 2]:
                z_lo = float(high[i])
                z_hi = float(low[i - 2])
                size_atr = float((z_hi - z_lo) / atr_safe[i])
                if size_atr >= min_size_atr:
                    active_fvgs.append((-1, z_lo, z_hi, i, size_atr))

            if len(active_fvgs) > max_active:
                active_fvgs = active_fvgs[-max_active:]

        # 3) emit features
        up = sum(1 for d, *_ in active_fvgs if d == 1)
        down = len(active_fvgs) - up
        fvg_count_up[i] = float(up)
        fvg_count_down[i] = float(down)

        if not active_fvgs:
            continue

        c = float(close[i])
        a = float(atr_safe[i])
        best_abs = float("inf")
        best = (0, 0.0, 0.0, 0, 0.0, 0.0)  # d, z_lo, z_hi, created, size_atr, dist
        for d, z_lo, z_hi, created, size_atr in active_fvgs:
            if c > z_hi:
                dist = (c - z_hi) / a
            elif c < z_lo:
                dist = -(z_lo - c) / a
            else:
                dist = 0.0
            abs_dist = abs(dist)
            # Prefer closer zones; tie-break by recency.
            if abs_dist < best_abs or (abs_dist == best_abs and created > best[3]):
                best_abs = abs_dist
                best = (d, z_lo, z_hi, created, size_atr, dist)

        fvg_direction[i] = float(best[0])
        fvg_size[i] = float(best[4])
        dist_to_fvg[i] = float(np.clip(best[5], -50.0, 50.0))

    # ---------------------------------------------------------------------
    # Order Blocks (OB) - last opposing candle before BOS (uses existing BOS flags if present)
    # ---------------------------------------------------------------------
    bos_bull = out["bos_bullish"].values.astype(np.float32, copy=False) if "bos_bullish" in out.columns else np.zeros(n, dtype=np.float32)
    bos_bear = out["bos_bearish"].values.astype(np.float32, copy=False) if "bos_bearish" in out.columns else np.zeros(n, dtype=np.float32)

    ob_direction = np.zeros(n, dtype=np.float32)
    ob_strength = np.zeros(n, dtype=np.float32)
    dist_to_ob = np.zeros(n, dtype=np.float32)
    price_in_ob = np.zeros(n, dtype=np.float32)

    active_obs: List[Tuple[int, float, float, int, float]] = []  # d, z_lo, z_hi, created, strength_atr
    ob_max_age = int(cfg.ob_max_age_bars)
    ob_max_active = int(cfg.ob_max_active)
    ob_lookback = int(cfg.ob_lookback)

    for i in range(n):
        # 1) remove mitigated/expired
        if active_obs:
            new_active = []
            lo = float(low[i])
            hi = float(high[i])
            for d, z_lo, z_hi, created, strength in active_obs:
                if i - created > ob_max_age:
                    continue
                if lo <= z_hi and hi >= z_lo:
                    continue
                new_active.append((d, z_lo, z_hi, created, strength))
            active_obs = new_active[-ob_max_active:]

        # 2) add on BOS
        if bos_bull[i] == 1.0:
            j0 = max(0, i - ob_lookback)
            j_found = None
            for j in range(i - 1, j0 - 1, -1):
                if close[j] < open_[j]:
                    j_found = j
                    break
            if j_found is not None:
                z_lo = float(low[j_found])
                z_hi = float(high[j_found])
                strength = float(abs(float(close[i]) - float(close[j_found])) / float(atr_safe[i]))
                active_obs.append((1, z_lo, z_hi, i, strength))

        elif bos_bear[i] == 1.0:
            j0 = max(0, i - ob_lookback)
            j_found = None
            for j in range(i - 1, j0 - 1, -1):
                if close[j] > open_[j]:
                    j_found = j
                    break
            if j_found is not None:
                z_lo = float(low[j_found])
                z_hi = float(high[j_found])
                strength = float(abs(float(close[i]) - float(close[j_found])) / float(atr_safe[i]))
                active_obs.append((-1, z_lo, z_hi, i, strength))

        if len(active_obs) > ob_max_active:
            active_obs = active_obs[-ob_max_active:]

        # 3) emit
        if not active_obs:
            continue

        c = float(close[i])
        a = float(atr_safe[i])
        best_abs = float("inf")
        best = (0, 0.0, 0.0, 0, 0.0, 0.0)  # d, z_lo, z_hi, created, strength, dist
        for d, z_lo, z_hi, created, strength in active_obs:
            if c > z_hi:
                dist = (c - z_hi) / a
            elif c < z_lo:
                dist = -(z_lo - c) / a
            else:
                dist = 0.0
            abs_dist = abs(dist)
            if abs_dist < best_abs or (abs_dist == best_abs and created > best[3]):
                best_abs = abs_dist
                best = (d, z_lo, z_hi, created, strength, dist)

        ob_direction[i] = float(best[0])
        ob_strength[i] = float(np.clip(best[4], 0.0, 50.0))
        dist_to_ob[i] = float(np.clip(best[5], -50.0, 50.0))
        price_in_ob[i] = 1.0 if (best[1] <= c <= best[2]) else 0.0

    # ---------------------------------------------------------------------
    # Liquidity pools & sweeps (derived from confirmed pivots)
    # ---------------------------------------------------------------------
    liquidity_above = np.zeros(n, dtype=np.float32)
    liquidity_below = np.zeros(n, dtype=np.float32)
    sweep_signal = np.zeros(n, dtype=np.float32)

    liq_highs: List[Tuple[float, int]] = []
    liq_lows: List[Tuple[float, int]] = []
    liq_lookback = int(cfg.liquidity_lookback_bars)

    for i in range(n):
        if piv_high_arr[i]:
            level = float(high[i - half])
            liq_highs.append((level, i))
        if piv_low_arr[i]:
            level = float(low[i - half])
            liq_lows.append((level, i))

        # Age filter
        if liq_highs:
            liq_highs = [(lvl, idx) for (lvl, idx) in liq_highs if i - idx <= liq_lookback]
        if liq_lows:
            liq_lows = [(lvl, idx) for (lvl, idx) in liq_lows if i - idx <= liq_lookback]

        c = float(close[i])
        a = float(atr_safe[i])

        tol = max(float(cfg.liquidity_range_pct) * c, float(cfg.liquidity_range_atr) * a)

        liq_above_level: Optional[float] = None
        if liq_highs:
            above_levels = [lvl for (lvl, _) in liq_highs if lvl > c]
            if above_levels:
                nearest = min(above_levels)
                cluster = [lvl for lvl in above_levels if abs(lvl - nearest) <= tol]
                liq_above_level = float(np.mean(cluster)) if cluster else float(nearest)
                liquidity_above[i] = float(np.clip((liq_above_level - c) / a, 0.0, 50.0))

        liq_below_level: Optional[float] = None
        if liq_lows:
            below_levels = [lvl for (lvl, _) in liq_lows if lvl < c]
            if below_levels:
                nearest = max(below_levels)
                cluster = [lvl for lvl in below_levels if abs(lvl - nearest) <= tol]
                liq_below_level = float(np.mean(cluster)) if cluster else float(nearest)
                liquidity_below[i] = float(np.clip((c - liq_below_level) / a, 0.0, 50.0))

        # Sweep signals: pierce level then close back through.
        if liq_below_level is not None and float(low[i]) < liq_below_level and float(close[i]) > liq_below_level:
            sweep_signal[i] = 1.0
        elif liq_above_level is not None and float(high[i]) > liq_above_level and float(close[i]) < liq_above_level:
            sweep_signal[i] = -1.0

    # Previous day/week high/low distances (ATR-normalized)
    prev_day_high, prev_day_low = _previous_period_levels(out, "D")
    prev_week_high, prev_week_low = _previous_period_levels(out, str(cfg.weekly_resample_rule))
    pdh_dist = _safe_div((prev_day_high.values.astype(np.float32) - close), atr_safe)
    pdl_dist = _safe_div((close - prev_day_low.values.astype(np.float32)), atr_safe)
    pwh_dist = _safe_div((prev_week_high.values.astype(np.float32) - close), atr_safe)
    pwl_dist = _safe_div((close - prev_week_low.values.astype(np.float32)), atr_safe)
    pdh_dist = np.clip(pdh_dist, -50.0, 50.0)
    pdl_dist = np.clip(pdl_dist, -50.0, 50.0)
    pwh_dist = np.clip(pwh_dist, -50.0, 50.0)
    pwl_dist = np.clip(pwl_dist, -50.0, 50.0)

    # ---------------------------------------------------------------------
    # Premium/Discount + OTE (Fibonacci retracement within last swing range)
    # ---------------------------------------------------------------------
    equilibrium = (swing_high + swing_low) / 2.0
    premium_discount = np.zeros(n, dtype=np.float32)
    valid_swing = ~np.isnan(swing_range_safe)
    premium_discount[valid_swing] = (2.0 * (close[valid_swing] - equilibrium[valid_swing]) / swing_range_safe[valid_swing]).astype(np.float32)
    premium_discount = np.clip(premium_discount, -1.0, 1.0).astype(np.float32)

    in_ote_zone = np.zeros(n, dtype=np.float32)
    ote_distance = np.zeros(n, dtype=np.float32)
    swing_retracement = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if not np.isfinite(swing_range_safe[i]) or swing_dir[i] == 0.0:
            continue
        rng = float(swing_range_safe[i])
        a = float(atr_safe[i])

        if swing_dir[i] == 1.0:
            # Up swing ended at swing high -> OTE is discount retracement from high.
            ote_low = float(swing_high[i] - 0.79 * rng)
            ote_high = float(swing_high[i] - 0.62 * rng)
            retr = (float(swing_high[i]) - float(close[i])) / rng
        else:
            # Down swing ended at swing low -> OTE is premium retracement from low.
            ote_low = float(swing_low[i] + 0.62 * rng)
            ote_high = float(swing_low[i] + 0.79 * rng)
            retr = (float(close[i]) - float(swing_low[i])) / rng

        retr = float(np.clip(retr, 0.0, 1.0))
        swing_retracement[i] = retr * 100.0

        if ote_low <= float(close[i]) <= ote_high:
            in_ote_zone[i] = 1.0
            ote_distance[i] = 0.0
        else:
            dist = min(abs(float(close[i]) - ote_low), abs(float(close[i]) - ote_high)) / a
            ote_distance[i] = float(np.clip(dist, 0.0, 50.0))

    # ---------------------------------------------------------------------
    # Displacement candles (momentum / institutional footprint proxy)
    # ---------------------------------------------------------------------
    body = np.abs(close - open_).astype(np.float32)
    bar_range = (high - low).astype(np.float32)
    bar_range_safe = np.where(bar_range > 1e-8, bar_range, 1e-8).astype(np.float32)

    upper_wick = (high - np.maximum(open_, close)).astype(np.float32)
    lower_wick = (np.minimum(open_, close) - low).astype(np.float32)
    wick_sum = (upper_wick + lower_wick).astype(np.float32)

    body_wick_ratio = (body / (wick_sum + 1e-8)).astype(np.float32)
    body_wick_ratio = np.clip(body_wick_ratio, 0.0, float(cfg.body_wick_ratio_clip)).astype(np.float32)

    avg_body = _rolling_mean(body, int(cfg.displacement_lookback))
    std_body = _rolling_std(body, int(cfg.displacement_lookback))
    momentum_score = _safe_div(body - avg_body, std_body + 1e-6)
    momentum_score = np.clip(momentum_score, -float(cfg.momentum_z_clip), float(cfg.momentum_z_clip)).astype(np.float32)

    body_ratio = (body / bar_range_safe).astype(np.float32)
    displacement_mask = (body > (avg_body * float(cfg.displacement_multiplier))) & (body_ratio >= float(cfg.displacement_min_body_ratio))
    displacement = displacement_mask.astype(np.float32)
    displacement_direction = np.where(displacement_mask, np.sign(close - open_), 0.0).astype(np.float32)

    # ---------------------------------------------------------------------
    # Buying/Selling pressure proxies (OHLC-only)
    # ---------------------------------------------------------------------
    close_position = ((close - low) / bar_range_safe).astype(np.float32)
    close_position = np.clip(close_position, 0.0, 1.0).astype(np.float32)

    delta_proxy = ((close - open_) / bar_range_safe).astype(np.float32)
    delta_proxy = np.clip(delta_proxy, -1.0, 1.0).astype(np.float32)

    cvd_win = int(cfg.cumulative_delta_window)
    cumulative_delta = (
        pd.Series(delta_proxy)
        .rolling(window=cvd_win, min_periods=1)
        .sum()
        .values.astype(np.float32)
    )
    cumulative_delta = np.clip(cumulative_delta, -float(cvd_win), float(cvd_win)).astype(np.float32)

    # Rolling buying pressure windows (bars in cfg.buying_pressure_windows)
    w1, w2, w3 = cfg.buying_pressure_windows
    buying_pressure_15m = _rolling_mean(close_position, int(w2))
    buying_pressure_45m = _rolling_mean(close_position, int(w3))

    # ---------------------------------------------------------------------
    # Session/Killzones + Asian session levels
    # ---------------------------------------------------------------------
    in_killzone_s, killzone_type_s = _killzones(out, cfg)
    asian_high, asian_low = _asian_session_levels(out)
    asian_high_dist = _safe_div((asian_high.values.astype(np.float32) - close), atr_safe)
    asian_low_dist = _safe_div((close - asian_low.values.astype(np.float32)), atr_safe)
    asian_high_dist = np.clip(asian_high_dist, -50.0, 50.0)
    asian_low_dist = np.clip(asian_low_dist, -50.0, 50.0)

    # ---------------------------------------------------------------------
    # Attach columns (float32) - fill NaNs to avoid pipeline row drops.
    # ---------------------------------------------------------------------
    out["fvg_direction"] = fvg_direction.astype(np.float32)
    out["fvg_size"] = fvg_size.astype(np.float32)
    out["dist_to_fvg"] = np.nan_to_num(dist_to_fvg, nan=0.0).astype(np.float32)
    out["fvg_count_up"] = fvg_count_up.astype(np.float32)
    out["fvg_count_down"] = fvg_count_down.astype(np.float32)

    out["ob_direction"] = ob_direction.astype(np.float32)
    out["ob_strength"] = ob_strength.astype(np.float32)
    out["dist_to_ob"] = np.nan_to_num(dist_to_ob, nan=0.0).astype(np.float32)
    out["price_in_ob"] = price_in_ob.astype(np.float32)

    out["sweep_signal"] = sweep_signal.astype(np.float32)
    out["liquidity_above"] = liquidity_above.astype(np.float32)
    out["liquidity_below"] = liquidity_below.astype(np.float32)
    out["pdh_dist"] = np.nan_to_num(pdh_dist, nan=0.0).astype(np.float32)
    out["pdl_dist"] = np.nan_to_num(pdl_dist, nan=0.0).astype(np.float32)
    out["pwh_dist"] = np.nan_to_num(pwh_dist, nan=0.0).astype(np.float32)
    out["pwl_dist"] = np.nan_to_num(pwl_dist, nan=0.0).astype(np.float32)

    out["premium_discount"] = np.nan_to_num(premium_discount, nan=0.0).astype(np.float32)
    out["in_ote_zone"] = in_ote_zone.astype(np.float32)
    out["ote_distance"] = np.nan_to_num(ote_distance, nan=0.0).astype(np.float32)
    out["swing_retracement"] = np.nan_to_num(swing_retracement, nan=0.0).astype(np.float32)

    out["displacement"] = displacement.astype(np.float32)
    out["displacement_direction"] = displacement_direction.astype(np.float32)
    out["body_wick_ratio"] = body_wick_ratio.astype(np.float32)
    out["momentum_score"] = momentum_score.astype(np.float32)

    out["close_position"] = close_position.astype(np.float32)
    out["buying_pressure_15m"] = buying_pressure_15m.astype(np.float32)
    out["buying_pressure_45m"] = buying_pressure_45m.astype(np.float32)
    out["delta_proxy"] = delta_proxy.astype(np.float32)
    out["cumulative_delta"] = cumulative_delta.astype(np.float32)

    out["in_killzone"] = in_killzone_s.values.astype(np.float32)
    out["killzone_type"] = killzone_type_s.values.astype(np.float32)
    out["asian_high_dist"] = np.nan_to_num(asian_high_dist, nan=0.0).astype(np.float32)
    out["asian_low_dist"] = np.nan_to_num(asian_low_dist, nan=0.0).astype(np.float32)

    # Ensure no NaNs introduced by mapping/resampling
    for col in SMC_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(0.0).astype(np.float32)

    return out
