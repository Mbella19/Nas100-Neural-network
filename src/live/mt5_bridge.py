"""
MT5 ↔ Python bridge for live inference.

Design goals:
- Match the offline training/backtest conditions as closely as possible:
  - Ingest 1-minute OHLC from MT5 (server time) + UTC offset
  - Convert timestamps to UTC (timezone-naive like the training pipeline)
  - Rebuild 5m/15m/45m bars using the same pandas resampling semantics
    (label='right', closed='left') to avoid look-ahead bias
  - Apply the same feature engineering and saved normalizers
  - Build observations with the same ordering/scaling as TradingEnv
  - Run frozen Analyst + PPO Agent inference and return trade instructions

The MT5-side EA is expected to connect via TCP, send a length-prefixed JSON
payload, then read a length-prefixed JSON response.
"""

from __future__ import annotations

import json
import socketserver
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config.settings import Config, get_device
from src.agents.sniper_agent import SniperAgent
from src.data.features import engineer_all_features
from src.data.microstructure import add_smc_microstructure_features
from src.data.feature_names import SMC_FEATURE_COLUMNS, AGENT_TF_BASE_FEATURE_COLUMNS
from src.data.normalizer import FeatureNormalizer
from src.data.resampler import resample_all_timeframes, align_timeframes
from src.models.analyst import load_analyst
from src.utils.logging_config import setup_logging, get_logger

from .bridge_constants import MODEL_FEATURE_COLS, MARKET_FEATURE_COLS, POSITION_SIZES

logger = get_logger(__name__)


# =============================================================================
# Feature/Observation conventions (must match training)
# =============================================================================


@dataclass(frozen=True)
class MarketFeatureStats:
    """Z-score stats applied to `market_features` inside the observation."""

    cols: Tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "MarketFeatureStats":
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        cols = tuple(data["cols"].tolist())
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)
        std = np.where(std > 1e-8, std, 1.0).astype(np.float32)
        return cls(cols=cols, mean=mean, std=std)


@dataclass
class BridgeConfig:
    """Runtime configuration for the MT5 bridge server."""

    host: str = "127.0.0.1"
    port: int = 5555
    main_symbol: str = "NAS100"
    decision_tf_minutes: int = 5

    # Persist incoming M1 bars so restarts don't require MT5 bootstrap.
    history_dir: Path = field(default_factory=lambda: Path("data") / "live")
    max_m1_rows: int = 60 * 24 * 30  # ~30 days

    # Minimum M1 rows required before trading is enabled.
    # This is primarily driven by 45m SMA(200): 200 * 45 = 9000 minutes.
    min_m1_rows: int = 10_000

    # Execution mapping (EA expects lots).
    lot_scale: float = 1.0

    # Optional component mapping: ticker -> MT5 symbol name.
    component_symbols: Dict[str, str] = field(
        default_factory=lambda: {
            "AAPL": "AAPL",
            "MSFT": "MSFT",
            "NVDA": "NVDA",
            "AMZN": "AMZN",
            "GOOG": "GOOG",
            "AVGO": "AVGO",
        }
    )

    # Feature pipeline window sizes (recompute on a tail window for speed).
    tail_5m_bars: int = 600
    tail_15m_bars: int = 400
    tail_45m_bars: int = 260

    # Safety / testing
    dry_run: bool = False


def _read_exact(sock, n: int) -> bytes:
    """Read exactly n bytes from a socket."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while reading")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _decode_length_prefixed_json(sock) -> Dict[str, Any]:
    header = _read_exact(sock, 4)
    (length,) = struct.unpack(">I", header)
    if length <= 0 or length > 50_000_000:
        raise ValueError(f"Invalid payload length: {length}")
    payload = _read_exact(sock, length)
    return json.loads(payload.decode("utf-8"))


def _encode_length_prefixed_json(obj: Dict[str, Any]) -> bytes:
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _rates_array_to_df(
    rates: list,
    utc_offset_sec: int,
) -> pd.DataFrame:
    """
    Convert an MT5 rates array to a DataFrame indexed by UTC (timezone-naive).

    Expected per-row formats:
      [time, open, high, low, close] or [time, open, high, low, close, ...]
    Where `time` is in broker/server time; `utc_offset_sec` converts it to UTC.
    """
    if not rates:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    rows = []
    for row in rates:
        if not isinstance(row, (list, tuple)) or len(row) < 5:
            continue
        t_server = int(row[0])
        t_utc = t_server - int(utc_offset_sec)
        rows.append((t_utc, float(row[1]), float(row[2]), float(row[3]), float(row[4])))

    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])

    arr = np.array(rows, dtype=np.float64)
    idx = pd.to_datetime(arr[:, 0].astype(np.int64), unit="s", utc=True).tz_localize(None)
    df = pd.DataFrame(
        {
            "open": arr[:, 1].astype(np.float32),
            "high": arr[:, 2].astype(np.float32),
            "low": arr[:, 3].astype(np.float32),
            "close": arr[:, 4].astype(np.float32),
        },
        index=idx,
    )
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _merge_append_ohlc(existing: pd.DataFrame, new_df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if existing is None or existing.empty:
        merged = new_df.copy()
    else:
        merged = pd.concat([existing, new_df], axis=0)
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if len(merged) > max_rows:
        merged = merged.iloc[-max_rows:].copy()
    return merged


def _save_history(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Small, fast persistence format
    df.to_parquet(path)


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"History file has invalid index: {path}")
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].astype(np.float32)
    return df.sort_index()


def _compute_top6_features(
    main_index: pd.DatetimeIndex,
    component_m1: Dict[str, pd.DataFrame],
    component_symbols: Dict[str, str],
    resample_rule: str,
) -> pd.DataFrame:
    """
    Compute top6_momentum/top6_dispersion on `main_index` from component close returns.

    This mirrors `src.data.components.merge_component_features` but uses in-memory
    component OHLC data received live from MT5 instead of CSV files.
    """
    from src.data.components import COMPONENT_WEIGHTS

    aligned_returns = pd.DataFrame(index=main_index)
    weights: Dict[str, float] = {}

    for ticker, mt5_symbol in component_symbols.items():
        df_1m = component_m1.get(mt5_symbol)
        if df_1m is None or df_1m.empty or "close" not in df_1m.columns:
            continue

        resampled = (
            df_1m["close"]
            .resample(resample_rule, label="right", closed="left")
            .last()
            .dropna()
        )
        aligned_close = resampled.reindex(main_index, method="ffill", limit=12)
        ret = aligned_close.pct_change(fill_method=None)
        if ret.count() > 0:
            aligned_returns[ticker] = ret
            weights[ticker] = float(COMPONENT_WEIGHTS.get(ticker, 0.01))

    if aligned_returns.empty:
        return pd.DataFrame(
            {"top6_momentum": np.zeros(len(main_index), dtype=np.float32),
             "top6_dispersion": np.zeros(len(main_index), dtype=np.float32)},
            index=main_index,
        )

    total_w = sum(weights.values())
    if total_w <= 0:
        norm_weights = {k: 1.0 / len(weights) for k in weights}
    else:
        norm_weights = {k: v / total_w for k, v in weights.items()}

    weighted = []
    for tkr, w in norm_weights.items():
        if tkr in aligned_returns.columns:
            weighted.append(aligned_returns[tkr] * w)
    if not weighted:
        top6_mom = pd.Series(0.0, index=main_index)
    else:
        top6_mom = pd.concat(weighted, axis=1).sum(axis=1)
    top6_disp = aligned_returns.std(axis=1)

    return pd.DataFrame(
        {
            "top6_momentum": top6_mom.fillna(0).astype(np.float32),
            "top6_dispersion": top6_disp.fillna(0).astype(np.float32),
        },
        index=main_index,
    )


def _compute_component_sequence(
    main_index: pd.DatetimeIndex,
    component_m1: Dict[str, pd.DataFrame],
    component_symbols: Dict[str, str],
    resample_rule: str,
    seq_len: int,
    end_pos: int,
) -> Optional[np.ndarray]:
    """
    Compute a single component sequence tensor aligned to `main_index`.

    Output matches `prepare_component_sequences` shape for the LAST index only:
      [n_components, seq_len, 4] where 4 = open_ret/high_ret/low_ret/close_ret.
    """
    from src.data.components import compute_component_returns

    feature_cols = ["open_ret", "high_ret", "low_ret", "close_ret"]
    if end_pos < 0 or end_pos >= len(main_index):
        raise ValueError(f"end_pos out of range: {end_pos} (len={len(main_index)})")

    n_components = len(component_symbols)
    sequences = np.zeros((n_components, seq_len, 4), dtype=np.float32)

    for comp_idx, (ticker, mt5_symbol) in enumerate(component_symbols.items()):
        df_1m = component_m1.get(mt5_symbol)
        if df_1m is None or df_1m.empty:
            continue

        ohlc_agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
        resampled = df_1m.resample(resample_rule, label="right", closed="left").agg(ohlc_agg).dropna()
        aligned = resampled.reindex(main_index, method="ffill", limit=12)
        returns_df = compute_component_returns(aligned)
        for col in feature_cols:
            if col not in returns_df.columns:
                returns_df[col] = 0.0
        ret_array = returns_df[feature_cols].values.astype(np.float32)

        # Build window ending at `end_pos` (aligned to main_index).
        if end_pos + 1 < seq_len:
            tail = ret_array[: end_pos + 1]
            sequences[comp_idx, -tail.shape[0]:, :] = tail
        else:
            start = end_pos - seq_len + 1
            sequences[comp_idx, :, :] = ret_array[start:end_pos + 1, :]

    return sequences


def _build_observation(
    *,
    analyst: torch.nn.Module,
    agent_env_cfg: Config,
    market_feat_stats: MarketFeatureStats,
    x_5m: np.ndarray,
    x_15m: np.ndarray,
    x_45m: np.ndarray,
    market_feat_row: np.ndarray,
    returns_row_window: np.ndarray,
    component_seq: Optional[np.ndarray],
    position: int,
    entry_price: float,
    current_price: float,
    position_size: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Construct an observation vector with the same ordering as TradingEnv._get_observation().
    """
    device = next(analyst.parameters()).device if hasattr(analyst, "parameters") else torch.device("cpu")

    x_5m_t = torch.tensor(x_5m[None, ...], device=device, dtype=torch.float32)
    x_15m_t = torch.tensor(x_15m[None, ...], device=device, dtype=torch.float32)
    x_45m_t = torch.tensor(x_45m[None, ...], device=device, dtype=torch.float32)

    comp_t = None
    if component_seq is not None:
        comp_t = torch.tensor(component_seq[None, ...], device=device, dtype=torch.float32)

    with torch.no_grad():
        try:
            context, probs, weights = analyst.get_probabilities(
                x_5m_t, x_15m_t, x_45m_t, component_data=comp_t
            )
        except TypeError:
            # Backwards compatible: some checkpoints may not accept component_data kwarg.
            context, probs, weights = analyst.get_probabilities(x_5m_t, x_15m_t, x_45m_t)

    context_np = context.cpu().numpy().flatten().astype(np.float32)
    probs_np = probs.cpu().numpy().flatten().astype(np.float32)
    weights_np = None
    if weights is not None:
        weights_np = weights.cpu().numpy().flatten().astype(np.float32)

    # Analyst metrics (binary vs multi-class)
    if len(probs_np) == 2:
        p_down = float(probs_np[0])
        p_up = float(probs_np[1])
        confidence = max(p_down, p_up)
        edge = p_up - p_down
        uncertainty = 1.0 - confidence
        analyst_metrics = np.array([p_down, p_up, edge, confidence, uncertainty], dtype=np.float32)
    else:
        p_down = float(probs_np[0])
        p_neutral = float(probs_np[1])
        p_up = float(probs_np[2])
        confidence = float(np.max(probs_np))
        edge = p_up - p_down
        uncertainty = 1.0 - confidence
        analyst_metrics = np.array(
            [p_down, p_neutral, p_up, edge, confidence, uncertainty], dtype=np.float32
        )

    # Position state (mirrors TradingEnv normalization)
    atr = float(market_feat_row[0]) if len(market_feat_row) > 0 else 1.0
    atr_safe = max(atr, 1e-6)

    if position != 0:
        if position == 1:
            entry_price_norm = (current_price - entry_price) / (atr_safe * 100.0)
        else:
            entry_price_norm = (entry_price - current_price) / (atr_safe * 100.0)
        entry_price_norm = float(np.clip(entry_price_norm, -10.0, 10.0))

        pip_value = agent_env_cfg.instrument.pip_value
        if position == 1:
            unrealized_pnl = (current_price - entry_price) / pip_value
        else:
            unrealized_pnl = (entry_price - current_price) / pip_value
        unrealized_pnl *= float(position_size)
        unrealized_pnl_norm = float(unrealized_pnl / 100.0)
    else:
        entry_price_norm = 0.0
        unrealized_pnl_norm = 0.0

    position_state = np.array([float(position), entry_price_norm, unrealized_pnl_norm], dtype=np.float32)

    # Market feature normalization (second-stage zscore used in env)
    # Ensure the incoming row ordering matches the saved stats ordering.
    if tuple(MARKET_FEATURE_COLS) != tuple(market_feat_stats.cols):
        raise ValueError(
            "Market feature columns mismatch. "
            f"Expected {market_feat_stats.cols}, got {MARKET_FEATURE_COLS}"
        )
    market_feat_norm = ((market_feat_row - market_feat_stats.mean) / market_feat_stats.std).astype(np.float32)

    # SL/TP distance features (mirrors TradingEnv)
    dist_sl_norm = 0.0
    dist_tp_norm = 0.0
    if position != 0 and atr > 1e-8:
        pip_value = agent_env_cfg.instrument.pip_value
        sl_pips = max((atr * agent_env_cfg.trading.sl_atr_multiplier) / pip_value, 5.0)
        tp_pips = max((atr * agent_env_cfg.trading.tp_atr_multiplier) / pip_value, 5.0)

        if position == 1:
            sl_price = entry_price - sl_pips * pip_value
            tp_price = entry_price + tp_pips * pip_value
            dist_sl_norm = (current_price - sl_price) / atr
            dist_tp_norm = (tp_price - current_price) / atr
        else:
            sl_price = entry_price + sl_pips * pip_value
            tp_price = entry_price - tp_pips * pip_value
            dist_sl_norm = (sl_price - current_price) / atr
            dist_tp_norm = (current_price - tp_price) / atr

    obs = np.concatenate(
        [
            context_np,
            position_state,
            market_feat_norm,
            analyst_metrics,
            np.array([dist_sl_norm, dist_tp_norm], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)

    # Full-eyes returns window (already normalized in the pipeline; env multiplies by 100)
    lookback = int(agent_env_cfg.trading.agent_lookback_window)
    if lookback > 0:
        if returns_row_window.shape[0] != lookback:
            raise ValueError(f"returns window mismatch: {returns_row_window.shape[0]} != {lookback}")
        obs = np.concatenate([obs, (returns_row_window.astype(np.float32) * 100.0)], axis=0)

    # Attention weights features
    if bool(agent_env_cfg.trading.include_attention_features):
        n_components = int(agent_env_cfg.analyst.n_components)
        if weights_np is None:
            att = np.ones(n_components, dtype=np.float32) / max(n_components, 1)
        else:
            if weights_np.size > n_components:
                att = weights_np.reshape(-1, n_components).mean(axis=0).astype(np.float32)
            elif weights_np.size == n_components:
                att = weights_np.astype(np.float32)
            else:
                att = np.ones(n_components, dtype=np.float32) / max(n_components, 1)
        obs = np.concatenate([obs, att], axis=0)

    info = {
        "p_down": float(probs_np[0]) if probs_np.size >= 1 else 0.5,
        "p_up": float(probs_np[-1]) if probs_np.size >= 2 else 0.5,
    }
    return obs.astype(np.float32), info


class MT5BridgeState:
    def __init__(self, cfg: BridgeConfig, system_cfg: Config):
        self.cfg = cfg
        self.system_cfg = system_cfg

        self.history_dir = cfg.history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)

        self.m1_path = self.history_dir / f"{cfg.main_symbol}_M1.parquet"
        self.m1: pd.DataFrame = _load_history(self.m1_path)

        self.components_m1: Dict[str, pd.DataFrame] = {}
        for _, mt5_symbol in cfg.component_symbols.items():
            path = self.history_dir / f"{mt5_symbol}_M1.parquet"
            self.components_m1[mt5_symbol] = _load_history(path)

        # Load artifacts
        self.normalizers: Dict[str, FeatureNormalizer] = {
            "5m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_5m.pkl"),
            "15m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_15m.pkl"),
            "45m": FeatureNormalizer.load(system_cfg.paths.models_analyst / "normalizer_45m.pkl"),
        }

        market_stats_path = system_cfg.paths.models_agent / "market_feat_stats.npz"
        self.market_feat_stats = MarketFeatureStats.load(market_stats_path)

        feature_dims = {k: len(MODEL_FEATURE_COLS) for k in ("5m", "15m", "45m")}
        analyst_path = system_cfg.paths.models_analyst / "best.pt"
        self.analyst = load_analyst(str(analyst_path), feature_dims, device=system_cfg.device, freeze=True)
        self.analyst.eval()

        obs_dim = self._expected_obs_dim()
        dummy_env = _make_dummy_env(obs_dim)
        agent_path = system_cfg.paths.models_agent / "final_model.zip"
        self.agent = SniperAgent.load(str(agent_path), dummy_env, device="cpu")

        self.last_decision_label_utc: Optional[pd.Timestamp] = None

        logger.info(
            "MT5 bridge ready | symbol=%s | obs_dim=%d | feature_dim=%d",
            cfg.main_symbol,
            obs_dim,
            len(MODEL_FEATURE_COLS),
        )

    def _expected_obs_dim(self) -> int:
        context_dim = int(getattr(self.analyst, "context_dim", self.system_cfg.analyst.context_dim))
        analyst_metrics_dim = 5 if int(getattr(self.analyst, "num_classes", 2)) == 2 else 6
        n_market = len(MARKET_FEATURE_COLS)
        returns_dim = int(self.system_cfg.trading.agent_lookback_window)
        attention_dim = int(self.system_cfg.analyst.n_components) if self.system_cfg.trading.include_attention_features else 0
        return context_dim + 3 + n_market + analyst_metrics_dim + 2 + returns_dim + attention_dim

    def update_from_payload(self, payload: Dict[str, Any]) -> None:
        utc_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))

        rates = payload.get("rates", {})
        m1_rates = rates.get("m1") or rates.get("1m") or rates.get("M1") or []
        m1_df = _rates_array_to_df(m1_rates, utc_offset_sec=utc_offset_sec)
        if not m1_df.empty:
            self.m1 = _merge_append_ohlc(self.m1, m1_df, self.cfg.max_m1_rows)
            _save_history(self.m1, self.m1_path)

        comps = payload.get("components", {}) or {}
        for _, mt5_symbol in self.cfg.component_symbols.items():
            comp_rates = comps.get(mt5_symbol) or comps.get(mt5_symbol.lower()) or comps.get(mt5_symbol.upper())
            if not comp_rates:
                continue
            comp_df = _rates_array_to_df(comp_rates, utc_offset_sec=utc_offset_sec)
            if comp_df.empty:
                continue
            merged = _merge_append_ohlc(self.components_m1.get(mt5_symbol), comp_df, self.cfg.max_m1_rows)
            self.components_m1[mt5_symbol] = merged
            _save_history(merged, self.history_dir / f"{mt5_symbol}_M1.parquet")

    def _should_decide_now(self, payload: Dict[str, Any]) -> Tuple[bool, Optional[pd.Timestamp]]:
        utc_offset_sec = int(payload.get("time", {}).get("utc_offset_sec", 0))
        rates = payload.get("rates", {})
        m1_rates = rates.get("m1") or rates.get("1m") or []
        if not m1_rates:
            return False, None

        last_row = m1_rates[-1]
        if not isinstance(last_row, (list, tuple)) or len(last_row) < 1:
            return False, None

        t_server = int(last_row[0])
        t_utc_open = t_server - utc_offset_sec
        t_utc_close = t_utc_open + 60

        if t_utc_close % (self.cfg.decision_tf_minutes * 60) != 0:
            return False, None

        label = pd.to_datetime(t_utc_close, unit="s", utc=True).tz_localize(None)
        if self.last_decision_label_utc is not None and label <= self.last_decision_label_utc:
            return False, label
        return True, label

    def decide(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.update_from_payload(payload)

        should_decide, label = self._should_decide_now(payload)
        if not should_decide:
            return {"action": 999, "reason": "no_new_5m_bar"}

        if len(self.m1) < self.cfg.min_m1_rows:
            return {
                "action": 999,
                "reason": f"warming_up_m1_history ({len(self.m1)}/{self.cfg.min_m1_rows})",
            }

        # Build features on a tail window for speed
        df_1m = self.m1.copy()

        # Resample to native bars (no gap filling, no weekend flats)
        resampled = resample_all_timeframes(df_1m, self.system_cfg.data.timeframes)
        df_5m = resampled["5m"]
        df_15m = resampled["15m"]
        df_45m = resampled["45m"]

        # Add Top6 component scalar features BEFORE feature engineering (mirrors pipeline)
        top6_5m = _compute_top6_features(
            df_5m.index, self.components_m1, self.cfg.component_symbols, resample_rule=self.system_cfg.data.timeframes["5m"]
        )
        top6_15m = _compute_top6_features(
            df_15m.index, self.components_m1, self.cfg.component_symbols, resample_rule=self.system_cfg.data.timeframes["15m"]
        )
        top6_45m = _compute_top6_features(
            df_45m.index, self.components_m1, self.cfg.component_symbols, resample_rule=self.system_cfg.data.timeframes["45m"]
        )
        df_5m = df_5m.join(top6_5m, how="left").fillna({"top6_momentum": 0.0, "top6_dispersion": 0.0})
        df_15m = df_15m.join(top6_15m, how="left").fillna({"top6_momentum": 0.0, "top6_dispersion": 0.0})
        df_45m = df_45m.join(top6_45m, how="left").fillna({"top6_momentum": 0.0, "top6_dispersion": 0.0})

        feature_cfg = {
            "pinbar_wick_ratio": self.system_cfg.features.pinbar_wick_ratio,
            "doji_body_ratio": self.system_cfg.features.doji_body_ratio,
            "fractal_window": self.system_cfg.features.fractal_window,
            "sr_lookback": self.system_cfg.features.sr_lookback,
            "sma_period": self.system_cfg.features.sma_period,
            "ema_fast": self.system_cfg.features.ema_fast,
            "ema_slow": self.system_cfg.features.ema_slow,
            "chop_period": self.system_cfg.features.chop_period,
            "adx_period": self.system_cfg.features.adx_period,
            "atr_period": self.system_cfg.features.atr_period,
        }

        df_5m = engineer_all_features(df_5m, feature_cfg)
        df_15m = engineer_all_features(df_15m, feature_cfg)
        df_45m = engineer_all_features(df_45m, feature_cfg)

        # Agent-only SMC microstructure features computed on NATIVE TF bars (pre-alignment)
        try:
            smc_base = getattr(self.system_cfg, "smc", None)
            if smc_base is not None:
                add_smc_microstructure_features(df_5m, smc_base.for_timeframe_minutes(5), inplace=True)
                if getattr(smc_base, "include_higher_timeframes", True):
                    add_smc_microstructure_features(df_15m, smc_base.for_timeframe_minutes(15), inplace=True)
                    add_smc_microstructure_features(df_45m, smc_base.for_timeframe_minutes(45), inplace=True)
        except Exception as e:
            logger.warning("Failed to compute SMC features (native TFs): %s", e)

        # Align higher TFs to 5m grid AFTER feature engineering
        df_5m, df_15m, df_45m = align_timeframes(df_5m, df_15m, df_45m)

        # Drop rows with any NaN across timeframes (mirrors pipeline)
        common_valid = (~df_5m.isna().any(axis=1)) & (~df_15m.isna().any(axis=1)) & (~df_45m.isna().any(axis=1))
        df_5m = df_5m[common_valid]
        df_15m = df_15m[common_valid]
        df_45m = df_45m[common_valid]

        # Copy 15m/45m base+SMC features onto the 5m frame with suffixes for PPO visibility
        smc_base = getattr(self.system_cfg, "smc", None)
        if smc_base is None or getattr(smc_base, "include_higher_timeframes", True):
            tf_cols = list(AGENT_TF_BASE_FEATURE_COLUMNS) + list(SMC_FEATURE_COLUMNS)
            for tf_name, df_tf in (("15m", df_15m), ("45m", df_45m)):
                for col in tf_cols:
                    if col not in df_tf.columns:
                        df_tf[col] = 0.0
                block = df_tf[tf_cols].add_suffix(f"_{tf_name}").astype(np.float32)
                add_cols = [c for c in block.columns if c not in df_5m.columns]
                if add_cols:
                    df_5m = df_5m.join(block[add_cols], how="left")
                    df_5m[add_cols] = df_5m[add_cols].fillna(0.0).astype(np.float32)

        # 1m SMC microstructure features resampled onto the 5m grid (agent-only)
        if getattr(getattr(self.system_cfg, "smc", None), "include_m1_features", True):
            try:
                smc_base = getattr(self.system_cfg, "smc", None)
                smc_cfg_1m = smc_base.for_timeframe_minutes(1) if smc_base is not None else None
                add_smc_microstructure_features(df_1m, smc_cfg_1m, inplace=True)
                smc_1m_5m = (
                    df_1m[list(SMC_FEATURE_COLUMNS)]
                    .resample(self.system_cfg.data.timeframes["5m"], label="right", closed="left")
                    .last()
                    .dropna(how="all")
                    .astype(np.float32)
                )
                smc_1m_5m = smc_1m_5m.reindex(df_5m.index).fillna(0.0).astype(np.float32)
                smc_1m_5m.columns = [f"{c}_1m" for c in smc_1m_5m.columns]
                add_cols = [c for c in smc_1m_5m.columns if c not in df_5m.columns]
                if add_cols:
                    df_5m = df_5m.join(smc_1m_5m[add_cols], how="left")
                df_5m[smc_1m_5m.columns] = df_5m[smc_1m_5m.columns].fillna(0.0).astype(np.float32)
            except Exception as e:
                logger.warning("Failed to compute 1m SMC features in live pipeline: %s", e)

        # Apply saved normalizers (per timeframe). RAW columns remain unscaled.
        df_5m_n = self.normalizers["5m"].transform(df_5m)
        df_15m_n = self.normalizers["15m"].transform(df_15m)
        df_45m_n = self.normalizers["45m"].transform(df_45m)

        if label not in df_5m_n.index:
            # Not enough data to form this label in our UTC-resampled timeline.
            return {"action": 999, "reason": "label_not_ready"}

        # Find positional index
        pos = int(df_5m_n.index.get_loc(label))

        lookback_5m = int(self.system_cfg.analyst.lookback_5m)
        lookback_15m = int(self.system_cfg.analyst.lookback_15m)
        lookback_45m = int(self.system_cfg.analyst.lookback_45m)
        subsample_15m = 3
        subsample_45m = 9
        start_idx = max(
            lookback_5m,
            (lookback_15m - 1) * subsample_15m + 1,
            (lookback_45m - 1) * subsample_45m + 1,
        )

        if pos < start_idx:
            return {"action": 999, "reason": "insufficient_lookback"}

        features_5m = df_5m_n[MODEL_FEATURE_COLS].values.astype(np.float32)
        features_15m = df_15m_n[MODEL_FEATURE_COLS].values.astype(np.float32)
        features_45m = df_45m_n[MODEL_FEATURE_COLS].values.astype(np.float32)

        x_5m = features_5m[pos - lookback_5m + 1:pos + 1]

        idx_range_15m = list(
            range(pos - (lookback_15m - 1) * subsample_15m, pos + 1, subsample_15m)
        )
        x_15m = features_15m[idx_range_15m]

        idx_range_45m = list(
            range(pos - (lookback_45m - 1) * subsample_45m, pos + 1, subsample_45m)
        )
        x_45m = features_45m[idx_range_45m]

        # Market features row (order must match MARKET_FEATURE_COLS)
        for col in MARKET_FEATURE_COLS:
            if col not in df_5m_n.columns:
                df_5m_n[col] = 0.0
        df_5m_n[MARKET_FEATURE_COLS] = df_5m_n[MARKET_FEATURE_COLS].fillna(0.0)
        market_feat_row = df_5m_n[MARKET_FEATURE_COLS].iloc[pos].values.astype(np.float32)

        # Returns window (already normalized by FeatureNormalizer; env multiplies by 100)
        lookback_ret = int(self.system_cfg.trading.agent_lookback_window)
        returns_series = df_5m_n["returns"].values.astype(np.float32)
        returns_window = returns_series[pos - lookback_ret + 1:pos + 1] if lookback_ret > 0 else np.array([], dtype=np.float32)
        if lookback_ret > 0 and returns_window.shape[0] != lookback_ret:
            return {"action": 999, "reason": "returns_window_not_ready"}

        # Cross-asset component sequence for this timestep (5m grid)
        component_seq = None
        if getattr(self.analyst, "cross_asset_module", None) is not None:
            component_seq = _compute_component_sequence(
                df_5m_n.index,
                self.components_m1,
                self.cfg.component_symbols,
                resample_rule=self.system_cfg.data.timeframes["5m"],
                seq_len=int(self.system_cfg.analyst.component_seq_len),
                end_pos=pos,
            )

        # Position state from MT5
        pos_payload = payload.get("position", {}) or {}
        pos_type = int(pos_payload.get("type", -1))
        mt5_volume = float(pos_payload.get("volume", 0.0))
        mt5_entry = float(pos_payload.get("price", 0.0))

        if pos_type == 0:
            position = 1
        elif pos_type == 1:
            position = -1
        else:
            position = 0

        current_price = float(df_5m_n["close"].iloc[pos])
        entry_price = float(mt5_entry) if position != 0 else current_price
        position_size = float(mt5_volume) if position != 0 else 0.0

        obs, obs_info = _build_observation(
            analyst=self.analyst,
            agent_env_cfg=self.system_cfg,
            market_feat_stats=self.market_feat_stats,
            x_5m=x_5m,
            x_15m=x_15m,
            x_45m=x_45m,
            market_feat_row=market_feat_row,
            returns_row_window=returns_window,
            component_seq=component_seq,
            position=position,
            entry_price=entry_price,
            current_price=current_price,
            position_size=position_size,
        )

        if self.cfg.dry_run:
            self.last_decision_label_utc = label
            return {"action": 999, "reason": "dry_run", **obs_info}

        action, _ = self.agent.predict(
            obs,
            deterministic=True,
            min_action_confidence=float(self.system_cfg.trading.min_action_confidence),
        )
        action = np.array(action).astype(np.int32).flatten()
        if action.size < 2:
            return {"action": 999, "reason": "invalid_agent_action"}

        direction = int(action[0])
        size_idx = int(action[1])
        size_idx = int(np.clip(size_idx, 0, 3))

        atr = float(market_feat_row[0])
        pip_value = float(self.system_cfg.instrument.pip_value)

        base_size = float(POSITION_SIZES[size_idx])
        lots = base_size
        if bool(getattr(self.system_cfg.trading, "volatility_sizing", True)):
            sl_pips = (atr * float(self.system_cfg.trading.sl_atr_multiplier)) / pip_value
            sl_pips = max(sl_pips, 5.0)
            # NOTE: The training env used risk_pips_target=15.0 unless overridden.
            risk_pips_target = float(getattr(self.system_cfg.trading, "risk_pips_target", 15.0))
            vol_scalar = risk_pips_target / sl_pips
            lots = float(np.clip(base_size * vol_scalar, 0.1, 5.0))

        lots = float(np.clip(lots * float(self.cfg.lot_scale), 0.0, 1000.0))

        sl_price = 0.0
        tp_price = 0.0
        if direction in (1, 2):
            sl_pips = max((atr * float(self.system_cfg.trading.sl_atr_multiplier)) / pip_value, 5.0)
            tp_pips = max((atr * float(self.system_cfg.trading.tp_atr_multiplier)) / pip_value, 5.0)
            if direction == 1:
                sl_price = current_price - sl_pips * pip_value
                tp_price = current_price + tp_pips * pip_value
            else:
                sl_price = current_price + sl_pips * pip_value
                tp_price = current_price - tp_pips * pip_value

        self.last_decision_label_utc = label

        logger.info(
            "Decision @ %s | action=%d size=%.2f sl=%.2f tp=%.2f | p_up=%.3f p_down=%.3f",
            label,
            direction,
            round(lots, 2),
            float(sl_price),
            float(tp_price),
            float(obs_info.get("p_up", 0.5)),
            float(obs_info.get("p_down", 0.5)),
        )

        return {
            "action": direction,
            "size": round(lots, 2),
            "sl": float(sl_price),
            "tp": float(tp_price),
            "ts_utc": int(label.timestamp()),
            **obs_info,
        }


def _make_dummy_env(obs_dim: int):
    import gymnasium as gym
    from gymnasium import spaces

    class _DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.MultiDiscrete([3, 4])

        def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
            super().reset(seed=seed)
            return np.zeros((obs_dim,), dtype=np.float32), {}

        def step(self, action):
            return np.zeros((obs_dim,), dtype=np.float32), 0.0, True, False, {}

    return _DummyEnv()


class _BridgeHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        try:
            payload = _decode_length_prefixed_json(self.request)
            response = self.server.bridge_state.decide(payload)
        except Exception as e:
            logger.exception("Bridge error: %s", e)
            response = {"action": 999, "reason": "server_error"}

        self.request.sendall(_encode_length_prefixed_json(response))


class MT5BridgeServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True

    def __init__(self, server_address, handler_class, bridge_state: MT5BridgeState):
        super().__init__(server_address, handler_class)
        self.bridge_state = bridge_state


def run_mt5_bridge(
    bridge_cfg: BridgeConfig,
    system_cfg: Optional[Config] = None,
    log_dir: Optional[str | Path] = None,
) -> None:
    if system_cfg is None:
        system_cfg = Config()

    if system_cfg.device is None:
        system_cfg.device = get_device()

    setup_logging(str(log_dir) if log_dir is not None else None, name=__name__)

    state = MT5BridgeState(bridge_cfg, system_cfg)
    server = MT5BridgeServer((bridge_cfg.host, bridge_cfg.port), _BridgeHandler, state)

    logger.info("Listening on %s:%d", bridge_cfg.host, bridge_cfg.port)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
