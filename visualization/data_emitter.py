"""
Thread-safe data emitter for training metrics.

This module provides a singleton that training callbacks can push data to,
and the WebSocket server can pull from. Designed for zero training slowdown.

Usage in training code:
    from visualization import get_emitter
    emitter = get_emitter()
    emitter.push_agent_step(timestep=1000, reward=0.5, ...)

Usage in server:
    emitter = get_emitter()
    snapshot = emitter.get_latest()
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Deque
import numpy as np

from .models import (
    TrainingSnapshot,
    AnalystState,
    AgentState,
    MarketState,
    RewardComponents,
    SystemStatus,
    OHLCBar,
    TradeMarker,
)
from .config import VisualizationConfig


@dataclass
class DataEmitter:
    """
    Thread-safe data emitter for training visualization.

    Uses a non-blocking queue to prevent training slowdown.
    Maintains circular buffers for historical data.
    """

    config: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self):
        # Thread-safe queue for new snapshots
        self._queue: queue.Queue = queue.Queue(maxsize=1000)

        # Latest complete state
        self._latest_analyst: Optional[AnalystState] = None
        self._latest_agent: Optional[AgentState] = None
        self._latest_market: Optional[MarketState] = None
        self._latest_reward: Optional[RewardComponents] = None
        self._latest_system: Optional[SystemStatus] = None

        # History buffers (circular)
        self._snapshot_history: Deque[TrainingSnapshot] = deque(
            maxlen=self.config.max_snapshots
        )
        self._price_history: Deque[OHLCBar] = deque(
            maxlen=self.config.max_price_bars
        )
        self._trade_history: Deque[TradeMarker] = deque(
            maxlen=self.config.max_trades
        )
        self._loss_history: Deque[Dict[str, float]] = deque(maxlen=1000)
        self._reward_history: Deque[float] = deque(maxlen=10000)
        self._pnl_history: Deque[float] = deque(maxlen=10000)

        # Tracking
        self._start_time: float = time.time()
        self._last_broadcast_time: float = 0.0
        self._step_count: int = 0
        self._episode_count: int = 0

        # Lock for thread safety on history access
        self._lock = threading.RLock()

        # Enable flag
        self._enabled: bool = True
        
        # Start background worker
        self._start_background_worker()

    def enable(self):
        """Enable data emission."""
        self._enabled = True

    def disable(self):
        """Disable data emission (for production)."""
        self._enabled = False

    def is_enabled(self) -> bool:
        """Check if emission is enabled."""
        return self._enabled

    # =========================================================================
    # Push Methods (called from training threads)
    # =========================================================================

    def push_analyst_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        train_acc: float = 0.0,
        val_acc: float = 0.0,
        direction_acc: float = 0.0,
        grad_norm: float = 0.0,
        learning_rate: float = 0.0,
        attention_weights: Optional[List[float]] = None,
        p_down: float = 0.5,
        p_up: float = 0.5,
        encoder_norms: Optional[List[float]] = None,
        context_sample: Optional[List[float]] = None,
        activations: Optional[Dict[str, List[float]]] = None,
    ):
        """Push analyst training epoch data."""
        if not self._enabled:
            return

        try:
            analyst = AnalystState(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                direction_acc=direction_acc,
                grad_norm=grad_norm,
                learning_rate=learning_rate,
                attention_weights=attention_weights or [0.5, 0.5],
                p_down=p_down,
                p_up=p_up,
                confidence=max(p_down, p_up),
                edge=p_up - p_down,
                encoder_15m_norm=encoder_norms[0] if encoder_norms else 0.0,
                encoder_1h_norm=encoder_norms[1] if encoder_norms else 0.0,
                encoder_4h_norm=encoder_norms[2] if encoder_norms else 0.0,
                context_vector_sample=context_sample[:8] if context_sample else [],
                activations=activations,
            )

            self._latest_analyst = analyst

            # Add to loss history
            with self._lock:
                self._loss_history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                })

            # Create snapshot
            snapshot = TrainingSnapshot(
                timestamp=time.time(),
                message_type="analyst_epoch",
                analyst=analyst,
                system=self._get_system_status("analyst_training"),
            )
            self._push_snapshot(snapshot)

        except Exception:
            pass  # Never block training

    def push_agent_step(
        self,
        timestep: int,
        episode: int,
        reward: float,
        action_direction: int,
        action_size: int,
        position: int,
        position_size: float,
        entry_price: Optional[float],
        current_price: float,
        unrealized_pnl: float,
        total_pnl: float,
        n_trades: int,
        atr: float = 0.0,
        chop: float = 50.0,
        adx: float = 25.0,
        regime: int = 1,
        sma_distance: float = 0.0,
        p_down: float = 0.5,
        p_up: float = 0.5,
        attention_weights: Optional[List[float]] = None,
        value_estimate: float = 0.0,
        action_probs: Optional[List[float]] = None,
        size_probs: Optional[List[float]] = None,
        reward_components: Optional[Dict[str, float]] = None,
        ohlc: Optional[Dict[str, float]] = None,
        sl_level: Optional[float] = None,
        tp_level: Optional[float] = None,
        activations: Optional[Dict[str, List[float]]] = None, # New arg
    ):
        """Push agent training step data."""
        if not self._enabled:
            return

        try:
            self._step_count = timestep

            # Update agent state
            agent = AgentState(
                timestep=timestep,
                episode=episode,
                value_estimate=value_estimate,
                action_probs=action_probs or [0.33, 0.33, 0.34],
                size_probs=size_probs or [0.25, 0.25, 0.25, 0.25],
                last_action_direction=action_direction,
                last_action_size=action_size,
            )
            self._latest_agent = agent

            # Update analyst state (probabilities)
            if self._latest_analyst is None:
                self._latest_analyst = AnalystState()
            self._latest_analyst.p_down = p_down
            self._latest_analyst.p_up = p_up
            self._latest_analyst.confidence = max(p_down, p_up)
            self._latest_analyst.edge = p_up - p_down
            if attention_weights:
                self._latest_analyst.attention_weights = attention_weights
            if activations:
                self._latest_analyst.activations = activations

            # Update market state
            ohlc_bar = None
            if ohlc:
                # Use timestamp from training data if provided, otherwise current time
                bar_timestamp = ohlc.get("timestamp", time.time())
                ohlc_bar = OHLCBar(
                    timestamp=bar_timestamp,
                    open=ohlc.get("open", current_price),
                    high=ohlc.get("high", current_price),
                    low=ohlc.get("low", current_price),
                    close=ohlc.get("close", current_price),
                )
                with self._lock:
                    self._price_history.append(ohlc_bar)

            market = MarketState(
                current_price=current_price,
                ohlc=ohlc_bar,
                position=position,
                position_size=position_size,
                entry_price=entry_price,
                unrealized_pnl=unrealized_pnl,
                total_pnl=total_pnl,
                n_trades=n_trades,
                sl_level=sl_level,
                tp_level=tp_level,
                atr=atr,
                chop=chop,
                adx=adx,
                regime=regime,
                sma_distance=sma_distance,
            )
            self._latest_market = market

            # Update reward components
            if reward_components:
                self._latest_reward = RewardComponents(**reward_components)
            else:
                self._latest_reward = RewardComponents(total=reward)

            # Add to reward history
            with self._lock:
                self._reward_history.append(reward)
                self._pnl_history.append(total_pnl)

            # Create snapshot (throttled to avoid overwhelming)
            now = time.time()
            if now - self._last_broadcast_time >= 1.0 / self.config.update_hz:
                snapshot = TrainingSnapshot(
                    timestamp=now,
                    message_type="agent_step",
                    analyst=self._latest_analyst,
                    agent=agent,
                    market=market,
                    reward=self._latest_reward,
                    system=self._get_system_status("agent_training"),
                )
                self._push_snapshot(snapshot)
                self._last_broadcast_time = now

        except Exception:
            pass  # Never block training

    def push_episode_end(
        self,
        episode: int,
        episode_reward: float,
        episode_pnl: float,
        episode_trades: int,
        win_rate: float,
        episode_length: int,
    ):
        """Push episode completion data."""
        if not self._enabled:
            return

        try:
            self._episode_count = episode

            if self._latest_agent:
                self._latest_agent.episode = episode
                self._latest_agent.episode_reward = episode_reward
                self._latest_agent.episode_pnl = episode_pnl
                self._latest_agent.episode_trades = episode_trades
                self._latest_agent.win_rate = win_rate

            snapshot = TrainingSnapshot(
                timestamp=time.time(),
                message_type="episode_end",
                agent=self._latest_agent,
                market=self._latest_market,
                system=self._get_system_status("agent_training"),
                clear_chart=True,  # Signal frontend to clear price history
            )
            self._push_snapshot(snapshot)

        except Exception:
            pass

    def push_trade(
        self,
        price: float,
        direction: int,
        size: float,
        is_entry: bool = True,
        pnl: Optional[float] = None,
        close_reason: Optional[str] = None,
        timestamp: Optional[float] = None,
    ):
        """Push trade entry/exit event."""
        if not self._enabled:
            return

        try:
            # Use provided timestamp or fall back to current time
            trade_timestamp = timestamp if timestamp is not None else time.time()
            
            trade = TradeMarker(
                timestamp=trade_timestamp,
                price=price,
                direction=direction,
                size=size,
                is_entry=is_entry,
                pnl=pnl,
                close_reason=close_reason,
            )

            with self._lock:
                self._trade_history.append(trade)

            snapshot = TrainingSnapshot(
                timestamp=time.time(),
                message_type="trade",
                trade=trade,
                market=self._latest_market,
            )
            self._push_snapshot(snapshot)

        except Exception:
            pass

    # =========================================================================
    # Pull Methods (called from WebSocket server)
    # =========================================================================

    def get_latest(self) -> Optional[TrainingSnapshot]:
        """Get the most recent snapshot (non-blocking)."""
        latest = None

        # Drain queue to get most recent
        while not self._queue.empty():
            try:
                latest = self._queue.get_nowait()
                with self._lock:
                    self._snapshot_history.append(latest)
            except queue.Empty:
                break

        return latest

    def get_all_pending(self) -> List[TrainingSnapshot]:
        """Get all pending snapshots (non-blocking)."""
        snapshots = []

        while not self._queue.empty():
            try:
                snap = self._queue.get_nowait()
                snapshots.append(snap)
                with self._lock:
                    self._snapshot_history.append(snap)
            except queue.Empty:
                break

        return snapshots

    def get_current_state(self) -> TrainingSnapshot:
        """Get complete current state for new connections."""
        with self._lock:
            return TrainingSnapshot(
                timestamp=time.time(),
                message_type="full_state",
                analyst=self._latest_analyst,
                agent=self._latest_agent,
                market=self._latest_market,
                reward=self._latest_reward,
                system=self._get_system_status(),
                price_history=list(self._price_history),
                trade_history=list(self._trade_history),
                loss_history=list(self._loss_history),
                reward_history=list(self._reward_history),
            )

    def get_history(self) -> Dict[str, Any]:
        """Get all historical data for charts."""
        with self._lock:
            return {
                "price_bars": [bar.model_dump() for bar in self._price_history],
                "trades": [t.model_dump() for t in self._trade_history],
                "loss_history": list(self._loss_history),
                "reward_history": list(self._reward_history),
                "pnl_history": list(self._pnl_history),
            }

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _push_snapshot(self, snapshot: TrainingSnapshot):
        """Push snapshot to queue or remote server."""
        # Try to push to local queue first (for server mode)
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(snapshot)
            except:
                pass

        # Also try to push to remote server (for client mode / training script)
        # We NO LONGER spawn a thread per request to avoid OOM
        pass 

    def _start_background_worker(self):
        """Start background worker for sending data."""
        if not self.config.server_mode and not hasattr(self, '_worker_thread'):
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

    def _worker_loop(self):
        """Worker loop to consume queue and send to server."""
        import requests
        while True:
            try:
                snapshot = self._queue.get()
                self._send_to_server(snapshot)
                self._queue.task_done()
            except Exception:
                pass

    def _send_to_server(self, snapshot: TrainingSnapshot):
        """Send snapshot to visualization server via HTTP."""
        try:
            import requests
            # Use orjson if available for faster serialization
            try:
                import orjson
                data = orjson.loads(snapshot.model_dump_json())
            except ImportError:
                data = snapshot.model_dump(mode='json')
            
            # Use session for connection pooling
            if not hasattr(self, '_session'):
                self._session = requests.Session()
            
            self._session.post(
                f"{self.config.frontend_url.replace('3000', '8000')}/api/ingest",
                json=data,
                timeout=0.1 
            )
        except Exception:
            pass  # Fail silently if server is down

    def _get_system_status(self, phase: str = "idle") -> SystemStatus:
        """Get current system status."""
        elapsed = time.time() - self._start_time
        steps_per_sec = self._step_count / elapsed if elapsed > 0 else 0
        eps_per_hour = (self._episode_count / elapsed) * 3600 if elapsed > 0 else 0

        # Get memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
        except:
            memory_mb = 0

        return SystemStatus(
            phase=phase,
            memory_used_mb=memory_mb,
            steps_per_second=steps_per_sec,
            episodes_per_hour=eps_per_hour,
            elapsed_seconds=elapsed,
        )

    def reset(self):
        """Reset all state (for new training run)."""
        with self._lock:
            self._latest_analyst = None
            self._latest_agent = None
            self._latest_market = None
            self._latest_reward = None
            self._snapshot_history.clear()
            self._price_history.clear()
            self._trade_history.clear()
            self._loss_history.clear()
            self._reward_history.clear()
            self._pnl_history.clear()
            self._start_time = time.time()
            self._step_count = 0
            self._episode_count = 0

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except:
                pass


# =========================================================================
# Singleton Pattern
# =========================================================================

_emitter_instance: Optional[DataEmitter] = None
_emitter_lock = threading.Lock()


def get_emitter() -> DataEmitter:
    """Get the global DataEmitter singleton."""
    global _emitter_instance

    if _emitter_instance is None:
        with _emitter_lock:
            if _emitter_instance is None:
                _emitter_instance = DataEmitter()

    return _emitter_instance


def reset_emitter():
    """Reset the global emitter (for testing)."""
    global _emitter_instance

    if _emitter_instance:
        _emitter_instance.reset()
