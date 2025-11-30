# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **hybrid trading system for EURUSD** combining supervised learning (Market Analyst) and reinforcement learning (PPO Agent). The project is optimized for Apple M2 Silicon with 8GB RAM constraints.

## System Architecture

**Two-Phase Hierarchical Design:**

1. **Phase A - Market Analyst (Supervised Learning)**
   - Multi-timeframe encoder (15m, 1H, 4H)
   - Produces context vectors for RL agent
   - Trained on smoothed future returns (NOT next-step price)
   - Frozen after pre-training

2. **Phase B - Sniper Agent (Reinforcement Learning)**
   - PPO policy consuming Analyst context + market state
   - Actions: `[0: Flat/Exit, 1: Long, 2: Short]`
   - Reward engineering: PnL, transaction costs, FOMO penalty, chop avoidance

## Critical Hardware Constraints (Apple M2, 8GB RAM)

**ALWAYS enforce these rules:**

- Device: `device="mps"` (Metal Performance Shaders)
- Precision: `torch.float32` only - **NEVER use float64**
- Batch sizes: 32-64 max
- Clear cache regularly: `torch.mps.empty_cache(); gc.collect()`
- Process data in chunks, never load full dataset to GPU
- Use `torch.no_grad()` during inference
- Delete intermediate tensors immediately

## Tech Stack

Python 3.10+, PyTorch 2.0+, Stable Baselines 3, Gymnasium, Pandas, NumPy, pandas-ta/TA-Lib

## Data Requirements

- Source: EURUSD 1-minute OHLCV (5 years, ~2.6M rows)
- Multi-timeframe: Resample 1m → 15m, 1H, 4H using forward-fill on complete datetime index
- Gap handling: Create complete `pd.date_range()` index, then `.reindex().ffill()`

## Feature Engineering Categories

1. **Price Action Patterns**: Pinbar (wick > 2× body), Engulfing, Doji (body < 10% range)
2. **Market Structure**: Fractal S/R levels, ATR-normalized distance to S/R
3. **Trend Filters**: SMA(200) distance/ATR, EMA crossovers
4. **Regime Detection**: Choppiness Index (>61.8 = ranging, <38.2 = trending), ADX (>25 = trending)

## Key Implementation Details

### Market Analyst (src/models/analyst.py)

- Separate `TemporalEncoder` for each timeframe (15m, 1H, 4H)
- `AttentionFusion` layer combines to produce context vector `[batch, context_dim]`
- **Target**: Smoothed future return (NOT next-step price): `df['close'].shift(-12).rolling(12).mean() / df['close'] - 1`
- Training: AdamW optimizer, lr=1e-4 to 3e-4, batch_size=32-64, MSE/Huber loss, early stopping
- After training, **freeze all parameters** and set to `.eval()` mode

### Trading Environment (src/environments/trading_env.py)

**Action Space**: `gym.spaces.Discrete(3)`
- `0`: Flat/Exit (if position != 0, close immediately)
- `1`: Long (close short if exists, open long if flat)
- `2`: Short (close long if exists, open short if flat)

**Observation Space**: `gym.spaces.Box` containing context vector + market features (all float32)

**Reward Function**:
```python
reward = pnl_pips - (SPREAD_PIPS if opened_trade else 0) + fomo_penalty + chop_penalty
# FOMO: -0.5 if flat during high-momentum moves (|price_move| > 2*ATR)
# Chop: -0.3 if holding position when CHOP > 60
```

### PPO Agent (src/agents/sniper_agent.py)

**Stable Baselines 3 Configuration**:
```python
PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=64,
    n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, device="mps")
```

**MPS Fallback**: If MPS fails, fall back to CPU for SB3 (limited MPS support)

## Execution Pipeline

**Full pipeline** (`scripts/run_pipeline.py`):
1. Load & clean 1m OHLCV → resample to 15m/1H/4H → align timeframes → engineer features
2. Train Market Analyst on smoothed future returns (85% train, 15% val)
3. Freeze Analyst weights (`param.requires_grad = False`, `.eval()`)
4. Initialize TradingEnv with frozen Analyst → train PPO agent (500k timesteps)
5. Run out-of-sample backtest (final 15%) → compare to buy-and-hold baseline

**Individual modules**:
- Train Analyst only: `python -m src.training.train_analyst`
- Train Agent only: `python -m src.training.train_agent` (requires pre-trained Analyst)
- Run backtest: `python -m src.evaluation.backtest`

## Directory Structure

```
├── requirements.txt
├── config/settings.py           # Hyperparameters & constants
├── data/
│   ├── raw/                     # Raw 1m OHLCV
│   └── processed/               # Multi-timeframe data
├── src/
│   ├── data/                    # loader.py, resampler.py, features.py
│   ├── models/                  # analyst.py, encoders.py, fusion.py
│   ├── environments/            # trading_env.py (Gymnasium)
│   ├── agents/                  # sniper_agent.py (PPO wrapper)
│   ├── training/                # train_analyst.py, train_agent.py
│   └── evaluation/              # backtest.py, metrics.py
├── scripts/run_pipeline.py      # Main execution
├── notebooks/                   # exploration.ipynb
└── models/                      # Saved checkpoints
    ├── analyst/
    └── agent/
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Sortino Ratio | > 1.5 |
| Total Return | Beat buy-and-hold |
| Max Drawdown | < 20% |
| Win Rate | > 50% |
| Profit Factor | > 1.5 |

## Critical Code Patterns

### Device & Memory Management
```python
# Device selection
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# Always float32
tensor = tensor.to(device=device, dtype=torch.float32)
df = df.astype(np.float32)

# Clear cache every 50 batches
if batch_idx % 50 == 0:
    torch.mps.empty_cache()
    gc.collect()
    del intermediate_tensors
```

### SB3 MPS Fallback
```python
try:
    agent = PPO("MlpPolicy", env, device="mps")
except:
    agent = PPO("MlpPolicy", env, device="cpu")  # SB3 has limited MPS support
```

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| OOM on M2 | Reduce batch size, gradient checkpointing |
| float64 tensors | Enforce float32 everywhere |
| Look-ahead bias | Features use only past data |
| Overfitting | Dropout, early stopping, validation split |
| Passive agent | FOMO penalty, entropy bonus |
| Over-trading | Transaction cost modeling |
| Choppy markets | Chop avoidance penalty |

## Validation Checklist

- Multi-timeframe data aligned without NaN values
- Analyst loss decreases, context vectors have variance (not collapsed)
- Environment returns valid (obs, reward, done, info)
- Agent explores diverse actions (not stuck on one)
- Memory usage < 6GB, all tensors float32
- No look-ahead bias in features

## Commands

```bash
pip install -r requirements.txt          # Install dependencies
python scripts/run_pipeline.py           # Full pipeline
python -m src.training.train_analyst     # Train Analyst only
python -m src.training.train_agent       # Train Agent only
python -m src.evaluation.backtest        # Run backtest
```
