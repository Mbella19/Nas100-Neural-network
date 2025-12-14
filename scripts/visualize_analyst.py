#!/usr/bin/env python3
"""
Standalone Analyst Visualization Tool.

This script runs a lightweight Flask server to visualize the Analyst model's
performance on the last month of data. It is completely separate from the
main training dashboard.

Usage:
    python scripts/visualize_analyst.py

Dependencies:
    pip install flask flask-socketio
"""

import sys
import os
import time
import threading
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config, get_device
from src.models.analyst import load_analyst
from src.data.features import get_feature_columns
from src.data.components import load_component_sequences

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try importing Flask dependencies
try:
    from flask import Flask, render_template, send_from_directory
    from flask_socketio import SocketIO
except ImportError:
    logger.error("Flask or Flask-SocketIO not found.")
    logger.info("Please run: pip install flask flask-socketio eventlet")
    sys.exit(1)

# Initialize Flask
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'analyst_secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class SimulationState:
    def __init__(self):
        self.running = False
        self.paused = False
        self.speed = 1.0  # Seconds per step (lower is faster)
        self.current_idx = 0
        self.data = None
        self.model = None
        self.device = None
        self.lookbacks = {}
        self.feature_cols = []
        self.component_sequences = None

state = SimulationState()

def prepare_data():
    """Load model and data for the last 30 days."""
    config = Config()
    device = get_device()
    state.device = device

    # 1. Load Data
    logger.info("Loading data...")
    try:
        # Load normalized data for model (aligned 5m/15m/45m dataset)
        df_5m = pd.read_parquet(config.paths.data_processed / 'features_5m_normalized.parquet')
        df_15m = pd.read_parquet(config.paths.data_processed / 'features_15m_normalized.parquet')
        df_45m = pd.read_parquet(config.paths.data_processed / 'features_45m_normalized.parquet')

        # Load raw data for visualization (prices)
        df_5m_raw = pd.read_parquet(config.paths.data_processed / 'features_5m.parquet')
    except FileNotFoundError:
        logger.error("Processed data not found. Run pipeline first.")
        sys.exit(1)

    # 2. Filter last 30 days (5-minute bars: 12*24 per day)
    days_to_visualize = 30
    bars_5m = 12 * 24 * days_to_visualize
    
    # Ensure we have enough data
    if len(df_5m) < bars_5m:
        logger.warning(f"Not enough data for {days_to_visualize} days. Using all available.")
        start_idx = 0
    else:
        start_idx = len(df_5m) - bars_5m

    # Align all dataframes
    df_5m = df_5m.iloc[start_idx:].reset_index(drop=True)
    df_5m_raw = df_5m_raw.iloc[start_idx:].reset_index(drop=True)  # Align raw data
    df_15m = df_15m.iloc[start_idx:].reset_index(drop=True)
    df_45m = df_45m.iloc[start_idx:].reset_index(drop=True)

    # 3. Load Model
    analyst_path = config.paths.models_analyst / 'best.pt'
    if not analyst_path.exists():
        logger.error(f"Model not found at {analyst_path}")
        sys.exit(1)

    # Define features (must match training)
    feature_cols = get_feature_columns()
    feature_cols = [c for c in feature_cols if c in df_5m.columns]
    state.feature_cols = feature_cols

    feature_dims = {
        '5m': len(feature_cols),
        '15m': len(feature_cols),
        '45m': len(feature_cols)
    }
    
    logger.info(f"Loading model from {analyst_path}")
    state.model = load_analyst(str(analyst_path), feature_dims, device, freeze=True)
    state.model.eval()

    # 4. Prepare Tensor Data
    state.lookbacks = {
        '5m': getattr(config.analyst, 'lookback_5m', 48),
        '15m': getattr(config.analyst, 'lookback_15m', 16),
        '45m': getattr(config.analyst, 'lookback_45m', 6)
    }

    # Optional: load precomputed component sequences for cross-asset attention
    component_path = getattr(config.paths, 'component_sequences', None)
    if component_path is not None and Path(component_path).exists():
        try:
            sequences, _ = load_component_sequences(component_path)
            state.component_sequences = sequences[start_idx:].astype(np.float32)
            logger.info(f"Loaded component sequences for visualization: {state.component_sequences.shape}")
        except Exception as e:
            logger.warning(f"Failed to load component sequences: {e}")
            state.component_sequences = None

    # Store dataframes for access
    state.data = {
        '5m': df_5m,
        '15m': df_15m,
        '45m': df_45m,
        'raw_close': df_5m_raw['close'].values  # Use RAW close prices
    }
    
    logger.info(f"Ready to visualize {len(df_5m)} steps.")

def simulation_loop():
    """Background thread to run the simulation."""
    logger.info("Simulation loop started.")
    
    # Warmup period
    max_lookback = max(state.lookbacks.values())
    state.current_idx = max_lookback
    
    while True:
        if not state.running:
            time.sleep(0.1)
            continue
            
        if state.paused:
            time.sleep(0.1)
            continue
            
        if state.current_idx >= len(state.data['5m']):
            logger.info("Simulation finished. Restarting.")
            state.current_idx = max_lookback
            
        # 1. Prepare Input
        idx = state.current_idx
        
        # Extract windows
        # Note: This is a simplified extraction. In real pipeline we handle subsampling carefully.
        # Here we assume 1h/4h are aligned row-by-row (which they are in processed data usually)
        # But wait, processed data might be subsampled? 
        # The pipeline aligns them to 15m index. So we can just index them directly.
        
        def get_window(df, idx, lookback):
            if idx < lookback: return np.zeros((lookback, len(state.feature_cols)))
            return df[state.feature_cols].iloc[idx-lookback:idx].values.astype(np.float32)

        x_5m = get_window(state.data['5m'], idx, state.lookbacks['5m'])
        x_15m = get_window(state.data['15m'], idx, state.lookbacks['15m'])
        x_45m = get_window(state.data['45m'], idx, state.lookbacks['45m'])
        
        # To Tensor
        t_5m = torch.tensor(x_5m, device=state.device).unsqueeze(0) # [1, L, F]
        t_15m = torch.tensor(x_15m, device=state.device).unsqueeze(0)
        t_45m = torch.tensor(x_45m, device=state.device).unsqueeze(0)

        component_tensor = None
        if state.component_sequences is not None and idx < len(state.component_sequences):
            component_tensor = torch.tensor(
                state.component_sequences[idx],
                device=state.device
            ).unsqueeze(0)
        
        # 2. Model Inference
        with torch.no_grad():
            # Get activations and probs
            if hasattr(state.model, 'get_activations'):
                if component_tensor is not None and hasattr(state.model, 'use_cross_asset_attention'):
                    context, activations = state.model.get_activations(
                        t_5m, t_15m, t_45m, component_data=component_tensor
                    )
                    _, probs = state.model.get_probabilities(
                        t_5m, t_15m, t_45m, component_data=component_tensor
                    )
                else:
                    context, activations = state.model.get_activations(t_5m, t_15m, t_45m)
                    _, probs = state.model.get_probabilities(t_5m, t_15m, t_45m)
                
                # Process activations for JSON
                act_data = {k: v[0].cpu().numpy().tolist() for k, v in activations.items()}
            else:
                # Fallback
                probs = torch.tensor([[0.5, 0.5]], device=state.device)
                act_data = {}

        # 3. Emit Data
        current_price = float(state.data['raw_close'][idx-1]) # Previous close is current price effectively
        
        p_down = float(probs[0, 0].item())
        p_up = float(probs[0, 1].item())
        
        data = {
            'step': idx,
            'price': current_price,
            'p_up': p_up,
            'p_down': p_down,
            'activations': act_data,
            'timestamp': datetime.now().isoformat() # Simulated time
        }
        
        socketio.emit('new_step', data)
        
        # Advance
        state.current_idx += 1
        
        # Sleep
        time.sleep(state.speed)

# Routes
@app.route('/')
def index():
    return render_template('analyst_dashboard.html')

@app.route('/start')
def start():
    state.running = True
    state.paused = False
    return {'status': 'started'}

@app.route('/pause')
def pause():
    state.paused = True
    return {'status': 'paused'}

@app.route('/reset')
def reset():
    state.current_idx = max(state.lookbacks.values())
    return {'status': 'reset'}

@app.route('/speed/<float:speed>')
def set_speed(speed):
    state.speed = max(0.01, min(2.0, speed))
    return {'status': 'speed_set', 'speed': state.speed}

if __name__ == '__main__':
    prepare_data()
    
    # Start simulation thread
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    
    logger.info("Starting Flask server at http://localhost:5001")
    socketio.run(app, host='0.0.0.0', port=5001, debug=False)
