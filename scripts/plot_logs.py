import re
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

def parse_log_file(log_path):
    steps = []
    rewards = []
    trades = []
    
    pnl_steps = []
    pnls = []

    with open(log_path, 'r') as f:
        content = f.read()

    # Updated parsing logic to handle "Training Progress @" blocks which contain all info
    blocks = content.split("Training Progress @")
    for block in blocks[1:]: # Skip first empty split
        try:
            # Extract step number
            step_match = re.search(r"^ (\d+) steps:", block)
            if not step_match:
                continue
            current_step = int(step_match.group(1))
            
            # Extract Reward
            reward_match = re.search(r"Avg Episode Reward: ([-\d.]+)", block)
            if reward_match:
                rewards.append(float(reward_match.group(1)))
                steps.append(current_step)
            
            # Extract PnL
            pnl_match = re.search(r"Avg PnL: ([-\d.]+) pips", block)
            if pnl_match:
                pnl_steps.append(current_step)
                pnls.append(float(pnl_match.group(1)))
                
            # Extract Trades (Action Distribution)
            # Action Distribution: Flat=64.2%, Long=18.8%, Short=17.0%
            action_match = re.search(r"Action Distribution: Flat=([-\d.]+)%, Long=([-\d.]+)%, Short=([-\d.]+)%", block)
            if action_match:
                flat_pct = float(action_match.group(1))
                long_pct = float(action_match.group(2))
                short_pct = float(action_match.group(3))
                # Approximate "Trades per Episode" based on Long+Short percentage * 500 steps (Episode Length)
                # This is an estimate since we don't have the exact "Trades" count logged explicitly here, 
                # but (Long% + Short%) * 500 gives active bars. 
                # Note: One trade lasts multiple bars, so this overestimates *trades*, but accurately reflects *activity*.
                # Actually, wait. The log usually says "Avg Trade Duration". If not, we use Activity %.
                # Let's label it "Active Steps %" or similar.
                active_pct = long_pct + short_pct
                trades.append(active_pct) # Plotting Activity % instead of raw trade count is fine/better
                
        except Exception as e:
            continue

    df_metrics = pd.DataFrame({'step': steps, 'mean_reward': rewards, 'mean_trades': trades})
    df_pnl = pd.DataFrame({'step': pnl_steps, 'avg_pnl': pnls})
    
    return df_metrics, df_pnl

def plot_training_progress(metrics_df, pnl_df, save_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Plot Mean Reward
    axes[0].plot(metrics_df['step'], metrics_df['mean_reward'], color='#00ff00', linewidth=1.5)
    axes[0].set_title('Mean Reward per Episode', fontsize=14, color='white')
    axes[0].set_ylabel('Reward', fontsize=12, color='white')
    axes[0].grid(True, alpha=0.2)
    axes[0].set_facecolor('#1e1e1e')
    
    # Plot Mean Trades
    axes[1].plot(metrics_df['step'], metrics_df['mean_trades'], color='#00ccff', linewidth=1.5)
    axes[1].set_title('Mean Trades per Episode (Activity)', fontsize=14, color='white')
    axes[1].set_ylabel('Trades', fontsize=12, color='white')
    axes[1].grid(True, alpha=0.2)
    axes[1].set_facecolor('#1e1e1e')
    
    # Plot Avg PnL
    if not pnl_df.empty:
        axes[2].plot(pnl_df['step'], pnl_df['avg_pnl'], color='#ff9900', linewidth=2)
    axes[2].set_title('Average PnL (Pips)', fontsize=14, color='white')
    axes[2].set_xlabel('Training Steps', fontsize=12, color='white')
    axes[2].set_ylabel('PnL (Pips)', fontsize=12, color='white')
    axes[2].grid(True, alpha=0.2)
    axes[2].set_facecolor('#1e1e1e')
    
    # Style formatting
    fig.patch.set_facecolor('#121212')
    for ax in axes:
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        for spine in ax.spines.values():
            spine.set_color('#444444')

    plt.tight_layout()
    plt.savefig(save_path, facecolor='#121212')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_logs.py <log_file>")
        sys.exit(1)
        
    log_file = sys.argv[1]
    save_file = log_file.replace('.log', '_chart.png')
    
    df_m, df_p = parse_log_file(log_file)
    print(f"Found {len(df_m)} metric points and {len(df_p)} PnL points.")
    plot_training_progress(df_m, df_p, save_file)
