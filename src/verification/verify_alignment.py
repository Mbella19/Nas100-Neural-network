import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import Config
from src.data.loader import load_ohlcv
from src.data.components import load_component_data, PRIMARY_COMPONENTS

def verify_alignment():
    print("="*60)
    print("VERIFYING NAS100 vs COMPONENT DATA ALIGNMENT")
    print("="*60)
    
    config = Config()
    
    # 1. Load Main NAS100 Data
    nas100_path = config.paths.data_raw / config.data.raw_file
    print(f"Loading NAS100 data from: {nas100_path}")
    
    if not nas100_path.exists():
        print(f"ERROR: NAS100 file not found at {nas100_path}")
        return

    try:
        df_nas100 = load_ohlcv(nas100_path, datetime_format=config.data.datetime_format)
    except Exception as e:
        print(f"ERROR loading NAS100: {e}")
        return
        
    print(f"NAS100 Data: {len(df_nas100):,} rows")
    print(f"Range: {df_nas100.index.min()} to {df_nas100.index.max()}")
    print("-" * 60)
    
    # 2. Load Component Data
    components_dir = config.paths.components_dir
    print(f"Loading Components from: {components_dir}")
    
    if not components_dir.exists():
        print(f"ERROR: Components directory not found at {components_dir}")
        return
        
    components = load_component_data(components_dir)
    
    if not components:
        print("ERROR: No component files loaded.")
        return
        
    print(f"Loaded {len(components)} components: {list(components.keys())}")
    print("-" * 60)
    
    # 3. Analyze Alignment
    print(f"{'COMPONENT':<10} | {'ROWS':<10} | {'START DATE':<20} | {'END DATE':<20} | {'COVERAGE %':<10} | {'MISSING':<10}")
    print("-" * 90)
    
    nas100_idx = df_nas100.index
    total_nas100 = len(nas100_idx)
    
    perfect_alignment = True
    
    # Check specific primary components
    for ticker in PRIMARY_COMPONENTS:
        if ticker not in components:
            print(f"{ticker:<10} | {'MISSING':<10} | {'N/A':<20} | {'N/A':<20} | {'0.0%':<10} | {total_nas100:<10}")
            perfect_alignment = False
            continue
            
        df_comp = components[ticker]
        
        # Intersection
        # We reindex component to match NAS100 exactly (without filling yet) to see raw overlap
        common = df_comp.reindex(nas100_idx)
        valid_count = common['close'].notna().sum()
        missing_count = total_nas100 - valid_count
        coverage = (valid_count / total_nas100) * 100
        
        start_str = str(df_comp.index.min())[:19]
        end_str = str(df_comp.index.max())[:19]
        
        print(f"{ticker:<10} | {len(df_comp):<10,} | {start_str:<20} | {end_str:<20} | {coverage:6.1f}% | {missing_count:<10,}")
        
        if coverage < 95.0:
            perfect_alignment = False
            
    print("-" * 90)
    
    if perfect_alignment:
        print("\nSUCCESS: High alignment detected (>95%) for all primary components.")
    else:
        print("\nWARNING: Some components have significant gaps or mismatches with NAS100 data.")
        print("This may affect the accuracy of the Cross-Asset Attention module.")
        print("Recommendation: Ensure component data covers the same historical range as NAS100.")

if __name__ == "__main__":
    verify_alignment()
