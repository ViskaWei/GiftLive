#!/usr/bin/env python3
"""
Optimize apply_frozen_features: Vectorized version for better performance.

Current issue: apply_frozen_features uses iterrows() which is very slow for large datasets (4.9M rows).

Optimization strategy:
1. Use vectorized operations where possible
2. Use merge/join instead of iterrows()
3. Cache lookup tables if they don't change
4. Process in chunks for very large datasets

Author: Viska Wei
Date: 2026-01-18
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import json
from datetime import datetime

BASE_DIR = Path("/home/swei20/GiftLive")
OUTPUT_DIR = BASE_DIR / "gift_EVpred"
FEATURES_DIR = OUTPUT_DIR / "features_cache"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def apply_frozen_features_optimized(df, lookups, timestamp_col='timestamp'):
    """
    Optimized version of apply_frozen_features using vectorized operations.
    
    Key optimizations:
    1. Use merge/join for pair features instead of iterrows()
    2. Use map() for user/streamer features (already vectorized)
    3. Vectorize time gap calculation
    
    Args:
        df: DataFrame with user_id, streamer_id, timestamp columns
        lookups: Dict with 'pair', 'user', 'streamer' lookup tables
        timestamp_col: Name of timestamp column
    
    Returns:
        df with frozen features added
    """
    df = df.copy()
    
    # ========================================================================
    # Pair features: Use merge instead of iterrows()
    # ========================================================================
    # Convert pair lookup to DataFrame for merge
    pair_data = []
    for key, stats in lookups['pair'].items():
        pair_data.append({
            'user_id': key[0],
            'streamer_id': key[1],
            'pair_gift_count_past': stats['pair_gift_count'],
            'pair_gift_sum_past': stats['pair_gift_sum'],
            'pair_gift_mean_past': stats['pair_gift_mean'],
            'pair_last_gift_ts': stats['pair_last_gift_ts']
        })
    
    if pair_data:
        pair_df = pd.DataFrame(pair_data)
        # Merge pair features
        df = df.merge(
            pair_df[['user_id', 'streamer_id', 'pair_gift_count_past', 
                    'pair_gift_sum_past', 'pair_gift_mean_past', 'pair_last_gift_ts']],
            on=['user_id', 'streamer_id'],
            how='left'
        )
        
        # Fill missing values
        df['pair_gift_count_past'] = df['pair_gift_count_past'].fillna(0).astype(int)
        df['pair_gift_sum_past'] = df['pair_gift_sum_past'].fillna(0.0)
        df['pair_gift_mean_past'] = df['pair_gift_mean_past'].fillna(0.0)
        
        # Calculate time gap (vectorized)
        mask = df['pair_last_gift_ts'].notna()
        df.loc[mask, 'pair_last_gift_time_gap_past'] = (
            (df.loc[mask, timestamp_col] - df.loc[mask, 'pair_last_gift_ts']) / (1000 * 3600)
        )
        df['pair_last_gift_time_gap_past'] = df['pair_last_gift_time_gap_past'].fillna(999.0)
        df = df.drop(columns=['pair_last_gift_ts'])
    else:
        # No pair data, initialize with zeros
        df['pair_gift_count_past'] = 0
        df['pair_gift_sum_past'] = 0.0
        df['pair_gift_mean_past'] = 0.0
        df['pair_last_gift_time_gap_past'] = 999.0
    
    # ========================================================================
    # User features: Already vectorized (using map)
    # ========================================================================
    df['user_total_gift_7d_past'] = df['user_id'].map(lookups['user']).fillna(0)
    df['user_budget_proxy_past'] = df['user_total_gift_7d_past']
    
    # ========================================================================
    # Streamer features: Use merge instead of iterrows()
    # ========================================================================
    streamer_data = []
    for streamer_id, stats in lookups['streamer'].items():
        streamer_data.append({
            'streamer_id': streamer_id,
            'streamer_recent_revenue_past': stats['streamer_recent_revenue'],
            'streamer_recent_unique_givers_past': stats['streamer_recent_unique_givers']
        })
    
    if streamer_data:
        streamer_df = pd.DataFrame(streamer_data)
        df = df.merge(
            streamer_df,
            on='streamer_id',
            how='left'
        )
        df['streamer_recent_revenue_past'] = df['streamer_recent_revenue_past'].fillna(0.0)
        df['streamer_recent_unique_givers_past'] = df['streamer_recent_unique_givers_past'].fillna(0).astype(int)
    else:
        df['streamer_recent_revenue_past'] = 0.0
        df['streamer_recent_unique_givers_past'] = 0
    
    return df


def save_frozen_lookups(lookups, train_min_ts, train_max_ts, save_path=None):
    """
    Save frozen lookup tables to disk for reuse.
    
    Args:
        lookups: Dict with 'pair', 'user', 'streamer' lookup tables
        train_min_ts: Minimum timestamp of train window
        train_max_ts: Maximum timestamp of train window
        save_path: Optional path to save file
    """
    if save_path is None:
        save_path = FEATURES_DIR / f"frozen_lookups_{train_min_ts}_{train_max_ts}.pkl"
    
    metadata = {
        'train_min_ts': train_min_ts,
        'train_max_ts': train_max_ts,
        'n_pairs': len(lookups['pair']),
        'n_users': len(lookups['user']),
        'n_streamers': len(lookups['streamer']),
        'created_at': datetime.now().isoformat()
    }
    
    save_data = {
        'lookups': lookups,
        'metadata': metadata
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"✅ Saved frozen lookups to: {save_path}")
    print(f"   Pairs: {metadata['n_pairs']:,}, Users: {metadata['n_users']:,}, Streamers: {metadata['n_streamers']:,}")
    
    return save_path


def load_frozen_lookups(load_path):
    """Load frozen lookup tables from disk."""
    with open(load_path, 'rb') as f:
        save_data = pickle.load(f)
    
    print(f"✅ Loaded frozen lookups from: {load_path}")
    print(f"   Pairs: {save_data['metadata']['n_pairs']:,}, "
          f"Users: {save_data['metadata']['n_users']:,}, "
          f"Streamers: {save_data['metadata']['n_streamers']:,}")
    
    return save_data['lookups'], save_data['metadata']


def test_optimization():
    """Test the optimized function with a small sample."""
    import sys
    sys.path.insert(0, str(BASE_DIR))
    
    from scripts.train_leakage_free_baseline import (
        load_data, prepare_click_level_data,
        create_past_only_features_frozen, apply_frozen_features
    )
    
    print("="*60)
    print("Testing apply_frozen_features optimization")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    gift, user, streamer, room, click = load_data()
    
    # Prepare click-level data (small sample for testing)
    print("\n2. Preparing click-level data (sample: 100K rows)...")
    click_base = prepare_click_level_data(gift, click, label_window_hours=1)
    click_sample = click_base.head(100000).copy()
    
    # Split for train
    click_sample = click_sample.sort_values('timestamp').reset_index(drop=True)
    train_end = int(len(click_sample) * 0.7)
    train_df = click_sample.iloc[:train_end].copy()
    
    # Create frozen features
    print("\n3. Creating frozen lookups...")
    lookups = create_past_only_features_frozen(gift, click, train_df)
    
    # Test original function
    print("\n4. Testing original apply_frozen_features...")
    import time
    start = time.time()
    df_original = apply_frozen_features(click_sample, lookups)
    time_original = time.time() - start
    print(f"   Time: {time_original:.2f}s")
    
    # Test optimized function
    print("\n5. Testing optimized apply_frozen_features...")
    start = time.time()
    df_optimized = apply_frozen_features_optimized(click_sample, lookups)
    time_optimized = time.time() - start
    print(f"   Time: {time_optimized:.2f}s")
    
    # Verify results match
    print("\n6. Verifying results match...")
    pair_cols = ['pair_gift_count_past', 'pair_gift_sum_past', 'pair_gift_mean_past', 
                 'pair_last_gift_time_gap_past']
    user_cols = ['user_total_gift_7d_past', 'user_budget_proxy_past']
    streamer_cols = ['streamer_recent_revenue_past', 'streamer_recent_unique_givers_past']
    
    all_match = True
    for col in pair_cols + user_cols + streamer_cols:
        if col in df_original.columns and col in df_optimized.columns:
            diff = (df_original[col] - df_optimized[col]).abs().max()
            if diff > 1e-6:
                print(f"   ❌ {col}: max diff = {diff}")
                all_match = False
            else:
                print(f"   ✅ {col}: match")
    
    if all_match:
        print("\n✅ All features match! Optimization successful.")
        speedup = time_original / time_optimized if time_optimized > 0 else float('inf')
        print(f"   Speedup: {speedup:.2f}x")
    else:
        print("\n❌ Some features don't match. Check implementation.")
    
    # Test saving/loading
    print("\n7. Testing save/load...")
    save_path = save_frozen_lookups(lookups, train_df['timestamp'].min(), train_df['timestamp'].max())
    loaded_lookups, metadata = load_frozen_lookups(save_path)
    
    # Verify loaded lookups match
    if len(loaded_lookups['pair']) == len(lookups['pair']):
        print("   ✅ Loaded lookups match original")
    else:
        print(f"   ❌ Loaded lookups don't match: {len(loaded_lookups['pair'])} vs {len(lookups['pair'])}")


if __name__ == '__main__':
    test_optimization()
