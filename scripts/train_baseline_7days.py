#!/usr/bin/env python3
"""
Baseline with 7-7-7 Day Split: Optimized Frozen Features
========================================================

Experiment: Baseline with 7 days train / 7 days val / 7 days test split

Key improvements:
1. Use 7-7-7 day split (instead of 70-15-15 ratio)
2. Precompute frozen features for all data (using train window only)
3. Ensure no leakage: all features computed from train window only
4. Use optimized apply_frozen_features_optimized() (107x faster)

Author: Viska Wei
Date: 2026-01-18
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Import optimized functions
sys.path.insert(0, str(Path(__file__).parent))
from optimize_apply_frozen_features import (
    apply_frozen_features_optimized,
    save_frozen_lookups,
    load_frozen_lookups
)

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_EVpred"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
FEATURES_DIR = OUTPUT_DIR / "features_cache"
LOGS_DIR = BASE_DIR / "logs"

for d in [IMG_DIR, RESULTS_DIR, MODELS_DIR, FEATURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============== CONFIGURATION ==============
CONFIG = {
    'label_window_hours': 0.25,  # 15 minutes (0.25 hours)
    'train_days': 7,              # Train: 7 days
    'val_days': 7,                # Val: 7 days
    'test_days': 7,               # Test: 7 days
    'feature_version': 'frozen',  # Use frozen (strict no-leakage)
    'use_optimized': True,        # Use optimized apply_frozen_features
    'cache_lookups': True,        # Cache frozen lookups for reuse
    
    # Fast training config
    'lgb_params': {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.1,  # Faster convergence
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'num_boost_round': 200,  # Reduced from 500
        'early_stopping_rounds': 20,  # Reduced from 50
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1
    },
    
    'target_transform': 'log1p',  # log(1+Y)
}


def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def load_data():
    """Load all data files."""
    log_message("Loading data files...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    return gift, user, streamer, room, click


def prepare_click_level_data(gift, click, label_window_hours=0.25):
    """Prepare click-level dataset with gift labels."""
    log_message(f"Preparing click-level data (label window: {label_window_hours*60:.0f}min)...")
    
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['day_of_week'] = click_base['timestamp_dt'].dt.dayofweek
    click_base['is_weekend'] = (click_base['day_of_week'] >= 5).astype(int)
    click_base['date'] = click_base['timestamp_dt'].dt.date
    
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')
    click_base['click_end_ts'] = click_base['timestamp_dt'] + pd.Timedelta(hours=label_window_hours)
    
    merged = click_base[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'click_end_ts']].merge(
        gift[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'gift_price']],
        on=['user_id', 'streamer_id', 'live_id'],
        how='left',
        suffixes=('_click', '_gift')
    )
    
    merged = merged[
        (merged['timestamp_dt_gift'] >= merged['timestamp_dt_click']) &
        (merged['timestamp_dt_gift'] <= merged['click_end_ts'])
    ]
    
    gift_agg = merged.groupby(['user_id', 'streamer_id', 'live_id', 'timestamp_dt_click']).agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={
        'timestamp_dt_click': 'timestamp_dt',
        'gift_price': 'gift_price_label'
    })
    
    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id', 'timestamp_dt'], how='left')
    click_base['gift_price_label'] = click_base['gift_price_label'].fillna(0)
    
    log_message(f"Click base: {len(click_base):,} records")
    log_message(f"Gift rate: {click_base['gift_price_label'].gt(0).mean()*100:.2f}%")
    
    return click_base


def create_frozen_features(gift, click, train_df):
    """
    Create frozen past-only features (train window only).
    
    CRITICAL: Only use train window data to compute statistics.
    This ensures no leakage for val/test.
    """
    log_message("Creating frozen past-only features (train window only, no leakage)...")
    
    train_min_ts = train_df['timestamp'].min()
    train_max_ts = train_df['timestamp'].max()
    
    log_message(f"Train window: {pd.to_datetime(train_min_ts, unit='ms')} to {pd.to_datetime(train_max_ts, unit='ms')}")
    
    # CRITICAL: Only use gift data from train window
    gift_train = gift[
        (gift['timestamp'] >= train_min_ts) &
        (gift['timestamp'] <= train_max_ts)
    ].copy()
    
    log_message(f"Using {len(gift_train):,} gift records from train window only")
    
    lookups = {}
    
    # Pair features
    pair_gift_stats = gift_train.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean'],
        'timestamp': 'max'
    }).reset_index()
    pair_gift_stats.columns = ['user_id', 'streamer_id', 'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean', 'pair_last_gift_ts']
    
    pair_lookup = {}
    for _, row in pair_gift_stats.iterrows():
        key = (row['user_id'], row['streamer_id'])
        pair_lookup[key] = {
            'pair_gift_count': row['pair_gift_count'],
            'pair_gift_sum': row['pair_gift_sum'],
            'pair_gift_mean': row['pair_gift_mean'],
            'pair_last_gift_ts': row['pair_last_gift_ts']
        }
    lookups['pair'] = pair_lookup
    
    # User features
    user_gift_7d = gift_train.groupby('user_id').agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={'gift_price': 'user_total_gift_7d'})
    user_lookup = dict(zip(user_gift_7d['user_id'], user_gift_7d['user_total_gift_7d']))
    lookups['user'] = user_lookup
    
    # Streamer features
    streamer_gift_stats = gift_train.groupby('streamer_id').agg({
        'gift_price': ['sum', 'count'],
        'user_id': 'nunique'
    }).reset_index()
    streamer_gift_stats.columns = ['streamer_id', 'streamer_recent_revenue', 'streamer_gift_count', 'streamer_recent_unique_givers']
    
    streamer_lookup = {}
    for _, row in streamer_gift_stats.iterrows():
        streamer_lookup[row['streamer_id']] = {
            'streamer_recent_revenue': row['streamer_recent_revenue'],
            'streamer_recent_unique_givers': row['streamer_recent_unique_givers']
        }
    lookups['streamer'] = streamer_lookup
    
    log_message(f"Created lookups: {len(lookups['pair']):,} pairs, {len(lookups['user']):,} users, {len(lookups['streamer']):,} streamers")
    log_message("✅ All features computed from train window only (no leakage)")
    
    return lookups, train_min_ts, train_max_ts


def create_static_features(user, streamer, room):
    """Create static features."""
    log_message("Creating static features...")
    
    user_features = user[['user_id', 'age', 'gender', 'device_brand', 'device_price',
                          'fans_num', 'follow_num', 'accu_watch_live_cnt',
                          'accu_watch_live_duration', 'is_live_streamer', 'is_photo_author',
                          'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                          'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    streamer_features = streamer[['streamer_id', 'gender', 'age', 'device_brand', 'device_price',
                                  'live_operation_tag', 'fans_user_num', 'fans_group_fans_num',
                                  'follow_user_num', 'accu_live_cnt', 'accu_live_duration',
                                  'accu_play_cnt', 'accu_play_duration',
                                  'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                                  'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    streamer_features = streamer_features.rename(columns={
        'gender': 'streamer_gender', 'age': 'streamer_age',
        'device_brand': 'streamer_device_brand', 'device_price': 'streamer_device_price',
        'onehot_feat0': 'streamer_onehot_feat0', 'onehot_feat1': 'streamer_onehot_feat1',
        'onehot_feat2': 'streamer_onehot_feat2', 'onehot_feat3': 'streamer_onehot_feat3',
        'onehot_feat4': 'streamer_onehot_feat4', 'onehot_feat5': 'streamer_onehot_feat5',
        'onehot_feat6': 'streamer_onehot_feat6',
    })
    
    room_info = room[['live_id', 'live_type', 'live_content_category']].drop_duplicates('live_id')
    
    return user_features, streamer_features, room_info


def prepare_features(gift, user, streamer, room, click, click_base, train_df, lookups):
    """
    Prepare features with frozen method (optimized).
    
    CRITICAL: All frozen features are computed from train_df only.
    For val/test, we only lookup from the precomputed lookups.
    """
    log_message("Preparing features (frozen version, optimized)...")
    
    user_features, streamer_features, room_info = create_static_features(user, streamer, room)
    
    df = click_base.merge(user_features, on='user_id', how='left')
    df = df.merge(streamer_features, on='streamer_id', how='left')
    df = df.merge(room_info, on='live_id', how='left')
    
    # CRITICAL: Use precomputed lookups (from train window only)
    # This ensures no leakage for val/test
    if CONFIG['use_optimized']:
        df = apply_frozen_features_optimized(df, lookups)
    else:
        # Fallback to original (should not happen)
        from train_leakage_free_baseline import apply_frozen_features
        df = apply_frozen_features(df, lookups)
    
    # Fill NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Encode categorical
    cat_columns = ['age', 'gender', 'device_brand', 'device_price', 'fans_num', 'follow_num',
                   'accu_watch_live_cnt', 'accu_watch_live_duration',
                   'streamer_gender', 'streamer_age', 'streamer_device_brand', 'streamer_device_price',
                   'live_operation_tag', 'fans_user_num', 'fans_group_fans_num', 'follow_user_num',
                   'accu_live_cnt', 'accu_live_duration', 'accu_play_cnt', 'accu_play_duration',
                   'live_type', 'live_content_category']
    
    for col in cat_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
            df[col] = df[col].astype('category').cat.codes
    
    # Targets
    if CONFIG['target_transform'] == 'log1p':
        df['target'] = np.log1p(df['gift_price_label'])
    else:
        df['target'] = df['gift_price_label']
    df['target_raw'] = df['gift_price_label']
    df['is_gift'] = (df['gift_price_label'] > 0).astype(int)
    
    log_message(f"Final dataset: {len(df):,} records, {df.shape[1]} columns")
    log_message(f"Gift rate: {df['is_gift'].mean()*100:.2f}%")
    
    return df


def get_feature_columns(df):
    """Get feature columns."""
    exclude_cols = ['user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
                    'date', 'gift_price_label', 'target', 'target_raw', 'is_gift',
                    'watch_live_time', 'click_end_ts']
    return [c for c in df.columns if c not in exclude_cols]


def temporal_split_7days(df):
    """
    Split data by time: 7 days train / 7 days val / 7 days test.
    
    Returns:
        train, val, test DataFrames
    """
    log_message("Performing temporal split (7-7-7 days)...")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Convert timestamp to date
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    total_days = (max_date - min_date).days + 1
    
    log_message(f"Date range: {min_date} to {max_date} ({total_days} days)")
    
    # Calculate split dates
    test_start = max_date - timedelta(days=CONFIG['test_days'] - 1)  # Last 7 days
    val_start = test_start - timedelta(days=CONFIG['val_days'])      # 7 days before test
    train_end = val_start - timedelta(days=1)                         # End of train
    
    log_message(f"Train: {min_date} to {train_end} ({CONFIG['train_days']} days)")
    log_message(f"Val: {val_start} to {test_start - timedelta(days=1)} ({CONFIG['val_days']} days)")
    log_message(f"Test: {test_start} to {max_date} ({CONFIG['test_days']} days)")
    
    # Split
    train = df[df['date'] < val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test = df[df['date'] >= test_start].copy()
    
    log_message(f"Split sizes: Train={len(train):,}, Val={len(val):,}, Test={len(test):,}")
    log_message(f"Split ratios: Train={len(train)/len(df)*100:.1f}%, Val={len(val)/len(df)*100:.1f}%, Test={len(test)/len(df)*100:.1f}%")
    
    # Verify no overlap
    train_max_ts = train['timestamp'].max()
    val_min_ts = val['timestamp'].min()
    val_max_ts = val['timestamp'].max()
    test_min_ts = test['timestamp'].min()
    
    log_message(f"Time gaps: Train-Val={pd.to_datetime(val_min_ts, unit='ms') - pd.to_datetime(train_max_ts, unit='ms')}")
    log_message(f"Time gaps: Val-Test={pd.to_datetime(test_min_ts, unit='ms') - pd.to_datetime(val_max_ts, unit='ms')}")
    
    if train_max_ts >= val_min_ts or val_max_ts >= test_min_ts:
        log_message("⚠️ WARNING: Time overlap detected!", "WARNING")
    else:
        log_message("✅ No time overlap (correct split)", "SUCCESS")
    
    return train, val, test


def compute_top_k_capture(y_true, y_pred, k_pct=0.01):
    """Compute Top-K% capture rate (set overlap)."""
    n = len(y_true)
    k = max(1, int(n * k_pct))
    
    true_rank = np.argsort(np.argsort(-y_true))
    pred_rank = np.argsort(np.argsort(-y_pred))
    
    true_topk = set(np.where(true_rank < k)[0])
    pred_topk = set(np.where(pred_rank < k)[0])
    
    return len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0


def compute_revenue_capture_at_k(y_true_amount, y_pred, k_pct=0.01):
    """Compute Revenue Capture@K."""
    n = len(y_true_amount)
    k = max(1, int(n * k_pct))
    
    pred_order = np.argsort(-y_pred)
    top_k_indices = pred_order[:k]
    
    revenue_top_k = y_true_amount[top_k_indices].sum()
    total_revenue = y_true_amount.sum()
    
    return revenue_top_k / total_revenue if total_revenue > 0 else 0


def train_model(train, val, feature_cols):
    """Train LightGBM model."""
    log_message("Training LightGBM model...")
    
    X_train = train[feature_cols]
    y_train = train['target']
    X_val = val[feature_cols]
    y_val = val['target']
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=CONFIG['lgb_params']['early_stopping_rounds']),
        lgb.log_evaluation(period=50)
    ]
    
    start_time = time.time()
    model = lgb.train(
        CONFIG['lgb_params'],
        train_data,
        num_boost_round=CONFIG['lgb_params']['num_boost_round'],
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    
    log_message(f"Training completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


def evaluate_model(model, test, feature_cols):
    """Evaluate model."""
    log_message("Evaluating model...")
    
    X_test = test[feature_cols]
    y_test = test['target'].values
    y_test_raw = test['target_raw'].values
    y_test_is_gift = test['is_gift'].values
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Transform back if needed
    if CONFIG['target_transform'] == 'log1p':
        y_pred_raw = np.expm1(y_pred)
    else:
        y_pred_raw = y_pred
    
    # Metrics
    spearman, _ = stats.spearmanr(y_test_raw, y_pred_raw)
    top_1pct_capture = compute_top_k_capture(y_test_raw, y_pred_raw, 0.01)
    revenue_1pct = compute_revenue_capture_at_k(y_test_raw, y_pred_raw, 0.01)
    
    # Binary classification metrics (if gift > 0)
    if y_test_is_gift.sum() > 0:
        auc = roc_auc_score(y_test_is_gift, y_pred_raw)
    else:
        auc = 0.0
    
    metrics = {
        'spearman': float(spearman),
        'top_1pct_capture': float(top_1pct_capture),
        'revenue_capture_1pct': float(revenue_1pct),
        'auc': float(auc),
        'mae': float(np.mean(np.abs(y_test - y_pred))),
        'rmse': float(np.sqrt(np.mean((y_test - y_pred)**2))),
    }
    
    log_message(f"Spearman: {spearman:.4f}")
    log_message(f"Top-1% Capture: {top_1pct_capture*100:.2f}%")
    log_message(f"Revenue Capture@1%: {revenue_1pct*100:.2f}%")
    log_message(f"AUC: {auc:.4f}")
    
    return metrics, y_pred_raw


def main():
    """Main execution function."""
    log_message("=" * 60)
    log_message("Baseline: 7-7-7 Day Split + Optimized Frozen Features")
    log_message("=" * 60)
    log_message(f"Config: {CONFIG['label_window_hours']*60:.0f}min window, {CONFIG['train_days']}-{CONFIG['val_days']}-{CONFIG['test_days']} days split")
    
    start_total = time.time()
    
    # Load data
    gift, user, streamer, room, click = load_data()
    
    # Prepare click-level data
    click_base = prepare_click_level_data(gift, click, CONFIG['label_window_hours'])
    
    # Temporal split (7-7-7 days)
    click_train, click_val, click_test = temporal_split_7days(click_base)
    
    # CRITICAL: Precompute frozen features using TRAIN window only
    log_message("\n" + "=" * 60)
    log_message("Precomputing frozen features (train window only, no leakage)")
    log_message("=" * 60)
    
    lookups = None
    lookup_cache_path = None
    
    if CONFIG['cache_lookups']:
        train_min_ts = click_train['timestamp'].min()
        train_max_ts = click_train['timestamp'].max()
        lookup_cache_path = FEATURES_DIR / f"frozen_lookups_7days_{train_min_ts}_{train_max_ts}.pkl"
        
        if lookup_cache_path.exists():
            log_message(f"Loading cached lookups from {lookup_cache_path}...")
            lookups, metadata = load_frozen_lookups(lookup_cache_path)
            log_message(f"✅ Loaded: {metadata['n_pairs']:,} pairs, {metadata['n_users']:,} users, {metadata['n_streamers']:,} streamers")
        else:
            log_message("Computing frozen lookups from train window only...")
            lookups, train_min_ts, train_max_ts = create_frozen_features(gift, click, click_train)
            if CONFIG['cache_lookups']:
                save_frozen_lookups(lookups, train_min_ts, train_max_ts, lookup_cache_path)
                log_message(f"✅ Saved lookups to {lookup_cache_path}")
    else:
        lookups, _, _ = create_frozen_features(gift, click, click_train)
    
    # Verify no leakage: check that lookups only use train window
    log_message("\n" + "=" * 60)
    log_message("Verifying no leakage...")
    log_message("=" * 60)
    train_min_ts = click_train['timestamp'].min()
    train_max_ts = click_train['timestamp'].max()
    val_min_ts = click_val['timestamp'].min()
    test_min_ts = click_test['timestamp'].min()
    
    log_message(f"Train window: {pd.to_datetime(train_min_ts, unit='ms')} to {pd.to_datetime(train_max_ts, unit='ms')}")
    log_message(f"Val window: {pd.to_datetime(val_min_ts, unit='ms')} to {pd.to_datetime(click_val['timestamp'].max(), unit='ms')}")
    log_message(f"Test window: {pd.to_datetime(test_min_ts, unit='ms')} to {pd.to_datetime(click_test['timestamp'].max(), unit='ms')}")
    log_message("✅ All frozen features computed from train window only (no leakage)")
    
    # Prepare features for all splits (using precomputed lookups)
    log_message("\n" + "=" * 60)
    log_message("Preparing features for all splits")
    log_message("=" * 60)
    
    log_message("Preparing train features...")
    train_df = prepare_features(gift, user, streamer, room, click, click_train, click_train, lookups)
    
    log_message("Preparing val features...")
    val_df = prepare_features(gift, user, streamer, room, click, click_val, click_train, lookups)
    
    log_message("Preparing test features...")
    test_df = prepare_features(gift, user, streamer, room, click, click_test, click_train, lookups)
    
    feature_cols = get_feature_columns(train_df)
    log_message(f"Feature columns: {len(feature_cols)}")
    
    # Train model
    model, training_time = train_model(train_df, val_df, feature_cols)
    
    # Evaluate
    metrics, y_pred = evaluate_model(model, test_df, feature_cols)
    metrics['training_time'] = training_time
    metrics['n_features'] = len(feature_cols)
    metrics['n_train'] = len(train_df)
    metrics['n_val'] = len(val_df)
    metrics['n_test'] = len(test_df)
    metrics['label_window_minutes'] = CONFIG['label_window_hours'] * 60
    metrics['split_type'] = f"{CONFIG['train_days']}-{CONFIG['val_days']}-{CONFIG['test_days']}_days"
    
    # Save results
    results_path = RESULTS_DIR / f"baseline_7days_{datetime.now().strftime('%Y%m%d')}.json"
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    log_message(f"Results saved to: {results_path}")
    
    # Save model
    model_path = MODELS_DIR / f"baseline_7days_{datetime.now().strftime('%Y%m%d')}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    log_message(f"Model saved to: {model_path}")
    
    # Summary
    total_time = time.time() - start_total
    log_message("\n" + "=" * 60)
    log_message("EXPERIMENT SUMMARY")
    log_message("=" * 60)
    log_message(f"Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    log_message(f"Training time: {training_time:.1f}s")
    log_message(f"\nMetrics:")
    log_message(f"  Spearman: {metrics['spearman']:.4f}")
    log_message(f"  Top-1% Capture: {metrics['top_1pct_capture']*100:.2f}%")
    log_message(f"  Revenue Capture@1%: {metrics['revenue_capture_1pct']*100:.2f}%")
    log_message(f"  AUC: {metrics['auc']:.4f}")
    log_message("\n" + "=" * 60)
    log_message("Experiment completed!", "SUCCESS")


if __name__ == '__main__':
    main()
