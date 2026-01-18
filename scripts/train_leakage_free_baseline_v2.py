#!/usr/bin/env python3
"""
Leakage-Free Baseline V2: With Watch Time Truncation
=====================================================

Experiment: EXP-20260118-gift_EVpred-01 (MVP-1.0 V2)

Key Fix: Label window now uses min(watch_time, H) instead of fixed H
- Original: [t_click, t_click + H]
- Fixed: [t_click, min(t_click + watch_time, t_click + H)]

This prevents attributing gifts to clicks where user has already left.

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
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_EVpred"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

for d in [IMG_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def load_data():
    log_message("Loading data files...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    return gift, user, streamer, room, click


def prepare_click_level_data_v2(gift, click, label_window_hours=1):
    """
    V2: Prepare click-level dataset with watch_time truncation.

    Label window: [t_click, min(t_click + watch_time, t_click + H)]

    Args:
        gift: gift dataframe
        click: click dataframe
        label_window_hours: max hours after click to aggregate gifts

    Returns:
        click_base: click dataframe with gift_price label (0 or positive)
    """
    log_message(f"Preparing click-level data V2 (label window: min(watch_time, {label_window_hours}h))...")

    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['day_of_week'] = click_base['timestamp_dt'].dt.dayofweek
    click_base['is_weekend'] = (click_base['day_of_week'] >= 5).astype(int)
    click_base['date'] = click_base['timestamp_dt'].dt.date

    # V2 FIX: Use min(watch_time, H) for label window truncation
    max_window_ms = label_window_hours * 3600 * 1000  # Convert hours to ms
    click_base['effective_window_ms'] = np.minimum(click_base['watch_live_time'], max_window_ms)
    click_base['click_end_ts'] = click_base['timestamp_dt'] + pd.to_timedelta(click_base['effective_window_ms'], unit='ms')

    # For each click, find gifts within truncated label window
    gift = gift.copy()
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')

    # Merge click and gift on (user_id, streamer_id, live_id)
    merged = click_base[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'click_end_ts']].merge(
        gift[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'gift_price']],
        on=['user_id', 'streamer_id', 'live_id'],
        how='left',
        suffixes=('_click', '_gift')
    )

    # Filter: gift timestamp within truncated window
    merged = merged[
        (merged['timestamp_dt_gift'] >= merged['timestamp_dt_click']) &
        (merged['timestamp_dt_gift'] <= merged['click_end_ts'])
    ]

    # Aggregate gift amounts per click
    gift_agg = merged.groupby(['user_id', 'streamer_id', 'live_id', 'timestamp_dt_click']).agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={
        'timestamp_dt_click': 'timestamp_dt',
        'gift_price': 'gift_price_label'
    })

    # Merge back to click_base
    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id', 'timestamp_dt'], how='left')
    click_base['gift_price_label'] = click_base['gift_price_label'].fillna(0)

    log_message(f"Click base: {len(click_base):,} records")
    log_message(f"Gift rate: {click_base['gift_price_label'].gt(0).mean()*100:.3f}%")
    log_message(f"Total gift: {click_base['gift_price_label'].sum():,.0f}")

    return click_base


def create_past_only_features_frozen(gift, click, train_df):
    """Create past-only features using frozen method: train window statistics only."""
    log_message("Creating frozen past-only features (train window only)...")

    train_min_ts = train_df['timestamp'].min()
    train_max_ts = train_df['timestamp'].max()

    gift_train = gift[
        (gift['timestamp'] >= train_min_ts) &
        (gift['timestamp'] <= train_max_ts)
    ].copy()

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

    log_message(f"Created frozen lookups: {len(lookups['pair']):,} pairs, {len(lookups['user']):,} users, {len(lookups['streamer']):,} streamers")

    return lookups


def apply_frozen_features(df, lookups, timestamp_col='timestamp'):
    """Apply frozen features to dataframe using lookup tables."""
    df = df.copy()

    df['pair_gift_count_past'] = 0
    df['pair_gift_sum_past'] = 0.0
    df['pair_gift_mean_past'] = 0.0
    df['pair_last_gift_time_gap_past'] = np.nan

    for idx, row in df.iterrows():
        key = (row['user_id'], row['streamer_id'])
        if key in lookups['pair']:
            stats = lookups['pair'][key]
            df.at[idx, 'pair_gift_count_past'] = stats['pair_gift_count']
            df.at[idx, 'pair_gift_sum_past'] = stats['pair_gift_sum']
            df.at[idx, 'pair_gift_mean_past'] = stats['pair_gift_mean']
            if pd.notna(stats['pair_last_gift_ts']):
                gap = (row[timestamp_col] - stats['pair_last_gift_ts']) / (1000 * 3600)
                df.at[idx, 'pair_last_gift_time_gap_past'] = gap

    df['user_total_gift_7d_past'] = df['user_id'].map(lookups['user']).fillna(0)
    df['user_budget_proxy_past'] = df['user_total_gift_7d_past']

    df['streamer_recent_revenue_past'] = 0.0
    df['streamer_recent_unique_givers_past'] = 0

    for idx, row in df.iterrows():
        if row['streamer_id'] in lookups['streamer']:
            stats = lookups['streamer'][row['streamer_id']]
            df.at[idx, 'streamer_recent_revenue_past'] = stats['streamer_recent_revenue']
            df.at[idx, 'streamer_recent_unique_givers_past'] = stats['streamer_recent_unique_givers']

    df['pair_last_gift_time_gap_past'] = df['pair_last_gift_time_gap_past'].fillna(999)

    return df


def create_static_features(user, streamer, room):
    """Create static features (no leakage risk)."""
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


def prepare_features_frozen(gift, user, streamer, room, click, click_base, train_df):
    """Prepare features with frozen method."""
    log_message("Preparing features (frozen version)...")

    user_features, streamer_features, room_info = create_static_features(user, streamer, room)

    df = click_base.merge(user_features, on='user_id', how='left')
    df = df.merge(streamer_features, on='streamer_id', how='left')
    df = df.merge(room_info, on='live_id', how='left')

    lookups = create_past_only_features_frozen(gift, click, train_df)
    df = apply_frozen_features(df, lookups)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

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

    df['target'] = np.log1p(df['gift_price_label'])
    df['target_raw'] = df['gift_price_label']
    df['is_gift'] = (df['gift_price_label'] > 0).astype(int)

    log_message(f"Final dataset: {len(df):,} records, {df.shape[1]} columns")
    log_message(f"Gift rate: {df['is_gift'].mean()*100:.3f}%")

    return df


def get_feature_columns(df):
    """Get feature columns."""
    exclude_cols = ['user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
                    'date', 'gift_price_label', 'target', 'target_raw', 'is_gift',
                    'watch_live_time', 'click_end_ts', 'effective_window_ms']
    return [c for c in df.columns if c not in exclude_cols]


def train_direct_model(train, val, feature_cols):
    """Train Direct Regression model."""
    log_message("Training Direct Regression model...")

    X_train = train[feature_cols]
    y_train = train['target']
    X_val = val[feature_cols]
    y_val = val['target']

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(period=50)
    ]

    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    training_time = time.time() - start_time

    log_message(f"Direct model completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


def compute_revenue_capture_at_k(y_true, y_pred, k_pct=0.01):
    """Compute Revenue Capture@K."""
    n = len(y_true)
    k = max(1, int(n * k_pct))

    pred_order = np.argsort(-y_pred)
    top_k_indices = pred_order[:k]

    revenue_top_k = y_true[top_k_indices].sum()
    total_revenue = y_true.sum()

    if total_revenue == 0:
        return 0.0

    return revenue_top_k / total_revenue


def compute_top_k_capture(y_true, y_pred, k_pct=0.01):
    """Compute Top-K% capture rate (set overlap)."""
    n = len(y_true)
    k = max(1, int(n * k_pct))

    true_rank = np.argsort(np.argsort(-y_true))
    pred_rank = np.argsort(np.argsort(-y_pred))

    true_topk = set(np.where(true_rank < k)[0])
    pred_topk = set(np.where(pred_rank < k)[0])

    return len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0


def evaluate_model(model, test, feature_cols):
    """Evaluate model and return metrics."""
    log_message("Evaluating model...")

    X_test = test[feature_cols]
    y_test_log = test['target'].values
    y_test_raw = test['target_raw'].values

    y_pred_log = model.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)

    mae_log = np.mean(np.abs(y_test_log - y_pred_log))
    spearman, _ = stats.spearmanr(y_test_raw, y_pred_raw)

    top_1pct_capture = compute_top_k_capture(y_test_raw, y_pred_raw, 0.01)
    top_5pct_capture = compute_top_k_capture(y_test_raw, y_pred_raw, 0.05)

    revenue_capture_1pct = compute_revenue_capture_at_k(y_test_raw, y_pred_raw, 0.01)
    revenue_capture_5pct = compute_revenue_capture_at_k(y_test_raw, y_pred_raw, 0.05)

    metrics = {
        'mae_log': float(mae_log),
        'spearman': float(spearman),
        'top_1pct_capture': float(top_1pct_capture),
        'top_5pct_capture': float(top_5pct_capture),
        'revenue_capture_1pct': float(revenue_capture_1pct),
        'revenue_capture_5pct': float(revenue_capture_5pct),
    }

    log_message(f"MAE(log): {mae_log:.4f}")
    log_message(f"Spearman: {spearman:.4f}")
    log_message(f"Top-1% Capture: {top_1pct_capture*100:.1f}%")
    log_message(f"Revenue Capture@1%: {revenue_capture_1pct*100:.1f}%")

    return metrics, y_pred_raw


def main():
    log_message("="*60)
    log_message("Leakage-Free Baseline V2 (Watch Time Truncation)")
    log_message("="*60)

    start_time = time.time()

    # Load data
    gift, user, streamer, room, click = load_data()

    # Prepare click-level data with V2 (watch time truncation)
    click_base = prepare_click_level_data_v2(gift, click, label_window_hours=1)

    # Temporal split
    click_base_sorted = click_base.sort_values('timestamp').reset_index(drop=True)
    train_ratio, val_ratio = 0.70, 0.15
    n = len(click_base_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    click_train = click_base_sorted.iloc[:train_end].copy()
    click_val = click_base_sorted.iloc[train_end:val_end].copy()
    click_test = click_base_sorted.iloc[val_end:].copy()

    log_message(f"Temporal split: Train={len(click_train):,}, Val={len(click_val):,}, Test={len(click_test):,}")

    # Prepare features (frozen only - no rolling)
    train_df = prepare_features_frozen(gift, user, streamer, room, click, click_train, click_train)
    val_df = prepare_features_frozen(gift, user, streamer, room, click, click_val, click_train)
    test_df = prepare_features_frozen(gift, user, streamer, room, click, click_test, click_train)

    feature_cols = get_feature_columns(train_df)
    log_message(f"Feature columns: {len(feature_cols)}")

    # Train Direct model
    direct_model, direct_time = train_direct_model(train_df, val_df, feature_cols)
    direct_metrics, direct_pred = evaluate_model(direct_model, test_df, feature_cols)

    # Save results
    results = {
        'version': 'V2_watch_time_truncation',
        'direct_frozen': direct_metrics,
        'data_stats': {
            'total_samples': len(click_base),
            'gift_rate': float((click_base['gift_price_label'] > 0).mean() * 100),
            'total_gift': float(click_base['gift_price_label'].sum()),
        }
    }

    results_path = RESULTS_DIR / 'leakage_free_v2_eval_20260118.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    pickle.dump(direct_model, open(MODELS_DIR / 'direct_frozen_v2_20260118.pkl', 'wb'))

    elapsed = time.time() - start_time

    log_message("\n" + "="*60)
    log_message("RESULTS SUMMARY (V2 - Watch Time Truncation)")
    log_message("="*60)
    log_message(f"Top-1% Capture: {direct_metrics['top_1pct_capture']*100:.1f}%")
    log_message(f"Revenue Capture@1%: {direct_metrics['revenue_capture_1pct']*100:.1f}%")
    log_message(f"Spearman: {direct_metrics['spearman']:.4f}")
    log_message(f"\nTotal time: {elapsed:.1f}s")
    log_message(f"Results saved to: {results_path}")
    log_message("="*60, "SUCCESS")


if __name__ == '__main__':
    main()
