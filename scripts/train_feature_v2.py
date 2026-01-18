#!/usr/bin/env python3
"""
Feature Engineering V2: Exploring New Signal Sources
=====================================================

Experiment: EXP-20260118-gift_EVpred-04 (MVP-1.2)

Goal: Explore new feature signals (sequence, realtime, content matching, cold-start)
to improve prediction performance on the leakage-free baseline.

Author: Viska Wei
Date: 2026-01-18
"""

import os
import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
import matplotlib.pyplot as plt

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

log_file = LOGS_DIR / f"feature_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_line = f"[{timestamp}] [{level}] {msg}"
    print(log_line)
    with open(log_file, 'a') as f:
        f.write(log_line + '\n')


def load_data():
    """Load all data files."""
    log_message("Loading data files...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    log_message(f"Loaded: gift={len(gift):,}, click={len(click):,}, user={len(user):,}")
    return gift, user, streamer, room, click


def prepare_click_level_data(gift, click, label_window_hours=1):
    """Prepare click-level dataset with labels (vectorized)."""
    log_message(f"Preparing click-level data (label window: {label_window_hours}h)...")

    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['day_of_week'] = click_base['timestamp_dt'].dt.dayofweek
    click_base['is_weekend'] = (click_base['day_of_week'] >= 5).astype(int)

    gift = gift.copy()
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')

    # For efficiency, aggregate gifts per (user, streamer, live_id) first
    gift_agg = gift.groupby(['user_id', 'streamer_id', 'live_id']).agg({
        'gift_price': 'sum',
        'timestamp': 'min'  # first gift time
    }).reset_index().rename(columns={'gift_price': 'total_gift', 'timestamp': 'first_gift_ts'})

    # Merge with click
    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id'], how='left')

    # Check if gift is within window (simplified: just use total if any gift in same live)
    click_base['gift_price_label'] = click_base['total_gift'].fillna(0)
    click_base = click_base.drop(columns=['total_gift', 'first_gift_ts'], errors='ignore')

    log_message(f"Click base: {len(click_base):,} records")
    log_message(f"Gift rate: {click_base['gift_price_label'].gt(0).mean()*100:.2f}%")

    return click_base


def create_baseline_features(gift, train_df):
    """Create baseline features using vectorized operations."""
    log_message("Creating baseline features (vectorized)...")

    train_min_ts = train_df['timestamp'].min()
    train_max_ts = train_df['timestamp'].max()

    gift_train = gift[
        (gift['timestamp'] >= train_min_ts) &
        (gift['timestamp'] <= train_max_ts)
    ].copy()

    # Pair features
    pair_stats = gift_train.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'timestamp': 'max'
    }).reset_index()
    pair_stats.columns = ['user_id', 'streamer_id', 'pair_gift_count', 'pair_gift_sum',
                          'pair_gift_mean', 'pair_gift_std', 'pair_gift_max', 'pair_last_gift_ts']
    pair_stats['pair_gift_std'] = pair_stats['pair_gift_std'].fillna(0)

    # User features
    user_stats = gift_train.groupby('user_id').agg({
        'gift_price': ['sum', 'count', 'mean', 'std', 'max'],
        'streamer_id': 'nunique',
        'timestamp': 'max'
    }).reset_index()
    user_stats.columns = ['user_id', 'user_total_gift', 'user_gift_count', 'user_gift_mean',
                          'user_gift_std', 'user_gift_max', 'user_unique_streamers', 'user_last_gift_ts']
    user_stats['user_gift_std'] = user_stats['user_gift_std'].fillna(0)

    # Streamer features
    streamer_stats = gift_train.groupby('streamer_id').agg({
        'gift_price': ['sum', 'count', 'mean', 'std', 'max'],
        'user_id': 'nunique',
        'timestamp': 'max'
    }).reset_index()
    streamer_stats.columns = ['streamer_id', 'streamer_total_revenue', 'streamer_gift_count',
                              'streamer_gift_mean', 'streamer_gift_std', 'streamer_gift_max',
                              'streamer_unique_givers', 'streamer_last_gift_ts']
    streamer_stats['streamer_gift_std'] = streamer_stats['streamer_gift_std'].fillna(0)

    log_message(f"Baseline features: {len(pair_stats):,} pairs, {len(user_stats):,} users, {len(streamer_stats):,} streamers")
    return {'pair': pair_stats, 'user': user_stats, 'streamer': streamer_stats}


def apply_baseline_features(df, baseline_stats):
    """Apply baseline features using merge (vectorized)."""
    df = df.copy()

    # Merge pair features
    df = df.merge(baseline_stats['pair'], on=['user_id', 'streamer_id'], how='left', suffixes=('', '_pair'))

    # Compute time gap
    df['pair_last_gift_gap'] = (df['timestamp'] - df['pair_last_gift_ts']) / (1000 * 3600)
    df['pair_last_gift_gap'] = df['pair_last_gift_gap'].clip(lower=0).fillna(999)
    df = df.drop(columns=['pair_last_gift_ts'], errors='ignore')

    # Fill NaN for pair features
    pair_cols = ['pair_gift_count', 'pair_gift_sum', 'pair_gift_mean', 'pair_gift_std', 'pair_gift_max']
    for col in pair_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Merge user features
    df = df.merge(baseline_stats['user'], on='user_id', how='left', suffixes=('', '_user'))
    df['user_last_gift_gap'] = (df['timestamp'] - df['user_last_gift_ts']) / (1000 * 3600)
    df['user_last_gift_gap'] = df['user_last_gift_gap'].clip(lower=0).fillna(999)
    df = df.drop(columns=['user_last_gift_ts'], errors='ignore')

    user_cols = ['user_total_gift', 'user_gift_count', 'user_gift_mean', 'user_gift_std',
                 'user_gift_max', 'user_unique_streamers']
    for col in user_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Merge streamer features
    df = df.merge(baseline_stats['streamer'], on='streamer_id', how='left', suffixes=('', '_streamer'))
    df['streamer_last_gift_gap'] = (df['timestamp'] - df['streamer_last_gift_ts']) / (1000 * 3600)
    df['streamer_last_gift_gap'] = df['streamer_last_gift_gap'].clip(lower=0).fillna(999)
    df = df.drop(columns=['streamer_last_gift_ts'], errors='ignore')

    streamer_cols = ['streamer_total_revenue', 'streamer_gift_count', 'streamer_gift_mean',
                     'streamer_gift_std', 'streamer_gift_max', 'streamer_unique_givers']
    for col in streamer_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def create_sequence_features(gift, train_df, n_recent=[3, 5]):
    """Create sequence features (vectorized)."""
    log_message("Creating sequence features...")

    train_min_ts = train_df['timestamp'].min()
    train_max_ts = train_df['timestamp'].max()

    gift_train = gift[
        (gift['timestamp'] >= train_min_ts) &
        (gift['timestamp'] <= train_max_ts)
    ].copy().sort_values('timestamp')

    # User sequence features
    user_seq_features = []
    for user_id, grp in gift_train.groupby('user_id'):
        amounts = grp['gift_price'].values
        timestamps = grp['timestamp'].values

        row = {'user_id': user_id}
        for n in n_recent:
            recent = amounts[-n:] if len(amounts) >= n else amounts
            if len(recent) > 0:
                row[f'user_seq_{n}_mean'] = np.mean(recent)
                row[f'user_seq_{n}_std'] = np.std(recent) if len(recent) > 1 else 0
                row[f'user_seq_{n}_max'] = np.max(recent)
            else:
                row[f'user_seq_{n}_mean'] = 0
                row[f'user_seq_{n}_std'] = 0
                row[f'user_seq_{n}_max'] = 0

            # Intervals
            if len(timestamps) >= 2:
                recent_ts = timestamps[-n:] if len(timestamps) >= n else timestamps
                intervals = np.diff(recent_ts) / (1000 * 3600)
                row[f'user_seq_{n}_interval_mean'] = np.mean(intervals) if len(intervals) > 0 else 999
            else:
                row[f'user_seq_{n}_interval_mean'] = 999

        user_seq_features.append(row)

    user_seq_df = pd.DataFrame(user_seq_features) if user_seq_features else pd.DataFrame({'user_id': []})

    # Pair sequence features (simplified)
    pair_seq = gift_train.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': lambda x: np.mean(x.values[-3:]) if len(x) > 0 else 0
    }).reset_index().rename(columns={'gift_price': 'pair_seq_3_mean'})

    log_message(f"Sequence features: {len(user_seq_df):,} users")
    return {'user_seq': user_seq_df, 'pair_seq': pair_seq}


def apply_sequence_features(df, seq_stats):
    """Apply sequence features."""
    df = df.merge(seq_stats['user_seq'], on='user_id', how='left')
    df = df.merge(seq_stats['pair_seq'], on=['user_id', 'streamer_id'], how='left')

    # Fill NaN
    seq_cols = [c for c in df.columns if 'seq_' in c]
    for col in seq_cols:
        default = 999 if 'interval' in col else 0
        df[col] = df[col].fillna(default)

    return df


def create_realtime_features(click, train_df):
    """Create realtime context features."""
    log_message("Creating realtime features...")

    train_min_ts = train_df['timestamp'].min()
    train_max_ts = train_df['timestamp'].max()

    click_train = click[
        (click['timestamp'] >= train_min_ts) &
        (click['timestamp'] <= train_max_ts)
    ].copy()

    user_avg_watch = click_train.groupby('user_id')['watch_live_time'].mean().reset_index()
    user_avg_watch.columns = ['user_id', 'user_avg_watch_time']

    global_avg = click_train['watch_live_time'].mean()

    return {'user_avg_watch': user_avg_watch, 'global_avg': global_avg}


def apply_realtime_features(df, rt_stats):
    """Apply realtime features."""
    df = df.merge(rt_stats['user_avg_watch'], on='user_id', how='left')
    df['user_avg_watch_time'] = df['user_avg_watch_time'].fillna(rt_stats['global_avg'])

    df['watch_time_log'] = np.log1p(df['watch_live_time'].fillna(0))
    df['watch_time_ratio'] = df['watch_live_time'].fillna(0) / (df['user_avg_watch_time'] + 1)

    df['is_peak_hour'] = df['hour'].isin([18, 19, 20, 21, 22]).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    return df


def create_content_features(gift, room, train_df):
    """Create content matching features."""
    log_message("Creating content features...")

    train_min_ts = train_df['timestamp'].min()
    train_max_ts = train_df['timestamp'].max()

    gift_train = gift[
        (gift['timestamp'] >= train_min_ts) &
        (gift['timestamp'] <= train_max_ts)
    ].copy()

    # Get category for each gift
    room_cat = room[['live_id', 'live_content_category']].drop_duplicates('live_id')
    gift_with_cat = gift_train.merge(room_cat, on='live_id', how='left')
    gift_with_cat['live_content_category'] = gift_with_cat['live_content_category'].fillna('unknown')

    # User preference by category
    user_cat_pref = gift_with_cat.groupby(['user_id', 'live_content_category']).agg({
        'gift_price': 'sum'
    }).reset_index()

    # Get top category per user
    user_top_cat = user_cat_pref.loc[user_cat_pref.groupby('user_id')['gift_price'].idxmax()][['user_id', 'live_content_category']]
    user_top_cat.columns = ['user_id', 'user_top_category']

    # Streamer primary category
    room_streamer_cat = room.groupby('streamer_id')['live_content_category'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown'
    ).reset_index()
    room_streamer_cat.columns = ['streamer_id', 'streamer_category']

    return {'user_top_cat': user_top_cat, 'streamer_cat': room_streamer_cat, 'room_cat': room_cat}


def apply_content_features(df, content_stats):
    """Apply content features."""
    df = df.merge(content_stats['user_top_cat'], on='user_id', how='left')
    df = df.merge(content_stats['streamer_cat'], on='streamer_id', how='left')
    df = df.merge(content_stats['room_cat'], on='live_id', how='left')

    df['user_top_category'] = df['user_top_category'].fillna('unknown')
    df['streamer_category'] = df['streamer_category'].fillna('unknown')
    df['live_content_category'] = df['live_content_category'].fillna('unknown')

    # Category match
    df['category_match'] = (df['user_top_category'] == df['live_content_category']).astype(int)

    # Encode categories
    for col in ['user_top_category', 'streamer_category', 'live_content_category']:
        df[col] = df[col].astype('category').cat.codes

    return df


def create_coldstart_features(gift, train_df):
    """Create cold-start generalization features."""
    log_message("Creating coldstart features...")

    train_min_ts = train_df['timestamp'].min()
    train_max_ts = train_df['timestamp'].max()

    gift_train = gift[
        (gift['timestamp'] >= train_min_ts) &
        (gift['timestamp'] <= train_max_ts)
    ].copy()

    # User tiers
    user_spend = gift_train.groupby('user_id').agg({
        'gift_price': 'sum',
        'streamer_id': 'nunique'
    }).reset_index()
    user_spend.columns = ['user_id', 'user_total_spend', 'user_streamer_count']
    user_spend['user_avg_per_streamer'] = user_spend['user_total_spend'] / (user_spend['user_streamer_count'] + 1)

    # Quantile-based tiers
    user_spend['user_tier'] = pd.qcut(
        user_spend['user_total_spend'].rank(method='first'),
        q=4, labels=[0, 1, 2, 3]
    ).astype(int)

    # Streamer tiers
    streamer_recv = gift_train.groupby('streamer_id').agg({
        'gift_price': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    streamer_recv.columns = ['streamer_id', 'streamer_total_recv', 'streamer_giver_count']
    streamer_recv['streamer_avg_per_user'] = streamer_recv['streamer_total_recv'] / (streamer_recv['streamer_giver_count'] + 1)

    streamer_recv['streamer_tier'] = pd.qcut(
        streamer_recv['streamer_total_recv'].rank(method='first'),
        q=4, labels=[0, 1, 2, 3]
    ).astype(int)

    return {
        'user_tier': user_spend[['user_id', 'user_tier', 'user_avg_per_streamer']],
        'streamer_tier': streamer_recv[['streamer_id', 'streamer_tier', 'streamer_avg_per_user']]
    }


def apply_coldstart_features(df, cs_stats):
    """Apply coldstart features."""
    df = df.merge(cs_stats['user_tier'], on='user_id', how='left')
    df = df.merge(cs_stats['streamer_tier'], on='streamer_id', how='left')

    df['user_tier'] = df['user_tier'].fillna(0).astype(int)
    df['streamer_tier'] = df['streamer_tier'].fillna(0).astype(int)
    df['user_avg_per_streamer'] = df['user_avg_per_streamer'].fillna(0)
    df['streamer_avg_per_user'] = df['streamer_avg_per_user'].fillna(0)

    df['tier_interaction'] = df['user_tier'] * 4 + df['streamer_tier']
    df['tier_diff'] = np.abs(df['user_tier'] - df['streamer_tier'])
    df['tier_match'] = (df['user_tier'] == df['streamer_tier']).astype(int)

    df['user_avg_per_streamer_log'] = np.log1p(df['user_avg_per_streamer'])
    df['streamer_avg_per_user_log'] = np.log1p(df['streamer_avg_per_user'])

    return df


def create_static_features(user, streamer, room):
    """Create static features."""
    log_message("Creating static features...")

    user_cols = ['user_id', 'age', 'gender', 'device_brand', 'device_price',
                 'fans_num', 'follow_num', 'accu_watch_live_cnt',
                 'accu_watch_live_duration', 'is_live_streamer', 'is_photo_author']
    user_features = user[[c for c in user_cols if c in user.columns]].copy()

    streamer_cols = ['streamer_id', 'gender', 'age', 'device_brand', 'device_price',
                     'live_operation_tag', 'fans_user_num', 'fans_group_fans_num',
                     'follow_user_num', 'accu_live_cnt', 'accu_live_duration']
    streamer_features = streamer[[c for c in streamer_cols if c in streamer.columns]].copy()

    streamer_features = streamer_features.rename(columns={
        'gender': 'streamer_gender', 'age': 'streamer_age',
        'device_brand': 'streamer_device_brand', 'device_price': 'streamer_device_price'
    })

    room_info = room[['live_id', 'live_type']].drop_duplicates('live_id')

    return user_features, streamer_features, room_info


def get_feature_columns(df):
    """Get feature columns."""
    exclude_cols = {'user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
                    'gift_price_label', 'target', 'target_raw', 'is_gift',
                    'watch_live_time', 'click_end_ts', 'date'}
    return [c for c in df.columns if c not in exclude_cols]


def train_model(train, val, feature_cols):
    """Train LightGBM model."""
    log_message(f"Training model with {len(feature_cols)} features...")

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
        lgb.log_evaluation(period=100)
    ]

    start = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    elapsed = time.time() - start

    log_message(f"Training done in {elapsed:.1f}s, best_iter={model.best_iteration}", "SUCCESS")
    return model, elapsed


def compute_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    n = len(y_true)

    # Top-K% capture
    def top_k_capture(k_pct):
        k = max(1, int(n * k_pct))
        true_rank = np.argsort(np.argsort(-y_true))
        pred_rank = np.argsort(np.argsort(-y_pred))
        true_topk = set(np.where(true_rank < k)[0])
        pred_topk = set(np.where(pred_rank < k)[0])
        return len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0

    # Revenue capture
    def revenue_capture(k_pct):
        k = max(1, int(n * k_pct))
        pred_order = np.argsort(-y_pred)
        top_k_idx = pred_order[:k]
        return y_true[top_k_idx].sum() / y_true.sum() if y_true.sum() > 0 else 0

    spearman, _ = stats.spearmanr(y_true, y_pred)

    return {
        'top_1pct_capture': top_k_capture(0.01),
        'top_5pct_capture': top_k_capture(0.05),
        'revenue_capture_1pct': revenue_capture(0.01),
        'revenue_capture_5pct': revenue_capture(0.05),
        'spearman': spearman if not np.isnan(spearman) else 0
    }


def evaluate_model(model, test, feature_cols):
    """Evaluate model."""
    log_message("Evaluating model...")

    X_test = test[feature_cols]
    y_test_raw = test['target_raw'].values

    y_pred_log = model.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)

    metrics = compute_metrics(y_test_raw, y_pred_raw)

    log_message(f"Top-1%: {metrics['top_1pct_capture']*100:.1f}%, RevCap@1%: {metrics['revenue_capture_1pct']*100:.1f}%")

    return metrics, y_pred_raw


def plot_ablation(results, save_path):
    """Plot ablation study."""
    fig, ax = plt.subplots(figsize=(6, 5))

    names = list(results.keys())
    values = [results[n]['top_1pct_capture'] * 100 for n in names]

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(range(len(names)), values, color=colors)

    ax.set_xlabel('Feature Group', fontsize=12)
    ax.set_ylabel('Top-1% Capture (%)', fontsize=12)
    ax.set_title('Feature Ablation: Top-1% Capture', fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_revenue(results, save_path):
    """Plot revenue capture."""
    fig, ax = plt.subplots(figsize=(6, 5))

    names = list(results.keys())
    values = [results[n]['revenue_capture_1pct'] * 100 for n in names]

    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(range(len(names)), values, color=colors)

    ax.set_xlabel('Feature Group', fontsize=12)
    ax.set_ylabel('Revenue Capture@1% (%)', fontsize=12)
    ax.set_title('Feature Ablation: Revenue Capture@1%', fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_importance(model, feature_cols, save_path, top_n=25):
    """Plot feature importance."""
    fig, ax = plt.subplots(figsize=(6, 5))

    importance = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False).head(top_n)

    # Color by type
    colors = []
    for f in imp_df['feature']:
        if 'seq_' in f:
            colors.append('coral')
        elif 'watch_time' in f or 'hour' in f or 'peak' in f:
            colors.append('green')
        elif 'category' in f or 'match' in f:
            colors.append('purple')
        elif 'tier' in f or 'avg_per' in f:
            colors.append('orange')
        else:
            colors.append('steelblue')

    ax.barh(range(len(imp_df)), imp_df['importance'].values, color=colors)
    ax.set_yticks(range(len(imp_df)))
    ax.set_yticklabels(imp_df['feature'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title(f'Feature Importance (Top {top_n})', fontsize=14)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', label='Baseline'),
        Patch(facecolor='coral', label='Sequence'),
        Patch(facecolor='green', label='Realtime'),
        Patch(facecolor='purple', label='Content'),
        Patch(facecolor='orange', label='Coldstart'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_coldstart(results, baseline_stats, test, y_pred_raw, save_path):
    """Plot cold-start slice analysis."""
    fig, ax = plt.subplots(figsize=(6, 5))

    y_test_raw = test['target_raw'].values

    # Create pair keys set
    pair_keys = set(zip(baseline_stats['pair']['user_id'], baseline_stats['pair']['streamer_id']))
    test_pairs = list(zip(test['user_id'], test['streamer_id']))
    cold_mask = np.array([p not in pair_keys for p in test_pairs])
    warm_mask = ~cold_mask

    slices = {}
    slices['All'] = compute_metrics(y_test_raw, y_pred_raw)

    if cold_mask.sum() > 100:
        slices['Cold-pair'] = compute_metrics(y_test_raw[cold_mask], y_pred_raw[cold_mask])
    if warm_mask.sum() > 100:
        slices['Warm-pair'] = compute_metrics(y_test_raw[warm_mask], y_pred_raw[warm_mask])

    slice_names = list(slices.keys())
    top_1 = [slices[s]['top_1pct_capture'] * 100 for s in slice_names]
    rev_1 = [slices[s]['revenue_capture_1pct'] * 100 for s in slice_names]

    x = np.arange(len(slice_names))
    width = 0.35

    ax.bar(x - width/2, top_1, width, label='Top-1% Capture', color='steelblue')
    ax.bar(x + width/2, rev_1, width, label='Revenue Capture@1%', color='coral')

    ax.set_xlabel('Slice', fontsize=12)
    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title('Cold-start Slice Analysis', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(slice_names)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_learning_curve(results, save_path):
    """Plot learning curve."""
    fig, ax = plt.subplots(figsize=(6, 5))

    names = list(results.keys())
    top_1 = [results[n]['top_1pct_capture'] * 100 for n in names]
    rev_1 = [results[n]['revenue_capture_1pct'] * 100 for n in names]

    ax.plot(range(len(names)), top_1, 'o-', label='Top-1% Capture', linewidth=2, markersize=8)
    ax.plot(range(len(names)), rev_1, 's-', label='Revenue Capture@1%', linewidth=2, markersize=8)

    ax.set_xlabel('Feature Group', fontsize=12)
    ax.set_ylabel('Performance (%)', fontsize=12)
    ax.set_title('Learning Curve', fontsize=14)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def main():
    """Main experiment."""
    log_message("=" * 60)
    log_message("Feature Engineering V2 Experiment")
    log_message("EXP-20260118-gift_EVpred-04")
    log_message("=" * 60)

    # Load data
    gift, user, streamer, room, click = load_data()

    # Prepare click-level data
    click_base = prepare_click_level_data(gift, click)

    # Temporal split
    click_base = click_base.sort_values('timestamp').reset_index(drop=True)
    n = len(click_base)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_click_full = click_base.iloc[:train_end].copy()
    val_click_full = click_base.iloc[train_end:val_end].copy()
    test_click_full = click_base.iloc[val_end:].copy()

    # Sample for efficiency: keep all gift records + sample non-gift records
    MAX_TRAIN = 500000
    MAX_VAL = 100000
    MAX_TEST = 100000

    def sample_with_all_gifts(df, max_size):
        """Sample data keeping all gift records."""
        gift_records = df[df['gift_price_label'] > 0]
        non_gift_records = df[df['gift_price_label'] == 0]
        n_non_gift = max(0, max_size - len(gift_records))
        if len(non_gift_records) > n_non_gift:
            non_gift_records = non_gift_records.sample(n=n_non_gift, random_state=SEED)
        return pd.concat([gift_records, non_gift_records]).sort_values('timestamp').reset_index(drop=True)

    train_click = sample_with_all_gifts(train_click_full, MAX_TRAIN)
    val_click = sample_with_all_gifts(val_click_full, MAX_VAL)
    test_click = sample_with_all_gifts(test_click_full, MAX_TEST)

    log_message(f"Full: Train={len(train_click_full):,}, Val={len(val_click_full):,}, Test={len(test_click_full):,}")
    log_message(f"Sampled: Train={len(train_click):,}, Val={len(val_click):,}, Test={len(test_click):,}")
    log_message(f"Gift rates: Train={train_click['gift_price_label'].gt(0).mean()*100:.2f}%, Val={val_click['gift_price_label'].gt(0).mean()*100:.2f}%, Test={test_click['gift_price_label'].gt(0).mean()*100:.2f}%")

    # Create all feature lookups
    log_message("\n--- Creating Feature Lookups ---")
    baseline_stats = create_baseline_features(gift, train_click)
    seq_stats = create_sequence_features(gift, train_click)
    rt_stats = create_realtime_features(click, train_click)
    content_stats = create_content_features(gift, room, train_click)
    cs_stats = create_coldstart_features(gift, train_click)

    # Static features
    user_features, streamer_features, room_info = create_static_features(user, streamer, room)

    # Prepare all data with all features ONCE
    log_message("\n--- Preparing Full Feature Sets ---")

    def prepare_full_features(df):
        """Prepare dataframe with all possible features."""
        result = df.copy()
        result = result.merge(user_features, on='user_id', how='left')
        result = result.merge(streamer_features, on='streamer_id', how='left')
        result = result.merge(room_info, on='live_id', how='left')
        result = apply_baseline_features(result, baseline_stats)
        result = apply_sequence_features(result, seq_stats)
        result = apply_realtime_features(result, rt_stats)
        result = apply_content_features(result, content_stats)
        result = apply_coldstart_features(result, cs_stats)

        # Targets
        result['target'] = np.log1p(result['gift_price_label'])
        result['target_raw'] = result['gift_price_label']
        result['is_gift'] = (result['gift_price_label'] > 0).astype(int)

        # Fill NaN
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        result[numeric_cols] = result[numeric_cols].fillna(0)

        # Encode all object columns
        obj_cols = result.select_dtypes(include=['object']).columns.tolist()
        for col in obj_cols:
            result[col] = result[col].fillna('unknown').astype('category').cat.codes

        return result

    train_full = prepare_full_features(train_click)
    log_message(f"Train prepared: {len(train_full):,} rows, {len(train_full.columns)} cols")
    val_full = prepare_full_features(val_click)
    log_message(f"Val prepared: {len(val_full):,} rows")
    test_full = prepare_full_features(test_click)
    log_message(f"Test prepared: {len(test_full):,} rows")

    # Define feature groups
    baseline_cols = ['pair_gift_count', 'pair_gift_sum', 'pair_gift_mean', 'pair_gift_std', 'pair_gift_max',
                     'pair_last_gift_gap', 'user_total_gift', 'user_gift_count', 'user_gift_mean',
                     'user_gift_std', 'user_gift_max', 'user_unique_streamers', 'user_last_gift_gap',
                     'streamer_total_revenue', 'streamer_gift_count', 'streamer_gift_mean',
                     'streamer_gift_std', 'streamer_gift_max', 'streamer_unique_givers', 'streamer_last_gift_gap',
                     'hour', 'day_of_week', 'is_weekend']
    seq_cols = [c for c in train_full.columns if 'seq_' in c]
    rt_cols = ['watch_time_log', 'watch_time_ratio', 'user_avg_watch_time', 'is_peak_hour', 'hour_sin', 'hour_cos']
    content_cols = ['user_top_category', 'streamer_category', 'live_content_category', 'category_match']
    cold_cols = ['user_tier', 'streamer_tier', 'user_avg_per_streamer', 'streamer_avg_per_user',
                 'tier_interaction', 'tier_diff', 'tier_match', 'user_avg_per_streamer_log', 'streamer_avg_per_user_log']
    static_cols = [c for c in train_full.columns if c in user_features.columns or
                   c in streamer_features.columns or c == 'live_type']

    # Filter to columns that actually exist
    baseline_cols = [c for c in baseline_cols if c in train_full.columns]
    seq_cols = [c for c in seq_cols if c in train_full.columns]
    rt_cols = [c for c in rt_cols if c in train_full.columns]
    content_cols = [c for c in content_cols if c in train_full.columns]
    cold_cols = [c for c in cold_cols if c in train_full.columns]
    static_cols = [c for c in static_cols if c in train_full.columns and c not in ['user_id', 'streamer_id']]

    log_message(f"Feature groups: baseline={len(baseline_cols)}, seq={len(seq_cols)}, rt={len(rt_cols)}, content={len(content_cols)}, cold={len(cold_cols)}, static={len(static_cols)}")

    # Experiments: map names to feature column lists
    experiments = {
        'baseline': baseline_cols + static_cols,
        'baseline+seq': baseline_cols + seq_cols + static_cols,
        'baseline+rt': baseline_cols + rt_cols + static_cols,
        'baseline+content': baseline_cols + content_cols + static_cols,
        'baseline+cold': baseline_cols + cold_cols + static_cols,
        'all': baseline_cols + seq_cols + rt_cols + content_cols + cold_cols + static_cols
    }

    all_results = {}
    all_models = {}
    best_model = None
    best_test = None
    best_pred = None
    best_features = None

    for exp_name, feature_cols in experiments.items():
        log_message(f"\n{'='*60}")
        log_message(f"Experiment: {exp_name}")
        log_message(f"{'='*60}")

        # Deduplicate and filter feature columns
        feature_cols = list(dict.fromkeys(feature_cols))  # Remove duplicates, preserve order
        feature_cols = [c for c in feature_cols if c in train_full.columns and c in test_full.columns]
        log_message(f"Features: {len(feature_cols)}")

        # Train
        model, train_time = train_model(train_full, val_full, feature_cols)

        # Evaluate
        metrics, y_pred = evaluate_model(model, test_full, feature_cols)
        metrics['n_features'] = len(feature_cols)
        metrics['train_time'] = train_time

        all_results[exp_name] = metrics
        all_models[exp_name] = model

        # Track best for plotting
        if exp_name == 'all':
            best_model = model
            best_test = test_full
            best_pred = y_pred
            best_features = feature_cols

    # Plots
    log_message("\n--- Generating Plots ---")
    plot_ablation(all_results, IMG_DIR / 'feature_v2_ablation.png')
    plot_revenue(all_results, IMG_DIR / 'feature_v2_revenue.png')
    plot_importance(best_model, best_features, IMG_DIR / 'feature_v2_importance.png')
    plot_coldstart(all_results, baseline_stats, best_test, best_pred, IMG_DIR / 'feature_v2_coldstart.png')
    plot_learning_curve(all_results, IMG_DIR / 'feature_v2_learning_curve.png')

    # Save results
    with open(RESULTS_DIR / 'feature_v2_eval_20260118.json', 'w') as f:
        json.dump({'experiments': all_results}, f, indent=2)

    # Summary
    log_message("\n" + "="*60)
    log_message("EXPERIMENT SUMMARY")
    log_message("="*60)

    for name, m in all_results.items():
        log_message(f"{name}: Top-1%={m['top_1pct_capture']*100:.1f}%, RevCap@1%={m['revenue_capture_1pct']*100:.1f}%, Spearman={m['spearman']:.4f}")

    log_message("\n" + "="*60)
    log_message("Experiment completed!", "SUCCESS")
    log_message(f"Results: {RESULTS_DIR / 'feature_v2_eval_20260118.json'}")
    log_message(f"Figures: {IMG_DIR}")


if __name__ == '__main__':
    main()
