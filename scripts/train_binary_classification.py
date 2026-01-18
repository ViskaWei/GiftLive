#!/usr/bin/env python3
"""
Binary Classification: Task Degradation Validation
===================================================

Experiment: EXP-20260118-gift_EVpred-05 (MVP-1.3)

Goal: Validate whether binary classification (P(gift>0)) is more feasible than EV regression,
as an alternative or first stage for gift prediction.

Experiments:
- Exp A: LightGBM Default
- Exp B: LightGBM + scale_pos_weight
- Exp C: LightGBM + Undersampling
- Exp D: XGBoost + Focal Loss
- Exp E: Two-Stage Improved (best classifier + regression)

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
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    roc_curve, f1_score, precision_score, recall_score
)
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


def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "", "SUCCESS": "[OK]", "WARNING": "[WARN]", "ERROR": "[ERR]"}.get(level, "")
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


def prepare_click_level_data(gift, click, label_window_hours=1):
    """Prepare click-level dataset with gift labels."""
    log_message(f"Preparing click-level data (label window: {label_window_hours}h)...")

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
    """Create frozen past-only features (train window only)."""
    log_message("Creating frozen past-only features...")

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

    log_message(f"Created lookups: {len(lookups['pair']):,} pairs, {len(lookups['user']):,} users, {len(lookups['streamer']):,} streamers")

    return lookups


def apply_frozen_features(df, lookups, timestamp_col='timestamp'):
    """Apply frozen features to dataframe."""
    df = df.copy()

    # Initialize columns
    df['pair_gift_count_past'] = 0
    df['pair_gift_sum_past'] = 0.0
    df['pair_gift_mean_past'] = 0.0
    df['pair_last_gift_time_gap_past'] = 999.0

    # Apply pair features (vectorized where possible)
    pair_keys = list(zip(df['user_id'], df['streamer_id']))

    for idx, key in enumerate(pair_keys):
        if key in lookups['pair']:
            stats = lookups['pair'][key]
            df.iloc[idx, df.columns.get_loc('pair_gift_count_past')] = stats['pair_gift_count']
            df.iloc[idx, df.columns.get_loc('pair_gift_sum_past')] = stats['pair_gift_sum']
            df.iloc[idx, df.columns.get_loc('pair_gift_mean_past')] = stats['pair_gift_mean']
            if pd.notna(stats['pair_last_gift_ts']):
                gap = (df.iloc[idx][timestamp_col] - stats['pair_last_gift_ts']) / (1000 * 3600)
                df.iloc[idx, df.columns.get_loc('pair_last_gift_time_gap_past')] = gap

    # User features
    df['user_total_gift_7d_past'] = df['user_id'].map(lookups['user']).fillna(0)
    df['user_budget_proxy_past'] = df['user_total_gift_7d_past']

    # Streamer features
    df['streamer_recent_revenue_past'] = 0.0
    df['streamer_recent_unique_givers_past'] = 0

    for idx, row in df.iterrows():
        if row['streamer_id'] in lookups['streamer']:
            stats = lookups['streamer'][row['streamer_id']]
            df.at[idx, 'streamer_recent_revenue_past'] = stats['streamer_recent_revenue']
            df.at[idx, 'streamer_recent_unique_givers_past'] = stats['streamer_recent_unique_givers']

    return df


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


def prepare_features(gift, user, streamer, room, click, click_base, train_df):
    """Prepare features with frozen method."""
    log_message("Preparing features (frozen version)...")

    user_features, streamer_features, room_info = create_static_features(user, streamer, room)

    df = click_base.merge(user_features, on='user_id', how='left')
    df = df.merge(streamer_features, on='streamer_id', how='left')
    df = df.merge(room_info, on='live_id', how='left')

    lookups = create_frozen_features(gift, click, train_df)
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
    df['target'] = np.log1p(df['gift_price_label'])
    df['target_raw'] = df['gift_price_label']
    df['is_gift'] = (df['gift_price_label'] > 0).astype(int)

    log_message(f"Final dataset: {len(df):,} records, {df.shape[1]} columns")
    log_message(f"Gift rate: {df['is_gift'].mean()*100:.2f}%")

    return df, lookups


def get_feature_columns(df):
    """Get feature columns."""
    exclude_cols = ['user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
                    'date', 'gift_price_label', 'target', 'target_raw', 'is_gift',
                    'watch_live_time', 'click_end_ts']
    return [c for c in df.columns if c not in exclude_cols]


# ============== EXPERIMENT A: LightGBM Default ==============
def train_exp_a_default(train, val, feature_cols):
    """Exp A: LightGBM Default binary classification."""
    log_message("Exp A: LightGBM Default...")

    X_train = train[feature_cols]
    y_train = train['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']

    params = {
        'objective': 'binary',
        'metric': 'auc',
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

    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    training_time = time.time() - start_time

    log_message(f"Exp A completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


# ============== EXPERIMENT B: LightGBM + scale_pos_weight ==============
def train_exp_b_weighted(train, val, feature_cols):
    """Exp B: LightGBM with scale_pos_weight."""
    log_message("Exp B: LightGBM + scale_pos_weight...")

    X_train = train[feature_cols]
    y_train = train['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = min(n_neg / n_pos, 100.0) if n_pos > 0 else 1.0
    log_message(f"scale_pos_weight = {scale_pos_weight:.2f}")

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'min_data_in_leaf': 100,
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

    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    training_time = time.time() - start_time

    log_message(f"Exp B completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


# ============== EXPERIMENT C: LightGBM + Undersampling ==============
def train_exp_c_undersample(train, val, feature_cols, target_ratio=10):
    """Exp C: LightGBM with undersampling (1:target_ratio)."""
    log_message(f"Exp C: LightGBM + Undersampling (1:{target_ratio})...")

    # Undersample negative class
    pos_samples = train[train['is_gift'] == 1]
    neg_samples = train[train['is_gift'] == 0]

    n_pos = len(pos_samples)
    n_neg_sample = min(n_pos * target_ratio, len(neg_samples))

    neg_sampled = neg_samples.sample(n=n_neg_sample, random_state=SEED)
    train_sampled = pd.concat([pos_samples, neg_sampled]).sample(frac=1, random_state=SEED)

    log_message(f"Sampled: {len(pos_samples):,} pos + {n_neg_sample:,} neg = {len(train_sampled):,} total")

    X_train = train_sampled[feature_cols]
    y_train = train_sampled['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']

    params = {
        'objective': 'binary',
        'metric': 'auc',
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

    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    training_time = time.time() - start_time

    log_message(f"Exp C completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


# ============== EXPERIMENT D: LightGBM DART ==============
def train_exp_d_dart(train, val, feature_cols):
    """Exp D: LightGBM with DART boosting (dropout for regularization)."""
    log_message("Exp D: LightGBM DART boosting...")

    X_train = train[feature_cols]
    y_train = train['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = min(n_neg / n_pos, 100.0) if n_pos > 0 else 1.0

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'dart',  # DART boosting
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
        'drop_rate': 0.1,  # DART specific
        'skip_drop': 0.5,  # DART specific
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

    start_time = time.time()
    model = lgb.train(params, train_data, num_boost_round=500,
                      valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                      callbacks=callbacks)
    training_time = time.time() - start_time

    log_message(f"Exp D completed in {training_time:.1f}s, iter={model.best_iteration}", "SUCCESS")
    return model, training_time


# ============== EXPERIMENT E: Two-Stage Improved ==============
def train_exp_e_twostage(train, val, feature_cols, best_stage1_model):
    """Exp E: Two-Stage Improved (best classifier + regression)."""
    log_message("Exp E: Two-Stage Improved...")

    # Stage 2: Regression on gift samples only
    train_gift = train[train['is_gift'] == 1].copy()
    val_gift = val[val['is_gift'] == 1].copy()

    log_message(f"Stage 2 training on {len(train_gift):,} gift samples")

    X_train = train_gift[feature_cols]
    y_train = train_gift['target_raw']  # Raw amount, not log
    X_val = val_gift[feature_cols]
    y_val = val_gift['target_raw']

    params = {
        'objective': 'regression',
        'metric': 'mae',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
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

    start_time = time.time()
    stage2_model = lgb.train(params, train_data, num_boost_round=500,
                              valid_sets=[train_data, val_data], valid_names=['train', 'val'],
                              callbacks=callbacks)
    training_time = time.time() - start_time

    log_message(f"Exp E Stage 2 completed in {training_time:.1f}s, iter={stage2_model.best_iteration}", "SUCCESS")
    return best_stage1_model, stage2_model, training_time


# ============== EVALUATION FUNCTIONS ==============
def compute_precision_recall_at_k(y_true, y_pred_prob, k_pct=0.01):
    """Compute Precision@K and Recall@K."""
    n = len(y_true)
    k = max(1, int(n * k_pct))

    pred_order = np.argsort(-y_pred_prob)
    top_k_indices = pred_order[:k]

    # Precision@K = TP@K / K
    tp_at_k = y_true[top_k_indices].sum()
    precision_at_k = tp_at_k / k

    # Recall@K = TP@K / Total Positive
    total_positive = y_true.sum()
    recall_at_k = tp_at_k / total_positive if total_positive > 0 else 0

    return precision_at_k, recall_at_k


def compute_revenue_capture_at_k(y_true_amount, y_pred, k_pct=0.01):
    """Compute Revenue Capture@K."""
    n = len(y_true_amount)
    k = max(1, int(n * k_pct))

    pred_order = np.argsort(-y_pred)
    top_k_indices = pred_order[:k]

    revenue_top_k = y_true_amount[top_k_indices].sum()
    total_revenue = y_true_amount.sum()

    return revenue_top_k / total_revenue if total_revenue > 0 else 0


def compute_top_k_capture(y_true, y_pred, k_pct=0.01):
    """Compute Top-K% capture rate (set overlap)."""
    n = len(y_true)
    k = max(1, int(n * k_pct))

    true_rank = np.argsort(np.argsort(-y_true))
    pred_rank = np.argsort(np.argsort(-y_pred))

    true_topk = set(np.where(true_rank < k)[0])
    pred_topk = set(np.where(pred_rank < k)[0])

    return len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0


def evaluate_classifier(model, test, feature_cols):
    """Evaluate binary classifier."""
    X_test = test[feature_cols]
    y_test = test['is_gift'].values
    y_test_amount = test['target_raw'].values

    y_pred_prob = model.predict(X_test)

    # Metrics
    auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc = average_precision_score(y_test, y_pred_prob)

    # Precision/Recall at various K
    prec_1pct, recall_1pct = compute_precision_recall_at_k(y_test, y_pred_prob, 0.01)
    prec_5pct, recall_5pct = compute_precision_recall_at_k(y_test, y_pred_prob, 0.05)
    prec_10pct, recall_10pct = compute_precision_recall_at_k(y_test, y_pred_prob, 0.10)

    # Optimal F1
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_prob)
    f1_scores = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5

    metrics = {
        'auc': float(auc),
        'pr_auc': float(pr_auc),
        'precision_1pct': float(prec_1pct),
        'recall_1pct': float(recall_1pct),
        'precision_5pct': float(prec_5pct),
        'recall_5pct': float(recall_5pct),
        'precision_10pct': float(prec_10pct),
        'recall_10pct': float(recall_10pct),
        'best_f1': float(best_f1),
        'best_threshold': float(best_threshold)
    }

    log_message(f"AUC: {auc:.4f}, PR-AUC: {pr_auc:.4f}")
    log_message(f"Precision@1%: {prec_1pct:.4f} ({prec_1pct*100:.2f}%), Recall@1%: {recall_1pct:.4f} ({recall_1pct*100:.2f}%)")

    # ROC curve data for plotting
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

    return metrics, y_pred_prob, (fpr, tpr), (precision_curve, recall_curve)


def evaluate_twostage(stage1_model, stage2_model, test, feature_cols):
    """Evaluate Two-Stage model."""
    X_test = test[feature_cols]
    y_test = test['is_gift'].values
    y_test_amount = test['target_raw'].values

    # Stage 1: probability
    p_pred = stage1_model.predict(X_test)

    # Stage 2: amount
    m_pred = stage2_model.predict(X_test)

    # Combine: p * m
    ev_pred = p_pred * m_pred

    # Metrics
    spearman, _ = stats.spearmanr(y_test_amount, ev_pred)
    top_1pct_capture = compute_top_k_capture(y_test_amount, ev_pred, 0.01)
    revenue_1pct = compute_revenue_capture_at_k(y_test_amount, ev_pred, 0.01)

    metrics = {
        'spearman': float(spearman),
        'top_1pct_capture': float(top_1pct_capture),
        'revenue_capture_1pct': float(revenue_1pct),
        'stage1_auc': float(roc_auc_score(y_test, p_pred))
    }

    log_message(f"Two-Stage: Spearman={spearman:.4f}, Top-1%={top_1pct_capture*100:.2f}%, RevCap@1%={revenue_1pct*100:.2f}%")

    return metrics, ev_pred


# ============== PLOTTING FUNCTIONS ==============
def plot_roc_curves(roc_data, save_path):
    """Fig1: ROC curves comparison."""
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, color=colors[idx % len(colors)],
                linewidth=2, label=f'{name} (AUC={auc_val:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves: Binary Classification Methods', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_pr_curves(pr_data, save_path):
    """Fig2: Precision-Recall curves."""
    fig, ax = plt.subplots(figsize=(6, 5))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for idx, (name, (precision, recall, pr_auc)) in enumerate(pr_data.items()):
        ax.plot(recall, precision, color=colors[idx % len(colors)],
                linewidth=2, label=f'{name} (PR-AUC={pr_auc:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_precision_recall_at_k(results_dict, save_path):
    """Fig3: Precision and Recall at different K."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    k_values = [1, 5, 10]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (name, metrics) in enumerate(results_dict.items()):
        prec_values = [metrics[f'precision_{k}pct'] * 100 for k in k_values]
        recall_values = [metrics[f'recall_{k}pct'] * 100 for k in k_values]

        ax1.plot(k_values, prec_values, 'o-', color=colors[idx % len(colors)],
                linewidth=2, markersize=8, label=name)
        ax2.plot(k_values, recall_values, 'o-', color=colors[idx % len(colors)],
                linewidth=2, markersize=8, label=name)

    ax1.set_xlabel('Top-K%', fontsize=12)
    ax1.set_ylabel('Precision@K (%)', fontsize=12)
    ax1.set_title('Precision@K', fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xticks(k_values)

    ax2.set_xlabel('Top-K%', fontsize=12)
    ax2.set_ylabel('Recall@K (%)', fontsize=12)
    ax2.set_title('Recall@K', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xticks(k_values)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_model_comparison(results_dict, save_path):
    """Fig4: Model comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    models = list(results_dict.keys())
    auc_values = [results_dict[m]['auc'] for m in models]
    pr_auc_values = [results_dict[m]['pr_auc'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    ax1.bar(x - width/2, auc_values, width, label='AUC', color='steelblue')
    ax1.bar(x + width/2, pr_auc_values, width, label='PR-AUC', color='coral')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('AUC and PR-AUC Comparison', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')
    ax1.axhline(y=0.7, color='r', linestyle='--', linewidth=1, label='Target AUC=0.7')

    prec_1pct = [results_dict[m]['precision_1pct'] * 100 for m in models]
    recall_1pct = [results_dict[m]['recall_1pct'] * 100 for m in models]

    ax2.bar(x - width/2, prec_1pct, width, label='Precision@1%', color='steelblue')
    ax2.bar(x + width/2, recall_1pct, width, label='Recall@1%', color='coral')
    ax2.set_xlabel('Model', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title('Precision@1% and Recall@1% Comparison', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_twostage_improvement(results_dict, baseline_metrics, save_path):
    """Fig5: Two-Stage improvement comparison."""
    fig, ax = plt.subplots(figsize=(6, 5))

    models = ['Baseline (MVP-1.0)', 'Two-Stage Improved']
    top_1pct = [baseline_metrics['top_1pct_capture'] * 100,
                results_dict['top_1pct_capture'] * 100]
    revenue_1pct = [baseline_metrics['revenue_capture_1pct'] * 100,
                    results_dict['revenue_capture_1pct'] * 100]

    x = np.arange(len(models))
    width = 0.35

    ax.bar(x - width/2, top_1pct, width, label='Top-1% Capture', color='steelblue')
    ax.bar(x + width/2, revenue_1pct, width, label='Revenue Capture@1%', color='coral')
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Two-Stage Improvement vs Baseline', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # Add value labels
    for i, (t, r) in enumerate(zip(top_1pct, revenue_1pct)):
        ax.text(i - width/2, t + 0.5, f'{t:.1f}%', ha='center', fontsize=9)
        ax.text(i + width/2, r + 0.5, f'{r:.1f}%', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


# ============== MAIN ==============
def main():
    """Main execution function."""
    log_message("=" * 60)
    log_message("Binary Classification Experiment (MVP-1.3)")
    log_message("=" * 60)

    # Load data
    gift, user, streamer, room, click = load_data()

    # Prepare click-level data
    click_base = prepare_click_level_data(gift, click, label_window_hours=1)

    # Temporal split
    click_base_sorted = click_base.sort_values('timestamp').reset_index(drop=True)
    n = len(click_base_sorted)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    click_train = click_base_sorted.iloc[:train_end].copy()
    click_val = click_base_sorted.iloc[train_end:val_end].copy()
    click_test = click_base_sorted.iloc[val_end:].copy()

    log_message(f"Split: Train={len(click_train):,}, Val={len(click_val):,}, Test={len(click_test):,}")

    # Prepare features (frozen version)
    train_df, lookups = prepare_features(gift, user, streamer, room, click, click_train, click_train)
    val_df, _ = prepare_features(gift, user, streamer, room, click, click_val, click_train)
    test_df, _ = prepare_features(gift, user, streamer, room, click, click_test, click_train)

    feature_cols = get_feature_columns(train_df)
    log_message(f"Feature columns: {len(feature_cols)}")

    # Data validation
    gift_rate = train_df['is_gift'].mean()
    log_message(f"Training gift rate: {gift_rate*100:.2f}% (expected ~1.5%)")

    all_results = {}
    roc_data = {}
    pr_data = {}

    # ========== Exp A: LightGBM Default ==========
    log_message("\n" + "=" * 40)
    model_a, time_a = train_exp_a_default(train_df, val_df, feature_cols)
    metrics_a, pred_a, roc_a, pr_a = evaluate_classifier(model_a, test_df, feature_cols)
    metrics_a['training_time'] = time_a
    all_results['Exp A: Default'] = metrics_a
    roc_data['Exp A: Default'] = (roc_a[0], roc_a[1], metrics_a['auc'])
    pr_data['Exp A: Default'] = (pr_a[0], pr_a[1], metrics_a['pr_auc'])

    # ========== Exp B: LightGBM + scale_pos_weight ==========
    log_message("\n" + "=" * 40)
    model_b, time_b = train_exp_b_weighted(train_df, val_df, feature_cols)
    metrics_b, pred_b, roc_b, pr_b = evaluate_classifier(model_b, test_df, feature_cols)
    metrics_b['training_time'] = time_b
    all_results['Exp B: Weighted'] = metrics_b
    roc_data['Exp B: Weighted'] = (roc_b[0], roc_b[1], metrics_b['auc'])
    pr_data['Exp B: Weighted'] = (pr_b[0], pr_b[1], metrics_b['pr_auc'])

    # ========== Exp C: LightGBM + Undersampling ==========
    log_message("\n" + "=" * 40)
    model_c, time_c = train_exp_c_undersample(train_df, val_df, feature_cols, target_ratio=10)
    metrics_c, pred_c, roc_c, pr_c = evaluate_classifier(model_c, test_df, feature_cols)
    metrics_c['training_time'] = time_c
    all_results['Exp C: Undersample'] = metrics_c
    roc_data['Exp C: Undersample'] = (roc_c[0], roc_c[1], metrics_c['auc'])
    pr_data['Exp C: Undersample'] = (pr_c[0], pr_c[1], metrics_c['pr_auc'])

    # ========== Exp D: LightGBM DART ==========
    log_message("\n" + "=" * 40)
    model_d, time_d = train_exp_d_dart(train_df, val_df, feature_cols)
    metrics_d, pred_d, roc_d, pr_d = evaluate_classifier(model_d, test_df, feature_cols)
    metrics_d['training_time'] = time_d
    all_results['Exp D: DART'] = metrics_d
    roc_data['Exp D: DART'] = (roc_d[0], roc_d[1], metrics_d['auc'])
    pr_data['Exp D: DART'] = (pr_d[0], pr_d[1], metrics_d['pr_auc'])

    # ========== Exp E: Two-Stage Improved ==========
    log_message("\n" + "=" * 40)
    # Select best Stage 1 model based on AUC
    best_model_name = max(['Exp A: Default', 'Exp B: Weighted', 'Exp C: Undersample', 'Exp D: DART'],
                          key=lambda x: all_results[x]['auc'])
    log_message(f"Best Stage 1 model: {best_model_name} (AUC={all_results[best_model_name]['auc']:.4f})")

    if 'Exp D' in best_model_name:
        best_stage1_model = model_d
    elif 'Exp C' in best_model_name:
        best_stage1_model = model_c
    elif 'Exp B' in best_model_name:
        best_stage1_model = model_b
    else:
        best_stage1_model = model_a

    stage1, stage2, time_e = train_exp_e_twostage(train_df, val_df, feature_cols,
                                                   best_stage1_model)
    metrics_e, pred_e = evaluate_twostage(stage1, stage2, test_df, feature_cols)
    metrics_e['training_time'] = time_e
    metrics_e['best_stage1'] = best_model_name
    all_results['Exp E: Two-Stage'] = metrics_e

    # ========== Generate Plots ==========
    log_message("\n" + "=" * 60)
    log_message("Generating visualizations...")
    log_message("=" * 60)

    # Fig 1: ROC Curves
    plot_roc_curves(roc_data, IMG_DIR / 'binary_roc_curves.png')

    # Fig 2: PR Curves
    plot_pr_curves(pr_data, IMG_DIR / 'binary_pr_curves.png')

    # Fig 3: Precision/Recall at K
    classifier_results = {k: v for k, v in all_results.items() if 'Two-Stage' not in k}
    plot_precision_recall_at_k(classifier_results, IMG_DIR / 'binary_precision_recall_at_k.png')

    # Fig 4: Model Comparison
    plot_model_comparison(classifier_results, IMG_DIR / 'binary_model_comparison.png')

    # Fig 5: Two-Stage Improvement
    baseline_metrics = {
        'top_1pct_capture': 0.116,  # From MVP-1.0 Frozen Direct
        'revenue_capture_1pct': 0.216
    }
    plot_twostage_improvement(metrics_e, baseline_metrics, IMG_DIR / 'binary_twostage_improvement.png')

    # ========== Save Results ==========
    with open(RESULTS_DIR / 'binary_classification_eval_20260118.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ========== Summary ==========
    log_message("\n" + "=" * 60)
    log_message("EXPERIMENT SUMMARY")
    log_message("=" * 60)

    print("\n| Model | AUC | PR-AUC | Prec@1% | Recall@1% |")
    print("|-------|-----|--------|---------|-----------|")
    for name, m in all_results.items():
        if 'auc' in m:
            print(f"| {name} | {m['auc']:.4f} | {m['pr_auc']:.4f} | {m.get('precision_1pct', 0)*100:.2f}% | {m.get('recall_1pct', 0)*100:.2f}% |")

    if 'Exp E: Two-Stage' in all_results:
        m = all_results['Exp E: Two-Stage']
        print(f"\nTwo-Stage Improved:")
        print(f"  - Spearman: {m['spearman']:.4f}")
        print(f"  - Top-1% Capture: {m['top_1pct_capture']*100:.2f}%")
        print(f"  - Revenue Capture@1%: {m['revenue_capture_1pct']*100:.2f}%")

    # Check hypothesis
    best_auc = max(m['auc'] for m in all_results.values() if 'auc' in m)
    log_message(f"\nH3.1 (AUC > 0.70): {'PASS' if best_auc > 0.70 else 'FAIL'} (Best AUC = {best_auc:.4f})")

    best_prec = max(m.get('precision_1pct', 0) for m in all_results.values())
    log_message(f"H3.3 (Precision@1% > 5%): {'PASS' if best_prec > 0.05 else 'FAIL'} (Best = {best_prec*100:.2f}%)")

    log_message("\n" + "=" * 60)
    log_message("Experiment completed!", "SUCCESS")
    log_message(f"Results: {RESULTS_DIR / 'binary_classification_eval_20260118.json'}")
    log_message(f"Figures: {IMG_DIR}/binary_*.png")


if __name__ == '__main__':
    main()
