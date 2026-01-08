#!/usr/bin/env python3
"""
Two-Stage LightGBM Model for Gift Amount Prediction
=====================================================

Experiment: EXP-20260108-gift-allocation-03 (MVP-1.1)

Goal: Compare two-stage model (p(x) * m(x)) vs direct regression baseline.
- Stage 1: Binary classification (is_gift)
- Stage 2: Regression on log(1+Y) for gift samples only
- Final: v(x) = p(x) * m(x)

Outputs:
- Models: experiments/gift_allocation/models/two_stage_*.pkl
- Results: experiments/gift_allocation/results/two_stage_results_20260108.json
- Figures: experiments/gift_allocation/img/two_stage_*.png

Author: Viska Wei
Date: 2026-01-08
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
from sklearn.metrics import precision_recall_curve, auc, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set random seed
SEED = 42
np.random.seed(SEED)

# Paths
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "experiments" / "gift_allocation"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"

# Create directories
for d in [IMG_DIR, RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Plot settings
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Baseline metrics for comparison
BASELINE_METRICS = {
    'top_1pct_capture': 0.562,
    'top_5pct_capture': 0.763,
    'top_10pct_capture': 0.823,
    'spearman': 0.891,
    'mae_log': 0.263,
    'ndcg_100': 0.716
}


def log_message(msg: str, level: str = "INFO"):
    """Print log message with timestamp."""
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
    
    log_message(f"Gift records: {len(gift):,}")
    log_message(f"Users: {len(user):,}")
    log_message(f"Streamers: {len(streamer):,}")
    log_message(f"Rooms: {len(room):,}")
    log_message(f"Clicks: {len(click):,}")
    
    return gift, user, streamer, room, click


def create_user_features(gift: pd.DataFrame, click: pd.DataFrame, user: pd.DataFrame) -> pd.DataFrame:
    """Create user-level features from historical data."""
    log_message("Creating user features...")
    
    user_gift_stats = gift.groupby('user_id').agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'streamer_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    user_gift_stats.columns = [
        'user_id', 'user_gift_count', 'user_gift_sum', 'user_gift_mean', 
        'user_gift_std', 'user_gift_max', 'user_unique_streamers', 'user_unique_rooms'
    ]
    user_gift_stats['user_gift_std'] = user_gift_stats['user_gift_std'].fillna(0)
    
    user_click_stats = click.groupby('user_id').agg({
        'watch_live_time': ['count', 'sum', 'mean'],
        'streamer_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    user_click_stats.columns = [
        'user_id', 'user_watch_count', 'user_watch_time_sum', 'user_watch_time_mean',
        'user_watch_unique_streamers', 'user_watch_unique_rooms'
    ]
    
    user_features = user[['user_id', 'age', 'gender', 'device_brand', 'device_price',
                           'fans_num', 'follow_num', 'accu_watch_live_cnt', 
                           'accu_watch_live_duration', 'is_live_streamer', 'is_photo_author',
                           'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                           'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    user_features = user_features.merge(user_gift_stats, on='user_id', how='left')
    user_features = user_features.merge(user_click_stats, on='user_id', how='left')
    
    numeric_cols = user_features.select_dtypes(include=[np.number]).columns
    user_features[numeric_cols] = user_features[numeric_cols].fillna(0)
    
    return user_features


def create_streamer_features(gift: pd.DataFrame, room: pd.DataFrame, streamer: pd.DataFrame) -> pd.DataFrame:
    """Create streamer-level features from historical data."""
    log_message("Creating streamer features...")
    
    streamer_gift_stats = gift.groupby('streamer_id').agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'user_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    streamer_gift_stats.columns = [
        'streamer_id', 'streamer_gift_count', 'streamer_gift_sum', 'streamer_gift_mean',
        'streamer_gift_std', 'streamer_gift_max', 'streamer_unique_givers', 'streamer_unique_rooms'
    ]
    streamer_gift_stats['streamer_gift_std'] = streamer_gift_stats['streamer_gift_std'].fillna(0)
    
    streamer_room_stats = room.groupby('streamer_id').agg({
        'live_id': 'count',
        'live_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
    }).reset_index()
    streamer_room_stats.columns = ['streamer_id', 'streamer_room_count', 'streamer_main_live_type']
    
    streamer_features = streamer[['streamer_id', 'gender', 'age', 'device_brand', 'device_price',
                                   'live_operation_tag', 'fans_user_num', 'fans_group_fans_num',
                                   'follow_user_num', 'accu_live_cnt', 'accu_live_duration',
                                   'accu_play_cnt', 'accu_play_duration',
                                   'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                                   'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    streamer_features = streamer_features.rename(columns={
        'gender': 'streamer_gender',
        'age': 'streamer_age',
        'device_brand': 'streamer_device_brand',
        'device_price': 'streamer_device_price',
        'onehot_feat0': 'streamer_onehot_feat0',
        'onehot_feat1': 'streamer_onehot_feat1',
        'onehot_feat2': 'streamer_onehot_feat2',
        'onehot_feat3': 'streamer_onehot_feat3',
        'onehot_feat4': 'streamer_onehot_feat4',
        'onehot_feat5': 'streamer_onehot_feat5',
        'onehot_feat6': 'streamer_onehot_feat6',
    })
    
    streamer_features = streamer_features.merge(streamer_gift_stats, on='streamer_id', how='left')
    streamer_features = streamer_features.merge(streamer_room_stats, on='streamer_id', how='left')
    
    numeric_cols = streamer_features.select_dtypes(include=[np.number]).columns
    streamer_features[numeric_cols] = streamer_features[numeric_cols].fillna(0)
    
    return streamer_features


def create_context_features(gift: pd.DataFrame, room: pd.DataFrame) -> pd.DataFrame:
    """Create context features for each gift record."""
    log_message("Creating context features...")
    
    gift = gift.copy()
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')
    gift['hour'] = gift['timestamp_dt'].dt.hour
    gift['day_of_week'] = gift['timestamp_dt'].dt.dayofweek
    gift['is_weekend'] = (gift['day_of_week'] >= 5).astype(int)
    gift['date'] = gift['timestamp_dt'].dt.date
    
    room_info = room[['live_id', 'live_type', 'live_content_category']].drop_duplicates('live_id')
    gift = gift.merge(room_info, on='live_id', how='left')
    
    return gift


def create_interaction_features(gift: pd.DataFrame, click: pd.DataFrame) -> pd.DataFrame:
    """Create user-streamer interaction features."""
    log_message("Creating interaction features...")
    
    user_streamer_click = click.groupby(['user_id', 'streamer_id']).agg({
        'watch_live_time': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_click.columns = [
        'user_id', 'streamer_id', 
        'pair_watch_count', 'pair_watch_time_sum', 'pair_watch_time_mean'
    ]
    
    user_streamer_gift = gift.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_gift.columns = [
        'user_id', 'streamer_id',
        'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean'
    ]
    
    interaction = user_streamer_click.merge(user_streamer_gift, on=['user_id', 'streamer_id'], how='outer')
    interaction = interaction.fillna(0)
    
    return interaction


def encode_categorical(df: pd.DataFrame, cat_columns: list) -> pd.DataFrame:
    """Encode categorical columns using label encoding."""
    df = df.copy()
    for col in cat_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
            df[col] = df[col].astype('category').cat.codes
    return df


def prepare_features(gift, user, streamer, room, click):
    """Prepare all features for model training.
    
    Key difference from baseline: Use CLICK data as base (all interactions),
    then label with gift amount. This gives us negative samples for classification.
    """
    log_message("Preparing features (two-stage: click-based)...")
    
    # Create feature tables from historical data
    user_features = create_user_features(gift, click, user)
    streamer_features = create_streamer_features(gift, room, streamer)
    interaction = create_interaction_features(gift, click)
    
    # ============ KEY CHANGE: Use CLICK as base instead of GIFT ============
    # This gives us both positive (with gift) and negative (no gift) samples
    log_message("Using click data as base (contains both gift and non-gift interactions)...")
    
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['day_of_week'] = click_base['timestamp_dt'].dt.dayofweek
    click_base['is_weekend'] = (click_base['day_of_week'] >= 5).astype(int)
    click_base['date'] = click_base['timestamp_dt'].dt.date
    
    # Merge room info
    room_info = room[['live_id', 'live_type', 'live_content_category']].drop_duplicates('live_id')
    click_base = click_base.merge(room_info, on='live_id', how='left')
    
    # Aggregate gift amounts per click (user-streamer-live session)
    # A click may have multiple gifts, sum them up
    gift_agg = gift.groupby(['user_id', 'streamer_id', 'live_id']).agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={'gift_price': 'total_gift_price'})
    
    # Merge gift info to clicks
    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id'], how='left')
    click_base['total_gift_price'] = click_base['total_gift_price'].fillna(0)
    
    # Rename for consistency
    click_base['gift_price'] = click_base['total_gift_price']
    
    log_message(f"Click base: {len(click_base):,} records")
    
    # ============ Merge all features ============
    log_message("Merging all features...")
    df = click_base.merge(user_features, on='user_id', how='left')
    df = df.merge(streamer_features, on='streamer_id', how='left')
    df = df.merge(interaction, on=['user_id', 'streamer_id'], how='left')
    
    # Fill remaining NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Encode categorical features
    cat_columns = [
        'age', 'gender', 'device_brand', 'device_price', 'fans_num', 'follow_num',
        'accu_watch_live_cnt', 'accu_watch_live_duration',
        'streamer_gender', 'streamer_age', 'streamer_device_brand', 'streamer_device_price',
        'live_operation_tag', 'fans_user_num', 'fans_group_fans_num', 'follow_user_num',
        'accu_live_cnt', 'accu_live_duration', 'accu_play_cnt', 'accu_play_duration',
        'live_type', 'live_content_category'
    ]
    df = encode_categorical(df, cat_columns)
    
    # Targets
    df['target'] = np.log1p(df['gift_price'])  # log(1+Y) for regression
    df['is_gift'] = (df['gift_price'] > 0).astype(int)  # Binary for classification
    
    log_message(f"Final dataset: {len(df):,} records, {df.shape[1]} columns")
    log_message(f"Gift rate: {df['is_gift'].mean()*100:.2f}% ({df['is_gift'].sum():,} gifts)")
    
    return df


def temporal_split(df: pd.DataFrame):
    """Split data by time: train, val (7 days), test (last 7 days)."""
    log_message("Performing temporal split...")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    log_message(f"Date range: {min_date} to {max_date}")
    
    test_start = max_date - pd.Timedelta(days=6)
    val_start = test_start - pd.Timedelta(days=7)
    
    log_message(f"Train: {min_date} to {val_start - pd.Timedelta(days=1)}")
    log_message(f"Val: {val_start} to {test_start - pd.Timedelta(days=1)}")
    log_message(f"Test: {test_start} to {max_date}")
    
    train = df[df['date'] < val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test = df[df['date'] >= test_start].copy()
    
    log_message(f"Train size: {len(train):,} ({len(train)/len(df)*100:.1f}%)")
    log_message(f"Val size: {len(val):,} ({len(val)/len(df)*100:.1f}%)")
    log_message(f"Test size: {len(test):,} ({len(test)/len(df)*100:.1f}%)")
    
    return train, val, test


def get_feature_columns(df: pd.DataFrame):
    """Get list of feature columns (exclude target and identifiers)."""
    exclude_cols = [
        'user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
        'date', 'gift_price', 'total_gift_price', 'target', 'is_gift',
        'watch_live_time'  # This is from click, not a feature
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def train_stage1_classifier(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list):
    """Train Stage 1: Binary classification (is_gift)."""
    log_message("=" * 50)
    log_message("Training Stage 1: Binary Classification (is_gift)")
    log_message("=" * 50)
    
    X_train = train[feature_cols]
    y_train = train['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']
    
    # Calculate scale_pos_weight for imbalanced data
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    # Ensure scale_pos_weight is reasonable (between 1 and 100)
    if n_pos > 0 and n_neg > 0:
        scale_pos_weight = min(n_neg / n_pos, 100.0)  # Cap at 100
    else:
        scale_pos_weight = 1.0
    
    log_message(f"Positive samples: {n_pos:,} ({n_pos/len(y_train)*100:.2f}%)")
    log_message(f"Negative samples: {n_neg:,} ({n_neg/len(y_train)*100:.2f}%)")
    log_message(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'scale_pos_weight': scale_pos_weight,
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
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    log_message(f"Stage 1 training completed in {training_time:.1f}s", "SUCCESS")
    log_message(f"Best iteration: {model.best_iteration}")
    
    return model, training_time


def train_stage2_regressor(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list):
    """Train Stage 2: Regression on log(1+Y) for gift samples only."""
    log_message("=" * 50)
    log_message("Training Stage 2: Regression (conditional amount)")
    log_message("=" * 50)
    
    # Filter to gift samples only
    train_gift = train[train['is_gift'] == 1].copy()
    val_gift = val[val['is_gift'] == 1].copy()
    
    log_message(f"Training on {len(train_gift):,} gift samples (from {len(train):,})")
    log_message(f"Validation on {len(val_gift):,} gift samples (from {len(val):,})")
    
    X_train = train_gift[feature_cols]
    y_train = train_gift['target']  # log(1+Y)
    X_val = val_gift[feature_cols]
    y_val = val_gift['target']
    
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
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=callbacks
    )
    
    training_time = time.time() - start_time
    log_message(f"Stage 2 training completed in {training_time:.1f}s", "SUCCESS")
    log_message(f"Best iteration: {model.best_iteration}")
    
    return model, training_time


def compute_pr_auc(y_true, y_pred_proba):
    """Compute PR-AUC."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    return auc(recall, precision)


def compute_ece(y_true, y_pred_proba, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_pred_proba[mask].mean()
            bin_weight = mask.sum() / total_samples
            ece += bin_weight * np.abs(bin_accuracy - bin_confidence)
    
    return ece


def evaluate_two_stage(clf_model, reg_model, df: pd.DataFrame, feature_cols: list, split_name: str = "test"):
    """Evaluate two-stage model."""
    log_message(f"Evaluating two-stage model on {split_name} set...")
    
    X = df[feature_cols]
    y_true_raw = df['gift_price'].values
    y_true_log = df['target'].values
    y_true_binary = df['is_gift'].values
    
    # Stage 1: Predict probability of gift
    p_x = clf_model.predict(X)
    
    # Stage 2: Predict conditional amount (in log space)
    m_x_log = reg_model.predict(X)
    m_x = np.expm1(m_x_log)  # Convert back to original scale
    
    # Combined prediction: v(x) = p(x) * m(x)
    v_x = p_x * m_x
    
    # Also compute log-space combined prediction for MAE comparison
    # v_log = log(1 + p*m) approximately
    v_x_log = np.log1p(v_x)
    
    # ============ Stage 1 Metrics ============
    pr_auc = compute_pr_auc(y_true_binary, p_x)
    ece = compute_ece(y_true_binary, p_x)
    logloss = log_loss(y_true_binary, np.clip(p_x, 1e-10, 1-1e-10))
    
    log_message(f"Stage 1 PR-AUC: {pr_auc:.4f}")
    log_message(f"Stage 1 ECE: {ece:.4f}")
    log_message(f"Stage 1 Log Loss: {logloss:.4f}")
    
    # ============ Combined Metrics ============
    # MAE in log space (comparing with baseline)
    mae_log = np.mean(np.abs(y_true_log - v_x_log))
    rmse_log = np.sqrt(np.mean((y_true_log - v_x_log) ** 2))
    
    # Spearman correlation
    spearman_corr, _ = stats.spearmanr(y_true_raw, v_x)
    
    # Top-K% capture rates
    n = len(y_true_raw)
    y_true_rank = np.argsort(np.argsort(-y_true_raw))
    y_pred_rank = np.argsort(np.argsort(-v_x))
    
    def capture_rate(top_pct):
        k = max(1, int(n * top_pct))
        true_topk = set(np.where(y_true_rank < k)[0])
        pred_topk = set(np.where(y_pred_rank < k)[0])
        if len(true_topk) == 0:
            return 0.0
        return len(true_topk & pred_topk) / len(true_topk)
    
    top_1pct_capture = capture_rate(0.01)
    top_5pct_capture = capture_rate(0.05)
    top_10pct_capture = capture_rate(0.10)
    
    # NDCG@100
    def ndcg_at_k(y_true, y_pred, k=100):
        pred_order = np.argsort(-y_pred)[:k]
        dcg = 0.0
        for i, idx in enumerate(pred_order):
            rel = y_true[idx]
            dcg += rel / np.log2(i + 2)
        idcg = 0.0
        sorted_true = np.sort(y_true)[::-1][:k]
        for i, rel in enumerate(sorted_true):
            idcg += rel / np.log2(i + 2)
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    ndcg_100 = ndcg_at_k(y_true_raw, v_x, k=100)
    
    log_message(f"MAE(log): {mae_log:.4f}")
    log_message(f"RMSE(log): {rmse_log:.4f}")
    log_message(f"Spearman: {spearman_corr:.4f}")
    log_message(f"Top-1% Capture: {top_1pct_capture:.4f} ({top_1pct_capture*100:.1f}%)")
    log_message(f"Top-5% Capture: {top_5pct_capture:.4f}")
    log_message(f"Top-10% Capture: {top_10pct_capture:.4f}")
    log_message(f"NDCG@100: {ndcg_100:.4f}")
    
    metrics = {
        'stage1_pr_auc': pr_auc,
        'stage1_ece': ece,
        'stage1_logloss': logloss,
        'mae_log': mae_log,
        'rmse_log': rmse_log,
        'spearman': spearman_corr,
        'top_1pct_capture': top_1pct_capture,
        'top_5pct_capture': top_5pct_capture,
        'top_10pct_capture': top_10pct_capture,
        'ndcg_100': ndcg_100,
    }
    
    predictions = {
        'p_x': p_x,
        'm_x': m_x,
        'v_x': v_x,
        'v_x_log': v_x_log
    }
    
    return metrics, predictions


def plot_comparison_topk(two_stage_metrics, baseline_metrics, save_path):
    """Fig1: Bar chart comparing Top-K% capture rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Top-1%', 'Top-5%', 'Top-10%']
    baseline_vals = [
        baseline_metrics['top_1pct_capture'] * 100,
        baseline_metrics['top_5pct_capture'] * 100,
        baseline_metrics['top_10pct_capture'] * 100
    ]
    two_stage_vals = [
        two_stage_metrics['top_1pct_capture'] * 100,
        two_stage_metrics['top_5pct_capture'] * 100,
        two_stage_metrics['top_10pct_capture'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline (Direct Regression)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, two_stage_vals, width, label='Two-Stage (p×m)', color='coral', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, baseline_vals):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, two_stage_vals):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Capture Rate (%)', fontsize=12)
    ax.set_title('Two-Stage vs Baseline: Top-K% Capture Rate Comparison', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_topk_curve(y_true_raw, v_x, baseline_v_x, save_path):
    """Fig2: Top-K% capture rate curves for both models."""
    n = len(y_true_raw)
    ks = np.arange(0.01, 0.51, 0.01)
    
    y_true_rank = np.argsort(np.argsort(-y_true_raw))
    
    def get_capture_rates(y_pred):
        y_pred_rank = np.argsort(np.argsort(-y_pred))
        rates = []
        for k in ks:
            top_k = max(1, int(n * k))
            true_topk = set(np.where(y_true_rank < top_k)[0])
            pred_topk = set(np.where(y_pred_rank < top_k)[0])
            rate = len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0
            rates.append(rate)
        return rates
    
    two_stage_rates = get_capture_rates(v_x)
    baseline_rates = get_capture_rates(baseline_v_x)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ks * 100, np.array(two_stage_rates) * 100, 'coral', lw=2, marker='o', 
            markersize=3, label='Two-Stage (p×m)')
    ax.plot(ks * 100, np.array(baseline_rates) * 100, 'steelblue', lw=2, marker='s', 
            markersize=3, label='Baseline', alpha=0.7)
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    
    ax.set_xlabel('Top-K% Threshold', fontsize=12)
    ax.set_ylabel('Capture Rate (%)', fontsize=12)
    ax.set_title('Top-K% Capture Rate: Two-Stage vs Baseline', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_prauc_comparison(two_stage_metrics, save_path):
    """Fig3: PR-AUC for Stage 1."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['PR-AUC', 'ECE']
    vals = [two_stage_metrics['stage1_pr_auc'], two_stage_metrics['stage1_ece']]
    colors = ['coral', 'steelblue']
    
    bars = ax.bar(metrics, vals, color=colors, alpha=0.8)
    
    for bar, val in zip(bars, vals):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=12)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Stage 1 Classification Metrics', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_calibration_stage1(y_true_binary, p_x, save_path, n_bins=10):
    """Fig4: Stage 1 calibration plot."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = (p_x >= bin_edges[i]) & (p_x < bin_edges[i+1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
            bin_accs.append(y_true_binary[mask].mean())
            bin_counts.append(mask.sum())
    
    ax.bar(bin_centers, bin_accs, width=0.08, alpha=0.7, color='coral', label='Actual')
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect Calibration')
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Positive Rate', fontsize=12)
    ax.set_title('Stage 1 Calibration: Predicted p(x) vs Actual Gift Rate', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_pred_vs_actual(y_true_raw, v_x, save_path):
    """Fig5: Scatter plot of predicted v(x) vs actual Y."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n = len(y_true_raw)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        y_true_plot = y_true_raw[idx]
        v_x_plot = v_x[idx]
    else:
        y_true_plot = y_true_raw
        v_x_plot = v_x
    
    # Use log scale for better visualization
    ax.scatter(np.log1p(v_x_plot), np.log1p(y_true_plot), alpha=0.3, s=10, c='coral')
    
    min_val = min(np.log1p(v_x_plot).min(), np.log1p(y_true_plot).min())
    max_val = max(np.log1p(v_x_plot).max(), np.log1p(y_true_plot).max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect')
    
    ax.set_xlabel('Predicted log(1 + v(x))', fontsize=12)
    ax.set_ylabel('Actual log(1 + Y)', fontsize=12)
    ax.set_title('Two-Stage Model: Predicted vs Actual', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_feature_importance(clf_model, reg_model, feature_cols, save_path, top_n=15):
    """Fig6: Feature importance for both stages."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    # Stage 1
    imp1 = clf_model.feature_importance(importance_type='gain')
    df1 = pd.DataFrame({'feature': feature_cols, 'importance': imp1}).sort_values('importance', ascending=False).head(top_n)
    
    axes[0].barh(range(len(df1)), df1['importance'].values, color='coral', alpha=0.8)
    axes[0].set_yticks(range(len(df1)))
    axes[0].set_yticklabels(df1['feature'].values)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance (Gain)', fontsize=11)
    axes[0].set_title('Stage 1: Classification (is_gift)', fontsize=12)
    
    # Stage 2
    imp2 = reg_model.feature_importance(importance_type='gain')
    df2 = pd.DataFrame({'feature': feature_cols, 'importance': imp2}).sort_values('importance', ascending=False).head(top_n)
    
    axes[1].barh(range(len(df2)), df2['importance'].values, color='steelblue', alpha=0.8)
    axes[1].set_yticks(range(len(df2)))
    axes[1].set_yticklabels(df2['feature'].values)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Importance (Gain)', fontsize=11)
    axes[1].set_title('Stage 2: Regression (amount|gift)', fontsize=12)
    
    plt.suptitle('Feature Importance: Two-Stage Model', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")
    
    return df1, df2


def save_results(metrics, baseline_metrics, train_size, val_size, test_size, 
                 training_time_s1, training_time_s2, clf_iterations, reg_iterations, save_path):
    """Save results to JSON."""
    
    # Compute deltas
    delta_top_1pct = metrics['top_1pct_capture'] - baseline_metrics['top_1pct_capture']
    delta_spearman = metrics['spearman'] - baseline_metrics['spearman']
    delta_mae = metrics['mae_log'] - baseline_metrics['mae_log']
    
    # Decision rule: If PR-AUC and Top-1% both improve → accept two-stage
    conclusion = "accept" if delta_top_1pct > 0 else "reject"
    
    results = {
        "experiment_id": "EXP-20260108-gift-allocation-03",
        "model": "Two-Stage LightGBM",
        "timestamp": datetime.now().isoformat(),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "models": {
            "baseline": {k: float(v) for k, v in baseline_metrics.items()},
            "two_stage": {k: float(v) for k, v in metrics.items()}
        },
        "comparison": {
            "delta_top_1pct": float(delta_top_1pct),
            "delta_spearman": float(delta_spearman),
            "delta_mae_log": float(delta_mae),
            "conclusion": conclusion,
            "decision_rule": "If Top-1% improves → accept two-stage"
        },
        "training": {
            "stage1_time_seconds": float(training_time_s1),
            "stage2_time_seconds": float(training_time_s2),
            "stage1_best_iteration": int(clf_iterations),
            "stage2_best_iteration": int(reg_iterations)
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"Results saved to: {save_path}", "SUCCESS")
    return results


def main():
    """Main training pipeline."""
    log_message("=" * 60)
    log_message("Two-Stage LightGBM Model - MVP-1.1")
    log_message("=" * 60)
    
    # Load data
    gift, user, streamer, room, click = load_data()
    
    # Prepare features
    df = prepare_features(gift, user, streamer, room, click)
    
    # Temporal split
    train, val, test = temporal_split(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    log_message(f"Number of features: {len(feature_cols)}")
    
    # ============ Train Two-Stage Model ============
    
    # Stage 1: Classification
    clf_model, time_s1 = train_stage1_classifier(train, val, feature_cols)
    
    # Stage 2: Regression
    reg_model, time_s2 = train_stage2_regressor(train, val, feature_cols)
    
    # Save models
    with open(MODELS_DIR / "two_stage_clf_20260108.pkl", 'wb') as f:
        pickle.dump(clf_model, f)
    with open(MODELS_DIR / "two_stage_reg_20260108.pkl", 'wb') as f:
        pickle.dump(reg_model, f)
    log_message("Models saved", "SUCCESS")
    
    # ============ Evaluate ============
    metrics, predictions = evaluate_two_stage(clf_model, reg_model, test, feature_cols, "test")
    
    # Load baseline model for comparison curves
    baseline_model_path = MODELS_DIR / "baseline_lgb_20260108.pkl"
    if baseline_model_path.exists():
        with open(baseline_model_path, 'rb') as f:
            baseline_model = pickle.load(f)
        baseline_v_x = baseline_model.predict(test[feature_cols])
    else:
        log_message("Baseline model not found, using two-stage for comparison", "WARNING")
        baseline_v_x = predictions['v_x']
    
    # ============ Generate Figures ============
    log_message("Generating figures...")
    
    y_true_raw = test['gift_price'].values
    y_true_binary = test['is_gift'].values
    v_x = predictions['v_x']
    p_x = predictions['p_x']
    
    # Fig1: Comparison bar chart
    plot_comparison_topk(metrics, BASELINE_METRICS, IMG_DIR / "two_stage_comparison_topk.png")
    
    # Fig2: Top-K curve
    plot_topk_curve(y_true_raw, v_x, baseline_v_x, IMG_DIR / "two_stage_topk_curve.png")
    
    # Fig3: PR-AUC
    plot_prauc_comparison(metrics, IMG_DIR / "two_stage_prauc.png")
    
    # Fig4: Stage 1 calibration
    plot_calibration_stage1(y_true_binary, p_x, IMG_DIR / "two_stage_calibration_stage1.png")
    
    # Fig5: Pred vs Actual
    plot_pred_vs_actual(y_true_raw, v_x, IMG_DIR / "two_stage_pred_vs_actual.png")
    
    # Fig6: Feature importance
    feat_imp_s1, feat_imp_s2 = plot_feature_importance(
        clf_model, reg_model, feature_cols, 
        IMG_DIR / "two_stage_feature_importance.png"
    )
    
    # ============ Save Results ============
    results = save_results(
        metrics, BASELINE_METRICS,
        len(train), len(val), len(test),
        time_s1, time_s2,
        clf_model.best_iteration, reg_model.best_iteration,
        RESULTS_DIR / "two_stage_results_20260108.json"
    )
    
    # ============ Print Summary ============
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT SUMMARY - MVP-1.1 Two-Stage Model")
    print("=" * 60)
    print(f"Train/Val/Test: {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Training time: Stage1={time_s1:.1f}s, Stage2={time_s2:.1f}s")
    print("-" * 60)
    print("STAGE 1 (Classification) METRICS:")
    print(f"  PR-AUC:          {metrics['stage1_pr_auc']:.4f}")
    print(f"  ECE:             {metrics['stage1_ece']:.4f}")
    print(f"  Log Loss:        {metrics['stage1_logloss']:.4f}")
    print("-" * 60)
    print("COMBINED (v=p×m) METRICS:")
    print(f"  MAE(log):        {metrics['mae_log']:.4f}  (Baseline: {BASELINE_METRICS['mae_log']:.4f})")
    print(f"  Spearman:        {metrics['spearman']:.4f}  (Baseline: {BASELINE_METRICS['spearman']:.4f})")
    print(f"  Top-1% Capture:  {metrics['top_1pct_capture']*100:.1f}%  (Baseline: {BASELINE_METRICS['top_1pct_capture']*100:.1f}%)")
    print(f"  Top-5% Capture:  {metrics['top_5pct_capture']*100:.1f}%  (Baseline: {BASELINE_METRICS['top_5pct_capture']*100:.1f}%)")
    print(f"  Top-10% Capture: {metrics['top_10pct_capture']*100:.1f}%  (Baseline: {BASELINE_METRICS['top_10pct_capture']*100:.1f}%)")
    print(f"  NDCG@100:        {metrics['ndcg_100']:.4f}  (Baseline: {BASELINE_METRICS['ndcg_100']:.4f})")
    print("-" * 60)
    print("COMPARISON:")
    delta = metrics['top_1pct_capture'] - BASELINE_METRICS['top_1pct_capture']
    emoji = "✅" if delta > 0 else "❌"
    print(f"  Δ Top-1%:        {delta*100:+.1f}% {emoji}")
    print(f"  Δ Spearman:      {metrics['spearman'] - BASELINE_METRICS['spearman']:+.4f}")
    print(f"  Conclusion:      {results['comparison']['conclusion'].upper()}")
    print("=" * 60)
    
    log_message("Experiment completed!", "SUCCESS")
    
    return results


if __name__ == "__main__":
    results = main()
