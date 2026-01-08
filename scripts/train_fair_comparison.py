#!/usr/bin/env python3
"""
Fair Comparison: Two-Stage vs Direct Regression on Click Data
==============================================================

Experiment: EXP-20260108-gift-allocation-04 (MVP-1.1-fair)

Goal: Fair comparison on the SAME click data (containing both gift and non-gift samples).
- Model A: Direct Regression on log(1+Y) using click data (includes Y=0)
- Model B: Two-Stage LightGBM (p(x) * m(x))

Key difference from previous experiments:
- Previous Baseline was trained on gift-only data (~72k)
- Previous Two-Stage was trained on click data (~4.9M)
- This experiment: BOTH models trained on click data for fair comparison

Outputs:
- Models: gift_allocation/models/fair_direct_reg_20260108.pkl
- Results: gift_allocation/results/fair_comparison_20260108.json
- Figures: gift_allocation/img/fair_*.png

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

# Paths - Note: output to gift_allocation/ (not experiments/gift_allocation/)
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_allocation"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for d in [IMG_DIR, RESULTS_DIR, MODELS_DIR, LOG_DIR]:
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
    
    IMPORTANT: Use CLICK data as base (all interactions including non-gift).
    This gives us both positive (with gift) and negative (no gift) samples.
    """
    log_message("=" * 50)
    log_message("Preparing features (click-based for fair comparison)...")
    log_message("=" * 50)
    
    # Create feature tables from historical data
    user_features = create_user_features(gift, click, user)
    streamer_features = create_streamer_features(gift, room, streamer)
    interaction = create_interaction_features(gift, click)
    
    # ============ Use CLICK as base (contains both gift and non-gift) ============
    log_message("Using CLICK data as base (both gift and non-gift interactions)...")
    
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
    gift_agg = gift.groupby(['user_id', 'streamer_id', 'live_id']).agg({
        'gift_price': 'sum'
    }).reset_index().rename(columns={'gift_price': 'total_gift_price'})
    
    # Merge gift info to clicks
    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id'], how='left')
    click_base['total_gift_price'] = click_base['total_gift_price'].fillna(0)
    click_base['gift_price'] = click_base['total_gift_price']
    
    log_message(f"Click base: {len(click_base):,} records")
    log_message(f"Gift samples: {(click_base['gift_price'] > 0).sum():,} ({(click_base['gift_price'] > 0).mean()*100:.2f}%)")
    
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
    
    # Log gift rates per split
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        gift_rate = split['is_gift'].mean() * 100
        log_message(f"  {name} gift rate: {gift_rate:.2f}%")
    
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


# ============ MODEL A: Direct Regression on Click ============

def train_direct_regression(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list):
    """Train Direct Regression model on log(1+Y) including Y=0 samples."""
    log_message("=" * 50)
    log_message("Training Model A: Direct Regression on Click (log(1+Y))")
    log_message("=" * 50)
    
    X_train = train[feature_cols]
    y_train = train['target']  # log(1+Y), many Y=0 → target=0
    X_val = val[feature_cols]
    y_val = val['target']
    
    log_message(f"Training samples: {len(X_train):,}")
    log_message(f"Target=0 ratio: {(y_train == 0).mean()*100:.2f}%")
    
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
    log_message(f"Direct Reg training completed in {training_time:.1f}s", "SUCCESS")
    log_message(f"Best iteration: {model.best_iteration}")
    
    return model, training_time


# ============ MODEL B: Two-Stage ============

def train_stage1_classifier(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list):
    """Train Stage 1: Binary classification (is_gift)."""
    log_message("=" * 50)
    log_message("Training Model B Stage 1: Binary Classification (is_gift)")
    log_message("=" * 50)
    
    X_train = train[feature_cols]
    y_train = train['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    if n_pos > 0 and n_neg > 0:
        scale_pos_weight = min(n_neg / n_pos, 100.0)
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
    log_message("Training Model B Stage 2: Regression (conditional amount)")
    log_message("=" * 50)
    
    # Filter to gift samples only
    train_gift = train[train['is_gift'] == 1].copy()
    val_gift = val[val['is_gift'] == 1].copy()
    
    log_message(f"Training on {len(train_gift):,} gift samples (from {len(train):,})")
    log_message(f"Validation on {len(val_gift):,} gift samples (from {len(val):,})")
    
    X_train = train_gift[feature_cols]
    y_train = train_gift['target']
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


# ============ Evaluation ============

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


def compute_metrics(y_true_raw, y_pred, y_true_log=None):
    """Compute all ranking and regression metrics."""
    n = len(y_true_raw)
    
    # Convert predictions to comparable scale
    y_pred_raw = np.expm1(np.maximum(y_pred, 0))  # Convert from log space if needed
    
    # If y_true_log not provided, compute it
    if y_true_log is None:
        y_true_log = np.log1p(y_true_raw)
    
    # MAE in log space
    y_pred_log = np.log1p(y_pred_raw)
    mae_log = np.mean(np.abs(y_true_log - y_pred_log))
    rmse_log = np.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
    
    # Spearman correlation
    spearman_corr, _ = stats.spearmanr(y_true_raw, y_pred_raw)
    
    # Top-K% capture rates
    y_true_rank = np.argsort(np.argsort(-y_true_raw))
    y_pred_rank = np.argsort(np.argsort(-y_pred_raw))
    
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
    
    ndcg_100 = ndcg_at_k(y_true_raw, y_pred_raw, k=100)
    
    return {
        'mae_log': mae_log,
        'rmse_log': rmse_log,
        'spearman': spearman_corr,
        'top_1pct_capture': top_1pct_capture,
        'top_5pct_capture': top_5pct_capture,
        'top_10pct_capture': top_10pct_capture,
        'ndcg_100': ndcg_100,
    }


def evaluate_direct_regression(model, df: pd.DataFrame, feature_cols: list, split_name: str = "test"):
    """Evaluate Direct Regression model."""
    log_message(f"Evaluating Direct Regression on {split_name} set...")
    
    X = df[feature_cols]
    y_true_raw = df['gift_price'].values
    y_true_log = df['target'].values
    
    # Predict (in log space)
    y_pred_log = model.predict(X)
    y_pred_raw = np.expm1(np.maximum(y_pred_log, 0))
    
    metrics = compute_metrics(y_true_raw, y_pred_log, y_true_log)
    
    log_message(f"MAE(log): {metrics['mae_log']:.4f}")
    log_message(f"Spearman: {metrics['spearman']:.4f}")
    log_message(f"Top-1% Capture: {metrics['top_1pct_capture']*100:.1f}%")
    log_message(f"Top-5% Capture: {metrics['top_5pct_capture']*100:.1f}%")
    log_message(f"NDCG@100: {metrics['ndcg_100']:.4f}")
    
    return metrics, y_pred_raw


def evaluate_two_stage(clf_model, reg_model, df: pd.DataFrame, feature_cols: list, split_name: str = "test"):
    """Evaluate Two-Stage model."""
    log_message(f"Evaluating Two-Stage on {split_name} set...")
    
    X = df[feature_cols]
    y_true_raw = df['gift_price'].values
    y_true_log = df['target'].values
    y_true_binary = df['is_gift'].values
    
    # Stage 1: Predict probability of gift
    p_x = clf_model.predict(X)
    
    # Stage 2: Predict conditional amount (in log space)
    m_x_log = reg_model.predict(X)
    m_x = np.expm1(np.maximum(m_x_log, 0))
    
    # Combined prediction: v(x) = p(x) * m(x)
    v_x = p_x * m_x
    
    # Stage 1 metrics
    pr_auc = compute_pr_auc(y_true_binary, p_x)
    ece = compute_ece(y_true_binary, p_x)
    logloss = log_loss(y_true_binary, np.clip(p_x, 1e-10, 1-1e-10))
    
    log_message(f"Stage 1 PR-AUC: {pr_auc:.4f}")
    log_message(f"Stage 1 ECE: {ece:.4f}")
    
    # Combined metrics
    v_x_log = np.log1p(v_x)
    metrics = compute_metrics(y_true_raw, v_x_log, y_true_log)
    
    # Add stage 1 metrics
    metrics['stage1_pr_auc'] = pr_auc
    metrics['stage1_ece'] = ece
    metrics['stage1_logloss'] = logloss
    
    log_message(f"MAE(log): {metrics['mae_log']:.4f}")
    log_message(f"Spearman: {metrics['spearman']:.4f}")
    log_message(f"Top-1% Capture: {metrics['top_1pct_capture']*100:.1f}%")
    log_message(f"Top-5% Capture: {metrics['top_5pct_capture']*100:.1f}%")
    log_message(f"NDCG@100: {metrics['ndcg_100']:.4f}")
    
    predictions = {'p_x': p_x, 'm_x': m_x, 'v_x': v_x}
    
    return metrics, predictions


# ============ Plotting Functions ============

def plot_comparison_topk(direct_metrics, two_stage_metrics, save_path):
    """Fig1: Grouped bar chart comparing Top-K% capture rates."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['Top-1%', 'Top-5%', 'Top-10%']
    direct_vals = [
        direct_metrics['top_1pct_capture'] * 100,
        direct_metrics['top_5pct_capture'] * 100,
        direct_metrics['top_10pct_capture'] * 100
    ]
    two_stage_vals = [
        two_stage_metrics['top_1pct_capture'] * 100,
        two_stage_metrics['top_5pct_capture'] * 100,
        two_stage_metrics['top_10pct_capture'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, direct_vals, width, label='Direct Regression', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, two_stage_vals, width, label='Two-Stage (p×m)', color='coral', alpha=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, direct_vals):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, two_stage_vals):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Capture Rate (%)', fontsize=12)
    ax.set_title('Fair Comparison: Top-K% Capture Rate', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_topk_curve(y_true_raw, direct_pred, two_stage_pred, save_path):
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
    
    direct_rates = get_capture_rates(direct_pred)
    two_stage_rates = get_capture_rates(two_stage_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(ks * 100, np.array(direct_rates) * 100, 'steelblue', lw=2, marker='s', 
            markersize=3, label='Direct Regression', alpha=0.8)
    ax.plot(ks * 100, np.array(two_stage_rates) * 100, 'coral', lw=2, marker='o', 
            markersize=3, label='Two-Stage (p×m)')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Perfect')
    
    ax.set_xlabel('Top-K% Threshold', fontsize=12)
    ax.set_ylabel('Capture Rate (%)', fontsize=12)
    ax.set_title('Fair Comparison: Top-K% Capture Rate Curve', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_scatter_comparison(y_true_raw, direct_pred, two_stage_pred, save_path):
    """Fig3: Scatter plot of predicted vs actual for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sample for visibility
    n = len(y_true_raw)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        y_true_plot = y_true_raw[idx]
        direct_plot = direct_pred[idx]
        two_stage_plot = two_stage_pred[idx]
    else:
        y_true_plot = y_true_raw
        direct_plot = direct_pred
        two_stage_plot = two_stage_pred
    
    # Direct Regression
    axes[0].scatter(np.log1p(direct_plot), np.log1p(y_true_plot), alpha=0.3, s=10, c='steelblue')
    min_val = min(np.log1p(direct_plot).min(), np.log1p(y_true_plot).min())
    max_val = max(np.log1p(direct_plot).max(), np.log1p(y_true_plot).max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    axes[0].set_xlabel('Predicted log(1+Y)', fontsize=11)
    axes[0].set_ylabel('Actual log(1+Y)', fontsize=11)
    axes[0].set_title('Direct Regression', fontsize=12)
    
    # Two-Stage
    axes[1].scatter(np.log1p(two_stage_plot), np.log1p(y_true_plot), alpha=0.3, s=10, c='coral')
    min_val = min(np.log1p(two_stage_plot).min(), np.log1p(y_true_plot).min())
    max_val = max(np.log1p(two_stage_plot).max(), np.log1p(y_true_plot).max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    axes[1].set_xlabel('Predicted log(1+v(x))', fontsize=11)
    axes[1].set_ylabel('Actual log(1+Y)', fontsize=11)
    axes[1].set_title('Two-Stage (p×m)', fontsize=12)
    
    plt.suptitle('Fair Comparison: Predicted vs Actual', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_feature_importance_comparison(direct_model, clf_model, reg_model, feature_cols, save_path, top_n=15):
    """Fig4: Feature importance for all models (2x1 subplot)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Direct Regression
    imp_direct = direct_model.feature_importance(importance_type='gain')
    df_direct = pd.DataFrame({'feature': feature_cols, 'importance': imp_direct}).sort_values('importance', ascending=False).head(top_n)
    
    axes[0].barh(range(len(df_direct)), df_direct['importance'].values, color='steelblue', alpha=0.8)
    axes[0].set_yticks(range(len(df_direct)))
    axes[0].set_yticklabels(df_direct['feature'].values)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Importance (Gain)', fontsize=11)
    axes[0].set_title('Direct Regression', fontsize=12)
    
    # Two-Stage (average of stage1 and stage2)
    imp1 = clf_model.feature_importance(importance_type='gain')
    imp2 = reg_model.feature_importance(importance_type='gain')
    # Normalize and average
    imp1_norm = imp1 / imp1.max() if imp1.max() > 0 else imp1
    imp2_norm = imp2 / imp2.max() if imp2.max() > 0 else imp2
    imp_avg = (imp1_norm + imp2_norm) / 2
    
    df_ts = pd.DataFrame({'feature': feature_cols, 'importance': imp_avg}).sort_values('importance', ascending=False).head(top_n)
    
    axes[1].barh(range(len(df_ts)), df_ts['importance'].values, color='coral', alpha=0.8)
    axes[1].set_yticks(range(len(df_ts)))
    axes[1].set_yticklabels(df_ts['feature'].values)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Importance (Normalized Avg)', fontsize=11)
    axes[1].set_title('Two-Stage (Avg of Stage1 & Stage2)', fontsize=12)
    
    plt.suptitle('Feature Importance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_all_metrics(direct_metrics, two_stage_metrics, save_path):
    """Fig5: Bar chart comparing all metrics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['Top-1%', 'Top-5%', 'Top-10%', 'Spearman', 'NDCG@100']
    direct_vals = [
        direct_metrics['top_1pct_capture'] * 100,
        direct_metrics['top_5pct_capture'] * 100,
        direct_metrics['top_10pct_capture'] * 100,
        direct_metrics['spearman'] * 100,
        direct_metrics['ndcg_100'] * 100
    ]
    two_stage_vals = [
        two_stage_metrics['top_1pct_capture'] * 100,
        two_stage_metrics['top_5pct_capture'] * 100,
        two_stage_metrics['top_10pct_capture'] * 100,
        two_stage_metrics['spearman'] * 100,
        two_stage_metrics['ndcg_100'] * 100
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, direct_vals, width, label='Direct Regression', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, two_stage_vals, width, label='Two-Stage (p×m)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Fair Comparison: All Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 110)
    
    # Add value labels
    for bar, val in zip(bars1, direct_vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, two_stage_vals):
        ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def save_results(direct_metrics, two_stage_metrics, train_size, val_size, test_size,
                 direct_time, ts_time_s1, ts_time_s2, direct_iter, clf_iter, reg_iter, save_path):
    """Save results to JSON."""
    
    # Compute deltas
    delta_top_1pct = two_stage_metrics['top_1pct_capture'] - direct_metrics['top_1pct_capture']
    delta_top_5pct = two_stage_metrics['top_5pct_capture'] - direct_metrics['top_5pct_capture']
    delta_spearman = two_stage_metrics['spearman'] - direct_metrics['spearman']
    
    # Decision rule: If Top-1% Capture improves ≥5% → accept two-stage
    delta_pct = delta_top_1pct * 100  # Convert to percentage points
    conclusion = "accept" if delta_pct >= 5.0 else "reject"
    
    results = {
        "experiment_id": "EXP-20260108-gift-allocation-04",
        "mvp": "MVP-1.1-fair",
        "timestamp": datetime.now().isoformat(),
        "data": {
            "train_size": int(train_size),
            "val_size": int(val_size),
            "test_size": int(test_size),
            "base": "click (contains both gift and non-gift samples)"
        },
        "models": {
            "direct_reg": {k: float(v) for k, v in direct_metrics.items()},
            "two_stage": {k: float(v) for k, v in two_stage_metrics.items()}
        },
        "comparison": {
            "delta_top_1pct": float(delta_top_1pct),
            "delta_top_1pct_pct_points": float(delta_pct),
            "delta_top_5pct": float(delta_top_5pct),
            "delta_spearman": float(delta_spearman),
            "conclusion": conclusion,
            "decision_rule": "If Top-1% Capture improves >= 5 percentage points → accept two-stage"
        },
        "training": {
            "direct_reg_time_seconds": float(direct_time),
            "direct_reg_best_iteration": int(direct_iter),
            "two_stage_s1_time_seconds": float(ts_time_s1),
            "two_stage_s2_time_seconds": float(ts_time_s2),
            "two_stage_s1_best_iteration": int(clf_iter),
            "two_stage_s2_best_iteration": int(reg_iter)
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"Results saved to: {save_path}", "SUCCESS")
    return results


def main():
    """Main training pipeline."""
    log_message("=" * 70)
    log_message("Fair Comparison: Two-Stage vs Direct Regression on Click Data")
    log_message("Experiment: MVP-1.1-fair | EXP-20260108-gift-allocation-04")
    log_message("=" * 70)
    
    # Load data
    gift, user, streamer, room, click = load_data()
    
    # Prepare features (using click as base)
    df = prepare_features(gift, user, streamer, room, click)
    
    # Temporal split (SAME split for both models)
    train, val, test = temporal_split(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    log_message(f"Number of features: {len(feature_cols)}")
    
    # ============ Train Model A: Direct Regression ============
    direct_model, direct_time = train_direct_regression(train, val, feature_cols)
    
    # Save direct model
    with open(MODELS_DIR / "fair_direct_reg_20260108.pkl", 'wb') as f:
        pickle.dump(direct_model, f)
    log_message("Direct Regression model saved", "SUCCESS")
    
    # ============ Train Model B: Two-Stage ============
    clf_model, ts_time_s1 = train_stage1_classifier(train, val, feature_cols)
    reg_model, ts_time_s2 = train_stage2_regressor(train, val, feature_cols)
    
    # Save two-stage models (for this fair comparison)
    with open(MODELS_DIR / "fair_two_stage_clf_20260108.pkl", 'wb') as f:
        pickle.dump(clf_model, f)
    with open(MODELS_DIR / "fair_two_stage_reg_20260108.pkl", 'wb') as f:
        pickle.dump(reg_model, f)
    log_message("Two-Stage models saved", "SUCCESS")
    
    # ============ Evaluate Both Models ============
    log_message("=" * 50)
    log_message("EVALUATION ON TEST SET")
    log_message("=" * 50)
    
    direct_metrics, direct_pred = evaluate_direct_regression(direct_model, test, feature_cols, "test")
    two_stage_metrics, ts_predictions = evaluate_two_stage(clf_model, reg_model, test, feature_cols, "test")
    
    y_true_raw = test['gift_price'].values
    two_stage_pred = ts_predictions['v_x']
    
    # ============ Generate Figures ============
    log_message("Generating figures...")
    
    # Fig1: Top-K% comparison bar chart
    plot_comparison_topk(direct_metrics, two_stage_metrics, IMG_DIR / "fair_comparison_topk.png")
    
    # Fig2: Top-K% curve
    plot_topk_curve(y_true_raw, direct_pred, two_stage_pred, IMG_DIR / "fair_comparison_topk_curve.png")
    
    # Fig3: Scatter plots
    plot_scatter_comparison(y_true_raw, direct_pred, two_stage_pred, IMG_DIR / "fair_comparison_scatter.png")
    
    # Fig4: Feature importance
    plot_feature_importance_comparison(direct_model, clf_model, reg_model, feature_cols, 
                                       IMG_DIR / "fair_comparison_feature_importance.png")
    
    # Fig5: All metrics comparison
    plot_all_metrics(direct_metrics, two_stage_metrics, IMG_DIR / "fair_comparison_all_metrics.png")
    
    # ============ Save Results ============
    results = save_results(
        direct_metrics, two_stage_metrics,
        len(train), len(val), len(test),
        direct_time, ts_time_s1, ts_time_s2,
        direct_model.best_iteration, clf_model.best_iteration, reg_model.best_iteration,
        RESULTS_DIR / "fair_comparison_20260108.json"
    )
    
    # ============ Print Summary ============
    print("\n" + "=" * 70)
    print("📊 EXPERIMENT SUMMARY - MVP-1.1-fair: Fair Comparison")
    print("=" * 70)
    print(f"Data: Click-based (contains gift + non-gift samples)")
    print(f"Train/Val/Test: {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"Features: {len(feature_cols)}")
    print("-" * 70)
    print("TRAINING TIME:")
    print(f"  Direct Regression: {direct_time:.1f}s (iter={direct_model.best_iteration})")
    print(f"  Two-Stage: S1={ts_time_s1:.1f}s (iter={clf_model.best_iteration}), S2={ts_time_s2:.1f}s (iter={reg_model.best_iteration})")
    print("-" * 70)
    print("MODEL A - DIRECT REGRESSION:")
    print(f"  MAE(log):        {direct_metrics['mae_log']:.4f}")
    print(f"  Spearman:        {direct_metrics['spearman']:.4f}")
    print(f"  Top-1% Capture:  {direct_metrics['top_1pct_capture']*100:.1f}%")
    print(f"  Top-5% Capture:  {direct_metrics['top_5pct_capture']*100:.1f}%")
    print(f"  Top-10% Capture: {direct_metrics['top_10pct_capture']*100:.1f}%")
    print(f"  NDCG@100:        {direct_metrics['ndcg_100']:.4f}")
    print("-" * 70)
    print("MODEL B - TWO-STAGE (p×m):")
    print(f"  PR-AUC (S1):     {two_stage_metrics['stage1_pr_auc']:.4f}")
    print(f"  ECE (S1):        {two_stage_metrics['stage1_ece']:.4f}")
    print(f"  MAE(log):        {two_stage_metrics['mae_log']:.4f}")
    print(f"  Spearman:        {two_stage_metrics['spearman']:.4f}")
    print(f"  Top-1% Capture:  {two_stage_metrics['top_1pct_capture']*100:.1f}%")
    print(f"  Top-5% Capture:  {two_stage_metrics['top_5pct_capture']*100:.1f}%")
    print(f"  Top-10% Capture: {two_stage_metrics['top_10pct_capture']*100:.1f}%")
    print(f"  NDCG@100:        {two_stage_metrics['ndcg_100']:.4f}")
    print("-" * 70)
    print("COMPARISON (Two-Stage - Direct):")
    delta_top1 = (two_stage_metrics['top_1pct_capture'] - direct_metrics['top_1pct_capture']) * 100
    delta_top5 = (two_stage_metrics['top_5pct_capture'] - direct_metrics['top_5pct_capture']) * 100
    delta_spearman = two_stage_metrics['spearman'] - direct_metrics['spearman']
    print(f"  Δ Top-1%:        {delta_top1:+.1f} pp {'✅' if delta_top1 >= 5 else '❌'} (threshold: ≥5pp)")
    print(f"  Δ Top-5%:        {delta_top5:+.1f} pp")
    print(f"  Δ Spearman:      {delta_spearman:+.4f}")
    print("-" * 70)
    print(f"🎯 DECISION: {results['comparison']['conclusion'].upper()}")
    if results['comparison']['conclusion'] == 'accept':
        print("   → Two-Stage model shows significant improvement. DG1 CLOSED ✅")
    else:
        print("   → Two-Stage improvement < 5pp. Consider keeping Direct Regression.")
    print("=" * 70)
    
    log_message("Experiment completed!", "SUCCESS")
    
    return results


if __name__ == "__main__":
    results = main()
