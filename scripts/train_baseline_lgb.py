#!/usr/bin/env python3
"""
Baseline LightGBM Regression Model for Gift Amount Prediction
==============================================================

Experiment: EXP-20260108-gift-allocation-02 (MVP-0.2)

Goal: Build a direct regression baseline that predicts log(1+Y) gift amounts.

Outputs:
- Model: experiments/gift_allocation/models/baseline_lgb_20260108.pkl
- Results: experiments/gift_allocation/results/baseline_results_20260108.json
- Figures: experiments/gift_allocation/img/baseline_*.png

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
    plt.style.use('seaborn-whitegrid')
except:
    pass  # Use default style if seaborn style not available
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
    
    # Load main datasets
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
    
    # User historical gift stats
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
    
    # User click stats (watch behavior)
    user_click_stats = click.groupby('user_id').agg({
        'watch_live_time': ['count', 'sum', 'mean'],
        'streamer_id': 'nunique',
        'live_id': 'nunique'
    }).reset_index()
    user_click_stats.columns = [
        'user_id', 'user_watch_count', 'user_watch_time_sum', 'user_watch_time_mean',
        'user_watch_unique_streamers', 'user_watch_unique_rooms'
    ]
    
    # Merge with user profile
    user_features = user[['user_id', 'age', 'gender', 'device_brand', 'device_price',
                           'fans_num', 'follow_num', 'accu_watch_live_cnt', 
                           'accu_watch_live_duration', 'is_live_streamer', 'is_photo_author',
                           'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                           'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    user_features = user_features.merge(user_gift_stats, on='user_id', how='left')
    user_features = user_features.merge(user_click_stats, on='user_id', how='left')
    
    # Fill NaN for users without history
    numeric_cols = user_features.select_dtypes(include=[np.number]).columns
    user_features[numeric_cols] = user_features[numeric_cols].fillna(0)
    
    return user_features


def create_streamer_features(gift: pd.DataFrame, room: pd.DataFrame, streamer: pd.DataFrame) -> pd.DataFrame:
    """Create streamer-level features from historical data."""
    log_message("Creating streamer features...")
    
    # Streamer historical gift stats
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
    
    # Streamer room stats
    streamer_room_stats = room.groupby('streamer_id').agg({
        'live_id': 'count',
        'live_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0,
    }).reset_index()
    streamer_room_stats.columns = ['streamer_id', 'streamer_room_count', 'streamer_main_live_type']
    
    # Merge with streamer profile
    streamer_features = streamer[['streamer_id', 'gender', 'age', 'device_brand', 'device_price',
                                   'live_operation_tag', 'fans_user_num', 'fans_group_fans_num',
                                   'follow_user_num', 'accu_live_cnt', 'accu_live_duration',
                                   'accu_play_cnt', 'accu_play_duration',
                                   'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3',
                                   'onehot_feat4', 'onehot_feat5', 'onehot_feat6']].copy()
    
    # Rename to avoid column name conflicts
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
    
    # Fill NaN
    numeric_cols = streamer_features.select_dtypes(include=[np.number]).columns
    streamer_features[numeric_cols] = streamer_features[numeric_cols].fillna(0)
    
    return streamer_features


def create_context_features(gift: pd.DataFrame, room: pd.DataFrame) -> pd.DataFrame:
    """Create context features for each gift record."""
    log_message("Creating context features...")
    
    # Extract time features from timestamp
    gift = gift.copy()
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')
    gift['hour'] = gift['timestamp_dt'].dt.hour
    gift['day_of_week'] = gift['timestamp_dt'].dt.dayofweek
    gift['is_weekend'] = (gift['day_of_week'] >= 5).astype(int)
    gift['date'] = gift['timestamp_dt'].dt.date
    
    # Merge room info
    room_info = room[['live_id', 'live_type', 'live_content_category']].drop_duplicates('live_id')
    gift = gift.merge(room_info, on='live_id', how='left')
    
    return gift


def create_interaction_features(gift: pd.DataFrame, click: pd.DataFrame) -> pd.DataFrame:
    """Create user-streamer interaction features."""
    log_message("Creating interaction features...")
    
    # User-streamer historical interactions from click data
    user_streamer_click = click.groupby(['user_id', 'streamer_id']).agg({
        'watch_live_time': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_click.columns = [
        'user_id', 'streamer_id', 
        'pair_watch_count', 'pair_watch_time_sum', 'pair_watch_time_mean'
    ]
    
    # User-streamer historical gift interactions
    user_streamer_gift = gift.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_gift.columns = [
        'user_id', 'streamer_id',
        'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean'
    ]
    
    # Merge
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
    """Prepare all features for model training."""
    log_message("Preparing features...")
    
    # Create feature tables
    user_features = create_user_features(gift, click, user)
    streamer_features = create_streamer_features(gift, room, streamer)
    
    # Add context features to gift
    gift_enriched = create_context_features(gift, room)
    
    # Create interaction features
    interaction = create_interaction_features(gift, click)
    
    # Merge all features
    log_message("Merging all features...")
    df = gift_enriched.merge(user_features, on='user_id', how='left')
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
    
    # Target: log(1+Y)
    df['target'] = np.log1p(df['gift_price'])
    
    log_message(f"Final dataset: {len(df):,} records, {df.shape[1]} columns")
    
    return df


def temporal_split(df: pd.DataFrame):
    """Split data by time: train (N-14 days), val (N-13 to N-7), test (last 7 days)."""
    log_message("Performing temporal split...")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Get date range
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    log_message(f"Date range: {min_date} to {max_date}")
    
    # Calculate split dates
    total_days = (max_date - min_date).days + 1
    test_start = max_date - pd.Timedelta(days=6)  # Last 7 days
    val_start = test_start - pd.Timedelta(days=7)  # 7 days before test
    
    log_message(f"Train: {min_date} to {val_start - pd.Timedelta(days=1)}")
    log_message(f"Val: {val_start} to {test_start - pd.Timedelta(days=1)}")
    log_message(f"Test: {test_start} to {max_date}")
    
    # Split
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
        'date', 'gift_price', 'target'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def train_lightgbm(train: pd.DataFrame, val: pd.DataFrame, feature_cols: list):
    """Train LightGBM model."""
    log_message("Training LightGBM model...")
    
    X_train = train[feature_cols]
    y_train = train['target']
    X_val = val[feature_cols]
    y_val = val['target']
    
    # Model parameters
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
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train with early stopping
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
    log_message(f"Training completed in {training_time:.1f}s", "SUCCESS")
    log_message(f"Best iteration: {model.best_iteration}")
    
    return model, training_time


def evaluate_model(model, df: pd.DataFrame, feature_cols: list, split_name: str = "test"):
    """Evaluate model and compute all metrics."""
    log_message(f"Evaluating on {split_name} set...")
    
    X = df[feature_cols]
    y_true = df['target'].values  # log(1+Y)
    y_true_raw = df['gift_price'].values  # Raw gift amount
    
    # Predict
    y_pred = model.predict(X)
    
    # Regression metrics in log space
    mae_log = np.mean(np.abs(y_true - y_pred))
    rmse_log = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Spearman correlation (ranking ability)
    spearman_corr, _ = stats.spearmanr(y_true, y_pred)
    
    # Top-K% capture rates
    n = len(y_true)
    y_true_rank = np.argsort(np.argsort(-y_true_raw))  # Higher is better (rank 0 = highest)
    y_pred_rank = np.argsort(np.argsort(-y_pred))
    
    def capture_rate(top_pct):
        """What fraction of true top-K% users are captured in predicted top-K%?"""
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
        """Compute NDCG@k."""
        # Get top-k predicted items
        pred_order = np.argsort(-y_pred)[:k]
        true_order = np.argsort(-y_true)[:k]
        
        # DCG
        dcg = 0.0
        for i, idx in enumerate(pred_order):
            # Find rank of this item in true order
            rel = y_true[idx]
            dcg += rel / np.log2(i + 2)
        
        # IDCG (ideal)
        idcg = 0.0
        sorted_true = np.sort(y_true)[::-1][:k]
        for i, rel in enumerate(sorted_true):
            idcg += rel / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    ndcg_100 = ndcg_at_k(y_true_raw, y_pred, k=100)
    
    metrics = {
        'mae_log': mae_log,
        'rmse_log': rmse_log,
        'spearman': spearman_corr,
        'top_1pct_capture': top_1pct_capture,
        'top_5pct_capture': top_5pct_capture,
        'top_10pct_capture': top_10pct_capture,
        'ndcg_100': ndcg_100,
    }
    
    log_message(f"MAE(log): {mae_log:.4f}")
    log_message(f"RMSE(log): {rmse_log:.4f}")
    log_message(f"Spearman: {spearman_corr:.4f}")
    log_message(f"Top-1% Capture: {top_1pct_capture:.4f} ({top_1pct_capture*100:.1f}%)")
    log_message(f"Top-5% Capture: {top_5pct_capture:.4f}")
    log_message(f"Top-10% Capture: {top_10pct_capture:.4f}")
    log_message(f"NDCG@100: {ndcg_100:.4f}")
    
    return metrics, y_pred


def plot_pred_vs_actual(y_true, y_pred, save_path):
    """Fig1: Scatter plot of predicted vs actual log(1+Y)."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Sample for visibility if too many points
    n = len(y_true)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        y_true_plot = y_true[idx]
        y_pred_plot = y_pred[idx]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    ax.scatter(y_pred_plot, y_true_plot, alpha=0.3, s=10, c='steelblue')
    
    # Diagonal line
    min_val = min(y_pred_plot.min(), y_true_plot.min())
    max_val = max(y_pred_plot.max(), y_true_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Predicted log(1+Y)', fontsize=12)
    ax.set_ylabel('Actual log(1+Y)', fontsize=12)
    ax.set_title('Baseline LightGBM: Predicted vs Actual', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_feature_importance(model, feature_cols, save_path, top_n=20):
    """Fig2: Feature importance bar chart."""
    importance = model.feature_importance(importance_type='gain')
    feature_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    bars = ax.barh(range(len(feature_imp)), feature_imp['importance'].values, color='steelblue')
    ax.set_yticks(range(len(feature_imp)))
    ax.set_yticklabels(feature_imp['feature'].values)
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance (Gain)', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")
    
    return feature_imp


def plot_topk_capture(y_true_raw, y_pred, save_path):
    """Fig3: Top-K% capture rate curve."""
    n = len(y_true_raw)
    
    # Compute capture rates for different K values
    ks = np.arange(0.01, 0.51, 0.01)
    capture_rates = []
    
    y_true_rank = np.argsort(np.argsort(-y_true_raw))
    y_pred_rank = np.argsort(np.argsort(-y_pred))
    
    for k in ks:
        top_k = max(1, int(n * k))
        true_topk = set(np.where(y_true_rank < top_k)[0])
        pred_topk = set(np.where(y_pred_rank < top_k)[0])
        rate = len(true_topk & pred_topk) / len(true_topk) if len(true_topk) > 0 else 0
        capture_rates.append(rate)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(ks * 100, np.array(capture_rates) * 100, 'b-', lw=2, marker='o', markersize=3)
    ax.axhline(y=100, color='r', linestyle='--', alpha=0.7, label='Perfect (100%)')
    
    # Mark key points
    key_ks = [0.01, 0.05, 0.10]
    for k in key_ks:
        idx = int(k * 100) - 1
        if idx < len(capture_rates):
            ax.axvline(x=k*100, color='gray', linestyle=':', alpha=0.5)
            ax.annotate(f'{capture_rates[idx]*100:.1f}%', 
                       xy=(k*100, capture_rates[idx]*100),
                       xytext=(k*100+2, capture_rates[idx]*100+5),
                       fontsize=9)
    
    ax.set_xlabel('Top-K% Threshold', fontsize=12)
    ax.set_ylabel('Capture Rate (%)', fontsize=12)
    ax.set_title('Baseline LightGBM: Top-K% Capture Rate', fontsize=14)
    ax.legend()
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_learning_curve(model, save_path):
    """Fig4: Training/validation loss curve."""
    # Get evaluation results from model
    evals_result = {}
    
    # LightGBM stores results differently, we'll create synthetic curve from model info
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Since we don't have iteration-by-iteration results stored, 
    # we'll note this limitation
    iterations = np.arange(1, model.best_iteration + 1)
    
    # Create a simple visualization showing final performance
    ax.text(0.5, 0.5, f"Best Iteration: {model.best_iteration}\n\n"
            f"Note: Full learning curve requires\n"
            f"callback logging during training.",
            ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('MAE (train/val)', fontsize=12)
    ax.set_title('Baseline LightGBM: Learning Curve', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def plot_calibration(y_true, y_pred, save_path, n_buckets=10):
    """Fig5: Calibration plot - predicted deciles vs actual mean."""
    # Convert back from log space for interpretability
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred)
    
    # Create prediction decile buckets
    pred_deciles = pd.qcut(y_pred, q=n_buckets, labels=False, duplicates='drop')
    
    df_calib = pd.DataFrame({
        'actual': y_true_exp,
        'predicted': y_pred_exp,
        'bucket': pred_deciles
    })
    
    bucket_stats = df_calib.groupby('bucket').agg({
        'actual': 'mean',
        'predicted': 'mean'
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(bucket_stats))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, bucket_stats['predicted'], width, label='Mean Predicted', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, bucket_stats['actual'], width, label='Mean Actual', color='coral', alpha=0.8)
    
    ax.set_xlabel('Prediction Bucket (Decile)', fontsize=12)
    ax.set_ylabel('Mean Gift Amount', fontsize=12)
    ax.set_title('Baseline LightGBM: Calibration by Prediction Decile', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f'D{i+1}' for i in range(len(bucket_stats))])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {save_path}")


def save_results(metrics, feature_imp, train_size, val_size, test_size, training_time, save_path):
    """Save results to JSON."""
    results = {
        "model": "LightGBM",
        "target": "log(1+Y)",
        "experiment_id": "EXP-20260108-gift-allocation-02",
        "timestamp": datetime.now().isoformat(),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "test_size": int(test_size),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "top_features": [
            {"name": row['feature'], "importance": float(row['importance'])}
            for _, row in feature_imp.head(10).iterrows()
        ],
        "training_time_seconds": float(training_time)
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    log_message(f"Results saved to: {save_path}", "SUCCESS")
    return results


def main():
    """Main training pipeline."""
    log_message("=" * 60)
    log_message("Baseline LightGBM Regression - MVP-0.2")
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
    
    # Train model
    model, training_time = train_lightgbm(train, val, feature_cols)
    
    # Save model
    model_path = MODELS_DIR / "baseline_lgb_20260108.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    log_message(f"Model saved to: {model_path}", "SUCCESS")
    
    # Evaluate on test set
    metrics, y_pred_test = evaluate_model(model, test, feature_cols, "test")
    
    # Generate figures
    log_message("Generating figures...")
    
    y_true_test = test['target'].values
    y_true_raw_test = test['gift_price'].values
    
    # Fig1: Pred vs Actual
    plot_pred_vs_actual(y_true_test, y_pred_test, IMG_DIR / "baseline_pred_vs_actual.png")
    
    # Fig2: Feature Importance
    feature_imp = plot_feature_importance(model, feature_cols, IMG_DIR / "baseline_feature_importance.png")
    
    # Fig3: Top-K Capture
    plot_topk_capture(y_true_raw_test, y_pred_test, IMG_DIR / "baseline_topk_capture.png")
    
    # Fig4: Learning Curve
    plot_learning_curve(model, IMG_DIR / "baseline_learning_curve.png")
    
    # Fig5: Calibration
    plot_calibration(y_true_test, y_pred_test, IMG_DIR / "baseline_calibration.png")
    
    # Save results
    results = save_results(
        metrics, feature_imp, 
        len(train), len(val), len(test),
        training_time,
        RESULTS_DIR / "baseline_results_20260108.json"
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Train/Val/Test: {len(train):,} / {len(val):,} / {len(test):,}")
    print(f"Features: {len(feature_cols)}")
    print(f"Training time: {training_time:.1f}s")
    print("-" * 60)
    print("TEST METRICS:")
    print(f"  MAE(log):        {metrics['mae_log']:.4f}")
    print(f"  RMSE(log):       {metrics['rmse_log']:.4f}")
    print(f"  Spearman:        {metrics['spearman']:.4f}")
    print(f"  Top-1% Capture:  {metrics['top_1pct_capture']*100:.1f}%")
    print(f"  Top-5% Capture:  {metrics['top_5pct_capture']*100:.1f}%")
    print(f"  Top-10% Capture: {metrics['top_10pct_capture']*100:.1f}%")
    print(f"  NDCG@100:        {metrics['ndcg_100']:.4f}")
    print("-" * 60)
    print("TOP 5 FEATURES:")
    for i, row in feature_imp.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.1f}")
    print("=" * 60)
    
    log_message("Experiment completed!", "SUCCESS")
    
    return results


if __name__ == "__main__":
    results = main()
