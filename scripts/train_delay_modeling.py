#!/usr/bin/env python3
"""
Delay Feedback Modeling: Baseline vs Chapelle Method
=====================================================

Experiment: EXP-20260108-gift-allocation-05 (MVP-1.2)

Goal: Validate whether delay correction (Chapelle method) improves calibration
for gift prediction. Based on pre-exploration, delay is NOT significant 
(median=0), so we expect limited gains.

Decision Rule (DG2):
- If ECE improves >= 0.02 -> Adopt delay correction
- Else -> Close DG2, no delay modeling needed

Methods:
- Baseline: Direct binary classification with observed labels (0/1)
- Chapelle: Soft labels for negative samples: w_i = 1 - F(H - t_i)
  where F(d) is the CDF of delay distribution, H is observation horizon

Outputs:
- Results: gift_allocation/results/delay_modeling_20260108.json
- Figures: gift_allocation/img/delay_*.png

Author: Viska Wei
Date: 2026-01-08
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import (
    precision_recall_curve, auc, log_loss, brier_score_loss,
    roc_auc_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Set random seed
SEED = 42
np.random.seed(SEED)

# Paths
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_allocation"
IMG_DIR = OUTPUT_DIR / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
LOG_DIR = BASE_DIR / "logs"

# Create directories
for d in [IMG_DIR, RESULTS_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Plot settings
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def log_message(msg: str, level: str = "INFO"):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def load_data():
    """Load click and gift data with timestamps."""
    log_message("Loading data files...")
    
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    
    log_message(f"Gift records: {len(gift):,}")
    log_message(f"Click records: {len(click):,}")
    
    return gift, click, user, streamer, room


def analyze_delay_distribution(click: pd.DataFrame, gift: pd.DataFrame):
    """Analyze delay distribution from gift data."""
    log_message("=" * 50)
    log_message("Analyzing delay distribution...")
    log_message("=" * 50)
    
    # Merge click and gift on (user_id, live_id, streamer_id)
    merged = click.merge(
        gift, 
        on=['user_id', 'live_id', 'streamer_id'], 
        suffixes=('_click', '_gift')
    )
    
    log_message(f"Matched (user, live, streamer) pairs with gift: {len(merged):,}")
    
    # Compute delay in seconds
    merged['delay_ms'] = merged['timestamp_gift'] - merged['timestamp_click']
    merged['delay_sec'] = merged['delay_ms'] / 1000
    
    # Filter valid delays (positive, within watch time)
    valid = merged[(merged['delay_sec'] >= 0) & (merged['delay_sec'] <= merged['watch_live_time'])]
    log_message(f"Valid delays (0 <= delay <= watch_time): {len(valid):,} ({100*len(valid)/len(merged):.1f}%)")
    
    delay_sec = valid['delay_sec'].values
    watch_time = valid['watch_live_time'].values
    
    # Compute statistics
    delay_stats = {
        "n_samples": len(delay_sec),
        "mean": float(np.mean(delay_sec)),
        "median": float(np.median(delay_sec)),
        "std": float(np.std(delay_sec)),
        "p25": float(np.percentile(delay_sec, 25)),
        "p75": float(np.percentile(delay_sec, 75)),
        "p90": float(np.percentile(delay_sec, 90)),
        "p95": float(np.percentile(delay_sec, 95)),
        "p99": float(np.percentile(delay_sec, 99)),
        "max": float(np.max(delay_sec)),
    }
    
    # Relative delay (delay / watch_time)
    relative_delay = delay_sec / np.maximum(watch_time, 1)
    delay_stats["relative_delay_mean"] = float(np.mean(relative_delay))
    delay_stats["relative_delay_p50"] = float(np.median(relative_delay))
    delay_stats["pct_late_50"] = float(100 * np.mean(relative_delay > 0.5))
    delay_stats["pct_late_80"] = float(100 * np.mean(relative_delay > 0.8))
    delay_stats["pct_late_90"] = float(100 * np.mean(relative_delay > 0.9))
    
    log_message(f"Delay mean: {delay_stats['mean']:.1f}s, median: {delay_stats['median']:.1f}s")
    log_message(f"Delay P90: {delay_stats['p90']:.1f}s, P99: {delay_stats['p99']:.1f}s")
    log_message(f"Relative delay (>50% of watch time): {delay_stats['pct_late_50']:.1f}%")
    log_message(f"Relative delay (>90% of watch time): {delay_stats['pct_late_90']:.1f}%")
    
    return delay_sec, delay_stats


def fit_weibull_distribution(delay_sec: np.ndarray):
    """Fit Weibull distribution to delay data."""
    log_message("Fitting Weibull distribution to delays...")
    
    # Filter positive delays for Weibull fitting
    positive_delays = delay_sec[delay_sec > 0]
    
    if len(positive_delays) < 100:
        log_message("Too few positive delays, using empirical CDF", "WARNING")
        return None, None
    
    # Fit Weibull using scipy
    try:
        # Weibull minimum: shape (c), loc, scale
        shape, loc, scale = stats.weibull_min.fit(positive_delays, floc=0)
        
        log_message(f"Weibull params: shape={shape:.3f}, scale={scale:.3f}")
        
        return shape, scale
    except Exception as e:
        log_message(f"Weibull fitting failed: {e}", "WARNING")
        return None, None


def create_delay_cdf(delay_sec: np.ndarray, weibull_shape: float, weibull_scale: float):
    """Create delay CDF function."""
    if weibull_shape is not None and weibull_scale is not None:
        # Use fitted Weibull
        def delay_cdf(d):
            return stats.weibull_min.cdf(d, weibull_shape, scale=weibull_scale)
    else:
        # Use empirical CDF
        positive_delays = np.sort(delay_sec[delay_sec > 0])
        n = len(positive_delays)
        def delay_cdf(d):
            return np.searchsorted(positive_delays, d, side='right') / n
    
    return delay_cdf


def plot_delay_distribution(delay_sec: np.ndarray, weibull_shape: float, weibull_scale: float):
    """Plot delay distribution and fitted Weibull."""
    log_message("Plotting delay distribution...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Plot 1: Histogram of delays
    ax = axes[0]
    positive_delays = delay_sec[delay_sec > 0]
    ax.hist(positive_delays, bins=50, density=True, alpha=0.7, label='Observed')
    
    if weibull_shape is not None:
        x = np.linspace(0, np.percentile(positive_delays, 99), 100)
        y = stats.weibull_min.pdf(x, weibull_shape, scale=weibull_scale)
        ax.plot(x, y, 'r-', linewidth=2, label=f'Weibull (k={weibull_shape:.2f})')
    
    ax.set_xlabel('Delay (seconds)')
    ax.set_ylabel('Density')
    ax.set_title('Delay Distribution (Positive Delays Only)')
    ax.legend()
    
    # Plot 2: CDF comparison
    ax = axes[1]
    sorted_delays = np.sort(positive_delays)
    ecdf = np.arange(1, len(sorted_delays) + 1) / len(sorted_delays)
    ax.plot(sorted_delays, ecdf, 'b-', linewidth=2, label='Empirical CDF')
    
    if weibull_shape is not None:
        x = np.linspace(0, np.percentile(positive_delays, 99), 100)
        y = stats.weibull_min.cdf(x, weibull_shape, scale=weibull_scale)
        ax.plot(x, y, 'r--', linewidth=2, label='Weibull CDF')
    
    ax.set_xlabel('Delay (seconds)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Delay CDF')
    ax.legend()
    
    # Plot 3: Delay distribution - all samples (including 0)
    ax = axes[2]
    zero_frac = np.mean(delay_sec == 0)
    ax.bar(['delay=0', 'delay>0'], [zero_frac * 100, (1 - zero_frac) * 100], 
           color=['steelblue', 'coral'])
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Delay Distribution: {zero_frac*100:.1f}% Immediate')
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "delay_distribution.png", bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {IMG_DIR}/delay_distribution.png", "SUCCESS")


def create_features(gift, click, user, streamer, room):
    """Create features for classification model."""
    log_message("Creating features...")
    
    # User features
    user_gift_stats = gift.groupby('user_id').agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    user_gift_stats.columns = ['user_id', 'user_gift_count', 'user_gift_sum', 'user_gift_mean']
    
    user_click_stats = click.groupby('user_id').agg({
        'watch_live_time': ['count', 'sum', 'mean']
    }).reset_index()
    user_click_stats.columns = ['user_id', 'user_watch_count', 'user_watch_time_sum', 'user_watch_time_mean']
    
    user_features = user[['user_id', 'age', 'gender', 'device_brand', 'device_price',
                          'fans_num', 'follow_num', 'accu_watch_live_cnt']].copy()
    user_features = user_features.merge(user_gift_stats, on='user_id', how='left')
    user_features = user_features.merge(user_click_stats, on='user_id', how='left')
    user_features = user_features.fillna(0)
    
    # Streamer features  
    streamer_gift_stats = gift.groupby('streamer_id').agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    streamer_gift_stats.columns = ['streamer_id', 'streamer_gift_count', 'streamer_gift_sum', 'streamer_gift_mean']
    
    streamer_features = streamer[['streamer_id', 'gender', 'age', 'device_brand', 
                                   'fans_user_num', 'accu_live_cnt']].copy()
    streamer_features = streamer_features.rename(columns={
        'gender': 'streamer_gender', 'age': 'streamer_age',
        'device_brand': 'streamer_device_brand'
    })
    streamer_features = streamer_features.merge(streamer_gift_stats, on='streamer_id', how='left')
    streamer_features = streamer_features.fillna(0)
    
    # Interaction features
    user_streamer_gift = gift.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_gift.columns = ['user_id', 'streamer_id', 'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean']
    
    user_streamer_click = click.groupby(['user_id', 'streamer_id']).agg({
        'watch_live_time': ['count', 'sum', 'mean']
    }).reset_index()
    user_streamer_click.columns = ['user_id', 'streamer_id', 'pair_watch_count', 'pair_watch_time_sum', 'pair_watch_time_mean']
    
    interaction = user_streamer_click.merge(user_streamer_gift, on=['user_id', 'streamer_id'], how='outer')
    interaction = interaction.fillna(0)
    
    # Build base dataset from clicks
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['date'] = click_base['timestamp_dt'].dt.date
    
    # Merge gift info
    gift_agg = gift.groupby(['user_id', 'streamer_id', 'live_id']).agg({
        'gift_price': 'sum',
        'timestamp': 'min'  # First gift time
    }).reset_index().rename(columns={'gift_price': 'total_gift_price', 'timestamp': 'gift_timestamp'})
    
    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id'], how='left')
    click_base['total_gift_price'] = click_base['total_gift_price'].fillna(0)
    click_base['is_gift'] = (click_base['total_gift_price'] > 0).astype(int)
    
    # Compute delay for gift samples
    click_base['delay_sec'] = np.where(
        click_base['gift_timestamp'].notna(),
        (click_base['gift_timestamp'] - click_base['timestamp']) / 1000,
        np.nan
    )
    
    # Merge features
    df = click_base.merge(user_features, on='user_id', how='left')
    df = df.merge(streamer_features, on='streamer_id', how='left')
    df = df.merge(interaction, on=['user_id', 'streamer_id'], how='left')
    
    # Encode all object/category columns (except date and timestamp columns)
    exclude_encode = ['date', 'timestamp', 'timestamp_dt', 'gift_timestamp']
    for col in df.columns:
        if col in exclude_encode:
            continue
        if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
            df[col] = df[col].fillna('unknown').astype('category').cat.codes
    
    df = df.fillna(0)
    
    # Ensure all numeric (except exclusions)
    for col in df.select_dtypes(include=['object']).columns:
        if col not in exclude_encode:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    log_message(f"Dataset: {len(df):,} records, gift rate: {df['is_gift'].mean()*100:.2f}%")
    
    return df


def temporal_split(df: pd.DataFrame):
    """Split data by time."""
    log_message("Performing temporal split...")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    test_start = max_date - pd.Timedelta(days=6)
    val_start = test_start - pd.Timedelta(days=7)
    
    train = df[df['date'] < val_start].copy()
    val = df[(df['date'] >= val_start) & (df['date'] < test_start)].copy()
    test = df[df['date'] >= test_start].copy()
    
    log_message(f"Train: {len(train):,}, Val: {len(val):,}, Test: {len(test):,}")
    
    return train, val, test


def get_feature_columns(df: pd.DataFrame):
    """Get feature columns."""
    exclude_cols = [
        'user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
        'date', 'total_gift_price', 'is_gift', 'gift_timestamp', 'delay_sec',
        'watch_live_time', 'hour'
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def compute_chapelle_weights(df: pd.DataFrame, delay_cdf, max_observation_horizon: float = None):
    """
    Compute Chapelle soft labels for negative samples.
    
    For positive samples: weight = 1
    For negative samples: weight = 1 - F(H - t_i)
        where H is observation horizon, t_i is time since click
        
    In our case, H = watch_live_time (user's observation window)
    t_i = watch_live_time (time elapsed = full observation window)
    So F(H - t_i) = F(0) = 0, meaning weight = 1 for negatives
    
    This confirms our expectation: if gifts happen immediately (median=0),
    the Chapelle correction has little effect because F(0) ≈ 0.
    """
    log_message("Computing Chapelle soft labels...")
    
    weights = np.ones(len(df))
    
    # For positive samples, weight = 1 (already set)
    
    # For negative samples, compute 1 - F(H - t_i)
    # H = watch_live_time (observation horizon)
    # t_i = watch_live_time (we observed full duration)
    # So remaining time = H - t_i = 0
    
    neg_mask = df['is_gift'] == 0
    observation_time = df.loc[neg_mask, 'watch_live_time'].values
    
    # Since we observe full watch_live_time, remaining = 0
    # But to be more realistic, let's assume we're at end of observation window
    # and compute probability of conversion in remaining time
    
    # Simplified: weight = 1 - F(max_delay - observation_time)
    # For observed negatives after full watch, F(0) ≈ 0, so weight ≈ 1
    
    # Alternative interpretation: for negatives, compute probability they would
    # have converted if observed longer. Since delay median=0, this is minimal.
    
    # For experiment, let's use: weight = 1 - F(remaining_time)
    # where remaining_time = some_horizon - observation_time
    # Use median watch time as horizon
    if max_observation_horizon is None:
        max_observation_horizon = df['watch_live_time'].median()
    
    remaining_time = np.maximum(max_observation_horizon - observation_time, 0)
    conversion_prob = np.array([delay_cdf(t) for t in remaining_time])
    
    # Soft label for negatives: 1 - P(would have converted in remaining time)
    weights[neg_mask] = 1 - conversion_prob
    
    log_message(f"Negative samples: {neg_mask.sum():,}")
    log_message(f"Weight range for negatives: [{weights[neg_mask].min():.4f}, {weights[neg_mask].max():.4f}]")
    log_message(f"Mean weight for negatives: {weights[neg_mask].mean():.4f}")
    
    return weights


def train_baseline(train, val, feature_cols):
    """Train baseline binary classifier (no delay correction)."""
    log_message("=" * 50)
    log_message("Training Baseline (no delay correction)")
    log_message("=" * 50)
    
    X_train = train[feature_cols]
    y_train = train['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = min(n_neg / n_pos, 100.0) if n_pos > 0 else 1.0
    
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
    
    log_message(f"Baseline training: {training_time:.1f}s, best iter: {model.best_iteration}", "SUCCESS")
    
    return model


def train_chapelle(train, val, feature_cols, weights):
    """Train Chapelle model with soft labels (weighted)."""
    log_message("=" * 50)
    log_message("Training Chapelle (delay correction)")
    log_message("=" * 50)
    
    X_train = train[feature_cols]
    # Soft labels: for positives, keep 1; for negatives, use weight
    y_train_soft = train['is_gift'].values * 1.0  # Start with original labels
    # Note: In Chapelle, negatives get soft labels based on weights
    # But LightGBM doesn't support continuous labels for binary
    # So we use weights as sample weights instead
    
    y_train = train['is_gift']
    X_val = val[feature_cols]
    y_val = val['is_gift']
    
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = min(n_neg / n_pos, 100.0) if n_pos > 0 else 1.0
    
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
    
    # Use weights
    train_weights = weights[:len(train)]
    train_data = lgb.Dataset(X_train, label=y_train, weight=train_weights)
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
    
    log_message(f"Chapelle training: {training_time:.1f}s, best iter: {model.best_iteration}", "SUCCESS")
    
    return model


def compute_ece(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def evaluate_model(model, test, feature_cols, name: str):
    """Evaluate model and return metrics."""
    log_message(f"Evaluating {name}...")
    
    X_test = test[feature_cols]
    y_test = test['is_gift'].values
    
    y_prob = model.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Metrics
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    ece = compute_ece(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)
    
    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "ece": float(ece),
        "brier_score": float(brier),
        "log_loss": float(logloss),
    }
    
    log_message(f"  PR-AUC: {pr_auc:.4f}")
    log_message(f"  ROC-AUC: {roc_auc:.4f}")
    log_message(f"  ECE: {ece:.4f}")
    log_message(f"  Brier Score: {brier:.4f}")
    
    return metrics, y_prob


def plot_calibration_comparison(y_test, y_prob_baseline, y_prob_chapelle):
    """Plot calibration curves comparison."""
    log_message("Plotting calibration comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Calibration curves
    ax = axes[0]
    for name, y_prob, color in [('Baseline', y_prob_baseline, 'blue'), 
                                 ('Chapelle', y_prob_chapelle, 'red')]:
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_prob, n_bins=10, strategy='uniform'
        )
        ax.plot(mean_predicted_value, fraction_of_positives, 's-', 
                label=name, color=color, linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curves')
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Prediction distribution
    ax = axes[1]
    ax.hist(y_prob_baseline, bins=50, alpha=0.5, label='Baseline', density=True)
    ax.hist(y_prob_chapelle, bins=50, alpha=0.5, label='Chapelle', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "delay_calibration_comparison.png", bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {IMG_DIR}/delay_calibration_comparison.png", "SUCCESS")


def plot_ece_comparison(metrics_baseline, metrics_chapelle):
    """Plot ECE comparison bar chart."""
    log_message("Plotting ECE comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # ECE comparison
    ax = axes[0]
    methods = ['Baseline', 'Chapelle']
    ece_values = [metrics_baseline['ece'], metrics_chapelle['ece']]
    colors = ['steelblue', 'coral']
    bars = ax.bar(methods, ece_values, color=colors)
    ax.axhline(y=0.02, color='green', linestyle='--', label='Target threshold')
    ax.set_ylabel('ECE (Expected Calibration Error)')
    ax.set_title('ECE Comparison')
    ax.legend()
    
    # Add values on bars
    for bar, val in zip(bars, ece_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{val:.4f}', ha='center', va='bottom')
    
    # All metrics comparison
    ax = axes[1]
    metrics_names = ['PR-AUC', 'ROC-AUC', 'Brier', 'Log Loss']
    baseline_vals = [metrics_baseline['pr_auc'], metrics_baseline['roc_auc'],
                     metrics_baseline['brier_score'], metrics_baseline['log_loss']]
    chapelle_vals = [metrics_chapelle['pr_auc'], metrics_chapelle['roc_auc'],
                     metrics_chapelle['brier_score'], metrics_chapelle['log_loss']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='steelblue')
    ax.bar(x + width/2, chapelle_vals, width, label='Chapelle', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel('Value')
    ax.set_title('All Metrics Comparison')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(IMG_DIR / "delay_ece_comparison.png", bbox_inches='tight')
    plt.close()
    log_message(f"Saved: {IMG_DIR}/delay_ece_comparison.png", "SUCCESS")


def main():
    """Main experiment function."""
    log_message("=" * 60)
    log_message("MVP-1.2: Delay Feedback Modeling Experiment")
    log_message("=" * 60)
    
    start_time = time.time()
    
    # 1. Load data
    gift, click, user, streamer, room = load_data()
    
    # 2. Analyze delay distribution
    delay_sec, delay_stats = analyze_delay_distribution(click, gift)
    
    # 3. Fit Weibull distribution
    weibull_shape, weibull_scale = fit_weibull_distribution(delay_sec)
    
    # 4. Plot delay distribution
    plot_delay_distribution(delay_sec, weibull_shape, weibull_scale)
    
    # 5. Create features
    df = create_features(gift, click, user, streamer, room)
    
    # 6. Split data
    train, val, test = temporal_split(df)
    
    # 7. Get feature columns
    feature_cols = get_feature_columns(df)
    log_message(f"Using {len(feature_cols)} features")
    
    # 8. Create delay CDF
    delay_cdf = create_delay_cdf(delay_sec, weibull_shape, weibull_scale)
    
    # 9. Compute Chapelle weights
    chapelle_weights = compute_chapelle_weights(train, delay_cdf)
    
    # 10. Train models
    model_baseline = train_baseline(train, val, feature_cols)
    model_chapelle = train_chapelle(train, val, feature_cols, chapelle_weights)
    
    # 11. Evaluate models
    metrics_baseline, y_prob_baseline = evaluate_model(model_baseline, test, feature_cols, "Baseline")
    metrics_chapelle, y_prob_chapelle = evaluate_model(model_chapelle, test, feature_cols, "Chapelle")
    
    # 12. Plot comparisons
    y_test = test['is_gift'].values
    plot_calibration_comparison(y_test, y_prob_baseline, y_prob_chapelle)
    plot_ece_comparison(metrics_baseline, metrics_chapelle)
    
    # 13. Compute improvement
    ece_improvement = metrics_baseline['ece'] - metrics_chapelle['ece']
    
    log_message("=" * 60)
    log_message("RESULTS SUMMARY")
    log_message("=" * 60)
    log_message(f"Baseline ECE: {metrics_baseline['ece']:.4f}")
    log_message(f"Chapelle ECE: {metrics_chapelle['ece']:.4f}")
    log_message(f"ECE Improvement: {ece_improvement:.4f}")
    
    if ece_improvement >= 0.02:
        decision = "ADOPT delay correction (ECE improvement >= 0.02)"
        log_message(f"Decision: {decision}", "SUCCESS")
    else:
        decision = "CLOSE DG2 - No delay correction needed (ECE improvement < 0.02)"
        log_message(f"Decision: {decision}", "WARNING")
    
    # 14. Save results
    results = {
        "experiment": "delay_modeling",
        "experiment_id": "EXP-20260108-gift-allocation-05",
        "mvp": "MVP-1.2",
        "timestamp": datetime.now().isoformat(),
        "delay_stats": delay_stats,
        "weibull_params": {
            "shape": float(weibull_shape) if weibull_shape else None,
            "scale": float(weibull_scale) if weibull_scale else None,
        },
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "baseline": metrics_baseline,
        "chapelle": metrics_chapelle,
        "ece_improvement": float(ece_improvement),
        "decision": decision,
        "decision_rule": "ECE improvement >= 0.02 -> adopt delay correction",
        "conclusion": "DG2 CLOSED" if ece_improvement < 0.02 else "Adopt Chapelle",
    }
    
    results_path = RESULTS_DIR / "delay_modeling_20260108.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    log_message(f"Saved results: {results_path}", "SUCCESS")
    
    total_time = time.time() - start_time
    log_message(f"Total experiment time: {total_time:.1f}s", "SUCCESS")
    
    return results


if __name__ == "__main__":
    results = main()
