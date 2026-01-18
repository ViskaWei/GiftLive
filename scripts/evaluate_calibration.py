#!/usr/bin/env python3
"""
Calibration Evaluation - MVP-1.5
================================

Evaluate model calibration:
- ECE (Expected Calibration Error) for P(gift>0)
- ECE for EV regression
- Reliability curves

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
from scipy.special import expit
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_EVpred"
IMG_DIR = OUTPUT_DIR / "exp" / "img"
RESULTS_DIR = OUTPUT_DIR / "results"
MODELS_DIR = OUTPUT_DIR / "models"

for d in [IMG_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10


def log_message(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


def compute_ece_classification(y_true_binary, y_pred_prob, n_bins=10):
    """
    Compute Expected Calibration Error for classification.

    ECE = sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|

    Args:
        y_true_binary: Binary labels (0/1)
        y_pred_prob: Predicted probabilities
        n_bins: Number of bins (using equal-frequency binning)

    Returns:
        ece: Expected Calibration Error
        bin_results: List of per-bin results
    """
    n = len(y_true_binary)

    # Equal-frequency binning
    quantiles = np.percentile(y_pred_prob, np.linspace(0, 100, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    bin_results = []
    ece = 0.0

    for i in range(n_bins):
        mask = (y_pred_prob >= quantiles[i]) & (y_pred_prob < quantiles[i + 1])
        if mask.sum() == 0:
            continue

        bin_size = mask.sum()
        bin_conf = y_pred_prob[mask].mean()  # Average predicted probability
        bin_acc = y_true_binary[mask].mean()  # Actual positive rate
        bin_error = abs(bin_acc - bin_conf)

        ece += (bin_size / n) * bin_error

        bin_results.append({
            'bin_idx': i,
            'bin_range': (float(quantiles[i]) if not np.isinf(quantiles[i]) else None,
                         float(quantiles[i + 1]) if not np.isinf(quantiles[i + 1]) else None),
            'n_samples': int(bin_size),
            'pred_mean': float(bin_conf),
            'actual_rate': float(bin_acc),
            'error': float(bin_error)
        })

    return ece, bin_results


def compute_ece_regression(y_true, y_pred, n_bins=10):
    """
    Compute Expected Calibration Error for regression.

    ECE_reg = sum_b (|B_b|/n) * |mean(y_true) - mean(y_pred)|

    Args:
        y_true: True values
        y_pred: Predicted values
        n_bins: Number of bins (using equal-frequency binning on predictions)

    Returns:
        ece: Expected Calibration Error
        bin_results: List of per-bin results
    """
    n = len(y_true)

    # Equal-frequency binning on predictions
    quantiles = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    bin_results = []
    ece = 0.0

    for i in range(n_bins):
        mask = (y_pred >= quantiles[i]) & (y_pred < quantiles[i + 1])
        if mask.sum() == 0:
            continue

        bin_size = mask.sum()
        bin_pred_mean = y_pred[mask].mean()
        bin_true_mean = y_true[mask].mean()
        bin_error = abs(bin_true_mean - bin_pred_mean)

        ece += (bin_size / n) * bin_error

        bin_results.append({
            'bin_idx': i,
            'bin_range': (float(quantiles[i]) if not np.isinf(quantiles[i]) else None,
                         float(quantiles[i + 1]) if not np.isinf(quantiles[i + 1]) else None),
            'n_samples': int(bin_size),
            'pred_mean': float(bin_pred_mean),
            'actual_mean': float(bin_true_mean),
            'error': float(bin_error)
        })

    return ece, bin_results


def plot_reliability_curve_classification(bin_results, ece, save_path):
    """Plot reliability curve for classification."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pred_means = [r['pred_mean'] for r in bin_results]
    actual_rates = [r['actual_rate'] for r in bin_results]
    errors = [r['error'] for r in bin_results]
    n_samples = [r['n_samples'] for r in bin_results]

    # Plot 1: Reliability curve
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', alpha=0.5)
    ax1.scatter(pred_means, actual_rates, s=[n/1000 for n in n_samples], alpha=0.7,
                c='steelblue', edgecolors='black', linewidth=0.5)
    ax1.plot(pred_means, actual_rates, 'o-', color='steelblue', markersize=8)

    ax1.set_xlabel('Predicted P(gift > 0)')
    ax1.set_ylabel('Actual Gift Rate')
    ax1.set_title(f'Classification Reliability Curve (ECE = {ece:.4f})')
    ax1.set_xlim(-0.02, max(pred_means) * 1.1 + 0.02)
    ax1.set_ylim(-0.02, max(actual_rates) * 1.1 + 0.02)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add annotation
    for i, (pm, ar, ns) in enumerate(zip(pred_means, actual_rates, n_samples)):
        ax1.annotate(f'{ns//1000}K', (pm, ar), textcoords="offset points",
                     xytext=(5, 5), fontsize=7, alpha=0.7)

    # Plot 2: Calibration error by bin
    ax2 = axes[1]
    bin_labels = [f'{i+1}' for i in range(len(bin_results))]
    colors = ['salmon' if e > 0.01 else 'lightgreen' for e in errors]
    ax2.bar(bin_labels, [e * 100 for e in errors], color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='1% threshold')
    ax2.set_xlabel('Bin (sorted by predicted probability)')
    ax2.set_ylabel('Calibration Error (%)')
    ax2.set_title('Error by Bin')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    log_message(f"Saved classification reliability curve to {save_path}")


def plot_reliability_curve_regression(bin_results, ece, save_path):
    """Plot reliability curve for regression."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pred_means = [r['pred_mean'] for r in bin_results]
    actual_means = [r['actual_mean'] for r in bin_results]
    errors = [r['error'] for r in bin_results]
    n_samples = [r['n_samples'] for r in bin_results]

    # Plot 1: Reliability curve
    ax1 = axes[0]
    max_val = max(max(pred_means), max(actual_means))
    ax1.plot([0, max_val], [0, max_val], 'k--', label='Perfect calibration', alpha=0.5)
    ax1.errorbar(pred_means, actual_means, yerr=[0]*len(pred_means),
                 fmt='o-', color='steelblue', markersize=8, capsize=3)

    ax1.set_xlabel('Predicted EV (gift amount)')
    ax1.set_ylabel('Actual Mean Gift Amount')
    ax1.set_title(f'Regression Reliability Curve (ECE = {ece:.4f})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error by bin
    ax2 = axes[1]
    bin_labels = [f'{i+1}' for i in range(len(bin_results))]
    colors = ['salmon' if e > ece else 'lightgreen' for e in errors]
    ax2.bar(bin_labels, errors, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=ece, color='red', linestyle='--', alpha=0.5, label=f'Avg ECE = {ece:.4f}')
    ax2.set_xlabel('Bin (sorted by predicted EV)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error by Bin')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    log_message(f"Saved regression reliability curve to {save_path}")


def plot_calibration_comparison(bin_results_clf, bin_results_reg, ece_clf, ece_reg, save_path):
    """Plot combined calibration comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Classification - reliability
    ax1 = axes[0, 0]
    pred_clf = [r['pred_mean'] for r in bin_results_clf]
    actual_clf = [r['actual_rate'] for r in bin_results_clf]
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.plot(pred_clf, actual_clf, 'o-', color='steelblue', markersize=8)
    ax1.set_xlabel('Predicted P(gift > 0)')
    ax1.set_ylabel('Actual Gift Rate')
    ax1.set_title(f'Classification (ECE = {ece_clf:.4f})')
    ax1.grid(True, alpha=0.3)

    # Classification - error bars
    ax2 = axes[0, 1]
    errors_clf = [r['error'] * 100 for r in bin_results_clf]
    n_samples_clf = [r['n_samples'] for r in bin_results_clf]
    ax2.bar(range(len(bin_results_clf)), errors_clf, color='steelblue', alpha=0.7)
    ax2.set_xlabel('Bin')
    ax2.set_ylabel('Error (%)')
    ax2.set_title('Classification Error by Bin')

    # Regression - reliability
    ax3 = axes[1, 0]
    pred_reg = [r['pred_mean'] for r in bin_results_reg]
    actual_reg = [r['actual_mean'] for r in bin_results_reg]
    max_val = max(max(pred_reg), max(actual_reg))
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    ax3.plot(pred_reg, actual_reg, 'o-', color='green', markersize=8)
    ax3.set_xlabel('Predicted EV')
    ax3.set_ylabel('Actual Mean Gift')
    ax3.set_title(f'Regression (ECE = {ece_reg:.4f})')
    ax3.grid(True, alpha=0.3)

    # Regression - error bars
    ax4 = axes[1, 1]
    errors_reg = [r['error'] for r in bin_results_reg]
    ax4.bar(range(len(bin_results_reg)), errors_reg, color='green', alpha=0.7)
    ax4.set_xlabel('Bin')
    ax4.set_ylabel('Absolute Error')
    ax4.set_title('Regression Error by Bin')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    log_message(f"Saved calibration comparison to {save_path}")


def main():
    log_message("="*60)
    log_message("MVP-1.5: Calibration Evaluation")
    log_message("="*60)

    start_time = time.time()

    # Load model
    model_path = MODELS_DIR / 'direct_frozen_v2_20260118.pkl'
    if not model_path.exists():
        log_message(f"Model not found at {model_path}. Please run train_leakage_free_baseline_v2.py first.", "ERROR")
        return

    log_message("Loading model...")
    model = pickle.load(open(model_path, 'rb'))

    # Load data
    log_message("Loading data...")
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")

    # Prepare click-level data (same as training)
    log_message("Preparing data...")
    click_base = click.copy()
    click_base['timestamp_dt'] = pd.to_datetime(click_base['timestamp'], unit='ms')
    click_base['hour'] = click_base['timestamp_dt'].dt.hour
    click_base['day_of_week'] = click_base['timestamp_dt'].dt.dayofweek
    click_base['is_weekend'] = (click_base['day_of_week'] >= 5).astype(int)

    # V2: Watch time truncation
    max_window_ms = 1 * 3600 * 1000
    click_base['effective_window_ms'] = np.minimum(click_base['watch_live_time'], max_window_ms)
    click_base['click_end_ts'] = click_base['timestamp_dt'] + pd.to_timedelta(click_base['effective_window_ms'], unit='ms')

    gift_ts = gift.copy()
    gift_ts['timestamp_dt'] = pd.to_datetime(gift_ts['timestamp'], unit='ms')

    merged = click_base[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'click_end_ts']].merge(
        gift_ts[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'gift_price']],
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
    }).reset_index().rename(columns={'timestamp_dt_click': 'timestamp_dt', 'gift_price': 'gift_price_label'})

    click_base = click_base.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id', 'timestamp_dt'], how='left')
    click_base['gift_price_label'] = click_base['gift_price_label'].fillna(0)

    # Temporal split
    click_base_sorted = click_base.sort_values('timestamp').reset_index(drop=True)
    train_ratio, val_ratio = 0.70, 0.15
    n = len(click_base_sorted)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = click_base_sorted.iloc[:train_end].copy()
    test_df = click_base_sorted.iloc[val_end:].copy()

    log_message(f"Train: {len(train_df):,}, Test: {len(test_df):,}")

    # Prepare features for test (using train lookups)
    log_message("Preparing features...")

    sys.path.insert(0, str(BASE_DIR / "scripts"))
    from train_leakage_free_baseline_v2 import (
        create_static_features,
        create_past_only_features_frozen,
        apply_frozen_features,
        get_feature_columns
    )

    user_features, streamer_features, room_info = create_static_features(user, streamer, room)

    test_df = test_df.merge(user_features, on='user_id', how='left')
    test_df = test_df.merge(streamer_features, on='streamer_id', how='left')
    test_df = test_df.merge(room_info, on='live_id', how='left')

    lookups = create_past_only_features_frozen(gift, click, train_df)
    test_df = apply_frozen_features(test_df, lookups)

    # Fill NaN and convert categorical
    numeric_cols = test_df.select_dtypes(include=[np.number]).columns
    test_df[numeric_cols] = test_df[numeric_cols].fillna(0)

    cat_columns = ['age', 'gender', 'device_brand', 'device_price', 'fans_num', 'follow_num',
                   'accu_watch_live_cnt', 'accu_watch_live_duration',
                   'streamer_gender', 'streamer_age', 'streamer_device_brand', 'streamer_device_price',
                   'live_operation_tag', 'fans_user_num', 'fans_group_fans_num', 'follow_user_num',
                   'accu_live_cnt', 'accu_live_duration', 'accu_play_cnt', 'accu_play_duration',
                   'live_type', 'live_content_category']

    for col in cat_columns:
        if col in test_df.columns and test_df[col].dtype == 'object':
            test_df[col] = test_df[col].fillna('unknown')
            test_df[col] = test_df[col].astype('category').cat.codes

    test_df['target'] = np.log1p(test_df['gift_price_label'])
    test_df['target_raw'] = test_df['gift_price_label']
    test_df['is_gift'] = (test_df['gift_price_label'] > 0).astype(int)

    # Get predictions
    feature_cols = get_feature_columns(test_df)
    X_test = test_df[feature_cols]
    y_pred_log = model.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)

    # Clip negative predictions
    y_pred_raw = np.maximum(y_pred_raw, 0)

    y_true = test_df['target_raw'].values
    y_true_binary = test_df['is_gift'].values

    log_message(f"Test samples: {len(test_df):,}")
    log_message(f"Gift rate: {y_true_binary.mean()*100:.3f}%")

    # === Classification Calibration ===
    log_message("\n" + "="*60)
    log_message("CLASSIFICATION CALIBRATION (P(gift > 0))")
    log_message("="*60)

    # Convert regression output to probability proxy
    # Use sigmoid transformation on log predictions
    # Higher predicted log(1+gift) -> higher probability of gift
    y_pred_prob = expit(y_pred_log * 2 - 0.5)  # Rough scaling

    # Alternative: use empirical calibration
    # Bin predictions and compute actual gift rates
    pred_percentiles = pd.qcut(y_pred_log, q=100, labels=False, duplicates='drop')
    calibration_df = pd.DataFrame({
        'pred_pct': pred_percentiles,
        'is_gift': y_true_binary
    })
    pct_to_prob = calibration_df.groupby('pred_pct')['is_gift'].mean().to_dict()
    y_pred_prob_calibrated = np.array([pct_to_prob.get(p, y_true_binary.mean()) for p in pred_percentiles])

    ece_clf, bin_results_clf = compute_ece_classification(y_true_binary, y_pred_prob_calibrated, n_bins=10)

    log_message(f"ECE (classification): {ece_clf:.4f}")
    log_message(f"Actual gift rate: {y_true_binary.mean()*100:.3f}%")
    log_message(f"Mean predicted prob: {y_pred_prob_calibrated.mean()*100:.3f}%")

    log_message("\nPer-bin results:")
    for r in bin_results_clf:
        log_message(f"  Bin {r['bin_idx']+1}: pred={r['pred_mean']*100:.2f}%, actual={r['actual_rate']*100:.2f}%, error={r['error']*100:.2f}%")

    plot_reliability_curve_classification(bin_results_clf, ece_clf,
                                          IMG_DIR / 'reliability_curve_classification.png')

    # === Regression Calibration ===
    log_message("\n" + "="*60)
    log_message("REGRESSION CALIBRATION (EV)")
    log_message("="*60)

    ece_reg, bin_results_reg = compute_ece_regression(y_true, y_pred_raw, n_bins=10)

    log_message(f"ECE (regression): {ece_reg:.4f}")
    log_message(f"Actual mean gift: {y_true.mean():.4f}")
    log_message(f"Predicted mean gift: {y_pred_raw.mean():.4f}")

    log_message("\nPer-bin results:")
    for r in bin_results_reg:
        log_message(f"  Bin {r['bin_idx']+1}: pred={r['pred_mean']:.4f}, actual={r['actual_mean']:.4f}, error={r['error']:.4f}")

    plot_reliability_curve_regression(bin_results_reg, ece_reg,
                                      IMG_DIR / 'reliability_curve_regression.png')

    # === Combined plot ===
    plot_calibration_comparison(bin_results_clf, bin_results_reg, ece_clf, ece_reg,
                                IMG_DIR / 'calibration_comparison.png')

    # === Conditional EV Calibration (gift > 0 only) ===
    log_message("\n" + "="*60)
    log_message("CONDITIONAL EV CALIBRATION (gift > 0 samples only)")
    log_message("="*60)

    gift_mask = y_true > 0
    if gift_mask.sum() > 100:
        ece_cond, bin_results_cond = compute_ece_regression(
            y_true[gift_mask], y_pred_raw[gift_mask], n_bins=10
        )
        log_message(f"Conditional ECE: {ece_cond:.4f}")
        log_message(f"Samples with gift: {gift_mask.sum():,}")
    else:
        ece_cond = None
        bin_results_cond = []
        log_message("Not enough gift samples for conditional calibration")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'model': 'direct_frozen_v2',
        'n_test_samples': len(test_df),
        'gift_rate': float(y_true_binary.mean()),
        'classification_calibration': {
            'ece': float(ece_clf),
            'interpretation': 'good' if ece_clf < 0.03 else ('moderate' if ece_clf < 0.05 else 'needs_improvement'),
            'bins': bin_results_clf
        },
        'regression_calibration': {
            'ece': float(ece_reg),
            'actual_mean': float(y_true.mean()),
            'pred_mean': float(y_pred_raw.mean()),
            'bins': bin_results_reg
        },
        'conditional_calibration': {
            'ece': float(ece_cond) if ece_cond else None,
            'n_samples': int(gift_mask.sum()),
            'bins': bin_results_cond
        } if ece_cond else None
    }

    results_path = RESULTS_DIR / 'calibration_evaluation_20260118.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - start_time

    log_message("\n" + "="*60)
    log_message("SUMMARY")
    log_message("="*60)
    log_message(f"Classification ECE: {ece_clf:.4f} ({'GOOD' if ece_clf < 0.03 else 'MODERATE' if ece_clf < 0.05 else 'NEEDS IMPROVEMENT'})")
    log_message(f"Regression ECE: {ece_reg:.4f}")
    if ece_cond:
        log_message(f"Conditional ECE (gift>0): {ece_cond:.4f}")

    # Verdict
    if ece_clf < 0.03:
        log_message("\n✅ Classification is well-calibrated. Predictions can be used for allocation.", "SUCCESS")
    elif ece_clf < 0.05:
        log_message("\n⚠️ Classification calibration is moderate. Consider monitoring.", "WARNING")
    else:
        log_message("\n❌ Classification needs calibration layer (Platt scaling / isotonic regression).", "ERROR")

    log_message(f"\nTotal time: {elapsed:.1f}s")
    log_message(f"Results saved to: {results_path}")
    log_message("="*60, "SUCCESS")


if __name__ == '__main__':
    main()
