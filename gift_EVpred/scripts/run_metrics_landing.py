#!/usr/bin/env python3
"""
Metrics Landing Experiment: 三层指标体系落地
============================================

EXP-20260119-EVpred-01 / MVP-3.1

Computes:
- RevCap@K curve (K ∈ {0.01%, 0.1%, 0.5%, 1%, 2%, 5%, 10%})
- RevCap@1% stability by day (7 days)
- Tail calibration (Sum/Mean for top buckets)
- Normalized RevCap@K (vs Oracle)

Author: Viska Wei
Date: 2026-01-19
"""

import sys
import os
sys.path.insert(0, '/home/swei20/GiftLive')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Import data utilities
from gift_EVpred.data_utils import prepare_dataset, get_feature_columns

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path('/home/swei20/GiftLive/gift_EVpred')
IMG_DIR = OUTPUT_DIR / 'img'
RESULTS_DIR = OUTPUT_DIR / 'results'
LOGS_DIR = OUTPUT_DIR / 'logs'

IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# K values for RevCap curve
K_VALUES = [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
K_LABELS = ['0.01%', '0.1%', '0.5%', '1%', '2%', '5%', '10%']

# Buckets for tail calibration
CALIBRATION_BUCKETS = [0.001, 0.005, 0.01, 0.05]
BUCKET_LABELS = ['Top 0.1%', 'Top 0.5%', 'Top 1%', 'Top 5%']


# =============================================================================
# Metric Functions
# =============================================================================
def revenue_capture_at_k(y_true, y_pred, k=0.01):
    """
    Compute Revenue Capture at top K%.

    Args:
        y_true: true labels (raw amounts)
        y_pred: predicted scores (for ranking)
        k: fraction of samples to consider as "top" (0.01 = 1%)

    Returns:
        float: fraction of total revenue captured by top K% predictions
    """
    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_pred)[-n_top:]
    total_revenue = y_true.sum()
    if total_revenue == 0:
        return 0.0
    return y_true[top_indices].sum() / total_revenue


def compute_revcap_curve(y_true, y_pred, k_values):
    """Compute RevCap for multiple K values."""
    return [revenue_capture_at_k(y_true, y_pred, k) for k in k_values]


def compute_tail_calibration(y_true, y_pred, buckets):
    """
    Compute tail calibration for top buckets.

    Returns:
        dict: {bucket: {'sum_ratio': float, 'mean_ratio': float}}
    """
    results = {}
    for k in buckets:
        n_top = max(1, int(len(y_true) * k))
        top_indices = np.argsort(y_pred)[-n_top:]

        pred_sum = y_pred[top_indices].sum()
        true_sum = y_true[top_indices].sum()
        pred_mean = y_pred[top_indices].mean()
        true_mean = y_true[top_indices].mean()

        results[k] = {
            'sum_ratio': pred_sum / true_sum if true_sum > 0 else np.nan,
            'mean_ratio': pred_mean / true_mean if true_mean > 0 else np.nan,
        }
    return results


def compute_revcap_by_day(test_df, y_true, y_pred, k=0.01):
    """
    Compute RevCap@K for each day in test set.

    Returns:
        dict: {day_idx: revcap_value}
    """
    test_df = test_df.copy()
    test_df['_date'] = pd.to_datetime(test_df['timestamp'], unit='ms').dt.date

    days = sorted(test_df['_date'].unique())
    revcap_by_day = {}

    for i, day in enumerate(days):
        mask = (test_df['_date'] == day).values
        if mask.sum() > 0 and y_true[mask].sum() > 0:
            revcap_by_day[i+1] = revenue_capture_at_k(y_true[mask], y_pred[mask], k)
        else:
            revcap_by_day[i+1] = np.nan

    return revcap_by_day


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_revcap_curve(model_revcap, oracle_revcap, random_revcap, k_labels, save_path):
    """
    Plot RevCap curve: Model vs Oracle vs Random.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(k_labels))

    ax.plot(x, model_revcap, 'b-o', linewidth=2, markersize=8, label='Model (Ridge)')
    ax.plot(x, oracle_revcap, 'g--^', linewidth=2, markersize=8, label='Oracle')
    ax.plot(x, random_revcap, 'r:s', linewidth=2, markersize=6, label='Random')

    ax.set_xlabel('Top K%', fontsize=12)
    ax.set_ylabel('Revenue Capture', fontsize=12)
    ax.set_title('Revenue Capture Curve: Model vs Oracle vs Random', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    # Add value annotations
    for i, (m, o, r) in enumerate(zip(model_revcap, oracle_revcap, random_revcap)):
        ax.annotate(f'{m:.1%}', (x[i], m), textcoords='offset points',
                   xytext=(0, 8), ha='center', fontsize=8, color='blue')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_revcap_stability(revcap_by_day, save_path):
    """
    Plot RevCap@1% by day with mean and std lines.
    """
    days = list(revcap_by_day.keys())
    values = list(revcap_by_day.values())

    mean_val = np.nanmean(values)
    std_val = np.nanstd(values)
    cv = std_val / mean_val if mean_val > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.bar(days, values, color='steelblue', alpha=0.8, edgecolor='black')
    ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.1%}')
    ax.axhline(y=mean_val + std_val, color='orange', linestyle=':', linewidth=1.5, label=f'+1 Std: {mean_val+std_val:.1%}')
    ax.axhline(y=mean_val - std_val, color='orange', linestyle=':', linewidth=1.5, label=f'-1 Std: {mean_val-std_val:.1%}')
    ax.fill_between(days, mean_val - std_val, mean_val + std_val, alpha=0.2, color='orange')

    ax.set_xlabel('Test Day', fontsize=12)
    ax.set_ylabel('RevCap@1%', fontsize=12)
    ax.set_title(f'RevCap@1% Stability by Day\nMean={mean_val:.1%}, Std={std_val:.1%}, CV={cv:.1%}', fontsize=14)
    ax.set_xticks(days)
    ax.set_xticklabels([f'Day {d}' for d in days])
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(loc='upper right', fontsize=9)

    # Add value annotations on bars
    for d, v in zip(days, values):
        ax.annotate(f'{v:.1%}', (d, v), textcoords='offset points',
                   xytext=(0, 5), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    return mean_val, std_val, cv


def plot_tail_calibration(calibration_results, bucket_labels, save_path):
    """
    Plot tail calibration heatmap.
    """
    # Prepare data for heatmap
    buckets = list(calibration_results.keys())
    data = []
    for k in buckets:
        data.append([
            calibration_results[k]['sum_ratio'],
            calibration_results[k]['mean_ratio']
        ])

    data = np.array(data).T  # rows: Sum/Mean, cols: buckets

    fig, ax = plt.subplots(figsize=(6, 5))

    # Custom colormap: green near 1.0, red when deviating
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    sns.heatmap(data, annot=True, fmt='.3f', cmap=cmap, center=1.0,
                xticklabels=bucket_labels, yticklabels=['Sum Ratio', 'Mean Ratio'],
                ax=ax, vmin=0.5, vmax=1.5,
                cbar_kws={'label': 'Calibration Ratio (pred/actual)'})

    ax.set_xlabel('Bucket', fontsize=12)
    ax.set_ylabel('Calibration Type', fontsize=12)
    ax.set_title('Tail Calibration: Pred/Actual Ratio\n(Green=1.0, Red=Deviated)', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main Experiment
# =============================================================================
def main():
    print("=" * 60)
    print("Metrics Landing Experiment")
    print("EXP-20260119-EVpred-01 / MVP-3.1")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load data
    # =========================================================================
    print("\n[Step 1] Loading data...")
    train_df, val_df, test_df = prepare_dataset()
    feature_cols = get_feature_columns(train_df)

    print(f"Features: {len(feature_cols)}")
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")

    # Prepare arrays
    X_train = train_df[feature_cols].values
    y_train = train_df['target_raw'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target_raw'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target_raw'].values

    # =========================================================================
    # Step 2: Standardize features
    # =========================================================================
    print("\n[Step 2] Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # =========================================================================
    # Step 3: Train Ridge Regression
    # =========================================================================
    print("\n[Step 3] Training Ridge Regression (alpha=1.0)...")
    model = Ridge(alpha=1.0, solver='auto', random_state=SEED)
    model.fit(X_train_scaled, y_train)

    y_pred_test = model.predict(X_test_scaled)
    print(f"Prediction range: [{y_pred_test.min():.2f}, {y_pred_test.max():.2f}]")

    # =========================================================================
    # Step 4: Compute RevCap curve
    # =========================================================================
    print("\n[Step 4] Computing RevCap curve...")
    model_revcap = compute_revcap_curve(y_test, y_pred_test, K_VALUES)
    oracle_revcap = compute_revcap_curve(y_test, y_test, K_VALUES)  # Oracle: sort by true y
    random_revcap = K_VALUES.copy()  # Random: RevCap@K = K

    print("RevCap@K:")
    for k, kl, mr, orc, rnd in zip(K_VALUES, K_LABELS, model_revcap, oracle_revcap, random_revcap):
        print(f"  @{kl}: Model={mr:.1%}, Oracle={orc:.1%}, Random={rnd:.1%}, Norm={mr/orc:.1%}")

    # Normalized RevCap
    normalized_revcap = [m/o if o > 0 else 0 for m, o in zip(model_revcap, oracle_revcap)]

    # =========================================================================
    # Step 5: Compute RevCap@1% by day (stability)
    # =========================================================================
    print("\n[Step 5] Computing RevCap@1% stability by day...")
    revcap_by_day = compute_revcap_by_day(test_df, y_test, y_pred_test, k=0.01)

    print("RevCap@1% by day:")
    for day, val in revcap_by_day.items():
        print(f"  Day {day}: {val:.1%}")

    mean_revcap = np.nanmean(list(revcap_by_day.values()))
    std_revcap = np.nanstd(list(revcap_by_day.values()))
    cv_revcap = std_revcap / mean_revcap if mean_revcap > 0 else np.nan

    print(f"\nStability: Mean={mean_revcap:.1%}, Std={std_revcap:.1%}, CV={cv_revcap:.1%}")

    # =========================================================================
    # Step 6: Compute Tail Calibration
    # =========================================================================
    print("\n[Step 6] Computing Tail Calibration...")
    calibration_results = compute_tail_calibration(y_test, y_pred_test, CALIBRATION_BUCKETS)

    print("Tail Calibration (Pred/Actual):")
    for k, bl in zip(CALIBRATION_BUCKETS, BUCKET_LABELS):
        res = calibration_results[k]
        print(f"  {bl}: Sum={res['sum_ratio']:.3f}, Mean={res['mean_ratio']:.3f}")

    # =========================================================================
    # Step 7: Generate plots
    # =========================================================================
    print("\n[Step 7] Generating plots...")

    # Fig 1: RevCap curve
    plot_revcap_curve(
        model_revcap, oracle_revcap, random_revcap, K_LABELS,
        IMG_DIR / 'revcap_curve.png'
    )

    # Fig 2: Stability
    mean_val, std_val, cv_val = plot_revcap_stability(
        revcap_by_day,
        IMG_DIR / 'revcap_stability.png'
    )

    # Fig 3: Tail calibration
    plot_tail_calibration(
        calibration_results, BUCKET_LABELS,
        IMG_DIR / 'tail_calibration.png'
    )

    # =========================================================================
    # Step 8: Save results
    # =========================================================================
    print("\n[Step 8] Saving results...")

    results = {
        'experiment_id': 'EXP-20260119-EVpred-01',
        'mvp': 'MVP-3.1',
        'date': '2026-01-19',
        'model': {
            'name': 'Ridge Regression',
            'alpha': 1.0,
            'target': 'raw Y (target_raw)',
            'scaling': 'StandardScaler'
        },
        'data': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'features': len(feature_cols),
            'split': '7-7-7 by days'
        },
        'revcap_curve': {
            k_label: {
                'model': round(m, 4),
                'oracle': round(o, 4),
                'random': round(r, 4),
                'normalized': round(n, 4)
            }
            for k_label, m, o, r, n in zip(K_LABELS, model_revcap, oracle_revcap, random_revcap, normalized_revcap)
        },
        'stability': {
            'revcap_by_day': {f'day_{d}': round(v, 4) for d, v in revcap_by_day.items()},
            'mean': round(mean_revcap, 4),
            'std': round(std_revcap, 4),
            'cv': round(cv_revcap, 4)
        },
        'tail_calibration': {
            bl: {
                'sum_ratio': round(calibration_results[k]['sum_ratio'], 4),
                'mean_ratio': round(calibration_results[k]['mean_ratio'], 4)
            }
            for k, bl in zip(CALIBRATION_BUCKETS, BUCKET_LABELS)
        }
    }

    results_file = RESULTS_DIR / 'metrics_landing_20260119.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {results_file}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    print(f"\nKey Metrics:")
    print(f"  RevCap@1%: {model_revcap[3]:.1%} (Model) vs {oracle_revcap[3]:.1%} (Oracle)")
    print(f"  Normalized RevCap@1%: {normalized_revcap[3]:.1%}")
    print(f"  Stability CV: {cv_revcap:.1%} (Target: < 10%)")
    print(f"\nOutput files:")
    print(f"  {IMG_DIR / 'revcap_curve.png'}")
    print(f"  {IMG_DIR / 'revcap_stability.png'}")
    print(f"  {IMG_DIR / 'tail_calibration.png'}")
    print(f"  {results_file}")

    return results


if __name__ == '__main__':
    results = main()
