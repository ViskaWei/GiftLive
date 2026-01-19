#!/usr/bin/env python3
"""
Metrics Landing Experiment V2: 完整三层指标体系
===============================================

补充指标：
- Whale Recall@K（whale 召回率）
- Precision@K（whale 精确率）
- Avg Revenue per Selected@K（人均真实收入）
- 稳定性：按天统计 + Bootstrap CI

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
from scipy import stats

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from gift_EVpred.data_utils import prepare_dataset, get_feature_columns

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = Path('/home/swei20/GiftLive/gift_EVpred')
IMG_DIR = OUTPUT_DIR / 'img'
RESULTS_DIR = OUTPUT_DIR / 'results'

IMG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# K values for metrics
K_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
K_LABELS = ['0.1%', '0.5%', '1%', '2%', '5%', '10%']

# Whale threshold: P90 of gifters (y > 0)
WHALE_PERCENTILE = 90  # P90 of gifters


# =============================================================================
# Metric Functions
# =============================================================================
def revenue_capture_at_k(y_true, y_pred, k=0.01):
    """RevCap@K: fraction of total revenue captured by top K%."""
    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_pred)[-n_top:]
    total_revenue = y_true.sum()
    if total_revenue == 0:
        return 0.0
    return y_true[top_indices].sum() / total_revenue


def whale_recall_at_k(y_true, y_pred, whale_threshold, k=0.01):
    """
    Whale Recall@K: 在 top K% 预测中，捕获了多少比例的真实 whale。

    whale_recall = |top_k ∩ whales| / |whales|
    """
    n_top = max(1, int(len(y_true) * k))
    top_indices = set(np.argsort(y_pred)[-n_top:])

    whale_indices = set(np.where(y_true >= whale_threshold)[0])
    if len(whale_indices) == 0:
        return np.nan

    captured_whales = len(top_indices & whale_indices)
    return captured_whales / len(whale_indices)


def whale_precision_at_k(y_true, y_pred, whale_threshold, k=0.01):
    """
    Whale Precision@K: top K% 预测中，有多少比例是真实 whale。

    whale_precision = |top_k ∩ whales| / |top_k|
    """
    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_pred)[-n_top:]

    whales_in_top_k = (y_true[top_indices] >= whale_threshold).sum()
    return whales_in_top_k / n_top


def avg_revenue_per_selected_at_k(y_true, y_pred, k=0.01):
    """
    Avg Revenue per Selected@K: top K% 样本的人均真实收入。

    帮助评估"池子质量"
    """
    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_pred)[-n_top:]
    return y_true[top_indices].mean()


def compute_all_metrics_at_k(y_true, y_pred, whale_threshold, k):
    """计算单个 K 值下的所有指标"""
    return {
        'revcap': revenue_capture_at_k(y_true, y_pred, k),
        'whale_recall': whale_recall_at_k(y_true, y_pred, whale_threshold, k),
        'whale_precision': whale_precision_at_k(y_true, y_pred, whale_threshold, k),
        'avg_revenue': avg_revenue_per_selected_at_k(y_true, y_pred, k),
    }


def compute_metrics_by_day(test_df, y_true, y_pred, whale_threshold, k=0.01):
    """按天计算所有指标，评估稳定性"""
    test_df = test_df.copy()
    test_df['_date'] = pd.to_datetime(test_df['timestamp'], unit='ms').dt.date

    days = sorted(test_df['_date'].unique())
    metrics_by_day = []

    for day in days:
        mask = (test_df['_date'] == day).values
        if mask.sum() < 100:  # 跳过样本太少的天
            continue

        y_day = y_true[mask]
        pred_day = y_pred[mask]

        if y_day.sum() == 0:  # 跳过无打赏的天
            continue

        metrics = compute_all_metrics_at_k(y_day, pred_day, whale_threshold, k)
        metrics['day'] = day
        metrics['n_samples'] = mask.sum()
        metrics['n_whales'] = (y_day >= whale_threshold).sum()
        metrics['total_revenue'] = y_day.sum()
        metrics_by_day.append(metrics)

    return pd.DataFrame(metrics_by_day)


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95):
    """Bootstrap 计算置信区间"""
    values = np.array([v for v in values if not np.isnan(v)])
    if len(values) < 2:
        return np.nan, np.nan, np.nan

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return np.mean(values), lower, upper


# =============================================================================
# Plotting Functions
# =============================================================================
def plot_comprehensive_metrics(metrics_df, k_labels, save_path):
    """绘制完整指标曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    x = np.arange(len(k_labels))

    # RevCap@K
    ax = axes[0, 0]
    ax.plot(x, metrics_df['revcap'], 'b-o', linewidth=2, markersize=8, label='Model')
    ax.plot(x, metrics_df['oracle_revcap'], 'g--^', linewidth=2, markersize=8, label='Oracle')
    ax.set_xlabel('Top K%', fontsize=11)
    ax.set_ylabel('Revenue Capture', fontsize=11)
    ax.set_title('Revenue Capture@K', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    for i, v in enumerate(metrics_df['revcap']):
        ax.annotate(f'{v:.1%}', (x[i], v), textcoords='offset points',
                   xytext=(0, 8), ha='center', fontsize=9)

    # Whale Recall@K
    ax = axes[0, 1]
    ax.plot(x, metrics_df['whale_recall'], 'r-o', linewidth=2, markersize=8)
    ax.set_xlabel('Top K%', fontsize=11)
    ax.set_ylabel('Whale Recall', fontsize=11)
    ax.set_title(f'Whale Recall@K (whale >= P{WHALE_PERCENTILE})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(metrics_df['whale_recall']):
        ax.annotate(f'{v:.1%}', (x[i], v), textcoords='offset points',
                   xytext=(0, 8), ha='center', fontsize=9)

    # Whale Precision@K
    ax = axes[1, 0]
    ax.plot(x, metrics_df['whale_precision'], 'm-o', linewidth=2, markersize=8)
    ax.set_xlabel('Top K%', fontsize=11)
    ax.set_ylabel('Whale Precision', fontsize=11)
    ax.set_title(f'Whale Precision@K (whale >= P{WHALE_PERCENTILE})', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.set_ylim(0, max(metrics_df['whale_precision']) * 1.2)
    ax.grid(True, alpha=0.3)
    for i, v in enumerate(metrics_df['whale_precision']):
        ax.annotate(f'{v:.1%}', (x[i], v), textcoords='offset points',
                   xytext=(0, 8), ha='center', fontsize=9)

    # Avg Revenue per Selected@K
    ax = axes[1, 1]
    ax.bar(x, metrics_df['avg_revenue'], color='teal', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Top K%', fontsize=11)
    ax.set_ylabel('Avg Revenue (yuan)', fontsize=11)
    ax.set_title('Avg Revenue per Selected@K', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(k_labels)
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(metrics_df['avg_revenue']):
        ax.annotate(f'{v:.1f}', (x[i], v), textcoords='offset points',
                   xytext=(0, 5), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_stability_analysis(daily_df, save_path):
    """绘制稳定性分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    days = range(1, len(daily_df) + 1)

    metrics_to_plot = [
        ('revcap', 'RevCap@1%', 'steelblue'),
        ('whale_recall', 'Whale Recall@1%', 'coral'),
        ('whale_precision', 'Whale Precision@1%', 'purple'),
        ('total_revenue', 'Daily Total Revenue (yuan)', 'green'),
    ]

    for ax, (metric, title, color) in zip(axes.flat, metrics_to_plot):
        values = daily_df[metric].values
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        cv = std_val / mean_val if mean_val > 0 else np.nan

        # Bootstrap CI
        _, ci_lower, ci_upper = bootstrap_ci(values)

        ax.bar(days, values, color=color, alpha=0.7, edgecolor='black')
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.2f}')

        if not np.isnan(ci_lower):
            ax.fill_between([0.5, len(days)+0.5], ci_lower, ci_upper,
                           alpha=0.2, color='red', label=f'95% CI')

        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f'{title}\nMean={mean_val:.3f}, Std={std_val:.3f}, CV={cv:.1%}', fontsize=11)
        ax.set_xticks(days)
        ax.set_xticklabels([f'D{d}' for d in days])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_daily_revenue_distribution(test_df, y_true, save_path):
    """绘制每日收入分布，识别异常大额"""
    test_df = test_df.copy()
    test_df['_date'] = pd.to_datetime(test_df['timestamp'], unit='ms').dt.date
    test_df['y_true'] = y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 每日总收入
    ax = axes[0]
    daily_revenue = test_df.groupby('_date')['y_true'].sum()
    days = range(1, len(daily_revenue) + 1)
    ax.bar(days, daily_revenue.values, color='steelblue', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Total Revenue (yuan)', fontsize=11)
    ax.set_title('Daily Total Revenue', fontsize=12)
    ax.set_xticks(days)
    ax.set_xticklabels([f'D{d}' for d in days])
    ax.grid(True, alpha=0.3, axis='y')

    # 标注最大单笔交易
    for i, (day, rev) in enumerate(zip(days, daily_revenue.values)):
        ax.annotate(f'{rev:,.0f}', (day, rev), textcoords='offset points',
                   xytext=(0, 5), ha='center', fontsize=9)

    # 每日最大单笔交易
    ax = axes[1]
    daily_max = test_df.groupby('_date')['y_true'].max()
    ax.bar(days, daily_max.values, color='coral', alpha=0.8, edgecolor='black')
    ax.set_xlabel('Day', fontsize=11)
    ax.set_ylabel('Max Single Gift (yuan)', fontsize=11)
    ax.set_title('Daily Max Single Gift (detect outliers)', fontsize=12)
    ax.set_xticks(days)
    ax.set_xticklabels([f'D{d}' for d in days])
    ax.grid(True, alpha=0.3, axis='y')

    for i, (day, max_gift) in enumerate(zip(days, daily_max.values)):
        ax.annotate(f'{max_gift:,.0f}', (day, max_gift), textcoords='offset points',
                   xytext=(0, 5), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 60)
    print("Metrics Landing V2: Complete Framework")
    print("=" * 60)

    # Load data
    print("\n[1] Loading data...")
    train_df, val_df, test_df = prepare_dataset()
    feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols].values
    y_train = train_df['target_raw'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target_raw'].values

    # Scale features
    print("\n[2] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    print("\n[3] Training Ridge Regression...")
    model = Ridge(alpha=1.0, random_state=SEED)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Define whale threshold: P90 of gifters
    gifters_mask = y_test > 0
    whale_threshold = np.percentile(y_test[gifters_mask], WHALE_PERCENTILE)
    n_whales = (y_test >= whale_threshold).sum()
    print(f"\n[4] Whale definition: y >= {whale_threshold:.0f} yuan (P{WHALE_PERCENTILE} of gifters)")
    print(f"    Total whales: {n_whales:,} ({n_whales/len(y_test)*100:.3f}% of all samples)")

    # Compute metrics for all K values
    print("\n[5] Computing metrics for all K values...")
    metrics_list = []
    for k, kl in zip(K_VALUES, K_LABELS):
        metrics = compute_all_metrics_at_k(y_test, y_pred, whale_threshold, k)
        metrics['k'] = k
        metrics['k_label'] = kl
        # Oracle RevCap
        metrics['oracle_revcap'] = revenue_capture_at_k(y_test, y_test, k)
        metrics['normalized_revcap'] = metrics['revcap'] / metrics['oracle_revcap'] if metrics['oracle_revcap'] > 0 else 0
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    print("\nMetrics Summary:")
    print("-" * 80)
    print(f"{'K':<8} {'RevCap':<10} {'Norm':<10} {'Recall':<10} {'Precision':<10} {'AvgRev':<10}")
    print("-" * 80)
    for _, row in metrics_df.iterrows():
        print(f"{row['k_label']:<8} {row['revcap']:<10.1%} {row['normalized_revcap']:<10.1%} "
              f"{row['whale_recall']:<10.1%} {row['whale_precision']:<10.1%} {row['avg_revenue']:<10.1f}")

    # Compute daily stability
    print("\n[6] Computing daily stability...")
    daily_df = compute_metrics_by_day(test_df, y_test, y_pred, whale_threshold, k=0.01)

    print("\nDaily Metrics (K=1%):")
    print("-" * 100)
    print(f"{'Day':<12} {'RevCap':<10} {'Recall':<10} {'Precision':<12} {'TotalRev':<12} {'Whales':<8}")
    print("-" * 100)
    for i, row in daily_df.iterrows():
        print(f"{str(row['day']):<12} {row['revcap']:<10.1%} {row['whale_recall']:<10.1%} "
              f"{row['whale_precision']:<12.1%} {row['total_revenue']:<12,.0f} {row['n_whales']:<8}")

    # Stability summary
    print("\n[7] Stability Summary (RevCap@1%):")
    revcap_values = daily_df['revcap'].values
    mean_val, ci_lower, ci_upper = bootstrap_ci(revcap_values)
    std_val = np.std(revcap_values)
    cv = std_val / mean_val
    print(f"    Mean: {mean_val:.1%}")
    print(f"    Std:  {std_val:.1%}")
    print(f"    CV:   {cv:.1%}")
    print(f"    95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]")
    print(f"    Min Day: {daily_df.loc[daily_df['revcap'].idxmin(), 'day']} ({daily_df['revcap'].min():.1%})")
    print(f"    Max Day: {daily_df.loc[daily_df['revcap'].idxmax(), 'day']} ({daily_df['revcap'].max():.1%})")

    # Check for outlier days
    print("\n[8] Daily Revenue Distribution (detecting outliers):")
    test_df_copy = test_df.copy()
    test_df_copy['_date'] = pd.to_datetime(test_df_copy['timestamp'], unit='ms').dt.date
    test_df_copy['y_true'] = y_test

    daily_max_gift = test_df_copy.groupby('_date')['y_true'].max()
    daily_total = test_df_copy.groupby('_date')['y_true'].sum()

    print(f"    Daily max single gift range: {daily_max_gift.min():.0f} ~ {daily_max_gift.max():.0f} yuan")
    print(f"    Daily total revenue range: {daily_total.min():,.0f} ~ {daily_total.max():,.0f} yuan")

    # Generate plots
    print("\n[9] Generating plots...")
    plot_comprehensive_metrics(metrics_df, K_LABELS, IMG_DIR / 'metrics_comprehensive.png')
    plot_stability_analysis(daily_df, IMG_DIR / 'stability_analysis.png')
    plot_daily_revenue_distribution(test_df, y_test, IMG_DIR / 'daily_revenue_dist.png')

    # Save results
    print("\n[10] Saving results...")
    results = {
        'experiment_id': 'EXP-20260119-EVpred-01-v2',
        'whale_threshold': float(whale_threshold),
        'whale_percentile': WHALE_PERCENTILE,
        'n_whales': int(n_whales),
        'whale_rate': float(n_whales / len(y_test)),
        'metrics_by_k': metrics_df.to_dict('records'),
        'stability': {
            'daily_revcap': daily_df['revcap'].tolist(),
            'daily_whale_recall': daily_df['whale_recall'].tolist(),
            'daily_whale_precision': daily_df['whale_precision'].tolist(),
            'daily_total_revenue': daily_df['total_revenue'].tolist(),
            'revcap_mean': float(mean_val),
            'revcap_std': float(std_val),
            'revcap_cv': float(cv),
            'revcap_ci_lower': float(ci_lower),
            'revcap_ci_upper': float(ci_upper),
        },
        'daily_stats': {
            'max_single_gift_range': [float(daily_max_gift.min()), float(daily_max_gift.max())],
            'total_revenue_range': [float(daily_total.min()), float(daily_total.max())],
        }
    }

    with open(RESULTS_DIR / 'metrics_landing_v2_20260119.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)

    return results


if __name__ == '__main__':
    results = main()
