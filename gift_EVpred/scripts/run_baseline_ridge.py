#!/usr/bin/env python3
"""
Baseline Ridge Regression Experiment
=====================================

实验目的：建立并记录 gift_EVpred 的标准 baseline
- 模型：Ridge Regression (alpha=1.0)
- 目标：Raw Y (gift_usd_sum)
- 数据划分：7-7-7 天 (Train/Val/Test)
- 特征：Day-Frozen 特征（无泄漏）

输出：
- 模型文件：models/baseline_ridge_v1.pkl
- 特征文件：features_cache/baseline_features_v1.pkl
- 评估结果：results/baseline_ridge_v1_results.json
- 图表：img/baseline_*.png

Author: Viska Wei
Date: 2026-01-19
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import Ridge

# 添加项目路径
sys.path.insert(0, '/home/swei20/GiftLive')

from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
from gift_EVpred.metrics import evaluate_model, format_metrics_table, EvalResult

# =============================================================================
# 配置
# =============================================================================

CONFIG = {
    'model_name': 'baseline_ridge_v1',
    'model_type': 'Ridge',
    'alpha': 1.0,
    'train_days': 7,
    'val_days': 7,
    'test_days': 7,
    'target_col': 'target_raw',
    'whale_threshold': 100,  # P90 of gifters
    'random_state': 42,
}

# 路径配置
BASE_DIR = Path('/home/swei20/GiftLive/gift_EVpred')
MODELS_DIR = BASE_DIR / 'models'
FEATURES_DIR = BASE_DIR / 'features_cache'
RESULTS_DIR = BASE_DIR / 'results'
IMG_DIR = BASE_DIR / 'img'

# 创建目录
for d in [MODELS_DIR, FEATURES_DIR, RESULTS_DIR, IMG_DIR]:
    d.mkdir(exist_ok=True)


def log(msg):
    """打印带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# =============================================================================
# 数据加载
# =============================================================================

def load_data():
    """加载并准备数据"""
    log("Loading data with prepare_dataset...")
    train_df, val_df, test_df = prepare_dataset(
        train_days=CONFIG['train_days'],
        val_days=CONFIG['val_days'],
        test_days=CONFIG['test_days']
    )

    feature_cols = get_feature_columns(train_df)
    target_col = CONFIG['target_col']

    log(f"Feature columns: {len(feature_cols)}")
    log(f"Target column: {target_col}")

    # 准备特征和标签
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    # 数据统计
    stats = {
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'n_features': len(feature_cols),
        'train_gift_rate': (y_train > 0).mean(),
        'val_gift_rate': (y_val > 0).mean(),
        'test_gift_rate': (y_test > 0).mean(),
        'train_total_revenue': float(y_train.sum()),
        'val_total_revenue': float(y_val.sum()),
        'test_total_revenue': float(y_test.sum()),
        'feature_cols': feature_cols,
    }

    log(f"Train: {stats['train_samples']:,} samples, gift_rate={stats['train_gift_rate']:.2%}")
    log(f"Val: {stats['val_samples']:,} samples, gift_rate={stats['val_gift_rate']:.2%}")
    log(f"Test: {stats['test_samples']:,} samples, gift_rate={stats['test_gift_rate']:.2%}")

    return (X_train, y_train, X_val, y_val, X_test, y_test,
            train_df, val_df, test_df, feature_cols, stats)


# =============================================================================
# 模型训练
# =============================================================================

def train_model(X_train, y_train):
    """训练 Ridge 回归模型"""
    log(f"Training Ridge regression (alpha={CONFIG['alpha']})...")

    model = Ridge(alpha=CONFIG['alpha'], random_state=CONFIG['random_state'])
    model.fit(X_train, y_train)

    # 训练集 MSE
    train_pred = model.predict(X_train)
    train_mse = np.mean((y_train - train_pred) ** 2)
    train_rmse = np.sqrt(train_mse)

    log(f"Training complete. Train RMSE: {train_rmse:.4f}")

    return model, {'train_mse': train_mse, 'train_rmse': train_rmse}


# =============================================================================
# 评估
# =============================================================================

def evaluate_all_sets(model, X_train, y_train, X_val, y_val, X_test, y_test,
                      train_df, val_df, test_df):
    """在所有数据集上评估"""
    results = {}

    # Train set
    log("Evaluating on train set...")
    train_pred = model.predict(X_train)
    train_result = evaluate_model(
        y_true=y_train,
        y_pred=train_pred,
        test_df=train_df,
        whale_threshold=CONFIG['whale_threshold'],
        compute_stability=True,
        compute_ndcg=True,
        timestamp_col='timestamp'
    )
    results['train'] = train_result
    log(f"  Train RevCap@1%: {train_result.revcap_1pct:.1%}")

    # Val set
    log("Evaluating on val set...")
    val_pred = model.predict(X_val)
    val_result = evaluate_model(
        y_true=y_val,
        y_pred=val_pred,
        test_df=val_df,
        whale_threshold=CONFIG['whale_threshold'],
        compute_stability=True,
        compute_ndcg=True,
        timestamp_col='timestamp'
    )
    results['val'] = val_result
    log(f"  Val RevCap@1%: {val_result.revcap_1pct:.1%}")

    # Test set
    log("Evaluating on test set...")
    test_pred = model.predict(X_test)
    test_result = evaluate_model(
        y_true=y_test,
        y_pred=test_pred,
        test_df=test_df,
        whale_threshold=CONFIG['whale_threshold'],
        compute_stability=True,
        compute_ndcg=True,
        timestamp_col='timestamp'
    )
    results['test'] = test_result
    log(f"  Test RevCap@1%: {test_result.revcap_1pct:.1%}")

    return results, {'train': train_pred, 'val': val_pred, 'test': test_pred}


# =============================================================================
# 可视化
# =============================================================================

def plot_revcap_curves(results, save_path):
    """绘制 RevCap 曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = [0.1, 0.5, 1, 2, 5, 10]

    for split, result in results.items():
        revcap_values = [result.revcap_curve.get(f'{k}%', 0) for k in k_values]
        ax.plot(k_values, [v * 100 for v in revcap_values],
                marker='o', label=f'{split.capitalize()} (RevCap@1%={result.revcap_1pct:.1%})')

    # Oracle
    oracle_values = [results['test'].oracle_curve.get(f'{k}%', 0) for k in k_values]
    ax.plot(k_values, [v * 100 for v in oracle_values],
            '--', color='gray', label='Oracle (Test)', alpha=0.7)

    ax.set_xlabel('Top K%', fontsize=12)
    ax.set_ylabel('Revenue Capture (%)', fontsize=12)
    ax.set_title('Baseline Ridge: Revenue Capture Curves', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved: {save_path}")


def plot_stability(result, save_path, title='Test Set'):
    """绘制稳定性图表"""
    if result.daily_metrics is None or len(result.daily_metrics) == 0:
        return

    daily = result.daily_metrics

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # RevCap by day
    ax = axes[0, 0]
    dates = [str(d) for d in daily['date']]
    ax.bar(range(len(dates)), daily['revcap'] * 100, color='steelblue', alpha=0.7)
    ax.axhline(result.stability['mean'] * 100, color='red', linestyle='--',
               label=f"Mean: {result.stability['mean']:.1%}")
    ax.axhline(result.stability['ci_lower'] * 100, color='orange', linestyle=':', alpha=0.7)
    ax.axhline(result.stability['ci_upper'] * 100, color='orange', linestyle=':', alpha=0.7)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[-5:] for d in dates], rotation=45)
    ax.set_ylabel('RevCap@1% (%)')
    ax.set_title(f'RevCap@1% by Day (CV={result.stability["cv"]:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Whale Recall by day
    ax = axes[0, 1]
    ax.bar(range(len(dates)), daily['whale_recall'] * 100, color='coral', alpha=0.7)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[-5:] for d in dates], rotation=45)
    ax.set_ylabel('Whale Recall@1% (%)')
    ax.set_title('Whale Recall@1% by Day')
    ax.grid(True, alpha=0.3)

    # Max single gift by day
    ax = axes[1, 0]
    ax.bar(range(len(dates)), daily['max_single'], color='forestgreen', alpha=0.7)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[-5:] for d in dates], rotation=45)
    ax.set_ylabel('Max Single Gift (yuan)')
    ax.set_title('Max Single Gift by Day (Outlier Detection)')
    ax.grid(True, alpha=0.3)

    # Total revenue by day
    ax = axes[1, 1]
    ax.bar(range(len(dates)), daily['total_revenue'], color='purple', alpha=0.7)
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([d[-5:] for d in dates], rotation=45)
    ax.set_ylabel('Total Revenue (yuan)')
    ax.set_title('Total Revenue by Day')
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Baseline Ridge: Stability Analysis ({title})', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved: {save_path}")


def plot_calibration(result, save_path):
    """绘制校准图"""
    if not result.calibration:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    buckets = list(result.calibration.keys())
    sum_ratios = [result.calibration[b]['sum_ratio'] for b in buckets]

    x = range(len(buckets))
    ax.bar(x, sum_ratios, color='steelblue', alpha=0.7)
    ax.axhline(1.0, color='red', linestyle='--', label='Perfect Calibration')

    ax.set_xticks(x)
    ax.set_xticklabels(buckets)
    ax.set_ylabel('Pred Sum / Actual Sum')
    ax.set_title('Baseline Ridge: Tail Calibration')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 标注数值
    for i, v in enumerate(sum_ratios):
        ax.text(i, v + 0.05, f'{v:.2f}x', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved: {save_path}")


def plot_feature_importance(model, feature_cols, save_path, top_n=20):
    """绘制特征重要性"""
    coefs = model.coef_
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coef': coefs,
        'abs_coef': np.abs(coefs)
    }).sort_values('abs_coef', ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['coral' if c < 0 else 'steelblue' for c in importance['coef']]
    ax.barh(range(len(importance)), importance['coef'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(importance['feature'])
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Coefficient')
    ax.set_title(f'Baseline Ridge: Top {top_n} Feature Coefficients')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"Saved: {save_path}")


# =============================================================================
# 保存
# =============================================================================

def save_model(model, feature_cols, stats, path):
    """保存模型及元信息"""
    save_dict = {
        'model': model,
        'feature_cols': feature_cols,
        'config': CONFIG,
        'stats': stats,
        'timestamp': datetime.now().isoformat(),
    }
    with open(path, 'wb') as f:
        pickle.dump(save_dict, f)
    log(f"Saved model: {path}")


def save_features(X_train, y_train, X_val, y_val, X_test, y_test,
                  feature_cols, stats, path):
    """保存预处理后的特征"""
    save_dict = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'config': CONFIG,
        'stats': stats,
        'timestamp': datetime.now().isoformat(),
    }
    with open(path, 'wb') as f:
        pickle.dump(save_dict, f)
    log(f"Saved features: {path}")


def save_results(results, train_metrics, stats, path):
    """保存评估结果"""
    output = {
        'config': CONFIG,
        'stats': {k: v for k, v in stats.items() if k != 'feature_cols'},
        'train_metrics': train_metrics,
        'evaluation': {
            split: result.to_dict()
            for split, result in results.items()
        },
        'timestamp': datetime.now().isoformat(),
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"Saved results: {path}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    log("=" * 60)
    log("Baseline Ridge Regression Experiment")
    log("=" * 60)

    # 1. 加载数据
    (X_train, y_train, X_val, y_val, X_test, y_test,
     train_df, val_df, test_df, feature_cols, stats) = load_data()

    # 2. 训练模型
    model, train_metrics = train_model(X_train, y_train)

    # 3. 评估
    results, predictions = evaluate_all_sets(
        model, X_train, y_train, X_val, y_val, X_test, y_test,
        train_df, val_df, test_df
    )

    # 4. 打印完整测试集评估
    log("\n" + "=" * 60)
    log("TEST SET EVALUATION")
    log("=" * 60)
    print(results['test'].summary())

    # 5. 生成图表
    log("\nGenerating plots...")
    plot_revcap_curves(results, IMG_DIR / 'baseline_revcap_curves.png')
    plot_stability(results['test'], IMG_DIR / 'baseline_stability.png')
    plot_calibration(results['test'], IMG_DIR / 'baseline_calibration.png')
    plot_feature_importance(model, feature_cols, IMG_DIR / 'baseline_feature_importance.png')

    # 6. 保存
    log("\nSaving artifacts...")
    model_path = MODELS_DIR / f"{CONFIG['model_name']}.pkl"
    features_path = FEATURES_DIR / f"{CONFIG['model_name']}_features.pkl"
    results_path = RESULTS_DIR / f"{CONFIG['model_name']}_results.json"

    save_model(model, feature_cols, stats, model_path)
    save_features(X_train, y_train, X_val, y_val, X_test, y_test,
                  feature_cols, stats, features_path)
    save_results(results, train_metrics, stats, results_path)

    # 7. 打印汇总
    log("\n" + "=" * 60)
    log("EXPERIMENT SUMMARY")
    log("=" * 60)

    print(f"""
Model: Ridge Regression (alpha={CONFIG['alpha']})
Target: Raw Y (gift amount)
Data Split: {CONFIG['train_days']}-{CONFIG['val_days']}-{CONFIG['test_days']} days

Performance:
  - Train RevCap@1%: {results['train'].revcap_1pct:.1%}
  - Val RevCap@1%:   {results['val'].revcap_1pct:.1%}
  - Test RevCap@1%:  {results['test'].revcap_1pct:.1%}

Test Set Diagnostics:
  - Whale Recall@1%:    {results['test'].whale_recall_1pct:.1%}
  - Whale Precision@1%: {results['test'].whale_precision_1pct:.1%}
  - Avg Revenue@1%:     {results['test'].avg_revenue_1pct:.1f} yuan
  - Stability CV:       {results['test'].stability.get('cv', 0):.1%}

Saved Artifacts:
  - Model:    {model_path}
  - Features: {features_path}
  - Results:  {results_path}
  - Plots:    {IMG_DIR}/baseline_*.png
""")

    log("Experiment complete!")
    return results, model, stats


if __name__ == '__main__':
    results, model, stats = main()
