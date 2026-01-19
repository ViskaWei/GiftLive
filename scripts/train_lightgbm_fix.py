#!/usr/bin/env python3
"""
LightGBM 修复实验：禁用 Early Stopping + LambdaRank
===================================================

修复原实验的问题：
1. Early stopping 第 1 轮就停了（Best iteration = 1）
2. 用 MAE 做 early stopping 在 98.5% 为 0 的数据上会误导

实验变体：
1. LightGBM (no early stop) - 强制跑完 500 棵树
2. LightGBM (LambdaRank) - 用排序损失函数

Author: Viska Wei
Date: 2026-01-18
"""

import sys
sys.path.insert(0, '/home/swei20/GiftLive')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
import json
from pathlib import Path
from datetime import datetime

from gift_EVpred.data_utils import prepare_dataset, get_feature_columns

# =============================================================================
# 评估函数
# =============================================================================
def revenue_capture_at_k(y_true, y_pred, k=0.01):
    """计算 Revenue Capture @k%"""
    n_top = int(len(y_true) * k)
    if n_top == 0:
        n_top = 1
    top_indices = np.argsort(y_pred)[-n_top:]
    total_revenue = y_true.sum()
    if total_revenue == 0:
        return 0.0
    return y_true[top_indices].sum() / total_revenue

def gift_rate_at_k(y_true, y_pred, k=0.01):
    """计算 Gift Rate @k%"""
    n_top = int(len(y_true) * k)
    if n_top == 0:
        n_top = 1
    top_indices = np.argsort(y_pred)[-n_top:]
    return (y_true[top_indices] > 0).mean()

def evaluate_model(y_true, y_pred, name="Model"):
    """评估模型"""
    results = {
        'name': name,
        'rev_cap_0.01': revenue_capture_at_k(y_true, y_pred, 0.01),
        'rev_cap_0.05': revenue_capture_at_k(y_true, y_pred, 0.05),
        'rev_cap_0.10': revenue_capture_at_k(y_true, y_pred, 0.10),
        'gift_rate_0.01': gift_rate_at_k(y_true, y_pred, 0.01),
        'gift_rate_0.05': gift_rate_at_k(y_true, y_pred, 0.05),
        'spearman': spearmanr(y_true, y_pred)[0],
    }
    return results

def print_results(results, baseline_rev_cap=None):
    """打印结果"""
    print(f"\n{'='*60}")
    print(f"📊 {results['name']}")
    print(f"{'='*60}")
    print(f"  RevCap@1%:  {results['rev_cap_0.01']*100:.2f}%", end="")
    if baseline_rev_cap:
        diff = (results['rev_cap_0.01'] - baseline_rev_cap) / baseline_rev_cap * 100
        print(f"  ({diff:+.2f}% vs Ridge)")
    else:
        print()
    print(f"  RevCap@5%:  {results['rev_cap_0.05']*100:.2f}%")
    print(f"  RevCap@10%: {results['rev_cap_0.10']*100:.2f}%")
    print(f"  GiftRate@1%: {results['gift_rate_0.01']*100:.2f}%")
    print(f"  Spearman:   {results['spearman']:.4f}")

# =============================================================================
# 主实验
# =============================================================================
def main():
    print("="*60)
    print("🍃 LightGBM 修复实验：禁用 Early Stopping + LambdaRank")
    print("="*60)

    # 加载数据
    print("\n📦 Loading data...")
    train_df, val_df, test_df = prepare_dataset()
    feature_cols = get_feature_columns(train_df)

    print(f"  Train: {len(train_df):,}")
    print(f"  Val:   {len(val_df):,}")
    print(f"  Test:  {len(test_df):,}")
    print(f"  Features: {len(feature_cols)}")

    # 准备数据
    X_train = train_df[feature_cols].values
    y_train = train_df['target_raw'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target_raw'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target_raw'].values

    all_results = {}

    # =========================================================================
    # Baseline: Ridge
    # =========================================================================
    print("\n" + "="*60)
    print("🔵 Training Ridge (Baseline)...")
    print("="*60)

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_pred = ridge.predict(X_test)

    ridge_results = evaluate_model(y_test, ridge_pred, "Ridge (Baseline)")
    print_results(ridge_results)
    all_results['ridge'] = ridge_results
    baseline_rev_cap = ridge_results['rev_cap_0.01']

    # =========================================================================
    # Exp 1: LightGBM 禁用 Early Stopping
    # =========================================================================
    print("\n" + "="*60)
    print("🟢 Training LightGBM (No Early Stopping, 500 trees)...")
    print("="*60)

    params_no_es = {
        'objective': 'regression',
        'metric': 'mae',
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1,
    }

    lgb_no_es = lgb.LGBMRegressor(**params_no_es)
    # 不传 eval_set 和 callbacks，强制跑完所有树
    lgb_no_es.fit(X_train, y_train)
    lgb_no_es_pred = lgb_no_es.predict(X_test)

    lgb_no_es_results = evaluate_model(y_test, lgb_no_es_pred, "LightGBM (No Early Stop, 500 trees)")
    print_results(lgb_no_es_results, baseline_rev_cap)
    all_results['lgb_no_early_stop'] = lgb_no_es_results

    # 特征重要性
    importance = lgb_no_es.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print("\n📊 Top 10 Feature Importance:")
    print(importance_df.head(10).to_string(index=False))

    # =========================================================================
    # Exp 2: LightGBM LambdaRank (按 streamer 分组)
    # =========================================================================
    print("\n" + "="*60)
    print("🟡 Training LightGBM (LambdaRank, 500 trees)...")
    print("="*60)

    # 为 LambdaRank 准备 group 信息 - 按 streamer_id 分组
    # 每个主播的观众作为一个 query group
    train_df_sorted = train_df.sort_values('streamer_id').reset_index(drop=True)
    train_group = train_df_sorted.groupby('streamer_id').size().values.tolist()

    val_df_sorted = val_df.sort_values('streamer_id').reset_index(drop=True)
    val_group = val_df_sorted.groupby('streamer_id').size().values.tolist()

    X_train_rank = train_df_sorted[feature_cols].values
    # LambdaRank 需要离散等级标签 (0-4)
    def to_relevance(y):
        """转换为 0-4 等级: 0=无, 1=小额, 2=中额, 3=大额, 4=whale"""
        rel = np.zeros(len(y), dtype=int)
        rel[y > 0] = 1      # 任何打赏
        rel[y > 10] = 2     # >10元
        rel[y > 50] = 3     # >50元
        rel[y > 100] = 4    # whale
        return rel

    y_train_rank = to_relevance(train_df_sorted['target_raw'].values)
    X_val_rank = val_df_sorted[feature_cols].values
    y_val_rank = to_relevance(val_df_sorted['target_raw'].values)

    print(f"  Train label dist: {np.bincount(y_train_rank)}")
    print(f"  Val label dist: {np.bincount(y_val_rank)}")

    print(f"  Train groups: {len(train_group)}, avg size: {np.mean(train_group):.1f}")
    print(f"  Val groups: {len(val_group)}, avg size: {np.mean(val_group):.1f}")

    params_rank = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'max_depth': 8,
        'learning_rate': 0.05,
        'num_leaves': 64,
        'min_child_samples': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'seed': 42,
        'verbose': -1,
        'n_jobs': -1,
        'lambdarank_truncation_level': 30,
    }

    # 创建 Dataset
    train_data = lgb.Dataset(X_train_rank, label=y_train_rank, group=train_group)
    val_data = lgb.Dataset(X_val_rank, label=y_val_rank, group=val_group, reference=train_data)

    # 训练
    lgb_rank = lgb.train(
        params_rank,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(100)],
    )
    lgb_rank_pred = lgb_rank.predict(X_test)

    lgb_rank_results = evaluate_model(y_test, lgb_rank_pred, "LightGBM (LambdaRank, 500 trees)")
    print_results(lgb_rank_results, baseline_rev_cap)
    all_results['lgb_lambdarank'] = lgb_rank_results

    # =========================================================================
    # 汇总对比
    # =========================================================================
    print("\n" + "="*60)
    print("📋 汇总对比")
    print("="*60)

    print(f"\n{'Model':<40} {'RevCap@1%':>12} {'vs Ridge':>12}")
    print("-" * 64)
    for key, res in all_results.items():
        rev_cap = res['rev_cap_0.01'] * 100
        diff = (res['rev_cap_0.01'] - baseline_rev_cap) / baseline_rev_cap * 100
        print(f"{res['name']:<40} {rev_cap:>11.2f}% {diff:>+11.2f}%")

    # 保存结果
    output_path = Path("/home/swei20/GiftLive/gift_EVpred/results/lightgbm_fix_20260118.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为可序列化格式
    save_results = {}
    for k, v in all_results.items():
        save_results[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                          for kk, vv in v.items()}

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\n✅ Results saved to {output_path}")

if __name__ == "__main__":
    main()
