#!/usr/bin/env python3
"""
Baseline Ridge 完整指标评估脚本
================================

运行 metrics.py v2.1 的全部 35 个指标，确保 Baseline 有完整的指标覆盖。

已跑（在其他脚本中）：
- L1-L4 基础指标
- L7 v2.1 决策核心指标

本脚本补充：
- L5 切片指标（详细）
- L6 生态指标（含 Overload）

Author: Viska Wei
Date: 2026-01-19
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.insert(0, '/home/swei20/GiftLive')

from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
from gift_EVpred.metrics import (
    evaluate_model,
    compute_slice_metrics,
    compute_ecosystem_metrics,
    compute_misallocation_cost,
    compute_capture_auc,
    compute_drift,
    compute_diversity,
    compute_whale_metrics,
    compute_user_level_whale_metrics,
    gini_coefficient,
)

# =============================================================================
# 配置
# =============================================================================

BASE_DIR = Path('/home/swei20/GiftLive/gift_EVpred')
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'

WHALE_THRESHOLD = 100


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_model_and_data():
    """加载模型和数据"""
    log("Loading model...")
    model_path = MODELS_DIR / 'baseline_ridge_v1.pkl'
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    model = model_dict['model']
    model_feature_cols = model_dict['feature_cols']
    log(f"Model expects {len(model_feature_cols)} features")
    
    log("Loading data...")
    train_df, val_df, test_df = prepare_dataset(
        train_days=7, val_days=7, test_days=7
    )
    
    # 使用数据中可用的特征列
    available_feature_cols = get_feature_columns(test_df)
    log(f"Data has {len(available_feature_cols)} features")
    
    # 检查是否需要重新训练
    if len(model_feature_cols) != len(available_feature_cols):
        log(f"⚠️ Feature mismatch: model={len(model_feature_cols)}, data={len(available_feature_cols)}")
        log("Retraining model with current features...")
        
        from sklearn.linear_model import Ridge
        X_train = train_df[available_feature_cols].values
        y_train = train_df['target_raw'].values
        
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X_train, y_train)
        log("Model retrained!")
        
        feature_cols = available_feature_cols
    else:
        feature_cols = model_feature_cols
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target_raw'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target_raw'].values
    
    log(f"Using {len(feature_cols)} features")
    return model, train_df, test_df, X_train, y_train, X_test, y_test


def run_complete_evaluation():
    """运行完整评估"""
    log("=" * 60)
    log("Baseline Ridge - Full Metrics Evaluation (v2.1)")
    log("=" * 60)
    
    model, train_df, test_df, X_train, y_train, X_test, y_test = load_model_and_data()
    
    # 生成预测
    log("Generating predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # ==========================================================================
    # 1. 完整评估（包含所有开关）
    # ==========================================================================
    log("\n--- Running full evaluate_model() ---")
    
    full_result = evaluate_model(
        y_true=y_test,
        y_pred=y_pred_test,
        test_df=test_df,
        whale_threshold=WHALE_THRESHOLD,
        compute_stability=True,
        compute_ndcg=True,
        timestamp_col='timestamp',
        # v2.0 新增
        y_prob=None,  # 回归模型无概率输出
        compute_slices=True,
        compute_ecosystem=True,
        # v2.1 决策核心
        y_pred_train=y_pred_train,
        compute_decision_metrics=True,
    )
    
    log("Full evaluation complete!")
    print(full_result.summary())
    
    # ==========================================================================
    # 2. 详细切片指标
    # ==========================================================================
    log("\n--- Running detailed slice metrics ---")
    
    slice_result = compute_slice_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        df=test_df,
        whale_threshold=WHALE_THRESHOLD,
        k_values=[0.001, 0.005, 0.01, 0.02, 0.05, 0.10],
        user_col='user_id',
        streamer_col='streamer_id',
        pair_hist_col='pair_gift_cnt_hist',       # 正确列名
        streamer_hist_col='str_gift_cnt_hist',    # 正确列名
        user_value_col='user_gift_sum_hist',      # 正确列名
        streamer_value_col='str_gift_sum_hist',   # 正确列名
    )
    
    # 打印切片摘要
    print("\n=== Slice Metrics Summary ===")
    for slice_name, slice_data in slice_result.items():
        if slice_name == 'skipped':
            continue
        n = slice_data.get('n', 0)
        revcap = slice_data.get('revcap_1pct', 0)
        whale_recall = slice_data.get('whale_recall_1pct', 0)
        print(f"  {slice_name}: n={n:,}, RevCap@1%={revcap:.1%}, Whale Recall={whale_recall:.1%}")
    
    if slice_result.get('skipped'):
        print(f"  Skipped: {list(slice_result['skipped'].keys())}")
    
    # ==========================================================================
    # 3. 详细生态指标
    # ==========================================================================
    log("\n--- Running detailed ecosystem metrics ---")
    
    eco_result = compute_ecosystem_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        df=test_df,
        k_select=0.01,
        user_col='user_id',
        streamer_col='streamer_id',
        timestamp_col='timestamp',
        streamer_hist_col='str_gift_cnt_hist',     # 正确列名
        streamer_value_col='str_gift_sum_hist',    # 正确列名
        user_value_col='user_gift_sum_hist',       # 正确列名
    )
    
    print("\n=== Ecosystem Metrics Summary ===")
    print(f"  Selection: Top {eco_result['selection']['k_select']*100:.0f}% = {eco_result['selection']['n_selected']:,} samples")
    print(f"  Gini: {eco_result['gini']['streamer_revenue_gini']:.3f}")
    print(f"  Top 10% Share: {eco_result['gini']['top10_share']:.1%}")
    print(f"  Streamer Coverage: {eco_result['coverage']['streamer_coverage']:.2%}")
    print(f"  Tail Coverage: {eco_result['coverage']['tail_coverage']:.2%}")
    print(f"  Cold Start Coverage: {eco_result['coverage']['cold_start_streamer_coverage']:.2%}")
    print(f"  Overload Bucket Rate: {eco_result['overload']['overload_bucket_rate']:.2%}")
    print(f"  Overloaded Streamer Rate: {eco_result['overload']['overloaded_streamer_rate']:.2%}")
    
    # ==========================================================================
    # 4. 详细 Whale 指标（6 件套）
    # ==========================================================================
    log("\n--- Running complete whale metrics ---")
    
    whale_result = compute_whale_metrics(
        y_true=y_test,
        y_pred=y_pred_test,
        whale_threshold=WHALE_THRESHOLD,
        k=0.01,
    )
    
    print("\n=== Whale Metrics Summary (K=1%, T=100) ===")
    print(f"  Base Whale Rate:  {whale_result['base_whale_rate']:.4%} (P(y>=T))")
    print(f"  ---")
    print(f"  WRec@1%:          {whale_result['wrec']:.1%} (Whale Recall)")
    print(f"  WRecLift@1%:      {whale_result['wrec_lift']:.1f}× (vs Random)")
    print(f"  WPrec@1%:         {whale_result['wprec']:.1%} (Whale Precision)")
    print(f"  WPrecLift@1%:     {whale_result['wprec_lift']:.1f}× (vs Random)")
    print(f"  WRevCap@1%:       {whale_result['wrevcap']:.1%} (Revenue Capture)")
    print(f"  ---")
    print(f"  OracleWRec@1%:    {whale_result['oracle_wrec']:.1%} (Upper Bound)")
    print(f"  nWRec@1%:         {whale_result['nwrec']:.1%} (vs Oracle)")
    
    # ==========================================================================
    # 5. User-Level Whale 指标（找大哥名单）
    # ==========================================================================
    log("\n--- Running user-level whale metrics (找大哥) ---")
    
    # 添加预测分数到 df
    test_df_with_pred = test_df.copy()
    test_df_with_pred['y_pred'] = y_pred_test
    
    user_whale_results = {}
    for k in [0.001, 0.01, 0.05, 0.10]:
        user_whale = compute_user_level_whale_metrics(
            df=test_df_with_pred,
            y_true_col='target_raw',
            y_pred_col='y_pred',
            user_col='user_id',
            whale_threshold=WHALE_THRESHOLD,
            k=k,
            user_score_agg='sum',
            whale_def='cumulative',
        )
        user_whale_results[f'k_{k}'] = user_whale
    
    # 打印 K=1% 的详细结果
    k1 = user_whale_results['k_0.01']
    print(f"\n=== User-Level Whale Metrics (K=1%, 累计口径≥100) ===")
    print(f"  总用户数:           {k1['n_users']:,}")
    print(f"  Whale 用户数:       {k1['n_whale_users']:,}")
    print(f"  BaseWhaleUserRate:  {k1['base_whale_user_rate']:.2%} (用户中大哥比例)")
    print(f"  ---")
    print(f"  WhaleUserRec@1%:    {k1['whale_user_rec']:.1%} (大哥用户召回)")
    print(f"  WhaleUserRecLift:   {k1['whale_user_rec_lift']:.1f}× (vs Random)")
    print(f"  WhaleUserPrec@1%:   {k1['whale_user_prec']:.1%} (池子中大哥比例)")
    print(f"  UserRevCap@1%:      {k1['user_revcap']:.1%} (用户级收入捕获)")
    print(f"  OracleUserRevCap:   {k1['oracle_user_revcap']:.1%} (理论上限)")
    
    # 打印各 K 值对比
    print(f"\n=== User-Level Metrics by K ===")
    print(f"  {'K':>6} | {'WhaleUserRec':>12} | {'Lift':>6} | {'UserRevCap':>10} | {'OracleRevCap':>12}")
    print(f"  {'-'*6} | {'-'*12} | {'-'*6} | {'-'*10} | {'-'*12}")
    for k_str, result in user_whale_results.items():
        k_val = result['k']
        rec = result['whale_user_rec'] or 0
        lift = result['whale_user_rec_lift'] or 0
        revcap = result['user_revcap'] or 0
        oracle = result['oracle_user_revcap'] or 0
        print(f"  {k_val:>6.1%} | {rec:>11.1%} | {lift:>5.1f}× | {revcap:>9.1%} | {oracle:>11.1%}")
    
    # ==========================================================================
    # 6. 保存完整结果
    # ==========================================================================
    log("\n--- Saving results ---")
    
    # 合并所有结果
    complete_results = {
        'config': {
            'model': 'Ridge',
            'alpha': 1.0,
            'metrics_version': '2.1',
            'timestamp': datetime.now().isoformat(),
        },
        'core_metrics': {
            'revcap_1pct': full_result.revcap_1pct,
            'whale_recall_1pct': full_result.whale_recall_1pct,
            'whale_precision_1pct': full_result.whale_precision_1pct,
            'avg_revenue_1pct': full_result.avg_revenue_1pct,
            'gift_rate_1pct': full_result.gift_rate_1pct,
            'cv': full_result.stability.get('cv', None),
            'ci_lower': full_result.stability.get('ci_lower', None),
            'ci_upper': full_result.stability.get('ci_upper', None),
        },
        'whale_metrics_sample_level': whale_result,
        'whale_metrics_user_level': user_whale_results,
        'revcap_curve': full_result.revcap_curve,
        'oracle_curve': full_result.oracle_curve,
        'calibration': full_result.calibration,
        'slice_metrics': slice_result,
        'ecosystem_metrics': eco_result,
        'decision_metrics': {
            'misallocation_cost': full_result.misallocation_cost,
            'capture_auc': {
                k: v for k, v in full_result.capture_auc.items() 
                if k not in ['curve', 'oracle_curve']  # 去掉大量曲线数据
            },
            'drift': {
                k: v for k, v in full_result.drift.items()
                if k != 'bins_detail'
            },
            'diversity': full_result.diversity,
        },
    }
    
    output_path = RESULTS_DIR / 'baseline_ridge_full_metrics_20260119.json'
    with open(output_path, 'w') as f:
        json.dump(complete_results, f, indent=2, default=str)
    log(f"Saved: {output_path}")
    
    # ==========================================================================
    # 7. 指标覆盖率统计
    # ==========================================================================
    log("\n" + "=" * 60)
    log("METRICS COVERAGE SUMMARY")
    log("=" * 60)
    
    coverage = {
        'L1_main': ['revcap', 'oracle'],
        'L2_diagnostic': ['whale_recall', 'whale_precision', 'avg_revenue', 'gift_rate'],
        'L3_stability': ['cv', 'ci_lower', 'ci_upper', 'daily_metrics'],
        'L4_calibration': ['tail_sum_ratio', 'tail_mean_ratio', 'ece_skipped'],
        'L5_slices': list(slice_result.keys()),
        'L6_ecosystem': list(eco_result.keys()),
        'L7_decision': ['misallocation', 'capture_auc', 'drift', 'diversity'],
    }
    
    total_metrics = 0
    completed_metrics = 0
    
    for layer, metrics in coverage.items():
        if layer == 'L5_slices':
            # 排除 skipped
            actual = [m for m in metrics if m != 'skipped']
            total = 10  # 理论应有 10 个切片
            completed = len(actual)
        elif layer == 'L6_ecosystem':
            actual = [m for m in metrics if m != 'meta']
            total = 4  # gini, coverage, overload, selection
            completed = len(actual)
        else:
            total = len(metrics)
            completed = sum(1 for m in metrics if 'skipped' not in m.lower())
        
        total_metrics += total
        completed_metrics += completed
        status = '✅' if completed >= total * 0.8 else '⚠️'
        print(f"  {status} {layer}: {completed}/{total}")
    
    print(f"\n  Total: {completed_metrics}/{total_metrics} ({completed_metrics/total_metrics:.0%})")
    
    log("\nDone!")
    return complete_results


if __name__ == '__main__':
    results = run_complete_evaluation()
