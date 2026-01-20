#!/usr/bin/env python3
"""
Baseline Ridge - Strict vs Benchmark 对比实验
==============================================

运行两个版本的 Ridge baseline：
1. Strict Mode (20 features): 只保留真正静态/在线可得字段
2. Benchmark Mode (31 features): 包含快照累计特征

所有指标使用 metrics.py v2.1 完整版

Author: Viska Wei
Date: 2026-01-19
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import Ridge

sys.path.insert(0, '/home/swei20/GiftLive')

from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
from gift_EVpred.metrics import (
    evaluate_model,
    compute_slice_metrics,
    compute_ecosystem_metrics,
)

BASE_DIR = Path('/home/swei20/GiftLive/gift_EVpred')
RESULTS_DIR = BASE_DIR / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

WHALE_THRESHOLD = 100

# 切片和生态配置
SLICE_CONFIG = {
    'pair_hist_col': 'pair_gift_cnt_hist',
    'streamer_hist_col': 'str_gift_cnt_hist',
    'user_value_col': 'user_gift_sum_hist',
    'streamer_value_col': 'str_gift_sum_hist',
}
ECOSYSTEM_CONFIG = {
    'streamer_hist_col': 'str_gift_cnt_hist',
    'streamer_value_col': 'str_gift_sum_hist',
    'user_value_col': 'user_gift_sum_hist',
}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def run_experiment(strict_mode: bool):
    """运行单个模式的实验"""
    mode_name = "Strict" if strict_mode else "Benchmark"
    log(f"\n{'='*60}")
    log(f"Running {mode_name} Mode (strict_mode={strict_mode})")
    log(f"{'='*60}")
    
    # 1. 加载数据
    log(f"Loading data ({mode_name} mode)...")
    train_df, val_df, test_df = prepare_dataset(
        train_days=7, val_days=7, test_days=7,
        strict_mode=strict_mode
    )
    
    feature_cols = get_feature_columns(train_df)
    log(f"Features: {len(feature_cols)}")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['target_raw'].values
    X_val = val_df[feature_cols].values
    y_val = val_df['target_raw'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['target_raw'].values
    
    log(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    
    # 2. 训练模型
    log("Training Ridge (alpha=1.0)...")
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    
    # 3. 生成预测
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # 4. 评估 (完整 v2.1 指标)
    log("Evaluating with full v2.1 metrics...")
    
    # Test set 完整评估
    test_result = evaluate_model(
        y_true=y_test,
        y_pred=y_pred_test,
        test_df=test_df,
        whale_threshold=WHALE_THRESHOLD,
        compute_stability=True,
        compute_ndcg=True,
        timestamp_col='timestamp',
        y_pred_train=y_pred_train,
        compute_slices=True,
        compute_ecosystem=True,
        compute_decision_metrics=True,
        slice_config=SLICE_CONFIG,
        ecosystem_config=ECOSYSTEM_CONFIG,
    )
    
    # Val set 简单评估
    val_result = evaluate_model(
        y_true=y_val,
        y_pred=y_pred_val,
        test_df=val_df,
        whale_threshold=WHALE_THRESHOLD,
        compute_stability=True,
    )
    
    # Train set 简单评估
    train_result = evaluate_model(
        y_true=y_train,
        y_pred=y_pred_train,
        test_df=train_df,
        whale_threshold=WHALE_THRESHOLD,
        compute_stability=True,
    )
    
    # 5. 打印结果摘要
    print(f"\n{'='*60}")
    print(f"{mode_name} Mode Results ({len(feature_cols)} features)")
    print(f"{'='*60}")
    print(f"Train RevCap@1%: {train_result.revcap_1pct:.1%}")
    print(f"Val RevCap@1%:   {val_result.revcap_1pct:.1%}")
    print(f"Test RevCap@1%:  {test_result.revcap_1pct:.1%}")
    print(f"")
    print(f"Test Diagnostics:")
    print(f"  Whale Recall@1%:    {test_result.whale_recall_1pct:.1%}")
    print(f"  Whale Precision@1%: {test_result.whale_precision_1pct:.1%}")
    print(f"  CV (Stability):     {test_result.stability.get('cv', 0):.1%}")
    print(f"  PSI (Drift):        {test_result.drift.get('psi', 0):.4f}")
    print(f"  Capture AUC:        {test_result.capture_auc.get('normalized_auc', 0):.1%}")
    print(f"  Efficiency:         {test_result.misallocation_cost.get('efficiency', 0):.1%}")
    print(f"  Effective#:         {test_result.diversity.get('effective_number', 0):.0f}")
    
    # 6. 返回结果
    return {
        'mode': mode_name,
        'strict_mode': strict_mode,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'train': {
            'revcap_1pct': train_result.revcap_1pct,
            'cv': train_result.stability.get('cv'),
        },
        'val': {
            'revcap_1pct': val_result.revcap_1pct,
            'cv': val_result.stability.get('cv'),
        },
        'test': {
            'revcap_1pct': test_result.revcap_1pct,
            'whale_recall_1pct': test_result.whale_recall_1pct,
            'whale_precision_1pct': test_result.whale_precision_1pct,
            'avg_revenue_1pct': test_result.avg_revenue_1pct,
            'gift_rate_1pct': test_result.gift_rate_1pct,
            'cv': test_result.stability.get('cv'),
            'ci_lower': test_result.stability.get('ci_lower'),
            'ci_upper': test_result.stability.get('ci_upper'),
            'revcap_curve': test_result.revcap_curve,
            'oracle_curve': test_result.oracle_curve,
            'calibration': test_result.calibration,
            'misallocation_cost': test_result.misallocation_cost,
            'capture_auc': {k: v for k, v in test_result.capture_auc.items() 
                           if k not in ['curve', 'oracle_curve']},
            'drift': {k: v for k, v in test_result.drift.items() if k != 'bins_detail'},
            'diversity': test_result.diversity,
            'ecosystem': test_result.ecosystem,
        },
        'slice_metrics': test_result.slice_metrics,
    }


def main():
    log("="*60)
    log("Baseline Ridge: Strict vs Benchmark Comparison")
    log("="*60)
    
    # 运行两个模式
    strict_result = run_experiment(strict_mode=True)
    benchmark_result = run_experiment(strict_mode=False)
    
    # 对比表格
    print("\n" + "="*70)
    print("COMPARISON: Strict vs Benchmark")
    print("="*70)
    
    s = strict_result['test']
    b = benchmark_result['test']
    
    print(f"""
| 指标 | Strict ({strict_result['n_features']} feat) | Benchmark ({benchmark_result['n_features']} feat) | Δ |
|------|--------|-----------|-----|
| RevCap@1% | {s['revcap_1pct']:.1%} | {b['revcap_1pct']:.1%} | {(b['revcap_1pct']-s['revcap_1pct'])*100:+.1f}pp |
| Whale Recall@1% | {s['whale_recall_1pct']:.1%} | {b['whale_recall_1pct']:.1%} | {(b['whale_recall_1pct']-s['whale_recall_1pct'])*100:+.1f}pp |
| CV (Stability) | {s['cv']:.1%} | {b['cv']:.1%} | {(b['cv']-s['cv'])*100:+.1f}pp |
| PSI (Drift) | {s['drift']['psi']:.4f} | {b['drift']['psi']:.4f} | {b['drift']['psi']-s['drift']['psi']:+.4f} |
| Capture AUC | {s['capture_auc']['normalized_auc']:.1%} | {b['capture_auc']['normalized_auc']:.1%} | {(b['capture_auc']['normalized_auc']-s['capture_auc']['normalized_auc'])*100:+.1f}pp |
| Efficiency | {s['misallocation_cost']['efficiency']:.1%} | {b['misallocation_cost']['efficiency']:.1%} | {(b['misallocation_cost']['efficiency']-s['misallocation_cost']['efficiency'])*100:+.1f}pp |
| Effective# | {s['diversity']['effective_number']:.0f} | {b['diversity']['effective_number']:.0f} | {b['diversity']['effective_number']-s['diversity']['effective_number']:+.0f} |
""")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'strict': strict_result,
        'benchmark': benchmark_result,
        'comparison': {
            'revcap_1pct_delta': b['revcap_1pct'] - s['revcap_1pct'],
            'whale_recall_delta': b['whale_recall_1pct'] - s['whale_recall_1pct'],
            'cv_delta': b['cv'] - s['cv'],
            'psi_delta': b['drift']['psi'] - s['drift']['psi'],
        }
    }
    
    output_path = RESULTS_DIR / 'baseline_ridge_strict_vs_benchmark_20260119.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log(f"\nSaved: {output_path}")
    
    log("\nDone!")
    return output


if __name__ == '__main__':
    results = main()
