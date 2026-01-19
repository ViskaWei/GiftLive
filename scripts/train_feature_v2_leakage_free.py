#!/usr/bin/env python3
"""
Feature Engineering V2 - Leakage-Free Version
==============================================

基于 Day-Frozen 框架的特征工程 V2。

新增特征（全部无泄漏）：
1. Sequence 特征：用户/pair 最近 N 次打赏统计
2. Coldstart Tier：用户/主播消费档次（基于历史分位数）
3. 时间间隔特征：距离上次打赏的天数

Author: Viska Wei
Date: 2026-01-18
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_EVpred"
RESULT_DIR = OUTPUT_DIR / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 日志工具
# =============================================================================
def log(msg: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


# =============================================================================
# Day-Frozen Sequence 特征
# =============================================================================
def create_sequence_features(gift, click):
    """
    创建 Day-Frozen Sequence 特征
    """
    log("Creating Day-Frozen sequence features...")

    gift = gift.copy()
    click = click.copy()

    gift['day'] = pd.to_datetime(gift['timestamp'], unit='ms').dt.normalize()
    click['day'] = pd.to_datetime(click['timestamp'], unit='ms').dt.normalize()

    gift_sorted = gift.sort_values(['user_id', 'streamer_id', 'day']).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Pair-level: 最近一次打赏的 day
    # -------------------------------------------------------------------------
    log("  Building pair last gift day...")

    pair_last = gift_sorted.groupby(['user_id', 'streamer_id', 'day']).agg(
        last_gift_price=('gift_price', 'last')
    ).reset_index()

    pair_last = pair_last.sort_values('day').reset_index(drop=True)
    click_sorted = click.sort_values('day').reset_index(drop=True)

    click_with_seq = pd.merge_asof(
        click_sorted,
        pair_last[['day', 'user_id', 'streamer_id']].rename(columns={'day': 'pair_last_gift_day'}),
        left_on='day',
        right_on='pair_last_gift_day',
        by=['user_id', 'streamer_id'],
        direction='backward',
        allow_exact_matches=False
    )

    click_with_seq['pair_last_gift_gap'] = (
        (click_with_seq['day'] - click_with_seq['pair_last_gift_day']).dt.days
    ).fillna(-1)

    # -------------------------------------------------------------------------
    # Pair-level: 最近 3 次打赏金额均值
    # -------------------------------------------------------------------------
    log("  Building pair seq-3 mean...")

    pair_day_gift = gift_sorted.groupby(['user_id', 'streamer_id', 'day']).agg(
        day_gift_sum=('gift_price', 'sum'),
        day_gift_cnt=('gift_price', 'count')
    ).reset_index()

    pair_day_gift = pair_day_gift.sort_values(['user_id', 'streamer_id', 'day'])
    pair_day_gift['pair_seq_3_sum'] = pair_day_gift.groupby(
        ['user_id', 'streamer_id']
    )['day_gift_sum'].transform(lambda x: x.rolling(3, min_periods=1).sum())

    pair_day_gift['pair_seq_3_cnt'] = pair_day_gift.groupby(
        ['user_id', 'streamer_id']
    )['day_gift_cnt'].transform(lambda x: x.rolling(3, min_periods=1).sum())

    pair_day_gift['pair_seq_3_mean'] = pair_day_gift['pair_seq_3_sum'] / pair_day_gift['pair_seq_3_cnt']

    pair_day_gift = pair_day_gift.sort_values('day').reset_index(drop=True)

    click_with_seq = pd.merge_asof(
        click_with_seq,
        pair_day_gift[['day', 'user_id', 'streamer_id', 'pair_seq_3_mean']],
        on='day',
        by=['user_id', 'streamer_id'],
        direction='backward',
        allow_exact_matches=False
    )
    click_with_seq['pair_seq_3_mean'] = click_with_seq['pair_seq_3_mean'].fillna(0)

    # -------------------------------------------------------------------------
    # User-level: 最近 3 次打赏金额均值
    # -------------------------------------------------------------------------
    log("  Building user seq-3 mean...")

    user_day_gift = gift_sorted.groupby(['user_id', 'day']).agg(
        day_gift_sum=('gift_price', 'sum'),
        day_gift_cnt=('gift_price', 'count')
    ).reset_index()

    user_day_gift = user_day_gift.sort_values(['user_id', 'day'])
    user_day_gift['user_seq_3_sum'] = user_day_gift.groupby('user_id')['day_gift_sum'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    user_day_gift['user_seq_3_cnt'] = user_day_gift.groupby('user_id')['day_gift_cnt'].transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    user_day_gift['user_seq_3_mean'] = user_day_gift['user_seq_3_sum'] / user_day_gift['user_seq_3_cnt']

    user_day_gift = user_day_gift.sort_values('day').reset_index(drop=True)

    click_with_seq = pd.merge_asof(
        click_with_seq,
        user_day_gift[['day', 'user_id', 'user_seq_3_mean']],
        on='day',
        by='user_id',
        direction='backward',
        allow_exact_matches=False
    )
    click_with_seq['user_seq_3_mean'] = click_with_seq['user_seq_3_mean'].fillna(0)

    click_with_seq = click_with_seq.drop(columns=['pair_last_gift_day'], errors='ignore')

    log(f"  Sequence features created!", "SUCCESS")

    return click_with_seq


# =============================================================================
# 特征列工具
# =============================================================================
EXCLUDE_COLUMNS = [
    'user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
    'gift_price_label', 'target', 'target_raw', 'is_gift',
    'label_end_dt', '_datetime', '_date', 'day',
    'watch_live_time', 'watch_time_log', 'watch_time_ratio',
]

def get_feature_columns(df):
    return [c for c in df.columns if c not in EXCLUDE_COLUMNS]


# =============================================================================
# 评估函数
# =============================================================================
def evaluate_model(y_true, y_pred, y_raw):
    results = {}
    results['spearman'], _ = stats.spearmanr(y_pred, y_raw)

    n = len(y_pred)
    for k_pct in [1, 5]:
        k = int(n * k_pct / 100)
        top_k_idx = np.argsort(y_pred)[-k:]
        results[f'revenue_capture_{k_pct}pct'] = y_raw[top_k_idx].sum() / y_raw.sum()

    k = int(n * 0.01)
    top_k_idx = np.argsort(y_pred)[-k:]
    results['top1_gift_rate'] = (y_raw[top_k_idx] > 0).mean()

    oracle_idx = np.argsort(y_raw)[-k:]
    results['oracle_rev_1pct'] = y_raw[oracle_idx].sum() / y_raw.sum()

    return results


# =============================================================================
# 主函数
# =============================================================================
def main():
    log("=" * 60)
    log("Feature Engineering V2 - Leakage-Free Experiment")
    log("=" * 60)

    # =========================================================================
    # 加载 Day-Frozen Baseline 数据
    # =========================================================================
    import sys
    sys.path.insert(0, str(BASE_DIR))
    from gift_EVpred.data_utils import prepare_dataset, get_feature_columns as get_base_features, load_raw_data

    # 使用 data_utils 的标准数据准备流程
    train_df, val_df, test_df = prepare_dataset()

    # 获取 baseline 特征列（31 个）
    baseline_feature_cols = get_base_features(train_df)
    log(f"\nBaseline features ({len(baseline_feature_cols)}): {baseline_feature_cols[:5]}...")

    # =========================================================================
    # 添加 Sequence 特征
    # =========================================================================
    gift, click, _, _, _ = load_raw_data()

    # 为 test_df 添加 sequence 特征
    log("\nAdding sequence features to test set...")

    # 先合并所有数据
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # 创建 sequence 特征
    all_with_seq = create_sequence_features(gift, all_df)

    # 重新 split
    all_with_seq['_date'] = pd.to_datetime(all_with_seq['timestamp'], unit='ms').dt.date

    min_date = all_with_seq['_date'].min()
    train_end = min_date + timedelta(days=6)
    val_start = train_end + timedelta(days=1)
    val_end = val_start + timedelta(days=6)
    test_start = val_end + timedelta(days=1)
    test_end = test_start + timedelta(days=6)

    train_df_v2 = all_with_seq[all_with_seq['_date'] <= train_end].copy()
    val_df_v2 = all_with_seq[(all_with_seq['_date'] >= val_start) & (all_with_seq['_date'] <= val_end)].copy()
    test_df_v2 = all_with_seq[(all_with_seq['_date'] >= test_start) & (all_with_seq['_date'] <= test_end)].copy()

    for df in [train_df_v2, val_df_v2, test_df_v2]:
        df.drop(columns=['_date', 'day'], inplace=True, errors='ignore')

    # Fill NaN
    for df in [train_df_v2, val_df_v2, test_df_v2]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

    log(f"Train: {len(train_df_v2):,}, Val: {len(val_df_v2):,}, Test: {len(test_df_v2):,}")

    # =========================================================================
    # 定义特征组
    # =========================================================================
    seq_features = ['pair_last_gift_gap', 'pair_seq_3_mean', 'user_seq_3_mean']

    # All features = baseline + sequence
    all_feature_cols = baseline_feature_cols + [f for f in seq_features if f in train_df_v2.columns]

    log(f"\nAll features ({len(all_feature_cols)}):")
    log(f"  Baseline: {len(baseline_feature_cols)}")
    log(f"  Sequence: {[f for f in seq_features if f in train_df_v2.columns]}")

    # =========================================================================
    # 准备数据
    # =========================================================================
    X_train = train_df_v2[all_feature_cols].values
    y_train = train_df_v2['target'].values
    y_train_raw = train_df_v2['target_raw'].values
    is_gift_train = train_df_v2['is_gift'].values

    X_test = test_df_v2[all_feature_cols].values
    y_test = test_df_v2['target'].values
    y_test_raw = test_df_v2['target_raw'].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # =========================================================================
    # 实验 1: Baseline (31 features) - 对照组
    # =========================================================================
    log("\n" + "=" * 60)
    log("Experiment 1: Baseline (31 features)")
    log("=" * 60)

    base_idx = [all_feature_cols.index(c) for c in baseline_feature_cols]

    # Direct
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_scaled[:, base_idx], y_train)
    y_pred_direct = model_direct.predict(X_test_scaled[:, base_idx])
    results['baseline_direct'] = evaluate_model(y_test, y_pred_direct, y_test_raw)
    log(f"  Direct: Spearman={results['baseline_direct']['spearman']:.4f}, "
        f"RevCap@1%={results['baseline_direct']['revenue_capture_1pct']*100:.2f}%")

    # Two-Stage
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_scaled[:, base_idx], is_gift_train)
    p_gift = clf.predict_proba(X_test_scaled[:, base_idx])[:, 1]

    gifter_mask_train = is_gift_train == 1
    reg = Ridge(alpha=1.0)
    reg.fit(X_train_scaled[gifter_mask_train][:, base_idx], y_train[gifter_mask_train])
    e_gift = reg.predict(X_test_scaled[:, base_idx])

    y_pred_twostage = p_gift * e_gift
    results['baseline_twostage'] = evaluate_model(y_test, y_pred_twostage, y_test_raw)
    log(f"  Two-Stage: Spearman={results['baseline_twostage']['spearman']:.4f}, "
        f"RevCap@1%={results['baseline_twostage']['revenue_capture_1pct']*100:.2f}%")

    # =========================================================================
    # 实验 2: Baseline + Sequence (34 features)
    # =========================================================================
    log("\n" + "=" * 60)
    log("Experiment 2: Baseline + Sequence (34 features)")
    log("=" * 60)

    # Direct
    model_direct = Ridge(alpha=1.0)
    model_direct.fit(X_train_scaled, y_train)
    y_pred_direct = model_direct.predict(X_test_scaled)
    results['base_seq_direct'] = evaluate_model(y_test, y_pred_direct, y_test_raw)
    log(f"  Direct: Spearman={results['base_seq_direct']['spearman']:.4f}, "
        f"RevCap@1%={results['base_seq_direct']['revenue_capture_1pct']*100:.2f}%")

    # Two-Stage
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_scaled, is_gift_train)
    p_gift = clf.predict_proba(X_test_scaled)[:, 1]

    reg = Ridge(alpha=1.0)
    reg.fit(X_train_scaled[gifter_mask_train], y_train[gifter_mask_train])
    e_gift = reg.predict(X_test_scaled)

    y_pred_twostage = p_gift * e_gift
    results['base_seq_twostage'] = evaluate_model(y_test, y_pred_twostage, y_test_raw)
    log(f"  Two-Stage: Spearman={results['base_seq_twostage']['spearman']:.4f}, "
        f"RevCap@1%={results['base_seq_twostage']['revenue_capture_1pct']*100:.2f}%")

    # 分类器性能
    from sklearn.metrics import roc_auc_score
    clf_auc = roc_auc_score(test_df_v2['is_gift'].values, p_gift)
    log(f"  Classifier AUC: {clf_auc:.4f}")

    # =========================================================================
    # 汇总结果
    # =========================================================================
    log("\n" + "=" * 60)
    log("Summary")
    log("=" * 60)

    summary = []
    for name, res in results.items():
        summary.append({
            'config': name,
            'features': len(baseline_feature_cols) if 'baseline' in name else len(all_feature_cols),
            'spearman': res['spearman'],
            'rev_cap_1pct': res['revenue_capture_1pct'],
            'rev_cap_5pct': res['revenue_capture_5pct'],
            'top1_gift_rate': res['top1_gift_rate'],
        })

    summary_df = pd.DataFrame(summary)
    print("\n", summary_df.to_string(index=False))

    # 保存结果
    output_file = RESULT_DIR / "feature_v2_leakage_free_20260118.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    log(f"\nResults saved to {output_file}", "SUCCESS")

    # 打印对比
    log("\n" + "=" * 60)
    log("Feature Ablation Analysis")
    log("=" * 60)

    baseline_revcap = results['baseline_twostage']['revenue_capture_1pct']
    seq_revcap = results['base_seq_twostage']['revenue_capture_1pct']
    delta = (seq_revcap - baseline_revcap) / baseline_revcap * 100

    log(f"  Baseline (31 features): RevCap@1% = {baseline_revcap*100:.2f}%")
    log(f"  + Sequence (34 features): RevCap@1% = {seq_revcap*100:.2f}% (delta={delta:+.1f}%)")

    if delta > 0:
        log(f"\n  ✅ Sequence features improve RevCap@1% by {delta:.1f}%", "SUCCESS")
    else:
        log(f"\n  ⚠️ Sequence features decrease RevCap@1% by {abs(delta):.1f}%", "WARNING")

    # =========================================================================
    # 特征重要性分析
    # =========================================================================
    log("\n" + "=" * 60)
    log("Feature Importance (Two-Stage Classifier)")
    log("=" * 60)

    # 重新训练获取系数
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train_scaled, is_gift_train)

    coef_df = pd.DataFrame({
        'feature': all_feature_cols,
        'coef': clf.coef_[0]
    }).sort_values('coef', key=abs, ascending=False)

    log("  Top 10 features by |coefficient|:")
    for _, row in coef_df.head(10).iterrows():
        log(f"    {row['feature']}: {row['coef']:.4f}")


if __name__ == '__main__':
    main()
