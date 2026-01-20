#!/usr/bin/env python3
"""
Gift EVpred Metrics v2.0 - Usage Examples

展示如何使用新增的决策相关指标：
1. 概率校准（ECE + Reliability Curve）
2. 切片指标（冷启动/whale/tier）
3. 生态指标（Gini/Coverage/Overload）

Author: Viska Wei
Date: 2026-01-19
"""

import sys
import numpy as np
import pandas as pd

# 添加项目根目录
sys.path.insert(0, '/home/swei20/GiftLive')

from gift_EVpred.metrics import (
    evaluate_model,
    compute_calibration,
    compute_slice_metrics,
    compute_ecosystem_metrics,
    gini_coefficient,
)


# =============================================================================
# 1. 基础用法：evaluate_model() 一站式评估
# =============================================================================

def example_basic_usage():
    """
    基础用法：一站式评估

    最简单的用法，传入 y_true, y_pred 即可获得完整评估。
    """
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # 模拟数据
    np.random.seed(42)
    n = 10000

    # 构造 DataFrame
    df = pd.DataFrame({
        'user_id': np.random.randint(1, 500, n),
        'streamer_id': np.random.randint(1, 200, n),
        'timestamp': 1700000000000 + np.arange(n) * 1000,
        'pair_gift_count': np.random.choice([0, 1, 2, 5, 10], n, p=[0.4, 0.2, 0.2, 0.1, 0.1]),
        'streamer_gift_count': np.random.choice([0, 1, 5, 10, 50], n),
        'user_gift_sum': np.random.exponential(100, n),
        'streamer_gift_sum': np.random.exponential(500, n),
    })

    # 模拟真实标签（大部分为 0，少部分有打赏）
    y_true = np.zeros(n)
    gifter_mask = np.random.random(n) < 0.015  # 1.5% gift rate
    y_true[gifter_mask] = np.random.exponential(80, gifter_mask.sum())

    # 模拟预测分数
    y_pred = np.random.uniform(0, 1, n)

    # 模拟预测概率（可选）
    y_prob = 1 / (1 + np.exp(-y_pred * 6 + 3))

    # =========================================================================
    # 完整评估
    # =========================================================================
    result = evaluate_model(
        y_true=y_true,
        y_pred=y_pred,
        test_df=df,
        y_prob=y_prob,               # 传入概率则计算 ECE
        compute_slices=True,         # 计算切片指标
        compute_ecosystem=True,      # 计算生态指标
        whale_threshold=100,         # 自定义 whale 阈值
    )

    # 打印摘要
    print(result.summary())

    # 保存 JSON
    # result.to_json('gift_EVpred/results/eval_example.json')

    return result


# =============================================================================
# 2. 单独使用概率校准
# =============================================================================

def example_calibration():
    """
    单独计算概率校准（ECE + Reliability Curve）

    场景：评估分类模型的概率输出是否校准良好
    """
    print("\n" + "=" * 60)
    print("Example 2: Probability Calibration")
    print("=" * 60)

    np.random.seed(42)
    n = 5000

    # 模拟：真实正例率 2%
    y_true = np.zeros(n)
    y_true[np.random.choice(n, int(n * 0.02), replace=False)] = 1

    # 场景 A：较好校准的模型
    y_prob_good = np.random.beta(0.5, 24.5, n)  # 分布接近真实 2%

    # 场景 B：过度自信的模型
    y_prob_overconfident = np.random.beta(2, 10, n)  # 预测偏高

    print("\n--- 较好校准的模型 ---")
    calib_good = compute_calibration(y_true, y_prob_good, n_bins=10, strategy='uniform')
    print(f"ECE: {calib_good['ece']:.4f}")
    print(f"Positive Rate: {calib_good['meta']['positive_rate']:.2%}")

    print("\n--- 过度自信的模型 ---")
    calib_bad = compute_calibration(y_true, y_prob_overconfident, n_bins=10)
    print(f"ECE: {calib_bad['ece']:.4f}")
    print(f"Positive Rate: {calib_bad['meta']['positive_rate']:.2%}")

    # 查看 reliability curve
    print("\nReliability Curve (过度自信模型):")
    for b in calib_bad['bins'][:5]:
        print(f"  [{b['bin_lower']:.2f}, {b['bin_upper']:.2f}]: "
              f"avg_pred={b['avg_pred']:.3f}, avg_true={b['avg_true']:.3f}, "
              f"gap={b['gap']:+.3f}, n={b['n']}")


# =============================================================================
# 3. 单独使用切片指标
# =============================================================================

def example_slice_metrics():
    """
    单独计算切片指标

    场景：诊断模型在不同用户群体上的表现差异
    """
    print("\n" + "=" * 60)
    print("Example 3: Slice Metrics")
    print("=" * 60)

    np.random.seed(42)
    n = 8000

    df = pd.DataFrame({
        'user_id': np.random.randint(1, 300, n),
        'streamer_id': np.random.randint(1, 100, n),
        'pair_gift_count': np.random.choice([0, 0, 0, 1, 2, 5], n),  # 50% cold-start
        'streamer_gift_count': np.random.choice([0, 0, 1, 5, 10], n),
        'user_gift_sum': np.random.exponential(100, n),
        'streamer_gift_sum': np.random.exponential(500, n),
    })

    y_true = np.zeros(n)
    gifter_mask = np.random.random(n) < 0.02
    y_true[gifter_mask] = np.random.exponential(60, gifter_mask.sum())
    # 添加一些大额打赏
    y_true[y_true > 100] *= 2

    y_pred = np.random.uniform(0, 1, n)

    # 计算切片指标
    slices = compute_slice_metrics(
        y_true, y_pred, df,
        whale_threshold=100,
        min_slice_n=200,
        # 自定义列名（可选）
        pair_hist_col='pair_gift_count',
        user_value_col='user_gift_sum',
    )

    print("\n切片指标汇总:")
    for slice_name, data in slices.items():
        if slice_name == 'skipped':
            continue
        print(f"\n  {slice_name}:")
        print(f"    样本数: {data['n']:,}")
        print(f"    总收入: {data['total_revenue']:,.0f}")
        if 'revcap_1pct' in data:
            print(f"    RevCap@1%: {data['revcap_1pct']:.1%}")

    # 查看被跳过的切片
    if slices.get('skipped'):
        print("\n  被跳过的切片:")
        for name, reason in slices['skipped'].items():
            print(f"    {name}: {reason}")


# =============================================================================
# 4. 单独使用生态指标
# =============================================================================

def example_ecosystem_metrics():
    """
    单独计算生态指标

    场景：评估分配策略对主播生态的影响
    """
    print("\n" + "=" * 60)
    print("Example 4: Ecosystem Metrics")
    print("=" * 60)

    np.random.seed(42)
    n = 10000

    df = pd.DataFrame({
        'user_id': np.random.randint(1, 500, n),
        'streamer_id': np.random.randint(1, 200, n),
        'timestamp': 1700000000000 + np.random.randint(0, 3600000, n),
        'streamer_gift_count': np.random.choice([0, 0, 1, 5, 10, 50], n),
        'streamer_gift_sum': np.random.exponential(500, n),
        'user_gift_sum': np.random.exponential(100, n),
    })

    y_true = np.zeros(n)
    gifter_mask = np.random.random(n) < 0.015
    y_true[gifter_mask] = np.random.exponential(70, gifter_mask.sum())

    y_pred = np.random.uniform(0, 1, n)

    # 计算生态指标
    eco = compute_ecosystem_metrics(
        y_true, y_pred, df,
        k_select=0.01,               # Top 1% 作为选中集合
        tail_streamer_quantile=0.8,  # Bottom 80% 为长尾
        overload_window_minutes=10,
        overload_cap_per_window=3,
    )

    print("\n生态指标:")
    print(f"\n  选择集合:")
    print(f"    K: {eco['selection']['k_select']*100:.0f}%")
    print(f"    选中样本数: {eco['selection']['n_selected']:,}")

    print(f"\n  集中度 (Gini):")
    gini = eco['gini']
    gini_val = gini['streamer_revenue_gini']
    top10 = gini['top10_share']
    print(f"    Streamer Revenue Gini: {gini_val:.3f}" if not np.isnan(gini_val) else "    Streamer Revenue Gini: N/A")
    print(f"    Top 10% Share: {top10:.1%}" if not np.isnan(top10) else "    Top 10% Share: N/A")

    print(f"\n  覆盖率 (Coverage):")
    cov = eco['coverage']
    for k, v in cov.items():
        v_str = f"{v:.1%}" if not np.isnan(v) else "N/A"
        print(f"    {k}: {v_str}")

    print(f"\n  过载率 (Overload):")
    ov = eco['overload']
    print(f"    Window: {ov['window_minutes']} min, Cap: {ov['cap']}")
    ob_rate = ov['overload_bucket_rate']
    os_rate = ov['overloaded_streamer_rate']
    print(f"    Overload Bucket Rate: {ob_rate:.1%}" if not np.isnan(ob_rate) else "    Overload Bucket Rate: N/A")
    print(f"    Overloaded Streamer Rate: {os_rate:.1%}" if not np.isnan(os_rate) else "    Overloaded Streamer Rate: N/A")

    if eco['meta']['warnings']:
        print(f"\n  Warnings: {eco['meta']['warnings']}")


# =============================================================================
# 5. Gini 系数单独使用
# =============================================================================

def example_gini():
    """
    单独使用 Gini 系数

    场景：分析任意分布的集中度
    """
    print("\n" + "=" * 60)
    print("Example 5: Gini Coefficient")
    print("=" * 60)

    # 场景 1：完全均匀
    uniform = np.ones(100) * 10
    print(f"\n完全均匀分布: Gini = {gini_coefficient(uniform):.4f}")

    # 场景 2：指数分布（自然不平等）
    np.random.seed(42)
    exponential = np.random.exponential(100, 1000)
    print(f"指数分布: Gini = {gini_coefficient(exponential):.4f}")

    # 场景 3：帕累托分布（严重不平等）
    pareto = np.random.pareto(1.5, 1000) * 10
    print(f"帕累托分布: Gini = {gini_coefficient(pareto):.4f}")

    # 场景 4：极端集中
    extreme = np.zeros(100)
    extreme[0] = 1000
    print(f"极端集中（1人拥有全部）: Gini = {gini_coefficient(extreme):.4f}")


# =============================================================================
# 6. 自定义配置
# =============================================================================

def example_custom_config():
    """
    使用自定义配置

    展示如何通过 config 参数定制各模块行为
    """
    print("\n" + "=" * 60)
    print("Example 6: Custom Configuration")
    print("=" * 60)

    np.random.seed(42)
    n = 5000

    df = pd.DataFrame({
        'uid': np.random.randint(1, 200, n),  # 自定义列名
        'sid': np.random.randint(1, 100, n),
        'ts': 1700000000000 + np.arange(n) * 500,
        'pair_cnt': np.random.choice([0, 1, 2], n),
        'user_value': np.random.exponential(100, n),
        'streamer_value': np.random.exponential(500, n),
    })

    y_true = np.zeros(n)
    y_true[np.random.random(n) < 0.02] = np.random.exponential(50, (np.random.random(n) < 0.02).sum())
    y_pred = np.random.uniform(0, 1, n)
    y_prob = 1 / (1 + np.exp(-y_pred * 5))

    # 自定义配置
    result = evaluate_model(
        y_true, y_pred, df,
        y_prob=y_prob,

        # 校准配置
        calibration_config={
            'n_bins': 5,
            'strategy': 'quantile',
        },

        # 切片配置
        slice_config={
            'user_col': 'uid',
            'streamer_col': 'sid',
            'pair_hist_col': 'pair_cnt',
            'user_value_col': 'user_value',
            'streamer_value_col': 'streamer_value',
            'min_slice_n': 100,
        },

        # 生态配置
        ecosystem_config={
            'user_col': 'uid',
            'streamer_col': 'sid',
            'timestamp_col': 'ts',
            'user_value_col': 'user_value',
            'streamer_value_col': 'streamer_value',
            'k_select': 0.02,
            'overload_window_minutes': 5,
        },
    )

    print("\n自定义配置的评估结果:")
    print(f"  ECE: {result.prob_calibration.get('ece', 'N/A'):.4f}")
    print(f"  切片数: {len([k for k in result.slice_metrics.keys() if k != 'skipped'])}")
    print(f"  生态 Gini: {result.ecosystem.get('gini', {}).get('streamer_revenue_gini', 'N/A')}")


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Gift EVpred Metrics v2.0 - Usage Examples")
    print("=" * 60)

    # 运行所有示例
    example_basic_usage()
    example_calibration()
    example_slice_metrics()
    example_ecosystem_metrics()
    example_gini()
    example_custom_config()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
