#!/usr/bin/env python3
"""
Gift EVpred Metrics Module
==========================

统一的评估指标模块，所有 gift_EVpred 实验必须使用此模块。

指标体系设计：
1. 主指标（选模型/调参）：RevCap@K，以 K=1% 为主
2. 诊断指标：Whale Recall@K, Whale Precision@K, Avg Revenue@K, Tail Calibration
3. 稳定性指标：按天 CV, Bootstrap CI
4. 可选：NDCG@K（Top 内微调）

Usage:
    from gift_EVpred.metrics import (
        evaluate_model,           # 完整评估
        revenue_capture_at_k,     # 单指标
        whale_recall_at_k,
        compute_revcap_curve,
        EvalResult,               # 结果类
    )

    # 完整评估
    result = evaluate_model(y_true, y_pred, test_df, whale_threshold=100)
    print(result.summary())
    result.to_json('results.json')

Author: Viska Wei
Date: 2026-01-19
Version: 1.0
"""

import numpy as np
import pandas as pd
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# =============================================================================
# 常量配置
# =============================================================================

# 默认 K 值
DEFAULT_K_VALUES = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]
DEFAULT_K_LABELS = ['0.1%', '0.5%', '1%', '2%', '5%', '10%']

# Whale 默认阈值（P90 of gifters）
DEFAULT_WHALE_PERCENTILE = 90

# Calibration buckets
DEFAULT_CALIBRATION_BUCKETS = [0.001, 0.005, 0.01, 0.05]


# =============================================================================
# 核心指标函数
# =============================================================================

def revenue_capture_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.01) -> float:
    """
    Revenue Capture @K（主指标）

    计算 Top K% 预测捕获的真实收入比例。

    Args:
        y_true: 真实标签（原始金额）
        y_pred: 预测分数（用于排序）
        k: Top 比例，0.01 = 1%

    Returns:
        float: RevCap@K ∈ [0, 1]

    Formula:
        RevCap@K = Σ_{i ∈ Top K%} y_i / Σ_i y_i

    Example:
        >>> y_true = np.array([0, 100, 50, 200, 0])
        >>> y_pred = np.array([0.1, 0.9, 0.5, 0.8, 0.2])
        >>> revenue_capture_at_k(y_true, y_pred, k=0.4)  # Top 2 samples
        0.857  # (100 + 200) / 350
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_pred)[-n_top:]

    total_revenue = y_true.sum()
    if total_revenue == 0:
        return 0.0

    return float(y_true[top_indices].sum() / total_revenue)


def whale_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                       whale_threshold: float, k: float = 0.01) -> float:
    """
    Whale Recall @K（诊断指标）

    计算 Top K% 预测捕获了多少比例的真实 whale。

    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值（如 100 元）
        k: Top 比例

    Returns:
        float: Whale Recall ∈ [0, 1]

    Formula:
        Whale Recall@K = |Top K% ∩ Whales| / |Whales|
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_top = max(1, int(len(y_true) * k))
    top_indices = set(np.argsort(y_pred)[-n_top:])

    whale_indices = set(np.where(y_true >= whale_threshold)[0])
    if len(whale_indices) == 0:
        return np.nan

    captured_whales = len(top_indices & whale_indices)
    return float(captured_whales / len(whale_indices))


def whale_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                          whale_threshold: float, k: float = 0.01) -> float:
    """
    Whale Precision @K（诊断指标）

    计算 Top K% 预测中有多少比例是真实 whale。

    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值
        k: Top 比例

    Returns:
        float: Whale Precision ∈ [0, 1]

    Formula:
        Whale Precision@K = |Top K% ∩ Whales| / |Top K%|
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_pred)[-n_top:]

    whales_in_top_k = (y_true[top_indices] >= whale_threshold).sum()
    return float(whales_in_top_k / n_top)


def avg_revenue_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.01) -> float:
    """
    Average Revenue per Selected @K（诊断指标）

    计算 Top K% 样本的人均真实收入，用于评估"池子质量"。

    Args:
        y_true: 真实标签
        y_pred: 预测分数
        k: Top 比例

    Returns:
        float: 人均收入（元）

    Formula:
        AvgRev@K = Σ_{i ∈ Top K%} y_i / |Top K%|
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_pred)[-n_top:]

    return float(y_true[top_indices].mean())


def gift_rate_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.01) -> float:
    """
    Gift Rate @K

    计算 Top K% 样本中有打赏的比例。

    Args:
        y_true: 真实标签
        y_pred: 预测分数
        k: Top 比例

    Returns:
        float: Gift Rate ∈ [0, 1]
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_top = max(1, int(len(y_true) * k))
    top_indices = np.argsort(y_pred)[-n_top:]

    return float((y_true[top_indices] > 0).mean())


def tail_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                      buckets: List[float] = None) -> Dict[str, Dict[str, float]]:
    """
    Tail Calibration（诊断指标）

    按预测分数分桶，计算预测金额 / 实际金额比值。
    目标是接近 1.0，避免分配时高估/低估。

    Args:
        y_true: 真实标签
        y_pred: 预测分数
        buckets: 分桶比例列表，如 [0.001, 0.005, 0.01, 0.05]

    Returns:
        dict: {bucket_label: {'sum_ratio': float, 'mean_ratio': float}}

    Formula:
        Sum Ratio = Σ_{i ∈ Top K%} pred_i / Σ_{i ∈ Top K%} y_i
        Mean Ratio = Mean(pred[Top K%]) / Mean(y[Top K%])
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if buckets is None:
        buckets = DEFAULT_CALIBRATION_BUCKETS

    results = {}
    bucket_labels = [f'top_{k*100:.1f}%' for k in buckets]

    for k, label in zip(buckets, bucket_labels):
        n_top = max(1, int(len(y_true) * k))
        top_indices = np.argsort(y_pred)[-n_top:]

        pred_sum = y_pred[top_indices].sum()
        true_sum = y_true[top_indices].sum()
        pred_mean = y_pred[top_indices].mean()
        true_mean = y_true[top_indices].mean()

        results[label] = {
            'sum_ratio': float(pred_sum / true_sum) if true_sum > 0 else np.nan,
            'mean_ratio': float(pred_mean / true_mean) if true_mean > 0 else np.nan,
        }

    return results


def normalized_dcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int = 100) -> float:
    """
    NDCG @K（可选，用于 Top 内微调）

    计算归一化折损累计增益，用于评估 Top K 内部的排序质量。

    Args:
        y_true: 真实标签（作为 relevance）
        y_pred: 预测分数
        k: Top K 个样本

    Returns:
        float: NDCG@K ∈ [0, 1]

    Note:
        - 适用于需要精细排序的场景
        - 主指标仍用 RevCap@K
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # 按预测排序
    pred_order = np.argsort(y_pred)[::-1][:k]
    # 按真实排序（ideal）
    ideal_order = np.argsort(y_true)[::-1][:k]

    # DCG
    dcg = 0.0
    for i, idx in enumerate(pred_order):
        rel = y_true[idx]
        dcg += rel / np.log2(i + 2)

    # IDCG
    idcg = 0.0
    for i, idx in enumerate(ideal_order):
        rel = y_true[idx]
        idcg += rel / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return float(dcg / idcg)


# =============================================================================
# 曲线计算
# =============================================================================

def compute_revcap_curve(y_true: np.ndarray, y_pred: np.ndarray,
                          k_values: List[float] = None) -> Dict[str, float]:
    """
    计算 RevCap 曲线（多 K 值）

    Args:
        y_true: 真实标签
        y_pred: 预测分数
        k_values: K 值列表

    Returns:
        dict: {k_label: revcap_value}
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    results = {}
    for k in k_values:
        label = f'{k*100:.1f}%' if k < 0.01 else f'{k*100:.0f}%'
        results[label] = revenue_capture_at_k(y_true, y_pred, k)

    return results


def compute_oracle_revcap_curve(y_true: np.ndarray,
                                 k_values: List[float] = None) -> Dict[str, float]:
    """
    计算 Oracle RevCap 曲线（理论上限）

    Oracle = 按真实 y 排序的理论最优。
    """
    return compute_revcap_curve(y_true, y_true, k_values)


def compute_all_metrics_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                              whale_threshold: float, k: float = 0.01) -> Dict[str, float]:
    """
    计算单个 K 值下的所有指标

    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值
        k: Top 比例

    Returns:
        dict: 所有指标的字典
    """
    return {
        'revcap': revenue_capture_at_k(y_true, y_pred, k),
        'whale_recall': whale_recall_at_k(y_true, y_pred, whale_threshold, k),
        'whale_precision': whale_precision_at_k(y_true, y_pred, whale_threshold, k),
        'avg_revenue': avg_revenue_at_k(y_true, y_pred, k),
        'gift_rate': gift_rate_at_k(y_true, y_pred, k),
    }


# =============================================================================
# 稳定性评估
# =============================================================================

def compute_stability_by_day(test_df: pd.DataFrame, y_true: np.ndarray,
                              y_pred: np.ndarray, whale_threshold: float,
                              k: float = 0.01,
                              timestamp_col: str = 'timestamp') -> pd.DataFrame:
    """
    按天计算所有指标，评估稳定性

    Args:
        test_df: 测试数据 DataFrame（需要 timestamp 列）
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值
        k: Top 比例
        timestamp_col: 时间戳列名

    Returns:
        DataFrame: 每天的指标
    """
    test_df = test_df.copy()
    test_df['_date'] = pd.to_datetime(test_df[timestamp_col], unit='ms').dt.date

    days = sorted(test_df['_date'].unique())
    metrics_by_day = []

    for day in days:
        mask = (test_df['_date'] == day).values
        if mask.sum() < 100:
            continue

        y_day = y_true[mask]
        pred_day = y_pred[mask]

        if y_day.sum() == 0:
            continue

        metrics = compute_all_metrics_at_k(y_day, pred_day, whale_threshold, k)
        metrics['date'] = day
        metrics['n_samples'] = int(mask.sum())
        metrics['n_whales'] = int((y_day >= whale_threshold).sum())
        metrics['total_revenue'] = float(y_day.sum())
        metrics['max_single'] = float(y_day.max())
        metrics_by_day.append(metrics)

    return pd.DataFrame(metrics_by_day)


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 1000,
                  ci: float = 0.95) -> Tuple[float, float, float]:
    """
    Bootstrap 计算置信区间

    Args:
        values: 数值数组
        n_bootstrap: bootstrap 次数
        ci: 置信水平

    Returns:
        tuple: (mean, ci_lower, ci_upper)
    """
    values = np.array([v for v in values if not np.isnan(v)])
    if len(values) < 2:
        return np.nan, np.nan, np.nan

    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return float(np.mean(values)), float(lower), float(upper)


def compute_stability_summary(daily_df: pd.DataFrame,
                               metric_col: str = 'revcap') -> Dict[str, float]:
    """
    计算稳定性汇总指标

    Args:
        daily_df: 每天指标的 DataFrame
        metric_col: 要汇总的指标列名

    Returns:
        dict: 稳定性指标
    """
    values = daily_df[metric_col].values
    mean_val, ci_lower, ci_upper = bootstrap_ci(values)
    std_val = np.nanstd(values)
    cv = std_val / mean_val if mean_val > 0 else np.nan

    return {
        'mean': mean_val,
        'std': std_val,
        'cv': cv,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'min': float(np.nanmin(values)),
        'max': float(np.nanmax(values)),
        'min_date': str(daily_df.loc[daily_df[metric_col].idxmin(), 'date']),
        'max_date': str(daily_df.loc[daily_df[metric_col].idxmax(), 'date']),
    }


# =============================================================================
# 评估结果类
# =============================================================================

@dataclass
class EvalResult:
    """
    评估结果类

    包含所有评估指标，支持格式化输出和 JSON 导出。
    """
    # 基础信息
    n_samples: int
    n_gifters: int
    n_whales: int
    whale_threshold: float
    gift_rate: float
    whale_rate: float
    total_revenue: float

    # 主指标
    revcap_1pct: float
    revcap_curve: Dict[str, float] = field(default_factory=dict)
    oracle_curve: Dict[str, float] = field(default_factory=dict)
    normalized_curve: Dict[str, float] = field(default_factory=dict)

    # 诊断指标
    whale_recall_1pct: float = 0.0
    whale_precision_1pct: float = 0.0
    avg_revenue_1pct: float = 0.0
    gift_rate_1pct: float = 0.0
    calibration: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 全 K 值指标
    metrics_by_k: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # 稳定性指标
    stability: Dict[str, float] = field(default_factory=dict)
    daily_metrics: Optional[pd.DataFrame] = None

    # 可选指标
    ndcg_100: Optional[float] = None
    spearman: Optional[float] = None

    def summary(self) -> str:
        """生成摘要字符串"""
        lines = [
            "=" * 60,
            "Evaluation Summary",
            "=" * 60,
            f"Samples: {self.n_samples:,} | Gifters: {self.n_gifters:,} ({self.gift_rate:.2%})",
            f"Whales: {self.n_whales:,} ({self.whale_rate:.3%}) | Threshold: {self.whale_threshold:.0f} yuan",
            f"Total Revenue: {self.total_revenue:,.0f} yuan",
            "",
            "--- Main Metrics (RevCap@K) ---",
        ]

        for k, v in self.revcap_curve.items():
            oracle_v = self.oracle_curve.get(k, 1.0)
            norm_v = v / oracle_v if oracle_v > 0 else 0
            lines.append(f"  @{k}: {v:.1%} (Norm: {norm_v:.1%})")

        lines.extend([
            "",
            "--- Diagnostic Metrics (@1%) ---",
            f"  Whale Recall: {self.whale_recall_1pct:.1%}",
            f"  Whale Precision: {self.whale_precision_1pct:.1%}",
            f"  Avg Revenue: {self.avg_revenue_1pct:.1f} yuan",
            f"  Gift Rate: {self.gift_rate_1pct:.1%}",
        ])

        if self.calibration:
            lines.append("\n--- Tail Calibration ---")
            for bucket, ratios in self.calibration.items():
                lines.append(f"  {bucket}: Sum={ratios['sum_ratio']:.3f}, Mean={ratios['mean_ratio']:.3f}")

        if self.stability:
            lines.extend([
                "",
                "--- Stability (@1%) ---",
                f"  Mean: {self.stability.get('mean', 0):.1%}",
                f"  Std: {self.stability.get('std', 0):.1%}",
                f"  CV: {self.stability.get('cv', 0):.1%}",
                f"  95% CI: [{self.stability.get('ci_lower', 0):.1%}, {self.stability.get('ci_upper', 0):.1%}]",
            ])

        if self.ndcg_100 is not None:
            lines.append(f"\n--- Optional ---\n  NDCG@100: {self.ndcg_100:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """转换为字典（用于 JSON 序列化）"""
        d = asdict(self)
        # 移除 DataFrame（不可序列化）
        d.pop('daily_metrics', None)
        return d

    def to_json(self, path: Union[str, Path]) -> None:
        """保存为 JSON 文件"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'EvalResult':
        """从 JSON 文件加载"""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls(**d)


# =============================================================================
# 主评估函数
# =============================================================================

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray,
                    test_df: Optional[pd.DataFrame] = None,
                    whale_threshold: Optional[float] = None,
                    whale_percentile: int = DEFAULT_WHALE_PERCENTILE,
                    k_values: List[float] = None,
                    compute_stability: bool = True,
                    compute_ndcg: bool = False,
                    timestamp_col: str = 'timestamp') -> EvalResult:
    """
    完整模型评估（主入口函数）

    计算所有指标并返回 EvalResult 对象。

    Args:
        y_true: 真实标签（原始金额）
        y_pred: 预测分数
        test_df: 测试数据 DataFrame（用于按天稳定性，可选）
        whale_threshold: whale 定义阈值，若为 None 则自动计算 P90
        whale_percentile: 自动计算 whale 阈值时的百分位数
        k_values: K 值列表
        compute_stability: 是否计算按天稳定性
        compute_ndcg: 是否计算 NDCG
        timestamp_col: 时间戳列名

    Returns:
        EvalResult: 完整评估结果

    Example:
        >>> from gift_EVpred.metrics import evaluate_model
        >>> result = evaluate_model(y_test, y_pred, test_df)
        >>> print(result.summary())
        >>> result.to_json('results.json')
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if k_values is None:
        k_values = DEFAULT_K_VALUES

    # 基础统计
    n_samples = len(y_true)
    n_gifters = int((y_true > 0).sum())
    gift_rate = n_gifters / n_samples
    total_revenue = float(y_true.sum())

    # Whale 阈值
    if whale_threshold is None:
        gifters_y = y_true[y_true > 0]
        if len(gifters_y) > 0:
            whale_threshold = float(np.percentile(gifters_y, whale_percentile))
        else:
            whale_threshold = 0.0

    n_whales = int((y_true >= whale_threshold).sum())
    whale_rate = n_whales / n_samples

    # 主指标：RevCap 曲线
    revcap_curve = compute_revcap_curve(y_true, y_pred, k_values)
    oracle_curve = compute_oracle_revcap_curve(y_true, k_values)
    normalized_curve = {
        k: revcap_curve[k] / oracle_curve[k] if oracle_curve[k] > 0 else 0
        for k in revcap_curve
    }

    # @1% 主指标
    revcap_1pct = revenue_capture_at_k(y_true, y_pred, 0.01)

    # 诊断指标 @1%
    whale_recall_1pct = whale_recall_at_k(y_true, y_pred, whale_threshold, 0.01)
    whale_precision_1pct = whale_precision_at_k(y_true, y_pred, whale_threshold, 0.01)
    avg_revenue_1pct = avg_revenue_at_k(y_true, y_pred, 0.01)
    gift_rate_1pct = gift_rate_at_k(y_true, y_pred, 0.01)

    # Tail Calibration
    calibration = tail_calibration(y_true, y_pred)

    # 全 K 值指标
    metrics_by_k = {}
    for k in k_values:
        label = f'{k*100:.1f}%' if k < 0.01 else f'{k*100:.0f}%'
        metrics_by_k[label] = compute_all_metrics_at_k(y_true, y_pred, whale_threshold, k)
        metrics_by_k[label]['oracle_revcap'] = oracle_curve[label]
        metrics_by_k[label]['normalized_revcap'] = normalized_curve[label]

    # 稳定性
    stability = {}
    daily_metrics = None
    if compute_stability and test_df is not None:
        daily_metrics = compute_stability_by_day(
            test_df, y_true, y_pred, whale_threshold, k=0.01, timestamp_col=timestamp_col
        )
        if len(daily_metrics) > 0:
            stability = compute_stability_summary(daily_metrics, 'revcap')

    # 可选指标
    ndcg_100 = None
    if compute_ndcg:
        ndcg_100 = normalized_dcg_at_k(y_true, y_pred, k=100)

    # Spearman（可选）
    try:
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(y_true, y_pred)
        spearman = float(spearman)
    except:
        spearman = None

    return EvalResult(
        n_samples=n_samples,
        n_gifters=n_gifters,
        n_whales=n_whales,
        whale_threshold=whale_threshold,
        gift_rate=gift_rate,
        whale_rate=whale_rate,
        total_revenue=total_revenue,
        revcap_1pct=revcap_1pct,
        revcap_curve=revcap_curve,
        oracle_curve=oracle_curve,
        normalized_curve=normalized_curve,
        whale_recall_1pct=whale_recall_1pct,
        whale_precision_1pct=whale_precision_1pct,
        avg_revenue_1pct=avg_revenue_1pct,
        gift_rate_1pct=gift_rate_1pct,
        calibration=calibration,
        metrics_by_k=metrics_by_k,
        stability=stability,
        daily_metrics=daily_metrics,
        ndcg_100=ndcg_100,
        spearman=spearman,
    )


def quick_eval(y_true: np.ndarray, y_pred: np.ndarray,
               whale_threshold: float = 100, k: float = 0.01) -> Dict[str, float]:
    """
    快速评估（轻量版）

    只计算核心指标，用于训练过程中的快速评估。

    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 阈值
        k: Top 比例

    Returns:
        dict: 核心指标
    """
    return {
        'revcap': revenue_capture_at_k(y_true, y_pred, k),
        'whale_recall': whale_recall_at_k(y_true, y_pred, whale_threshold, k),
        'whale_precision': whale_precision_at_k(y_true, y_pred, whale_threshold, k),
        'avg_revenue': avg_revenue_at_k(y_true, y_pred, k),
    }


# =============================================================================
# 辅助函数
# =============================================================================

def get_whale_threshold(y_true: np.ndarray, percentile: int = 90) -> float:
    """
    计算 whale 阈值

    Args:
        y_true: 真实标签
        percentile: 百分位数（基于 gifters）

    Returns:
        float: whale 阈值
    """
    gifters_y = y_true[y_true > 0]
    if len(gifters_y) == 0:
        return 0.0
    return float(np.percentile(gifters_y, percentile))


def format_metrics_table(result: EvalResult, k_values: List[str] = None) -> str:
    """
    格式化为 Markdown 表格

    Args:
        result: EvalResult 对象
        k_values: 要显示的 K 值列表

    Returns:
        str: Markdown 表格
    """
    if k_values is None:
        k_values = list(result.metrics_by_k.keys())

    lines = [
        "| K | RevCap | Norm | Whale Recall | Whale Prec | Avg Rev |",
        "|---|--------|------|--------------|------------|---------|",
    ]

    for k in k_values:
        m = result.metrics_by_k.get(k, {})
        lines.append(
            f"| {k} | {m.get('revcap', 0):.1%} | {m.get('normalized_revcap', 0):.1%} | "
            f"{m.get('whale_recall', 0):.1%} | {m.get('whale_precision', 0):.1%} | "
            f"{m.get('avg_revenue', 0):.1f} |"
        )

    return "\n".join(lines)


# =============================================================================
# CLI 入口
# =============================================================================

if __name__ == '__main__':
    print("Gift EVpred Metrics Module")
    print("=" * 60)
    print("Usage:")
    print("  from gift_EVpred.metrics import evaluate_model")
    print("  result = evaluate_model(y_true, y_pred, test_df)")
    print("  print(result.summary())")
