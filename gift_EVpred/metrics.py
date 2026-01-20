#!/usr/bin/env python3
"""
Gift EVpred Metrics Module
==========================

统一的评估指标模块，所有 gift_EVpred 实验必须使用此模块。

指标体系设计（v2.1 - 决策相关指标完整版）：

1. 主指标（选模型/调参）：
   - RevCap@K：Top K% 预测集合的真实收入占比，对应决策中"你选中的人贡献了多少实际收益"

2. 诊断指标：
   - Whale Recall/Precision@K, Avg Revenue@K, Gift Rate@K
   - Tail Calibration（金额/EV 校准）

3. 稳定性指标：
   - 按天 CV, Bootstrap CI

4. 校准指标：
   - Probability Calibration (ECE + Reliability Curve)：是否打赏概率校准
   - Value Calibration：金额/EV 的 sum_ratio（现有 tail_calibration）

5. 切片指标：
   - 冷启动切片：cold_start_pair, cold_start_streamer
   - Whale 切片：whale_true, non_whale_true
   - 分层切片：user_top_1pct, user_top_10pct, streamer_top_1pct 等
   用于回答"冷启动、whale、头/尾用户是否被系统性损害"

6. 生态指标（分配层护栏）：
   - Gini：主播收入集中度
   - Coverage：主播覆盖率、长尾覆盖率、冷启动覆盖率
   - Overload：高价值用户过载风险

7. **[NEW v2.1] 决策核心指标**：
   - Misallocation Cost：错分代价（Oracle@K - Achieved@K）
   - Capture Curve AUC：避免只盯 1% 导致 5%/10% 崩
   - PSI/Drift：训练/测试预测分布漂移
   - Diversity/Entropy：分配多样性（比 Gini 更灵敏）

Usage:
    from gift_EVpred.metrics import (
        evaluate_model,           # 完整评估
        revenue_capture_at_k,     # 单指标
        whale_recall_at_k,
        compute_revcap_curve,
        compute_calibration,      # 概率校准
        compute_slice_metrics,    # 切片指标
        compute_ecosystem_metrics,# 生态指标
        gini_coefficient,         # Gini 系数
        # [NEW v2.1] 决策核心指标
        compute_misallocation_cost,  # 错分代价
        compute_capture_auc,         # 收益曲线 AUC
        compute_drift,               # PSI/分布漂移
        compute_diversity,           # 多样性/熵
        EvalResult,               # 结果类
    )

    # 完整评估（含新增指标）
    result = evaluate_model(
        y_true, y_pred, test_df,
        y_prob=y_prob,           # 若提供则计算概率校准
        compute_slices=True,
        compute_ecosystem=True,
    )
    print(result.summary())
    result.to_json('results.json')

Author: Viska Wei
Date: 2026-01-19
Version: 2.2 (Calibration & Naming Revision)

v2.2 Changes:
    - Regret 命名（原 Opportunity Cost）
    - 金额版/比例版指标拆分（AchievedRev, OracleRev, Regret, Eff, RegretPct）
    - nAUC 规范（明确积分范围，推荐使用归一化版本）
    - Overload 拆分为用户侧/主播侧（U-OverTarget, S-OverLoad）
    - Tail Calibration 添加 EV 刻度前提说明
    - 工业界风格短名映射（METRIC_SHORT_NAMES）
"""

import numpy as np
import pandas as pd
import json
import warnings
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
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

# [NEW] 概率校准默认 bins
DEFAULT_PROB_CALIBRATION_BINS = 10

# [NEW] 切片最小样本数
DEFAULT_MIN_SLICE_N = 500

# [NEW] 生态指标默认配置
DEFAULT_TAIL_STREAMER_QUANTILE = 0.8
DEFAULT_OVERLOAD_WINDOW_MINUTES = 10
DEFAULT_OVERLOAD_CAP_PER_WINDOW = 3
DEFAULT_HIGH_VALUE_USER_QUANTILE = 0.99

# =============================================================================
# [v2.2] 工业界风格短名映射
# =============================================================================
# 命名规则：
#   - @K: Top 比例（如 @1% = k=0.01）
#   - @W: 时间窗口（如 @10m = 10分钟）
#   - n 前缀: 归一化版本（normalized）
#   - $ 后缀: 金额版（代码中用 _amount 或 _rev）
#
# 主指标:
#   RevCap@K        = revenue_capture_at_k()
#   OracleRevCap@K  = oracle 版本（按真实排序）
#   nRevCap@K       = RevCap@K / OracleRevCap@K（归一化）
#
# 决策核心（v2.2 新命名）:
#   AchievedRev@K   = 实际捕获收入（元）
#   OracleRev@K     = Oracle 收入（元）
#   Regret@K         = OracleRev - AchievedRev（元，原 OpportunityCost）
#   RegretPct@K     = 1 - Eff@K（比例，原 OpportunityCost%）
#   Eff@K           = AchievedRev / OracleRev（效率，≈ nRevCap）
#   nAUC@K          = CapAUC / OracleAUC（归一化 Capture AUC）
#   MeanRevCap@K    = CapAUC / K（平均 RevCap）
#
# 诊断:
#   WRec@K          = Whale Recall@K
#   WPrec@K         = Whale Precision@K
#   AvgRev@K        = avg_revenue_at_k()
#   GiftRate@K      = gift_rate_at_k()
#   RevLift@K       = RevCap@K / K（相对 random 提升倍数）
#
# 稳定性:
#   CV@K            = 变异系数
#   CI95@K          = 95% Bootstrap CI
#
# 校准:
#   pECE            = 概率 ECE（probability calibration）
#   SumRatio@K      = tail_calibration 的 sum_ratio
#   MeanRatio@K     = tail_calibration 的 mean_ratio
#
# 生态/多样性:
#   GiniS@K         = streamer revenue gini
#   CovS@K          = streamer coverage
#   TailCovS@K      = tail streamer coverage
#   ColdCovS@K      = cold streamer coverage
#   Ent@K           = Shannon entropy
#   EffN@K          = effective number = exp(entropy)
#   HHI@K           = Herfindahl-Hirschman Index
#   Top10Share@K    = Top 10% 分组份额
#
# Overload（v2.2 拆分为两类）:
#   U-OverTarget@W  = 用户侧过度触达（user_overtarget_rate）
#   S-OverLoad@W    = 主播侧承接过载（streamer_overload_rate）

METRIC_SHORT_NAMES = {
    # 主指标
    'revenue_capture': 'RevCap@K',
    'oracle_revcap': 'OracleRevCap@K',
    'normalized_revcap': 'nRevCap@K',
    # 决策核心
    'achieved_rev': 'AchievedRev@K',
    'oracle_rev': 'OracleRev@K',
    'regret': 'Regret@K',
    'regret_pct': 'RegretPct@K',
    'eff': 'Eff@K',
    'nAUC': 'nAUC@K',
    'mean_revcap': 'MeanRevCap@K',
    # 诊断
    'whale_recall': 'WRec@K',
    'whale_precision': 'WPrec@K',
    'avg_revenue': 'AvgRev@K',
    'gift_rate': 'GiftRate@K',
    # 稳定性
    'cv': 'CV@K',
    'ci_95': 'CI95@K',
    # 校准
    'ece': 'pECE',
    'sum_ratio': 'SumRatio@K',
    'mean_ratio': 'MeanRatio@K',
    # 生态
    'streamer_revenue_gini': 'GiniS@K',
    'streamer_coverage': 'CovS@K',
    'tail_coverage': 'TailCovS@K',
    'cold_start_streamer_coverage': 'ColdCovS@K',
    'entropy': 'Ent@K',
    'effective_number': 'EffN@K',
    'hhi': 'HHI@K',
    'top10_share': 'Top10Share@K',
    # Overload
    'user_overtarget_rate': 'U-OverTarget@W',
    'streamer_overload_rate': 'S-OverLoad@W',
    # PSI
    'psi': 'PSI',
}


# =============================================================================
# 核心指标函数（保持不变）
# =============================================================================

def revenue_capture_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: float = 0.01) -> float:
    """
    Revenue Capture @K（主指标）

    计算 Top K% 预测捕获的真实收入比例。
    对应决策中"你选中的人贡献了多少实际收益"（也可称 RevShare@K）。

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


def whale_recall_lift_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                            whale_threshold: float, k: float = 0.01) -> float:
    """
    Whale Recall Lift @K（相对随机提升倍数）
    
    WRecLift@K = WRec@K / K
    
    随机选择 TopK 时，期望 WRec ≈ K。Lift 消除 K 值影响，更稳健、更可比。
    
    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值
        k: Top 比例
        
    Returns:
        float: Lift 倍数（如 33.5 表示是随机的 33.5 倍）
    """
    wrec = whale_recall_at_k(y_true, y_pred, whale_threshold, k)
    if np.isnan(wrec) or k == 0:
        return np.nan
    return float(wrec / k)


def whale_precision_lift_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                               whale_threshold: float, k: float = 0.01) -> float:
    """
    Whale Precision Lift @K（纯度相对随机提升倍数）
    
    WPrecLift@K = WPrec@K / base_whale_rate
    
    其中 base_whale_rate = P(y >= T)，即全体样本中 whale 的比例。
    
    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值
        k: Top 比例
        
    Returns:
        float: Lift 倍数
    """
    y_true = np.asarray(y_true)
    wprec = whale_precision_at_k(y_true, y_pred, whale_threshold, k)
    
    # 全体 whale rate
    base_whale_rate = (y_true >= whale_threshold).mean()
    if base_whale_rate == 0:
        return np.nan
    return float(wprec / base_whale_rate)


def whale_revenue_capture_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                                whale_threshold: float, k: float = 0.01) -> float:
    """
    Whale Revenue Capture @K（金额口径）
    
    "所有 whale 的钱里，有多少被我 TopK 抓住了？"
    
    比 WRec（个数口径）更贴近"抓住大额营收"，
    对"一个 whale 反复大额 vs 很多小 whale"更稳健。
    
    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值
        k: Top 比例
        
    Returns:
        float: WRevCap ∈ [0, 1]
        
    Formula:
        WRevCap@K = Σ(y_i where i ∈ TopK ∩ Whales) / Σ(y_i where i ∈ Whales)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n_top = max(1, int(len(y_true) * k))
    top_indices = set(np.argsort(y_pred)[-n_top:])
    
    whale_mask = y_true >= whale_threshold
    whale_total_rev = y_true[whale_mask].sum()
    if whale_total_rev == 0:
        return np.nan
    
    # TopK 中捕获的 whale 收入
    captured_whale_rev = sum(
        y_true[i] for i in top_indices 
        if y_true[i] >= whale_threshold
    )
    return float(captured_whale_rev / whale_total_rev)


def oracle_whale_recall_at_k(y_true: np.ndarray, whale_threshold: float, 
                              k: float = 0.01) -> float:
    """
    Oracle Whale Recall @K（理论上限）
    
    用真实 y 排序选 TopK 时的 whale recall。
    
    Args:
        y_true: 真实标签
        whale_threshold: whale 定义阈值
        k: Top 比例
        
    Returns:
        float: OracleWRec ∈ [0, 1]
    """
    return whale_recall_at_k(y_true, y_true, whale_threshold, k)


def normalized_whale_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray,
                                  whale_threshold: float, k: float = 0.01) -> float:
    """
    Normalized Whale Recall @K（相对 Oracle 上限）
    
    nWRec@K = WRec@K / OracleWRec@K
    
    告诉你："在当前 K 下，你离能做到的最好还有多远"。
    
    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值
        k: Top 比例
        
    Returns:
        float: nWRec ∈ [0, 1]
    """
    wrec = whale_recall_at_k(y_true, y_pred, whale_threshold, k)
    oracle_wrec = oracle_whale_recall_at_k(y_true, whale_threshold, k)
    
    if np.isnan(wrec) or np.isnan(oracle_wrec) or oracle_wrec == 0:
        return np.nan
    return float(wrec / oracle_wrec)


def compute_whale_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          whale_threshold: float, k: float = 0.01) -> Dict[str, float]:
    """
    计算完整的 Whale 指标套件（6 件套）
    
    Args:
        y_true: 真实标签
        y_pred: 预测分数
        whale_threshold: whale 定义阈值
        k: Top 比例
        
    Returns:
        Dict with keys:
            - wrec: Whale Recall（覆盖）
            - wprec: Whale Precision（纯度）
            - wrec_lift: Whale Recall Lift（相对随机）
            - wprec_lift: Whale Precision Lift（纯度相对随机）
            - wrevcap: Whale Revenue Capture（金额口径）
            - oracle_wrec: Oracle Whale Recall（上限）
            - nwrec: Normalized Whale Recall（相对 Oracle）
            - base_whale_rate: 全体 whale 比例
    """
    y_true = np.asarray(y_true)
    
    wrec = whale_recall_at_k(y_true, y_pred, whale_threshold, k)
    wprec = whale_precision_at_k(y_true, y_pred, whale_threshold, k)
    wrec_lift = whale_recall_lift_at_k(y_true, y_pred, whale_threshold, k)
    wprec_lift = whale_precision_lift_at_k(y_true, y_pred, whale_threshold, k)
    wrevcap = whale_revenue_capture_at_k(y_true, y_pred, whale_threshold, k)
    oracle_wrec = oracle_whale_recall_at_k(y_true, whale_threshold, k)
    nwrec = normalized_whale_recall_at_k(y_true, y_pred, whale_threshold, k)
    base_whale_rate = float((y_true >= whale_threshold).mean())
    
    return {
        'wrec': wrec,
        'wprec': wprec,
        'wrec_lift': wrec_lift,
        'wprec_lift': wprec_lift,
        'wrevcap': wrevcap,
        'oracle_wrec': oracle_wrec,
        'nwrec': nwrec,
        'base_whale_rate': base_whale_rate,
    }


# =============================================================================
# User-Level Whale Metrics（找大哥名单）
# =============================================================================

def compute_user_level_whale_metrics(
    df: pd.DataFrame,
    y_true_col: str = 'target_raw',
    y_pred_col: str = 'y_pred',
    user_col: str = 'user_id',
    whale_threshold: float = 100,
    k: float = 0.01,
    user_score_agg: str = 'sum',
    whale_def: str = 'cumulative',
) -> Dict[str, Any]:
    """
    User-Level Whale 指标（找大哥名单）
    
    与 sample-level 不同，这里回答的是"能否找出大哥用户"。
    
    Args:
        df: 数据 DataFrame，需包含 user_col, y_true_col
        y_true_col: 真实金额列名
        y_pred_col: 预测分数列名（需提前添加到 df）
        user_col: 用户 ID 列名
        whale_threshold: Whale 定义阈值
        k: Top 比例（用户级）
        user_score_agg: 用户分数聚合方式 ('sum', 'max', 'mean')
        whale_def: Whale 用户定义 ('cumulative' = Σy≥T, 'single' = max(y)≥T)
        
    Returns:
        Dict with keys:
            - n_users: 总用户数
            - n_whale_users: Whale 用户数
            - base_whale_user_rate: 用户中大哥比例
            - whale_user_rec: Whale User Recall
            - whale_user_prec: Whale User Precision
            - whale_user_rec_lift: Recall Lift
            - user_revcap: User-level Revenue Capture
            - whale_def: Whale 定义方式
            - user_score_agg: 聚合方式
    """
    df = df.copy()
    
    # 1. 按用户聚合
    user_agg = df.groupby(user_col).agg({
        y_true_col: ['sum', 'max', 'count'],
        y_pred_col: [user_score_agg],
    })
    user_agg.columns = ['y_sum', 'y_max', 'n_samples', 'score']
    user_agg = user_agg.reset_index()
    
    n_users = len(user_agg)
    total_revenue = user_agg['y_sum'].sum()
    
    # 2. 定义 Whale 用户
    if whale_def == 'cumulative':
        user_agg['is_whale'] = user_agg['y_sum'] >= whale_threshold
    elif whale_def == 'single':
        user_agg['is_whale'] = user_agg['y_max'] >= whale_threshold
    else:
        raise ValueError(f"Unknown whale_def: {whale_def}")
    
    n_whale_users = user_agg['is_whale'].sum()
    base_whale_user_rate = n_whale_users / n_users if n_users > 0 else 0
    
    # 3. 选择 TopK 用户
    n_top_users = max(1, int(n_users * k))
    top_user_indices = user_agg.nlargest(n_top_users, 'score').index
    
    # 4. 计算指标
    top_users = user_agg.loc[top_user_indices]
    
    # Whale User Recall: 大哥用户中有多少被选中
    whale_in_top = top_users['is_whale'].sum()
    whale_user_rec = whale_in_top / n_whale_users if n_whale_users > 0 else np.nan
    
    # Whale User Precision: 选出的用户里有多少是大哥
    whale_user_prec = whale_in_top / n_top_users if n_top_users > 0 else np.nan
    
    # Recall Lift
    whale_user_rec_lift = whale_user_rec / k if not np.isnan(whale_user_rec) and k > 0 else np.nan
    
    # User Revenue Capture: TopK 用户贡献多少总收入
    top_revenue = top_users['y_sum'].sum()
    user_revcap = top_revenue / total_revenue if total_revenue > 0 else np.nan
    
    # Oracle User RevCap
    oracle_top_users = user_agg.nlargest(n_top_users, 'y_sum')
    oracle_revenue = oracle_top_users['y_sum'].sum()
    oracle_user_revcap = oracle_revenue / total_revenue if total_revenue > 0 else np.nan
    
    return {
        'n_users': n_users,
        'n_whale_users': int(n_whale_users),
        'base_whale_user_rate': float(base_whale_user_rate),
        'whale_user_rec': float(whale_user_rec) if not np.isnan(whale_user_rec) else None,
        'whale_user_prec': float(whale_user_prec) if not np.isnan(whale_user_prec) else None,
        'whale_user_rec_lift': float(whale_user_rec_lift) if not np.isnan(whale_user_rec_lift) else None,
        'user_revcap': float(user_revcap) if not np.isnan(user_revcap) else None,
        'oracle_user_revcap': float(oracle_user_revcap) if not np.isnan(oracle_user_revcap) else None,
        'whale_def': whale_def,
        'user_score_agg': user_score_agg,
        'whale_threshold': whale_threshold,
        'k': k,
    }


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
    Tail Calibration（Value Calibration - 金额/EV 校准）

    按预测分数分桶，计算预测金额 / 实际金额比值。
    目标是接近 1.0，避免分配时高估/低估。

    [v2.2 工业界命名]
    - SumRatio@K（原 sum_ratio）
    - MeanRatio@K（原 mean_ratio）

    Note:
        这是"金额/EV校准"，与 compute_calibration() 的"概率校准"不同。

    ⚠️ **重要前提**：
        此指标 **只对 "y_pred 是同一金额刻度的 EV" 有意义**。
        - ✅ 有意义：y_pred 是预测金额（如 Ridge 回归输出、反 log 变换后的值）
        - ❌ 无意义：y_pred 是 ranking score（如 tree 的 raw score、pairwise ranker 输出）

        若模型输出只是排序分数而非金额，此 ratio 会变得没有意义，
        此时跳过或仅作为相对比较。

    Args:
        y_true: 真实标签（原始金额）
        y_pred: 预测分数（**必须是金额刻度**，否则 ratio 无意义）
        buckets: 分桶比例列表，如 [0.001, 0.005, 0.01, 0.05]

    Returns:
        dict: {bucket_label: {'sum_ratio': float, 'mean_ratio': float}}

    Formula:
        SumRatio@K = Σ_{i ∈ Top K%} pred_i / Σ_{i ∈ Top K%} y_i
        MeanRatio@K = Mean(pred[Top K%]) / Mean(y[Top K%])
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
# 曲线计算（保持不变）
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
# 稳定性评估（保持不变）
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
# [NEW] Gini 系数（通用 helper）
# =============================================================================

def gini_coefficient(x: np.ndarray, eps: float = 1e-12) -> float:
    """
    计算 Gini 系数（集中度指标）

    Gini = 0 表示完全均匀，Gini = 1 表示完全集中。

    Args:
        x: 数值数组（如各主播收入）
        eps: 防止除零的小量

    Returns:
        float: Gini ∈ [0, 1]，空数组或全零返回 NaN

    Note:
        - 负数会被 clip 为 0 并记录 warning
        - 空数组返回 NaN
    """
    x = np.asarray(x, dtype=np.float64).flatten()

    if len(x) == 0:
        return np.nan

    # 处理负数
    if np.any(x < 0):
        warnings.warn("gini_coefficient: negative values clipped to 0")
        x = np.clip(x, 0, None)

    # 全零情况
    total = x.sum()
    if total < eps:
        return np.nan

    # 全等情况
    if np.allclose(x, x[0]):
        return 0.0

    # Gini 公式：基于 Lorenz 曲线
    # Gini = (Σ_i Σ_j |x_i - x_j|) / (2 * n * Σ x)
    # 更高效的向量化实现：
    n = len(x)
    x_sorted = np.sort(x)
    cumsum = np.cumsum(x_sorted)
    # Gini = 1 - 2 * Σ_{i=1}^{n} ((n - i + 0.5) * x_i) / (n * Σ x)
    # 等价于：Gini = (2 * Σ_{i=1}^{n} i * x_{(i)}) / (n * Σ x) - (n + 1) / n
    index = np.arange(1, n + 1)
    gini = (2.0 * np.dot(index, x_sorted)) / (n * total + eps) - (n + 1.0) / n

    return float(np.clip(gini, 0.0, 1.0))


# =============================================================================
# [NEW] 概率校准（ECE + Reliability Curve）
# =============================================================================

def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = DEFAULT_PROB_CALIBRATION_BINS,
    strategy: str = "uniform",
    sample_weight: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    计算概率校准指标（ECE + Reliability Curve）

    用于二分类概率校准（例如是否打赏 P(gift|x)）。

    Note:
        这是"概率校准"，与 tail_calibration() 的"金额/EV校准"不同。
        - 概率校准：P(gift|x) 预测是否准确
        - 金额校准：E[amount|x] 预测是否准确

    Args:
        y_true: 真实标签。支持：
            - 0/1 或 bool 二分类标签
            - 原始金额（自动转换为 y_true_bin = y_true > 0）
        y_prob: 预测概率，范围应为 [0,1]
            （若超出范围会 clip 并记录 warning）
        n_bins: 分桶数量
        strategy: 分桶策略
            - "uniform": 等宽 bins in [0,1]
            - "quantile": 按 y_prob 分位数切 bins
        sample_weight: 样本权重（可选）
        eps: 防止除零的小量

    Returns:
        dict: {
            "ece": float,  # Expected Calibration Error
            "bins": [      # Reliability curve 每个 bin 的详情
                {"bin_lower": 0.0, "bin_upper": 0.1, "n": 12345,
                 "avg_pred": 0.04, "avg_true": 0.03, "gap": -0.01},
                ...
            ],
            "meta": {
                "n": int,
                "positive_rate": float,
                "strategy": str,
                "n_bins": int,
                "warnings": [...]
            }
        }

    Formula:
        ECE = Σ_bin (n_bin / n_total) * |avg_true - avg_pred|
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_prob = np.asarray(y_prob, dtype=np.float64).flatten()

    if len(y_true) != len(y_prob):
        raise ValueError("y_true and y_prob must have the same length")

    n = len(y_true)
    warnings_list = []

    # 自动转换金额为二分类
    if not np.all((y_true == 0) | (y_true == 1)):
        y_true_bin = (y_true > 0).astype(float)
        warnings_list.append("y_true converted to binary (y_true > 0)")
    else:
        y_true_bin = y_true

    # Clip y_prob to [0, 1]
    if np.any(y_prob < 0) or np.any(y_prob > 1):
        warnings_list.append(f"y_prob clipped to [0,1], original range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")
        y_prob = np.clip(y_prob, 0, 1)

    # 处理权重
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64).flatten()
        if len(sample_weight) != n:
            raise ValueError("sample_weight must have the same length as y_true")
        total_weight = sample_weight.sum()
    else:
        sample_weight = np.ones(n)
        total_weight = float(n)

    positive_rate = float(np.average(y_true_bin, weights=sample_weight))

    # 边界情况：全同类
    if positive_rate < eps or positive_rate > 1 - eps:
        warnings_list.append(f"All samples are same class (positive_rate={positive_rate:.4f})")
        return {
            "ece": np.nan,
            "bins": [],
            "meta": {
                "n": n,
                "positive_rate": positive_rate,
                "strategy": strategy,
                "n_bins": n_bins,
                "warnings": warnings_list,
            }
        }

    # 分桶
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        # 按分位数，但要处理重复值
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(y_prob, percentiles)
        bin_edges = np.unique(bin_edges)  # 去重
        if len(bin_edges) < 2:
            bin_edges = np.array([0, 1])
            warnings_list.append("quantile bins collapsed to single bin due to duplicate values")
    else:
        raise ValueError(f"strategy must be 'uniform' or 'quantile', got '{strategy}'")

    # 计算每个 bin 的统计量
    bins_result = []
    ece = 0.0

    # 使用 np.digitize 进行高效分桶（向量化）
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])  # 0 to n_bins-1

    for i in range(len(bin_edges) - 1):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        mask = (bin_indices == i)
        bin_weight = sample_weight[mask].sum()

        if bin_weight < eps:
            continue

        avg_pred = float(np.average(y_prob[mask], weights=sample_weight[mask]))
        avg_true = float(np.average(y_true_bin[mask], weights=sample_weight[mask]))
        gap = avg_true - avg_pred

        bins_result.append({
            "bin_lower": float(bin_lower),
            "bin_upper": float(bin_upper),
            "n": int(mask.sum()),
            "weight": float(bin_weight),
            "avg_pred": avg_pred,
            "avg_true": avg_true,
            "gap": float(gap),
        })

        # ECE 累加
        ece += (bin_weight / total_weight) * abs(gap)

    return {
        "ece": float(ece),
        "bins": bins_result,
        "meta": {
            "n": n,
            "positive_rate": positive_rate,
            "strategy": strategy,
            "n_bins": n_bins,
            "warnings": warnings_list,
        }
    }


# =============================================================================
# [NEW] 切片指标（Cold-start / Whale / Tier）
# =============================================================================

def _eval_subset(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    whale_threshold: float,
    k_values: List[float],
    min_n: int = DEFAULT_MIN_SLICE_N,
) -> Optional[Dict[str, Any]]:
    """
    内部 helper：评估子集的指标

    Args:
        y_true: 全量真实标签
        y_pred: 全量预测分数
        mask: bool 掩码
        whale_threshold: whale 阈值
        k_values: K 值列表
        min_n: 最小样本数

    Returns:
        dict or None: 子集指标，若样本不足返回 None
    """
    n_subset = mask.sum()
    if n_subset < min_n:
        return None

    y_sub = y_true[mask]
    pred_sub = y_pred[mask]
    total_revenue = y_sub.sum()

    if total_revenue == 0:
        return {
            "n": int(n_subset),
            "total_revenue": 0.0,
            "gift_rate": 0.0,
            "notes": "no revenue in this slice",
            "revcap_curve": {},
            "metrics_by_k": {},
            "calibration": {},
        }

    # RevCap 曲线
    revcap_curve = compute_revcap_curve(y_sub, pred_sub, k_values)
    oracle_curve = compute_oracle_revcap_curve(y_sub, k_values)

    # @1% 详细指标
    metrics_1pct = compute_all_metrics_at_k(y_sub, pred_sub, whale_threshold, 0.01)

    # 全 K 值指标
    metrics_by_k = {}
    for k in k_values:
        label = f'{k*100:.1f}%' if k < 0.01 else f'{k*100:.0f}%'
        metrics_by_k[label] = compute_all_metrics_at_k(y_sub, pred_sub, whale_threshold, k)
        metrics_by_k[label]['oracle_revcap'] = oracle_curve[label]

    # Tail calibration
    calibration = tail_calibration(y_sub, pred_sub)

    return {
        "n": int(n_subset),
        "total_revenue": float(total_revenue),
        "gift_rate": float((y_sub > 0).mean()),
        "n_whales": int((y_sub >= whale_threshold).sum()),
        "revcap_1pct": metrics_1pct['revcap'],
        "whale_recall_1pct": metrics_1pct['whale_recall'],
        "whale_precision_1pct": metrics_1pct['whale_precision'],
        "avg_revenue_1pct": metrics_1pct['avg_revenue'],
        "revcap_curve": revcap_curve,
        "oracle_curve": oracle_curve,
        "metrics_by_k": metrics_by_k,
        "calibration": calibration,
    }


def compute_slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    whale_threshold: float,
    k_values: Optional[List[float]] = None,
    # 切片字段名配置
    user_col: str = "user_id",
    streamer_col: str = "streamer_id",
    pair_hist_col: str = "pair_gift_count",
    streamer_hist_col: str = "streamer_gift_count",
    user_value_col: str = "user_gift_sum",
    streamer_value_col: str = "streamer_gift_sum",
    user_tier_col: Optional[str] = None,
    streamer_tier_col: Optional[str] = None,
    min_slice_n: int = DEFAULT_MIN_SLICE_N,
) -> Dict[str, Any]:
    """
    计算切片指标（Cold-start / Whale / Tier）

    用于回答"冷启动、whale、头/尾用户是否被系统性损害"。

    Args:
        y_true: 真实标签（原始金额）
        y_pred: 预测分数
        df: 测试数据 DataFrame
        whale_threshold: whale 定义阈值
        k_values: K 值列表，默认使用 DEFAULT_K_VALUES
        user_col: 用户 ID 列名
        streamer_col: 主播 ID 列名
        pair_hist_col: pair 历史打赏次数列名（用于冷启动）
        streamer_hist_col: 主播历史被打赏次数列名
        user_value_col: 用户历史总打赏金额列名
        streamer_value_col: 主播历史总收入列名
        user_tier_col: 用户分层列名（可选，若无则自动生成）
        streamer_tier_col: 主播分层列名（可选）
        min_slice_n: 切片最小样本数

    Returns:
        dict: {
            "cold_start_pair": {...metrics...},
            "cold_start_streamer": {...},
            "whale_true": {...},
            "non_whale_true": {...},
            "user_top_1pct": {...},
            "user_top_10pct": {...},
            "user_tail": {...},
            "streamer_top_1pct": {...},
            ...
            "skipped": {
                "user_tier": "missing columns: user_gift_sum",
                ...
            }
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if k_values is None:
        k_values = DEFAULT_K_VALUES

    results = {}
    skipped = {}

    # =========================================================================
    # A) Cold-start slices
    # =========================================================================

    # cold_start_pair
    if pair_hist_col in df.columns:
        mask = (df[pair_hist_col] == 0).values
        subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
        if subset_result is not None:
            results["cold_start_pair"] = subset_result
        else:
            skipped["cold_start_pair"] = f"insufficient samples (n < {min_slice_n})"
    else:
        skipped["cold_start_pair"] = f"missing column: {pair_hist_col}"

    # cold_start_streamer
    if streamer_hist_col in df.columns:
        mask = (df[streamer_hist_col] == 0).values
        subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
        if subset_result is not None:
            results["cold_start_streamer"] = subset_result
        else:
            skipped["cold_start_streamer"] = f"insufficient samples (n < {min_slice_n})"
    elif streamer_value_col in df.columns:
        # 退化：用 value==0 代替
        mask = (df[streamer_value_col] == 0).values
        subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
        if subset_result is not None:
            results["cold_start_streamer"] = subset_result
            results["cold_start_streamer"]["notes"] = f"using {streamer_value_col}==0 as proxy"
        else:
            skipped["cold_start_streamer"] = f"insufficient samples (n < {min_slice_n})"
    else:
        skipped["cold_start_streamer"] = f"missing columns: {streamer_hist_col}, {streamer_value_col}"

    # =========================================================================
    # B) Whale / Non-whale slices（按真实标签切）
    # =========================================================================

    # whale_true: 真实大额打赏
    mask_whale = (y_true >= whale_threshold)
    subset_result = _eval_subset(y_true, y_pred, mask_whale, whale_threshold, k_values, min_slice_n)
    if subset_result is not None:
        results["whale_true"] = subset_result
    else:
        skipped["whale_true"] = f"insufficient samples (n < {min_slice_n})"

    # non_whale_true: 非大额（包括 0 和小额）
    mask_non_whale = (y_true < whale_threshold)
    subset_result = _eval_subset(y_true, y_pred, mask_non_whale, whale_threshold, k_values, min_slice_n)
    if subset_result is not None:
        results["non_whale_true"] = subset_result
    else:
        skipped["non_whale_true"] = f"insufficient samples (n < {min_slice_n})"

    # =========================================================================
    # C) Tier slices（用户分层 / 主播分层）
    # =========================================================================

    # 用户分层
    user_tier_source = None
    if user_tier_col is not None and user_tier_col in df.columns:
        user_tier_source = user_tier_col
    elif user_value_col in df.columns:
        user_tier_source = user_value_col

    if user_tier_source is not None:
        try:
            user_values = df[user_tier_source].values
            # 计算分位数阈值
            p99 = np.percentile(user_values, 99)
            p90 = np.percentile(user_values, 90)

            # user_top_1pct
            mask = (user_values >= p99)
            subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
            if subset_result is not None:
                results["user_top_1pct"] = subset_result
            else:
                skipped["user_top_1pct"] = f"insufficient samples (n < {min_slice_n})"

            # user_top_10pct
            mask = (user_values >= p90) & (user_values < p99)
            subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
            if subset_result is not None:
                results["user_top_10pct"] = subset_result
            else:
                skipped["user_top_10pct"] = f"insufficient samples (n < {min_slice_n})"

            # user_tail
            mask = (user_values < p90)
            subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
            if subset_result is not None:
                results["user_tail"] = subset_result
            else:
                skipped["user_tail"] = f"insufficient samples (n < {min_slice_n})"

        except Exception as e:
            skipped["user_tier"] = f"error computing user tier: {str(e)}"
    else:
        skipped["user_tier"] = f"missing columns: {user_tier_col or user_value_col}"

    # 主播分层
    streamer_tier_source = None
    if streamer_tier_col is not None and streamer_tier_col in df.columns:
        streamer_tier_source = streamer_tier_col
    elif streamer_value_col in df.columns:
        streamer_tier_source = streamer_value_col

    if streamer_tier_source is not None:
        try:
            streamer_values = df[streamer_tier_source].values
            p99 = np.percentile(streamer_values, 99)
            p90 = np.percentile(streamer_values, 90)

            # streamer_top_1pct
            mask = (streamer_values >= p99)
            subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
            if subset_result is not None:
                results["streamer_top_1pct"] = subset_result
            else:
                skipped["streamer_top_1pct"] = f"insufficient samples (n < {min_slice_n})"

            # streamer_top_10pct
            mask = (streamer_values >= p90) & (streamer_values < p99)
            subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
            if subset_result is not None:
                results["streamer_top_10pct"] = subset_result
            else:
                skipped["streamer_top_10pct"] = f"insufficient samples (n < {min_slice_n})"

            # streamer_tail
            mask = (streamer_values < p90)
            subset_result = _eval_subset(y_true, y_pred, mask, whale_threshold, k_values, min_slice_n)
            if subset_result is not None:
                results["streamer_tail"] = subset_result
            else:
                skipped["streamer_tail"] = f"insufficient samples (n < {min_slice_n})"

        except Exception as e:
            skipped["streamer_tier"] = f"error computing streamer tier: {str(e)}"
    else:
        skipped["streamer_tier"] = f"missing columns: {streamer_tier_col or streamer_value_col}"

    results["skipped"] = skipped
    return results


# =============================================================================
# [NEW] 生态指标（Gini / Coverage / Overload）
# =============================================================================

def compute_ecosystem_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    k_select: float = 0.01,
    user_col: str = "user_id",
    streamer_col: str = "streamer_id",
    timestamp_col: str = "timestamp",
    streamer_hist_col: str = "streamer_gift_count",
    streamer_value_col: str = "streamer_gift_sum",
    tail_streamer_quantile: float = DEFAULT_TAIL_STREAMER_QUANTILE,
    overload_window_minutes: int = DEFAULT_OVERLOAD_WINDOW_MINUTES,
    overload_cap_per_window: int = DEFAULT_OVERLOAD_CAP_PER_WINDOW,
    high_value_user_col: Optional[str] = None,
    high_value_user_quantile: float = DEFAULT_HIGH_VALUE_USER_QUANTILE,
    user_value_col: str = "user_gift_sum",
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    计算生态指标（Gini / Coverage / Overload）

    这是分配层的必备护栏指标。
    使用离线近似：把预测 Top k_select 的样本视为"系统会分配/曝光"的集合。

    Args:
        y_true: 真实标签（原始金额）
        y_pred: 预测分数
        df: 测试数据 DataFrame
        k_select: Top K% 作为"被分配/曝光集合"的离线近似
        user_col: 用户 ID 列名
        streamer_col: 主播 ID 列名
        timestamp_col: 时间戳列名
        streamer_hist_col: 主播历史被打赏次数列名
        streamer_value_col: 主播历史总收入列名
        tail_streamer_quantile: 长尾主播定义（bottom X%）
        overload_window_minutes: 过载检测的时间窗口（分钟）
        overload_cap_per_window: 每窗口高价值用户上限
        high_value_user_col: 高价值用户标识列（可选）
        high_value_user_quantile: 若无 high_value_user_col，按此分位数定义
        user_value_col: 用户历史打赏金额列名
        eps: 防止除零的小量

    Returns:
        dict: {
            "selection": {"k_select": 0.01, "n_selected": ..., "n_total": ...},
            "gini": {"streamer_revenue_gini": 0.93, "top10_share": 0.81},
            "coverage": {
                "streamer_coverage": ...,
                "tail_coverage": ...,
                "cold_start_streamer_coverage": ...
            },
            "overload": {
                "window_minutes": 10,
                "cap": 3,
                "overload_bucket_rate": ...,
                "overloaded_streamer_rate": ...
            },
            "meta": {"warnings": [...], "used_columns": {...}}
        }
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_total = len(y_true)
    n_selected = max(1, int(n_total * k_select))

    warnings_list = []
    used_columns = {}

    # 选择 Top K% 样本
    top_indices = np.argsort(y_pred)[-n_selected:]
    selected_mask = np.zeros(n_total, dtype=bool)
    selected_mask[top_indices] = True

    # 重置索引以避免后续操作的索引问题
    df_selected = df.iloc[top_indices].copy().reset_index(drop=True)
    y_selected = y_true[top_indices]

    # =========================================================================
    # 1) Gini（主播收入集中度）
    # =========================================================================

    gini_result = {"streamer_revenue_gini": np.nan, "top10_share": np.nan}

    if streamer_col in df.columns:
        used_columns["streamer_col"] = streamer_col
        try:
            # 按主播聚合：在 selected set 中的收入
            df_selected_with_y = df_selected.copy()
            df_selected_with_y['_captured_revenue'] = y_selected
            streamer_revenue = df_selected_with_y.groupby(streamer_col)['_captured_revenue'].sum()

            streamer_revenue_values = streamer_revenue.values

            if len(streamer_revenue_values) > 0 and streamer_revenue_values.sum() > eps:
                gini_result["streamer_revenue_gini"] = gini_coefficient(streamer_revenue_values)

                # Top 10% share
                sorted_rev = np.sort(streamer_revenue_values)[::-1]
                n_top10 = max(1, int(len(sorted_rev) * 0.1))
                total_rev = sorted_rev.sum()
                if total_rev > eps:
                    gini_result["top10_share"] = float(sorted_rev[:n_top10].sum() / total_rev)
            else:
                warnings_list.append("gini: no revenue in selected set")

        except Exception as e:
            warnings_list.append(f"gini computation error: {str(e)}")
    else:
        warnings_list.append(f"gini: missing column {streamer_col}")

    # =========================================================================
    # 2) Coverage（覆盖率）
    # =========================================================================

    coverage_result = {
        "streamer_coverage": np.nan,
        "tail_coverage": np.nan,
        "cold_start_streamer_coverage": np.nan,
    }

    if streamer_col in df.columns:
        try:
            # 全局主播集
            all_streamers = set(df[streamer_col].unique())
            selected_streamers = set(df_selected[streamer_col].unique())

            coverage_result["streamer_coverage"] = float(len(selected_streamers) / len(all_streamers)) \
                if len(all_streamers) > 0 else np.nan

            # 长尾主播覆盖率
            if streamer_value_col in df.columns:
                used_columns["streamer_value_col"] = streamer_value_col
                # 计算每个主播在全数据中的总价值
                streamer_total_value = df.groupby(streamer_col)[streamer_value_col].first()
                tail_threshold = np.percentile(streamer_total_value.values, tail_streamer_quantile * 100)
                tail_streamers = set(streamer_total_value[streamer_total_value <= tail_threshold].index)

                if len(tail_streamers) > 0:
                    tail_covered = tail_streamers & selected_streamers
                    coverage_result["tail_coverage"] = float(len(tail_covered) / len(tail_streamers))
            else:
                warnings_list.append(f"tail_coverage: missing column {streamer_value_col}")

            # 冷启动主播覆盖率
            if streamer_hist_col in df.columns:
                used_columns["streamer_hist_col"] = streamer_hist_col
                cold_start_streamers = set(df[df[streamer_hist_col] == 0][streamer_col].unique())
                if len(cold_start_streamers) > 0:
                    cold_covered = cold_start_streamers & selected_streamers
                    coverage_result["cold_start_streamer_coverage"] = float(
                        len(cold_covered) / len(cold_start_streamers)
                    )
            else:
                warnings_list.append(f"cold_start_streamer_coverage: missing column {streamer_hist_col}")

        except Exception as e:
            warnings_list.append(f"coverage computation error: {str(e)}")

    # =========================================================================
    # 3) Overload（过载率）- 拆分为用户侧和主播侧
    # [v2.2 更新] 明确区分两类 Overload：
    #   - U-OverTarget@W（用户侧）: 同一个用户在窗口内被选中次数 > cap
    #   - S-OverLoad@W（主播侧）: 同一个主播在窗口内接收高价值用户数 > cap
    # =========================================================================

    overload_result = {
        "window_minutes": overload_window_minutes,
        "cap": overload_cap_per_window,
        # 用户侧过度触达（user fatigue / spam risk）
        "user_overtarget_rate": np.nan,        # 被过度推送的用户比例
        "user_overtarget_bucket_rate": np.nan, # 过载的 (user, time_bucket) 比例
        # 主播侧承接过载（streamer overload）
        "streamer_overload_rate": np.nan,      # 过载的主播比例
        "streamer_overload_bucket_rate": np.nan, # 过载的 (streamer, time_bucket) 比例
        # 向后兼容（deprecated）
        "overload_bucket_rate": np.nan,
        "overloaded_streamer_rate": np.nan,
    }

    if timestamp_col in df.columns and streamer_col in df.columns and user_col in df.columns:
        used_columns["timestamp_col"] = timestamp_col
        used_columns["user_col"] = user_col

        try:
            df_sel = df_selected.copy()

            # 识别高价值用户
            if high_value_user_col is not None and high_value_user_col in df.columns:
                used_columns["high_value_user_col"] = high_value_user_col
                high_value_mask = df_sel[high_value_user_col].astype(bool).values
            elif user_value_col in df.columns:
                used_columns["user_value_col"] = user_value_col
                threshold = np.percentile(df[user_value_col].values, high_value_user_quantile * 100)
                high_value_mask = (df_sel[user_value_col] >= threshold).values
            else:
                warnings_list.append(f"overload: missing user value column for high-value detection")
                high_value_mask = np.ones(len(df_sel), dtype=bool)  # fallback: 全部视为高价值

            # 只保留高价值用户的记录
            df_hv = df_sel[high_value_mask].copy()

            if len(df_hv) > 0:
                # 时间分桶
                df_hv['_ts'] = pd.to_datetime(df_hv[timestamp_col], unit='ms')
                df_hv['_time_bucket'] = df_hv['_ts'].dt.floor(f'{overload_window_minutes}min')

                # ===============================================================
                # A) 用户侧过度触达 (U-OverTarget@W)
                # 同一个用户在窗口内被选中/推送次数 > cap
                # ===============================================================
                user_bucket_counts = df_hv.groupby([user_col, '_time_bucket']).size()
                total_user_buckets = len(user_bucket_counts)
                overtarget_user_buckets = (user_bucket_counts > overload_cap_per_window).sum()

                if total_user_buckets > 0:
                    overload_result["user_overtarget_bucket_rate"] = float(
                        overtarget_user_buckets / total_user_buckets
                    )

                # 被过度推送的用户比例
                unique_users = df_hv[user_col].nunique()
                if overtarget_user_buckets > 0:
                    overtarget_users = user_bucket_counts[
                        user_bucket_counts > overload_cap_per_window
                    ].reset_index()[user_col].nunique()
                else:
                    overtarget_users = 0

                if unique_users > 0:
                    overload_result["user_overtarget_rate"] = float(overtarget_users / unique_users)

                # ===============================================================
                # B) 主播侧承接过载 (S-OverLoad@W)
                # 同一个主播在窗口内接收高价值用户数 > cap
                # ===============================================================
                streamer_bucket_counts = df_hv.groupby([streamer_col, '_time_bucket'])[user_col].nunique()
                total_streamer_buckets = len(streamer_bucket_counts)
                overload_streamer_buckets = (streamer_bucket_counts > overload_cap_per_window).sum()

                if total_streamer_buckets > 0:
                    overload_result["streamer_overload_bucket_rate"] = float(
                        overload_streamer_buckets / total_streamer_buckets
                    )
                    # 向后兼容
                    overload_result["overload_bucket_rate"] = overload_result["streamer_overload_bucket_rate"]

                # 受影响的主播比例
                unique_streamers_in_selected = df_sel[streamer_col].nunique()
                if overload_streamer_buckets > 0:
                    overloaded_streamers = streamer_bucket_counts[
                        streamer_bucket_counts > overload_cap_per_window
                    ].reset_index()[streamer_col].nunique()
                else:
                    overloaded_streamers = 0

                if unique_streamers_in_selected > 0:
                    overload_result["streamer_overload_rate"] = float(
                        overloaded_streamers / unique_streamers_in_selected
                    )
                    # 向后兼容
                    overload_result["overloaded_streamer_rate"] = overload_result["streamer_overload_rate"]

            else:
                warnings_list.append("overload: no high-value users in selected set")

        except Exception as e:
            warnings_list.append(f"overload computation error: {str(e)}")
    else:
        missing = [c for c in [timestamp_col, streamer_col, user_col] if c not in df.columns]
        warnings_list.append(f"overload: missing columns {missing}")

    return {
        "selection": {
            "k_select": k_select,
            "n_selected": n_selected,
            "n_total": n_total,
        },
        "gini": gini_result,
        "coverage": coverage_result,
        "overload": overload_result,
        "meta": {
            "warnings": warnings_list,
            "used_columns": used_columns,
        }
    }


# =============================================================================
# [NEW v2.1] P0 决策指标（Misallocation / Capture AUC / Drift / Diversity）
# =============================================================================

def compute_misallocation_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: float = 0.01,
    cost_type: str = "opportunity",
) -> Dict[str, float]:
    """
    计算错分代价（Misallocation Cost / Regret）

    衡量"把稀缺大哥分给错误主播"的代价，同样 RevCap 的模型错分代价可能差 2×。

    [v2.2 更新] 输出同时包含：
    - **金额版**（元）：AchievedRev@K, OracleRev@K, Regret@K
    - **比例版**（无量纲）：Eff@K (= nRevCap@K), RegretPct@K

    Args:
        y_true: 真实标签（原始金额）
        y_pred: 预测分数
        k: Top 比例（默认 1%）
        cost_type: 代价类型（保留参数，默认返回全部）

    Returns:
        dict: {
            # 金额版（元）
            "achieved_rev": AchievedRev@K = Σ_{i ∈ Top K} y_i,
            "oracle_rev": OracleRev@K = Σ_{i ∈ Oracle Top K} y_i,
            "regret": Regret@K = OracleRev@K - AchievedRev@K（元）,

            # 比例版（无量纲）
            "eff": Eff@K = AchievedRev@K / OracleRev@K ∈ [0,1],
            "regret_pct": RegretPct@K = 1 - Eff@K ∈ [0,1],

            # 浪费曝光
            "wasted_exposure_rate": 选中但 y=0 的比例,
            "wasted_exposure_value": 浪费在 y=0 样本的"潜在"收入（元）,

            # 向后兼容（deprecated，建议用新字段）
            "efficiency": = eff,
            "opportunity_cost": = regret,
            "opportunity_cost_pct": = regret_pct,
        }

    Note:
        - Regret（遗憾值）是工业界/学术界更通用的命名
        - Eff@K ≈ nRevCap@K（当 OracleRevCap ≈ 1 时完全相等）
        - 金额版用于业务沟通，比例版用于跨实验对比

    Example:
        >>> cost = compute_misallocation_cost(y_true, y_pred, k=0.01)
        >>> print(f"Regret: {cost['regret']:,.0f} 元 ({cost['regret_pct']:.1%})")
        >>> print(f"Efficiency: {cost['eff']:.1%}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_total = len(y_true)
    n_top = max(1, int(n_total * k))

    # 模型选择的 Top K
    pred_top_indices = np.argsort(y_pred)[-n_top:]
    achieved_rev = y_true[pred_top_indices].sum()

    # Oracle 选择的 Top K（理论最优）
    oracle_top_indices = np.argsort(y_true)[-n_top:]
    oracle_rev = y_true[oracle_top_indices].sum()

    # Regret（遗憾值）：金额版和比例版
    regret = oracle_rev - achieved_rev
    eff = achieved_rev / oracle_rev if oracle_rev > 0 else 0.0
    regret_pct = 1 - eff

    # 浪费曝光率（选中但实际没打赏）
    wasted_mask = y_true[pred_top_indices] == 0
    wasted_exposure_rate = wasted_mask.mean()

    # 浪费曝光的"潜在价值"（用选中集合中有打赏的均值估计）
    non_zero_in_top = y_true[pred_top_indices][~wasted_mask]
    avg_if_gift = non_zero_in_top.mean() if len(non_zero_in_top) > 0 else 0.0
    wasted_exposure_value = wasted_mask.sum() * avg_if_gift

    return {
        "k": k,
        "n_selected": n_top,
        # 金额版（元）- 新命名
        "achieved_rev": float(achieved_rev),
        "oracle_rev": float(oracle_rev),
        "regret": float(regret),
        # 比例版（无量纲）- 新命名
        "eff": float(eff),
        "regret_pct": float(regret_pct),
        # 浪费曝光
        "wasted_exposure_rate": float(wasted_exposure_rate),
        "wasted_exposure_value": float(wasted_exposure_value),
        # 向后兼容（deprecated）
        "achieved_revenue": float(achieved_rev),
        "oracle_revenue": float(oracle_rev),
        "opportunity_cost": float(regret),
        "opportunity_cost_pct": float(regret_pct),
        "efficiency": float(eff),
    }


def compute_capture_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_k: float = 0.10,
    n_points: int = 100,
    normalize: bool = True,
) -> Dict[str, Any]:
    """
    计算 Capture Curve AUC（收益-覆盖曲线下面积）

    避免"只优化 1% 导致 5%/10% 崩"的过拟合，衡量整体排序质量。

    [v2.2 更新] 明确三种量的定义：
    - **CapAUC@K（原始积分）**: ∫_0^K RevCap(k) dk，数值在 [0, K]
    - **MeanRevCap@K**: CapAUC@K / K，数值在 [0, 1]，更直观
    - **nAUC@K（推荐）**: CapAUC@K / OracleAUC@K，数值在 [0, 1]

    Args:
        y_true: 真实标签（原始金额）
        y_pred: 预测分数
        max_k: 曲线最大 K 值（默认 10%）
        n_points: 曲线采样点数
        normalize: 是否归一化（除以 Oracle AUC）

    Returns:
        dict: {
            # 原始积分（数值范围 [0, max_k]）
            "capture_auc": ∫_0^max_k RevCap(k) dk,
            "oracle_auc": ∫_0^max_k OracleRevCap(k) dk,

            # 归一化版本（数值范围 [0, 1]）
            "nAUC": CapAUC / OracleAUC（推荐使用）,
            "mean_revcap": CapAUC / max_k（平均 RevCap）,

            # 向后兼容
            "normalized_auc": = nAUC,

            "curve": [(k, revcap), ...] 曲线数据点,
        }

    Formula:
        CapAUC@K = ∫_0^K RevCap(k) dk
        nAUC@K = CapAUC@K / OracleAUC@K ∈ [0, 1]

    Note:
        - **nAUC@10%** 是报告的推荐指标（归一化后可比）
        - 原始 CapAUC 的数值范围是 [0, max_k]，不要误读为百分比
        - 比单点 RevCap@1% 更全面评估排序质量

    Example:
        >>> auc = compute_capture_auc(y_true, y_pred, max_k=0.10)
        >>> print(f"nAUC@10%: {auc['nAUC']:.1%}")  # 推荐
        >>> print(f"MeanRevCap@10%: {auc['mean_revcap']:.1%}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total_revenue = y_true.sum()
    if total_revenue == 0:
        return {
            "capture_auc": 0.0,
            "oracle_auc": 0.0,
            "nAUC": np.nan,
            "mean_revcap": np.nan,
            "normalized_auc": np.nan,
            "curve": [],
        }

    n_total = len(y_true)
    k_values = np.linspace(0.001, max_k, n_points)

    # 预计算排序
    pred_order = np.argsort(y_pred)[::-1]
    oracle_order = np.argsort(y_true)[::-1]

    # 累积收入
    pred_cumsum = np.cumsum(y_true[pred_order])
    oracle_cumsum = np.cumsum(y_true[oracle_order])

    curve = []
    oracle_curve = []

    for k in k_values:
        n_top = max(1, int(n_total * k))
        revcap = pred_cumsum[n_top - 1] / total_revenue
        oracle_revcap = oracle_cumsum[n_top - 1] / total_revenue
        curve.append((k, revcap))
        oracle_curve.append((k, oracle_revcap))

    # 梯形积分计算 AUC
    capture_auc = np.trapz([p[1] for p in curve], [p[0] for p in curve])
    oracle_auc = np.trapz([p[1] for p in oracle_curve], [p[0] for p in oracle_curve])

    # 归一化指标
    nAUC = capture_auc / oracle_auc if oracle_auc > 0 else 0.0
    mean_revcap = capture_auc / max_k if max_k > 0 else 0.0

    return {
        # 原始积分
        "capture_auc": float(capture_auc),
        "oracle_auc": float(oracle_auc),
        # 归一化版本（推荐）
        "nAUC": float(nAUC),
        "mean_revcap": float(mean_revcap),
        # 元数据
        "max_k": max_k,
        "n_points": n_points,
        "curve": curve,
        "oracle_curve": oracle_curve,
        # 向后兼容
        "normalized_auc": float(nAUC),
    }


def compute_drift(
    y_pred_train: np.ndarray,
    y_pred_test: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    计算预测分布漂移（PSI / KL Divergence）

    系统依赖历史特征时，冷启动比例高可能导致严重漂移。

    Args:
        y_pred_train: 训练集预测分数
        y_pred_test: 测试集预测分数
        n_bins: 分桶数量
        eps: 防止除零/log(0) 的小量

    Returns:
        dict: {
            "psi": Population Stability Index,
            "kl_divergence": KL(test || train),
            "wasserstein": Wasserstein-1 距离,
            "max_bin_diff": 最大单桶差异,
            "interpretation": PSI 解读,
            "bins_detail": 每桶详情,
        }

    PSI 解读:
        - PSI < 0.1: 无显著漂移
        - 0.1 ≤ PSI < 0.25: 轻微漂移，需关注
        - PSI ≥ 0.25: 显著漂移，需调查

    Formula:
        PSI = Σ_i (p_test_i - p_train_i) * ln(p_test_i / p_train_i)
    """
    y_pred_train = np.asarray(y_pred_train).flatten()
    y_pred_test = np.asarray(y_pred_test).flatten()

    # 使用训练集分位数定义 bins
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(y_pred_train, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # 计算每桶比例
    train_counts, _ = np.histogram(y_pred_train, bins=bin_edges)
    test_counts, _ = np.histogram(y_pred_test, bins=bin_edges)

    train_pct = train_counts / len(y_pred_train)
    test_pct = test_counts / len(y_pred_test)

    # 避免零值
    train_pct = np.clip(train_pct, eps, 1 - eps)
    test_pct = np.clip(test_pct, eps, 1 - eps)

    # PSI
    psi_per_bin = (test_pct - train_pct) * np.log(test_pct / train_pct)
    psi = psi_per_bin.sum()

    # KL Divergence (test || train)
    kl_per_bin = test_pct * np.log(test_pct / train_pct)
    kl_divergence = kl_per_bin.sum()

    # Wasserstein-1（使用 CDF 差异）
    train_cdf = np.cumsum(train_pct)
    test_cdf = np.cumsum(test_pct)
    wasserstein = np.abs(train_cdf - test_cdf).mean()

    # 最大单桶差异
    max_bin_diff = np.abs(test_pct - train_pct).max()

    # 解读
    if psi < 0.1:
        interpretation = "无显著漂移 (PSI < 0.1)"
    elif psi < 0.25:
        interpretation = "轻微漂移，需关注 (0.1 ≤ PSI < 0.25)"
    else:
        interpretation = "显著漂移，需调查 (PSI ≥ 0.25)"

    # 每桶详情
    bins_detail = []
    for i in range(n_bins):
        bins_detail.append({
            "bin_idx": i,
            "train_pct": float(train_pct[i]),
            "test_pct": float(test_pct[i]),
            "diff": float(test_pct[i] - train_pct[i]),
            "psi_contribution": float(psi_per_bin[i]),
        })

    return {
        "psi": float(psi),
        "kl_divergence": float(kl_divergence),
        "wasserstein": float(wasserstein),
        "max_bin_diff": float(max_bin_diff),
        "interpretation": interpretation,
        "n_bins": n_bins,
        "bins_detail": bins_detail,
    }


def compute_diversity(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    k: float = 0.01,
    group_col: str = "streamer_id",
    value_col: Optional[str] = None,
    y_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    计算分配多样性（Entropy / Effective Number）

    比 Gini 更灵敏，能捕捉"看起来覆盖了很多主播，但流量仍高度集中"的情况。

    Args:
        df: 数据 DataFrame
        y_pred: 预测分数
        k: Top 比例（选中集合）
        group_col: 分组列（如 streamer_id）
        value_col: 权重列（如 revenue），若为 None 则按 count
        y_true: 真实标签（用于计算收入加权多样性）

    Returns:
        dict: {
            "entropy": Shannon 熵,
            "effective_number": exp(entropy)，有效分组数,
            "coverage": 覆盖的分组比例,
            "hhi": Herfindahl-Hirschman Index（集中度）,
            "top10_share": Top 10% 分组的份额,
            "gini": 分组分布的 Gini 系数,
        }

    Note:
        - Effective Number = exp(H) 表示"等效均匀分布下的分组数"
        - 例如：10 个主播各 10%，Effective Number = 10
        - 若 1 个主播 100%，Effective Number = 1
    """
    y_pred = np.asarray(y_pred)

    n_total = len(y_pred)
    n_top = max(1, int(n_total * k))

    # 选择 Top K
    top_indices = np.argsort(y_pred)[-n_top:]
    df_selected = df.iloc[top_indices].copy()

    if group_col not in df_selected.columns:
        return {
            "error": f"missing column: {group_col}",
            "entropy": np.nan,
            "effective_number": np.nan,
        }

    # 计算每组的权重
    if y_true is not None:
        df_selected['_weight'] = y_true[top_indices]
    elif value_col is not None and value_col in df_selected.columns:
        df_selected['_weight'] = df_selected[value_col]
    else:
        df_selected['_weight'] = 1.0

    # 按组聚合
    group_weights = df_selected.groupby(group_col)['_weight'].sum()
    group_weights = group_weights[group_weights > 0]  # 去零

    if len(group_weights) == 0:
        return {
            "entropy": 0.0,
            "effective_number": 1.0,
            "coverage": 0.0,
            "hhi": 1.0,
            "top10_share": 1.0,
            "gini": 1.0,
        }

    # 归一化为概率分布
    total_weight = group_weights.sum()
    p = group_weights.values / total_weight

    # Shannon 熵
    entropy = -np.sum(p * np.log(p + 1e-12))

    # Effective Number = exp(H)
    effective_number = np.exp(entropy)

    # Coverage（覆盖率）
    n_total_groups = df[group_col].nunique()
    n_covered_groups = len(group_weights)
    coverage = n_covered_groups / n_total_groups if n_total_groups > 0 else 0.0

    # HHI (Herfindahl-Hirschman Index)
    hhi = np.sum(p ** 2)

    # Top 10% 分组份额
    n_top10_groups = max(1, int(len(p) * 0.1))
    sorted_p = np.sort(p)[::-1]
    top10_share = sorted_p[:n_top10_groups].sum()

    # Gini
    gini = gini_coefficient(group_weights.values)

    return {
        "entropy": float(entropy),
        "effective_number": float(effective_number),
        "coverage": float(coverage),
        "n_covered_groups": int(n_covered_groups),
        "n_total_groups": int(n_total_groups),
        "hhi": float(hhi),
        "top10_share": float(top10_share),
        "gini": float(gini) if not np.isnan(gini) else None,
        "k": k,
    }


# =============================================================================
# 评估结果类（更新：添加新字段）
# =============================================================================

@dataclass
class EvalResult:
    """
    评估结果类

    包含所有评估指标，支持格式化输出和 JSON 导出。

    v2.0 新增字段（均为 Optional，保持向后兼容）：
    - prob_calibration: 概率校准（ECE + Reliability Curve）
    - slice_metrics: 切片指标（冷启动/whale/tier）
    - ecosystem: 生态指标（Gini/Coverage/Overload）
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

    # [NEW] v2.0 新增字段
    prob_calibration: Optional[Dict[str, Any]] = None
    slice_metrics: Dict[str, Any] = field(default_factory=dict)
    ecosystem: Dict[str, Any] = field(default_factory=dict)

    # [NEW] v2.1 决策核心指标
    misallocation_cost: Dict[str, Any] = field(default_factory=dict)
    capture_auc: Dict[str, Any] = field(default_factory=dict)
    drift: Dict[str, Any] = field(default_factory=dict)
    diversity: Dict[str, Any] = field(default_factory=dict)

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
            lines.append("\n--- Value Calibration (Tail) ---")
            for bucket, ratios in self.calibration.items():
                sum_r = ratios.get('sum_ratio', np.nan)
                mean_r = ratios.get('mean_ratio', np.nan)
                sum_str = f"{sum_r:.3f}" if not np.isnan(sum_r) else "N/A"
                mean_str = f"{mean_r:.3f}" if not np.isnan(mean_r) else "N/A"
                lines.append(f"  {bucket}: Sum={sum_str}, Mean={mean_str}")

        # [NEW] 概率校准
        if self.prob_calibration is not None:
            ece = self.prob_calibration.get('ece', np.nan)
            meta = self.prob_calibration.get('meta', {})
            pos_rate = meta.get('positive_rate', np.nan)
            ece_str = f"{ece:.4f}" if not np.isnan(ece) else "N/A"
            pos_str = f"{pos_rate:.2%}" if not np.isnan(pos_rate) else "N/A"
            lines.extend([
                "",
                "--- Probability Calibration ---",
                f"  ECE: {ece_str} | Positive Rate: {pos_str}",
            ])

        if self.stability:
            lines.extend([
                "",
                "--- Stability (@1%) ---",
                f"  Mean: {self.stability.get('mean', 0):.1%}",
                f"  Std: {self.stability.get('std', 0):.1%}",
                f"  CV: {self.stability.get('cv', 0):.1%}",
                f"  95% CI: [{self.stability.get('ci_lower', 0):.1%}, {self.stability.get('ci_upper', 0):.1%}]",
            ])

        # [NEW] 生态指标摘要
        if self.ecosystem:
            gini = self.ecosystem.get('gini', {})
            coverage = self.ecosystem.get('coverage', {})
            overload = self.ecosystem.get('overload', {})
            selection = self.ecosystem.get('selection', {})

            k_sel = selection.get('k_select', 0.01)

            gini_val = gini.get('streamer_revenue_gini', np.nan)
            top10 = gini.get('top10_share', np.nan)
            tail_cov = coverage.get('tail_coverage', np.nan)
            overload_rate = overload.get('overloaded_streamer_rate', np.nan)

            gini_str = f"{gini_val:.3f}" if not np.isnan(gini_val) else "N/A"
            top10_str = f"{top10:.1%}" if not np.isnan(top10) else "N/A"
            tail_str = f"{tail_cov:.1%}" if not np.isnan(tail_cov) else "N/A"
            overload_str = f"{overload_rate:.1%}" if not np.isnan(overload_rate) else "N/A"

            lines.extend([
                "",
                f"--- Ecosystem Guardrails (Top {k_sel*100:.0f}% selection) ---",
                f"  Streamer Gini: {gini_str} | Top10 Share: {top10_str}",
                f"  Tail Coverage: {tail_str} | Overload Streamer Rate: {overload_str}",
            ])

        # [NEW] 切片指标摘要
        if self.slice_metrics:
            slices = [k for k in self.slice_metrics.keys() if k != 'skipped']
            if slices:
                lines.extend([
                    "",
                    "--- Slice Metrics Summary ---",
                ])
                for slice_name in slices[:5]:  # 只显示前5个
                    s = self.slice_metrics[slice_name]
                    n = s.get('n', 0)
                    revcap = s.get('revcap_1pct', np.nan)
                    revcap_str = f"{revcap:.1%}" if not np.isnan(revcap) else "N/A"
                    lines.append(f"  {slice_name}: n={n:,}, RevCap@1%={revcap_str}")

                skipped = self.slice_metrics.get('skipped', {})
                if skipped:
                    lines.append(f"  (Skipped: {', '.join(skipped.keys())})")

        # [NEW v2.1] 决策核心指标
        if self.misallocation_cost:
            opp_cost = self.misallocation_cost.get('opportunity_cost_pct', np.nan)
            wasted = self.misallocation_cost.get('wasted_exposure_rate', np.nan)
            efficiency = self.misallocation_cost.get('efficiency', np.nan)
            opp_str = f"{opp_cost:.1%}" if not np.isnan(opp_cost) else "N/A"
            wasted_str = f"{wasted:.1%}" if not np.isnan(wasted) else "N/A"
            eff_str = f"{efficiency:.1%}" if not np.isnan(efficiency) else "N/A"
            lines.extend([
                "",
                "--- Decision Metrics (v2.1) ---",
                f"  Opportunity Cost: {opp_str} | Wasted Exposure: {wasted_str}",
                f"  Efficiency (vs Oracle): {eff_str}",
            ])

        if self.capture_auc:
            norm_auc = self.capture_auc.get('normalized_auc', np.nan)
            max_k = self.capture_auc.get('max_k', 0.1)
            auc_str = f"{norm_auc:.1%}" if not np.isnan(norm_auc) else "N/A"
            lines.append(f"  Capture AUC (0-{max_k*100:.0f}%): {auc_str}")

        if self.drift:
            psi = self.drift.get('psi', np.nan)
            interp = self.drift.get('interpretation', '')
            psi_str = f"{psi:.4f}" if not np.isnan(psi) else "N/A"
            lines.append(f"  PSI (Drift): {psi_str} - {interp}")

        if self.diversity:
            eff_num = self.diversity.get('effective_number', np.nan)
            coverage = self.diversity.get('coverage', np.nan)
            eff_str = f"{eff_num:.1f}" if not np.isnan(eff_num) else "N/A"
            cov_str = f"{coverage:.1%}" if not np.isnan(coverage) else "N/A"
            lines.append(f"  Diversity: Effective#={eff_str}, Coverage={cov_str}")

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
        # 处理旧版 JSON 缺少新字段的情况
        if 'prob_calibration' not in d:
            d['prob_calibration'] = None
        if 'slice_metrics' not in d:
            d['slice_metrics'] = {}
        if 'ecosystem' not in d:
            d['ecosystem'] = {}
        # v2.1 新字段
        if 'misallocation_cost' not in d:
            d['misallocation_cost'] = {}
        if 'capture_auc' not in d:
            d['capture_auc'] = {}
        if 'drift' not in d:
            d['drift'] = {}
        if 'diversity' not in d:
            d['diversity'] = {}
        return cls(**d)


# =============================================================================
# 主评估函数（更新：集成新指标）
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: Optional[pd.DataFrame] = None,
    whale_threshold: Optional[float] = None,
    whale_percentile: int = DEFAULT_WHALE_PERCENTILE,
    k_values: List[float] = None,
    compute_stability: bool = True,
    compute_ndcg: bool = False,
    timestamp_col: str = 'timestamp',
    # [NEW] v2.0 新增参数
    y_prob: Optional[np.ndarray] = None,
    compute_slices: bool = True,
    compute_ecosystem: bool = True,
    calibration_config: Optional[Dict[str, Any]] = None,
    slice_config: Optional[Dict[str, Any]] = None,
    ecosystem_config: Optional[Dict[str, Any]] = None,
    # [NEW] v2.1 决策核心指标
    y_pred_train: Optional[np.ndarray] = None,  # 用于计算 drift
    compute_decision_metrics: bool = True,  # 是否计算决策核心指标
) -> EvalResult:
    """
    完整模型评估（主入口函数）

    计算所有指标并返回 EvalResult 对象。

    Args:
        y_true: 真实标签（原始金额）
        y_pred: 预测分数
        test_df: 测试数据 DataFrame（用于按天稳定性/切片/生态指标，可选）
        whale_threshold: whale 定义阈值，若为 None 则自动计算 P90
        whale_percentile: 自动计算 whale 阈值时的百分位数
        k_values: K 值列表
        compute_stability: 是否计算按天稳定性
        compute_ndcg: 是否计算 NDCG
        timestamp_col: 时间戳列名

        [NEW] v2.0 新增参数：
        y_prob: 预测概率（若提供则计算概率校准）
        compute_slices: 是否计算切片指标（需要 test_df）
        compute_ecosystem: 是否计算生态指标（需要 test_df）
        calibration_config: compute_calibration() 的额外参数
        slice_config: compute_slice_metrics() 的额外参数
        ecosystem_config: compute_ecosystem_metrics() 的额外参数

        [NEW] v2.1 决策核心指标：
        y_pred_train: 训练集预测（用于计算 PSI/Drift）
        compute_decision_metrics: 是否计算决策核心指标（Misallocation/AUC/Drift/Diversity）

    Returns:
        EvalResult: 完整评估结果

    Example:
        >>> from gift_EVpred.metrics import evaluate_model
        >>> result = evaluate_model(
        ...     y_test, y_pred, test_df,
        ...     y_prob=y_prob,           # 若有概率输出
        ...     compute_slices=True,
        ...     compute_ecosystem=True,
        ... )
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

    # Tail Calibration（Value Calibration）
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
        try:
            daily_metrics = compute_stability_by_day(
                test_df, y_true, y_pred, whale_threshold, k=0.01, timestamp_col=timestamp_col
            )
            if len(daily_metrics) > 0:
                stability = compute_stability_summary(daily_metrics, 'revcap')
        except Exception:
            pass

    # 可选指标
    ndcg_100 = None
    if compute_ndcg:
        ndcg_100 = normalized_dcg_at_k(y_true, y_pred, k=100)

    # Spearman（可选）
    spearman = None
    try:
        from scipy.stats import spearmanr
        spearman, _ = spearmanr(y_true, y_pred)
        spearman = float(spearman)
    except:
        pass

    # =========================================================================
    # [NEW] v2.0 新增指标
    # =========================================================================

    # 概率校准
    prob_calibration = None
    if y_prob is not None:
        calib_cfg = calibration_config or {}
        try:
            prob_calibration = compute_calibration(y_true, y_prob, **calib_cfg)
        except Exception as e:
            prob_calibration = {"error": str(e)}

    # 切片指标
    slice_metrics = {}
    if compute_slices and test_df is not None:
        slice_cfg = slice_config or {}
        try:
            slice_metrics = compute_slice_metrics(
                y_true, y_pred, test_df, whale_threshold,
                k_values=k_values,
                **slice_cfg
            )
        except Exception as e:
            slice_metrics = {"error": str(e)}

    # 生态指标
    ecosystem = {}
    if compute_ecosystem and test_df is not None:
        eco_cfg = ecosystem_config or {}
        try:
            ecosystem = compute_ecosystem_metrics(
                y_true, y_pred, test_df,
                timestamp_col=timestamp_col,
                **eco_cfg
            )
        except Exception as e:
            ecosystem = {"error": str(e)}

    # =========================================================================
    # [NEW] v2.1 决策核心指标
    # =========================================================================

    misallocation_cost = {}
    capture_auc = {}
    drift = {}
    diversity = {}

    if compute_decision_metrics:
        # 错分代价
        try:
            misallocation_cost = compute_misallocation_cost(y_true, y_pred, k=0.01)
        except Exception as e:
            misallocation_cost = {"error": str(e)}

        # Capture Curve AUC
        try:
            capture_auc = compute_capture_auc(y_true, y_pred, max_k=0.10)
        except Exception as e:
            capture_auc = {"error": str(e)}

        # PSI / Drift（需要训练集预测）
        if y_pred_train is not None:
            try:
                drift = compute_drift(y_pred_train, y_pred)
            except Exception as e:
                drift = {"error": str(e)}

        # Diversity（需要 test_df）
        if test_df is not None:
            try:
                diversity = compute_diversity(
                    test_df, y_pred, k=0.01,
                    group_col='streamer_id',
                    y_true=y_true
                )
            except Exception as e:
                diversity = {"error": str(e)}

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
        # [NEW v2.0]
        prob_calibration=prob_calibration,
        slice_metrics=slice_metrics,
        ecosystem=ecosystem,
        # [NEW v2.1]
        misallocation_cost=misallocation_cost,
        capture_auc=capture_auc,
        drift=drift,
        diversity=diversity,
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
    print("Gift EVpred Metrics Module v2.2")
    print("=" * 60)
    print("Usage:")
    print("  from gift_EVpred.metrics import evaluate_model")
    print("  result = evaluate_model(y_true, y_pred, test_df, y_prob=y_prob)")
    print("  print(result.summary())")
    print("")
    print("v2.0 Features:")
    print("  - compute_calibration(): ECE + Reliability Curve")
    print("  - compute_slice_metrics(): Cold-start / Whale / Tier")
    print("  - compute_ecosystem_metrics(): Gini / Coverage / Overload")
    print("  - gini_coefficient(): Gini coefficient helper")
    print("")
    print("v2.1 Decision Metrics:")
    print("  - compute_misallocation_cost(): Regret / Wasted Exposure")
    print("  - compute_capture_auc(): nAUC@K (avoid overfitting to single K)")
    print("  - compute_drift(): PSI / KL Divergence")
    print("  - compute_diversity(): Entropy / Effective Number")
    print("")
    print("v2.2 Calibration & Naming Revision (NEW):")
    print("  - Regret naming (replaces OpportunityCost)")
    print("  - Amount vs Ratio split (AchievedRev, OracleRev, Regret, Eff, RegretPct)")
    print("  - nAUC clarification (normalized AUC is the recommended metric)")
    print("  - Overload split: U-OverTarget (user) vs S-OverLoad (streamer)")
    print("  - Tail Calibration prerequisite: requires EV-scale output")
    print("  - METRIC_SHORT_NAMES: industry-standard abbreviations")
