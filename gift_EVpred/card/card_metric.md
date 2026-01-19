# 📐 知识卡片：gift_EVpred 指标体系

> **ID:** KC-20260119-metrics
> **Topic:** gift_EVpred 评估指标
> **Author:** Viska Wei | **Date:** 2026-01-19

---

## 1. 指标体系总览

```
┌─────────────────────────────────────────────────────────────┐
│                    gift_EVpred 指标体系                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🎯 主指标（选模型/调参）                              │   │
│  │                                                     │   │
│  │   RevCap@K (K=1% 为主)                              │   │
│  │   = Top K% 捕获的收入占总收入的比例                   │   │
│  │                                                     │   │
│  │   目标：> 50%                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🔍 诊断指标（理解模型行为）                           │   │
│  │                                                     │   │
│  │   • Whale Recall@K   → 找到多少比例的大哥？           │   │
│  │   • Whale Precision@K → Top K% 池子里大哥占多少？     │   │
│  │   • Avg Revenue@K    → 池子人均真实收入（池子质量）    │   │
│  │   • Tail Calibration → 预测金额/实际金额（避免高估）   │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 📊 稳定性指标（评估可靠性）                           │   │
│  │                                                     │   │
│  │   • CV (Coefficient of Variation) → 目标 < 10%      │   │
│  │   • 95% Bootstrap CI              → 置信区间         │   │
│  │   • 按天/按直播间分析             → 识别异常          │   │
│  └─────────────────────────────────────────────────────┘   │
│                          ↓                                  │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 🔧 可选指标（特定场景）                               │   │
│  │                                                     │   │
│  │   • NDCG@K → Top 内微调（分配需要精细排序时）         │   │
│  │   • Spearman → 整体排序相关性（参考，非主指标）       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 指标详解

### 2.1 主指标：RevCap@K (Revenue Capture)

**定义**：
$$\text{RevCap@K} = \frac{\sum_{i \in \text{Top K\%}} y_i}{\sum_i y_i}$$

**含义**：如果只能曝光 Top K% 的用户，能捕获多少比例的收入？

**为什么是主指标？**
- 直接对齐业务目标：收入最大化
- 重尾分布下，头部贡献大部分收入
- 易于向业务方解释："选 1% 的人，能抓到 50% 的钱"

**K 值选择**：
| K | 适用场景 | 当前值 |
|---|---------|--------|
| 0.1% | 极端头部（VIP 专属） | 21.9% |
| **1%** | **主指标**（平衡精度和覆盖） | **52.6%** |
| 5% | 广覆盖场景 | 64.7% |

**Normalized RevCap@K**：
$$\text{NormRevCap@K} = \frac{\text{RevCap@K (Model)}}{\text{RevCap@K (Oracle)}}$$

衡量模型接近理论上限的程度。Oracle = 按真实 y 排序。

---

### 2.2 诊断指标

#### 2.2.1 Whale Recall@K（大哥召回率）

**定义**：
$$\text{Whale Recall@K} = \frac{|\text{Top K\%} \cap \text{Whales}|}{|\text{Whales}|}$$

**含义**：Top K% 预测捕获了多少比例的真实 whale？

**Whale 定义**：y >= P90 of gifters（本实验 = 100 元）

**用途**：
- 评估"有没有把大哥抓进池子"
- Recall 低说明漏掉了很多大哥
- 当前值 @1% = 35.2%（仅捕获 1/3）

#### 2.2.2 Whale Precision@K（大哥精确率）

**定义**：
$$\text{Whale Precision@K} = \frac{|\text{Top K\%} \cap \text{Whales}|}{|\text{Top K\%}|}$$

**含义**：Top K% 池子里有多少比例是真 whale？

**用途**：
- 评估"池子纯度"
- 高 Precision = 选的人里大哥多
- 当前值 @1% = 5.9%（基准 0.169%，提升 35 倍）

#### 2.2.3 Avg Revenue@K（人均收入）

**定义**：
$$\text{AvgRev@K} = \frac{\sum_{i \in \text{Top K\%}} y_i}{|\text{Top K\%}|}$$

**含义**：Top K% 样本的人均真实收入。

**用途**：
- 评估"池子质量"
- 用于分配层估算预算
- 当前值 @1% = 78.2 元（全局均值 1.5 元）

#### 2.2.4 Tail Calibration（尾部校准）

**定义**：
$$\text{Sum Ratio} = \frac{\sum_{i \in \text{Top K\%}} \hat{y}_i}{\sum_{i \in \text{Top K\%}} y_i}$$

**含义**：预测金额 vs 实际金额的比值。

**用途**：
- 避免分配时高估/低估
- 目标 ≈ 1.0
- 当前值 = 2.2-2.5x（高估，但不影响排序）

**注意**：Calibration ≠ Ranking。高估不影响排序能力，但影响分配预算估计。

---

### 2.3 稳定性指标

#### CV (Coefficient of Variation)

**定义**：CV = Std / Mean

**含义**：相对标准差，衡量波动大小。

**目标**：< 10%

**当前值**：10.8%（边界）

#### 95% Bootstrap CI

**含义**：通过 Bootstrap 重采样估计的 95% 置信区间。

**当前值**：[48.2%, 56.5%]

**用途**：
- 评估指标的可靠性
- CI 越窄越稳定

---

### 2.4 可选指标

#### NDCG@K (Normalized DCG)

**定义**：
$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}, \quad \text{DCG@K} = \sum_{i=1}^{K} \frac{y_i}{\log_2(i+1)}$$

**用途**：
- 评估 Top K 内部的排序质量
- 适用于分配需要精细排序的场景
- **不作为主指标**，仅在 Top 内微调时使用

#### Spearman Correlation

**用途**：
- 评估整体排序相关性
- **⚠️ 注意**：Spearman 和 RevCap 可能负相关！
- 高 Spearman ≠ 高 RevCap（参考即可，非决策依据）

---

## 3. 训练 Loss 建议

### 3.1 Baseline：MSE on raw Y

```python
loss = MSE(y_true, y_pred)  # 全量 click，预测 raw Y
```

**优点**：
- 简单有效
- 已验证 RevCap@1% = 52.6%

### 3.2 进阶：加权 MSE

```python
# 让 tail 更重要
weight = (1 + y_true) ** alpha  # alpha ∈ [0.3, 1.0]
loss = MSE(y_true, y_pred, sample_weight=weight)
```

**预期**：
- 提升 Whale Recall
- 提升低 K 值 RevCap

### 3.3 两阶段（可选）

1. **第一阶段**：全量 click 上回归 raw Y（筛选）
2. **第二阶段**：Top K% 上 ranking loss（微调排序）

```python
# 第二阶段：NDCG loss（仅在需要精细排序时）
loss = -NDCG(y_true[top_k], y_pred[top_k])
```

---

## 4. 使用指南

### 4.1 完整评估

```python
from gift_EVpred.metrics import evaluate_model

# 完整评估
result = evaluate_model(
    y_true=y_test,
    y_pred=y_pred,
    test_df=test_df,          # 可选，用于稳定性
    whale_threshold=100,       # 或自动计算 P90
    compute_stability=True,
    compute_ndcg=False,        # 仅在需要时开启
)

# 打印摘要
print(result.summary())

# 保存结果
result.to_json('results/eval_results.json')

# 获取特定指标
print(f"RevCap@1%: {result.revcap_1pct:.1%}")
print(f"Whale Recall@1%: {result.whale_recall_1pct:.1%}")
```

### 4.2 快速评估（训练过程中）

```python
from gift_EVpred.metrics import quick_eval

# 轻量评估
metrics = quick_eval(y_val, y_pred, whale_threshold=100, k=0.01)
print(f"RevCap: {metrics['revcap']:.1%}")
print(f"Whale Recall: {metrics['whale_recall']:.1%}")
```

### 4.3 单指标计算

```python
from gift_EVpred.metrics import (
    revenue_capture_at_k,
    whale_recall_at_k,
    whale_precision_at_k,
    avg_revenue_at_k,
    tail_calibration,
)

# RevCap
revcap = revenue_capture_at_k(y_true, y_pred, k=0.01)

# Whale Recall
recall = whale_recall_at_k(y_true, y_pred, whale_threshold=100, k=0.01)

# Calibration
calib = tail_calibration(y_true, y_pred)
```

### 4.4 生成 Markdown 表格

```python
from gift_EVpred.metrics import evaluate_model, format_metrics_table

result = evaluate_model(y_test, y_pred, test_df)
table = format_metrics_table(result)
print(table)
```

输出：
```
| K | RevCap | Norm | Whale Recall | Whale Prec | Avg Rev |
|---|--------|------|--------------|------------|---------|
| 0.1% | 21.9% | 26.1% | 9.9% | 16.7% | 325.9 |
| 1% | 52.6% | 52.8% | 35.2% | 5.9% | 78.2 |
| 5% | 64.7% | 64.7% | 52.7% | 1.8% | 19.2 |
```

---

## 5. 指标选择决策树

```
问题：选什么模型/参数？
    │
    ├── 比较 RevCap@1%（主指标）
    │       │
    │       ├── A > B by > 1% → 选 A
    │       ├── 差异 < 1%    → 看诊断指标
    │       │       │
    │       │       ├── Whale Recall 谁高？ → 选高的
    │       │       ├── Calibration 谁接近 1？ → 选接近的
    │       │       └── 稳定性 CV 谁低？ → 选低的
    │       │
    │       └── A < B → 选 B

问题：模型表现够不够好？
    │
    ├── RevCap@1% > 50%？ ✅ 达标
    ├── Whale Recall@1% > 40%？ 当前 35.2% ❌ 需提升
    ├── Stability CV < 10%？ 当前 10.8% ⚠️ 边界
    └── Calibration ≈ 1？ 当前 2.2x ⚠️ 高估

问题：需要 NDCG 吗？
    │
    ├── 分配层需要精细排序？ → 是 → 计算 NDCG@100
    └── 只需筛选池子？ → 否 → 不需要
```

---

## 6. 常见陷阱

| ⚠️ 陷阱 | 原因 | 正确做法 |
|---------|------|---------|
| 用 Spearman 选模型 | Spearman ≠ RevCap（可能负相关） | 用 RevCap@K |
| 用 AUC 选模型 | AUC 高 ≠ RevCap 高（分类丢失金额信息） | 用 RevCap@K |
| 只看 @1% | 曲线形态更重要 | 画 RevCap 曲线 |
| 过度关注 Calibration | 排序任务 Calibration 优先级低 | 先看 RevCap |
| 忽视稳定性 | 单日异常可能影响决策 | 按天分析 + CI |
| 用 log(1+Y) 优化收入 | log 压缩 whale 信号 | 用 raw Y |

---

## 7. 验收标准

| 指标 | 当前值 | 目标 | 状态 |
|------|--------|------|------|
| RevCap@1% | 52.6% | > 50% | ✅ |
| Whale Recall@1% | 35.2% | > 40% | ❌ 需提升 |
| Whale Precision@1% | 5.9% | > 5% | ✅ |
| Avg Revenue@1% | 78.2 元 | 越高越好 | ✅ |
| Stability CV | 10.8% | < 10% | ⚠️ 边界 |
| Calibration | 2.2x | ≈ 1.0 | ⚠️ 高估 |

---

## 8. API 参考

### 模块导入

```python
from gift_EVpred.metrics import (
    # 主评估函数
    evaluate_model,        # 完整评估
    quick_eval,            # 快速评估

    # 单指标函数
    revenue_capture_at_k,  # RevCap@K
    whale_recall_at_k,     # Whale Recall@K
    whale_precision_at_k,  # Whale Precision@K
    avg_revenue_at_k,      # Avg Revenue@K
    gift_rate_at_k,        # Gift Rate@K
    tail_calibration,      # Tail Calibration
    normalized_dcg_at_k,   # NDCG@K

    # 曲线计算
    compute_revcap_curve,
    compute_oracle_revcap_curve,
    compute_all_metrics_at_k,

    # 稳定性
    compute_stability_by_day,
    compute_stability_summary,
    bootstrap_ci,

    # 辅助
    get_whale_threshold,
    format_metrics_table,

    # 结果类
    EvalResult,
)
```

### 函数签名

```python
def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_df: Optional[pd.DataFrame] = None,
    whale_threshold: Optional[float] = None,
    whale_percentile: int = 90,
    k_values: List[float] = None,
    compute_stability: bool = True,
    compute_ndcg: bool = False,
    timestamp_col: str = 'timestamp'
) -> EvalResult
```

---

## 9. 文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| Metrics 模块 | `gift_EVpred/metrics.py` | 统一评估代码 |
| 知识卡片 | `gift_EVpred/card/card_metric.md` | 本文档 |
| 实验报告 | `gift_EVpred/exp/exp_metrics_landing_20260119.md` | 指标落地实验 |

---

> **更新时间**: 2026-01-19
