# Metrics v2.2 指标体系完整手册 — 5min 汇报

> **Experiment**: EVpred-20260119-metrics-v22  \
> **Author**: Viska Wei  \
> **Date**: 2026-01-19  \
> **Language**: 中文

---

## 7 层 38 指标，从选模型到上线监控全覆盖

- **目的**：为 gift_EVpred 建立统一、全面、可复用的评估指标体系，消除口径混乱
- **X 公式**：$X := \underbrace{\text{Metrics Framework v2.2}}_{\text{是什么}}\;\xrightarrow{\text{7 层分级 + 多粒度}}\;\underbrace{\text{标准化模型评估}}_{\text{统一口径}}$
- **结论**：7 层 **38 个指标**，覆盖 L1 主指标 → L7 决策核心，使用 `evaluate_model()` 一键评估

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| 🫐 输入 | $y_{true}$ | 真实标签 | ndarray (N,), 打赏金额 |
| 🫐 输入 | $y_{pred}$ | 预测值 | ndarray (N,), Ridge 预测 |
| 🫐 输入 | $\mathcal{D}_{test}$ | 测试集 | DataFrame (user_id, streamer_id...) |
| 🫐 输出 | $M$ | MetricsResult | 38+ 指标, .summary(), .to_json() |
| 📊 指标 | L1-L7 | 7 层体系 | RevCap, Whale, PSI, EffN... |
| 🍁 基线 | v2.1 | 旧版 | 缺金额版 Regret, nAUC 不规范 |

<details>
<summary><b>逐字稿</b></summary>

这个实验的目标是：为 gift_EVpred 建立统一的评估指标体系，解决之前实验口径混乱、难以对比的问题。

核心产出是一个 7 层 38 个指标的完整框架：
- L1 主指标：RevCap@K，用于选模型
- L2 诊断：Whale 七件套，理解模型行为
- L3 稳定性：CV、Bootstrap CI
- L4 校准：ECE、Tail Ratio
- L5 切片：10 个场景切片，分析公平性
- L6 生态：Gini、Coverage，监控分配健康
- L7 决策核心：Regret、PSI、EffN，支持上线决策

使用方式非常简单：调用 `evaluate_model()` 一键评估，输入是真实标签、预测值和测试集 DataFrame，输出是包含 38 个指标的 MetricsResult 对象。

</details>

---

## 7 层架构 + 一键评估

- **架构**：

| 层级 | 用途 | 指标数 | 核心指标 |
|------|------|--------|---------|
| L1 主指标 | 选模型 | 3 | RevCap@K, nRevCap@K |
| L2 诊断 | 理解行为 | 6 | WRecLift, WRevCap |
| L3 稳定性 | 评估鲁棒 | 5 | CV, Bootstrap CI |
| L4 校准 | 预测准度 | 3 | ECE, TailRatio |
| L5 切片 | 公平分析 | 10 | ColdStart, Whale |
| L6 生态 | 分配护栏 | 7 | Gini, Coverage |
| **L7 决策核心** | **上线判断** | **4 类** | **Regret, PSI, EffN** |

- **核心公式**：

$$
\text{RevCap@K} = \frac{\sum_{i \in \text{Top}_K} y_i}{\sum_i y_i}, \quad
\text{Regret@K} = \text{OracleRev@K} - \text{AchievedRev@K}
$$

$$
\text{nAUC@K} = \frac{\text{CapAUC@K}}{\text{OracleAUC@K}} \in [0,1], \quad
\text{PSI} = \sum_b (p_{test} - p_{train}) \ln\frac{p_{test}}{p_{train}}
$$

- **使用示例**：
```python
result = evaluate_model(y_true, y_pred, test_df, compute_decision_metrics=True)
# → 输出: {'RevCap@1%': 0.514, 'WRecLift@1%': 34.6, 'PSI': 0.155, ...}
result.to_json('results/metrics.json')
```

<details>
<summary><b>逐字稿</b></summary>

指标体系分 7 层，每层有明确职责：

L1 是主指标，RevCap@K，用于选模型和调参——Top K% 预测捕获多少收入。

L2 是诊断指标，Whale 七件套，帮助理解模型行为——抓大额事件的能力如何。

L3 到 L6 是稳定性、校准、切片、生态，分别评估鲁棒性、预测准确度、公平性、分配健康。

L7 是决策核心，这是 v2.2 新增的重点：
- Regret：错分代价，Oracle 减去实际收入，用金额量化
- nAUC：归一化 Capture AUC，避免只优化单点 K 导致过拟合
- PSI：分布漂移，判断是否可上线
- EffN：有效分组数，衡量多样性

使用非常简单，调用 `evaluate_model()` 一行代码，38 个指标全部算出来。

</details>

---

## 决策：v2.2 作为统一标准，≥90% 覆盖率强制要求

- **结论**：「7 层 38 指标体系闭环，口径规范化，可作为 gift_EVpred 统一评估标准」
    - 机制层：从 L1 选模型到 L7 上线决策，覆盖 ML 全生命周期
    - 证据层：38+ 指标，评估时间 ~30s（1.4M 样本），工程可行
- **决策**：✅ **所有实验必须使用 v2.2 评估**
    - 覆盖率要求 ≥ 90%（约 32/35 指标）
    - 不满足覆盖率的实验报告不得合入 roadmap
- **v2.2 vs v2.1 变更**：

| 变更 | v2.1 | v2.2 |
|------|------|------|
| Regret 命名 | opportunity_cost | regret（工业界通用） |
| 金额/比例 | 只有比例版 | 金额版（元）+ 比例版 |
| nAUC | 范围不清 | 明确 ∈ [0,1] |
| Overload | 单一指标 | U-OverTarget + S-OverLoad |

- **下一步**：
    - 🔴 **P0**：所有新实验使用 v2.2
    - 🟡 **P1**：补充 DW-ECE（金额加权 ECE）、Fairness Gap
    - 🟢 **P2**：性能优化（大数据量场景）

<details>
<summary><b>逐字稿</b></summary>

核心结论：7 层 38 个指标的体系已经闭环，口径规范化完成，可以作为 gift_EVpred 的统一评估标准。

决策非常明确：从现在起，所有实验必须使用 v2.2 进行评估。
而且有硬性要求：指标覆盖率必须达到 90%，也就是大约 32 个指标必须覆盖。不满足这个要求的实验报告不允许合入 roadmap。

v2.2 相比 v2.1 有几个重要变更：
- Regret 改名，从 opportunity_cost 改成 regret，和工业界对齐
- 同时提供金额版和比例版，金额版用于业务沟通，比例版用于跨实验对比
- nAUC 规范化到 0-1 区间
- Overload 拆分成用户侧和主播侧两个指标

下一步：P0 是所有新实验必须用 v2.2；P1 是补充 DW-ECE 和 Fairness Gap；P2 是性能优化。

</details>
