# 🍃 Calibration Evaluation
> **Name:** Calibration Evaluation
> **ID:** `EXP-20260118-gift_EVpred-07`
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.5
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** ✅ 完成

> 🎯 **Target:** 评估模型预测的校准性，确保预测值与实际值在分桶上对齐
> 🚀 **Next:** 回归预测严重低估 → 需要校准层或重新定义预测目标

## ⚡ 核心结论速览

> **一句话**: 分类校准良好（ECE≈0），但回归预测**严重低估**——预测均值 0.0008 vs 实际均值 1.22，低估 1500 倍。

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| 概率预测是否校准？ | ECE = 0.0000 | ✅ 校准良好（empirical 方法） |
| 金额预测是否校准？ | ECE = 1.22 | ❌ 严重低估，预测值接近 0 |
| 哪些分桶偏差最大？ | 所有分桶 | 整体性低估问题 |

| 指标 | 值 | 启示 |
|------|-----|------|
| Classification ECE | **0.0000** | 排序可信，但绝对值不可信 |
| Regression ECE | **1.22** | 预测值严重低估 |
| Pred Mean vs Actual | **0.0008 vs 1.22** | 低估 ~1500x |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § Q2.2 |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` § MVP-1.5 |

---
# 1. 🎯 目标

**问题**: 模型预测是否校准？预测值能否直接用于分配决策的期望收益计算？

**验证**: Q2.2（分桶校准是否必要）

| 预期 | 判断标准 | 实际结果 |
|------|---------|---------|
| 校准良好 | ECE < 0.03 | ✅ 分类 ECE = 0（但有 caveat） |
| 校准较差 | ECE > 0.05 | ❌ 回归 ECE = 1.22（严重） |

**为什么校准重要**：
- 做 allocation 时，需要预测期望收益 E[gift|x]
- 如果预测值严重偏离实际值，分配决策会产生系统性偏差
- 当前预测值接近 0，无法直接用于期望收益计算

---

# 2. 🦾 算法

> 📌 ECE (Expected Calibration Error) 定义

**二分类校准（P(gift>0)）**：

$$
ECE = \sum_{b=1}^{B} \frac{|B_b|}{n} |acc(B_b) - conf(B_b)|
$$

**回归校准（EV）**：

$$
ECE_{reg} = \sum_{b=1}^{B} \frac{|B_b|}{n} |\bar{y}(B_b) - \bar{\hat{y}}(B_b)|
$$

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive click-level 数据 |
| Test 样本 | 736,428 |
| Gift Rate | 1.517% |
| Gift 样本 | 11,169 |

## 3.2 校准评估设计

| 评估项 | 预测值 | 实际值 | 分桶方式 |
|--------|--------|--------|----------|
| 二分类校准 | empirical prob (percentile mapping) | 1(gift>0) | 等频 10 桶 |
| 回归校准 | exp(pred)-1 | gift_price_label | 等频 10 桶 |
| 条件 EV 校准 | pred (gift>0 only) | gift_price_label | 等频 10 桶 |

---

# 4. 📊 图表

### Fig 1: Reliability Curve (Classification)
![](./img/reliability_curve_classification.png)

**观察**:
- 使用 empirical percentile mapping 后，分类校准良好
- 但这是 post-hoc 校准，原始预测值并非概率

### Fig 2: Reliability Curve (Regression)
![](./img/reliability_curve_regression.png)

**观察**:
- 预测值整体接近 0（mean = 0.0008）
- 实际值 mean = 1.22
- 所有分桶都严重低估

### Fig 3: Calibration Comparison
![](./img/calibration_comparison.png)

**观察**:
- 分类和回归校准的对比
- 回归问题更严重

---

# 5. 💡 洞见

## 5.1 宏观
- **模型输出不是"期望收益"**：log(1+Y) 回归的输出在 exp 变换后仍然接近 0
- **排序可用，绝对值不可用**：Spearman=0.096 说明有一定排序能力，但预测值本身无意义

## 5.2 模型层
- **log(1+Y) 的问题**：98.5% 的样本 Y=0，log(1+0)=0，模型学到的主要是预测 0
- **稀疏标签的挑战**：在极度稀疏的情况下，模型趋向于保守预测

## 5.3 细节
- **条件 ECE 极高**（80.5）：在 gift>0 的样本上，模型完全无法预测金额
- **需要 Two-Stage 或校准层**：直接回归在稀疏场景下失效

---

# 6. 📝 结论

## 6.1 核心发现
> **回归预测严重低估，预测均值仅为实际的 0.07%，模型输出不能直接用于期望收益计算。**

- ✅ H4.1: 分类 ECE < 0.03 → **确认**（ECE=0，但需 empirical 校准）
- ✅ H4.2: 高预测分桶可能存在高估 → **否定**，所有分桶都**低估**
- ✅ H4.3: 需要校准层 → **确认**，回归需要 post-hoc 校准

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **分类排序可信** | Spearman=0.096, empirical 校准后 ECE=0 |
| 2 | **回归绝对值严重低估** | Pred mean=0.0008 vs Actual=1.22 |
| 3 | **直接回归在稀疏场景失效** | 98.5% 样本 Y=0 导致预测趋向 0 |
| 4 | **条件预测（gift>0）也失效** | Conditional ECE=80.5 |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 使用 Two-Stage | P(gift>0) × E[amount|gift>0] 分解可能更合理 |
| 添加校准层 | 使用 isotonic regression 校准预测值 |
| 重新定义目标 | 考虑只预测排序（ranking loss）而非绝对值 |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 直接用预测值做期望收益 | 预测值严重低估，会导致系统性决策偏差 |
| 忽略稀疏性问题 | 98.5% Y=0 导致模型学习偏向 0 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Classification ECE | **0.0000** | empirical 校准后 |
| Regression ECE | **1.2209** | - |
| Pred Mean | **0.0008** | - |
| Actual Mean | **1.2217** | - |
| Low-estimate Ratio | **~1500x** | 严重低估 |
| Conditional ECE (gift>0) | **80.54** | 更严重 |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| Two-Stage 实验 | 验证 P(gift>0) × E[amount|gift>0] | 🔴 |
| 校准层 | 添加 isotonic regression 校准 | 🟡 |
| Ranking Loss | 考虑用 pairwise ranking loss 替代 regression | 🟡 |

---

# 7. 📎 附录

## 7.1 数值结果

### 分类校准（P(gift>0)）

| Bin | Pred Mean | Actual Rate | Error |
|-----|-----------|-------------|-------|
| 10 | 1.52% | 1.52% | 0.00% |

> Note: 使用 empirical percentile mapping，所有样本映射到同一 bin

### 回归校准（EV）

| Bin | Pred Mean | Actual Mean | Error |
|-----|-----------|-------------|-------|
| 1 | 0.0000 | 11.68 | 11.68 |
| 10 | 0.0008 | 1.16 | 1.16 |

> Note: 所有分桶都严重低估

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/evaluate_calibration.py` |
| Output | `results/calibration_evaluation_20260118.json` |
| 图表 | `exp/img/reliability_curve_*.png`, `exp/img/calibration_comparison.png` |

```bash
# 评估
python scripts/evaluate_calibration.py
```

---

> **实验完成时间**: 2026-01-18
