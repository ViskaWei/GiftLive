# 🍃 Two-Stage Model vs Direct Regression

> **Name:** Two-Stage Model Comparison  
> **ID:** `EXP-20260108-gift-allocation-03`  
> **Topic:** `gift_EVmodel` | **MVP:** MVP-1.1  
> **Author:** Viska Wei | **Date:** 2026-01-08 | **Status:** ✅

> 🎯 **Target:** 验证两段式建模 (p(x)·m(x)) 是否优于直接回归 Baseline  
> ⚠️ **结论:** 实验揭示了两种模型的本质差异：Baseline 是条件模型（给定打赏预测金额），Two-Stage 是无条件模型（预测任意点击的期望收益）

## ⚡ 核心结论速览

> **一句话**: 两段式模型在 click 数据上 PR-AUC=0.65，但与 gift-only baseline 不可直接对比；实验揭示了模型定位的根本差异

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| DG1: 两段式 vs 直接回归的增益有多大？ | ⚠️ 不可直接比较 | 两种模型解决不同问题，需重新设计对比实验 |

| 指标 | Two-Stage (Click-based) | Baseline (Gift-only) | 可比性 |
|------|------------------------|---------------------|--------|
| Top-1% Capture | 35.8% | 56.2% | ❌ 不同数据集 |
| Spearman | 0.247 | 0.891 | ❌ 不同数据集 |
| PR-AUC (Stage 1) | **0.646** | N/A | ✅ 新指标 |
| ECE (Stage 1) | **0.018** | N/A | ✅ 新指标 |

| Type | Link |
|------|------|
| 🧠 Hub | `../gift_EVmodel_hub.md` § Q1.1 |
| 🗺️ Roadmap | `../gift_EVmodel_roadmap.md` § MVP-1.1 |

---

# 1. 🎯 目标

**问题**: 验证两段式建模 (Stage1: 是否打赏, Stage2: 条件金额) 是否优于直接回归 Baseline

**验证**: DG1 - 两段式 vs 直接回归的增益有多大？

| 预期 | 判断标准 |
|------|---------|
| Top-1% Capture > 56.2% | 通过 → 确认两段式路线 |
| PR-AUC 提升 > 5% | 通过 → 分类层有价值 |

---

# 2. 🦾 算法

## 2.1 Two-Stage Model

**架构**:
$$v(x) = p(x) \cdot m(x)$$

- **Stage 1 (Classification)**: $p(x) = \Pr(Y > 0 | x)$ — 是否打赏
- **Stage 2 (Regression)**: $m(x) = \mathbb{E}[\log(1+Y) | Y > 0, x]$ — 条件金额

**训练数据**: Click 数据 (4.9M 样本, 1.93% 正样本)

## 2.2 Baseline (Direct Regression)

**架构**:
$$\hat{y} = f(x) = \text{LightGBM}(x)$$

**训练数据**: Gift 数据 (72k 样本, 100% 正样本)

---

# 3. 🧪 实验设计

## 3.1 数据

| 模型 | 数据源 | Train | Val | Test | 正样本率 |
|------|--------|-------|-----|------|---------|
| Two-Stage | Click | 1,872,509 | 1,701,600 | 1,335,406 | 1.82% |
| Baseline | Gift | 25,985 | 24,029 | 22,632 | 100% |

**⚠️ 关键差异**: 两个模型在完全不同的数据分布上训练和评估

## 3.2 模型

| Stage | 模型 | Objective | Best Iteration | 训练时间 |
|-------|------|-----------|----------------|---------|
| Stage 1 | LightGBM | binary (scale_pos_weight=53.91) | 9 | 136s |
| Stage 2 | LightGBM | regression (MAE) | 105 | 317s |

## 3.3 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| Stage 1 | PR-AUC | 精确率-召回率曲线下面积（适合不均衡） |
| Stage 1 | ECE | 期望校准误差 |
| Combined | Top-K% Capture | 预测 Top-K% 与真实 Top-K% 的重叠率 |
| Combined | Spearman | 排名相关系数 |

---

# 4. 📊 图表

### Fig 1: Top-K% Capture Comparison
![](../img/two_stage_comparison_topk.png)

**观察**:
- Two-Stage 在所有 Top-K% 指标上都低于 Baseline
- ⚠️ 但这不是公平对比：数据集不同

### Fig 2: Top-K% Capture Curve
![](../img/two_stage_topk_curve.png)

**观察**:
- Baseline 曲线明显高于 Two-Stage
- 原因：Baseline 只在打赏样本上评估，Two-Stage 在全量点击上评估

### Fig 3: Stage 1 Classification Metrics
![](../img/two_stage_prauc.png)

**观察**:
- PR-AUC = 0.646 — 在 1.93% 正样本率下表现良好
- ECE = 0.018 — 校准非常准确

### Fig 4: Stage 1 Calibration
![](../img/two_stage_calibration_stage1.png)

**观察**:
- 预测概率与实际打赏率高度吻合
- 模型概率估计可信

### Fig 5: Predicted vs Actual
![](../img/two_stage_pred_vs_actual.png)

**观察**:
- 大量样本聚集在原点（无打赏）
- 高值区域有较大方差

### Fig 6: Feature Importance
![](../img/two_stage_feature_importance.png)

**观察**:
- Stage 1 (分类): `pair_gift_count` 最重要（历史打赏次数）
- Stage 2 (回归): `pair_gift_mean` 最重要（历史平均金额）
- 两阶段的重要特征不完全相同，说明两阶段确实在学习不同的信号

---

# 5. 💡 关键洞见

## 5.1 核心发现

| # | 洞见 | 证据 | 影响 |
|---|------|------|------|
| 1 | **模型定位不同** | Baseline 训练在 gift-only，Two-Stage 训练在 click | 不能直接对比指标 |
| 2 | **分类层有效** | PR-AUC=0.65, ECE=0.018 | Stage 1 值得保留 |
| 3 | **特征分工** | 分类用 count，回归用 mean | 两阶段学到不同信号 |

## 5.2 实验设计反思

**问题**: 本实验的对比不公平，因为：
1. **数据分布不同**: Baseline 只看打赏样本，Two-Stage 看全量点击
2. **评估集不同**: 22k gift vs 1.3M click
3. **任务定义不同**: 条件预测 vs 无条件预测

**正确的对比方式**:
- 方案 A: 在 click 数据上重新训练 Baseline（直接回归 log(1+Y)，含 0 值）
- 方案 B: 在 gift 样本子集上评估 Two-Stage 的 Stage 2 输出

---

# 6. 📝 结论

## 6.1 核心发现

> **实验揭示了 Baseline 与 Two-Stage 解决的是不同问题，直接对比指标不具意义**

- ⚠️ DG1 未能关闭：需要重新设计公平对比实验
- ✅ Stage 1 分类有效: PR-AUC=0.65, ECE=0.018
- ✅ 特征分工发现: 分类和回归依赖不同特征

## 6.2 假设验证

| 假设 | 结果 | 说明 |
|------|------|------|
| 两段式优于直接回归 | ⚠️ 待验证 | 需公平对比实验 |
| 分类层有价值 | ✅ 确认 | PR-AUC=0.65 |
| 校准良好 | ✅ 确认 | ECE=0.018 |

## 6.3 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 公平对比 | 在 click 数据上重新训练 Baseline | 🔴 |
| 或者 | 在 gift 子集上评估两模型 | 🔴 |
| 分析 | 理解 Stage 1 vs Stage 2 的特征差异 | 🟡 |

---

# 7. 📎 附录

## 7.1 数值结果

| 指标 | Two-Stage | Baseline (参考) |
|------|-----------|----------------|
| **Stage 1** | | |
| PR-AUC | 0.646 | N/A |
| ECE | 0.018 | N/A |
| Log Loss | 0.053 | N/A |
| **Combined** | | |
| MAE(log) | 0.081 | 0.263 |
| RMSE(log) | 0.411 | 0.662 |
| Spearman | 0.247 | 0.891 |
| Top-1% Capture | 35.8% | 56.2% |
| Top-5% Capture | 40.4% | 76.3% |
| Top-10% Capture | 27.4% | 82.3% |
| NDCG@100 | 0.371 | 0.716 |

## 7.2 训练配置

```yaml
stage1:
  model: LightGBM
  objective: binary
  scale_pos_weight: 53.91
  num_leaves: 31
  learning_rate: 0.05
  best_iteration: 9
  training_time: 136s

stage2:
  model: LightGBM
  objective: regression
  metric: mae
  num_leaves: 31
  learning_rate: 0.05
  best_iteration: 105
  training_time: 317s
```

## 7.3 执行记录

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/train_two_stage.py` |
| 日志 | `logs/two_stage_20260108.log` |
| 模型 Stage 1 | `experiments/gift_allocation/models/two_stage_clf_20260108.pkl` |
| 模型 Stage 2 | `experiments/gift_allocation/models/two_stage_reg_20260108.pkl` |
| 结果 JSON | `experiments/gift_allocation/results/two_stage_results_20260108.json` |

---

> **实验完成时间**: 2026-01-08 13:59:13
