# 🍃 Three-Stage EV Prediction: Non-gifter / Small / Whale
> **Name:** Three-Stage EV Prediction
> **ID:** `EXP-20260118-gift_EVpred-03`
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.1
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** ✅

> 🎯 **Target:** 将 Two-Stage 扩展为 Three-Stage，专注预测"超级大哥"（Whale）
> 🚀 **Next:** Whale-only 策略 RevCap@1%=43.60%，但后续发现 **Direct + raw Y 更优（52.68%）**

## ⚡ 核心结论速览

> **一句话**: Whale-only 策略 `P(whale)*E[whale]` 在 RevCap@1% 达到 **43.60%**，比 Two-Stage 提升 36.2%

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H1.3: Three-Stage 是否优于 Two-Stage？ | ✅ +36.2% | Whale-only 策略最优 |
| 最佳阈值？ | P90 = 100元 | 专注"超级大哥" |

| 指标 | Two-Stage | Three-Stage (P90) | 提升 |
|------|-----------|-------------------|------|
| **Revenue Capture @1%** | 32.01% | **43.60%** | **+36.2%** |
| Revenue Capture @5% | 54.62% | 62.91% | +15.2% |
| Gift Rate @1% | 23.60% | 19.01% | -19.4% |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § Q1.3 |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` § MVP-1.1 |
| 📊 Baseline | `exp/exp_baseline_day_frozen_20260118.md` |

---
# 1. 🎯 目标

**问题**: Two-Stage 模型将所有 gifters 视为一个群体，但实际上存在"小额打赏者"（给 1-2 元）和"大哥/Whale"（大额打赏）的区分。能否通过三阶段建模更精准地捕获高价值用户？

**验证**: H1.3 - Three-Stage 是否在 Revenue Capture @1% 上优于 Two-Stage？

**动机分析**:
- 当前 Two-Stage: EV = P(gift>0) × E[gift | gift>0]
- 问题：E[gift | gift>0] 受小额打赏者主导，难以区分 Whale
- Three-Stage 思路：将 gifters 进一步分为 Small / Whale

| 预期 | 判断标准 |
|------|---------|
| 正常 | RevCap@1% > 32.01% (Two-Stage) |
| 异常 | RevCap@1% ≤ 32.01%，说明分类边界不够好 |

---

# 2. 🦾 算法

## 2.1 Two-Stage (Baseline)

$$\text{EV} = P(\text{gift}>0) \times E[\text{gift} | \text{gift}>0]$$

## 2.2 Three-Stage (原始设计)

$$\text{EV} = P_1 \times (1-P_2) \times E_{small} + P_1 \times P_2 \times E_{whale}$$

- Stage 1: P(gift > 0)
- Stage 2: P(whale | gift > 0)
- Stage 3: E[gift | small] 和 E[gift | whale]

## 2.3 Whale-only (最终方案) ⭐

**核心洞察**：对于 Revenue Capture 任务，我们只关心"大哥"，不需要准确估计小额打赏者的 EV。

$$\text{Score} = P(\text{gift}>0) \times P(\text{whale} | \text{gift}>0) \times E[\text{gift} | \text{whale}]$$

简化为：
$$\text{Score} = P(\text{whale}) \times E[\text{whale}]$$

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive |
| 路径 | `data/KuaiLive/` |
| Train/Val/Test | 1,629,415 / 1,717,199 / 1,409,533 |
| 特征维度 | 31 (Day-Frozen) |
| Gift Rate | Train 1.40%, Test 1.68% |

## 3.2 Gift 分布

| 分位数 | 阈值 | Whale 占比 (in Gifters) |
|--------|------|------------------------|
| P50 | 2元 | 57.4% |
| P60 | 4元 | 41.9% |
| P70 | 10元 | 31.0% |
| P75 | 10元 | 31.0% |
| P80 | 20元 | 20.2% |
| **P90** | **100元** | **10.0%** |

## 3.3 模型

| 阶段 | 任务 | 模型 | 训练数据 |
|------|------|------|---------|
| Stage 1 | P(gift > 0) | LogisticRegression | 全量 (1.6M) |
| Stage 2 | P(whale \| gifter) | LogisticRegression | 仅 gifters (22K) |
| Stage 3 | E[gift \| whale] | Ridge | 仅 whales (~2.3K for P90) |

## 3.4 训练

| 参数 | 值 |
|------|-----|
| seed | 42 |
| 标签窗口 | click 后 1h |
| 评估集 | Test (Day 15-21) |
| 主指标 | Revenue Capture @1%, @5% |

---

# 4. 📊 结果

## 4.1 不同阈值对比

| Threshold | Value | Whale% | RevCap@1% | RevCap@5% | GiftRate@1% |
|-----------|-------|--------|-----------|-----------|-------------|
| P50 | 2元 | 57.4% | 32.02% | 56.14% | 23.38% |
| P60 | 4元 | 41.9% | 36.15% | 58.72% | 22.72% |
| P70 | 10元 | 31.0% | 37.86% | 60.76% | 22.04% |
| P80 | 20元 | 20.2% | 41.88% | 62.96% | 21.09% |
| **P90** | **100元** | **10.0%** | **43.60%** | **62.91%** | **19.01%** |

**发现**：阈值越高（越专注于超级大哥），RevCap@1% 越高！

## 4.2 不同 EV 公式对比 (P90 阈值)

| Method | RevCap@1% | RevCap@5% | GiftRate@1% | Spearman |
|--------|-----------|-----------|-------------|----------|
| Two-Stage (baseline) | 32.01% | 54.62% | 23.60% | 0.0997 |
| Three-Stage 标准 | 29.04% | 53.43% | 24.21% | 0.1128 |
| **P(whale)*E[whale]** | **43.60%** | **62.91%** | 19.01% | 0.0681 |
| P(gift)*E[gift] + P(whale)*E[whale] | 34.76% | 57.92% | 23.06% | 0.0973 |
| P(gift)*E[gift]*(1+P(whale)) | 34.27% | 57.43% | 23.17% | 0.0986 |

---

# 5. 💡 洞见

## 5.1 宏观

- **Whale-only 策略完胜**：专注预测"超级大哥"比预测全体 gifters 更有效
- **阈值越高越好（到 P90）**：说明 top 10% 的大哥贡献了绝大部分 revenue

## 5.2 模型层

- **P(whale|gift) AUC = 0.66**：在 gifters 中区分大小额的能力中等
- **Spearman 下降但 RevCap 上升**：说明整体排序不重要，关键是头部精准

## 5.3 细节

- **GiftRate@1% 下降**（23.60% → 19.01%）：Top-1% 中 gifters 变少
- **但 RevCap 上升**：说明这些 gifters 都是"大哥"，单笔金额高

---

# 6. 📝 结论

## 6.1 核心发现

> **Whale-only 策略 `P(whale)*E[whale]` 在 RevCap@1% 达到 43.60%，比 Two-Stage 提升 36.2%**

- ✅ H1.3: Three-Stage（Whale-only 变体）显著优于 Two-Stage
- 关键：使用 P90=100元 作为 Whale 阈值

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **专注大哥最有效** | P90 阈值 RevCap 最高 |
| 2 | **Spearman 不重要** | Spearman 下降但 RevCap 上升 |
| 3 | **GiftRate 不等于 Revenue** | GiftRate 下降但捕获更多 revenue |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 目标对齐 | 如果目标是 Revenue，就专注高价值用户 |
| 阈值选择 | P90（top 10% gifters）是好的分界点 |
| 评估指标 | 用 RevCap，不用 GiftRate 或 Spearman |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 用 P50 分类 | 小额打赏者噪音大，稀释信号 |
| 优化 Spearman | 与 Revenue 目标不一致 |
| 标准 Three-Stage | 反而不如 Two-Stage |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| **Best RevCap@1%** | **43.60%** | Whale-only, P90 |
| Two-Stage RevCap@1% | 32.01% | Baseline |
| 相对提升 | **+36.2%** | - |
| Whale 阈值 | 100元 | P90 |

## 6.5 后续发展

> ⚠️ **注意**：本实验完成后，发现 **Direct + raw Y** 方案更优（RevCap@1%=52.68%），超过 Whale-only（43.60%）。
>
> 原因：raw Y 作为预测目标已经隐式让模型学会"找大哥"，不需要显式的 Three-Stage 分类。

| 方向 | 结论 |
|------|------|
| Whale-only 策略 | ❌ 被 Direct + raw Y 超越 |
| 推荐方案 | ✅ Direct + raw Y（见 exp_raw_vs_log） |

---

# 7. 📎 附录

## 7.1 数值结果

| 配置 | Spearman | RevCap@1% | RevCap@5% |
|------|----------|-----------|-----------|
| Two-Stage (Baseline) | 0.0997 | 32.01% | 54.62% |
| Three-Stage 标准 (P50) | 0.1128 | 29.04% | 53.43% |
| **Whale-only (P90)** | 0.0681 | **43.60%** | 62.91% |

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 数据处理 | `gift_EVpred/data_utils.py` |
| 结果文件 | `results/three_stage_optimized_20260118.json` |

```python
# Whale-only 策略实现
from gift_EVpred.data_utils import prepare_dataset, get_feature_columns

train_df, val_df, test_df = prepare_dataset()
feature_cols = get_feature_columns(train_df)

# P90 阈值
WHALE_THRESHOLD = 100.0  # P90

# 定义 whale
train_df['is_whale'] = (train_df['target_raw'] >= WHALE_THRESHOLD).astype(int)

# Stage 1: P(gift > 0)
clf_gift = LogisticRegression()
clf_gift.fit(X_train, train_df['is_gift'])
p_gift = clf_gift.predict_proba(X_test)[:, 1]

# Stage 2: P(whale | gift > 0)
clf_whale = LogisticRegression()
clf_whale.fit(X_gifters_train, train_df.loc[gifters_mask, 'is_whale'])
p_whale = clf_whale.predict_proba(X_test)[:, 1]

# Stage 3: E[whale]
reg_whale = Ridge()
reg_whale.fit(X_whale_train, train_df.loc[whale_mask, 'target'])
e_whale = reg_whale.predict(X_test)

# Final score
score = p_gift * p_whale * e_whale
```

---

> **实验完成时间**: 2026-01-18
