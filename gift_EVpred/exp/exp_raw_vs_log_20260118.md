# 🍃 Raw Y vs Log(1+Y): 预测目标选择实验
> **Name:** Raw Y vs Log(1+Y) Comparison
> **ID:** `EXP-20260118-gift_EVpred-03`
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.0
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** ✅

> 🎯 **Target:** 对比预测 raw Y（原始金额）和 log(1+Y) 哪个更适合 EV 预测排序任务
> 🚀 **Next:** 使用 **Direct + raw Y** 作为最优方案，RevCap@1% 从 37.73% 提升到 **52.68%**

## ⚡ 核心结论速览

> **一句话**: 预测 raw Y 比 log(1+Y) 在 Revenue Capture @1% 上提升 **39.6%**（52.68% vs 37.73%），代价是 Spearman 下降

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Raw Y vs Log Y 哪个 RevCap 更高? | ✅ Raw Y +39.6% | Raw Y 更适合收入最大化 |
| Spearman 和 RevCap 是否一致? | ❌ 相反 | 需根据业务目标选择 |
| Two-Stage vs Direct? | Direct 更优 | Direct raw Y 最佳 |

| 指标 | Best | Baseline (log Y) | 提升 |
|------|------|------------------|------|
| RevCap@1% | **52.68%** (Direct raw) | 37.73% | **+39.6%** |
| Spearman | 0.106 (log Y) | 0.067 (raw Y) | -37% |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § I5 |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` § MVP-1.0 |

---

# 1. 🎯 目标

**问题**: 预测 EV（期望收益）时，目标变量应该用 raw Y（原始金额）还是 log(1+Y)（对数变换）？

**验证**:
- Q1: Raw Y 和 log(1+Y) 在 Revenue Capture 上的差异？
- Q2: Spearman 和 RevCap 是否一致？
- Q3: Direct vs Two-Stage 哪个更优？

| 预期 | 判断标准 |
|------|---------|
| Raw Y 更优 | RevCap 提升 > 10% → 使用 raw Y |
| Log Y 更优 | RevCap 提升 > 10% → 使用 log Y |
| 差异不大 | 差异 < 5% → 根据其他因素选择 |

---

# 2. 🦾 算法

## 2.1 预测目标定义

| 方法 | 目标 | 预测 | 排序依据 |
|------|------|------|----------|
| **Raw Y** | $Y$ (原始金额) | $\hat{Y}$ | $\hat{Y}$ |
| **Log Y (exp)** | $\log(1+Y)$ | $\hat{L}$ | $e^{\hat{L}} - 1$ |
| **Log Y (rank)** | $\log(1+Y)$ | $\hat{L}$ | $\hat{L}$ |

## 2.2 Two-Stage 公式

$$\text{EV} = P(\text{gift}>0) \times E[\text{gift} | \text{gift}>0]$$

- **Stage 1**: Logistic Regression 预测 $P(\text{gift}>0)$
- **Stage 2**: Ridge Regression 预测 $E[\text{gift} | \text{gift}>0]$
  - Raw Y: 直接预测原始金额
  - Log Y: 预测 $\log(1+Y)$，然后 $\exp$ 转回

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive (Day-Frozen) |
| 路径 | `data/KuaiLive/` |
| Train/Val/Test | 1,629,415 / 1,717,199 / 1,409,533 |
| 特征维度 | 31 |
| Gift Rate | Train 1.40%, Test 1.68% |

## 3.2 模型

| 模型 | 参数 |
|------|------|
| Direct | Ridge (alpha=1.0) |
| Two-Stage Clf | LogisticRegression (C=1.0, max_iter=1000) |
| Two-Stage Reg | Ridge (alpha=1.0) |

## 3.3 实验矩阵

| 实验 | 模型 | Stage2 目标 | 排序依据 |
|------|------|-------------|----------|
| direct_raw | Direct | raw Y | $\hat{Y}$ |
| direct_log_exp | Direct | log(1+Y) | $e^{\hat{L}}-1$ |
| direct_log_rank | Direct | log(1+Y) | $\hat{L}$ |
| twostage_raw | Two-Stage | raw Y | $P \times \hat{Y}$ |
| twostage_log_exp | Two-Stage | log(1+Y) | $P \times (e^{\hat{L}}-1)$ |
| twostage_log_rank | Two-Stage | log(1+Y) | $P \times \hat{L}$ |
| clf_only | Classifier | - | $P$ |

---

# 4. 📊 图表

### Fig 1: RevCap@1% Comparison

```
RevCap@1% (higher is better)
============================================================
direct_raw        ████████████████████████████████  52.68% ← BEST
twostage_raw      ██████████████████████████        45.66%
twostage_log_exp  █████████████████████████         44.89%
direct_log_exp    ██████████████████████            37.73%
twostage_log_rank ██████████████████                32.01%
clf_only          █████████                         15.23%
```

**观察**:
- Raw Y 方法（direct_raw, twostage_raw）显著优于 log Y 方法
- Direct raw Y 最优（52.68%），甚至超过 Two-Stage

### Fig 2: Spearman vs RevCap Trade-off

| 方法 | Spearman | RevCap@1% | Gift Rate @1% |
|------|----------|-----------|---------------|
| clf_only | **0.114** | 15.23% | 21.3% |
| direct_log | 0.106 | 37.73% | 24.1% |
| twostage_log_rank | 0.100 | 32.01% | 23.6% |
| twostage_log_exp | 0.090 | 44.89% | 20.3% |
| direct_raw | 0.067 | **52.68%** | 11.1% |
| twostage_raw | 0.011 | 45.66% | 11.0% |

**观察**:
- Spearman 和 RevCap 呈现明显的 **负相关**
- 高 Spearman → 更多 gifters（Gift Rate 高）
- 高 RevCap → 更多 revenue（但 gifters 少）

---

# 5. 💡 洞见

## 5.1 宏观

- **Spearman 和 RevCap 是 Trade-off**：优化整体排序（Spearman）≠ 优化头部收益（RevCap）
- **Log 变换压缩了 whale 信号**：10元和1000元在 log 空间差距变小，模型难以区分

## 5.2 模型层

- **Raw Y 让模型学会找 whale**：直接预测金额，大额打赏者有更大梯度
- **Log Y 让模型学会找 gifters**：log 变换使小额和大额贡献相近，模型倾向找更多人

## 5.3 细节

- **Gift Rate @1% 的差异**：raw Y 方法只有 11.1%（vs log Y 的 24.1%），但捕获的 revenue 更高
- **Classifier only 最差**：只用 P(gift) 排序，RevCap 仅 15.23%，说明金额预测很重要

## 5.4 为什么 Two-Stage 不如 Direct？

**数据特征**：per-click 打赏率仅 1.48%（72,646 / 4,909,515），极稀疏

**机制分析**：
- Two-Stage 的 Stage2 只在 gifters 上训练（~1.5% 样本），有效样本量极小
- Stage2 模型误差大，再乘上 Stage1 的 p(x)，噪声被放大
- Direct 在全量数据上回归，虽然大部分是 0，但梯度由非零样本主导

**例外情况**：
- 若关心"金主内部的精细金额排序"（如 NDCG@100 on gifters），Two-Stage 可能更好
- Two-Stage 不是完全没用，只是对 RevCap@K 这个指标不如 Direct

---

# 6. 📝 结论

## 6.1 核心发现

> **预测 raw Y 比 log(1+Y) 在 RevCap@1% 上提升 39.6%（52.68% vs 37.73%），是收入最大化的最优选择**

- ✅ Q1: Raw Y 显著优于 log Y（+39.6% RevCap）
- ✅ Q2: Spearman 和 RevCap 呈负相关（trade-off）
- ✅ Q3: Direct raw Y > Two-Stage raw Y > Two-Stage log Y

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **Raw Y 最优** | RevCap 52.68% vs 37.73% |
| 2 | **Direct > Two-Stage** | 52.68% vs 45.66% |
| 3 | **Spearman ≠ RevCap** | 负相关，不能用 Spearman 做决策 |
| 4 | **金额预测很重要** | Clf only 15.23% vs Direct 52.68% |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| **收入最大化** | 使用 raw Y 作为预测目标 |
| **用户覆盖** | 使用 log Y 或 binary classification |
| **评估指标** | 用 RevCap 而非 Spearman 做决策 |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 用 log Y 优化收入 | Log 压缩 whale 信号，损失 39.6% RevCap |
| 用 Spearman 选模型 | Spearman 高 ≠ RevCap 高 |
| 只用 P(gift) 排序 | 忽略金额信息，损失 71% RevCap |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| **Best RevCap@1%** | **52.68%** | Direct + raw Y |
| Two-Stage raw Y | 45.66% | P * E[Y] |
| Baseline (log Y) | 37.73% | Direct + log(1+Y) |
| 提升幅度 | **+39.6%** | raw vs log |
| Classifier AUC | 0.757 | - |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 模型升级 | LightGBM + raw Y | 🔴 |
| 特征工程 | 针对 whale 的特征 | 🟡 |
| 冷启动 | 新 pair 场景优化 | 🟡 |

---

# 7. 📎 附录

## 7.1 数值结果

| 配置 | RevCap@1% | Spearman | Gift Rate @1% |
|------|-----------|----------|---------------|
| **direct_raw** | **52.68%** | 0.0666 | 11.1% |
| direct_log_exp | 37.73% | 0.1061 | 24.1% |
| direct_log_rank | 37.73% | 0.1060 | 24.1% |
| twostage_raw | 45.66% | 0.0110 | 11.0% |
| twostage_log_exp | 44.89% | 0.0905 | 20.3% |
| twostage_log_rank | 32.01% | 0.0997 | 23.6% |
| clf_only | 15.23% | 0.1142 | 21.3% |

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 数据处理 | `gift_EVpred/data_utils.py` |
| 结果文件 | `gift_EVpred/results/raw_vs_log_comparison_20260118.json` |
| 缓存文件 | `gift_EVpred/features_cache/day_frozen_features_lw1h.parquet` |

```python
# 关键代码
from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
train_df, val_df, test_df = prepare_dataset()

# Direct raw Y
model = Ridge(alpha=1.0)
model.fit(X_train_scaled, y_train_raw)  # 预测原始金额
y_pred = model.predict(X_test_scaled)

# Two-Stage raw Y
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train_scaled, is_gift_train)
p_gift = clf.predict_proba(X_test_scaled)[:, 1]

reg = Ridge(alpha=1.0)
reg.fit(X_train_scaled[gifter_mask], y_train_raw[gifter_mask])  # 在 gifters 上预测 raw Y
e_gift = reg.predict(X_test_scaled)

y_pred = p_gift * e_gift  # EV = P * E[Y|Y>0]
```

## 7.3 业界参考

| 场景 | 常见做法 | 原因 |
|------|----------|------|
| 收入预测（电商 GMV） | Raw Y | 目标是最大化收入 |
| 广告 eCPM | pCTR × bid (raw) | bid 是原始出价 |
| 视频推荐（watch time） | Raw 或 Log | 取决于是否强调长视频 |
| 打赏预测 | **Raw Y（推荐）** | 捕获 whale 更重要 |

**业界惯例总结**：
- **收入最大化场景** → Raw Y（电商 GMV、广告出价、打赏）
- **用户覆盖/参与场景** → Log Y（视频观看时长、用户活跃度）
- **本质区别**：Raw Y 让模型关注"金额大小"；Log Y 让模型关注"有没有"

## 7.4 建模路线建议（务实版）

| 阶段 | 方案 | 说明 |
|------|------|------|
| **Baseline** | Direct 回归 raw Y | 当前最优，RevCap@1%=52.68% |
| **增强** | Multi-task（EV + is_gift） | 用密集信号扶起稀疏打赏信号 |
| **进阶** | Tweedie (1<p<2) 或 Quantile | 优化尾部分布，需公平对比 |

---

> **实验完成时间**: 2026-01-18
