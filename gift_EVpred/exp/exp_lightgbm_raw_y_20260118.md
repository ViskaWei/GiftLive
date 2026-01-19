# 🍃 LightGBM + Raw Y: 非线性模型提升
> **Name:** LightGBM Direct Regression
> **ID:** `EXP-20260118-gift_EVpred-04`
> **Topic:** `gift_EVpred` | **MVP:** MVP-2.0
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** ✅

> 🎯 **Target:** 验证 LightGBM 是否能通过非线性特征交互进一步提升 RevCap
> 🚀 **Next:** LightGBM 失败 → 转向特征工程或其他模型架构

## ⚡ 核心结论速览

> **一句话**: LightGBM 在 RevCap@1% 上显著劣于 Ridge（44.97% vs 52.60%），树模型不适合此任务

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q2.1: LightGBM 是否优于 Linear？ | ❌ **-3.1% ~ -14.5%** | 树模型失败 |

| 指标 | Ridge (Baseline) | LightGBM (default) | LightGBM (best) |
|------|-----------------|----------|------|
| Revenue Capture @1% | **52.60%** | 44.97% (-14.5%) | 49.49% (-3.1%) |
| Revenue Capture @5% | **64.72%** | 52.80% | - |
| Spearman | **0.0644** | 0.0504 | - |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § Q2.1 |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` § MVP-2.0 |
| 📊 Baseline | `exp/exp_raw_vs_log_20260118.md` |

---
# 1. 🎯 目标

**问题**: 当前 Direct + raw Y (Ridge) 达到 52.68% RevCap@1%，但 Linear 模型无法捕获非线性特征交互。LightGBM 能否进一步提升？

**验证**: Q2.1 - LightGBM 是否在 RevCap@1% 上优于 Ridge？

**动机分析**:
- Ridge 是线性模型，只能学习特征的加权组合
- LightGBM 可以学习特征交互（如 pair_history * user_spend_level）
- 历史特征之间可能存在非线性关系

| 预期 | 判断标准 |
|------|---------|
| 成功 | RevCap@1% > 55%（相对提升 >5%） |
| 持平 | RevCap@1% ∈ [52%, 55%] |
| 失败 | RevCap@1% < 52%（过拟合或超参不当） |

---

# 2. 🦾 算法

## 2.1 Baseline: Ridge Regression

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b$$

- 线性模型，无法捕获特征交互
- 当前 RevCap@1% = 52.68%

## 2.2 LightGBM Regression

$$\hat{y} = \sum_{k=1}^{K} f_k(\mathbf{x})$$

- 梯度提升树，可学习非线性特征交互
- 自动特征选择和特征重要性

**预期优势**：
- 自动发现 pair_history × user_level 等交互
- 对异常值（whale）更鲁棒
- 无需手动特征标准化

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive |
| 路径 | `data/KuaiLive/` |
| Train/Val/Test | 1,629,415 / 1,717,199 / 1,409,533 |
| 特征维度 | 31 (Day-Frozen) |
| 预测目标 | **raw Y**（gift_price_label） |

## 3.2 模型对比

| 模型 | 类型 | 参数 |
|------|------|------|
| Ridge (Baseline) | Linear | alpha=1.0 |
| LightGBM | Tree | 见下方 |

## 3.3 LightGBM 超参

| 参数 | 值 | 说明 |
|------|-----|------|
| objective | regression | 回归任务 |
| metric | mae | 监控指标（但评估用 RevCap） |
| n_estimators | 500 | 树数量 |
| max_depth | 8 | 防止过拟合 |
| learning_rate | 0.05 | 学习率 |
| num_leaves | 64 | 叶子数 |
| min_child_samples | 100 | 叶子最小样本 |
| subsample | 0.8 | 行采样 |
| colsample_bytree | 0.8 | 列采样 |
| reg_alpha | 0.1 | L1 正则 |
| reg_lambda | 0.1 | L2 正则 |
| seed | 42 | 随机种子 |

## 3.4 训练

| 参数 | 值 |
|------|-----|
| early_stopping_rounds | 50 |
| eval_set | Val |
| 评估集 | Test (Day 15-21) |
| 主指标 | Revenue Capture @1%, @5% |

## 3.5 消融实验

| 变体 | 说明 |
|------|------|
| LightGBM default | 默认超参 |
| LightGBM tuned | 调优后超参 |
| LightGBM + Two-Stage | 分类 + 回归 |

---

# 4. 📊 结果

## 4.1 主结果

| 模型 | RevCap@1% | RevCap@5% | RevCap@10% | Spearman |
|------|-----------|-----------|------------|----------|
| **Ridge (Baseline)** | **52.60%** | **64.72%** | **70.14%** | **0.0644** |
| LightGBM | 44.97% | 52.80% | 64.85% | 0.0504 |
| **相对变化** | **-14.5%** | **-18.4%** | **-7.5%** | **-21.7%** |

**关键观察**：
- LightGBM 在所有指标上均劣于 Ridge
- Best iteration = 1，说明模型在第一轮就触发 early stopping
- 即使尝试多种超参配置，结果仍不理想

## 4.2 Gift Rate 对比

| 模型 | GiftRate@1% | GiftRate@5% | GiftRate@10% |
|------|-------------|-------------|--------------|
| Ridge | 11.07% | 8.03% | 5.81% |
| LightGBM | 11.05% | 4.31% | 2.95% |

**发现**：LightGBM 在 @1% 的 GiftRate 与 Ridge 相近，但 @5%/@10% 显著下降，说明树模型的预测分布更集中。

## 4.3 特征重要性 (Top 10)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | user_gift_mean_hist | 8 |
| 2 | user_gift_cnt_hist | 5 |
| 3 | device_price | 4 |
| 4 | pair_gift_mean_hist | 3 |
| 5 | pair_gift_sum_hist | 3 |
| 6 | age | 3 |
| 7 | accu_watch_live_duration | 3 |
| 8 | str_accu_play_duration | 2 |
| 9 | hour | 2 |
| 10 | accu_watch_live_cnt | 2 |

**分析**：
- 历史打赏特征（user/pair_gift）仍是最重要的
- 但 importance 数值很低（总共只有 35 次分裂），说明模型几乎没有学到有效模式

---

# 5. 💡 洞见

## 5.1 失败原因分析

### 数据稀疏性问题

| 统计 | 值 |
|------|-----|
| 总样本 | ~4.9M |
| Gift > 0 | ~1.5% |
| Whale (>100元) | ~0.15% |
| 目标分布 | 98.5% 为 0 |

**核心问题**：98.5% 的样本 target = 0，树模型倾向于预测众数（0）。

### 树模型 vs 线性模型

| 特性 | Ridge | LightGBM |
|------|-------|----------|
| 预测类型 | 连续平滑 | 阶跃函数 |
| 对稀疏数据 | 给微小正值 | 预测 0 |
| 排序能力 | 保持相对排序 | 区分度差 |

**根本原因**：树模型的预测是离散的阶跃函数，当大部分样本为 0 时，树的叶节点倾向于预测 0。Ridge 的线性预测虽然绝对值可能不准，但能保持样本间的相对排序。

### Early Stopping 异常

- Best iteration = 1，说明验证集 loss 在第一轮后就开始上升
- 原因：树模型对这类高度偏斜数据的拟合很不稳定

## 5.2 超参调优结果（均失败）

| 配置 | RevCap@1% | vs Ridge | 说明 |
|------|-----------|----------|------|
| Default (no early stop) | 48.97% | -3.63% | 禁用 early stopping |
| Shallow trees (max_depth=3) | 38.59% | -14.01% | 限制树深度 |
| **Strong regularization** | **49.49%** | **-3.11%** | **最佳配置** |
| MAE objective | 0.71% | -51.89% | 完全失败 |
| Huber loss | 34.33% | -18.27% | 鲁棒损失无效 |

> **结论**：即使最佳配置（强正则化）仍比 Ridge 低 3.11%

## 5.3 深度分析：分类 vs 回归

### 悖论发现

LightGBM 在分类任务上大幅领先，但 RevCap 仍输给 Ridge！

| 任务 | LightGBM | LogisticRegression | 提升 |
|------|----------|-------------------|------|
| P(gift > 0) AUC | 0.7730 | 0.6434 | **+20.1%** |
| P(whale) AUC | **0.8533** | 0.6195 | **+37.7%** |
| P(whale\|gift) AUC | 0.8545 | 0.7662 | **+11.5%** |

**但 RevCap@1%**：Ridge 52.60% > LGB P(whale) 36.71%

### Top 1% 样本对比

| 指标 | Ridge Top 1% | LGB P(whale) Top 1% |
|------|--------------|---------------------|
| **Whale 数量** | **836** | 367 |
| **>1000元 大哥** | **222** | 96 |
| **>5000元 超级大哥** | **42** | 15 |
| **总 revenue** | **1,102,204 元** | 515,408 元 |
| **RevCap@1%** | **52.60%** | 24.60% |

### 根本原因：分类丢失金额信息

```
一个 100 元的 whale 和一个 10000 元的 whale：
├─ P(whale) 分类器：都是 whale，分数都 ≈ 1.0，无法区分
└─ E[Y] 回归器：10000 元的得分更高，会被优先选中
```

**Ridge E[Y] 天然按金额排序**：
- 预测值 = 特征加权和 ≈ 期望收入
- 大金额用户的特征值（历史打赏均值等）更高 → 预测值更高
- 自动把"超级大哥"排在"普通大哥"前面

### 方法论启示

| 问题类型 | 信息保留 | 适用场景 |
|---------|---------|---------|
| **分类 P(whale)** | 只保留"是/否" | 只需找到大哥，不关心金额 |
| **回归 E[Y]** | 保留金额大小 | 需要按价值排序 ✅ RevCap |

---

# 6. 📝 结论

## 6.1 核心发现

> **LightGBM 在 RevCap@1% 上劣于 Ridge 14.5%（44.97% vs 52.60%），树模型不适合此类高度稀疏的排序任务**

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **树模型回归失败** | RevCap 下降 14.5% |
| 2 | **树模型分类成功** | P(whale) AUC 0.8533 (+38%) |
| 3 | **分类 ≠ RevCap** | AUC 高但 RevCap 低 |
| 4 | **回归保留金额信息** | Ridge 选中更多超级大哥 |
| 5 | **数据稀疏是回归失败主因** | 98.5% 样本为 0 |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 任务建模 | RevCap 是回归问题，不是分类问题 |
| 模型选择 | 高度稀疏数据优先考虑线性模型 |
| 评估指标 | 排序任务看 RevCap，不看 AUC/RMSE |
| 信息保留 | 回归保留金额，分类丢失金额 |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 用分类做 RevCap | 分类丢失"大哥有多大"的信息 |
| 盲目用 GBDT | 不是所有问题 GBDT 都优于线性 |
| 信任 AUC | AUC 高不代表 RevCap 高 |
| 过度调参 | 数据特性决定上限 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Ridge RevCap@1% | **52.60%** | **当前最优** |
| LightGBM RevCap@1% | 44.97% | 失败 |
| 相对下降 | **-14.5%** | - |
| Best iteration | 1 | 立即 early stop |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| ✅ 确认 | Ridge + raw Y 仍是最优 baseline | - |
| 🟡 P1 | 特征工程：历史观看先验特征 | 替代 watch_live_time |
| 🟡 P1 | 特征工程：whale 专属特征 | 历史大额打赏次数/金额 |
| 🟢 P2 | 尝试 Neural Network | MLP 可能比树更适合 |

---

# 7. 📎 附录

## 7.1 执行记录

| 项 | 值 |
|----|-----|
| 数据处理 | `gift_EVpred/data_utils.py` |
| 结果文件 | `results/lightgbm_comparison_20260118.json` |
| 执行环境 | Python 3.x, LightGBM |

## 7.2 完整结果 JSON

```json
{
  "ridge": {
    "name": "Ridge",
    "rev_cap_0.01": 0.526,
    "gift_rate_0.01": 0.1107,
    "rev_cap_0.05": 0.6472,
    "rev_cap_0.1": 0.7014,
    "spearman": 0.0644
  },
  "lightgbm": {
    "name": "LightGBM",
    "rev_cap_0.01": 0.4497,
    "gift_rate_0.01": 0.1105,
    "rev_cap_0.05": 0.5280,
    "rev_cap_0.1": 0.6485,
    "spearman": 0.0504
  },
  "relative_improvement_1pct": -14.51,
  "best_iteration": 1
}
```

## 7.3 实验代码

```python
from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
import lightgbm as lgb
from sklearn.linear_model import Ridge
import numpy as np

# 加载数据
train_df, val_df, test_df, lookups = prepare_dataset()
feature_cols = get_feature_columns(train_df)

X_train = train_df[feature_cols].values
y_train = train_df['target_raw'].values  # raw Y
X_val = val_df[feature_cols].values
y_val = val_df['target_raw'].values
X_test = test_df[feature_cols].values
y_test = test_df['target_raw'].values

# Ridge baseline
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)

# LightGBM
params = {
    'objective': 'regression',
    'metric': 'mae',
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_child_samples': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'seed': 42,
    'verbose': -1
}

lgb_model = lgb.LGBMRegressor(**params)
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
)
lgb_pred = lgb_model.predict(X_test)

# 评估函数
def revenue_capture_at_k(y_true, y_pred, k=0.01):
    n_top = int(len(y_true) * k)
    top_indices = np.argsort(y_pred)[-n_top:]
    return y_true[top_indices].sum() / y_true.sum()
```

---

> **实验完成时间**: 2026-01-18
