# 🍃 LightGBM + Raw Y: 非线性模型提升
> **Name:** LightGBM Direct Regression
> **ID:** `EXP-20260118-gift_EVpred-04`
> **Topic:** `gift_EVpred` | **MVP:** MVP-2.0
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** 🔄

> 🎯 **Target:** 验证 LightGBM 是否能通过非线性特征交互进一步提升 RevCap
> 🚀 **Next:** If RevCap > 55% → 确认非线性有效；Else → 转向特征工程

## ⚡ 核心结论速览

> **一句话**: [待实验]

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q2.1: LightGBM 是否优于 Linear？ | ⏳ 待验证 | - |

| 指标 | Ridge (Baseline) | LightGBM | 提升 |
|------|-----------------|----------|------|
| Revenue Capture @1% | 52.68% | ⏳ | ⏳ |
| Revenue Capture @5% | - | ⏳ | ⏳ |

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

> 待实验后填充

## 4.1 主结果

| 模型 | RevCap@1% | RevCap@5% | Spearman |
|------|-----------|-----------|----------|
| Ridge (Baseline) | 52.68% | - | 0.0666 |
| LightGBM | ⏳ | ⏳ | ⏳ |

## 4.2 特征重要性

> 待填充 LightGBM 特征重要性 Top 10

---

# 5. 💡 洞见

> 待实验后填充

---

# 6. 📝 结论

> 待实验后填充

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 若成功 | 固化 LightGBM 为新 baseline | 🔴 |
| 若持平 | 分析特征重要性，转向特征工程 | 🟡 |
| 若失败 | 检查过拟合，调优超参 | 🟡 |

---

# 7. 📎 附录

## 7.1 执行记录

| 项 | 值 |
|----|-----|
| 数据处理 | `gift_EVpred/data_utils.py` |
| 实验脚本 | 待创建 |

```python
# LightGBM 实现
from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
import lightgbm as lgb

train_df, val_df, test_df = prepare_dataset()
feature_cols = get_feature_columns(train_df)

X_train = train_df[feature_cols]
y_train = train_df['target_raw']  # raw Y
X_val = val_df[feature_cols]
y_val = val_df['target_raw']
X_test = test_df[feature_cols]

# LightGBM 模型
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

model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50)]
)

# 预测
y_pred = model.predict(X_test)

# 评估 RevCap
# ... (与之前相同的评估函数)
```

---

> **实验创建时间**: 2026-01-18
