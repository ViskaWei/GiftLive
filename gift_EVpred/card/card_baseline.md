# 🧠 Card Baseline｜Ridge + Raw Y + Day-Frozen 特征 = 52.6% RevCap@1%

> **结论（可指导决策）**
> 简单线性模型 (Ridge) 配合历史打赏特征已能捕获 50%+ 收入，非线性模型边际收益有限；后续改进应聚焦 Whale Recall 和稳定性，而非追求更复杂模型。

---

## 1️⃣ 理论 / 原理依据

* **假设**：打赏行为具有强持续性，历史打赏金额是未来打赏的最佳预测因子
* **关键结论**：Ridge Regression (alpha=1.0) 回归 Raw Y 已达 RevCap@1% = 52.6%
* **含义**：
  - 打赏预测本质是"找到历史高价值用户"的问题，不是复杂非线性建模
  - LightGBM 等非线性模型仅能提升 ~0.5%，说明特征的非线性交互有限
  - 改进方向应是：(1) 更好的 Whale 召回 (2) 更稳定的预测 (3) 更准确的校准

---

## 2️⃣ 实验结果（关键证据）

### 核心指标

| 指标 | 值 | 说明 |
|------|-----|------|
| **RevCap@1%** | 52.6% | 主指标，Top 1% 预测捕获 52.6% 收入 |
| **Whale Recall@1%** | 35.2% | 35% 的 whale 被召回 |
| **Whale Precision@1%** | 5.9% | Top 1% 中 5.9% 是 whale |
| **CV (稳定性)** | 10.8% | 按天波动系数，略高于 10% 阈值 |
| **Tail Calibration** | 2.2-2.5x | 预测系统性高估 |

### RevCap@K 完整曲线

| K | RevCap | 归一化 |
|---|--------|--------|
| 0.1% | 21.9% | 26.1% |
| 0.5% | 45.6% | 46.4% |
| 1% | 52.6% | 52.8% |
| 2% | 57.8% | 57.8% |
| 5% | 64.7% | 64.7% |
| 10% | 70.1% | 70.1% |

### 模型配置

```python
# 模型
model = Ridge(alpha=1.0, random_state=42)

# 数据
train_days, val_days, test_days = 7, 7, 7  # 2025-05-04 ~ 05-24

# 特征（31 维）
# 核心特征：pair_gift_sum_hist, user_gift_sum_hist, str_gift_sum_hist
# 所有 *_hist 特征使用 Day-Frozen 版本，避免数据泄漏

# Whale 定义
whale_threshold = 100  # P90 of gifters, 2376 whales (0.169%)
```

---

## 3️⃣ 工件清单

| 类型 | 路径 | 说明 |
|------|------|------|
| 模型 | `gift_EVpred/models/baseline_ridge_v1.pkl` | 包含模型 + 特征列 + 配置 |
| 特征 | `gift_EVpred/features_cache/baseline_ridge_v1_features.pkl` | 预处理后的 X/y |
| 结果 | `gift_EVpred/results/baseline_ridge_v1_results.json` | 完整评估结果 |
| 图表 | `gift_EVpred/img/baseline_*.png` | 4 张图表 |

---

## 4️⃣ 使用方法

### 加载模型

```python
import pickle
from gift_EVpred.metrics import evaluate_model

# 加载模型
with open('gift_EVpred/models/baseline_ridge_v1.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
feature_cols = data['feature_cols']
config = data['config']

# 加载特征
with open('gift_EVpred/features_cache/baseline_ridge_v1_features.pkl', 'rb') as f:
    features = pickle.load(f)
X_test, y_test = features['X_test'], features['y_test']

# 预测
y_pred = model.predict(X_test)

# 评估
result = evaluate_model(y_test, y_pred, whale_threshold=100)
print(result.summary())
```

### 对比新模型

```python
from gift_EVpred.metrics import evaluate_model, quick_eval

# 新模型预测
y_pred_new = new_model.predict(X_test)

# 快速对比
baseline = quick_eval(y_test, y_pred_baseline, whale_threshold=100)
new = quick_eval(y_test, y_pred_new, whale_threshold=100)

print(f"Baseline RevCap@1%: {baseline['revcap']:.1%}")
print(f"New Model RevCap@1%: {new['revcap']:.1%}")
print(f"Improvement: {(new['revcap'] - baseline['revcap'])*100:.2f}pp")
```

---

## 5️⃣ 改进方向

| 方向 | 策略 | 预期收益 |
|------|------|----------|
| Whale Recall | 加权损失函数 (upweight tail) | +5-10pp Recall |
| 稳定性 | 移除超大单或 robust 估计 | CV < 10% |
| 校准 | Platt Scaling / 分位数回归 | Calibration < 1.5x |
| 排序 | NDCG loss 微调 Top 内排序 | +1-2pp RevCap |

**注意**：任何改进必须超过 **RevCap@1% = 52.6%**，否则不采纳。

---

## 6️⃣ 实验链接

| 来源 | 路径 |
|------|------|
| 主实验 | `gift_EVpred/exp/exp_baseline_ridge_20260119.md` |
| 指标模块 | `gift_EVpred/metrics.py` |
| 指标卡片 | `gift_EVpred/card/card_metric.md` |
| 脚本 | `gift_EVpred/scripts/run_baseline_ridge.py` |
| LightGBM 对比 | `gift_EVpred/exp/exp_lightgbm_raw_y_20260118.md` |

---

<!--
✅ 使用规则（强烈建议）

* **一句话结论必须能决定"下一步做不做某类实验"**
  → 本卡片指导：不要追求复杂模型，应聚焦 Whale Recall 和稳定性

* **理论 ≠ 证明，只是"为什么合理"**
  → 历史打赏行为的持续性是核心假设

* **实验只放"支持结论的最小证据"**
  → 52.6% RevCap + LightGBM 边际收益有限 = 线性模型足够

* **不指导下一步实验**（这是 hub 的职责，不是 card 的）
  → 改进方向仅供参考，具体实验计划见 hub
-->
