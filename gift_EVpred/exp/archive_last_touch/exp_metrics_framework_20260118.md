# 🍃 三层指标体系设计
> **Name:** Metrics Framework Design
> **ID:** `EXP-20260118-gift_EVpred-metrics`
> **Topic:** `gift_EVpred` | **MVP:** MVP-3.0
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** 🔄

> 🎯 **Target:** 建立"识别大哥 → 预测价值 → 分配"全链路的统一评估体系
> 🚀 **Next:** 标准化评估协议 → 所有后续实验使用同一套指标

## ⚡ 核心结论速览

> **一句话**: RevCap@1% 可做 North Star，但需配套三层指标体系 + 护栏指标

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| RevCap@1% 足够吗？ | 🟡 合理但不够 | 需配套 RevCap 曲线 + Tail Calibration + 稳定性指标 |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § 10 |
| 🔗 Allocation Hub | `gift_allocation/gift_allocation_hub.md` |

---

# 1. 🎯 目标

**问题**: RevCap@1% 做"找大哥"指标合理吗？需要什么护栏指标？

**背景**:
- 当前已验证 RevCap@1%=52.60%（Direct + raw Y）
- 但单一指标无法覆盖"识别 → 估值 → 分配"全链路
- 进入分配后，需要金额预测可用（不仅是排序好）

| 预期 | 判断标准 |
|------|---------|
| 三层指标体系落地 | 所有实验报告统一口径 |
| 护栏指标明确 | 按天稳定性、tail 校准 |

---

# 2. 📐 三层指标体系设计

## 2.1 识别层（Find Whales / 找大哥）

> 目标："在有限名额（Top K%）里抓到尽可能多的真实收入"

### 主指标（North Star）

| 指标 | 定义 | 公式 | 当前值 |
|------|------|------|--------|
| **RevCap@K** | Top K% 预测捕获的收入占比 | `sum(y[top_k]) / sum(y)` | 52.60%@1% |
| **Normalized RevCap@K** | 接近 Oracle 的程度 | `RevCap@K / Oracle@K` | 52.60/99.54 = 52.9% |

**建议**：报告 `K ∈ {0.1%, 0.5%, 1%, 2%, 5%, 10%}` 的 RevCap 曲线

### 诊断指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **Whale Recall@K** | TopK 选中多少真 whale（y≥P90） | 解释为什么好/坏 |
| **Precision@K** | TopK 里 whale 比例 | 池子质量 |
| **Avg Revenue per Selected@K** | TopK 人均真实收入 | 分配层预算估算 |
| **Stability (by day)** | 按天 RevCap@1% 均值±标准差 | 避免单日异常拉爆 |

### 评估泄漏警告

> ⚠️ "Best RevCap@1%" 如果是在 Test 上挑最好版本，会产生评估泄漏

**规范做法**：
- 用 **Val** 挑 best（含阈值、超参、特征版本）
- **Test 只做一次最终报告**（不参与选择）

---

## 2.2 估值层（Predict EV / 预测大哥期望赏金）

> 目标："金额预测在 tail 上可用/可校准"（不仅是排序好）

### Tail Calibration（头部校准）

按预测分数分桶（尤其看 top 0.1%、0.5%、1%、5%），对每个桶计算：

| 指标 | 公式 | 目标 |
|------|------|------|
| **Total Calibration** | `Sum(pred) / Sum(actual)` | 接近 1.0 |
| **Mean Calibration** | `Mean(pred) vs Mean(actual)` | 不系统性偏高/偏低 |

### 偏差方向

| 指标 | 定义 | 用途 |
|------|------|------|
| **Over/Under-estimation Ratio** | top 桶 over/under 的比例与幅度 | 分配更怕"高估" |
| **Tail MAE / Pinball Loss (Q90/Q95)** | 头部误差，不用整体 MAE | 聚焦 tail |

---

## 2.3 分配层（Allocation / 把大哥分给谁）

> 目标："最大化全局收益，同时控制生态集中度"

### 优化目标（来自 gift_allocation hub）

$$
\max \sum_s g_s(V_s), \quad V_s = \sum_{u \to s} v_{u,s}
$$

其中 $g$ 是凹函数（边际收益递减），边际增益近似：

$$
\Delta(u \to s) \approx g'(V_s) \cdot v_{u,s}
$$

### 分配层指标

| 指标 | 定义 | 用途 |
|------|------|------|
| **Total Revenue (Linear)** | `∑_s V_s` | 贪心 baseline |
| **Concave Revenue** | `∑_s g_s(V_s)` | 避免头部集中 |
| **Streamer Gini** | 主播收入集中度 | 生态护栏 |
| **Top-1% Share** | 头部主播占比 | 马太效应 |
| **Constraint Violations** | 约束违反率 | 容量/频控/风控 |

---

## 2.4 在线护栏（上线后必配）

| 类别 | 指标 | 用途 |
|------|------|------|
| **收益** | Revenue/DAU, ARPPU | 核心业务指标 |
| **留存** | D1/D7 留存率 | 长期健康 |
| **生态** | Gini 变化, 中小主播覆盖 | 公平性 |
| **风控** | 退出率, 投诉率 | 负面信号 |

---

# 3. 🦾 模型优化路径（P0-P3）

基于 LightGBM 失败（-14.5%）后的调整：

| 优先级 | 任务 | 方法 | 预期收益 | 判断标准 |
|--------|------|------|----------|----------|
| 🔴 **P0-Feature** | 历史观看先验 | `user_watch_mean_7d`, `pair_watch_mean` | 替代 watch_live_time | RevCap > 54% |
| 🟡 **P1-Tail** | 样本加权回归 | `w_i = (1+y_i)^α`, α∈[0.3,1] | 头部误差↓ | RevCap > 55% |
| 🟡 **P1-LTR** | 分位数回归 / LambdaRank | 直接学"上分位金额" | 更贴近找大哥 | RevCap > 55% |
| 🟢 **P2-Model** | MLP | 神经网络比树更适合稀疏 | 非线性提升 | RevCap > 55% |
| 🟢 **P3-Cold** | 冷启动/新 pair 泛化 | 相似主播特征 | 分配层基础 | warm/cold 分群评估 |

---

# 4. 🚀 分配 MVP 设计

## 4.1 两个 Baseline 对比

### Baseline A: 纯贪心（不凹）

```python
# 每个大哥 u 选 argmax_s v_{u,s}
for u in whales:
    s_best = argmax(v[u, :])
    assign(u, s_best)
```

- 结果：收益最大化，但极度集中

### Baseline B: 凹收益贪心（带影子价格）

```python
# 维护每个 streamer 当前 V_s
V = defaultdict(float)

for u in whales:
    # 边际增益 = g'(V_s) * v_{u,s}
    scores = {s: g_prime(V[s]) * v[u, s] for s in candidates}
    s_best = argmax(scores)
    V[s_best] += v[u, s_best]
    assign(u, s_best)
```

- g(V) 可用：`log(1+V)` 或 `B(1-exp(-V/B))`

## 4.2 最小约束集

| 约束 | 定义 | 优先级 |
|------|------|--------|
| 单用户频控 | 1 窗口内最多分给 N 个主播 | 🔴 必须 |
| 单主播容量 | 每主播最多接收 M 个大哥 | 🔴 必须 |
| 风控黑名单 | 违规主播/异常大哥剔除 | 🔴 必须 |
| 生态公平 | 中小主播最小流量 | 🟡 可选（先用 g(V) 软控制） |

## 4.3 离线评估（反事实问题）

**启动方式建议**：

1. **第一步（最小闭环）**：在用户已进入的候选集内做 re-rank（不太反事实）
2. **第二步（可控扩展）**：简单 Simulator（用历史分布拟合）
3. **第三步（上线前）**：OPE (IPS/DR) 或小流量 A/B

---

# 5. 📋 落地顺序建议

| 序号 | 任务 | 产出 | 验收 |
|------|------|------|------|
| 1 | 指标体系落地 | 统一报告模板 + RevCap 曲线 | 所有实验使用 |
| 2 | P0-Feature | 历史观看先验特征 | RevCap > 54% |
| 3 | P1-Tail | 样本加权回归 | RevCap > 55% |
| 4 | MVP-Alloc | 贪心 vs 凹收益贪心 | 总收益 + Gini 对比 |

---

# 6. 📎 附录

## 6.1 相关文件

| 文件 | 用途 |
|------|------|
| `gift_EVpred_hub.md` § 10 | 三层指标体系定义 |
| `gift_allocation_hub.md` | 分配层设计原则 |
| `data_utils.py` | 统一数据处理 |

## 6.2 来源

本实验计划基于 2026-01-18 专家建议整理，回答三个核心问题：
1. RevCap@1% 做"找大哥"指标合理吗？→ 合理但不够，需配套护栏
2. "预测大哥期望赏金"下一步往哪优化？→ P0-Feature > P1-Tail > P2-Model
3. 怎么开始做"分配"？→ MVP: 贪心 vs 凹收益贪心

---

> **实验完成时间**: 2026-01-18（计划创建）
