# 🍃 Simulator V1: 策略评估模拟环境

> **Name:** Simulator V1
> **ID:** `EXP-20260118-gift_EVpred-02`
> **Topic:** `gift_EVpred` | **MVP:** MVP-2.0
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** 🟡 **计划中**

> 🎯 **Target:** 构建可用于策略评估的模拟环境，填补 KuaiLive 数据的局限（长期留存/因果 uplift/分配优化）
> 🚀 **Next:** 若模拟器统计量与 KuaiLive 矩匹配误差 < 10%，则可用于策略迭代和 OPE 验证

---

## 🟡 实验优先级：远期

> **前置条件**：MVP-1.0（Leakage-Free Baseline）通过 Gate-0 后启动

| 背景问题 | 为什么需要模拟器 |
|----------|------------------|
| **长期留存** | KuaiLive 时间窗有限（~3 周），无法观测长期效应 |
| **因果 uplift** | 日志缺少随机探索/propensity，难做可靠反事实 |
| **分配优化** | 需要可控环境系统迭代策略、做可靠 OPE |
| **主播生态** | 需要建模外部性（过度集中伤生态） |

---

## ⚡ 核心结论速览

> **一句话**: [待实验完成后填写]

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H2.1: 模拟器能否复现 KuaiLive 统计特性？ | ⏳ 待验证 | 若关键统计量误差 < 10%，则模拟有效 |
| H2.2: 模拟器上的策略排序与 OPE 一致？ | ⏳ 待验证 | 若一致，则可用于策略评估 |
| H2.3: 分配优化能否在模拟器上验证？ | ⏳ 待验证 | 若收敛且符合约束，则框架有效 |

| 指标 | 目标值 | 说明 | 状态 |
|------|--------|------|------|
| 稀疏率误差 | **< 0.5%** | vs KuaiLive ~1.48% | ⏳ |
| Gini 误差 | **< 0.1** | 用户/主播集中度 | ⏳ |
| 金额分布 KS | **< 0.1** | vs KuaiLive 分布 | ⏳ |
| 策略排序一致性 | **> 80%** | 与 OPE 排序一致 | ⏳ |

| Type | Link |
|------|------|
| 🧠 Hub | `../gift_EVpred_hub.md` § Q4 |
| 🗺️ Roadmap | `../gift_EVpred_roadmap.md` § MVP-2.0, Gate-3 |
| 📋 上游实验 | `exp_leakage_free_baseline_20260118.md` |

---

# 1. 🎯 目标

**为什么需要模拟器？**

用户提供的分析指出，KuaiLive 数据有以下局限：

| 局限 | 影响 | 模拟器如何解决 |
|------|------|----------------|
| **时间窗有限** | 无法观测长期留存效应 | 模拟多个时间步，观测长期奖励 |
| **无随机探索** | 无法做可靠反事实估计 | 模拟器可控制探索策略 |
| **样本选择偏差** | 23,400 用户是活跃子集 | 可模拟不同用户分布 |
| **无外部性观测** | 无法评估生态健康 | 建模主播容量和集中度惩罚 |

**验证假设**：

| 假设 | 验证方法 | 通过标准 |
|------|----------|----------|
| **H2.1**: 模拟器能复现 KuaiLive 统计特性 | 矩匹配校准 | 关键统计量误差 < 10% |
| **H2.2**: 策略排序一致性 | 对比模拟器结果与 OPE | 排序一致性 > 80% |
| **H2.3**: 分配优化框架有效 | 在模拟器上运行约束优化 | 收敛且满足约束 |

---

# 2. 🦾 模拟器设计

## 2.1 最小闭环架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Simulator V1                           │
├─────────────────────────────────────────────────────────────┤
│  Entities:                                                  │
│    - Users (u): 财富/预算, 偏好向量, 疲劳度, 满意度         │
│    - Streamers (s): 质量, 容量上限, 曝光/收入, 生态健康分   │
│    - Time (t): 离散时间步                                   │
├─────────────────────────────────────────────────────────────┤
│  State Transition:                                          │
│    s_t → Action(a_t) → Observation(o_t) → Reward(r_t) → s_t+1│
├─────────────────────────────────────────────────────────────┤
│  Action:                                                    │
│    a_t = 分配策略(u_t → s_t)，给用户分配主播               │
├─────────────────────────────────────────────────────────────┤
│  Observation:                                               │
│    o_t = {click, watch_time, comment, like, gift}（带延迟） │
├─────────────────────────────────────────────────────────────┤
│  Reward:                                                    │
│    R_t = α·Revenue_t + β·UserSat_t - γ·ConcentrationPenalty │
└─────────────────────────────────────────────────────────────┘
```

## 2.2 实体建模

### 用户 (User)

| 属性 | 类型 | 说明 | 初始化 |
|------|------|------|--------|
| `budget` | float | 财富/预算 | 从 KuaiLive 用户消费分布采样 |
| `preference` | vector | 偏好向量（对主播类型/内容） | 随机初始化或从数据学习 |
| `fatigue` | float | 疲劳度（影响打赏概率） | 初始 0，随时间累积 |
| `satisfaction` | float | 满意度（影响留存） | 初始 1，随体验变化 |
| `gift_history` | list | 历史打赏记录 | 空 |

### 主播 (Streamer)

| 属性 | 类型 | 说明 | 初始化 |
|------|------|------|--------|
| `quality` | float | 内容质量 | 从分布采样 |
| `capacity` | int | 容量上限（同时服务人数） | 从分布采样 |
| `exposure` | float | 近期曝光量 | 初始 0 |
| `revenue` | float | 近期收入 | 初始 0 |
| `health_score` | float | 生态健康分 | 初始 1 |

## 2.3 行为模型

### Click 模型
$$P(\text{click} | u, s) = \sigma(\text{match}(u, s) + \text{quality}(s) - \text{fatigue}(u))$$

### Watch 模型
$$\text{watch\_time} \sim \text{Exponential}(\lambda = f(\text{match}, \text{quality}))$$

### Gift 模型（两阶段）
$$P(\text{gift} > 0 | \text{click}) = \sigma(\text{match}(u, s) + \text{budget}(u) - \text{fatigue}(u))$$
$$\text{amount} | \text{gift} > 0 \sim \text{Pareto}(\alpha, \min(\text{budget}(u), \text{max\_gift}))$$

### 延迟模型
$$\text{delay} \sim \text{Weibull}(k, \lambda) \cdot \mathbb{1}_{\text{delay} > 0} + (1-p_0) \cdot \delta_0$$

## 2.4 奖励函数

$$R_t = \alpha \cdot \text{Revenue}_t + \beta \cdot \text{UserSat}_t - \gamma \cdot \text{ConcentrationPenalty}_t$$

| 组件 | 定义 | 权重建议 |
|------|------|----------|
| **Revenue** | $\sum_i \text{gift\_amount}_i$ | α = 1.0 |
| **UserSat** | 用户满意度变化 | β = 0.1 |
| **ConcentrationPenalty** | Gini(exposure) 或 Top-share | γ = 0.05 |

### 生态健康约束

| 约束 | 定义 | 阈值 |
|------|------|------|
| 曝光 Gini | $\text{Gini}(\text{exposure})$ | < 0.8 |
| Top-10% 占比 | Top 10% 主播曝光占比 | < 60% |
| 长尾覆盖率 | 被服务的主播比例 | > 50% |
| 单主播过载 | 单主播 1h 内分配的高价值用户数 | < 100 |

---

# 3. 🧪 校准到 KuaiLive

## 3.1 矩匹配目标

| 统计量 | KuaiLive 真值 | 模拟目标 | 权重 |
|--------|---------------|----------|------|
| **gift 稀疏率** | ~1.48% (per-click) | ±0.5% | 高 |
| **金额分布 P50** | 2 | ±1 | 高 |
| **金额分布 P99** | ~1488 | ±200 | 高 |
| **用户 Gini** | ~0.7-0.8 | ±0.1 | 中 |
| **主播 Gini** | ~0.7-0.8 | ±0.1 | 中 |
| **pair 重复率** | ~30%（同一用户-主播多次打赏） | ±5% | 中 |
| **延迟分布** | Weibull 拟合参数 | 相似分布 | 低 |

## 3.2 校准方法

1. **ABC (Approximate Bayesian Computation)**：
   - 模拟器参数 → 生成数据 → 计算统计量 → 与 KuaiLive 比较 → 调整参数

2. **矩匹配损失**：
   $$L = \sum_i w_i \cdot |\hat{m}_i - m_i|^2$$
   其中 $m_i$ 是 KuaiLive 统计量，$\hat{m}_i$ 是模拟器统计量

---

# 4. 📊 评估指标

## 4.1 模拟有效性

| 指标 | 定义 | 目标 |
|------|------|------|
| **稀疏率误差** | $|\hat{p} - p| / p$ | < 10% |
| **金额 KS 统计量** | KS 距离 | < 0.1 |
| **Gini 误差** | $|\hat{G} - G|$ | < 0.1 |

## 4.2 策略评估有效性

| 指标 | 定义 | 目标 |
|------|------|------|
| **排序一致性** | 模拟器排序 vs OPE 排序 | > 80% |
| **相对收益估计** | 模拟器 Δ vs OPE Δ | 相关系数 > 0.8 |

---

# 5. 📝 结论（待实验完成后填写）

## 5.1 核心发现
> [待填写]

## 5.2 关键结论
| # | 结论 | 证据 |
|---|------|------|
| 1 | [待填写] | [数据] |

## 5.3 下一步
| 方向 | 任务 | 优先级 |
|------|------|--------|
| 分配优化 | 在模拟器上验证约束优化 | 🟡 |
| 多目标 | 验证 Revenue + UserSat + Eco 的平衡 | 🟢 |

---

# 6. 📎 附录

## 6.1 执行计划

| 步骤 | 任务 | 输出 |
|------|------|------|
| 1 | 设计模拟器架构 | `simulator/core.py` |
| 2 | 实现用户/主播实体 | `simulator/entities.py` |
| 3 | 实现行为模型 | `simulator/behavior_models.py` |
| 4 | 校准到 KuaiLive | `simulator/calibration.py` |
| 5 | 验证矩匹配 | `results/calibration_report.md` |
| 6 | 策略评估验证 | `results/policy_eval_report.md` |

## 6.2 相关文件

- 上游实验：`exp_leakage_free_baseline_20260118.md`
- Hub 链接：`../gift_EVpred_hub.md` § Q4
- Roadmap 链接：`../gift_EVpred_roadmap.md` § MVP-2.0

---

> **实验计划创建时间**: 2026-01-18
> **状态**: 🟡 计划中（等待 MVP-1.0 完成）
