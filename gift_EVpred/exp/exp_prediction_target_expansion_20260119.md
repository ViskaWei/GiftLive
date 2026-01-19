# 🍃 预测目标扩展：从 Gift 到行动后果
> **Name:** Prediction Target Expansion
> **ID:** `EXP-20260119-EVpred-05`
> **Topic:** `gift_EVpred` | **MVP:** MVP-4.0
> **Author:** Viska Wei | **Date:** 2026-01-19 | **Status:** 🔄 立项

> 🎯 **Target:** 从"预测 gift"扩展到"预测行动的后果"，建立多目标预测框架
> 🚀 **Next:** 定义新标签 → 扩展 data_utils.py → 多目标模型

## ⚡ 核心结论速览

> **一句话**: 待实验

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q2.4: 多目标预测是否优于单一 gift? | ⏳ | - |

| 指标 | 值 | 启示 |
|------|-----|------|
| Best | - | - |
| vs baseline (51.4%) | - | - |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred_hub.md` § Q2.4 |
| 🗺️ Roadmap | `gift_EVpred_roadmap.md` § MVP-4.0 |

---

# 1. 🎯 目标

## 1.1 问题

**核心问题**：当前预测的是"这条样本是否/会送多少礼"，但长期目标是预测"给了这个行动（分配），会发生什么"。

**动机**：
- 当前 Phase 1 只预测短期收益（gift_amount）
- 长期需要考虑：用户留存、主播生态健康
- 需要建立统一的"行动后果预测"框架

## 1.2 预测目标分解

| 维度 | 符号 | 定义 | 数据来源 |
|------|------|------|---------|
| **短期收益** | $r^{rev}_t$ | $E[\text{gift\_amount}_{t:t+H} \mid u, s, ctx, a]$ | ✅ 已有（gift_price_label） |
| **用户留存** | $r^{usr}_t$ | $E[\text{return}_{t+1d} \mid u, history, a]$ | ⏳ 需构建 |
| **生态健康** | $r^{eco}_t$ | $-\lambda \cdot \text{concentration}(\text{exposure/revenue})$ | ⏳ 需构建 |

## 1.3 验证假设

**H2.4**：多目标预测框架可以：
1. 保持短期 RevCap@1% ≥ 51.4%（不退化）
2. 提供用户留存预测信号
3. 支持生态约束的分配优化

| 预期 | 判断标准 |
|------|---------|
| 多目标有效 | RevCap ≥ 51.4% 且 用户留存 AUC > 0.6 |
| 多目标无效 | RevCap 下降 > 5% 或 留存 AUC < 0.55 |

---

# 2. 🦾 算法

## 2.1 多目标预测框架

**总奖励函数**：
$$
R(a) = w_{rev} \cdot r^{rev}(a) + w_{usr} \cdot r^{usr}(a) + w_{eco} \cdot r^{eco}(a)
$$

其中：
- $a = (u, s)$：行动（将用户 $u$ 分配/曝光给主播 $s$）
- $w_{rev}, w_{usr}, w_{eco}$：权重（初始 = 1, 0, 0，逐步调整）

## 2.2 各维度定义

### 2.2.1 短期收益 $r^{rev}$（已有）

$$
r^{rev}_t = E[\text{gift\_amount}_{t:t+H} \mid u, s, ctx]
$$

- 窗口 $H = 1$ 分钟（当前）
- 模型：Ridge Regression
- 指标：RevCap@1%

### 2.2.2 用户留存 $r^{usr}$（待构建）

$$
r^{usr}_t = E[\mathbb{1}[\text{user\_return}_{t+1d}] \mid u, history, s]
$$

**定义**：用户次日是否回访（任意直播间）

**候选标签**：
| 标签 | 定义 | 类型 |
|------|------|------|
| `return_1d` | 次日是否有 click | 二分类 |
| `return_7d` | 7 天内是否有 click | 二分类 |
| `watch_time_delta` | 观看时长变化 | 回归 |
| `engagement_delta` | 互动行为变化 | 回归 |

**指标**：AUC（分类）/ MAE（回归）

### 2.2.3 生态健康 $r^{eco}$（待构建）

$$
r^{eco}_t = -\lambda \cdot \text{Gini}(\text{exposure}_s) - \mu \cdot \text{overload}(s)
$$

**定义**：主播侧曝光/收入的集中度惩罚

**候选指标**：
| 指标 | 定义 | 目标 |
|------|------|------|
| Gini(exposure) | 曝光分布的基尼系数 | 越低越好 |
| Gini(revenue) | 收入分布的基尼系数 | 越低越好 |
| overload(s) | 主播负载超限惩罚 | 越低越好 |
| tail_coverage | 长尾主播覆盖率 | 越高越好 |

**注意**：生态健康不是逐条预测，而是分配层的约束/惩罚项。

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive (`data/KuaiLive/`) |
| 基础数据 | 复用 `data_utils.prepare_dataset()` |
| **新增标签** | 见 §3.2 |
| Train/Val/Test | 1.6M / 1.7M / 1.4M (7-7-7) |

## 3.2 新增标签构建

### 3.2.1 用户留存标签

```
构建逻辑（伪代码，不执行）：

1. 对每个 click，找该 user 在 click_day + 1 天内是否有 click
2. return_1d = 1 if exists click in [day+1, day+2) else 0
3. return_7d = 1 if exists click in [day+1, day+8) else 0
```

**数据来源**：`click.csv`（已有）

### 3.2.2 观看时长变化

```
构建逻辑（伪代码，不执行）：

1. 计算 user 在 click_day 的总观看时长 watch_today
2. 计算 user 在 click_day - 7d 的平均观看时长 watch_past_avg
3. watch_time_delta = watch_today / watch_past_avg - 1
```

**数据来源**：`click.csv`（需要 watch_live_time 的历史，非当前 session）

### 3.2.3 生态指标

```
构建逻辑（伪代码，不执行）：

1. 计算每个 streamer 的日曝光量 exposure_s
2. 计算 Gini(exposure) = gini_coefficient(exposure_s)
3. 计算 tail_coverage = 收到曝光的 streamer 数量 / 总 streamer 数量
```

**数据来源**：`click.csv` + `streamer.csv`

## 3.3 模型方案

| 方案 | 描述 | 复杂度 |
|------|------|--------|
| **A. 独立模型** | 每个目标单独训练 Ridge/LR | 低 |
| B. 多任务学习 | 共享底层 + 多头输出 | 中 |
| C. 端到端 RL | 直接优化总奖励 | 高 |

**推荐**：从方案 A 开始，验证各目标可预测性后再考虑联合建模。

## 3.4 评估指标

| 目标 | 主指标 | 辅助指标 |
|------|--------|---------|
| 短期收益 | RevCap@1% | RevCap 曲线 |
| 用户留存 | AUC | Recall@K |
| 生态健康 | Gini 变化 | tail_coverage |

## 3.5 实验步骤

| 步骤 | 任务 | 产出 |
|------|------|------|
| **Step 1** | 构建用户留存标签 | `return_1d`, `return_7d` 列 |
| **Step 2** | 验证留存可预测性 | AUC 报告 |
| **Step 3** | 计算生态基线指标 | Gini, tail_coverage |
| **Step 4** | 多目标联合评估 | 权衡曲线 |

---

# 4. 📊 图表（待实验）

### Fig 1: 留存预测 AUC
（待补充）

### Fig 2: RevCap vs 留存 AUC 权衡曲线
（待补充）

### Fig 3: 生态 Gini 分布
（待补充）

---

# 5. 💡 洞见（待实验）

## 5.1 预期洞见

- **用户留存 vs 短期收益的权衡**：是否存在"榨取式"分配？
- **生态约束的必要性**：无约束分配是否会导致头部主播过载？
- **多目标联合建模的价值**：是否优于独立模型？

---

# 6. 📝 结论（待实验）

## 6.1 核心发现
（待补充）

## 6.2 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 构建留存标签 | 扩展 data_utils.py | 🔴 P0 |
| 验证留存可预测性 | 训练 LR 分类器 | 🔴 P0 |
| 计算生态基线 | Gini / tail_coverage | 🟡 P1 |
| 多目标联合评估 | 权衡曲线 | 🟡 P1 |

---

# 7. 📎 附录

## 7.1 参考资料

| 资源 | 说明 |
|------|------|
| Hub § Quick Reference | Baseline、Data、Metric 定义 |
| `data_utils.py` | 当前数据处理模块 |
| `metrics.py` | 评估模块 |

## 7.2 依赖

- 需要扩展 `data_utils.py` 添加留存标签构建
- 需要扩展 `metrics.py` 添加 Gini 计算

## 7.3 风险

| 风险 | 应对 |
|------|------|
| 留存标签稀疏（用户不回访） | 用 7d 窗口替代 1d |
| 生态指标难以逐条预测 | 作为分配层约束，非预测目标 |
| 多目标权重难以确定 | 从 Pareto 前沿分析开始 |

---

> **实验完成时间**: 待定
