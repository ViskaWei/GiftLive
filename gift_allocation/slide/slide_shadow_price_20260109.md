# 影子价格框架验证：Gate-5B FAIL — 5min 汇报

> **Experiment**: VIT-20260109-gift_allocation-16  \
> **Author**: Viska Wei  \
> **Date**: 2026-01-09  \
> **Language**: 中文

---

## 影子价格 +2.74% 收益，容量满足率 82.8% — Gate-5B 未通过

- **目的**：Primal-Dual 影子价格框架能否统一处理多约束？成功标准：收益 ≥ Greedy+5% AND 约束满足率 >90%
- **X 一句话**：$X := \underbrace{\text{Shadow Price}}_{\text{Primal-Dual}}\;\xrightarrow{\lambda_c \cdot \Delta g_c}\;\underbrace{\text{统一多约束}}_{\text{容量/冷启动/头部...}}$
  - Why: 硬编码规则分散，难以扩展
  - How: 用对偶变量 λ 作为约束的"价格"，统一处理
- **I/O（带符号）**：输入多约束配置，输出分配决策 + 约束满足率

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| 🫐 输入 | $\mathcal{E}$ | 模拟环境 | 10k 用户 × 500 主播，5 种约束 |
| 🫐 输入 | $\lambda_c$ | 对偶变量（影子价格） | 容量/冷启动/头部/鲸鱼/频控 |
| 🫐 输出 | $s^*$ | 分配决策 | $\arg\max[\widehat{EV} - \sum\lambda_c \cdot \Delta g_c]$ |
| 📊 指标 | $R, CSR, G$ | 收益/约束满足率/Gini | +2.74% / 82.8% / 0.534 |
| 🍁 基线 | Greedy+Rules | 简单规则 | +4.36% 收益 |

<details>
<summary><b>note</b></summary>

今天分享影子价格框架验证实验。

核心结论是：**Gate-5B FAIL**。影子价格收益 +2.74%（目标 +5%），容量约束满足率 82.8%（目标 >90%）。

我们想解决的问题是：当前约束用硬编码规则实现，难以扩展。比如冷启动是一个规则，频控是另一个规则，头部限制又是一个规则。能不能用统一框架处理？

方法是 Primal-Dual：分配时减去约束惩罚 $\sum\lambda_c \cdot \Delta g_c$，每轮对偶更新 λ。违反约束时 λ 增大，下一轮自动回避。

看 I/O 表：我们测试了 5 种约束（容量、冷启动、头部限制、鲸鱼分散、频控），对比 Greedy+Rules 简单规则方案。

</details>

---

## Primal-Dual 对偶更新 + 实验流程 + 关键结果

- **算法**：分配减去约束惩罚，对偶更新调节 λ

$$s^* = \arg\max_s \left[ \widehat{EV}(u,s) - \sum_c \lambda_c \cdot \Delta g_c(u \to s) \right]$$
$$\lambda_c^{(t+1)} = \left[ \lambda_c^{(t)} + \eta \cdot (g_c^{(t)} - b_c) \right]_+$$

- **流程（5 步）**：
```
实验流程
├── 1. 准备环境
│   └── SimulatorV1: 10k 用户 × 500 主播，5 种约束
├── 2. 构建对比组
│   └── Greedy / Greedy+Rules / Shadow-Core / Shadow-All
├── 3. 核心循环 ⭐
│   ├── × 50 次模拟 × 50 轮 × 200 用户
│   └── 用户到达 → s* = argmax[EV - Σλ·Δg] → 观察约束违反 → λ 更新
│       └── 单步输出: {'user_id': 0, 'streamer_id': 42, 'did_gift': True, 'amount': 52.0}
├── 4. 学习率扫描
│   └── η ∈ [0.001, 0.01, 0.05, 0.1, 0.2]
└── 5. 评估
    └── Revenue / Capacity Satisfy / Cold Start Rate
```

- **结果**：

| Policy | Revenue Δ | Capacity Satisfy | Cold Start Rate |
|--------|-----------|------------------|-----------------|
| greedy | 0% | N/A | 90.1% |
| **greedy_with_rules** | **+4.36%** | N/A | **97.8%** |
| shadow_price_all | +2.74% | 82.8% | 91.5% |

<details>
<summary><b>note</b></summary>

算法核心是 Primal-Dual：分配时扣除约束惩罚 $\sum\lambda_c \cdot \Delta g_c$。约束被违反时，对偶更新让 λ 增大，下一轮分配器就自动回避。

实验流程：4 种策略对比，学习率扫描 η 从 0.001 到 0.2。

看结果表——**简单规则 Greedy+Rules 表现更好**：
- Greedy+Rules 收益 +4.36%，Shadow Price 只有 +2.74%
- Greedy+Rules 冷启动率 97.8%，Shadow Price 只有 91.5%
- Shadow Price 容量满足率 82.8%，未达 90% 目标

**λ 收敛性参差**：cold_start 和 head_cap 收敛良好，但 whale_spread 收敛较慢（稳定性 0.2335）。

</details>

---

## 简单规则 > 复杂框架 — 决策：保留 Greedy+Rules

- **结论**：「简单规则 Greedy+Rules 表现优于 Shadow Price 统一框架」
  - 机制层：约束越多 ≠ 越好，添加约束导致收益持续下降
  - 证据层：Greedy+Rules +4.36% vs Shadow Price +2.74%
- **决策**：❌ **Gate-5B FAIL → 保留 Greedy+Rules 方案**
  - 收益 +2.74% < +5% 目标
  - 容量满足率 82.8% < 90% 目标
- **权衡**：统一框架可扩展，但当前收益不如简单规则
- **配置建议**：推荐 η=0.05（收益-稳定性平衡点）
- **下一步**：
  - 🔴 P0: 保留 Greedy+Rules 方案
  - 🟡 P1: 优化 Shadow Price 约束惩罚函数设计
  - 🟢 P2: 考虑在线学习 λ，适应流量动态变化

<details>
<summary><b>note</b></summary>

金句是：**简单规则 Greedy+Rules 表现优于 Shadow Price 统一框架**。

这个结论有点意外。我们本来期望统一框架更优雅、更可扩展。但实验表明，至少在当前设置下，简单规则更有效。

原因可能是：
1. 约束惩罚函数设计不够好，导致收益损失
2. λ 收敛性参差，部分约束（如 whale_spread）收敛太慢
3. 约束越多，相互干扰越大，并非越多越好

决策：Gate-5B FAIL，保留 Greedy+Rules。但 Shadow Price 框架值得继续优化——它的可扩展性在约束复杂场景下仍有价值。

下一步：保留简单规则，同时继续优化 Shadow Price 的惩罚函数设计。

</details>
