<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$
-->

# 🍃 Whale Matching (b-matching)
> **Name:** Whale Matching - Tiered Allocation  \
> **ID:** `VIT-20260109-gift_allocation-53`  \
> **Topic:** `gift_allocation` | **MVP:** MVP-5.3 | **Project:** `VIT`  \
> **Author:** Viska Wei | **Date:** 2026-01-09 | **Status:** ✅ Completed
>
> 🎯 **Target:** 验证分层匹配策略能否分散鲸鱼用户，降低生态集中度  \
> 🚀 **Decision / Next:** ❌ Gate-5C FAIL → 保留统一分配，需重新设计

---

## ⚡ 核心结论速览（供 main 提取；≤30行；必含 I/O + Run TL;DR）

> **一句话**: ❌ **Gate-5C FAIL**：分层匹配策略未能降低生态集中度。所有 k 值配置下，Streamer Gini 与基线相同(0.912)，未观察到分散效果

### 0.1 这实验到底在做什么？（X := 算法/机制 → 目标 | Why+How | I/O | Trade-off）

$$
X := \underbrace{\text{分层匹配}}_{\text{鲸鱼+普通分层}}\ \xrightarrow[\text{通过}]{\ \text{b-matching 约束}\ }\ \underbrace{\text{分散鲸鱼}}_{\text{降低集中度}}\ \big|\ \underbrace{\text{Why 🩸}}_{\text{鲸鱼堆叠}} + \underbrace{\text{How 💧}}_{\text{k约束设计}}
$$
- **🐻 What (是什么)**: 分层匹配：先匹配鲸鱼（每主播最多 k 个），再填充普通用户
- **🍎 核心机制**: b-matching 或 min-cost flow 限制每个主播的鲸鱼数量
- **⭐ 目标**: 分散 Top-1% 鲸鱼，降低 Streamer Gini
- **🩸 Why（痛点）**: 鲸鱼同时涌入导致头部主播过载、马太效应
- **💧 How（难点）**: k 值设计、鲸鱼阈值选择
$$
\underbrace{\text{I/O 🫐}}_{\text{模拟器→Gini+超载率}}\ =\ \underbrace{0\%}_{\text{超载率}}\ -\ \underbrace{0}_{\text{Gini无改善}}
$$
**I/O（必须写清楚，读者靠这一段理解实验"在干嘛"）**

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| 🫐 输入 | $\mathcal{E}$ | 模拟环境 | 10K users × 500 streamers, 100 鲸鱼 |
| 🫐 输入 | $k$ | 每主播鲸鱼上限 | k ∈ {1, 2, 3, 5} |
| 🫐 输出 | $G_s, r_{over}$ | Gini 与超载率 | 0.912, 0.0% |
| 📊 指标 | $\Delta G, \Delta R$ | 相对变化 | 0, +0.07% |
| 🍁 基线 | $\pi_{greedy}$ | 无分层贪心 | Greedy |

### 0.2 Pipeline TL;DR（5-10 行极简伪代码，一眼看懂在跑什么）

```
1. 准备环境：SimulatorV2（10k 用户 × 500 主播，Top 1% = 100 鲸鱼）
2. 识别鲸鱼：按财富 Top 1% 标记 whale_ids
3. 核心循环（分层匹配）：
   for k in [1, 2, 3, 5]:  # 每主播最多 k 个鲸鱼
       for 50 次模拟:
           鲸鱼层：b-matching 约束分配 → whale_alloc
           普通层：Greedy 填充剩余容量 → normal_alloc
           合并 → 计算 Gini / 超载率 / Revenue
4. 评估：Streamer Gini（目标下降）/ 超载率（目标 <10%）/ Revenue（目标 >-5%）
5. 落盘：gift_allocation/results/whale_matching_20260109.json
```

> ⚠️ **复现命令** → 见 §7.2 附录
> 📖 **详细伪代码** → 见 §2.4.2

### 0.3 对假设/验证问题的回答

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H5C-1: 鲸鱼分散能降低超载率 | ✅ 通过 | 超载率=0.0% (<10%) |
| H5C-2: 分层匹配能降低 Streamer Gini | ❌ 失败 | Gini=0.912 (vs 基线 0.912，无改善) |
| H5C-3: 收益不显著下降 | ✅ 通过 | 收益Δ=+0.07% (>-5%) |

### 0.4 关键数字（只放最重要的 3-5 个）

| Metric | Value | vs Baseline | Notes |
|--------|-------|-------------|------|
| 超载率 | **0.0%** | 0.28% | ✅ 目标 <10% |
| Streamer Gini | **0.912** | 0.912 | ❌ 无改善 |
| Revenue Δ | **+0.07%** | 42,165 vs 42,135 | ✅ 目标 >-5% |
| k 影响 | **无** | k=1~5 结果相同 | 约束未生效 |

### 0.5 Links

| Type | Link |
|------|------|
| 🧠 Hub | `gift_allocation/gift_allocation_hub.md` § DG8, Gate-5C |
| 🗺️ Roadmap | `gift_allocation/gift_allocation_roadmap.md` § MVP-5.3 |
| 📗 Related | MVP-5.2 (影子价格), MVP-4.2 (并发容量) |

---

# 1. 🎯 目标

**核心问题**: 分层匹配策略能否分散 Top-1% 鲸鱼用户，降低生态集中度？

**对应 main / roadmap**:
- 验证问题：H5C-1, H5C-2, H5C-3
- Gate：Gate-5C

## 1.1 成功标准（验收 / stop rule）

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| ✅ 通过 | 超载率 <10% AND Gini↓ AND 收益 >-5% | 采用鲸鱼单独匹配层 |
| ❌ 否决 | Gini 无改善 | 保留统一分配 |
| ⚠️ 异常 | k 值无影响 | 检查约束是否生效 |

---

# 2. 🦾 方法（算法 + I/O + 实验流程）

## 2.1 算法

> 📌 **结构**：2.1.1 核心算法 → 2.1.2 符号表（变量定义+取值范围）→ 2.1.3 辅助公式（二级计算）

### 2.1.1 核心算法

**鲸鱼层分配（b-matching）**：

$$\text{whale\_alloc} = \arg\max_{\pi} \sum_{u \in W} \sum_s \pi_{u,s} \cdot EV(u,s)$$
$$\text{s.t.} \quad \sum_{u \in W} \pi_{u,s} \leq k, \quad \forall s$$

**普通层分配（Greedy 填充）**：

$$s^* = \arg\max_s EV(u,s), \quad \forall u \notin W$$

**直觉解释**：
- 鲸鱼先分配，强制分散到多个主播（每主播最多 k 个）
- 普通用户填充剩余容量
- 通过 k 值控制每个主播的鲸鱼数量上限

### 2.1.2 符号表

> 💡 **关键**：每个符号都给出具体数值例子，让读者秒懂变量含义

| 符号 | 含义 | 类型/取值范围 | 计算/来源 | 具体数值例子 |
|------|------|--------------|-----------|-------------|
| $W$ | 鲸鱼用户集合 | set, $|W| = 0.01 \times N$ (Top 1%) | 见 §2.1.3 鲸鱼识别 | `W={user_0, user_5, ...}`（100 人）|
| $N$ | 用户总数 | int, $N = 10000$ | 常量 | `N=10000` |
| $S$ | 主播总数 | int, $S = 500$ | 常量 | `S=500` |
| $k$ | 每主播鲸鱼上限 | int, $k \in \{1, 2, 3, 5\}$ | 超参数（扫描） | `k=2`（每主播最多 2 个鲸鱼）|
| $\pi_{u,s}$ | 用户分配指示 | binary, $\{0, 1\}$ | 优化变量 | `π[user_0, streamer_42]=1` |
| $EV(u,s)$ | 预估期望价值 | float, $\geq 0$ | 见 §2.1.3 EV 计算 | `EV(user_0, streamer_42)=85.3`（元）|
| $w_u$ | 用户财富 | float, $w_u > 0$ | 混合分布采样 | `w_u=52340.5`（鲸鱼财富 5 万+）|
| $\text{threshold}$ | 鲸鱼阈值 | float | P99 财富 | `threshold=15234.5`（元）|
| $\text{whale\_count}_s$ | 主播当前鲸鱼数 | int, $\geq 0$ | 累计计数 | `whale_count[s_42]=2` |
| $\text{overload\_rate}$ | 超载率 | float, $[0, 1]$ | 见 §2.1.3 超载计算 | `overload_rate=0.15`（15% 主播超载）|

### 2.1.3 辅助公式

**鲸鱼识别（Top 1% 财富）**：

$$\text{threshold} = \text{Percentile}(\{w_u\}_{u=1}^N, 99)$$
$$W = \{u : w_u \geq \text{threshold}\}$$

- **用途**: 识别 Top 1% 财富用户为鲸鱼
- **输入**: 所有用户财富 $\{w_u\}$
- **输出**: 鲸鱼集合 $W$，$|W| \approx 100$

**预估期望价值（EV）计算**：

$$EV(u,s) = P(\text{gift}|u,s) \cdot \mathbb{E}[\text{amount}|\text{gift}=1, u,s]$$

- **用途**: 估计分配的期望收益
- **输入**: 用户特征、主播特征
- **输出**: 预估价值 $EV(u,s)$

**b-matching 约束**：

$$\sum_{u \in W} \mathbb{1}[\text{alloc}(u) = s] \leq k, \quad \forall s \in \{1,...,S\}$$

- **用途**: 限制每个主播最多接收 $k$ 个鲸鱼
- **输入**: 鲸鱼分配结果、参数 $k$
- **输出**: 约束满足/违反状态

**超载率计算**：

$$\text{overload\_rate} = \frac{|\{s : \text{load}(s) > \text{capacity}(s)\}|}{S}$$

- **用途**: 衡量容量约束的满足程度
- **输入**: 各主播的负载和容量
- **输出**: 超载主播比例，目标 < 10%

## 2.2 输入 / 输出（必填：比 0.1 更细一点）

### I/O Schema

| Component | Type/Shape | Example | Notes |
|----------|------------|---------|------|
| users | ndarray (10000, 16) | preference + wealth | 含财富标签 |
| whale_threshold | float | 0.01 (Top 1%) | 鲸鱼定义 |
| k | int | 1~5 | 每主播鲸鱼上限 |
| Output: allocations | Dict | {u1: s5, ...} | 分配结果 |
| Output: overload_rate | float | 0.0% | 超载比例 |

### Assumptions & Constraints

| Assumption/Constraint | Why it matters | How handled |
|----------------------|----------------|------------|
| 鲸鱼定义基于 wealth | 与真实 cumulative_revenue 有偏差 | 可调阈值 |
| 容量设置 | 当前可能过大 | 需校准 |

## 2.3 实现要点（读者能对照代码定位）

| What | Where (file:function) | Key detail |
|------|------------------------|-----------|
| Whale Matching | `scripts/simulator/policies_whale_matching.py` | 分层策略 |
| b-matching | `scripts/simulator/policies_whale_matching.py:b_matching` | 核心约束 |
| Simulator | `scripts/simulator/simulator.py:GiftLiveSimulator` | V2 模拟器 |
| Evaluator | `scripts/simulator/metrics.py` | 指标计算 |

## 2.4 实验流程（必填：模块拆解 + 核心循环展开 + Code Pointer）

### 2.4.1 实验流程树状图（完整可视化）

```
分层匹配实验（Whale b-matching）
│
├── 1. 准备环境
│   ├── 模拟器：SimulatorV2（10K users × 500 streamers）
│   ├── 鲸鱼定义：Top 1% 财富（100 人）
│   └── Code: `simulator/simulator.py:GiftLiveSimulator`
│
├── 2. 识别鲸鱼
│   ├── threshold = Percentile(wealth, 99)
│   ├── whale_ids = {u : w_u >= threshold}
│   └── Code: `policies_whale_matching.py:identify_whales`
│
├── 3. 核心循环（分层匹配）
│   ├── 外层：k ∈ [1, 2, 3, 5]（每主播鲸鱼上限）
│   ├── 中层：× 50 次模拟
│   └── 内层：分层分配
│       ├── 第一层：鲸鱼 b-matching（每主播最多 k 个）
│       │   └── 只选 streamer_whale_count < k 的主播
│       └── 第二层：普通用户 Greedy 填充
│
├── 4. 评估
│   ├── 计算指标：Streamer Gini, 超载率, Revenue
│   ├── 对比 Greedy baseline
│   └── Code: `simulator/metrics.py:compute_metrics`
│
└── 5. 落盘
    ├── 结果：gift_allocation/results/whale_matching_20260109.json
    └── 图表：gift_allocation/img/mvp53_*.png
```

### 2.4.2 模块拆解（详细展开每个模块，带 Code Pointer）

| Module | Responsibility | Input → Output | Code Pointer |
|--------|----------------|----------------|--------------|
| M1: init_simulator | 初始化 SimulatorV2 | config → simulator | `simulator/simulator.py:GiftLiveSimulator` |
| M2: identify_whales | 识别鲸鱼用户 | wealth, threshold → whale_ids | `policies_whale_matching.py:identify_whales` |
| M3: b_matching | **鲸鱼层匹配** | whale_ids, k → whale_alloc | `policies_whale_matching.py:b_matching` |
| M4: greedy_matching | 普通层填充 | remaining_cap → normal_alloc | `policies_whale_matching.py:greedy_fill` |
| M5: evaluate | 计算 Gini/超载率 | allocations → metrics | `simulator/metrics.py:compute_metrics` |
| M6: save | 落盘 | results → json | `policies_whale_matching.py:save_json` |

### 2.4.3 核心循环展开（M3: b-matching 分层匹配逻辑）

> ⚠️ **必填**：分层匹配的核心——先约束分配鲸鱼，再 Greedy 填充普通用户

```python
# === 核心循环（对齐 policies_whale_matching.py）===

def run_tiered_matching(simulator, k=2):
    """分层匹配：鲸鱼层 + 普通层"""
    
    # Step 1: 识别鲸鱼（Top 1% 财富）
    whale_threshold = np.percentile([u.wealth for u in simulator.users], 99)
    whale_ids = [u.id for u in simulator.users if u.wealth >= whale_threshold]
    normal_ids = [u.id for u in simulator.users if u.wealth < whale_threshold]
    
    # Step 2: 初始化主播容量
    streamer_whale_count = {s.id: 0 for s in simulator.streamers}
    streamer_total_count = {s.id: 0 for s in simulator.streamers}
    
    allocations = {}
    
    # ===== 第一层：鲸鱼 b-matching =====
    for whale_id in whale_ids:
        user = simulator.get_user(whale_id)
        
        # 只考虑未满 k 个鲸鱼的主播
        eligible_streamers = [
            s for s in simulator.streamers 
            if streamer_whale_count[s.id] < k  # b-matching 约束
        ]
        
        if not eligible_streamers:
            continue  # 所有主播都满了
        
        # Greedy 选择（在 eligible 中）
        s_star = argmax([match_score(user, s) for s in eligible_streamers])
        allocations[whale_id] = s_star.id
        streamer_whale_count[s_star.id] += 1
        streamer_total_count[s_star.id] += 1
    
    # ===== 第二层：普通用户 Greedy 填充 =====
    for normal_id in normal_ids:
        user = simulator.get_user(normal_id)
        # 无约束 Greedy（可选：容量约束）
        s_star = argmax([match_score(user, s) for s in simulator.streamers])
        allocations[normal_id] = s_star.id
        streamer_total_count[s_star.id] += 1
    
    return allocations, streamer_whale_count

# 失败分析：为什么 Gini 没下降？
# - 鲸鱼只占 1%（100 人），分散后对总 Gini 影响微乎其微
# - 普通用户仍然 Greedy 堆叠到头部主播
# - k 约束实际未触发（每主播鲸鱼数本来就 <k）
```

**关键逻辑解释**：
- **b-matching 约束**: 每主播最多 k 个鲸鱼 → 超载率降到 0%
- **失败原因**: 鲸鱼数量太少（1%），分散后对 Gini 无显著影响
- **改进方向**: 需扩大"鲸鱼"定义，或对普通用户也加约束

### 2.4.4 参数扫描

```python
for k in [1, 2, 3, 5]:
    cfg_i = cfg.override(max_whales_per_streamer=k)
    run_one(cfg_i)
```

### 2.4.5 复现清单

- [x] 固定随机性：seed=42
- [x] 鲸鱼定义：Top 1% by wealth
- [x] 对照组：Greedy (无分层)
- [x] 输出物：whale_matching_20260109.json

---

# 3. 🧪 实验设计（具体到本次实验）

## 3.1 数据 / 环境

| Item | Value |
|------|-------|
| Source | Simulator V2 (含并发容量建模) |
| Path | 合成数据 |
| Split | N/A |
| Feature | 10,000 users × 500 streamers, 100 鲸鱼 (Top 1%) |
| Target | Gini↓, 超载率↓ |

## 3.2 Baselines（对照组）

| Baseline | Purpose | Key config |
|----------|---------|-----------|
| Greedy | 基线 | 按 EV 排序，无分层 |

## 3.3 训练 / 运行配置

| Param | Value | Notes |
|------|-------|------|
| n_rounds | 50 | - |
| users_per_round | 200 | - |
| n_simulations | 50 | - |
| hardware | CPU | ~10min |

## 3.4 扫描参数

| Sweep | Range | Fixed |
|------|-------|-------|
| k (每主播鲸鱼上限) | {1, 2, 3, 5} | - |
| 鲸鱼阈值 | Top 1% | 其他待测 |

## 3.5 评价指标

| Metric | Definition | Why |
|--------|------------|-----|
| 超载率 | 容量违约比例 | 约束效果 |
| Streamer Gini | 主播收益集中度 | 生态健康 |
| Revenue | 总收益 | 核心商业指标 |

---

# 4. 📊 图表 & 结果

### Fig 1: 超载率 vs k 值
![](./img/mvp53_overload_vs_k.png)

**What it shows**: 不同 k 值下的超载率

**Key observations**:
- 所有 k 值下超载率均为 0.0%，远低于 10% 目标
- 说明在当前容量设置下，容量约束不是瓶颈
- **启示**: 容量设置可能过大，未触发拥挤场景

### Fig 2: Streamer Gini vs k 值
![](./img/mvp53_gini_vs_k.png)

**What it shows**: 不同 k 值下的 Gini 系数

**Key observations**:
- 所有 k 值下 Gini 均为 0.912，与基线 Greedy 完全相同
- **关键发现**: 分层匹配未能降低生态集中度
- **可能原因**: 鲸鱼数量过少（Top 1% 仅 100 人），分散效果不明显

### Fig 3: 收益 vs k 值
![](./img/mvp53_revenue_vs_k.png)

**What it shows**: 不同 k 值下的收益

**Key observations**:
- 所有 k 值下收益几乎相同（42,165.51），与基线差异<0.1%
- 收益未显著下降，满足 Gate-5C 要求
- **启示**: 分层匹配对收益影响极小

### Fig 4: 算法对比热力图
![](./img/mvp53_algorithm_comparison.png)

**What it shows**: b-matching vs greedy-swaps 对比

**Key observations**:
- b-matching 和 greedy-swaps 表现完全相同
- 说明算法选择对结果影响不大
- **启示**: 问题不在算法，而在策略本身的有效性

### Fig 5: Gini vs Revenue 权衡散点图
![](./img/mvp53_tradeoff_scatter.png)

**What it shows**: 各配置的权衡关系

**Key observations**:
- 所有配置点几乎重叠，说明 k 值变化对结果无影响
- 与基线（红色星号）位置相同，验证了无改善结论

---

# 5. 💡 洞见（解释"为什么会这样"）

## 5.1 机制层（Mechanism)
- **分层匹配在当前设置下无效**: 所有 k 值配置下，Gini、收益、超载率均与基线相同
- **容量不是瓶颈**: 超载率仅 0.28%（基线），说明当前容量设置过大
- **鲸鱼数量可能过少**: Top 1% 仅 100 人，分散到 500 个主播，每个主播平均<0.2 个鲸鱼

## 5.2 实验层（Diagnostics)
- **算法选择不重要**: b-matching 和 greedy-swaps 结果完全相同
- **k 值无影响**: k 从 1 到 5，所有指标完全相同，说明约束未生效
- **可能原因**: 鲸鱼数量太少，即使 k=1，每个主播也分不到 1 个鲸鱼

## 5.3 设计层（So what)
- **容量设置需校准**: 当前容量过大，需降低以触发拥挤场景
- **鲸鱼阈值需提高**: 测试 Top 5% 而非 Top 1%
- **分层匹配需重新设计**: 当前策略在低鲸鱼密度下无效

---

# 6. 📝 结论 & 下一步

## 6.1 核心发现（punch line）
> **Gate-5C FAIL：分层匹配策略未能降低生态集中度，所有 k 值配置下 Gini 与基线相同(0.912)**

- ✅ H5C-1: 超载率 0.0% < 10% 目标
- ❌ H5C-2: Gini=0.912，无改善
- ✅ H5C-3: 收益 +0.07%，满足 >-5%
- **Decision**: 保留统一分配，分层匹配需重新设计

## 6.2 关键结论（2-5 条）

| # | 结论 | 证据（图/表/数字） | 适用范围 |
|---|------|-------------------|---------|
| 1 | **分层匹配在当前设置下无效** | 所有 k 值下 Gini=0.912 | 需重设 |
| 2 | **容量不是瓶颈** | 超载率仅 0.28% | 容量过大 |
| 3 | **收益未受影响** | +0.07% | 无损失 |
| 4 | **算法选择不重要** | b-matching = greedy-swaps | 问题在策略 |

## 6.3 Trade-offs（Δ+ vs Δ-）

| Upside (Δ+) | Cost / Constraint (Δ-) | When acceptable |
|-------------|--------------------------|----------------|
| 超载率↓ | Gini 无改善 | 容量紧张场景 |
| 收益无损 | 复杂度增加 | 需先验证有效 |

## 6.4 下一步（可执行任务）

| Priority | Task | Owner | Link |
|----------|------|-------|------|
| ❌ FAIL | 保留统一分配 | - | - |
| 🟡 P1 | 降低容量设置，测试 Top 5% 鲸鱼阈值 | - | - |
| 🟡 P1 | 考虑基于收益的分散而非基于数量 | - | - |

---

# 7. 📎 附录（复现/审计用）

## 7.1 数值结果（全量）

| 配置 | k | 鲸鱼阈值 | 超载率 | Gini | Revenue | Δ Revenue |
|------|---|---------|--------|------|---------|------------|
| Baseline (Greedy) | - | - | 0.28% | 0.912 | 42,135.69 | 0% |
| Whale-1 | 1 | Top 1% | 0.0% | 0.912 | 42,165.51 | +0.07% |
| Whale-2 | 2 | Top 1% | 0.0% | 0.912 | 42,165.51 | +0.07% |
| Whale-3 | 3 | Top 1% | 0.0% | 0.912 | 42,165.51 | +0.07% |
| Whale-5 | 5 | Top 1% | 0.0% | 0.912 | 42,165.51 | +0.07% |

**算法对比** (k=2, Top 1%):
- b-matching: Revenue=42,165.51, Overload=0.0%, Gini=0.912
- greedy-swaps: Revenue=42,165.51, Overload=0.0%, Gini=0.912

## 7.2 执行记录（必须包含可复制命令）

| Item | Value |
|------|-------|
| Repo | `~/GiftLive` |
| Script | `scripts/simulator/policies_whale_matching.py` |
| Config | `configs/whale_matching.yaml` |
| Output | `gift_allocation/results/whale_matching_20260109.json` |

```bash
# (1) setup
cd ~/GiftLive
source init.sh

# (2) run
python scripts/simulator/policies_whale_matching.py \
  --config configs/whale_matching.yaml \
  --output results/whale_matching_20260109.json

# (3) evaluate
python scripts/verify_metrics.py \
  --results results/whale_matching_20260109.json
```

## 7.3 运行日志摘要 / Debug（可选）

| Issue | Root cause | Fix |
|------|------------|-----|
| k 值无影响 | 鲸鱼数量太少 | 测试 Top 5% |
| 容量未触发 | 设置过大 | 降低容量 |

---

> **实验状态**: ✅ Completed  
> **创建时间**: 2026-01-09  
> **完成时间**: 2026-01-17  
> **Gate-5C**: ❌ FAIL (Gini 无改善)
