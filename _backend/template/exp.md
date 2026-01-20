<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$

🎯 本模板的目标：
- 前 30 行：读者不仅知道"结论是什么"，还要知道"这是在做什么 & 怎么跑出来的"。
- 正文：读者可以按『实验流程伪代码』复现/重跑。
-->

# 🍃 [实验名称]
> **Name:** [Name]  \
> **ID:** `[PROJECT]-[YYYYMMDD]-[topic]-[##]`  \
> **Topic:** `[topic]` | **MVP:** MVP-X.X | **Project:** `VIT`/`SD`/`BS`  \
> **Author:** [Name] | **Date:** YYYY-MM-DD | **Status:** 🔄 In Progress / ✅ Completed
>
> 🎯 **Target:** [一句话概括实验目的]  \
> 🚀 **Decision / Next:** [实验结论 → 影响什么决策/下一步动作]

---

## ⚡ 核心结论速览（供 main 提取；≤30行；必含 I/O + Pipeline TL;DR）

> **一句话**: [最重要发现 + 关键数字 + 结论性动作（采用/拒绝/待定）]

### 0.1 这实验到底在做什么？（X := 算法/机制 → 目标 | Why+How | I/O | Trade-off）

$$
X := \underbrace{\text{[算法/范式/策略 X]}}_{\text{是什么}}\ \xrightarrow[\text{基于/通过}]{\ \text{[核心机制 🍎]}\ }\ \underbrace{\text{[目标 ⭐]}}_{\text{用于/达到}}\ \big|\ \underbrace{\text{Why 🩸}}_{\text{[痛点]}} + \underbrace{\text{How 💧}}_{\text{[难点]}}
$$
- **🐻 What (是什么) **: [一句话：X 的对象/范式]
- **🍎 核心机制**: [一句话：关键 trick / 关键约束 / 关键结构]
- **⭐ 目标**: [优化什么/验证什么/用于什么决策]
- **🩸 Why（痛点）**: [为什么要做]
- **💧 How（难点）**: [最容易踩坑/最难实现的点]
$$
\underbrace{\text{I/O 🫐}}_{\text{输入→输出}}\ =\ \underbrace{\Delta^+}_{\text{相对基线/替代方案 🍁 的优势 🍀}}\ -\ \underbrace{\Delta^-}_{\text{主要限制/代价 👿}}
$$
**I/O（必须写清楚，读者靠这一段理解实验"在干嘛"）**

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| **🫐 输入** | $\mathcal{E}$ | 环境/数据（规模、分布、约束） | 10k 用户 × 500 主播（20% 新主播） |
| **🫐 输入** | $\widehat{EV}(u,s)$ | 决策所依据的"价值信号" | 用户 $u$ 给主播 $s$ 的期望收益 |
| **🫐 输出** | $\pi$ | 分配策略/规则 | $\pi(u) = s^*$ |
| **🫐 输出** | $A$ | 一次 round 的分配结果 | $\{u_1\!\to\!s_3, u_2\!\to\!s_{10}, \dots\}$ |
| **📊 指标** | $R, CSR, G$ | 核心指标（≤3 个） | 收益 / 冷启动成功率 / Gini |
| **🍁 基线** | $\pi_0$ | 基线策略 | Greedy / no-constraint |
| **🍀 指标Δ** | $\Delta R, \Delta G$ | 关注的 trade-off | 收益-公平 / 成本-精度 / 速度-质量 |

### 0.2 Pipeline TL;DR（5-10 行极简伪代码，一眼看懂在跑什么）

> ⚠️ **关键**：核心循环必须写清楚每步的**输出是什么**，用具体数据结构示例说明

```
1. 准备环境/数据：[什么数据/模拟器]
2. 构建对比组：[哪些策略/模型]
3. 核心循环：
   for each [策略/模型]:
       for each [轮次/样本]:
           [核心操作] → 输出: {用户, 主播, 是否打赏, 金额, ...}
4. 循环后输出：[trajectory 是什么数据结构]
5. 评估：[计算什么指标，对比什么 baseline]
6. 落盘：[输出到哪里]
```

**填写示例（以 Simulator 为例）**：
```
1. 准备环境/数据：SimulatorV2+ (10k 用户 × 500 主播)
2. 构建对比组：Greedy / Concave / Random
3. 核心循环：
   for each 策略:
       for 50 次模拟:
           for 50 轮 × 200 用户:
               用户到达 → 策略分配 → 观察打赏
               → 单步输出: {'user_id': 0, 'streamer_id': 42, 'did_gift': True, 'amount': 52.0}
4. 循环后输出：trajectory = [{'user_id':0, 'streamer_id':42, ...}, {...}, ...] (50×200 条记录)
5. 评估：计算 Revenue/Gini/Cold-Start-Rate，对比 Greedy baseline
6. 落盘：results/[exp_id].json + img/*.png
```

> ⚠️ **复现命令**（repo/entry/config/seed）→ 见 §7.2 附录
> 📖 **详细流程树状图**（完整可视化）→ 见 §2.4.1
> 📖 **详细伪代码**（对齐真实代码）→ 见 §2.4.2

### 0.3 对假设/验证问题的回答

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| Q[X] / H[X.X]: [问题]? | ✅/❌ [关键数值] | [结论 + 决策动作] |

### 0.4 关键数字（只放最重要的 3-5 个）

| Metric | Value | vs Baseline | Notes |
|--------|-------|------------|------|
| [Primary] | | | |
| [Secondary] | | | |
| [Cost/Constraint] | | | |

### 0.5 Links

| Type | Link |
|------|------|
| 🧠 Hub | `logg/[topic]/[topic]_hub.md` § [Q/H] |
| 🗺️ Roadmap | `logg/[topic]/[topic]_roadmap.md` § MVP-X.X |
| 📋 Kanban | `status/kanban.md` |

---

# 1. 🎯 目标

**核心问题**: [一句话]

**对应 main / roadmap**:
- 验证问题：Q[X]
- 子假设：H[X.X]
- Gate（如有）：Gate-X

## 1.1 成功标准（验收 / stop rule）

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| ✅ 通过 | [预期范围/方向] | If [条件] → [决策] |
| ❌ 否决 | [触发否决的结果] | If [条件] → [pivot/停止] |
| ⚠️ 异常 | [异常模式] | [先排查什么] |

---

# 2. 🦾 方法（算法 + I/O + 实验流程）

> 📌 **本章至少要填 2.2 I/O 与 2.4 实验流程**；否则读者无法知道实验怎么做的。

## 2.1 算法

> 📌 **结构**：2.1.1 核心算法 → 2.1.2 符号表（变量定义+取值范围）→ 2.1.3 辅助公式（二级计算）

### 2.1.1 核心算法

**[核心算法/公式名称]**：

$$[公式]$$

**直觉解释（2-4 行）**：
- [为什么这样设计]
- [它改变了什么]

### 2.1.2 符号表

> ⚠️ **必填**：核心公式中的每个变量都要在这里定义，包含含义、类型/取值范围、计算方式或来源
> 💡 **关键**：必须给出具体数值例子，让读者秒懂每个变量的实际意义

| 符号 | 含义 | 类型/取值范围 | 计算/来源 | 具体数值例子 |
|------|------|--------------|-----------|-------------|
| $X$ | [变量含义] | [int/float/vector, 范围] | [常量/输入/公式, 见§2.1.3] | 例: `X=42` |
| $Y$ | [变量含义] | [类型, 范围] | [数据来源或公式引用] | 例: `Y=[0.3, 0.7]` |

**符号表填写示例（以 Simulator 为例）**：

| 符号 | 含义 | 类型/取值范围 | 计算/来源 | 具体数值例子 |
|------|------|--------------|-----------|-------------|
| $N_u$ | 用户数 | int, $N_u > 0$ | 配置 | `N_u=10000` |
| $N_s$ | 主播数 | int, $N_s > 0$ | 配置 | `N_s=500` |
| $\text{tier}_k$ | 第 $k$ 档金额 | int, 10 级 | 常量 | `tiers=(1,2,10,52,100,520,...)` |
| $w_u$ | 用户财富 | float, $w_u > 0$ | Pareto 分布 | `w_u=1523.5`（元） |
| $\pi(u)$ | 用户 $u$ 的分配结果 | streamer_id | 策略输出 | `π(user_0)=streamer_42` |

### 2.1.3 辅助公式

> 放置二级计算公式：如相似度计算、engagement 计算、归一化等。使读者理解核心公式中每个变量的来源。

**[辅助公式1名称]**：

$$[公式]$$

- **用途**: [这个公式用于计算什么]
- **输入**: [输入变量]
- **输出**: [输出变量，用于核心公式的哪个位置]

**[辅助公式2名称]** (可选):

$$[公式]$$

- **用途**: [...]

## 2.2 输入 / 输出（必填：详细展开，事无巨细）

> ⚠️ **必填**：必须详细展开所有输入输出的细节，包括每个字段的含义、类型、取值范围、单位等。
> 目标：读者不看代码，也能完全知道"喂进去什么 → 吐出来什么"，包括所有细节。

### I/O Schema

| Component | Type/Shape | Example | Notes |
|----------|------------|---------|------|
| Input A | | | |
| Input B | | | |
| Output Y | | | |

### Assumptions & Constraints

| Assumption/Constraint | Why it matters | How handled |
|----------------------|----------------|------------|
| [e.g., i.i.d] | | |
| [e.g., budget cap] | | |

## 2.3 实现要点（详细展开，读者能对照代码定位）

> ⚠️ **必填**：必须详细列出所有关键实现点，包括配置解析、数据加载、模型/策略构建、评估等各个环节。

| What | Where (file:function) | Key detail |
|------|------------------------|-----------|
| config parsing | `...` | |
| data loader | `...` | |
| model/policy | `...` | |
| evaluator | `...` | |

> 如果你不想贴真代码，至少给出 **入口脚本 + 关键函数名**。

## 2.4 实验流程（必填：树状可视化 + 模块拆解 + 核心循环展开 + Code Pointer）

> ⚠️ **必填**：本节必须事无巨细地展开每个步骤，让读者完全理解实验是如何执行的。
> 目标：读者看完这一节，（1）能复述实验流水线，（2）能定位代码位置，（3）能理解每个步骤的细节。

### 2.4.1 实验流程树状图（完整可视化）

> 📌 **必填**：用树状结构或方框图完整展示实验流程，包括所有步骤和子步骤
> ⚠️ **关键**：每一步必须备注**输入/输出的具体数值例子**，让读者秒懂 trajectory 是什么、循环后输出是什么
> 
> **两种格式可选**：
> - **树状图**：适合展示层级关系、配置项
> - **方框流程图**：适合展示数据流、明确的输入→输出关系

**格式 A：树状图（展示层级、配置、I/O 示例）**

```
实验流程
│
├── 1. 准备环境/数据
│   ├── 数据源：[具体数据/模拟器名称]
│   ├── 规模：[用户数、主播数、特征维度等]
│   ├── 配置：[关键参数设置]
│   └── 输出: env 对象（含 N_u=10000, N_s=500, ...）
│
├── 2. 构建对比组
│   ├── Baseline: [策略名称 + 关键配置]
│   ├── 实验组: [策略名称 + 关键配置]
│   └── 输出: policies = [GreedyPolicy, ConcavePolicy, ...]
│
├── 3. 核心循环 ⭐
│   ├── 外层：对每个策略
│   ├── 中层：× N 次模拟/epoch
│   └── 内层：× M 轮 round × K 用户
│       ├── Step 1: 用户到达 → user = {'id': 0, 'wealth': 1523.5, 'features': [...]}
│       ├── Step 2: 策略分配 → action = {'assigned_streamer': 42}
│       ├── Step 3: 环境反馈 → reward = {'did_gift': True, 'amount': 52.0}
│       └── Step 4: 记录轨迹 → record = {'user_id': 0, 'streamer_id': 42, 'did_gift': True, 'amount': 52.0}
│   └── 循环后输出: trajectory = [record_1, record_2, ...] (共 N×M×K 条)
│
├── 4. 评估
│   ├── 输入: trajectory (list of dicts)
│   ├── 计算指标：Revenue = sum(amount), Gini = gini(streamer_revenue)
│   └── 输出: metrics = {'revenue': 52340.5, 'gini': 0.534, 'csr': 0.92}
│
├── 5. 汇总（如有）
│   ├── 输入: 多次模拟的 metrics 列表
│   ├── 统计量：均值/方差/置信区间
│   └── 输出: summary = {'revenue_mean': 51234.5, 'revenue_std': 1023.4}
│
└── 6. 落盘
    ├── 结果文件：results/[exp_id].json
    ├── 图表文件：img/[exp_id]_*.png
    └── 输出示例: {"greedy": {"revenue": 51234.5, ...}, "concave": {...}}
```

**格式 B：方框流程图（展示数据流和 I/O，含具体示例）**

> 适合模块化流程，需要展示每个模块的输入→输出

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        完整模拟/实验流程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │  1. 初始化环境   │  输入: config = {n_users: 10000, n_streamers: 500}    │
│  │                  │  输出: env = SimulatorV2+(...)                        │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  2. 构建策略     │  输入: env                                            │
│  │                  │  输出: policies = [Greedy, Concave, Random]           │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  3. 核心循环 ⭐  │  输入: env + policy                                   │
│  │    for round:    │  单步输出: {'user_id':0, 'streamer_id':42,            │
│  │      分配→反馈   │            'did_gift':True, 'amount':52.0}            │
│  │                  │  循环输出: trajectory = [{...}, {...}, ...]           │
│  └────────┬─────────┘  ← 这是理解实验的关键！                               │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  4. 评估         │  输入: trajectory                                     │
│  │                  │  输出: {'revenue': 52340.5, 'gini': 0.534}            │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────┐                                                       │
│  │  5. 落盘         │  输入: metrics dict                                   │
│  │                  │  输出: results/exp_xxx.json                           │
│  └──────────────────┘                                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**方框图绘制规则**：
- 使用 Unicode box-drawing 字符：`┌ ┐ └ ┘ ─ │ ├ ┤ ┬ ┴ ┼ ▼ ▲`
- 每个方框标注：模块名称 + 输入 + 输出
- ⚠️ **必须给出具体数据结构示例**，如 `{'user_id': 0, 'streamer_id': 42, ...}`
- 箭头 `▼` 表示数据流向
- 关键输出用 `← 说明！` 标注

### 2.4.2 模块拆解（详细展开每个模块，带 Code Pointer）

| Module | Responsibility | Input → Output | Code Pointer |
|--------|----------------|----------------|--------------|
| M1: load_config | 读取配置 | cli → cfg | `scripts/xxx.py:parse_args` |
| M2: load_data | 读取数据/环境 | cfg → data/env | `data/loader.py:load` |
| M3: build_X | 构建核心对象 | cfg → model/policy | `models/xxx.py:build` |
| M4: run_loop | **核心循环** | X + data → artifacts | `core/runner.py:run` |
| M5: evaluate | 计算指标 | artifacts → metrics | `eval/metrics.py:compute` |
| M6: save | 落盘 | metrics → files | `utils/io.py:save` |

> ⚠️ **M4 run_loop 必须展开**：不能只写 `sim.run()`，要写清楚里面在干什么（见 2.4.2）

### 2.4.3 核心循环展开（对齐真实代码的详细伪代码）

> ⚠️ **必填**：必须详细展开核心循环的每一步，包括所有细节。不能只写 `sim.run()`，要让读者不打开代码也能完全理解核心算法是怎么跑的。
> 💡 **关键**：每一步都要标注**输出的具体数据结构**，让读者秒懂 trajectory 是什么

```python
# === 核心循环（对齐 core/runner.py:run）===

def run_loop(policy, env, cfg):
    """
    输入:
        policy: 分配策略 (e.g., GreedyPolicy)
        env: 模拟环境 (e.g., SimulatorV2+)
        cfg: 配置 (e.g., {n_rounds: 50, n_users_per_round: 200})
    
    输出:
        trajectory: list of dict, 每条记录格式如下:
            {'user_id': 0, 'streamer_id': 42, 'did_gift': True, 'amount': 52.0, 'round': 0}
    """
    trajectory = []
    state = env.reset()  # state = {'available_streamers': [0,1,2,...,499], 'streamer_load': {...}}
    
    for t in range(cfg.n_rounds):  # e.g., 50 轮
        # 本轮到达的用户
        users = env.sample_users(cfg.n_users_per_round)  
        # users = [{'id': 0, 'wealth': 1523.5}, {'id': 1, 'wealth': 234.2}, ...]
        
        for user in users:
            # Step 1: 策略分配 — policy 决定把用户分配给哪个主播
            action = policy.select(user, state)  
            # action = {'assigned_streamer': 42}
            
            # Step 2: 环境反馈 — 用户是否打赏、打赏金额
            reward = env.step(user, action)      
            # reward = {'did_gift': True, 'amount': 52.0}
            
            # Step 3: 记录轨迹
            record = {
                'user_id': user['id'],           # 0
                'streamer_id': action['assigned_streamer'],  # 42
                'did_gift': reward['did_gift'],  # True
                'amount': reward['amount'],      # 52.0
                'round': t                       # 0
            }
            trajectory.append(record)
            
            # Step 4: 更新状态（如有）
            policy.update(user, action, reward)  # e.g., 更新对偶变量 λ
            state = env.update_state(action)     # e.g., 更新主播负载
    
    return trajectory
    # 返回: [{'user_id':0, 'streamer_id':42, 'did_gift':True, 'amount':52.0, 'round':0}, 
    #        {'user_id':1, 'streamer_id':17, 'did_gift':False, 'amount':0.0, 'round':0}, ...]
    # 共 n_rounds × n_users_per_round = 50 × 200 = 10000 条记录
```

### 2.4.4 参数扫描（如果有 sweep，必须详细展开）

```python
for setting in grid(cfg.sweep):
    cfg_i = cfg.override(setting)
    run_one(cfg_i)
aggregate_and_compare(all_runs)
```

### 2.4.5 复现清单（必须完整填写）

- [ ] 固定随机性：seed / deterministic flags
- [ ] 固定数据版本：dataset version / simulator version
- [ ] 固定对照组：baseline 的实现与参数
- [ ] 输出物：metrics.json / results.csv / figs/*.png / logs

---

# 3. 🧪 实验设计（具体到本次实验）

## 3.1 数据 / 环境

| Item | Value |
|------|-------|
| Source | [dataset/simulator name + version] |
| Path | `...` |
| Split | Train/Val/Test = N / N / N |
| Feature | dim=N, schema=... |
| Target | [Y 定义] |

## 3.2 Baselines（对照组）

| Baseline | Purpose | Key config |
|----------|---------|-----------|
| B0 | sanity | |
| B1 | SOTA/prev | |

## 3.3 训练 / 运行配置

| Param | Value | Notes |
|------|-------|------|
| epochs / rounds | | |
| batch | | |
| lr / eta | | |
| optimizer | | |
| hardware | | |
| time budget | | |

## 3.4 扫描参数（可选）

| Sweep | Range | Fixed |
|------|-------|-------|
| [param] | [...] | ... |

## 3.5 评价指标

| Metric | Definition | Why |
|--------|------------|-----|
| [primary] | | |
| [secondary] | | |

---

# 4. 📊 图表 & 结果

> ⚠️ 图表文字必须全英文！

### Fig 1: [Title]
![](./img/[filename].png)

**What it shows**: [一句话]

**Key observations**:
- [Obs 1]
- [Obs 2]

---

# 5. 💡 洞见（解释"为什么会这样"）

## 5.1 机制层（Mechanism)

> ⚠️ **必填**：详细解释为什么会得到这样的结果，从机制层面深入分析。

- [机制解释：链接回 2.1/2.2，详细展开]
- [为什么这个机制会导致这个结果]
- [机制的内在逻辑和因果关系]

## 5.2 实验层（Diagnostics)

> ⚠️ **必填**：详细诊断实验结果，排除所有可能的 confounder。

- [排除 confounder：数据/seed/实现差异，详细说明]
- [验证实验设计的合理性]
- [分析可能的偏差来源]

## 5.3 设计层（So what)

> ⚠️ **必填**：详细阐述对系统/产品/研究路线的影响和启示。

- [对系统/产品/研究路线的影响，详细展开]
- [设计原则和最佳实践]
- [适用范围和边界条件]

---

# 6. 📝 结论 & 下一步

## 6.1 核心发现（punch line）
> **[一句话 punch line + 关键数字]**

- ✅/❌ Q[X]/H[X.X]: [结果]
- **Decision**: [采用/拒绝/继续验证]

## 6.2 关键结论（详细展开，不限制条数）

| # | 结论 | 证据（图/表/数字） | 适用范围 |
|---|------|-------------------|---------|
| 1 | | | |
| 2 | | | |

## 6.3 Trade-offs（Δ+ vs Δ-）

| Upside (Δ+) | Cost / Constraint (Δ-) | When acceptable |
|-------------|--------------------------|----------------|
| | | |

## 6.4 下一步（可执行任务）

| Priority | Task | Owner | Link |
|----------|------|-------|------|
| 🔴 P0 | | | |
| 🟡 P1 | | | |

---

# 7. 📎 附录（复现/审计用）

## 7.1 数值结果（全量）

| Config | Metric1 | Metric2 | Notes |
|--------|---------|---------|------|
| | | | |

## 7.2 执行记录（复现命令）

| Item | Value |
|------|-------|
| Repo | `~/[repo]` |
| Script | `scripts/xxx.py` |
| Config | `configs/xxx.yaml` |
| Seed | 42 |
| Output | `results/[exp_id]/` |

```bash
# (1) setup
cd ~/repo
source init.sh

# (2) run
python scripts/xxx.py --config configs/xxx.yaml --seed 42

# (3) eval (if separated)
python scripts/eval.py --input results/[exp_id]/...

# (4) plot
python scripts/plot.py --exp_id [exp_id]
```

## 7.3 运行日志摘要 / Debug（可选）

| Issue | Root cause | Fix |
|------|------------|-----|
| | | |

---

> **实验完成时间**: YYYY-MM-DD
