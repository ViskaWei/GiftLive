<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$
-->

# 🍃 KuaiLive 数据探索性分析
> **Name:** KuaiLive EDA  \
> **ID:** `KuaiLive-20260108-eda-01`  \
> **Topic:** `kuailive` | **MVP:** MVP-0.1 | **Project:** `VIT`  \
> **Author:** Viska Wei | **Date:** 2026-01-08 | **Status:** ✅ Completed
>
> 🎯 **Target:** 对 KuaiLive 数据集进行全面探索，理解打赏行为的分布特征  \
> 🚀 **Decision / Next:** 提出两段式建模假设，待公平对比验证（后续 Hub 已更新：公平对比下直接回归更优）

---

## ⚡ 核心结论速览（供 main 提取；≤30行；必含 I/O + Pipeline TL;DR）

> **一句话**: 打赏金额呈极端重尾分布（Gini=0.94），Top 1% 用户贡献 60% 收益，click-level 打赏率仅 1.48%

### 0.1 这实验到底在做什么？（X := 算法/机制 → 目标 | Why+How | I/O | Trade-off）

$$
X := \underbrace{\text{EDA (探索性数据分析)}}_{\text{是什么}}\ \xrightarrow[\text{基于}]{\ \text{多维度统计分析}\ }\ \underbrace{\text{理解打赏行为分布}}_{\text{用于建模设计}}\ \big|\ \underbrace{\text{Why: 需了解数据特性}}_{\text{痛点}} + \underbrace{\text{How: 多角度量化}}_{\text{难点}}
$$

- **🐻 What (是什么)**: 对 KuaiLive 数据集进行全方位探索，包括金额分布、集中度、稀疏性、时间模式
- **🍎 核心机制**: 多维度统计（Gini、分位数、稀疏度）+ 可视化（Lorenz 曲线、分布图）
- **⭐ 目标**: 验证打赏行为假设，指导建模方案设计
- **🩸 Why（痛点）**: 需了解数据特性才能设计有效模型
- **💧 How（难点）**: 需区分多种口径（click-level vs user-level）

$$
\underbrace{\text{I/O 🫐}}_{\text{输入→输出}}\ =\ \underbrace{\Delta^+: \text{量化重尾+稀疏特性}}_{\text{洞见}}\ -\ \underbrace{\Delta^-: \text{user.csv 采样偏差}}_{\text{陷阱}}
$$

**I/O（必须写清楚，读者靠这一段理解实验"在干嘛"）**

| 类型 | 符号 | 说明 | 示例 |
|------|------|------|------|
| 🫐 输入 | $\mathcal{D}_{gift}$ | 打赏记录 | gift.csv (72,646 条) |
| 🫐 输入 | $\mathcal{D}_{click}$ | 观看记录 | click.csv (4,909,515 条) |
| 🫐 输出 | $S$ | 统计结果 | eda_stats.json |
| 🫐 输出 | $F$ | 图表文件 | img/*.png (9 张图) |
| 📊 指标 | $G, r, \rho$ | 核心统计量 | Gini=0.94, gift_rate=1.48%, density=0.0064% |
| 🍁 基线 | 无 | 首次 EDA | — |

### 0.2 Pipeline TL;DR（5-10 行极简伪代码，一眼看懂在跑什么）

```
1. 加载数据：gift.csv, click.csv, user.csv, streamer.csv
   → 加载后: gift_df = [{'user_id': 123, 'streamer_id': 456, 'gift_price': 52.0, ...}, ...]
2. 基础统计：
   - gift_rate = 72,646 / 4,909,515 = 1.48%（click-level）
   - amount_stats: mean=82.7, median=2, P99=1,488
3. 集中度分析：
   - user_gini = compute_gini(user_total_amount)  # 0.942
   → 输出: {'user_gini': 0.942, 'top_1pct_share': 0.599, ...}
4. 稀疏性分析：
   - matrix_density = n_interactions / (n_users * n_streamers)  # 0.0064%
   → 输出: {'density': 0.000064, 'cold_start_ratio': 0.922, ...}
5. 时间模式：peak_hour = 12, 20-22
6. 输出：eda_stats.json + img/*.png
   → eda_stats = {'gift_rate': 0.0148, 'user_gini': 0.942, 'amount_mean': 82.7, ...}
```

> ⚠️ **复现命令**（repo/entry/config/seed）→ 见 §7.2 附录

### 0.3 对假设/验证问题的回答

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H1.1: 打赏行为极稀疏（打赏率 < 1%）? | ⚠️ 1.48% (per-click) | 稀疏但略高于 1%，需不均衡处理 |
| H1.2: 打赏金额呈重尾分布（Pareto 型）? | ✅ Gini=0.94, P99/P50=744× | 极端重尾，需 log 变换 |
| H1.3: 金主高度集中? | ✅ Top 1% → 60% 收益 | 需重点识别金主 |

### 0.4 关键数字（只放最重要的 3-5 个）

| Metric | Value | vs Baseline | Notes |
|--------|-------|------------|------|
| Gift Rate (click-level) | **1.48%** | - | 真实稀疏性 |
| User Gini | **0.942** | - | 金主集中 |
| Top 1% User Share | **59.9%** | - | 1% 贡献 60% |
| Matrix Density | **0.0064%** | - | 极稀疏 |
| Amount P99/P50 | **744×** | - | 极端重尾 |

### 0.5 Links

| Type | Link |
|------|------|
| 🧠 Hub | `KuaiLive/kuailive_hub.md` § I8 |
| 🗺️ Roadmap | `KuaiLive/kuailive_roadmap.md` § MVP-0.1 |
| 📊 Data | `data/KuaiLive/` |

---

# 1. 🎯 目标

**核心问题**: KuaiLive 数据集中打赏行为的分布特征是什么？是否支持两段式建模假设？

**对应 main / roadmap**:
- 验证问题：H1.1 (打赏稀疏性), H1.2 (金额重尾分布), H1.3 (金主集中)
- MVP：MVP-0.1 基础 EDA

## 1.1 成功标准（验收 / stop rule）

| 场景 | 预期结果 | 判断标准 |
|------|---------|---------|
| ✅ 通过 | Gini > 0.8, gift_rate < 5% | 确认重尾+稀疏 |
| ❌ 否决 | Gini < 0.5 或 gift_rate > 10% | 重新评估建模方案 |
| ⚠️ 异常 | 数据口径不一致 | 检查采样偏差 |

---

# 2. 🦾 方法（算法 + I/O + 实验流程）

## 2.1 算法（Gini 系数计算）

$$
G = \frac{2\sum_{i=1}^n i \cdot x_{(i)}}{n\sum_{i=1}^n x_i} - \frac{n+1}{n}
$$

其中 $x_{(i)}$ 为按升序排列的用户/主播总打赏金额。

**直觉解释**：
- G=0：完全均匀分布
- G=1：完全集中（一人占全部）
- G>0.8：高度集中

## 2.2 输入 / 输出（必填）

### I/O Schema

| Component | Type/Shape | Example | Notes |
|----------|------------|---------|------|
| gift.csv | DataFrame (72,646, 5) | user_id, live_id, streamer_id, timestamp, gift_price | 打赏记录 |
| click.csv | DataFrame (4.9M, 5) | user_id, live_id, streamer_id, timestamp, watch_live_time | 观看记录 |
| Output: stats | dict | gini, gift_rate, amount_stats... | JSON 格式 |
| Output: figs | png files | 9 张图 | img/ 目录 |

### Assumptions & Constraints

| Assumption/Constraint | Why it matters | How handled |
|----------------------|----------------|------------|
| user.csv 只含有打赏用户 | 导致 user-level gift_rate=100% | 使用 click-level 口径 |
| timestamp 毫秒级 | 需转换为小时/天 | Unix timestamp 转换 |

## 2.3 实现要点

| What | Where (file:function) | Key detail |
|------|------------------------|-----------|
| 数据加载 | `scripts/eda_kuailive.py:load_data` | CSV 读取 |
| Gini 计算 | `scripts/eda_kuailive.py:compute_gini` | 升序排序 |
| 绘图 | `scripts/eda_kuailive.py:plot_*` | matplotlib |

## 2.4 实验流程

### 2.4.1 实验流程树状图（完整可视化）

> ⚠️ **每一步都带 I/O 数值例子**，让读者秒懂 EDA 的输出是什么

```
EDA 流程
│
├── 1. 加载数据
│   ├── gift.csv：72,646 条打赏记录
│   ├── click.csv：4,909,515 条观看记录
│   ├── user.csv, streamer.csv
│   └── 输出: gift_df = [{'user_id': 123, 'streamer_id': 456, 'gift_price': 52.0}, ...]
│
├── 2. 基础统计
│   ├── gift_rate = 72,646 / 4,909,515 = 1.48%
│   └── 输出: {'n_gifts': 72646, 'n_clicks': 4909515, 'gift_rate': 0.0148}
│
├── 3. 金额分布分析
│   ├── 计算: mean, median, P90, P95, P99, max
│   └── 输出: {'mean': 82.7, 'median': 2.0, 'P90': 88.0, 'P99': 1488.0, 'max': 13140.0}
│
├── 4. 集中度分析 ⭐
│   ├── user_gini = compute_gini(user_totals)  # 0.942
│   ├── streamer_gini = compute_gini(streamer_totals)  # 0.930
│   ├── Top-K% share 计算
│   └── 输出: {'user_gini': 0.942, 'top_1pct_share': 0.599, 'top_10pct_share': 0.92}
│
├── 5. 稀疏性分析
│   ├── matrix_density = n / (n_users × n_streamers)
│   ├── cold_start_ratio
│   └── 输出: {'density': 0.000064, 'cold_start_streamer_ratio': 0.922}
│
├── 6. 绘图
│   ├── 金额分布、Lorenz 曲线、时间模式等
│   └── 输出: img/amount_distribution.png, img/lorenz_curve.png, ... (共 9 张图)
│
└── 7. 落盘
    ├── eda_stats.json
    └── 输出示例: {'gift_rate': 0.0148, 'user_gini': 0.942, 'density': 0.000064, ...}
```

### 2.4.2 模块拆解（详细展开每个模块，带 Code Pointer）

| Module | Responsibility | Input → Output | Code Pointer |
|--------|----------------|----------------|--------------|
| M1: load_data | 加载 CSV | path → df | `load_data()` |
| M2: analyze_basics | 基础统计 | gift_df, click_df → stats | `analyze_gift_basics()` |
| M3: analyze_amount | 金额分布 | gift_df → amount_stats | `analyze_amount_distribution()` |
| M4: analyze_gini | 集中度 | gift_df → gini | `analyze_user/streamer_dimension()` |
| M5: analyze_sparsity | 稀疏性 | gift_df, click_df → density | `analyze_sparsity()` |
| M6: plot_all | 绘图 | stats → png files | `plot_fig*()` |
| M7: save | 保存 | stats → JSON | `save_json()` |

### 2.4.3 核心循环展开

```python
# === EDA 主流程 ===

def main():
    # 1. Load
    gift_df = pd.read_csv('data/KuaiLive/gift.csv')
    click_df = pd.read_csv('data/KuaiLive/click.csv')
    
    # 2. Basic stats
    gift_rate = len(gift_df) / len(click_df)  # 1.48%
    
    # 3. Amount distribution
    amounts = gift_df['gift_price'].values
    amount_stats = {
        'mean': np.mean(amounts),      # 82.7
        'median': np.median(amounts),  # 2
        'p99': np.percentile(amounts, 99),  # 1,488
    }
    
    # 4. Gini concentration
    user_totals = gift_df.groupby('user_id')['gift_price'].sum()
    user_gini = compute_gini(user_totals.values)  # 0.942
    
    # 5. Sparsity
    n_users = gift_df['user_id'].nunique()
    n_streamers = click_df['streamer_id'].nunique()
    n_interactions = len(gift_df)
    matrix_density = n_interactions / (n_users * n_streamers)  # 0.0064%
    
    # 6. Plot & Save
    plot_all_figures(...)
    save_json(stats, 'eda_stats.json')
```

### 2.4.5 复现清单

- [x] 数据版本：KuaiLive 公开数据集 v1.0
- [x] 输出物：eda_stats.json, img/*.png

---

# 3. 🧪 实验设计

## 3.1 数据

| Item | Value |
|------|-------|
| Source | KuaiLive 公开数据集 |
| Path | `data/KuaiLive/` |
| 打赏记录 | 72,646 条 |
| 观看记录 | 4,909,515 条 |
| 用户数 | 23,772 |
| 主播数 | 452,621 (有打赏: 35,370) |

## 3.2 分析维度

| 维度 | 指标 |
|------|------|
| 金额分布 | Mean, Median, P90, P95, P99, Max |
| 集中度 | Gini, Top-K% share |
| 稀疏性 | Matrix density, Cold start ratio |
| 时间模式 | Peak hour |

---

# 4. 📊 图表 & 结果

### Fig 1: Gift Amount Distribution (Log Scale)
![](../img/gift_amount_distribution.png)

**What it shows**: log(1+x) 变换后的打赏金额分布

**Key observations**:
- log 变换后近似正态分布，适合 MSE 损失
- Mean ≈ 2.1, Median ≈ 1.1（log scale）

### Fig 4: User Lorenz Curve
![](../img/user_lorenz_curve.png)

**What it shows**: 用户收入集中度

**Key observations**:
- **Gini = 0.942**，极高集中度
- Top 1% 用户贡献 60% 收益
- Top 10% 用户贡献 93% 收益

### Fig 5: Streamer Lorenz Curve
![](../img/streamer_lorenz_curve.png)

**What it shows**: 主播收入集中度

**Key observations**:
- **Gini = 0.930**，头部主播效应显著
- Top 1% 主播获得 53% 收益

---

# 5. 💡 洞见

## 5.1 机制层（Mechanism)
- **两段式建模必要性**：打赏稀疏（1.48%）+ 金额重尾（P99/P50=744×）
- **金主识别是核心**：1% 用户贡献 60% 收益

## 5.2 实验层（Diagnostics)
- **采样偏差警告**：user.csv 仅含有打赏用户，user-level gift_rate=100% 不可外推
- **真实稀疏性**：使用 click-level 口径 = 1.48%

## 5.3 设计层（So what)
- **金额变换**：log(1+Y) 变换后分布接近正态
- **稀疏处理**：需使用负采样或加权损失
- **冷启动**：92.2% 主播无打赏记录

---

# 6. 📝 结论 & 下一步

## 6.1 核心发现（punch line）
> **打赏行为极度集中（Gini=0.94），金额极端重尾（P99/P50=744×），click-level 稀疏（1.48%）**

- ✅ H1.2: 金额重尾分布确认
- ✅ H1.3: 金主集中确认
- ⚠️ H1.1: 稀疏率 1.48%（略高于 1%）

## 6.2 关键结论

| # | 结论 | 证据 | 适用范围 |
|---|------|------|---------|
| 1 | **金主集中** | Top 1% → 60% | 用户侧 |
| 2 | **头部主播** | Top 1% → 53% | 主播侧 |
| 3 | **极端稀疏** | density=0.0064% | 交互矩阵 |
| 4 | **金额重尾** | P99/P50=744× | 金额分布 |

## 6.3 Trade-offs

| Upside (Δ+) | Cost / Constraint (Δ-) | When acceptable |
|-------------|--------------------------|----------------|
| 量化重尾特性 | user.csv 有采样偏差 | 使用 click-level 口径 |

## 6.4 下一步

| Priority | Task | Owner | Link |
|----------|------|-------|------|
| 🔴 P0 | 实现 Baseline 模型 | Model | MVP-0.2 |
| 🟡 P1 | Session 级分析 | EDA | 综合 EDA |

---

# 7. 📎 附录

## 7.1 数值结果

| 指标 | 值 |
|------|-----|
| total_gift_records | 72,646 |
| unique_gift_users | 23,772 |
| gift_rate_per_click | 1.48% |
| amount_mean | 82.68 |
| amount_median | 2.0 |
| amount_p99 | 1,488.2 |
| user_gini | 0.9420 |
| streamer_gini | 0.9304 |
| matrix_density | 0.0064% |
| cold_start_streamer | 92.2% |

## 7.2 执行记录

| Item | Value |
|------|-------|
| Repo | `~/GiftLive` |
| Script | `scripts/eda_kuailive.py` |
| Output | `KuaiLive/img/`, `gift_allocation/results/eda_stats_20260108.json` |

```bash
# (1) setup
cd ~/GiftLive
source init.sh

# (2) run
python scripts/eda_kuailive.py 2>&1 | tee logs/eda_kuailive.log
```

---

> **实验完成时间**: 2026-01-08
