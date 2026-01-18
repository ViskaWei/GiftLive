# 🧠 gift_EVmodel Hub
> **ID:** EXP-20260118-gift_EVmodel-hub | **Status:** 🌱探索 |
> **Date:** 2026-01-18 | **Update:** 2026-01-18

---

## 🔴 Gate-0 结果：泄漏已消除，但性能不达标（2026-01-18）

> **核心发现**：数据泄漏已成功消除（特征重要性比 1.23 < 2x），但去除泄漏后性能大幅下降（Top-1% 从 56.2% 降至 **11.6%**），揭示原 baseline **几乎全部预测能力来自数据泄漏**。

| 指标 | Gate-0 目标 | 实际值 | 状态 |
|------|-------------|--------|------|
| 特征重要性比 | < 2x | **1.23** | ✅ 泄漏消除 |
| Top-1% Capture | > 40% | **11.6%** | ❌ 不达标 |
| Revenue Capture@1% | > 50% | **21.6%** | ❌ 不达标 |
| Spearman | 下降 < 0.2 | **0.14**（-0.75） | ❌ 不达标 |

### 关键结论
1. **原 baseline 是"开卷考试"**：模型通过泄漏的 `pair_gift_mean` 直接"看到"答案
2. **礼物预测极具挑战性**：98.5% 稀疏性 + 极端重尾分布，当前特征预测力接近随机
3. **需要全面重构特征工程**：探索序列特征、实时特征、内容匹配度等新信号源
4. **⚠️ Rolling 版本有实现问题**：81.1% vs Frozen 11.5%，cumsum+shift 实现可能有时间泄漏，以 Frozen 为准

---

## ⚠️ 关键问题诊断（原始分析）

> **背景**：现有 baseline（EXP-20260108-gift-allocation-02）存在**三个致命问题**，导致 Top-1%=56.2%、Spearman=0.891 的结果**不可信**。

| # | 问题 | 严重性 | 证据 | 影响 |
|---|------|--------|------|------|
| **P1** | **数据泄漏 (Target Leakage)** | 🔴 致命 | `pair_gift_mean` 重要性=328,571，远超第二名 3 倍；在 pair_gift_count=1 的样本上，`pair_gift_mean == gift_price`（特征=答案） | Baseline 性能虚高，无法作为可靠参照 |
| **P2** | **任务定义不匹配** | 🔴 致命 | 样本单元=gift-only（72,646 条已送礼记录），学的是 $E[\text{gift}|\text{已送礼}]$，而非 $EV = E[\text{gift}|\text{session}]$（含 0） | 模型无法学到"谁不会送礼"，不能支持分配决策 |
| **P3** | **评估指标偏差** | 🟡 中等 | Top-K% Capture 是"集合重叠比例"，不关心金额差异（漏掉超大额 vs 漏掉边缘 top1% 在业务上完全不同） | 应改用 Revenue Capture@K = $\frac{\sum_{i \in \text{TopK}(\hat{v})} Y_i}{\sum_i Y_i}$ |
| **P4** | **样本选择偏差** | 🟡 中等 | KuaiLive 23,400 用户是"多行为活跃用户子集"，`gift_rate` 接近 100%（全站稀疏率 ~1.48%） | 适合研究"金主分配"，但不能外推全站转化/留存 |

### 为什么 `pair_gift_mean/sum` 是泄漏？（直觉版）

```
在 gift-only 数据里，对很多 (user, streamer) 配对：
- pair_gift_count 很可能是 1（只送过一次）
- 那么 pair_gift_sum == gift_price，pair_gift_mean == gift_price

→ 特征里直接包含了当前样本的真实 label（甚至就是同一个数）
→ 模型当然"预测很准"，重要性也会爆炸
→ 这不是"特征很强"，而是"特征在很多样本上等于答案"
```

### 合理 vs 泄漏的特征设计

| 允许 ✅ | 泄漏 ❌ |
|---------|---------|
| `pair_gift_sum_past(t)` = 只用 **t 之前** 的历史累计 | `pair_gift_sum_all` = 用全量数据聚合后回填（包含 t 之后、包含当条样本自身） |
| `user_total_gift_7d_past` = 只用过去 7 天 | `user_total_gift` = 用全量数据 |
| 冻结版：只用 train window 统计 | 回填版：groupby 后 merge 回原表 |

---

| # | 💡 共识[抽象洞见] | 证据 | 决策 |
|---|----------------------------|----------------|------|
| K1 | **Direct Regression 优于 Two-Stage 做全量排序** | Direct Top-1%=54.5% vs Two-Stage=35.7%（⚠️ 需在无泄漏版本复验） | 采用 Direct 做召回层 |
| K2 | **Two-Stage 在精排场景有优势** | Stage2 在 gift 子集 Spearman=0.89 > Direct=0.74 | 推荐召回-精排分工策略 |
| K3 | **Revenue Capture@K 是核心评估指标** | ~~Baseline Top-1%=56.2%~~（虚高） | 聚焦 Revenue Capture@K 而非集合重叠 |
| K4 | **⚠️ 现有 Baseline 因泄漏不可信** | `pair_gift_mean` 重要性异常、Spearman=0.891 虚高 | 必须先跑 Past-only 无泄漏版本 |
| K5 | **任务定义必须改为 Click-Level EV** | gift-only 无法学到"谁不会送礼" | 从 gift-only 改为 click（含 0） |

<!-- ⚠️ 写作指导（此注释不要复制到实际报告）：
     共识 ≠ 数值。共识是抽象的、可泛化的洞见；数值放证据列。
-->

**🦾 现阶段信念 [≤10条，写"所以呢"]**
- **✅ 数据泄漏已消除**（特征重要性比 1.23） → 基础设施正确，可以做可靠实验
- **❌ 当前特征体系预测力极弱**（Top-1%=11.6%，Spearman=0.14） → 需要全面重构特征工程
- **礼物预测是极具挑战性的任务**（98.5% 稀疏性 + 重尾分布） → 可能需要降低任务难度或探索新信号源
- **原 baseline 高性能是假象**（泄漏导致 Top-1%=56.2%） → 所有基于泄漏版本的结论需重新评估
- **Direct vs Two-Stage 结论无法验证**（两者都很差） → 需在性能合理的版本上重新比较
- **Revenue Capture@K 是正确的评估指标** → 已实现，可用于后续实验
- **时间切分 + Past-only 特征是正确的** → 方法论正确，问题在于特征信号不足
- **可能需要考虑序列特征或实时特征** → 用户观看时长、当前 session 互动可能是关键信号

**👣 下一步最有价值  [≤2条，直接可进 Roadmap Gate]**
- 🔴 **P0**：重构特征工程 → 探索序列特征（观看时长序列）、实时特征（当前 session 互动）、内容特征（直播内容匹配度）
- 🔴 **P0**：验证任务降级 → 先验证二分类（P(gift>0)）是否可行，若 AUC > 0.7 则再考虑金额预测
- 🟡 **P1**：探索冷启动策略 → 对无历史 pair，设计基于 user/streamer 的泛化特征


> **权威数字（一行即可）**：~~Baseline=56.2%~~（⚠️ 泄漏无效）；公平对比 Direct=54.5% vs Two-Stage=35.7%（需复验）；条件=click 全量，但可能有 pair_* 特征泄漏

| 模型/方法 | 指标值 | 配置 | 备注 |
|-----------|--------|------|------|
| Direct Regression | Top-1%=54.5%, Spearman=0.331 | LightGBM, log(1+Y), click全量 | ⚠️ 需无泄漏版本复验 |
| Two-Stage | Top-1%=35.7%, NDCG@100=0.359 | p(x)×m(x), click全量 | ⚠️ 需无泄漏版本复验 |
| ~~Baseline (gift-only)~~ | ~~Top-1%=56.2%, Spearman=0.891~~ | ~~LightGBM, log(1+Y), gift-only~~ | ❌ 数据泄漏，结果无效 |
| Oracle p (Two-Stage) | Top-1%=90.7% | 真实 p(x) | Stage1 理论上界 |
| Oracle m (Two-Stage) | Top-1%=38.9% | 真实 m(x) | Stage2 理论上界 |
| **Leakage-Free (Frozen)** | **Top-1%=11.5%, Spearman=0.10, RevCap@1%=21.3%** | **Past-only, click-level, frozen** | ❌ Gate-0 失败，性能不达标 |
| Leakage-Free (Rolling) | Top-1%=81.1%, Spearman=0.52, RevCap@1%=98.7% | Past-only, click-level, rolling | ⚠️ 疑似泄漏，不可信 |



## 1) 🌲 核心假设树
```
🌲 核心: 如何建模和评估 EV（期望打赏）以支持排序/分配决策？
│
├── Q0: 基础设施正确性【✅ 已验证，❌ 性能不达标】
│   ├── Q0.1: 现有特征是否有泄漏？ → ✅ 已消除（特征重要性比 1.23 < 2x）
│   ├── Q0.2: 任务定义是否正确？ → ✅ 已修正（click-level 含 0，Y=0 占比 98.5%）
│   ├── Q0.3: 评估指标是否对齐业务？ → ✅ 已实现（Revenue Capture@K）
│   └── Q0.4: Past-only 特征能否支撑预测？ → ❌ 不能（Top-1%=11.6%，性能接近随机）
│
├── Q1: 如何建模 EV？【需在 Q0 解决后验证】
│   ├── Q1.1: Direct Regression vs Two-Stage？ → ⚠️ 需在无泄漏版本复验
│   ├── Q1.2: 多任务学习是否有帮助？ → ❌ 无收益（PR-AUC 反降 1.76pp）
│   ├── Q1.3: 延迟建模是否需要？ → ✅ Baseline 足够（ECE=0.018 优于 Chapelle=0.028）
│   └── Q1.4: 召回-精排分工是否有效？ → 🔆 待在无泄漏版本验证
│
├── Q2: 如何评估 EV 模型？
│   ├── Q2.1: Revenue Capture@K 是否更贴近业务？ → 🔆 进行中（MVP-1.0）
│   ├── Q2.2: 分桶校准（ECE）是否必要？ → 🔆 进行中（MVP-1.0）
│   ├── Q2.3: 切片评估（冷启动/Top-1%/长尾）是否必要？ → 🔆 进行中（MVP-1.0）
│   ├── Q2.4: OPE 方法有效性？ → ✅ SNIPS 可用（相对误差 < 10%）
│   └── Q2.5: 时间切分 + Past-only 是否必须？ → ✅ 必须（避免泄漏）
│
├── Q3: 如何优化 EV 模型？
│   ├── Q3.1: 实时上下文特征是否有帮助？ → ⏳ 待验证（需先完成 Q0）
│   └── Q3.2: 长期画像 vs 实时会话？ → ⏳ 待验证
│
└── Q4: 如何支持长期全局目标（分配优化）？【新增】
    ├── Q4.1: 估计层输出是否足够支持分配决策？ → ⏳ 待验证
    ├── Q4.2: 如何建模主播生态健康（外部性/约束）？ → ⏳ 待验证
    ├── Q4.3: 模拟器能否填补数据不足？ → ⏳ 待验证
    └── Q4.4: KuaiLive 数据能否外推到全站？ → ❌ 否（样本选择偏差，只能研究活跃用户子集）

Legend: ✅ 已验证 | ❌ 已否定 | 🔆 进行中 | ⏳ 待验证 | 🗑️ 已关闭 | ⚠️ 需复验
```

## 2) 口径冻结（唯一权威）

> **⚠️ 重要更新（2026-01-18）**：修正任务定义和特征规范

| 项目 | 规格 | 变更说明 |
|---|---|---|
| Dataset / Version | KuaiLive 数据集 | - |
| **样本单元** | **Click-level（含 0）** | ❌ 不再使用 gift-only |
| **样本构造** | click 后 H=1h 内 gift 总额 | 包含未送礼的 click（Y=0） |
| Train / Val / Test | 按时间切分：前 70% / 中间 15% / 最后 15% 天 | - |
| **特征约束** | **Past-only**（冻结版 + 滚动版） | ❌ 禁止全量聚合回填 |
| **主指标** | **Revenue Capture@K%**（Top-1%, Top-5%） | ❌ 不再使用"集合重叠"定义 |
| 辅助指标 | Spearman, NDCG@100, ECE（校准） | 新增校准和切片评估 |
| Seed / Repeats | 固定 seed=42，多次运行取均值 | - |
| 目标变量 | log(1 + gift_amount)，Y=0 表示无打赏 | - |

### 指标定义（唯一权威）

| 指标 | 定义 | 用途 |
|---|---|---|
| **Revenue Capture@K%** | $\text{RevShare@K} = \frac{\sum_{i \in \text{TopK}(\hat{v})} Y_i}{\sum_i Y_i}$ | 主指标：捕获多少收入（金额占比） |
| Top-K% Capture（旧） | $\text{Overlap@K} = \frac{|\text{TopK}(\hat{v}) \cap \text{TopK}(Y)|}{|\text{TopK}(Y)|}$ | ❌ 废弃：仅是集合重叠 |
| ECE | Expected Calibration Error | 校准：分桶预测 vs 实际 |

> 规则：任何口径变更必须写入 §8 变更日志。
---
## 3) 当前答案 & 战略推荐（对齐问题树）

### 3.1 战略推荐（只保留"当前推荐"）

> **⚠️ 当前阻塞**：所有架构选择需在 **Q0（基础设施正确性）** 解决后复验

- **推荐路线：Route 0 (Fix Foundation)**（先修复泄漏和任务定义，再验证架构选择）
- 需要 Roadmap 关闭的 Gate：**Gate-0（无泄漏 Baseline）** → Gate-1（架构选择复验）

| Route | 一句话定位 | 当前倾向 | 关键理由 | 需要的 Gate |
|---|---|---|---|---|
| **Route 0 (Fix)** | 修复泄漏 + 任务定义 | 🔴 **最高优先级** | 现有结果不可信 | **Gate-0** |
| Route Direct | Direct Regression 全量排序 | 🟡 待复验 | Top-1%=54.5%（可能有泄漏） | Gate-1 |
| Route Two-Stage | Two-Stage 全量排序 | 🟡 待复验 | Top-1%=35.7%（可能有泄漏） | Gate-1 |
| Route 分工 | Direct 召回 + Two-Stage 精排 | ⏳ 暂缓 | 需先完成 Gate-0 和 Gate-1 | Gate-2 |
| Route 分配优化 | EV → 约束优化 → 分配决策 | ⏳ 远期 | 估计层完成后探索 | Gate-3 |

### 3.2 分支答案表（每行必须回答"所以呢"）

| 分支 | 当前答案（1句话） | 置信度 | 决策含义（So what） | 证据（exp/MVP） |
|---|---|---|---|---|
| **Q0.1: 现有特征是否泄漏** | ❌ 是，`pair_gift_mean/sum` 是泄漏特征 | 🔴 确认 | 必须改为 past-only 特征 | exp_baseline（重要性异常） |
| **Q0.2: 任务定义是否正确** | ❌ 否，gift-only ≠ EV | 🔴 确认 | 必须改为 click-level（含 0） | 任务定义分析 |
| **Q0.3: 评估指标是否对齐** | ❌ 否，Top-K% ≠ Revenue Capture | 🔴 确认 | 必须改用 Revenue Capture@K | 指标定义分析 |
| Q0.4: Past-only 特征能否消除泄漏 | 待验证：Top-1% 是否仍 > 40% | 🔆 进行中 | 若通过则特征体系可信 | MVP-1.0 |
| Q1.1: Direct vs Two-Stage | Direct 领先 18.8pp | ⚠️ 需复验 | 需在无泄漏版本上确认结论稳定性 | exp_fair_comparison（可能有泄漏） |
| Q1.2: 多任务学习 | 无收益，PR-AUC 反降 1.76pp | 🟢 高 | 保留单任务方案 | exp_multitask |
| Q1.3: 延迟建模 | Baseline 足够，无需复杂校正 | 🟢 中 | 保持简单架构 | exp_delay_modeling |
| Q1.4: 召回-精排分工 | Stage2 精排有潜力 | ⏳ 暂缓 | 需先完成 Q0 | exp_two_stage_diagnosis |
| Q2.1: Revenue Capture@K | 待验证：是否更贴近业务 | 🔆 进行中 | 作为新主指标 | MVP-1.0 |
| Q2.2: OPE 方法 | SNIPS 可用 | 🟢 中 | 可用于策略离线评估 | exp_ope_validation |
| **Q4.4: 数据能否外推全站** | ❌ 否，样本选择偏差 | 🔴 确认 | 重新表述问题为"活跃用户子集的金主分配" | 数据分析 |
---
## 4) 洞见汇合（多实验 → 共识）

## 🔴 新增关键洞见（2026-01-18 更新）

### I0: 数据泄漏会制造"虚假高性能"——必须先修复基础设施

**观察**：Baseline Top-1%=56.2%、Spearman=0.891 看起来很强，但 `pair_gift_mean` 重要性=328,571，远超其他特征 3 倍以上。

**解释**：在 gift-only 数据里，很多 (user, streamer) 配对只有 1 次送礼记录：
- `pair_gift_count = 1` → `pair_gift_mean == pair_gift_sum == gift_price`（当前样本的真实 label）
- 模型用"答案"预测"答案"，当然准确率爆表
- 这不是"特征很强"，而是"作弊特征"

**决策影响**：
- ❌ 现有 Baseline 不能作为可靠参照
- ✅ 必须用 **past-only 特征**（冻结版或滚动版）重跑
- ✅ 所有聚合特征必须满足：`t < t_impression`（只用该样本发生之前的历史）

### I0a: 任务定义决定模型能学到什么——gift-only ≠ EV

**观察**：gift-only baseline 学的是 $E[\text{gift\_amount} | \text{已送礼}]$，但业务需要的是 $EV = E[\text{gift\_amount} | \text{session}]$（包含 0）。

**解释**：
- gift-only 模型**看不到"谁不会送礼"**——因为样本里没有 Y=0 的 click
- KuaiLive 真实稀疏率 ~1.48%（per-click），但 gift-only 里 100% 都是送礼
- 这导致模型无法学到"区分会/不会送礼"的能力

**决策影响**：
- ❌ 不再使用 gift-only 作为样本单元
- ✅ 改用 **click-level**（click 后 H=1h 内的 gift 总额，未送礼则 Y=0）
- ✅ 如果需要更细粒度，可用 session-level 或 impression-level

### I0b: 评估指标必须对齐业务——"捕获多少人" ≠ "捕获多少钱"

**观察**：Top-K% Capture 是"集合重叠比例"，不关心金额差异。

**解释**：
- 漏掉 1 个超大额（如 1488 元） vs 漏掉 1 个边缘 Top-1%（如 10 元），在业务上完全不同
- 但 Top-K% Capture 只看"有没有捕获"，不看"捕获了多少钱"
- 这会误导模型选择

**决策影响**：
- ❌ 废弃 Top-K% Capture（集合重叠）
- ✅ 改用 **Revenue Capture@K** = $\frac{\sum_{i \in \text{TopK}(\hat{v})} Y_i}{\sum_i Y_i}$
- ✅ 新增分桶校准（ECE）和切片评估（冷启动/Top-1%/长尾）

---

## 核心洞见：EV 模型建模方法论

### I1: 无常 ≠ 不能建模：预测的是"分布"，不是"确定性"

**观察**：单次打赏受情绪、偶然事件、社交氛围影响很大，难以精确预测。

**解释**：但在规模上（成千上万用户×直播间×时段），条件概率是稳定的：
- 同一个人：不同时间、不同主播、不同互动深度，打赏概率差很多
- 不同人：有的人"几乎不打赏"，有的人"偶尔爆发"，还有少数"稳定高消费"

**决策影响**：模型要学到这些系统性差异，哪怕只把排序/分配的"Top 区域"做对一点点，也会有价值。

### I2: "真的有用"的地方通常在：排序/分配，而不是逐条预测

**观察**：最常见的可用目标是 EV（期望打赏）：
$$EV = P(gift > 0 | context) \times E(amount | gift > 0, context)$$

**解释**：你拿 EV 做：
- 给用户推荐哪些直播间/主播（排序）
- 给主播分配曝光/流量（资源分配）
- 识别"高潜会话"做干预（券、引导、互动策略）

**决策影响**：不需要"预测准每一次"，需要"把更可能产生价值的放到更好的位置"。

### I3: 哪些特征/历史数据通常最有信号

**观察**：
- **用户侧（长期）**：过去是否打赏、打赏频率、金额分位数（重尾里特别重要）、最近一次打赏时间（recency）、对哪些内容/主播类型更容易打赏（偏好）
- **会话侧（实时上下文）**：进房时长、互动行为（点赞/评论/关注/停留曲线）、主播热度、当前氛围（在线人数、礼物雨/活动）、时间段、节假日、活动

**解释**：现实里往往是：没有实时上下文只靠静态画像 → 上限很低；加上会话行为 → 效果明显变好。

**决策影响**：必须同时建模长期画像和实时上下文。

### I4: 把不确定性变成策略

**观察**：模型输出概率=不确定性本身。很多样本就是接近底噪（接近全局低打赏率）。

**解释**：接受不可预测性，模型"很不确定"时就别激进分配；模型"很确定"时才集中资源。

**决策影响**：不确定性可以作为策略信号，而非噪声。

### I5: 离线评估：Revenue Capture@K 是关键指标

**观察**：你要的不是"预测准"，而是**把钱集中在更少的曝光里**。

**解释**：
- **Capture@K%**：Top K% 曝光捕获的收入占比
- **Lift@K%**：相对随机提升（Capture(K%) / K%）

**决策影响**：MVP 验收标准："Top 1% 曝光捕获 20% 收入（Lift=20x）"，或"Top 5% 捕获 45% 收入（Lift=9x）"。

### I6: 时间窗纪律：避免数据泄漏

**观察**：任何特征都必须是 **impression 时刻之前**可得。

**解释**：
- ✅ 用户历史累计到 t-ε
- ✅ 主播历史累计到 t-ε
- ❌ 使用跨同一窗口直接统计到未来的特征（极易泄漏）
- ❌ 会话结束后才知道的信息（例如最终停留时长、最终总互动）

**决策影响**：必须严格按时间切分数据，并做泄漏检查。

### I7: 数据切分：时间切分优先

**观察**：必须按时间切分（否则未来信息泄漏很常见）。

**解释**：
- Train：前 70% 天
- Valid：中间 15% 天
- Test：最后 15% 天

**决策影响**：再加一个"冷启动切分"（新用户/新主播测试集）可提升泛化评估。

### I8: Baseline 三层级：保证能对齐"是否有用"

**观察**：
- **B0（最朴素）**：`score(room) = 历史平均打赏金额`（或历史 EV）或 `score = 热度(popularity)`
- **B1（弱个性化）**：`score = user_spend_level * room_ev_level`
- **B2（ML baseline）**：LightGBM / LR，Two-stage 或 Direct 回归

**解释**：Two-stage：`P(gift>0) * E(amount | gift)`；Direct：直接回归 `log1p(amount)`

**决策影响**：MVP 主力是 B2，但必须同时跑 B0/B1 作为对照。

### I9: 上线 A/B 设计：最小可行

**观察**：
- **随机化单位**：user_id（推荐/排序最常用）
- **主指标**：Revenue per DAU / Revenue per impression、Payer conversion、ARPPU/ARPU
- **护栏指标**：用户留存、会话时长、投诉/负反馈、主播曝光集中度（Gini）、活跃主播覆盖、长尾主播收入变化

**解释**：MVP 的 treat 先只改一件事：推荐候选不变，只改排序（rerank by EV），或只在 Top-N 里替换 1-2 个位置（降低风险）。

**决策影响**：逐步放量（1% → 5% → 20% → 50%），护栏不过线（留存/生态指标不恶化）。

### I10: MVP 一句话验收标准

**观察**：
- **离线**：Top 1% / 5% 的 **Capture** 相对 B0 有显著提升，并且在"时间切 test"+"去掉可疑特征"后仍成立
- **在线**：`Revenue per DAU` 或 `Revenue per impression` 正向，护栏不过线（留存/生态指标不恶化）

**解释**：如果模型能把"Top 1% / Top 5% 的 EV 捕获"显著提高，即使整体准确率一般，也通常是有业务价值的。因为打赏就是重尾：价值集中在头部。

**决策影响**：不纠结模型细节，关注业务价值。

> 只收录"会改变决策"的洞见，建议 5–8 条。

| # | 洞见（标题） | 观察（What） | 解释（Why） | 决策影响（So what） | 证据 |
|---|---|---|---|---|---|
| **I0** | **⚠️ 数据泄漏导致 Baseline 无效** | `pair_gift_mean` 重要性=328,571，Spearman=0.891 | 特征直接等于答案（pair_gift_count=1 时） | **必须用 past-only 特征重跑** | exp_baseline（重要性异常） |
| **I0a** | **任务定义不匹配** | gift-only 学的是 E[gift|已送礼]，不是 EV | 模型看不到"谁不会送礼" | **改为 click-level（含 0）** | 任务定义分析 |
| **I0b** | **评估指标偏差** | Top-K% Capture 是集合重叠，不关心金额 | 漏掉大额 vs 漏掉小额，业务影响完全不同 | **改用 Revenue Capture@K** | 指标定义分析 |
| I1 | Direct vs Two-Stage（⚠️ 需复验） | Direct Top-1%=54.5% vs Two-Stage=35.7% | 可能有泄漏，结论不稳定 | 需在无泄漏版本复验 | exp_fair_comparison |
| I2 | Two-Stage 精排有潜力 | Stage2 在 gift 子集 Spearman=0.89 | Stage2 在正样本内排序准 | 若复验通过可用于精排 | exp_two_stage_diagnosis |
| I3 | 多任务学习无收益 | PR-AUC 反降 1.76pp | 密集信号对稀疏打赏迁移效果不显著 | 保留单任务方案 | exp_multitask |
| I4 | 时间切分 + Past-only 是必须的 | 所有实验必须按时间切分 | 避免未来信息泄漏 | **特征必须是 t < t_impression** | exp_baseline |
| **I5** | **KuaiLive 数据适合研究"金主分配"** | 23,400 用户是多行为活跃子集 | 样本选择偏差 | 重新表述问题为"活跃用户子集的金主分配"，不外推全站 | 数据分析 |
---
## 5) 决策空白（Decision Gaps）
> 写"要回答什么"，不写"怎么做实验"。建议 3–6 条。

| DG | 我们缺的答案 | 为什么重要（会改哪个决策） | 什么结果能关闭它 | 决策规则 |
|---|---|---|---|---|
| **DG0** | **Past-only 特征能否消除泄漏且保持性能？** | 决定特征体系是否可信，是所有后续实验的前提 | Top-1% > 40% 且 Spearman 下降 < 0.2 | If 通过 → 特征可信，进入 DG1；Else → 重新设计特征 |
| **DG0a** | **Click-level EV 预测的 Revenue Capture@K 如何？** | 决定任务定义是否正确 | Revenue Capture@1% > 50% | If 通过 → 任务定义正确；Else → 检查数据构造 |
| DG1 | Direct vs Two-Stage 在无泄漏版本上的相对差距？ | 决定模型架构选择 | Direct 仍领先 > 10pp | If 是 → 结论稳定，Direct 为主；Else → 重新评估 |
| DG2 | 召回-精排分工策略是否有效？ | 决定是否采用混合架构 | 分工策略 Top-1% > Direct 单独 | If 是 → 采用分工；Else → 继续优化 Direct |
| DG3 | 实时上下文特征对 EV 预测的提升有多大？ | 决定是否投入实时特征工程 | Top-1% Capture 提升 > 5pp | If 是 → 上线实时特征；Else → 聚焦长期画像 |
| **DG4** | **如何建模主播生态健康（外部性/约束）？** | 决定是否需要从纯 EV 扩展到多目标 | 定义可量化的生态健康指标 | If 有 → 加入约束；Else → 纯 EV 优化 |
| **DG5** | **模拟器能否填补数据不足（长期留存/因果 uplift）？** | 决定是否需要构建模拟器 | 模拟器统计量与 KuaiLive 矩匹配 | If 是 → 用模拟器做策略评估；Else → 仅用离线数据 |
---
## 6) 设计原则（可复用规则）

### 6.1 已确认原则

| # | 原则 | 建议（做/不做） | 适用范围 | 证据 |
|---|---|---|---|---|
| **P0** | **⚠️ Past-only 特征是必须的** | ✅ 做：所有聚合特征必须只用 t < t_impression 的历史 | **所有实验** | I0（泄漏分析） |
| **P0a** | **⚠️ 样本单元必须是 Click-level（含 0）** | ✅ 做：从 gift-only 改为 click + gift join | **所有实验** | I0a（任务定义） |
| **P0b** | **⚠️ 主指标必须是 Revenue Capture@K** | ✅ 做：用收入占比，不用集合重叠 | **所有实验** | I0b（指标定义） |
| P1 | **时间切分是必须的** | ✅ 做：严格按时间切分（前 70% / 中间 15% / 最后 15% 天） | 所有实验设计 | exp_baseline |
| P2 | ~~Direct Regression 做召回，Two-Stage 做精排~~ | ⚠️ 需复验 | 生产环境架构设计 | 需在无泄漏版本复验 |
| P3 | **简单架构优先** | ✅ 做：LightGBM + log(1+Y) | 模型选择 | exp_baseline, exp_delay_modeling |
| P4 | **多任务学习无收益** | ❌ 不做：保留单任务方案 | 模型选择 | exp_multitask |
| **P5** | **重新表述问题边界** | ✅ 做：问题是"活跃用户子集的金主分配"，不是"全站转化/留存" | 研究定位 | I5（数据分析） |

### 6.2 特征工程原则（无泄漏版）

| # | 特征类型 | 正确做法 ✅ | 错误做法 ❌ |
|---|---|---|---|
| F1 | 用户-主播交互 | `pair_gift_sum_past(t)` = cumsum 到 t-ε，shift(1) | `pair_gift_sum` = 全量 groupby 后回填 |
| F2 | 用户侧统计 | `user_total_gift_7d_past(t)` = 只用 t 之前 7 天 | `user_total_gift` = 全量统计 |
| F3 | 主播侧统计 | `streamer_recent_revenue_past(t)` = 只用 t 之前 | `streamer_total_revenue` = 全量统计 |
| F4 | 实现方式 | **冻结版**：只用 train window 统计；**滚动版**：cumsum + shift(1) | groupby 后 merge 回原表（会包含未来） |

### 6.3 待验证原则

| # | 原则 | 初步建议 | 需要验证（MVP/Gate） |
|---|---|---|---|
| P6 | Direct vs Two-Stage 在无泄漏版本上的相对差距 | 🟡 假设 Direct 仍占优 | MVP-1.0 (Gate-0) |
| P7 | 实时上下文特征显著提升 EV 预测 | 🟡 假设成立 | Gate-2 (实时特征实验) |
| P8 | 模拟器能填补数据不足 | 🟡 假设成立 | MVP-2.0 (模拟器) |
### 6.4 关键数字速查（只留会反复用到的）

| 指标 | 值 | 条件 | 来源 | 状态 |
|---|---|---|---|---|
| ~~Baseline Top-1% Capture~~ | ~~56.2%~~ | ~~Direct Regression, gift-only~~ | ~~exp_baseline~~ | ❌ 泄漏无效 |
| Direct Top-1% Capture | 54.5% | Direct Regression, click全量 | exp_fair_comparison | ⚠️ 需复验 |
| Two-Stage Top-1% Capture | 35.7% | Two-Stage, click全量 | exp_fair_comparison | ⚠️ 需复验 |
| Two-Stage Stage2 Spearman | 0.89 | Stage2 在 gift 子集 | exp_two_stage_diagnosis | ⚠️ 需复验 |
| Oracle p 上界 | 90.7% | 真实 p(x) 替换 | exp_two_stage_diagnosis | ✅ 可信 |
| OPE 最佳方法 | SNIPS | 相对误差 < 10% | exp_ope_validation | ✅ 可信 |
| **无泄漏 Baseline 目标** | **> 40%** | **Past-only, click-level** | **MVP-1.0** | 🔆 待验证 |
| **KuaiLive 稀疏率** | **~1.48%** | **per-click gift rate** | **数据分析** | ✅ 参考 |
| **金额分布** | **P50=2, P99≈1488** | **重尾分布** | **数据分析** | ✅ 参考 |

### 6.5 已关闭方向（避免重复踩坑）

| 方向 | 否定证据 | 关闭原因 | 教训 |
|---|---|---|---|
| ~~gift-only 任务定义~~ | 任务分析：学的是 E[gift\|已送礼]，不是 EV | 模型无法学到"谁不会送礼" | **必须用 click-level（含 0）** |
| ~~全量聚合特征~~ | 泄漏分析：pair_gift_mean 重要性异常 | 特征=答案（pair_gift_count=1 时） | **必须用 past-only 特征** |
| ~~Top-K% Capture 指标~~ | 指标分析：集合重叠不关心金额差异 | 漏掉大额 vs 漏掉小额，业务影响不同 | **改用 Revenue Capture@K** |
| ~~Two-Stage 全量排序~~ | exp_fair_comparison: Top-1%=35.7% vs Direct=54.5% | Stage1 分类能力不足（⚠️ 需复验） | 仅用于精排场景 |
| ~~多任务学习~~ | exp_multitask: PR-AUC 反降 1.76pp | 密集信号对稀疏打赏迁移效果不显著 | 保留单任务方案 |
| ~~复杂延迟校正~~ | exp_delay_modeling: Baseline ECE=0.018 优于 Chapelle=0.028 | 简单架构已足够 | 保持简单架构 |
| ~~外推到全站~~ | 数据分析：23,400 用户是活跃子集 | 样本选择偏差 | **只研究"活跃用户子集的金主分配"** |
---
## 7) 指针（详细信息在哪）
| 类型 | 路径 | 说明 |
|---|---|---|
| 🗺️ Roadmap | `./gift_EVmodel_roadmap.md` | Decision Gates + MVP 执行 |
| 📋 Kanban | `status/kanban.md` | 进度中控 |
| 📗 Experiments | `exp/exp_*.md` | 单实验报告 |
| 📊 Consolidated | `gift_EVmodel_consolidated_*.md` | 跨实验综合 |
---
## 8) 变更日志（只记录"知识改变"）

| 日期 | 变更 | 影响 |
|---|---|---|
| 2026-01-18 | 创建 | - |
| 2026-01-18 | 添加数据泄漏相关假设（Q1.1a, Q1.1b, Q2.1a） | 基于 baseline 分析，发现泄漏和任务不匹配问题，需复验结论 |
| **2026-01-18** | **重大更新：发现 Baseline 三个致命问题** | **所有现有结论需复验** |
| | **P1 数据泄漏**：`pair_gift_mean/sum` 是泄漏特征 | Baseline 56.2%/0.891 无效 |
| | **P2 任务不匹配**：gift-only ≠ EV | 必须改为 click-level（含 0） |
| | **P3 评估偏差**：Top-K% Capture ≠ Revenue Capture@K | 必须用收入占比指标 |
| | **P4 样本偏差**：KuaiLive 是活跃用户子集 | 重新表述问题边界 |
| | 新增 Q0 基础设施正确性分支 | 最高优先级 |
| | 新增 Q4 长期全局目标分支 | 分配优化、模拟器 |
| | 新增 DG0/DG4/DG5 决策空白 | 无泄漏验证、生态健康、模拟器 |
| | 新增 P0/P0a/P0b/F1-F4 设计原则 | 特征工程规范 |
| | 废弃 gift-only、全量聚合、Top-K% Capture | 关闭方向更新 |
<details>
<summary><b>附录：术语表 / 背景（可选）</b></summary>

### 术语表
| 术语 | 定义 |
|---|---|
| [...] | [...] |

### 背景
- [...]

</details>
