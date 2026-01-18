根据 **2026-01-08 的 Baseline 实验报告**，你现在这套 baseline（EXP-20260108-gift-allocation-02）“作为对比基准”在报告里看起来很强（Top-1% Capture=56.2%，Spearman=0.891，且最重要特征是 `pair_gift_mean`/`pair_gift_sum`）。但如果你的目标是“金主分配 / 全局长期收益最大化”，这套 baseline **在定义、特征、评估口径上都有关键问题**，其中最致命的是**数据泄漏**与**任务不匹配**。

下面我按你问的两部分回答。

---

## （1）这个 baseline 对吗？输入输出是什么？合理吗？指标合理吗？`pair_gift_mean / pair_gift_sum` 是不是泄漏？

### 1.1 baseline 的“输入/输出”到底是什么？

**报告口径（baseline 任务定义）**：

* 目标：用 LightGBM 回归预测 `log(1+gift_amount)`，并用 MAE(log)、Spearman、Top-K% Capture 做评估。
* 数据切分：按天时间切分 Train/Val/Test = 25,985 / 24,029 / 22,632（总计约等于 gift 记录数 72,646）。

**这意味着：样本单元 = “一次已经发生的打赏记录（gift event）”**。
所以它学到的是：

> **E[ gift_amount | 这条样本本来就发生了 gift ]**（“已送礼条件下的金额”）

而不是你写的：

> **EV_gift = E[gift_amount | session, item]**（包含 0 的“期望收益”）

这两者差别非常大：后者必须包含大量 **未送礼（Y=0）** 的 session/click/impression 才能学到“谁根本不会送礼”。KuaiLive 的真实稀疏性在 per-click 口径约 1.48%（即大量 0），但 gift-only baseline 完全看不到这些 0。

**结论**：

* **如果**你的任务只是“在已发生打赏的样本里预测金额”，这个任务定义本身可以成立；
* **但**如果你的任务是“给推荐/分配决策提供 EV”，那 gift-only baseline **不对题**。

---

### 1.2 `pair_gift_mean / pair_gift_sum` 用来预测 gift 是否数据泄漏？

你问的非常关键：**在你当前 baseline 的实现方式下，它几乎必然是泄漏（target leakage + future leakage）**，原因不是“用历史行为不允许”，而是“你构造特征的方式把标签喂回了特征”。

#### 为什么说它是泄漏（直觉版）

在 gift-only 数据里，对很多 (user, streamer) 配对：

* `pair_gift_count` 很可能是 1（只送过一次）
* 那么 `pair_gift_sum == gift_price`，`pair_gift_mean == gift_price`

也就是说：**特征里直接包含了当前样本的真实 label**（甚至就是同一个数），模型当然“预测很准”，重要性也会爆炸——这正是你看到的现象：`pair_gift_mean` 重要性 328,571，远超第二名 `pair_gift_sum` 102,973。

> 这不是“特征很强”，而是“特征在很多样本上等于答案”。

#### 更严格的判定标准

* **允许**：`pair_gift_sum_past(t)` = 只用 **t 之前** 的历史累计（在线系统可用）
* **泄漏**：`pair_gift_sum_all` = 用全量数据聚合后回填到每条样本（包含 t 之后、包含当条样本自身）

你报告里的高 Spearman=0.891、NDCG@100=0.716 也与这种泄漏现象高度一致。

**结论**：

* `pair_gift_mean / sum` 本身不是原罪；
* **“用全量 gift 表 groupby 得到的 pair_gift_* 再回填到同一张表”** 才是原罪。
* 在当前 baseline 中，它更像“作弊特征”，会把你的 baseline 变成一个**虚高上限**，无法作为后续实验的可靠参照。

---

### 1.3 指标合理吗？

报告里用的指标：MAE(log)、Spearman、Top-K% Capture、NDCG@100。

这些指标在“纯预测任务”里是可以的，但对“金主分配/全局长期收益”来说存在 4 个问题：

1. **评估集合不对（gift-only）**
   Top-K% Capture 在 gift-only 上测的是：

> 在“已经送礼”的事件里，能不能把“大额事件”排前面
> 这与“给用户/主播做分配决策”不是一回事。

2. **Top-K% Capture 的定义不等于“捕获收入”**
   报告里 Top-K Capture 是“集合重叠比例”：

* 你捕获了 Top 1% 的哪些样本（recall@top1%）
  但它不关心**金额差异**：漏掉 1 个超大额 vs 漏掉 1 个边缘 top1% 在业务上完全不同。

3. **Spearman 是全局相关，不是“受限资源下的收益”**
   资源约束/容量约束下，你关心的是：

* 给定只能分配 K 次曝光/机会，能拿到多少 revenue + 留存 + 生态健康
  这更像是一个“policy evaluation”问题，而不只是相关性。

4. **log 目标的“期望”不是 money 的期望**
   如果你用 log(1+Y) 回归，你得到的是 **E[log(1+Y)|X]**，不是 **E[Y|X]**。

* 对排序来说 log 是单调变换，很多时候还行
* 对“收益最大化”来说，必须关注 **原始金额空间的 EV**（否则会系统性偏差）

---

### 1.4 你提到的数据 bias：23400 用户是筛选过的，有局限吗？

是的，而且这个局限必须写清楚，否则后面所有“长期全局目标”都会被质疑。

你在 EDA 里已经观察到一个关键现象：

* `gift_rate`（用户级）可能接近 100%，因为 `user.csv` 很可能只包含“有打赏的用户”；
* `unique_click_users == unique_gift_users` 的统计也支持这种口径偏差。

**这意味着 KuaiLive 更像是“活跃用户子集/多行为完整用户子集”的日志**，不是全站用户。

对你的课题（金主分配）影响是：

* ✅ **你可以研究**：在“已知会送礼/活跃用户”里，如何做更好的分配与生态约束（这对“金主分配”反而很贴合）
* ❌ **你不能直接回答**：全站层面的“转化率提升多少”“新增金主多少”“对普通用户留存影响多少”（因为样本选择偏差）

**建议**：把问题重新表述为

> “在高价值/高活跃用户池中，如何在资源约束下做主播侧分配，最大化长期全局目标”
> 这会让你的数据与目标更一致，也更不容易被质疑“样本有 bias”。

---

### 1.5 “直接回归 EV” vs “Two-stage”——你现在的结论合理吗？

你们的公平对比实验（同一 click 全量候选集）报告结论是：

* Direct Regression Top-1% Capture = 54.5%
* Two-Stage = 35.7%
  差 18.8pp，且 V2 修复后仍如此。
  并且 Two-stage 在 NDCG@100 更好。

这在“极度稀疏”场景下**是可能合理的**：

* 直接回归在含大量 0 的数据上，能学到“谁不会送礼”（负样本信息量巨大）
* two-stage 的 stage2 只用很少的正样本，容易外推不稳；`p×m` 的组合会放大校准误差

**但必须强调：这个结论要在“无泄漏特征”版本上再确认一次**。因为一旦 `pair_gift_sum` 之类的特征含有未来信息，direct/two-stage 都会被“抬到很高”，相对差距是否稳定不一定。

---

## （2）如果不合理，怎么设计“合理实验”，服务长期全局目标？数据不够怎么办，如何模拟？

下面我给你一个“估计层（prediction layer）→ 决策层（allocation layer）”的实验设计骨架，重点是：**你到底要预测什么，才真的能服务长期目标**。

---

## 2.1 先定“估计层”的正确预测目标：不是预测 gift，而是预测“行动的后果”

你写的长期目标是：

> Total Revenue + 用户留存/满意度 + 主播生态健康

这不是单一 reward。一个可落地的分解是：对每次“把某用户分配/曝光给某主播”的行动 a，要预测：

1. **短期收益**

* ( r^{rev}*{t} = \mathbb{E}[\text{gift_amount}*{t:t+H} \mid u,s,ctx,a] )

2. **用户侧长期**（留存/满意度的代理指标）

* ( r^{usr}*{t} = \mathbb{E}[\text{return}*{t+1d} \mid u,history,a] )
* 或 ( \mathbb{E}[\Delta \text{watch_time}, \Delta \text{engagement} \mid a] )

3. **主播生态健康**（外部性/约束项）

* 例如：主播侧 exposure / revenue 的集中度、长尾扶持程度、过载惩罚
* ( r^{eco}_{t} = -\lambda \cdot \text{concentration}(\text{exposure/revenue}) )
* 或者每个主播一个 concave utility：( U_s(x) ) 边际递减（避免“让某个主播爆”）

> **关键点**：你要预测的是“给了这个行动，会发生什么”，而不是“这条样本是否送礼”。

---

## 2.2 重新定义训练样本：从 gift-only 改为 click/impression/session（含 0）

### 推荐的三种样本单元（按可用数据由易到难）

**A) Click-level（最现实/最接近你们已做的公平对比）**

* 一条 click 代表用户进入某直播间的一次机会
* label：在该 click 的观看窗口内（或 H=1h）发生的 gift 总额，没发生则 0
* 这是你写的 ( EV_gift ) 的一个合理近似

**B) Session-level（更贴近“延迟/多次打赏”）**

* session 可以用“进入同一 live_id 的连续观看”构造
* label：session gift 总额（0/正）
* 可以天然处理“进房很久才打赏 / 多次打赏”

**C) Impression-level（真正的推荐/分配决策点）**

* 需要 negative/impression 日志（如有 `negative.csv` / 曝光日志）
* label：曝光 → 点击 → 观看 → 打赏 的链路收益
* 这才是严格意义上的 allocation 输入

> 你们 EDA/HUB 已提到 Gift-Click 匹配率很高（>90%）可以支撑 click/session 级构造。

---

## 2.3 做一个“无泄漏”的特征体系：所有聚合特征都必须是 past-only

你现在最强的信号是 `pair_gift_*`。要保留它，但必须改成：

* `pair_gift_sum_1d/7d/30d_past`
* `pair_last_gift_time_gap`
* `pair_watch_time_7d_past`
* `user_total_gift_7d_past`, `user_budget_proxy`（比如最近 7 天总额）
* `streamer_recent_revenue`, `streamer_recent_unique_givers`
* `streamer_overload_proxy`：最近 1 小时/1 天的观看人数、互动量（承接上限代理）

**实现上两种方式：**

1. **最简单的“冻结特征”**：

* 所有聚合特征只用 train window 统计
* val/test 只查这张 train 统计表（不会用到未来）
  优点：简单、绝对无泄漏；缺点：低估了线上“特征随时间更新”的能力

2. **更真实的“在线滚动特征”**：

* 按 timestamp 排序
* 对 user/pair/streamer 做 cumsum/expanding，并 shift(1)（排除当前样本）
  优点：接近线上；缺点：工程复杂

你要向主管展示专业性，我建议做 **冻结版 + 在线滚动版** 两套：冻结版当严格下界，滚动版当接近线上上界。

---

## 2.4 模型设计：direct vs two-stage 怎么做才“数学上对业务有意义”？

你提出的两种：

* 直接回归：( \hat{v}(x) \approx \mathbb{E}[Y|X] )
* two-stage：( \hat{v}(x) = \hat{p}(x)\cdot \hat{m}(x) )

都可以，但要注意一个经常踩坑的点：

> **如果 stage2 学的是 ( \mathbb{E}[\log(1+Y)|Y>0,X] )**，那 ( p\times m ) 不是 ( \mathbb{E}[Y|X] )（量纲不对）。

更推荐的 two-stage 组合是：

* Stage1：( p(x)=P(Y>0|X) )
* Stage2：直接预测 **raw amount 的条件期望** 或者预测 log 后做校准：

  * 方案1：Stage2 用 Tweedie/Gamma 类目标回归 raw (Y)
  * 方案2：Stage2 预测 (\mu=\mathbb{E}[\log(1+Y)])，再用校准把 (\mu) 映射为 ( \mathbb{E}[Y] )

**为什么你们“极稀疏下 direct 更强”是合理的**：公平对比里 direct Top-1% 更好，这是典型现象；但要注意：two-stage 在 NDCG@100 更好，意味着它可能更擅长“金主内部精细金额区分”。所以后续可以走：

* 用 direct 做粗排 EV（强召回 top spender opportunity）
* 用 stage2/m(x) 做精排（在候选里更细的金额排序）

---

## 2.5 评估指标：从“预测好不好”升级为“能不能支持分配决策”

### 估计层必须加的“决策相关指标”

除了 MAE/Spearman 这类预测指标，你要加：

1. **Revenue Capture（收入捕获）**

* 定义：取预测 top-K% 的样本集合，计算它们在真实上的总收入占比
  [
  \text{RevShare@K}=\frac{\sum_{i\in \text{TopK}(\hat{v})}Y_i}{\sum_i Y_i}
  ]
  比“集合重叠的 Top-K Capture”更贴近业务。

2. **分桶校准（对 EV 或 P(gift)）**

* 你要做 allocation，就怕“把 0.01 误当 0.1”这种放大效应
* 所以必须看 calibration（ECE / reliability curve），公平对比也强调了校准。

3. **切片评估（必须做）**

* 冷启动 pair（train 中 pair_gift_count=0）
* 冷启动 streamer（历史收礼=0，KuaiLive 冷启动主播比例很高）
* top1% 用户、top10% 用户、长尾用户分别看（因为分布极端集中）

### 决策层的“生态健康指标”

你要回答“不是让某个主播爆”，那必须把指标写进评估：

* Streamer 曝光/收入的 Gini / Top-share（惩罚过度集中）
* 覆盖率：被服务的主播比例、长尾主播被分配到的曝光
* 单主播过载 proxy：某主播在短时间被分配的高价值用户数（上限约束）

---

## 2.6 你当前数据能不能完成“长期全局目标”？

我会给一个坦诚但专业的判断：

### 能做的（在 KuaiLive 范围内）

* 在“活跃/多行为用户子集”内，做 click/session 级 EV 估计与排序
* 研究稀疏、重尾、延迟带来的建模问题
* 在 offline 上构造“受限预算/容量”的分配优化（用预测 EV + 约束求解）

### 难做/做不了的（需要额外假设或模拟）

* 严格的“长期留存”——数据时间窗有限、且用户子集有选择偏差
* 严格的“因果 uplift”——日志缺少随机探索/propensity，难做可靠反事实
* 主播承接上限/生态外部性——需要额外的系统指标或长期观测

---

## 2.7 如果数据不足：怎么生成“能研究分配优化”的模拟数据？

你们 roadmap 里已经写了 Simulator V1 的方向（用户财富分布 + 偏好匹配 + 概率模型 + 金额模型 + 延迟 + 边际递减）。我建议把模拟器明确成一个 **可用于“策略评估”的环境**：

### 模拟器最小闭环（推荐）

**实体**：用户 u、主播 s、时间 t
**状态**：

* 用户：财富/预算、偏好向量、疲劳度、近期满意度
* 主播：质量、容量上限、近期曝光/收入、生态健康分
  **行动**：给每个用户在 t 时刻分配一个主播（或 Top-K）
  **观测**：click/watch/comment/like/gift（带延迟）
  **奖励**：
  [
  R_t = \alpha\cdot \text{Revenue}_t + \beta\cdot \text{UserSat}_t - \gamma\cdot \text{ConcentrationPenalty}_t
  ]
  其中 ConcentrationPenalty 可用 Gini/Top-share 或 concave utility 形式。

### 如何校准到 KuaiLive（让模拟器“像真的”）

用 KuaiLive 的关键统计量做 **矩匹配/ABC**：

* gift 稀疏率（per-click ~1.48%）
* 金额分布重尾（P50=2, P99≈1488）
* 用户/主播集中度（Gini）
* 重复消费/绑定关系强（由 pair_* 特征主导也侧面说明）
* 延迟分布（可以用你们 delay modeling 拟合的 Weibull + 0 质量点）

### 为什么模拟对你们“分配优化”特别重要？

因为你们真正要解决的是：

* 大哥稀缺 + 主播容量 + 外部性（过度集中伤生态）
  这本质是 **在线资源分配/匹配** 问题。没有可控环境，很难系统迭代策略、也很难做可靠 OPE。

---

# 给你一个“落地版结论”（你可以直接写进下一版报告）

1. 现有 baseline 的 Top-1% / Spearman 很可能被 `pair_gift_mean/sum` 的泄漏抬高，不能作为可信上限；需要做 **past-only 特征** 的重跑。
2. gift-only 回归预测的是“已送礼条件下金额”，不是 EV；估计层必须转到 click/session（含 0）任务定义。
3. two-stage 在当前 click 全量公平对比下 Top-1% 不占优是可能合理的，但要在“无泄漏”版本上复验；并且 two-stage 的优势可能体现在 Top-100 精排（NDCG）。
4. 数据确有样本选择偏差（多行为活跃用户子集），适合研究“金主分配”，但不适合直接外推全站转化/留存；长期目标与外部性建议用模拟器补齐。

如果你愿意，我也可以基于你们现有的 click/session 构造方式，把“无泄漏特征的最小可复现实验清单（5 个实验 + 每个实验的验收指标 + 预期现象）”直接写出来，用来指导你下一轮快速跑通并对齐主管的关注点。
