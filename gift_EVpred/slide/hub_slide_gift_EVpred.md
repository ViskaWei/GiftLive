# Gift EVpred（估计层）— 8min 汇报

> **Topic**: Gift EVpred Hub Overview  \
> **Author**: Viska Wei  \
> **Date**: 2026-01-20  \
> **Language**: 中文

---

## 估计层定位：预测每个 click 的打赏期望值 (EV)，为分配层排序打分

- **目标**：预测"给了这个行动（把用户分配给某主播），会发生什么"——即打赏期望值 EV
- **任务定义**：

| 项 | 规格 |
|---|---|
| 输入 | `(user_id, streamer_id, live_id, t_click)` + 进入时刻可用特征 |
| 输出 | 未来窗口 `[t_click, t_click+W]` 内礼物金额总和的期望 |
| 标签 | `gift_price_label = sum(gift_price)` within window W |

- **数据现实**
  - 打赏率：**1.48%**（per-click，极稀疏）
  - 金额：P50/P90/P99 = **2 / 88 / 1488**（极重尾）
  - 集中度：User Gini **0.942**，Streamer Gini **0.926**
- **当前成果**：RevCap@1% = **51.4%**（Top 1% 预测捕获 51.4% 收入）

<details>
<summary><b>note</b></summary>

大家好，今天汇报的是 **gift EVpred 估计层**——这是分配系统的基础模块。

任务很直接：对每个用户进入直播间的事件（click），预测未来一段时间内他会打赏多少钱。注意，我们预测的不是"是否打赏"，而是"打赏期望值"——因为 100 元和 10000 元的大哥对业务的价值完全不同。

先看数据现实：打赏率只有 **1.48%**，极稀疏；金额分布极重尾，P50 才 2 块，但 P99 高达 1488；而且收入高度集中，用户侧 Gini 0.942。

当前的核心成果是：我们的模型能让 Top 1% 预测捕获 **51.4%** 的实际收入——这意味着如果分配层只服务这 1% 用户，就能拿到一半以上的收入。

</details>

---

## 数据处理：Day-Frozen 防泄漏 + Last-Touch 归因 + 1min 窗口

**核心设计**：任何特征必须是 `t_click` 之前可得，绝对禁止未来信息泄漏

```
数据处理流程（无泄漏设计）
├── 1. 时间划分：7-7-7 按天（Train/Val/Test）
├── 2. 归因方式 ⭐
│   ├── Last-Touch：每个 gift 只归因最近 1 条 click
│   └── gift_id 去重：避免同一 gift 被多条 click 计入
├── 3. 标签窗口 ⭐
│   ├── 默认 1 分钟（覆盖 92.6% gift）
│   └── 98.2% gift 在 click 同毫秒内发生
├── 4. 特征冻结 ⭐
│   ├── Day-Frozen：历史特征只用 day < click_day 的数据
│   └── 严格 merge_asof(allow_exact_matches=False)
└── 5. Strict 模式：drop 快照特征（fans/accu_*），避免 KuaiLive 快照泄漏
```

- **关键验证**
  - 200/200 样本无泄漏验证通过
  - Over-Attribution 修复：8.43% gift 曾被多条 click 计入 → Last-Touch 修复
- **Cold Start 效应（不是泄漏）**：Train RevCap=29.4% << Test=52.6%，原因是 Train 早期历史特征几乎全为 0

<details>
<summary><b>note</b></summary>

数据处理是这个项目最花时间的部分，因为打赏预测极易泄漏——如果不小心，模型会学到"未来信息"而产生虚高的指标。

我们的设计有三个关键点：

第一是 **Day-Frozen 特征**：所有历史统计量都只用 `day < click_day` 的数据。比如计算用户历史打赏总额，只能用昨天及之前的数据。这通过 `merge_asof(allow_exact_matches=False)` 严格保证。

第二是 **Last-Touch 归因**：同一个用户可能短时间内多次进入同一直播间，同一个 gift 可能被多条 click 的窗口覆盖。我们只把 gift 归因给最近的那条 click，并按 gift_id 去重，避免金额膨胀。

第三是 **1 分钟窗口**：分析发现 98.2% 的 gift 在 click 同毫秒内发生，1 分钟窗口已经覆盖 92.6%。更大窗口增加归因噪声。

有一个看起来像泄漏但其实不是的现象：Train 的 RevCap 只有 29%，远低于 Test 的 52%。原因是数据集开始时没有历史——第一天的样本完全没有历史特征可用。这是 **Cold Start 效应**，不是泄漏。

</details>

---

## Baseline v1.0：Ridge 回归 + raw Y + 20 特征 = RevCap 51.4%

**模型选择结论**：Direct + raw Y (Ridge) 是当前最优方案

| 模型/方法 | RevCap@1% | 结论 |
|-----------|-----------|------|
| **Ridge (raw Y, Strict)** | **51.4%** | ✅ Final Baseline |
| Two-Stage (P(gift)×E[Y\|gift]) | 45.66% | ❌ 引入额外误差 |
| LightGBM + raw Y | ~45% | ❌ Tree 不适合稀疏回归 |
| Direct + log(1+Y) | ~35% | ❌ log 压缩 whale 信号 |

- **关键洞见**
  - **Raw Y >> Log Y**：log 变换把 10 元和 1000 元压缩成 1 和 3，丢失大哥信号（RevCap +39.6%）
  - **回归 > 分类**：分类丢失金额信息，AUC=0.85 但 RevCap=36.7%
  - **Linear > Tree（稀疏场景）**：98.5% 样本为 0，树模型预测众数

- **20 特征（Strict Mode）**

| 类别 | 特征 |
|------|------|
| Pair 历史 (3) | `pair_gift_cnt_hist`, `pair_gift_sum_hist`, `pair_gift_mean_hist` |
| User 历史 (3) | `user_gift_cnt_hist`, `user_gift_sum_hist`, `user_gift_mean_hist` |
| Streamer 历史 (3) | `str_gift_cnt_hist`, `str_gift_sum_hist`, `str_gift_mean_hist` |
| User Profile (6) | `age`, `gender`, `device_brand`, `device_price`, `is_live_streamer`, `is_photo_author` |
| Room (2) | `live_type`, `live_content_category` |
| Time (3) | `hour`, `day_of_week`, `is_weekend` |

<details>
<summary><b>note</b></summary>

这一页讲我们怎么确定 Baseline。我们尝试了很多方案，最终结论非常清晰：

**Ridge 回归 + raw Y 是最优的**。

几个关键对比：

第一，**raw Y vs log(1+Y)**：log 变换看起来合理（处理长尾），但它把 10 元和 1000 元的差距从 100 倍压缩成 3 倍。我们的目标是识别大哥，log 反而削弱了这个信号。换成 raw Y 后，RevCap 提升了 39.6%。

第二，**回归 vs 分类**：我们也试过用 LightGBM 做二分类（是否打赏），AUC 高达 0.85，看起来很好。但 AUC 不等于 RevCap——分类丢失了金额信息，100 元和 10000 元的大哥得分相同。

第三，**Linear vs Tree**：LightGBM 回归失败了，RevCap 反而下降。原因是 98.5% 样本打赏为 0，极度稀疏，树模型容易预测众数。Ridge 这种简单线性模型反而更稳定。

特征方面，我们用 Strict Mode 只保留 20 个真正可用的特征。最强的信号是 **pair 历史**——这个用户给这个主播打赏过多少次、多少钱。

</details>

---

## 指标体系 v2.2：7 层 38 指标，从选模型到上线决策全覆盖

**为什么需要指标体系**：打赏预测场景下，传统指标（MAE/AUC）与业务目标脱钩

$$\text{RevCap@K} = \frac{\sum_{i \in \text{Top}_K(\hat{y})} y_i}{\sum_{i=1}^{N} y_i}$$

| 层级 | 用途 | 核心指标 |
|------|------|---------|
| L1 主指标 | 选模型 | RevCap@K（Top K% 捕获收入占比） |
| L2 诊断 | 理解行为 | WRecLift（大额事件识别）、WRevCap |
| L3 稳定性 | 评估鲁棒 | CV < 10%、Bootstrap CI |
| L4 校准 | 预测准度 | ECE、Tail Ratio |
| L5 切片 | 公平分析 | ColdStart、Whale 分层 |
| L6 生态 | 分配护栏 | Gini、Coverage |
| **L7 决策核心** | **上线判断** | **Regret、PSI < 0.25、EffN** |

- **当前评估结果**

| 指标 | 值 | 解读 |
|------|-----|------|
| RevCap@1% | **51.4%** | 主指标达标 (>50%) |
| CV (稳定性) | **9.4%** | 首次 < 10% 阈值 |
| WRecLift@1% | **34.6×** | 大额事件识别强 |
| PSI | **0.155** | < 0.25 可上线 |
| WhaleUserPrec@1% | **72.6%** | 池子纯度高 |

<details>
<summary><b>note</b></summary>

指标体系是我们花大力气建立的。传统 ML 指标在打赏场景完全不适用：

- **MAE**：把头部误差和尾部同权，但一个 10000 元大哥预测错比 1000 个 0 元用户重要得多
- **AUC**：衡量区分能力，但不关心金额大小——100 元和 10000 元打赏的 AUC 贡献相同

我们用的核心指标是 **RevCap@K**：Top K% 预测捕获的实际收入占比。这直接对齐业务目标——如果分配层只服务 Top K% 用户，能拿到多少收入。

指标体系分 7 层：
- L1-L2 用于选模型和调参
- L3-L4 评估稳定性和校准
- L5-L6 是公平性和生态护栏
- L7 是上线决策核心：Regret（错分代价）、PSI（分布漂移）、EffN（多样性）

当前评估结果：RevCap@1%=51.4% 达标，CV=9.4% 首次低于 10% 阈值，PSI=0.155 低于 0.25 上线阈值。模型可以上线。

</details>

---

## 核心洞见：识别大哥已够用，真正杠杆在"分配"

**已验证的关键共识（K1-K18）**

| # | 共识 | 证据 | 决策影响 |
|---|------|------|---------|
| K1 | **排序/分配 > 逐条预测** | RevCap vs MAE | 评估用 RevCap@K |
| K2 | **Raw Y >> Log(1+Y)** | RevCap +39.6% | 预测目标用 raw Y |
| K4 | **历史打赏是最强信号** | pair_hist 系数 0.194 | 重点挖掘 pair 特征 |
| K7 | **回归 > 分类（for RevCap）** | AUC↑但RevCap↓ | 用回归不用分类 |
| K8 | **Linear > Tree（稀疏场景）** | LGB -14.5% | 稀疏回归用 Linear |
| K15 | **🔥 打赏损害留存** | 打赏用户留存 -10.3pp | 短期收益 ≠ 长期价值 |
| K16 | **收入极度集中** | Gini=0.926 | 分配层需生态约束 |

**⚠️ 当前局限（需分配层解决）**

- 首次打赏预测弱：56% 收入来自无历史 pair，但 RevCap 只有 29.6%（模型依赖复购信号）
- 留存不可预测：AUC=0.57（接近随机），当前特征无法预测次日留存
- 预测更准的边际收益有限：Ridge 已达 Linear 瓶颈，LightGBM 无法突破

<details>
<summary><b>note</b></summary>

这一页总结我们在估计层学到的核心洞见。

首先是 **方法论层面**：
- 我们的目标不是"预测准每一条"，而是"把高价值放对位置"——所以用 RevCap 评估
- raw Y 比 log Y 好 39.6%，因为 log 压缩了大哥信号
- 回归比分类好，因为分类丢失金额信息
- Linear 比 Tree 好，因为 98.5% 为 0，树模型容易预测众数

然后是 **业务层面**：
- 历史打赏是最强信号，特别是 pair 级别的历史
- 但这也意味着首次打赏很难预测——56% 收入来自第一次打赏的 pair，模型能力有限

最重要的洞见是 **K15 和 K16**：
- 打赏用户的留存反而比未打赏用户低 10.3pp——"满足后离开"效应
- 收入极度集中，Gini=0.926

这告诉我们：**预测更准的边际收益已经很有限了**。真正的杠杆不在"预测更准"，而在"分配更对"。

</details>

---

## 下一步：从"预测 gift"到"预测行动后果"→ 进入分配层

**当前预测目标 vs 长期目标**

| 阶段 | 预测目标 | 指标 |
|------|---------|------|
| **Phase 1（当前）** | $E[\text{gift\_amount} \mid \text{click}]$ | RevCap@1%=51.4% ✅ |
| **Phase 2（下一步）** | $E[\text{行动后果} \mid u, s, ctx, a]$ | 多目标分解 |

**长期目标分解**：预测"给了这个行动（分配），会发生什么"

$$r^{rev}_t = E[\text{gift\_amount}_{t:t+H} \mid u, s, ctx, a]$$
$$r^{usr}_t = E[\text{return}_{t+1d} \mid u, history, a]$$
$$r^{eco}_t = -\lambda \cdot \text{concentration}(\text{exposure/revenue})$$

**🔥 关键结论：杠杆在分配层，不在预测层**

| 策略 | 收益倍数 | 来源 |
|------|---------|------|
| Greedy vs Random | **~3×** | 分配策略差异 |
| + 冷启动软约束 | **+32%** | 探索发现高潜主播 |
| 预测模型优化 | ~+5% | 边际递减 |

→ **下一阶段**：进入 [gift_allocation_hub](../gift_allocation/gift_allocation_hub.md)，把分配问题当成"带约束、带外部性、带风险"的在线资源分配问题来做。

<details>
<summary><b>note</b></summary>

最后一页我要讲清楚：为什么我们不继续优化预测模型，而是要进入分配层。

当前我们的预测目标是"这条 click 会产生多少打赏"。但长期目标其实是"全局收益最大化"，这包括三部分：
- **短期收益**：打赏金额
- **用户留存**：用户明天还会不会来
- **生态健康**：收入是否过度集中在头部主播

我们在预测层做了很多实验，发现：
- 留存基本不可预测（AUC=0.57）
- 生态是全局指标，不是逐条标签

更关键的是，**杠杆对比**非常悬殊：
- 分配策略从 Random 换成 Greedy，收益翻 3 倍
- 加上冷启动软约束，再涨 32%
- 而预测模型从 Ridge 换成更复杂模型，提升只有 5% 左右

所以结论很清晰：**从"预测更准"转向"分配更对"**。

下一阶段进入 gift_allocation，把分配问题当成"带约束、带外部性、带风险"的在线资源分配问题来做。这才是真正的杠杆所在。

</details>

---

## 备忘：8分钟口播节奏建议

| 页数 | 时长 | 内容 |
|------|------|------|
| 1 | 55s | 问题定义 + 任务 I/O |
| 2 | 65s | 数据处理（防泄漏 + 归因 + 窗口） |
| 3 | 60s | Baseline + 关键洞见 |
| 4 | 55s | 指标体系 + 评估结果 |
| 5 | 55s | 核心共识 + 局限 |
| 6 | 55s | 下一步 → 分配层 |
