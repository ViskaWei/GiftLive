下面我给你一个**“最终可落地（可写进 Card / PR + 可直接对应 `data_utils.py` 修改点）”**的 Gift EVpred 数据处理方案，并顺带把你现在 `data_utils.py` 里**合理的/需要改的**地方逐条审计一遍。

---

# Gift EVpred 最终数据处理方案（Final）

## 0) 任务定义与数据口径

**目标：**做 **click-level** 的 Gift EV prediction（每次进入直播间 click，预测其带来的礼物金额 EV）。

* 样本粒度：**click**
* 回归目标：

  * `target_raw = gift_price_label`（归因到该 click 的礼物金额）
  * `target = log1p(gift_price_label)`（训练用）
* 辅助二分类：`is_gift = 1[gift_price_label > 0]`

---

## 1) Label 归因设计（最终定版）

### 1.1 归因模型：Last-Touch（Last-Click）

**规则：**每个 gift 只归因给**最近的一条 click**（同 `user_id, streamer_id, live_id`），并且要求 gift 发生在 lookback 窗口内：

* 候选 click：`click_ts ≤ gift_ts ≤ click_ts + W`
* 选择：`click_ts` 最大那条（Last-touch）
* 聚合：把归因后的 gift 金额 `sum` 到 click 上

> 你之前的“重复匹配导致 16.78% 金额膨胀”已经证明：**必须从 gift 粒度 dedup，再聚合到 click 粒度**。

### 1.2 归因窗口：默认 **1 分钟**（可调）

你现在全量口径下的 Coverage 分解非常清晰：

* **无 click**：4.8% gift / 5.7% 金额（本质不是窗口问题）
* **有 click 且 Δt ≤ 1min**：92.6% gift / 90.0% 金额
* **有 click 且 Δt ≤ 60min**：93.8% gift / 92.4% 金额
  → 相比 1min 只多 **+1.2% gift / +2.4% 金额**
* **Δt > 60min**：0.3% gift / ~0.2% 金额（几乎可忽略）

**最终选择：默认 W = 1min。**理由：

* 增加到 60min 的收益很小（+2.4% 金额）
* 但长窗口更容易引入“非进房触发”的噪声与混杂
* KuaiLive 论文明确提到**时间戳做过 offset + rounding**，所以不建议用“只匹配同毫秒（0ms window）”这种超脆弱口径，否则很容易因为四舍五入/对齐误伤真实归因。 ([arXiv][1])

> 如果你之后想做敏感性分析：保留 `label_window_minutes` 参数（1/5/10/60），但主线实验统一用 1min。

### 1.3 去重规则：每个 gift 只归因一次（强约束）

**必须用 gift 的唯一标识做 dedup**。这点你当前代码还有一个潜在坑（下面“代码审计”会说）：现在是用 `groupby([... , gift_ts])` 去 dedup，如果同一 (user, streamer, live) 在同一时间戳存在多条 gift 记录，会被错误折叠成 1 条。

**最终要求：**

* `gift_id` = gift 行号（或原始 id，如果有）
* Last-touch dedup：`groupby(gift_id).idxmax(click_ts)`

---

## 2) 未归因 gift 怎么处理（你问的“这些怎么处理”的最终答案）

你现在主要有两类未归因：

### 2.1 类别 A：**完全没有 click 的 gift（4.8% / 5.7% 金额）**

**最终处理：作为 “Out-of-scope revenue / orphan_gifts_no_click” 单独记录，不回填到 click label。**

原因（很务实）：

* click-level 任务要求：label 必须落在某条 click 上
* 没有 click ⇒ 没有样本承载这笔钱 ⇒ **无法监督学习**
* 强行“造一条 click”会引入虚假特征与虚假训练信号（得不偿失）

**但要做两件事：**

1. **监控**：把这部分金额单独报出来（否则团队会误以为“label 少算了”）
2. **可用作历史特征**：它发生在过去天（day < 当前天）时，完全可以计入 pair/user/streamer 的历史礼物统计（这不矛盾）

> 你也可以把它解释成“用户通过其他入口进入直播间但该入口未被记为 click”，这在 KuaiLive 里是合理假设；论文也描述了多入口（双列 live tab、短视频 feed 点击头像进房等）并记录交互。 ([arXiv][1])
> 但无论真实原因是什么：**现有 click-level pipeline 没法归因，就应当显式记为 orphan。**

### 2.2 类别 B：有 click，但在窗口外（主要是 1min~60min 那段 + 极少数 >60min）

**最终处理：默认不归因（window 外就是 window 外），同样落在 “orphan_gifts_outside_window” 里单独记录。**

你已经算清楚了：这部分金额规模很小（相对 1min 额外只有 2.4% 金额，>60min 甚至只有 ~0.2%）。
如果你未来发现它对某些关键主播/关键用户段很重要，再考虑开一个“长窗 label”的副目标（但不建议把它混进主目标里）。

---

## 3) 特征设计（Day-Frozen）最终口径

你写的 Day-Frozen 核心思想是正确的，我建议保留：

* 历史只用 `day < click_day`（严格小于当天）
* 特征三层：

  * Pair (user, streamer)：`pair_gift_cnt/sum/mean_hist`
  * User：`user_gift_cnt/sum/mean_hist`
  * Streamer：`str_gift_cnt/sum/mean_hist`

**这 9 个历史特征 + 基础静态信息 + 时间特征**，就足够跑出一个稳定 baseline，并且不依赖 watch_time 这类高泄漏风险特征。

---

## 4) 数据划分（最终建议）

* 采用 **7-7-7** 按天切分（Train/Val/Test 各 7 天）
* KuaiLive 本身就是 21 天（May 5–May 25, 2025），7-7-7 非常自然。 ([arXiv][1])
* gap_days 默认 0（你现在实现 OK）

---

# 对 `data_utils.py` 的审计结论（你问“同时检查一下其他 data 处理是否合理”）

下面是我直接看你上传的 `data_utils.py`（你当前版本已经实现 last-touch + 1min）后的审计结果。

## A. ✅ 做得好的点（建议保留）

1. **反向归因（gift→click）再聚合**：这是修复 double counting 的正确姿势
2. **Last-touch dedup**：逻辑正确
3. **Day-Frozen 历史特征 + merge_asof(allow_exact_matches=False)**：严格 day < 当前天，属于强无泄漏口径
4. **类别编码 train-fit / val-test reuse**：流程正确
5. **特征列黑名单 + verify_no_leakage**：方向正确（虽然后面我建议再加强一点点）

## B. 🔧 必改项 1：Last-touch dedup 应该按 gift_id，而不是按 gift_ts

你现在是：

```python
merged.groupby(['user_id','streamer_id','live_id','timestamp'])['click_ts'].idxmax()
```

这里的 `timestamp` 是 gift_ts。**如果同一毫秒存在多条 gift（或重复记录）**，这会把多条 gift 当成一条 gift 处理，造成**漏算 label**。

**建议改成（最稳）：**

* 在 merge 前加一列 `gift_id = gift.index`（或 reset_index）
* dedup 用 `groupby(gift_id)`

这属于“不会改变你的核心结论，但能把 pipeline 做扎实”的硬修。

## C. 🔧 必改项 2：现在的“总金额守恒”日志会误导

你在 label 函数里只对 `ratio > 1.01` 报警，否则就打印“总金额守恒 ✅”。

但你现在已经确认：**4.8% gifts 没 click + 一部分超窗**，所以 `sum(label) < sum(gift)` 是正常现象，不叫“守恒”。

**建议把日志改成两类指标：**

* `attributed_value / total_gift_value`（金额覆盖率）
* `attributed_count / total_gift_count`（数量覆盖率）
  并把 orphan 的 breakdown 打印出来（no_click / outside_window / negative_delta 等）。

这能显著降低后续 PR review 的沟通成本。

## D. ⚠️ 需要你明确立场的点：user/streamer “静态特征”可能有时间泄漏

你目前在 `add_static_features()` 里用了很多类似 `fans_num/follow_num/accu_*` 的累计特征，并把它们当成“无泄漏风险静态特征”。

但 KuaiLive 论文写得很清楚：**对 time-sensitive features（例如 fans_num）采用了“截至 2025-05-25 的统一快照”**。 ([arXiv][1])
这意味着：在你做 Train/Val/Test 时间切分时，训练集样本也能“看到未来几天增长后的粉丝数”等信息——严格意义上就是 leakage。

**最终建议（给你一个可落地的二选一）：**

* **模式 Strict（推荐作为 Gift EVpred 主线，真正无泄漏）**：
  静态特征只保留“本质不会随时间变化或可视为已知”的字段，例如：`age/gender/country/device_*/reg_timestamp/直播间类目/时间特征(hour,dow)`；
  把 `fans/follow/accu_*/play_cnt/duration` 这类快照累计全部 drop。

* **模式 Benchmark（如果你要对齐 KuaiLive 论文 benchmark）**：
  保留这些快照特征，但你要在 Card/PR 写清楚：
  “side info 是 snapshot as of May 25, 2025，因此不是严格在线可用特征”。

> 你这个项目本身一直强调 leakage-free，我会建议默认走 **Strict**，并把 Benchmark 当成可选实验开关。

## E. ✅ watch_live_time 禁用是对的，但还可以加一道“缓存后清理”

你在 label 构造时 drop `watch_live_time` 很好。
但当你从 parquet cache 直接加载时，理论上也要做一次 **drop forbidden**（防止缓存版本不一致把泄漏列带回来）。

---

# 你可以直接复制进 Card/PR 的“最终方案总结”（整理版）

**Gift EVpred 数据处理最终口径**

1. **Label（Click-level EV）**

* Attribution model：Last-touch (Last-click)
* Lookback window：**1 minute（默认，可调 5/10/60）**
* Dedup：每个 gift 只归因一次（按 `gift_id` 去重）
* Aggregation：归因后的 gift_price sum 到 click-level，得到 `gift_price_label`

2. **Orphan gifts（未归因）处理**

* A: 无 click 的 gift（约 4.8% / 5.7% 金额）→ 记为 `orphan_gifts_no_click`，不回填到 click label；用于监控 + 可用于历史特征（day-frozen）
* B: 超出窗口的 gift（主要 1–60min 少量、>60min 极少）→ `orphan_gifts_outside_window`，默认不回填；如需覆盖可开长窗实验

3. **Features（Day-Frozen，无泄漏）**

* 只用 `day < click_day` 的历史 gift 构造：pair/user/streamer 的 cnt/sum/mean（9 个）
* 禁用 watch_time 类结果泄漏特征：`watch_live_time` 等

4. **Static features**

* 推荐 Strict 模式：只保留真正静态/在线可得字段
* 若使用粉丝数/累计量等快照字段，需在文档注明 KuaiLive 是 May 25 快照（存在时间泄漏风险） ([arXiv][1])

5. **Split**

* 按天 7-7-7（KuaiLive 21 天：May 5–25, 2025） ([arXiv][1])

6. **Sanity checks**

* 每个 gift_id 只归因一次（无 double counting）
* 打印金额覆盖率/数量覆盖率 + orphan breakdown（不再用“守恒”误导）
* train/val/test 时间严格不重叠

---

如果你愿意，我可以基于你这份 `data_utils.py` 直接把**“必改项 1/2/静态特征 Strict 模式开关”**整理成一段最小 diff（就改 30 行左右），保证你一 merge 就能跑全链路。

[1]: https://arxiv.org/pdf/2508.05633 "KuaiLive: A Real-time Interactive Dataset for Live Streaming Recommendation"
