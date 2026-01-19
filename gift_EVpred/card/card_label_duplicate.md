# 🧠 Card Label-Duplicate｜重叠归因窗口导致 Double Counting

> **结论（可指导决策）**
> ~~当前标签构建存在典型的 **Over-Attribution Bug**~~ → ✅ **已修复 (2026-01-19)**
> 使用 **Last-Touch Attribution** 修复后，Label/Gift ratio 从 116.78% 降至 92.37%。

---

## 0️⃣ 业界背景

这个问题在业界非常典型：**"重叠归因窗口导致同一笔转化被多次计入（Double Counting / Over-Attribution）"**。

主流做法不是"对每条 click 去找转化"，而是反过来——**先在转化（gift）层面做归因，确保每个转化只被归到 1 个触点，再汇总到 click 级 label**。

### 业界标准归因模型

| 归因模型 | 描述 | 应用场景 |
|----------|------|----------|
| **Last-Touch** | 100% credit 给最后一次触点 | Google Ads/LinkedIn 默认，最常用 baseline |
| First-Touch | 100% credit 给第一次触点 | 品牌曝光评估 |
| Multi-Touch (MTA) | 按规则或算法分摊 credit | 渠道贡献分析、预算分配 |
| Data-Driven | 算法学习权重分配 | GA4、Google Ads 高级功能 |

**我们的场景**：训练 click-level EV 模型 → **Last-Touch 最合理**

### 推荐的业界标准表述

```
Attribution Model: Last-touch (Last-click) within 1h lookback
Dedup Rule: 每个 gift 只能归因给 1 条 click（最近的一条）
Aggregation: 再把 gift 金额 sum 到 click-level label
```

---

## 1️⃣ 问题描述

### 背景

在直播打赏场景中，用户可能**多次进入同一个直播间**（产生多条 click 记录）。当前标签构建逻辑是：

```python
# 对每条 click，找 1h 窗口内同一 (user, streamer, live_id) 的 gift
click['label_end'] = click['timestamp'] + 1h
merged = click.merge(gift, on=['user_id', 'streamer_id', 'live_id'])
merged = merged[
    (gift_ts >= click_ts) &
    (gift_ts <= click_ts + 1h)
]
label = merged.groupby(click_keys)['gift_price'].sum()
```

### 问题

如果用户多次进入，**同一笔 gift 会被多条 click 都匹配到**：

```
时间线：
├─ 10:00  click_1 进入（1h窗口 = 10:00~11:00）
├─ 10:30  click_2 再次进入（1h窗口 = 10:30~11:30）
└─ 10:35  gift 打赏 100 元

结果：
  click_1.label = 100  ← gift 在 click_1 的 1h 窗口内
  click_2.label = 100  ← gift 也在 click_2 的 1h 窗口内

问题：同一笔 100 元被算了 2 次！
```

### 物理意义上的问题

- 用户只打赏了 **1 次**，金额 **100 元**
- 但被 **2 条 click** 都计入了 label
- 这 100 元的"功劳"被错误地分给了 2 条 click
- 从归因角度看，只有 click_2 才是真正触发打赏的 click

---

## 2️⃣ 数据验证

### 统计结果

| 指标 | 值 | 说明 |
|------|-----|------|
| 原始 gift 数 | 72,646 | - |
| (gift, click) 匹配对数 | 74,863 | 多于 gift 数，说明有重复 |
| 被多条 click 匹配的 gift | **8.43%** (5,740) | 这些 gift 被重复计算 |
| 重复计算的金额 | **16.78%** (1,007,739) | 总金额被高估 |

### 匹配分布

```
每个 gift 被匹配的 click 数:
  1 次: 62,389 (91.57%)  ← 正常
  2 次:  4,911 (7.21%)   ← 重复
  3 次:    693 (1.02%)   ← 重复
  4 次:    112 (0.16%)   ← 重复
  5+次:     24 (0.04%)   ← 重复
```

### 具体案例

**案例 1**：User 14 -> Streamer 120082 (live 2183717)
```
Gift: 2025-05-18 22:07:31, 金额=38元
被 2 条 click 匹配:
  - Click 2025-05-18 21:42:58: gift 在 click 后 1473s (24分钟)
  - Click 2025-05-18 22:07:31: gift 在 click 后 0s (同时)

→ 这 38 元被算了 2 次
→ click_1 不应该获得这笔 label（用户当时已离开）
```

**案例 2**：User 14 -> Streamer 120082 (live 8785539)
```
Gift: 2025-05-07 13:55:48, 金额=1元
被 3 条 click 匹配:
  - Click 2025-05-07 13:22:36: gift 在 click 后 1991s (33分钟)
  - Click 2025-05-07 13:54:03: gift 在 click 后 105s (1.7分钟)
  - Click 2025-05-07 13:55:48: gift 在 click 后 0s (同时)

→ 这 1 元被算了 3 次
```

---

## 3️⃣ 影响分析

### 对标签的影响

| 影响项 | 程度 | 说明 |
|--------|------|------|
| 总 revenue | 高估 16.78% | 同一金额被重复计入 |
| 正样本数 | 略高估 | 多条 click 都被标记为正样本 |
| 标签均值 | 高估 | 特别是多次进入的用户 |

### 对模型的影响

| 影响项 | 程度 | 说明 |
|--------|------|------|
| 归因错误 | 🔴 严重 | click_1 不应获得 click_2 时打赏的"功劳" |
| 特征-标签关系 | 🔴 严重 | click_1 的特征无法解释为什么会有 gift（用户当时已走） |
| RevCap 计算 | 🟡 中等 | 总 revenue 被高估，RevCap 分母错误 |
| 模型学习 | 🟡 中等 | 模型可能学到虚假的模式 |

### 与之前分析的关系

**之前 `exp_estimation_layer_audit` 的 16.51% 差异**：
- 原以为是"固定窗口 vs watch_time 截断"的差异
- 实际是 **gift 重复匹配**造成的
- watch_time 截断"意外地"避免了重复匹配问题

---

## 4️⃣ 修复方案

### 方案对比

| 方案 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **A: Last-Touch** | 每个 gift 归因给最近的 click | 简单、业界默认 | 需要 gift×click 大 join |
| **B: Session Boundary** | 用 next_click 截断窗口 | 结构上消灭重叠 | 稍复杂，但更优雅 |

### 方案 A: Last-Touch（业界默认 baseline）

**思路**：对每个 gift，找满足条件的 click（同 user/streamer/live，且 click_ts ≤ gift_ts ≤ click_ts + 1h），选 **click_ts 最大**的那条。

```python
def prepare_labels_last_touch(gift, click, window_hours=1):
    """Last-Touch Attribution: 每个 gift 只归因给最近的 click"""

    # Step 1: Merge gift 和 click
    merged = gift.merge(
        click[['user_id', 'streamer_id', 'live_id', 'timestamp']].rename(
            columns={'timestamp': 'click_ts'}
        ),
        on=['user_id', 'streamer_id', 'live_id']
    )

    # Step 2: 筛选 gift 在 click 的窗口内
    merged = merged[
        (merged['timestamp'] >= merged['click_ts']) &
        (merged['timestamp'] <= merged['click_ts'] + window_hours * 3600000)
    ]

    # Step 3: 每个 gift 只保留最近的 click（Last-Touch）
    merged = merged.loc[
        merged.groupby(['user_id', 'streamer_id', 'live_id', 'timestamp'])['click_ts'].idxmax()
    ]

    # Step 4: 聚合到 click 级别
    labels = merged.groupby(
        ['user_id', 'streamer_id', 'live_id', 'click_ts']
    )['gift_price'].sum().reset_index().rename(columns={
        'click_ts': 'timestamp', 'gift_price': 'gift_price_label'
    })

    # Step 5: Merge 回 click
    click = click.merge(labels, on=['user_id', 'streamer_id', 'live_id', 'timestamp'], how='left')
    click['gift_price_label'] = click['gift_price_label'].fillna(0)

    return click
```

### 方案 B: Session Boundary（推荐，更优雅）

**思路**：把 click 看成 session start，用**下一次 click 作为 session 结束边界**，从结构上消灭窗口重叠。

```
定义：
  session_start = click_ts
  session_end = min(click_ts + 1h, next_click_ts)

效果：
  ├─ 10:00  click_1 → session_1: [10:00, 10:30)  ← 被 next_click 截断
  ├─ 10:30  click_2 → session_2: [10:30, 11:30]
  └─ 10:35  gift → 只落在 session_2 中（无重叠！）
```

```python
def prepare_labels_session_boundary(gift, click, window_hours=1):
    """Session Boundary: 用 next_click 截断窗口，结构上消灭重叠"""

    click = click.sort_values(['user_id', 'streamer_id', 'live_id', 'timestamp'])

    # Step 1: 计算每条 click 的 next_click_ts
    click['next_click_ts'] = click.groupby(
        ['user_id', 'streamer_id', 'live_id']
    )['timestamp'].shift(-1)

    # Step 2: session_end = min(click_ts + 1h, next_click_ts)
    click['session_end'] = click[['timestamp', 'next_click_ts']].apply(
        lambda x: min(
            x['timestamp'] + window_hours * 3600000,
            x['next_click_ts'] if pd.notna(x['next_click_ts']) else float('inf')
        ), axis=1
    )

    # Step 3: 每个 gift 找落在哪个 session（用 merge_asof 或条件 join）
    # 简化版：用 pd.merge_asof 找最近的 click（backward）
    gift_sorted = gift.sort_values('timestamp')
    click_sorted = click.sort_values('timestamp')

    attributed = pd.merge_asof(
        gift_sorted,
        click_sorted[['user_id', 'streamer_id', 'live_id', 'timestamp', 'session_end']].rename(
            columns={'timestamp': 'click_ts'}
        ),
        left_on='timestamp',
        right_on='click_ts',
        by=['user_id', 'streamer_id', 'live_id'],
        direction='backward'
    )

    # Step 4: 只保留 gift 在 session 内的（gift_ts < session_end）
    attributed = attributed[attributed['timestamp'] <= attributed['session_end']]

    # Step 5: 聚合到 click 级别
    labels = attributed.groupby(
        ['user_id', 'streamer_id', 'live_id', 'click_ts']
    )['gift_price'].sum().reset_index().rename(columns={
        'click_ts': 'timestamp', 'gift_price': 'gift_price_label'
    })

    # Step 6: Merge 回原始 click
    click = click.drop(columns=['next_click_ts', 'session_end'])
    click = click.merge(labels, on=['user_id', 'streamer_id', 'live_id', 'timestamp'], how='left')
    click['gift_price_label'] = click['gift_price_label'].fillna(0)

    return click
```

### SQL 版本（工业级实现）

```sql
-- Last-Touch: ROW_NUMBER + QUALIFY
WITH gift_attributed AS (
    SELECT
        g.*,
        c.click_ts,
        ROW_NUMBER() OVER (
            PARTITION BY g.user_id, g.streamer_id, g.live_id, g.gift_ts
            ORDER BY c.click_ts DESC  -- Last-Touch: 取最近的 click
        ) AS rn
    FROM gift g
    JOIN click c
        ON g.user_id = c.user_id
        AND g.streamer_id = c.streamer_id
        AND g.live_id = c.live_id
        AND g.gift_ts >= c.click_ts
        AND g.gift_ts <= c.click_ts + INTERVAL '1 hour'
    QUALIFY rn = 1  -- 每个 gift 只保留一行
)
SELECT
    c.*,
    COALESCE(SUM(ga.gift_price), 0) AS gift_price_label
FROM click c
LEFT JOIN gift_attributed ga
    ON c.user_id = ga.user_id
    AND c.streamer_id = ga.streamer_id
    AND c.live_id = ga.live_id
    AND c.click_ts = ga.click_ts
GROUP BY c.*
```

---

## 5️⃣ 验证护栏（必须实现）

### 护栏 1: 总金额守恒

```python
total_label = click['gift_price_label'].sum()
total_gift = gift['gift_price'].sum()
assert total_label <= total_gift * 1.01, f"金额膨胀: {total_label} > {total_gift}"
print(f"总金额守恒: label={total_label:,.0f}, gift={total_gift:,.0f}, ratio={total_label/total_gift:.4f}")
```

### 护栏 2: 一对一约束

```python
# 在归因中间表检查：每个 gift 只出现一次
gift_count = attributed.groupby(['user_id', 'streamer_id', 'live_id', 'gift_ts']).size()
assert (gift_count == 1).all(), f"存在重复归因: {(gift_count > 1).sum()} gifts"
```

---

## 6️⃣ 其他归因方式（了解即可）

| 归因方式 | 描述 | 适用场景 |
|----------|------|----------|
| **Last-Touch** | 100% 给最近的 click | ✅ 训练标签（推荐） |
| First-Touch | 100% 给第一次 click | 品牌曝光评估 |
| Multi-Touch | 按规则分摊 | 渠道贡献分析 |
| Data-Driven | 算法学习权重 | 需要大量数据 |

**结论**：训练 EV 模型用 **Last-Touch**，简单且符合因果。

---

## 7️⃣ 实验链接

| 来源 | 路径 |
|------|------|
| 数据处理代码 | `gift_EVpred/data_utils.py` |
| 数据处理卡片 | `gift_EVpred/card/card_data.md` |
| 之前的审计实验 | `gift_EVpred/exp/archive_leaky/exp_estimation_layer_audit_20260118.md` |
| 窗口对比实验 | `gift_EVpred/scripts/exp_label_window_compare.py` |

---

## 8️⃣ 修复记录 (2026-01-19)

- [x] 修复 `data_utils.py` 中的 `prepare_click_level_labels` 函数 ✅
- [x] 删除旧的特征缓存 (`features_cache/*.parquet`) ✅
- [x] 实现验证护栏（总金额守恒、一对一约束）✅
- [x] 更新 `card_data.md` 中的标签构建说明 ✅
- [ ] 重跑 baseline，对比修复前后的 RevCap（待后续验证）

### 修复验证结果

```
=== 修复验证 (2026-01-19) ===
Gift-Click pairs before dedup: 74,863
Gift-Click pairs after dedup: 68,129 (每个 gift 只归因 1 条 click) ✅

原始 Gift 总金额: 6,006,113
Label 总金额: 5,547,977
Ratio: 0.9237 (≤ 1.01) ✅

多次匹配的 Gift: 5,740 (8.43%)
Last-Touch 后: 每个 Gift 只归因 1 条 Click ✅
```

---

## 9️⃣ 参考资料

| 来源 | 链接 |
|------|------|
| Google Ads Attribution | https://support.google.com/google-ads/answer/6259715 |
| LinkedIn Conversion Attribution | https://www.linkedin.com/help/linkedin/answer/a426349 |
| GA4 Attribution | https://support.google.com/analytics/answer/10596866 |
| Nielsen MTA Guide | https://www.nielsen.com/insights/2019/methods-models-a-guide-to-multi-touch-attribution/ |
| Snowflake ROW_NUMBER | https://docs.snowflake.com/en/sql-reference/functions/row_number |
| Snowflake ASOF JOIN | https://docs.snowflake.com/en/sql-reference/constructs/asof-join |

---

<!--
Card 作者: Viska Wei
创建日期: 2026-01-19
更新日期: 2026-01-19
版本: 3.0 (Last-Touch Attribution 已实现)
状态: ✅ 已修复
修复方案: Last-Touch Attribution (方案A)
验证结果: Label/Gift ratio = 0.9237 (无膨胀)
-->
