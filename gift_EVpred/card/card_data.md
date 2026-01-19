# 🧠 Card Data｜EVpred 无泄漏数据处理流程

> **结论（可指导决策）**
> 必须使用 Day-Frozen 特征构建 + 7-7-7 按天划分，才能避免时间穿越泄漏；禁止使用 watch_live_time。

---

## 1️⃣ 数据源

### 原始数据文件

| 文件 | 描述 | 关键字段 |
|------|------|---------|
| `gift.csv` | 打赏记录 | user_id, streamer_id, live_id, timestamp, gift_price |
| `click.csv` | 点击/进入直播间记录 | user_id, streamer_id, live_id, timestamp, ~~watch_live_time~~ |
| `user.csv` | 用户画像 | user_id, age, gender, device_brand, fans_num, ... |
| `streamer.csv` | 主播画像 | streamer_id, fans_user_num, accu_live_cnt, ... |
| `room.csv` | 直播间信息 | live_id, live_type, live_content_category |

### 数据时间范围

```
KuaiLive 数据: 2025-05-04 ~ 2025-05-25 (共 22 天)
样本量: click ≈ 1.2M, gift ≈ 50K
```

---

## 2️⃣ 标签构建 (Click-Level)

### 核心逻辑

```
对每个 click 事件，Label = 该 click 后窗口内的 gift 总额
  - Label = 0: 窗口内无打赏
  - Label > 0: 窗口内有打赏，值为 gift_price 总和
```

### ⚠️ 窗口问题（重要）

**当前实现的问题**：固定 1 小时窗口，没有考虑直播结束时间

```
问题场景：
- 平均观看时间只有 ~4 秒
- 很多直播在 1 小时前就结束了
- 如果 click 发生在直播结束前 5 分钟，用户只有 5 分钟打赏机会
- 但窗口设为 1 小时 → label 定义不一致
```

**正确做法**：窗口上限 = min(固定窗口, 直播结束时间)

```python
# room.csv 有 start_timestamp 和 end_timestamp
room_times = room[['live_id', 'end_timestamp']]

# 窗口 = min(click + 1h, live_end)
click = click.merge(room_times, on='live_id', how='left')
click['label_end_dt'] = click[['timestamp_dt + 1h', 'end_timestamp']].min(axis=1)
```

**或者**：使用更短的固定窗口（如 5-10 分钟），减少直播结束的影响

### 已有实验结果（exp_estimation_layer_audit_20260118）

| 对比项 | 差异 | 说明 |
|--------|------|------|
| 整体 | **16.51%** | 固定 1h vs watch_time 截断 |
| <5s 分桶 | **65.57%** | 用户看了 <5s 就走，但 1h 内的礼物都算进去 |
| 5-30s 分桶 | **68.32%** | 同上 |
| >300s 分桶 | 9.50% | 长观看差异较小 |

### 深入分析：gift 发生时间 vs watch_time（exp_label_window_compare_20260119）

| 统计项 | 值 | 说明 |
|--------|-----|------|
| 在 watch_time 内的 gift | **98.77%** | 绝大部分 gift 在用户观看期间发生 |
| 在 1h 内的 gift | 99.71% | - |
| 伪标签（1h内但watch_time外）| **0.94%** | 只有极少数 |
| 伪标签金额占比 | **1.93%** | 影响很小 |

**"用户走了但礼物算给他" 会不会发生？**

物理上：**不会**。用户必须在直播间内才能打赏，离开后不能打赏。

数据上：**1.23% 的 gift 不在最近一次 click 的 watch_time 内**
- 42% 来自多次进入（用户第一次走了，后来又进来打赏，第一条 click 匹配到后来的 gift）
- 58% 是数据异常（单次进入但 gift 在 watch_time 后，可能是 watch_time 统计不准或 click 记录丢失）

### ✅ 已修复：gift 被多条 click 重复匹配 (Over-Attribution)

| 统计 | 修复前 | 修复后 | 说明 |
|------|--------|--------|------|
| 被多条 click 匹配的 gift | **8.43%** | 0% | 每个 gift 只归因 1 条 click |
| Label/Gift 金额比 | **116.78%** | **92.37%** | 不再膨胀 |

**问题场景（已修复）**：
```
click_1: 10:00 进入，1h 窗口 = 10:00~11:00
click_2: 10:30 进入，1h 窗口 = 10:30~11:30
gift:    10:35 打赏 100 元

修复前：
  click_1.label = 100 (gift 在窗口内)
  click_2.label = 100 (gift 也在窗口内)
  → 同一笔 100 元被算了 2 次！

修复后 (Last-Touch Attribution)：
  click_1.label = 0 (不是最近的 click)
  click_2.label = 100 (✅ 最近的 click 获得归因)
  → 只算 1 次
```

**修复方案**：**Last-Touch Attribution**（行业标准）

每个 gift 只归因给**最近的一条 click**（click_ts 最大）：

```python
# 从 gift 角度出发（先归因，再聚合）
merged = gift.merge(
    click[['user_id', 'streamer_id', 'live_id', 'timestamp']].rename(
        columns={'timestamp': 'click_ts'}
    ),
    on=['user_id', 'streamer_id', 'live_id'],
    how='inner'
)

# 筛选 gift 在 click 的 1h 窗口内
merged = merged[
    (merged['timestamp'] >= merged['click_ts']) &
    (merged['timestamp'] <= merged['click_ts'] + window_ms)
]

# Last-Touch: 每个 gift 只保留最近的 click
merged = merged.loc[
    merged.groupby(['user_id', 'streamer_id', 'live_id', 'timestamp'])['click_ts'].idxmax()
]

# 聚合到 click 级别
gift_agg = merged.groupby(
    ['user_id', 'streamer_id', 'live_id', 'click_ts']
)['gift_price'].sum()
```

**验证护栏**：
- 总金额守恒：`label_sum <= gift_sum`（ratio ≤ 1.01）
- 一对一约束：每个 gift 只归因 1 条 click

**状态**：✅ 已修复 (2026-01-19)

### Ridge 窗口对比实验

| 配置 | RevCap@1% | Revenue | vs 1h_fixed |
|------|-----------|---------|-------------|
| 1h_fixed | 22.78% | 2.09M | 基准 |
| 1h_live_end | 22.78% | 2.09M | **完全相同** |
| 30min_fixed | 21.90% | 1.94M | -0.89pp |
| 10min_fixed | 20.93% | 1.81M | -1.85pp |

**结论**：
1. **伪标签问题很小**（<2%），固定 1h 窗口可以接受
2. `live_end` 截断无效（KuaiLive 直播时长 >= 1h）
3. 更短窗口会损失正样本和 RevCap
4. 之前 16.51% 差异可能是计算方式不同，不是伪标签造成

**建议**：使用固定 1h 窗口即可，伪标签影响可忽略

### 当前实现（Last-Touch Attribution）

```python
# 使用 data_utils.prepare_click_level_labels()
# 核心流程:
# 1. 从 gift 角度 merge click（反向思路）
# 2. 筛选 gift 在 click 的 1h 窗口内
# 3. Last-Touch: 每个 gift 只保留最近的 click
# 4. 聚合到 click 级别
# 5. 验证护栏: 总金额守恒

from gift_EVpred.data_utils import prepare_click_level_labels
click_with_labels = prepare_click_level_labels(gift, click, label_window_hours=1)

# 输出:
#   Gift-Click pairs before dedup: 74,863
#   Gift-Click pairs after dedup: 68,129 (每个 gift 只归因 1 条 click)
#   总金额守恒: label=5,547,977, gift=6,006,113, ratio=0.9237 ✅
```

### 关键统计

```
Gift Rate: ~4% (正样本占比)
Label 分布: 高度右偏 (大部分为 0，少数高额打赏)
→ 使用 log1p 变换: target = log(1 + gift_price_label)
```

---

## 3️⃣ 数据划分 (7-7-7)

### 划分原则

| 原则 | 说明 |
|------|------|
| **按天划分** | 按自然日划分，而非按样本比例 |
| **时间顺序** | Train < Val < Test，严格按时间顺序 |
| **无重叠** | 三个集合的时间范围完全不重叠 |

### 具体划分

```
Train: Day 1-7  (2025-05-04 ~ 2025-05-10) → 7 天
Val:   Day 8-14 (2025-05-11 ~ 2025-05-17) → 7 天
Test:  Day 15-21 (2025-05-18 ~ 2025-05-24) → 7 天
剩余:  Day 22 不使用
```

### 代码实现

```python
from gift_EVpred.data_utils import split_by_days

train_df, val_df, test_df = split_by_days(
    df,
    train_days=7,
    val_days=7,
    test_days=7,
    gap_days=0  # 可选：Train-Val 间隔天数
)
```

---

## 4️⃣ Day-Frozen 特征构建（核心防泄漏机制）

### 核心设计

```
对每个 click 的历史特征，只允许用 day < 当前 day 的历史数据
  - 不会用到"未来"数据
  - 保守但安全：丢掉"同一天更早发生的历史"
  - 训练/验证/测试都用同一套逻辑
```

### 实现方式

使用 `pd.merge_asof` + `allow_exact_matches=False` 实现严格 < 当前天:

```python
# 1. 按天聚合 gift 历史
pair_day = gift.groupby(['day', 'user_id', 'streamer_id'])['gift_price'].agg(
    gift_cnt_day='count',
    gift_sum_day='sum'
)

# 2. 累计统计
pair_day['pair_gift_cnt_hist'] = pair_day.groupby(
    ['user_id', 'streamer_id']
)['gift_cnt_day'].cumsum()

# 3. merge_asof: 查找 strictly before 的最近记录
click_with_pair = pd.merge_asof(
    click_sorted,
    pair_day,
    on='day',
    by=['user_id', 'streamer_id'],
    direction='backward',
    allow_exact_matches=False  # 严格 < 当前天
)
```

### 构建的历史特征 (9 个)

| 层级 | 特征名 | 含义 |
|------|--------|------|
| **Pair-level** | `pair_gift_cnt_hist` | (user, streamer) 对的历史打赏次数 |
| | `pair_gift_sum_hist` | (user, streamer) 对的历史打赏总额 |
| | `pair_gift_mean_hist` | (user, streamer) 对的历史打赏均值 |
| **User-level** | `user_gift_cnt_hist` | 用户的历史打赏次数 |
| | `user_gift_sum_hist` | 用户的历史打赏总额 |
| | `user_gift_mean_hist` | 用户的历史打赏均值 |
| **Streamer-level** | `str_gift_cnt_hist` | 主播的历史收礼次数 |
| | `str_gift_sum_hist` | 主播的历史收礼总额 |
| | `str_gift_mean_hist` | 主播的历史收礼均值 |

---

## 5️⃣ 静态特征（无泄漏风险）

### 用户画像特征

```python
['age', 'gender', 'device_brand', 'device_price',
 'fans_num', 'follow_num',
 'accu_watch_live_cnt', 'accu_watch_live_duration',
 'is_live_streamer', 'is_photo_author']
```

### 主播画像特征

```python
['str_fans_user_num', 'str_fans_group_fans_num',
 'str_follow_user_num', 'str_accu_live_cnt',
 'str_accu_live_duration', 'str_accu_play_cnt',
 'str_accu_play_duration']
```

### 直播间特征

```python
['live_type', 'live_content_category']
```

### 时间特征

```python
['hour', 'day_of_week', 'is_weekend']
```

---

## 6️⃣ 特征使用规范

### 禁止直接使用（当前 session 值）

| 特征 | 泄漏类型 | 原因 | 替代方案 |
|------|----------|------|----------|
| `watch_live_time` | 🔴 结果泄漏 | 包含打赏后的观看时长 | ✅ `user_watch_hist` / `pair_watch_hist` |
| `watch_time_log` | 🔴 结果泄漏 | 同上 | 同上 |
| `pair_gift_mean` | 🔴 未来泄漏 | groupby 包含未来样本 | ✅ `pair_gift_mean_hist` |
| `user_total_gift_7d` | 🔴 未来泄漏 | 同上 | ✅ `user_gift_sum_hist` |

### 关键区分

```
❌ 当前 session 的 watch_live_time = 结果泄漏（包含打赏后时间）
✅ 历史观看时长 (day < 当前 day) = 有效特征（用户过去行为）

原理：
- 当前 session: 用户看了 5 分钟后打赏 → watch_time 包含打赏后时间 → 泄漏
- 历史 session: 用户昨天看了这个主播 30 分钟 → 可预测今天的打赏倾向 → 有效
```

### 可构建的历史观看特征（待实现）

| 特征名 | 含义 | 状态 |
|--------|------|------|
| `user_watch_hist` | 用户历史总观看时长 | 🟡 待实现 |
| `pair_watch_hist` | 用户对该主播的历史观看时长 | 🟡 待实现 |
| `user_watch_cnt_hist` | 用户历史观看次数 | 🟡 待实现 |
| `pair_watch_cnt_hist` | 用户对该主播的历史观看次数 | 🟡 待实现 |

### 历史观看特征的正确实现（重要）

```python
# ⚠️ 关键：必须用 Day-Frozen，不能用全量 groupby

# ❌ 错误：对 train 内样本，会把"未来 session 的 watch_time"灌给过去 click
click_train = click[click['day'] <= train_end]
user_watch_stats = click_train.groupby('user_id')['watch_live_time'].sum()  # 泄漏！

# ✅ 正确：Day-Frozen，只用 day < 当前 day 的历史
click['day'] = pd.to_datetime(click['timestamp'], unit='ms').dt.normalize()

# 按天聚合
user_day = click.groupby(['day', 'user_id'])['watch_live_time'].agg(
    watch_sum_day='sum',
    watch_cnt_day='count'
)

# 累计 + merge_asof (strictly before)
user_day['user_watch_hist'] = user_day.groupby('user_id')['watch_sum_day'].cumsum()

click_with_watch = pd.merge_asof(
    click.sort_values('day'),
    user_day[['day', 'user_id', 'user_watch_hist']],
    on='day',
    by='user_id',
    direction='backward',
    allow_exact_matches=False  # 严格 < 当前天
)
```

### 口径说明

| 场景 | 使用的数据 | 说明 |
|------|-----------|------|
| **Train 内样本** | day < 当前 day 的 click | Day-Frozen，避免未来泄漏 |
| **Val/Test 样本** | 同上逻辑 | 口径一致 |
| **线上推理** | 历史 session 的观看记录 | 是"历史先验"，不是"当前 session 即时停留" |

**注意**：当前 `data_utils.py` 直接删除了 `watch_live_time`，未实现历史特征。如需添加，要先保留原始列再构建。

---

## 7️⃣ 类别编码规范

### 问题：独立编码导致口径不一致

```python
# ❌ 错误：每个 split 独立编码
train_df['gender'] = pd.Categorical(train_df['gender']).codes  # male=0, female=1
val_df['gender'] = pd.Categorical(val_df['gender']).codes      # 可能 female=0, male=1

# 树模型认为 code=0 是同一类别，但 train 是 male，val 是 female
# → 评估不公平（不是泄漏，但污染评估）
```

### 正确做法：划分前在整个数据集上编码

```python
# ✅ 正确：划分前统一编码（不是泄漏，只是映射关系）
click['gender'] = pd.Categorical(click['gender']).codes  # 全局统一映射

# 然后再划分
train_df, val_df, test_df = split_by_days(click, ...)
```

**为什么不是泄漏？**
- 类别编码只是 `{male: 0, female: 1}` 的映射关系
- 不涉及标签信息（gift_price）
- 类似于 feature name → feature index，是元信息

### 当前实现

`data_utils.py` 当前在划分后用 Train 拟合 categories：
```python
train_categories = list(train_df[col].unique())
# Val/Test 未见过的值 → 'unknown'
df[col] = pd.Categorical(df[col], categories=train_categories).codes
```

**优化建议**：可以改为划分前统一编码，代码更简洁，避免 unknown 处理。

---

## 8️⃣ 完整使用流程

### 标准用法

```python
from gift_EVpred.data_utils import (
    prepare_dataset,
    get_feature_columns,
    verify_no_leakage
)

# 1. 准备数据
train_df, val_df, test_df = prepare_dataset(
    train_days=7,
    val_days=7,
    test_days=7
)

# 2. 获取特征列
feature_cols = get_feature_columns(train_df)

# 3. 验证无泄漏
gift, _, _, _, _ = load_raw_data()
verify_no_leakage(train_df, gift, n_samples=100)
```

### 输出说明

```python
train_df.columns 包含:
  - 原始 ID: user_id, streamer_id, live_id, timestamp
  - 标签: gift_price_label, target (log1p), is_gift (binary)
  - 历史特征: pair_gift_*_hist, user_gift_*_hist, str_gift_*_hist
  - 静态特征: age, gender, hour, ...
```

---

## 9️⃣ 验证清单

每个 EVpred 实验必须通过以下验证:

- [ ] 使用 `prepare_dataset()` 加载数据
- [ ] 使用 `get_feature_columns()` 获取特征
- [ ] 运行 `verify_no_leakage()` 验证通过
- [ ] 特征列不包含 `watch_live_time`（当前 session）
- [ ] 时间划分满足 `train_max < val_min < test_min`
- [ ] 类别编码口径一致（同一类别在 Train/Val/Test 中 code 相同）

---

## 🔟 实验链接

| 来源 | 路径 |
|------|------|
| 数据处理代码 | `gift_EVpred/data_utils.py` |
| 数据处理指南 | `gift_EVpred/DATA_PROCESSING_GUIDE.md` |
| Prompt 模板 | `gift_EVpred/prompts/prompt_template_evpred.md` |

---

<!--
Card 作者: Viska Wei
创建日期: 2026-01-18
更新日期: 2026-01-19
版本: 2.0 (Last-Touch Attribution)
更新内容:
  - 修复 Over-Attribution bug: 每个 gift 只归因给最近的 click
  - Label/Gift ratio 从 116.78% 降至 92.37%
  - 添加验证护栏：总金额守恒检查
-->
