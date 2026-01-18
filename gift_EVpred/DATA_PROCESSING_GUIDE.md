# 数据处理指南：无泄漏特征工程与 7-7-7 数据划分

> **适用范围**: 所有 gift_EVpred 相关实验
> **创建日期**: 2026-01-18
> **状态**: 强制执行

---

## 目录

1. [数据泄漏问题汇总](#1-数据泄漏问题汇总)
2. [7-7-7 数据划分规范](#2-7-7-7-数据划分规范)
3. [无泄漏特征构建规范](#3-无泄漏特征构建规范)
4. [统一数据处理代码](#4-统一数据处理代码)
5. [验证清单](#5-验证清单)

---

## 1. 数据泄漏问题汇总

### 1.1 已确认的泄漏问题

| 问题编号 | 泄漏特征 | 泄漏类型 | 严重性 | 证据 |
|---------|---------|---------|--------|------|
| **P1** | `watch_live_time` | 结果泄漏 | 🔴 致命 | 包含打赏后的观看时长 |
| **P2** | `pair_gift_mean/sum/count` | 未来泄漏 | 🔴 致命 | Feature Importance = 1.4M（异常高） |
| **P3** | `pair_seq_N_mean` | 未来泄漏 | 🔴 致命 | 使用全量数据计算最近 N 次 |
| **P4** | `user_total_gift_7d` | 未来泄漏 | 🔴 致命 | groupby 包含未来样本 |
| **P5** | `streamer_recent_revenue` | 未来泄漏 | 🔴 致命 | groupby 包含未来样本 |

### 1.2 泄漏问题详解

#### P1: watch_live_time 结果泄漏

**数据来源**: `click.csv`
```csv
user_id,live_id,streamer_id,timestamp,watch_live_time
8505,9342705,392199,1746374400022,2852
```

**问题**:
- `watch_live_time = 2852ms` 是该 session 的**总观看时长**
- 包含用户进入直播间后的**所有时间**，包括打赏后的时间
- 因果关系：长观看 → 打赏，而非反过来
- 模型可以从 watch_time 直接推断打赏概率（泄漏）

**影响**:
- 如果用户看了 5 分钟后打赏，watch_live_time 包含打赏后的时间
- 这是**结果泄漏**：watch_time 是打赏行为的结果，而非原因

**解决方案**:
```python
# 方案 1: 完全移除（推荐）
# 不使用 watch_live_time 相关特征

# 方案 2: 截断到预测时刻
# 需要额外数据支持（知道每次打赏发生的具体时间）
# 目前数据不支持，因此选择方案 1
```

#### P2-P5: 聚合特征未来泄漏

**问题代码**（错误示例）:
```python
# ❌ 错误：使用 train 时间范围内的 ALL gifts
gift_train = gift[
    (gift['timestamp'] >= train_min_ts) &
    (gift['timestamp'] <= train_max_ts)
].copy()

# ❌ 错误：groupby 包含当前和未来样本
pair_stats = gift_train.groupby(['user_id', 'streamer_id']).agg({
    'gift_price': ['count', 'sum', 'mean']
})

# ❌ 错误：直接 merge 回去
df = df.merge(pair_stats, on=['user_id', 'streamer_id'], how='left')
```

**为什么错误**:
对于 train_df 中时间为 T 的样本，pair_gift_mean 包含了：
1. T 之前的历史礼物 ✅ 合法
2. **T 时刻的当前礼物（如果是正样本）❌ 泄漏！**
3. **T 之后、train_max_ts 之前的未来礼物 ❌ 泄漏！**

**诊断证据**:
```
首次打赏样本检查:
- 首次打赏（应该 count=0）的样本中，16.8% 有 count > 1
- 100% 的样本存在时间穿越（特征值包含未来信息）
- 平均 count 差异：1.3，最大差异：8
```

---

## 2. 7-7-7 数据划分规范

### 2.1 基本原则

| 原则 | 说明 |
|------|------|
| **按天划分** | 按自然日划分，而非按样本比例 |
| **时间顺序** | Train < Val < Test，严格按时间顺序 |
| **无重叠** | 三个集合的时间范围完全不重叠 |
| **Gap 可选** | 可在 Train/Val 和 Val/Test 之间添加 gap 防止边界泄漏 |

### 2.2 KuaiLive 数据时间范围

```
数据时间范围: 2025-05-04 ~ 2025-05-25 (共 22 天)

7-7-7 划分 (无 gap):
- Train: Day 1-7  (2025-05-04 ~ 2025-05-10) → 7 天
- Val:   Day 8-14 (2025-05-11 ~ 2025-05-17) → 7 天
- Test:  Day 15-21 (2025-05-18 ~ 2025-05-24) → 7 天
- 剩余:  Day 22 (2025-05-25) → 不使用或作为 buffer

7-7-7 划分 (带 1 天 gap):
- Train: Day 1-7  (2025-05-04 ~ 2025-05-10) → 7 天
- Gap:   Day 8    (2025-05-11)              → 1 天（不使用）
- Val:   Day 9-15 (2025-05-12 ~ 2025-05-18) → 7 天
- Gap:   Day 16   (2025-05-19)              → 1 天（不使用）
- Test:  Day 17-22 (2025-05-20 ~ 2025-05-25) → 6 天（最后一天不完整可能）
```

### 2.3 时间划分代码

```python
import pandas as pd
from datetime import datetime, timedelta

def split_by_days(df, train_days=7, val_days=7, test_days=7, gap_days=0,
                  timestamp_col='timestamp'):
    """
    按天划分数据集

    Args:
        df: DataFrame with timestamp column (in milliseconds)
        train_days: 训练集天数
        val_days: 验证集天数
        test_days: 测试集天数
        gap_days: Train-Val 和 Val-Test 之间的 gap 天数
        timestamp_col: 时间戳列名（毫秒）

    Returns:
        train_df, val_df, test_df
    """
    # 转换为 datetime
    df = df.copy()
    df['_datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
    df['_date'] = df['_datetime'].dt.date

    # 获取日期范围
    dates = sorted(df['_date'].unique())
    min_date = dates[0]

    # 计算切分点
    train_end_date = min_date + timedelta(days=train_days - 1)
    val_start_date = train_end_date + timedelta(days=gap_days + 1)
    val_end_date = val_start_date + timedelta(days=val_days - 1)
    test_start_date = val_end_date + timedelta(days=gap_days + 1)
    test_end_date = test_start_date + timedelta(days=test_days - 1)

    print(f"Data range: {dates[0]} ~ {dates[-1]} ({len(dates)} days)")
    print(f"Train: {min_date} ~ {train_end_date}")
    print(f"Val:   {val_start_date} ~ {val_end_date}")
    print(f"Test:  {test_start_date} ~ {test_end_date}")

    # 划分
    train_df = df[df['_date'] <= train_end_date].copy()
    val_df = df[(df['_date'] >= val_start_date) & (df['_date'] <= val_end_date)].copy()
    test_df = df[(df['_date'] >= test_start_date) & (df['_date'] <= test_end_date)].copy()

    # 清理临时列
    for d in [train_df, val_df, test_df]:
        d.drop(columns=['_datetime', '_date'], inplace=True)

    print(f"Split result: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    return train_df, val_df, test_df


def get_date_boundaries(df, train_days=7, val_days=7, gap_days=0,
                        timestamp_col='timestamp'):
    """
    获取日期边界（用于 frozen 特征计算）

    Returns:
        dict with train_end_ts, val_start_ts, val_end_ts, test_start_ts
    """
    df = df.copy()
    df['_datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
    df['_date'] = df['_datetime'].dt.date

    min_date = df['_date'].min()

    train_end_date = min_date + timedelta(days=train_days - 1)
    val_start_date = train_end_date + timedelta(days=gap_days + 1)
    val_end_date = val_start_date + timedelta(days=val_days - 1)
    test_start_date = val_end_date + timedelta(days=gap_days + 1)

    # 转换为 timestamp（毫秒）
    def date_to_ts_start(d):
        """日期转换为当天 00:00:00 的毫秒时间戳"""
        dt = datetime.combine(d, datetime.min.time())
        return int(dt.timestamp() * 1000)

    def date_to_ts_end(d):
        """日期转换为当天 23:59:59.999 的毫秒时间戳"""
        dt = datetime.combine(d, datetime.max.time())
        return int(dt.timestamp() * 1000)

    return {
        'train_start_ts': date_to_ts_start(min_date),
        'train_end_ts': date_to_ts_end(train_end_date),
        'val_start_ts': date_to_ts_start(val_start_date),
        'val_end_ts': date_to_ts_end(val_end_date),
        'test_start_ts': date_to_ts_start(test_start_date),
    }
```

---

## 3. 无泄漏特征构建规范

### 3.1 特征构建原则

| 原则 | 说明 | 实现方式 |
|------|------|---------|
| **Past-Only** | 只用 t < t_current 的历史数据 | cumsum + shift 或 frozen lookup |
| **严格不等式** | gift_timestamp < click_timestamp | searchsorted(side='left') - 1 |
| **Train 隔离** | Val/Test 只能用 Train 期间的统计 | Frozen lookup table |
| **无 watch_time** | 完全移除 watch_live_time | 从特征列表中删除 |

### 3.2 两种特征构建方式

#### 方式 1: Frozen（推荐用于生产）

**原理**: 只用 Train 时间窗口内的数据计算统计量，Val/Test 只查表

**优点**:
- 简单、无泄漏风险
- 符合线上推理场景（模型部署后不会实时更新统计）
- 计算效率高（预计算 lookup table）

**缺点**:
- Val/Test 期间的新 pair 没有历史信息（冷启动）

```python
def create_frozen_lookups(gift, train_end_ts):
    """
    创建 Frozen 版本的特征查找表

    Args:
        gift: 全量 gift 数据
        train_end_ts: Train 结束时间戳（毫秒）

    Returns:
        dict of lookup tables
    """
    # 只用 train 时间窗口内的 gifts
    gift_train = gift[gift['timestamp'] <= train_end_ts].copy()

    lookups = {}

    # 1. Pair-level features
    pair_stats = gift_train.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'timestamp': 'max'  # 最后一次打赏时间
    }).reset_index()
    pair_stats.columns = ['user_id', 'streamer_id',
                          'pair_gift_count', 'pair_gift_sum', 'pair_gift_mean',
                          'pair_gift_std', 'pair_gift_max', 'pair_last_gift_ts']
    pair_stats['pair_gift_std'] = pair_stats['pair_gift_std'].fillna(0)

    lookups['pair'] = {}
    for _, row in pair_stats.iterrows():
        key = (row['user_id'], row['streamer_id'])
        lookups['pair'][key] = {
            'count': row['pair_gift_count'],
            'sum': row['pair_gift_sum'],
            'mean': row['pair_gift_mean'],
            'std': row['pair_gift_std'],
            'max': row['pair_gift_max'],
            'last_ts': row['pair_last_gift_ts']
        }

    # 2. User-level features
    user_stats = gift_train.groupby('user_id').agg({
        'gift_price': ['count', 'sum', 'mean'],
        'streamer_id': 'nunique'
    }).reset_index()
    user_stats.columns = ['user_id', 'user_gift_count', 'user_gift_sum',
                          'user_gift_mean', 'user_unique_streamers']

    lookups['user'] = {}
    for _, row in user_stats.iterrows():
        lookups['user'][row['user_id']] = {
            'count': row['user_gift_count'],
            'sum': row['user_gift_sum'],
            'mean': row['user_gift_mean'],
            'unique_streamers': row['user_unique_streamers']
        }

    # 3. Streamer-level features
    streamer_stats = gift_train.groupby('streamer_id').agg({
        'gift_price': ['count', 'sum', 'mean'],
        'user_id': 'nunique'
    }).reset_index()
    streamer_stats.columns = ['streamer_id', 'streamer_gift_count', 'streamer_gift_sum',
                              'streamer_gift_mean', 'streamer_unique_givers']

    lookups['streamer'] = {}
    for _, row in streamer_stats.iterrows():
        lookups['streamer'][row['streamer_id']] = {
            'count': row['streamer_gift_count'],
            'sum': row['streamer_gift_sum'],
            'mean': row['streamer_gift_mean'],
            'unique_givers': row['streamer_unique_givers']
        }

    print(f"Created frozen lookups: {len(lookups['pair']):,} pairs, "
          f"{len(lookups['user']):,} users, {len(lookups['streamer']):,} streamers")

    return lookups


def apply_frozen_features(df, lookups, timestamp_col='timestamp'):
    """
    应用 Frozen 特征到 DataFrame

    使用向量化操作加速（避免 iterrows）
    """
    df = df.copy()

    # 创建 lookup 映射
    pair_keys = list(zip(df['user_id'], df['streamer_id']))

    # Pair features
    df['pair_gift_count_past'] = [
        lookups['pair'].get(k, {}).get('count', 0) for k in pair_keys
    ]
    df['pair_gift_sum_past'] = [
        lookups['pair'].get(k, {}).get('sum', 0) for k in pair_keys
    ]
    df['pair_gift_mean_past'] = [
        lookups['pair'].get(k, {}).get('mean', 0) for k in pair_keys
    ]

    # Time gap from last gift
    last_ts = np.array([
        lookups['pair'].get(k, {}).get('last_ts', np.nan) for k in pair_keys
    ])
    df['pair_last_gift_gap_hours'] = np.where(
        ~np.isnan(last_ts),
        (df[timestamp_col].values - last_ts) / (1000 * 3600),  # ms to hours
        999  # 无历史
    )

    # User features
    df['user_gift_count_past'] = df['user_id'].map(
        lambda x: lookups['user'].get(x, {}).get('count', 0)
    )
    df['user_gift_sum_past'] = df['user_id'].map(
        lambda x: lookups['user'].get(x, {}).get('sum', 0)
    )

    # Streamer features
    df['streamer_gift_count_past'] = df['streamer_id'].map(
        lambda x: lookups['streamer'].get(x, {}).get('count', 0)
    )
    df['streamer_gift_sum_past'] = df['streamer_id'].map(
        lambda x: lookups['streamer'].get(x, {}).get('sum', 0)
    )

    return df
```

#### 方式 2: Rolling（实验用，需严格验证）

**原理**: 每个样本使用 t < t_current 的历史数据

**优点**:
- 信息更丰富（Train 中后期样本能看到更多历史）
- 更接近实时更新场景

**缺点**:
- 实现复杂，容易出错（100% 样本曾出现泄漏）
- 计算效率低（需要 per-sample 计算或复杂的 cumsum+searchsorted）

```python
import numpy as np

def create_rolling_features_vectorized(gift, df):
    """
    使用 binary search 创建 Rolling 特征（无泄漏）

    关键: searchsorted(side='left') - 1 确保严格 <
    """
    df = df.copy().sort_values('timestamp').reset_index(drop=True)
    gift_sorted = gift.sort_values('timestamp').copy()

    # 计算累积统计
    gift_sorted['pair_gift_count_cum'] = gift_sorted.groupby(
        ['user_id', 'streamer_id']
    ).cumcount() + 1
    gift_sorted['pair_gift_sum_cum'] = gift_sorted.groupby(
        ['user_id', 'streamer_id']
    )['gift_price'].cumsum()
    gift_sorted['pair_gift_mean_cum'] = (
        gift_sorted['pair_gift_sum_cum'] / gift_sorted['pair_gift_count_cum']
    )

    # 构建 lookup 结构
    pair_lookup = {}
    for (user_id, streamer_id), grp in gift_sorted.groupby(['user_id', 'streamer_id']):
        grp = grp.sort_values('timestamp')
        pair_lookup[(user_id, streamer_id)] = {
            'ts': grp['timestamp'].values,
            'count': grp['pair_gift_count_cum'].values,
            'sum': grp['pair_gift_sum_cum'].values,
            'mean': grp['pair_gift_mean_cum'].values
        }

    # 向量化查找
    n = len(df)
    pair_count = np.zeros(n)
    pair_sum = np.zeros(n)
    pair_mean = np.zeros(n)
    pair_last_ts = np.full(n, np.nan)

    for idx in range(n):
        row = df.iloc[idx]
        key = (row['user_id'], row['streamer_id'])
        click_ts = row['timestamp']

        if key in pair_lookup:
            lookup = pair_lookup[key]
            ts_arr = lookup['ts']

            # 关键: searchsorted(side='left') 找第一个 >= click_ts 的位置
            # 然后 -1 得到最后一个 < click_ts 的位置
            pos = np.searchsorted(ts_arr, click_ts, side='left') - 1

            if pos >= 0:  # 有历史记录
                pair_count[idx] = lookup['count'][pos]
                pair_sum[idx] = lookup['sum'][pos]
                pair_mean[idx] = lookup['mean'][pos]
                pair_last_ts[idx] = ts_arr[pos]

    df['pair_gift_count_past'] = pair_count
    df['pair_gift_sum_past'] = pair_sum
    df['pair_gift_mean_past'] = pair_mean
    df['pair_last_gift_gap_hours'] = np.where(
        ~np.isnan(pair_last_ts),
        (df['timestamp'].values - pair_last_ts) / (1000 * 3600),
        999
    )

    return df
```

### 3.3 禁止使用的特征

| 特征 | 原因 | 替代方案 |
|------|------|---------|
| `watch_live_time` | 结果泄漏 | 完全移除 |
| `watch_time_log` | 同上 | 完全移除 |
| `watch_time_ratio` | 同上 | 完全移除 |
| `pair_gift_mean` (非 past-only) | 未来泄漏 | `pair_gift_mean_past` |
| `user_total_gift_7d` (非 past-only) | 未来泄漏 | `user_gift_sum_past` |

---

## 4. 统一数据处理代码

### 4.1 主函数

```python
#!/usr/bin/env python3
"""
Leakage-Free Data Processing Pipeline
=====================================

统一的无泄漏数据处理流程，所有实验必须使用此代码。

Usage:
    from data_processing import prepare_leakage_free_dataset

    train_df, val_df, test_df = prepare_leakage_free_dataset(
        train_days=7, val_days=7, test_days=7
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pickle

DATA_DIR = Path("/home/swei20/GiftLive/data/KuaiLive")
CACHE_DIR = Path("/home/swei20/GiftLive/gift_EVpred/features_cache")
CACHE_DIR.mkdir(exist_ok=True)


def load_raw_data():
    """加载原始数据"""
    gift = pd.read_csv(DATA_DIR / "gift.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")

    print(f"Loaded: gift={len(gift):,}, click={len(click):,}")
    return gift, click, user, streamer, room


def prepare_click_level_labels(gift, click, label_window_hours=1):
    """
    构建 Click-level 标签

    Label = click 后 label_window_hours 内的 gift 总额（0 或正数）
    """
    click = click.copy()
    gift = gift.copy()

    # 转换时间
    click['timestamp_dt'] = pd.to_datetime(click['timestamp'], unit='ms')
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')

    # Label window
    click['label_end_dt'] = click['timestamp_dt'] + pd.Timedelta(hours=label_window_hours)

    # Merge and filter
    merged = click[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'label_end_dt']].merge(
        gift[['user_id', 'streamer_id', 'live_id', 'timestamp_dt', 'gift_price']],
        on=['user_id', 'streamer_id', 'live_id'],
        how='left',
        suffixes=('_click', '_gift')
    )

    # Filter: gift within window
    merged = merged[
        (merged['timestamp_dt_gift'] >= merged['timestamp_dt_click']) &
        (merged['timestamp_dt_gift'] <= merged['label_end_dt'])
    ]

    # Aggregate
    gift_agg = merged.groupby(
        ['user_id', 'streamer_id', 'live_id', 'timestamp_dt_click']
    )['gift_price'].sum().reset_index().rename(columns={
        'timestamp_dt_click': 'timestamp_dt',
        'gift_price': 'gift_price_label'
    })

    # Merge back
    click = click.merge(
        gift_agg,
        on=['user_id', 'streamer_id', 'live_id', 'timestamp_dt'],
        how='left'
    )
    click['gift_price_label'] = click['gift_price_label'].fillna(0)

    # 清理: 删除 watch_live_time（泄漏特征）
    if 'watch_live_time' in click.columns:
        click = click.drop(columns=['watch_live_time'])
        print("Removed watch_live_time (leakage feature)")

    print(f"Click-level labels: {len(click):,} records")
    print(f"Gift rate: {(click['gift_price_label'] > 0).mean()*100:.2f}%")

    return click


def split_by_days(df, train_days=7, val_days=7, test_days=7, gap_days=0):
    """按天划分数据集（7-7-7）"""
    df = df.copy()
    df['_datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['_date'] = df['_datetime'].dt.date

    dates = sorted(df['_date'].unique())
    min_date = dates[0]

    train_end = min_date + timedelta(days=train_days - 1)
    val_start = train_end + timedelta(days=gap_days + 1)
    val_end = val_start + timedelta(days=val_days - 1)
    test_start = val_end + timedelta(days=gap_days + 1)
    test_end = test_start + timedelta(days=test_days - 1)

    print(f"Data range: {dates[0]} ~ {dates[-1]} ({len(dates)} days)")
    print(f"Train: {min_date} ~ {train_end} ({train_days} days)")
    print(f"Val:   {val_start} ~ {val_end} ({val_days} days)")
    print(f"Test:  {test_start} ~ {test_end} ({test_days} days)")

    train_df = df[df['_date'] <= train_end].copy()
    val_df = df[(df['_date'] >= val_start) & (df['_date'] <= val_end)].copy()
    test_df = df[(df['_date'] >= test_start) & (df['_date'] <= test_end)].copy()

    for d in [train_df, val_df, test_df]:
        d.drop(columns=['_datetime', '_date'], inplace=True)

    print(f"Split: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    return train_df, val_df, test_df


def create_frozen_lookups(gift, train_end_ts):
    """创建 Frozen 特征查找表"""
    gift_train = gift[gift['timestamp'] <= train_end_ts].copy()

    lookups = {}

    # Pair-level
    pair_stats = gift_train.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean'],
        'timestamp': 'max'
    }).reset_index()
    pair_stats.columns = ['user_id', 'streamer_id', 'count', 'sum', 'mean', 'last_ts']

    lookups['pair'] = {
        (r['user_id'], r['streamer_id']): {
            'count': r['count'], 'sum': r['sum'], 'mean': r['mean'], 'last_ts': r['last_ts']
        } for _, r in pair_stats.iterrows()
    }

    # User-level
    user_stats = gift_train.groupby('user_id')['gift_price'].agg(['count', 'sum', 'mean']).reset_index()
    lookups['user'] = {
        r['user_id']: {'count': r['count'], 'sum': r['sum'], 'mean': r['mean']}
        for _, r in user_stats.iterrows()
    }

    # Streamer-level
    str_stats = gift_train.groupby('streamer_id')['gift_price'].agg(['count', 'sum', 'mean']).reset_index()
    lookups['streamer'] = {
        r['streamer_id']: {'count': r['count'], 'sum': r['sum'], 'mean': r['mean']}
        for _, r in str_stats.iterrows()
    }

    print(f"Frozen lookups: {len(lookups['pair']):,} pairs, "
          f"{len(lookups['user']):,} users, {len(lookups['streamer']):,} streamers")

    return lookups


def apply_frozen_features(df, lookups):
    """应用 Frozen 特征"""
    df = df.copy()

    pair_keys = list(zip(df['user_id'], df['streamer_id']))

    # Pair features
    df['pair_gift_count_past'] = [lookups['pair'].get(k, {}).get('count', 0) for k in pair_keys]
    df['pair_gift_sum_past'] = [lookups['pair'].get(k, {}).get('sum', 0) for k in pair_keys]
    df['pair_gift_mean_past'] = [lookups['pair'].get(k, {}).get('mean', 0) for k in pair_keys]

    last_ts = np.array([lookups['pair'].get(k, {}).get('last_ts', np.nan) for k in pair_keys])
    df['pair_last_gift_gap_hours'] = np.where(
        ~np.isnan(last_ts),
        (df['timestamp'].values - last_ts) / (1000 * 3600),
        999
    )

    # User features
    df['user_gift_count_past'] = df['user_id'].map(lambda x: lookups['user'].get(x, {}).get('count', 0))
    df['user_gift_sum_past'] = df['user_id'].map(lambda x: lookups['user'].get(x, {}).get('sum', 0))

    # Streamer features
    df['str_gift_count_past'] = df['streamer_id'].map(lambda x: lookups['streamer'].get(x, {}).get('count', 0))
    df['str_gift_sum_past'] = df['streamer_id'].map(lambda x: lookups['streamer'].get(x, {}).get('sum', 0))

    return df


def add_static_features(df, user, streamer, room):
    """添加静态特征（无泄漏风险）"""
    # User features
    user_cols = ['user_id', 'age', 'gender', 'device_brand', 'device_price',
                 'fans_num', 'follow_num', 'accu_watch_live_cnt', 'accu_watch_live_duration']
    df = df.merge(user[user_cols], on='user_id', how='left')

    # Streamer features
    str_cols = ['streamer_id', 'fans_user_num', 'fans_group_fans_num', 'accu_live_cnt']
    df = df.merge(streamer[str_cols], on='streamer_id', how='left')

    # Room features
    room_cols = ['live_id', 'live_type', 'live_content_category']
    room_dedup = room[room_cols].drop_duplicates('live_id')
    df = df.merge(room_dedup, on='live_id', how='left')

    # Time features
    df['hour'] = pd.to_datetime(df['timestamp'], unit='ms').dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp'], unit='ms').dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    return df


def prepare_leakage_free_dataset(train_days=7, val_days=7, test_days=7,
                                  gap_days=0, use_cache=True):
    """
    主函数: 准备无泄漏的数据集

    Returns:
        train_df, val_df, test_df: 包含特征和标签的 DataFrames
    """
    print("="*60)
    print("Preparing Leakage-Free Dataset")
    print("="*60)

    # Load data
    gift, click, user, streamer, room = load_raw_data()

    # Click-level labels (removes watch_live_time)
    click_with_labels = prepare_click_level_labels(gift, click)

    # 7-7-7 split
    train_df, val_df, test_df = split_by_days(
        click_with_labels, train_days, val_days, test_days, gap_days
    )

    # Get train end timestamp for frozen features
    train_end_ts = train_df['timestamp'].max()

    # Cache key
    cache_key = f"frozen_{train_days}_{val_days}_{test_days}_{gap_days}.pkl"
    cache_path = CACHE_DIR / cache_key

    if use_cache and cache_path.exists():
        print(f"Loading cached lookups from {cache_path}")
        with open(cache_path, 'rb') as f:
            lookups = pickle.load(f)
    else:
        lookups = create_frozen_lookups(gift, train_end_ts)
        with open(cache_path, 'wb') as f:
            pickle.dump(lookups, f)
        print(f"Cached lookups to {cache_path}")

    # Apply features
    train_df = apply_frozen_features(train_df, lookups)
    val_df = apply_frozen_features(val_df, lookups)
    test_df = apply_frozen_features(test_df, lookups)

    # Add static features
    train_df = add_static_features(train_df, user, streamer, room)
    val_df = add_static_features(val_df, user, streamer, room)
    test_df = add_static_features(test_df, user, streamer, room)

    # Create targets
    for df in [train_df, val_df, test_df]:
        df['target'] = np.log1p(df['gift_price_label'])
        df['target_raw'] = df['gift_price_label']
        df['is_gift'] = (df['gift_price_label'] > 0).astype(int)

    print("="*60)
    print("Dataset preparation complete!")
    print(f"Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    print(f"Train gift rate: {train_df['is_gift'].mean()*100:.2f}%")
    print("="*60)

    return train_df, val_df, test_df


# 特征列获取函数
def get_feature_columns(df):
    """获取特征列（排除 metadata 和 target）"""
    exclude = {
        'user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
        'gift_price_label', 'target', 'target_raw', 'is_gift',
        'label_end_dt', 'watch_live_time'  # 确保 watch_live_time 被排除
    }
    return [c for c in df.columns if c not in exclude]
```

### 4.2 使用示例

```python
# 标准用法
from data_processing import prepare_leakage_free_dataset, get_feature_columns

# 准备数据（7-7-7 划分，无 gap）
train_df, val_df, test_df = prepare_leakage_free_dataset(
    train_days=7, val_days=7, test_days=7, gap_days=0
)

# 获取特征列
feature_cols = get_feature_columns(train_df)
print(f"Features: {len(feature_cols)}")

# 检查无泄漏
assert 'watch_live_time' not in feature_cols
assert all('past' in f for f in feature_cols if 'gift' in f)

# 训练模型
X_train = train_df[feature_cols]
y_train = train_df['target']
# ...
```

---

## 5. 验证清单

### 5.1 数据划分验证

```python
def verify_time_split(train_df, val_df, test_df):
    """验证时间划分正确性"""
    train_max = train_df['timestamp'].max()
    val_min = val_df['timestamp'].min()
    val_max = val_df['timestamp'].max()
    test_min = test_df['timestamp'].min()

    assert train_max < val_min, f"Train/Val overlap: {train_max} >= {val_min}"
    assert val_max < test_min, f"Val/Test overlap: {val_max} >= {test_min}"

    print("Time split verification: PASSED")
    print(f"  Train max: {pd.to_datetime(train_max, unit='ms')}")
    print(f"  Val min:   {pd.to_datetime(val_min, unit='ms')}")
    print(f"  Val max:   {pd.to_datetime(val_max, unit='ms')}")
    print(f"  Test min:  {pd.to_datetime(test_min, unit='ms')}")
```

### 5.2 特征泄漏验证

```python
def verify_no_leakage(df, gift, n_samples=100):
    """验证特征无泄漏"""
    import numpy as np

    gift_sorted = gift.sort_values('timestamp')
    errors = []

    sample_idx = np.random.choice(len(df), min(n_samples, len(df)), replace=False)

    for idx in sample_idx:
        row = df.iloc[idx]
        click_ts = row['timestamp']
        user_id = row['user_id']
        streamer_id = row['streamer_id']

        # 计算真实的 past-only count
        true_past = gift_sorted[
            (gift_sorted['user_id'] == user_id) &
            (gift_sorted['streamer_id'] == streamer_id) &
            (gift_sorted['timestamp'] < click_ts)  # 严格 <
        ]
        true_count = len(true_past)

        # 对比特征值
        feature_count = row['pair_gift_count_past']

        if feature_count != true_count:
            errors.append({
                'idx': idx,
                'expected': true_count,
                'got': feature_count,
                'diff': feature_count - true_count
            })

    if errors:
        print(f"Leakage verification: FAILED ({len(errors)}/{n_samples} samples)")
        for e in errors[:3]:
            print(f"  idx={e['idx']}: expected={e['expected']}, got={e['got']}")
        return False
    else:
        print(f"Leakage verification: PASSED ({n_samples} samples)")
        return True
```

### 5.3 特征列验证

```python
def verify_feature_columns(feature_cols):
    """验证特征列不包含泄漏特征"""
    forbidden = ['watch_live_time', 'watch_time_log', 'watch_time_ratio']

    for f in forbidden:
        assert f not in feature_cols, f"Forbidden feature found: {f}"

    # 检查 gift 相关特征必须带 _past 后缀
    gift_features = [f for f in feature_cols if 'gift' in f.lower()]
    for f in gift_features:
        if not f.endswith('_past') and 'label' not in f:
            print(f"WARNING: Gift feature without _past suffix: {f}")

    print("Feature column verification: PASSED")
```

### 5.4 完整验证流程

```python
def run_full_verification(train_df, val_df, test_df, gift, feature_cols):
    """运行完整验证"""
    print("="*60)
    print("Running Full Verification")
    print("="*60)

    # 1. 时间划分
    verify_time_split(train_df, val_df, test_df)

    # 2. 特征泄漏
    print("\nVerifying train set...")
    verify_no_leakage(train_df, gift, n_samples=100)

    print("\nVerifying val set...")
    verify_no_leakage(val_df, gift, n_samples=100)

    print("\nVerifying test set...")
    verify_no_leakage(test_df, gift, n_samples=100)

    # 3. 特征列
    verify_feature_columns(feature_cols)

    print("\n" + "="*60)
    print("All verifications PASSED!")
    print("="*60)
```

---

## 附录: 快速参考

### 必须遵守的规则

| # | 规则 | 检查方式 |
|---|------|---------|
| 1 | **不使用 watch_live_time** | `'watch_live_time' not in features` |
| 2 | **7-7-7 按天划分** | `train_max < val_min < test_min` |
| 3 | **Frozen 特征用 Train 数据** | `gift_ts <= train_end_ts` |
| 4 | **Past-only 严格 <** | `gift_ts < click_ts` |
| 5 | **所有 gift 特征带 _past 后缀** | `feature.endswith('_past')` |

### 常见错误

| 错误 | 后果 | 正确做法 |
|------|------|---------|
| 使用 `groupby().agg()` 全量数据 | 100% 样本泄漏 | 使用 frozen lookup 或 cumsum+searchsorted |
| 使用 watch_live_time | 结果泄漏 | 完全移除 |
| 按样本比例划分 | 时间穿越 | 按天划分 |
| `searchsorted(side='right')` | 包含当前样本 | 使用 `side='left'` 然后 `-1` |

---

> **文档版本**: v1.0
> **最后更新**: 2026-01-18
> **维护者**: Viska Wei
