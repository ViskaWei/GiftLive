#!/usr/bin/env python3
"""
Gift EV Prediction - Leakage-Free Data Utilities (Day-Frozen Version)
======================================================================

统一的无泄漏数据处理模块，所有 gift_EVpred 实验必须使用此模块。

核心设计：按天冻结（Day-Frozen / Day-Snapshot）
- 对每个 click 的特征，只允许用 **之前的天（day < 当前 day）**的历史
- 完全不会用到"未来"，但会丢掉"同一天更早发生的历史"（保守但安全）
- 训练/验证/测试都用同一套构造逻辑

Usage:
    from gift_EVpred.data_utils import (
        load_raw_data,
        prepare_dataset,
        get_feature_columns,
        verify_no_leakage
    )

    # 标准用法
    train_df, val_df, test_df = prepare_dataset()

    # 获取特征列
    feature_cols = get_feature_columns(train_df)

Author: Viska Wei
Date: 2026-01-18
Version: 2.0 (Day-Frozen)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_EVpred"

# =============================================================================
# 常量配置
# =============================================================================
SEED = 42
np.random.seed(SEED)

# 泄漏特征黑名单（绝对禁止使用）
FORBIDDEN_FEATURES = [
    'watch_live_time',      # 结果泄漏：包含打赏后的观看时长
    'watch_time_log',       # 同上
    'watch_time_ratio',     # 同上
]

# 必须排除的列（非特征）
EXCLUDE_COLUMNS = [
    'user_id', 'live_id', 'streamer_id', 'timestamp', 'timestamp_dt',
    'gift_price_label', 'target', 'target_raw', 'is_gift',
    'label_end_dt', '_datetime', '_date', 'day',
] + FORBIDDEN_FEATURES


# =============================================================================
# 日志工具
# =============================================================================
def log(msg: str, level: str = "INFO"):
    """打印日志"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    prefix = {"INFO": "📝", "SUCCESS": "✅", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "")
    print(f"[{timestamp}] {prefix} {msg}")


# =============================================================================
# 数据加载
# =============================================================================
def load_raw_data():
    """
    加载原始数据

    Returns:
        tuple: (gift, click, user, streamer, room) DataFrames
    """
    log("Loading raw data...")

    gift = pd.read_csv(DATA_DIR / "gift.csv")
    click = pd.read_csv(DATA_DIR / "click.csv")
    user = pd.read_csv(DATA_DIR / "user.csv")
    streamer = pd.read_csv(DATA_DIR / "streamer.csv")
    room = pd.read_csv(DATA_DIR / "room.csv")

    log(f"Loaded: gift={len(gift):,}, click={len(click):,}, "
        f"user={len(user):,}, streamer={len(streamer):,}, room={len(room):,}")

    return gift, click, user, streamer, room


# =============================================================================
# Click-Level 标签构建
# =============================================================================
def prepare_click_level_labels(gift, click, label_window_hours=1):
    """
    构建 Click-level 标签

    Args:
        gift: gift DataFrame
        click: click DataFrame
        label_window_hours: 标签窗口（小时）

    Returns:
        DataFrame: click 数据 + gift_price_label 列（0 或正数）

    注意:
        - 会自动删除 watch_live_time 列（泄漏特征）
        - Label = click 后 label_window_hours 内的 gift 总额
    """
    log(f"Preparing click-level labels (window={label_window_hours}h)...")

    click = click.copy()
    gift = gift.copy()

    # 删除泄漏特征
    if 'watch_live_time' in click.columns:
        click = click.drop(columns=['watch_live_time'])
        log("Removed watch_live_time (leakage feature)", "WARNING")

    # 转换时间
    click['timestamp_dt'] = pd.to_datetime(click['timestamp'], unit='ms')
    gift['timestamp_dt'] = pd.to_datetime(gift['timestamp'], unit='ms')

    # Label window
    click['label_end_dt'] = click['timestamp_dt'] + pd.Timedelta(hours=label_window_hours)

    # Merge and filter
    merged = click[['user_id', 'streamer_id', 'live_id', 'timestamp', 'timestamp_dt', 'label_end_dt']].merge(
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
        ['user_id', 'streamer_id', 'live_id', 'timestamp']
    )['gift_price'].sum().reset_index().rename(columns={'gift_price': 'gift_price_label'})

    # Merge back
    click = click.merge(gift_agg, on=['user_id', 'streamer_id', 'live_id', 'timestamp'], how='left')
    click['gift_price_label'] = click['gift_price_label'].fillna(0)

    # 清理临时列
    click = click.drop(columns=['label_end_dt'], errors='ignore')

    log(f"Click-level data: {len(click):,} records, gift_rate={click['gift_price_label'].gt(0).mean()*100:.2f}%")

    return click


# =============================================================================
# 7-7-7 数据划分
# =============================================================================
def split_by_days(df, train_days=7, val_days=7, test_days=7, gap_days=0):
    """
    按天划分数据集（7-7-7）

    Args:
        df: DataFrame with timestamp column (milliseconds)
        train_days: 训练集天数 (default: 7)
        val_days: 验证集天数 (default: 7)
        test_days: 测试集天数 (default: 7)
        gap_days: Train-Val 和 Val-Test 之间的 gap 天数 (default: 0)

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    log(f"Splitting data: {train_days}-{val_days}-{test_days} days (gap={gap_days})...")

    df = df.copy()
    df['_datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['_date'] = df['_datetime'].dt.date

    dates = sorted(df['_date'].unique())
    min_date = dates[0]
    max_date = dates[-1]

    # 计算切分点
    train_end = min_date + timedelta(days=train_days - 1)
    val_start = train_end + timedelta(days=gap_days + 1)
    val_end = val_start + timedelta(days=val_days - 1)
    test_start = val_end + timedelta(days=gap_days + 1)
    test_end = test_start + timedelta(days=test_days - 1)

    log(f"Data range: {min_date} ~ {max_date} ({len(dates)} days)")
    log(f"  Train: {min_date} ~ {train_end} ({train_days} days)")
    log(f"  Val:   {val_start} ~ {val_end} ({val_days} days)")
    log(f"  Test:  {test_start} ~ {test_end} ({test_days} days)")

    # 划分
    train_df = df[df['_date'] <= train_end].copy()
    val_df = df[(df['_date'] >= val_start) & (df['_date'] <= val_end)].copy()
    test_df = df[(df['_date'] >= test_start) & (df['_date'] <= test_end)].copy()

    # 清理临时列
    for d in [train_df, val_df, test_df]:
        d.drop(columns=['_datetime', '_date'], inplace=True, errors='ignore')

    log(f"Split result: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

    return train_df, val_df, test_df


# =============================================================================
# Day-Frozen 历史特征（无泄漏）
# =============================================================================
def create_day_frozen_features(gift, click):
    """
    创建按天冻结的历史特征（Day-Frozen / Day-Snapshot）

    核心设计：
    - 对每个 click 的特征，只允许用 **之前的天（day < 当前 day）**的历史
    - 使用 pd.merge_asof(..., by=..., allow_exact_matches=False) 高效实现
    - 训练/验证/测试都用同一套逻辑
    - 无泄漏、口径一致

    Args:
        gift: 全量 gift DataFrame
        click: 全量 click DataFrame (已删除 watch_live_time)

    Returns:
        DataFrame: click 数据 + 历史特征
    """
    log("Creating day-frozen historical features...")

    click = click.copy()
    gift = gift.copy()

    # 添加 day 列（转换为 datetime 以便 merge_asof）
    click['day'] = pd.to_datetime(click['timestamp'], unit='ms').dt.normalize()
    gift['day'] = pd.to_datetime(gift['timestamp'], unit='ms').dt.normalize()

    # =========================================================================
    # Pair-level 历史特征 (user, streamer)
    # =========================================================================
    log("  Building pair-level features...")

    # 按天聚合
    pair_day = gift.groupby(['day', 'user_id', 'streamer_id'])['gift_price'].agg(
        gift_cnt_day='count',
        gift_sum_day='sum'
    ).reset_index()

    # 按 day 全局排序 (merge_asof 要求 on 列全局排序)
    pair_day = pair_day.sort_values('day').reset_index(drop=True)

    # Cumsum: 截至当天（含）的累计
    pair_day[['pair_gift_cnt_hist', 'pair_gift_sum_hist']] = pair_day.groupby(
        ['user_id', 'streamer_id']
    )[['gift_cnt_day', 'gift_sum_day']].cumsum()

    # 准备 click 数据用于 merge_asof (按 day 全局排序)
    click_sorted = click.sort_values('day').reset_index(drop=True)

    # merge_asof with by: 按 (user, streamer) 分组，查找 strictly before 的最近记录
    click_with_pair = pd.merge_asof(
        click_sorted,
        pair_day[['day', 'user_id', 'streamer_id', 'pair_gift_cnt_hist', 'pair_gift_sum_hist']],
        on='day',
        by=['user_id', 'streamer_id'],
        direction='backward',
        allow_exact_matches=False  # 严格 < 当前天
    )

    # 填充 NaN 为 0
    click_with_pair[['pair_gift_cnt_hist', 'pair_gift_sum_hist']] = \
        click_with_pair[['pair_gift_cnt_hist', 'pair_gift_sum_hist']].fillna(0)

    # 计算均值
    click_with_pair['pair_gift_mean_hist'] = (
        click_with_pair['pair_gift_sum_hist'] / click_with_pair['pair_gift_cnt_hist'].replace(0, np.nan)
    ).fillna(0)

    log(f"    Pair features: {(click_with_pair['pair_gift_cnt_hist'] > 0).sum():,} samples have history")

    # =========================================================================
    # User-level 历史特征
    # =========================================================================
    log("  Building user-level features...")

    user_day = gift.groupby(['day', 'user_id'])['gift_price'].agg(
        gift_cnt_day='count',
        gift_sum_day='sum'
    ).reset_index()

    # 按 day 全局排序
    user_day = user_day.sort_values('day').reset_index(drop=True)
    user_day[['user_gift_cnt_hist', 'user_gift_sum_hist']] = user_day.groupby('user_id')[
        ['gift_cnt_day', 'gift_sum_day']
    ].cumsum()

    # merge_asof by user_id (click_with_pair 已按 day 排序)
    click_with_user = pd.merge_asof(
        click_with_pair,
        user_day[['day', 'user_id', 'user_gift_cnt_hist', 'user_gift_sum_hist']],
        on='day',
        by='user_id',
        direction='backward',
        allow_exact_matches=False
    )

    click_with_user[['user_gift_cnt_hist', 'user_gift_sum_hist']] = \
        click_with_user[['user_gift_cnt_hist', 'user_gift_sum_hist']].fillna(0)
    click_with_user['user_gift_mean_hist'] = (
        click_with_user['user_gift_sum_hist'] / click_with_user['user_gift_cnt_hist'].replace(0, np.nan)
    ).fillna(0)

    log(f"    User features: {(click_with_user['user_gift_cnt_hist'] > 0).sum():,} samples have history")

    # =========================================================================
    # Streamer-level 历史特征
    # =========================================================================
    log("  Building streamer-level features...")

    str_day = gift.groupby(['day', 'streamer_id'])['gift_price'].agg(
        gift_cnt_day='count',
        gift_sum_day='sum'
    ).reset_index()

    # 按 day 全局排序
    str_day = str_day.sort_values('day').reset_index(drop=True)
    str_day[['str_gift_cnt_hist', 'str_gift_sum_hist']] = str_day.groupby('streamer_id')[
        ['gift_cnt_day', 'gift_sum_day']
    ].cumsum()

    # merge_asof by streamer_id (click_with_user 已按 day 排序)
    click_with_str = pd.merge_asof(
        click_with_user,
        str_day[['day', 'streamer_id', 'str_gift_cnt_hist', 'str_gift_sum_hist']],
        on='day',
        by='streamer_id',
        direction='backward',
        allow_exact_matches=False
    )

    click_with_str[['str_gift_cnt_hist', 'str_gift_sum_hist']] = \
        click_with_str[['str_gift_cnt_hist', 'str_gift_sum_hist']].fillna(0)
    click_with_str['str_gift_mean_hist'] = (
        click_with_str['str_gift_sum_hist'] / click_with_str['str_gift_cnt_hist'].replace(0, np.nan)
    ).fillna(0)

    log(f"    Streamer features: {(click_with_str['str_gift_cnt_hist'] > 0).sum():,} samples have history")

    click = click_with_str

    # =========================================================================
    # 历史观看时长特征（可选，使用过去天的 click/watch_time）
    # =========================================================================
    # 注意：当前实现不使用 watch_time 特征，因为需要原始 click 保留 watch_live_time
    # 如需添加，可以在此处实现类似的 cumsum + shift 逻辑

    log("Day-frozen features created!", "SUCCESS")
    log(f"  Total historical features: 9 (pair: 3, user: 3, streamer: 3)")

    return click


# =============================================================================
# 静态特征（无泄漏风险）
# =============================================================================
def add_static_features(df, user, streamer, room):
    """
    添加静态特征（profile 特征，无泄漏风险）

    Args:
        df: DataFrame
        user: user DataFrame
        streamer: streamer DataFrame
        room: room DataFrame

    Returns:
        DataFrame with static features added
    """
    log("Adding static features...")

    df = df.copy()

    # -------------------------------------------------------------------------
    # User profile features
    # -------------------------------------------------------------------------
    user_cols = ['user_id', 'age', 'gender', 'device_brand', 'device_price',
                 'fans_num', 'follow_num', 'accu_watch_live_cnt', 'accu_watch_live_duration',
                 'is_live_streamer', 'is_photo_author']
    user_cols = [c for c in user_cols if c in user.columns]
    df = df.merge(user[user_cols], on='user_id', how='left')

    # -------------------------------------------------------------------------
    # Streamer profile features
    # -------------------------------------------------------------------------
    str_cols = ['streamer_id', 'fans_user_num', 'fans_group_fans_num',
                'follow_user_num', 'accu_live_cnt', 'accu_live_duration',
                'accu_play_cnt', 'accu_play_duration']
    str_cols = [c for c in str_cols if c in streamer.columns]

    # Rename to avoid conflict with user features
    streamer_subset = streamer[str_cols].copy()
    rename_map = {c: f'str_{c}' for c in str_cols if c != 'streamer_id'}
    streamer_subset = streamer_subset.rename(columns=rename_map)

    df = df.merge(streamer_subset, on='streamer_id', how='left')

    # -------------------------------------------------------------------------
    # Room features
    # -------------------------------------------------------------------------
    room_cols = ['live_id', 'live_type', 'live_content_category']
    room_cols = [c for c in room_cols if c in room.columns]
    room_dedup = room[room_cols].drop_duplicates('live_id')
    df = df.merge(room_dedup, on='live_id', how='left')

    # -------------------------------------------------------------------------
    # Time features (从 timestamp 提取)
    # -------------------------------------------------------------------------
    ts_dt = pd.to_datetime(df['timestamp'], unit='ms')
    df['hour'] = ts_dt.dt.hour
    df['day_of_week'] = ts_dt.dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    log(f"  Added static features, total columns: {df.shape[1]}", "SUCCESS")

    return df


# =============================================================================
# 主函数
# =============================================================================
def prepare_dataset(train_days=7, val_days=7, test_days=7, gap_days=0,
                    label_window_hours=1, use_cache=True):
    """
    准备无泄漏的数据集（主函数）

    这是所有 gift_EVpred 实验的标准入口。

    核心设计：Day-Frozen（按天冻结）
    - 对每个 click 的特征，只允许用 **之前的天（day < 当前 day）**的历史
    - 训练/验证/测试都用同一套构造逻辑
    - 无泄漏、口径一致

    Args:
        train_days: 训练集天数 (default: 7)
        val_days: 验证集天数 (default: 7)
        test_days: 测试集天数 (default: 7)
        gap_days: Train-Val/Val-Test gap 天数 (default: 0)
        label_window_hours: 标签窗口（小时）(default: 1)
        use_cache: 是否使用缓存 (default: True)

    Returns:
        tuple: (train_df, val_df, test_df)

    Example:
        >>> from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
        >>> train_df, val_df, test_df = prepare_dataset()
        >>> feature_cols = get_feature_columns(train_df)
    """
    log("=" * 60)
    log("Preparing Leakage-Free Dataset (Day-Frozen Version)")
    log("=" * 60)

    # 缓存文件路径
    CACHE_DIR = OUTPUT_DIR / "features_cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"day_frozen_features_lw{label_window_hours}h.parquet"

    # 1. Load raw data
    gift, click, user, streamer, room = load_raw_data()

    # 2. 尝试从缓存加载
    if use_cache and cache_file.exists():
        log(f"Loading cached features from {cache_file}")
        click_with_features = pd.read_parquet(cache_file)
        log(f"  Loaded {len(click_with_features):,} records from cache", "SUCCESS")
    else:
        # 2. Click-level labels (removes watch_live_time automatically)
        click_with_labels = prepare_click_level_labels(gift, click, label_window_hours)

        # 3. Create day-frozen historical features (before split!)
        # 这样训练/验证/测试都用同一套逻辑
        click_with_features = create_day_frozen_features(gift, click_with_labels)

        # 保存缓存
        if use_cache:
            click_with_features.to_parquet(cache_file)
            log(f"Saved features to cache: {cache_file}", "SUCCESS")

    # 4. 7-7-7 split
    train_df, val_df, test_df = split_by_days(
        click_with_features, train_days, val_days, test_days, gap_days
    )

    # 5. Add static features
    train_df = add_static_features(train_df, user, streamer, room)
    val_df = add_static_features(val_df, user, streamer, room)
    test_df = add_static_features(test_df, user, streamer, room)

    # 6. Create targets
    for df in [train_df, val_df, test_df]:
        df['target'] = np.log1p(df['gift_price_label'])
        df['target_raw'] = df['gift_price_label']
        df['is_gift'] = (df['gift_price_label'] > 0).astype(int)

    # 7. Encode all object/category columns (Train 拟合，Val/Test 复用)
    # 先识别需要编码的列
    cat_cols = []
    for col in train_df.columns:
        if train_df[col].dtype == 'object' or str(train_df[col].dtype) == 'category':
            cat_cols.append(col)

    # Train 拟合 categories，Val/Test 复用同一映射
    for col in cat_cols:
        # 填充 NaN
        train_df[col] = train_df[col].fillna('unknown').astype(str)
        val_df[col] = val_df[col].fillna('unknown').astype(str)
        test_df[col] = test_df[col].fillna('unknown').astype(str)

        # 用 Train 的唯一值建立 categories（加上 'unknown' 兜底）
        train_categories = list(train_df[col].unique())
        if 'unknown' not in train_categories:
            train_categories.append('unknown')

        # 创建带固定 categories 的 Categorical，未知值映射为 'unknown'
        for df in [train_df, val_df, test_df]:
            # 将 Val/Test 中 Train 没见过的值替换为 'unknown'
            df[col] = df[col].apply(lambda x: x if x in train_categories else 'unknown')
            df[col] = pd.Categorical(df[col], categories=train_categories).codes

    log(f"  Encoded {len(cat_cols)} categorical columns (train-fitted)", "SUCCESS")

    # 8. Fill NaN for numeric columns
    for df in [train_df, val_df, test_df]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

    log("=" * 60)
    log("Dataset preparation complete!", "SUCCESS")
    log(f"  Train: {len(train_df):,} (gift_rate={train_df['is_gift'].mean()*100:.2f}%)")
    log(f"  Val:   {len(val_df):,} (gift_rate={val_df['is_gift'].mean()*100:.2f}%)")
    log(f"  Test:  {len(test_df):,} (gift_rate={test_df['is_gift'].mean()*100:.2f}%)")
    log("=" * 60)

    return train_df, val_df, test_df


# =============================================================================
# 特征列工具
# =============================================================================
def get_feature_columns(df):
    """
    获取特征列（排除 metadata、target 和泄漏特征）

    Args:
        df: DataFrame

    Returns:
        list: feature column names
    """
    exclude = set(EXCLUDE_COLUMNS)
    features = [c for c in df.columns if c not in exclude]

    # 额外检查：确保没有泄漏特征
    for f in FORBIDDEN_FEATURES:
        if f in features:
            features.remove(f)
            log(f"Removed forbidden feature: {f}", "WARNING")

    return features


def verify_feature_columns(feature_cols):
    """
    验证特征列不包含泄漏特征

    Args:
        feature_cols: list of feature column names

    Raises:
        AssertionError if forbidden features found
    """
    for f in FORBIDDEN_FEATURES:
        assert f not in feature_cols, f"Forbidden feature found: {f}"

    log("Feature column verification: PASSED", "SUCCESS")


# =============================================================================
# 验证函数
# =============================================================================
def verify_time_split(train_df, val_df, test_df):
    """
    验证时间划分正确性（无时间穿越）

    Args:
        train_df, val_df, test_df: DataFrames

    Raises:
        AssertionError if time overlap detected
    """
    train_max = train_df['timestamp'].max()
    val_min = val_df['timestamp'].min()
    val_max = val_df['timestamp'].max()
    test_min = test_df['timestamp'].min()

    assert train_max < val_min, f"Train/Val overlap: {train_max} >= {val_min}"
    assert val_max < test_min, f"Val/Test overlap: {val_max} >= {test_min}"

    log("Time split verification: PASSED", "SUCCESS")
    log(f"  Train max: {pd.to_datetime(train_max, unit='ms')}")
    log(f"  Val min:   {pd.to_datetime(val_min, unit='ms')}")
    log(f"  Test min:  {pd.to_datetime(test_min, unit='ms')}")


def verify_no_leakage(df, gift, n_samples=100):
    """
    验证特征无泄漏（抽样检查）

    对于每个样本，验证 pair_gift_cnt_hist 等于真实的 past-day count。
    Day-Frozen: 只用 day < 当前 day 的历史

    Args:
        df: DataFrame with features
        gift: original gift DataFrame
        n_samples: number of samples to check

    Returns:
        bool: True if passed, False if leakage detected
    """
    log(f"Verifying no leakage ({n_samples} samples)...")

    gift = gift.copy()
    gift['day'] = pd.to_datetime(gift['timestamp'], unit='ms').dt.normalize()

    errors = []

    sample_idx = np.random.choice(len(df), min(n_samples, len(df)), replace=False)

    for idx in sample_idx:
        row = df.iloc[idx]
        click_day = pd.to_datetime(row['timestamp'], unit='ms').normalize()
        user_id = row['user_id']
        streamer_id = row['streamer_id']

        # 计算真实的 past-day count (day < click_day)
        true_past = gift[
            (gift['user_id'] == user_id) &
            (gift['streamer_id'] == streamer_id) &
            (gift['day'] < click_day)  # 严格 < 当前天
        ]
        true_count = len(true_past)

        # 对比特征值
        feature_count = row['pair_gift_cnt_hist']

        if feature_count != true_count:
            errors.append({
                'idx': idx,
                'day': click_day,
                'expected': true_count,
                'got': feature_count,
                'diff': feature_count - true_count
            })

    if errors:
        log(f"Leakage verification: FAILED ({len(errors)}/{n_samples} samples)", "ERROR")
        for e in errors[:3]:
            log(f"  idx={e['idx']}, day={e['day']}: expected={e['expected']}, got={e['got']}, diff={e['diff']}")
        return False
    else:
        log(f"Leakage verification: PASSED ({n_samples}/{n_samples} samples)", "SUCCESS")
        return True


def run_full_verification(train_df, val_df, test_df, gift, feature_cols):
    """
    运行完整验证流程

    Args:
        train_df, val_df, test_df: DataFrames
        gift: original gift DataFrame
        feature_cols: list of feature column names

    Returns:
        bool: True if all verifications passed
    """
    log("=" * 60)
    log("Running Full Verification")
    log("=" * 60)

    all_passed = True

    # 1. 时间划分
    try:
        verify_time_split(train_df, val_df, test_df)
    except AssertionError as e:
        log(f"Time split verification FAILED: {e}", "ERROR")
        all_passed = False

    # 2. 特征列
    try:
        verify_feature_columns(feature_cols)
    except AssertionError as e:
        log(f"Feature column verification FAILED: {e}", "ERROR")
        all_passed = False

    # 3. 泄漏检查
    log("\nVerifying train set...")
    if not verify_no_leakage(train_df, gift, n_samples=50):
        all_passed = False

    log("\nVerifying val set...")
    if not verify_no_leakage(val_df, gift, n_samples=50):
        all_passed = False

    log("\nVerifying test set...")
    if not verify_no_leakage(test_df, gift, n_samples=100):
        all_passed = False

    log("=" * 60)
    if all_passed:
        log("All verifications PASSED!", "SUCCESS")
    else:
        log("Some verifications FAILED!", "ERROR")
    log("=" * 60)

    return all_passed


# =============================================================================
# CLI 入口
# =============================================================================
if __name__ == '__main__':
    # 示例用法
    print("Gift EVpred Data Utils - Day-Frozen Version")
    print("=" * 60)

    # 准备数据
    train_df, val_df, test_df = prepare_dataset()

    # 获取特征列
    feature_cols = get_feature_columns(train_df)
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(feature_cols[:10], "...")

    # 运行验证
    gift, _, _, _, _ = load_raw_data()
    run_full_verification(train_df, val_df, test_df, gift, feature_cols)
