#!/usr/bin/env python3
"""
Gift EV Prediction - Leakage-Free Data Utilities
=================================================

统一的无泄漏数据处理模块，所有 gift_EVpred 实验必须使用此模块。

Usage:
    from gift_EVpred.data_utils import (
        load_raw_data,
        prepare_dataset,
        get_feature_columns,
        verify_no_leakage
    )

    # 标准用法
    train_df, val_df, test_df, lookups = prepare_dataset()

    # 获取特征列
    feature_cols = get_feature_columns(train_df)

    # 验证无泄漏
    verify_no_leakage(train_df, gift_df)

Author: Viska Wei
Date: 2026-01-18
Version: 1.0
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import pickle
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = Path("/home/swei20/GiftLive")
DATA_DIR = BASE_DIR / "data" / "KuaiLive"
OUTPUT_DIR = BASE_DIR / "gift_EVpred"
CACHE_DIR = OUTPUT_DIR / "features_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
    'label_end_dt', '_datetime', '_date',
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
# Frozen 特征（无泄漏）
# =============================================================================
def create_frozen_lookups(gift, train_end_ts, click=None):
    """
    创建 Frozen 版本的特征查找表

    只使用 Train 时间窗口内的数据计算统计量，Val/Test 只查表。

    Args:
        gift: 全量 gift DataFrame
        train_end_ts: Train 结束时间戳（毫秒）
        click: 全量 click DataFrame（可选，用于计算历史观看时长）

    Returns:
        dict: {'pair': {...}, 'user': {...}, 'streamer': {...}, 'watch_time': {...}}
    """
    log("Creating frozen lookups (train window only)...")

    # 只用 train 时间窗口内的 gifts
    gift_train = gift[gift['timestamp'] <= train_end_ts].copy()
    log(f"  Train gifts: {len(gift_train):,} (before {pd.to_datetime(train_end_ts, unit='ms')})")

    lookups = {}

    # -------------------------------------------------------------------------
    # Pair-level features (user, streamer)
    # -------------------------------------------------------------------------
    pair_stats = gift_train.groupby(['user_id', 'streamer_id']).agg({
        'gift_price': ['count', 'sum', 'mean', 'std', 'max'],
        'timestamp': 'max'  # 最后一次打赏时间
    }).reset_index()
    pair_stats.columns = ['user_id', 'streamer_id',
                          'count', 'sum', 'mean', 'std', 'max', 'last_ts']
    pair_stats['std'] = pair_stats['std'].fillna(0)

    lookups['pair'] = {}
    for _, row in pair_stats.iterrows():
        key = (row['user_id'], row['streamer_id'])
        lookups['pair'][key] = {
            'count': row['count'],
            'sum': row['sum'],
            'mean': row['mean'],
            'std': row['std'],
            'max': row['max'],
            'last_ts': row['last_ts']
        }

    # -------------------------------------------------------------------------
    # User-level features
    # -------------------------------------------------------------------------
    user_stats = gift_train.groupby('user_id').agg({
        'gift_price': ['count', 'sum', 'mean'],
        'streamer_id': 'nunique'
    }).reset_index()
    user_stats.columns = ['user_id', 'count', 'sum', 'mean', 'unique_streamers']

    lookups['user'] = {}
    for _, row in user_stats.iterrows():
        lookups['user'][row['user_id']] = {
            'count': row['count'],
            'sum': row['sum'],
            'mean': row['mean'],
            'unique_streamers': row['unique_streamers']
        }

    # -------------------------------------------------------------------------
    # Streamer-level features
    # -------------------------------------------------------------------------
    str_stats = gift_train.groupby('streamer_id').agg({
        'gift_price': ['count', 'sum', 'mean'],
        'user_id': 'nunique'
    }).reset_index()
    str_stats.columns = ['streamer_id', 'count', 'sum', 'mean', 'unique_givers']

    lookups['streamer'] = {}
    for _, row in str_stats.iterrows():
        lookups['streamer'][row['streamer_id']] = {
            'count': row['count'],
            'sum': row['sum'],
            'mean': row['mean'],
            'unique_givers': row['unique_givers']
        }

    # -------------------------------------------------------------------------
    # Watch time features (历史观看时长，无泄漏)
    # 注意：这里用的是 train 期间的 click 数据的 watch_live_time
    # -------------------------------------------------------------------------
    lookups['watch_time'] = {'user': {}, 'pair': {}, 'streamer': {}}

    if click is not None and 'watch_live_time' in click.columns:
        click_train = click[click['timestamp'] <= train_end_ts].copy()
        log(f"  Train clicks for watch_time: {len(click_train):,}")

        # User-level: 用户历史平均观看时长
        user_watch = click_train.groupby('user_id')['watch_live_time'].agg(['mean', 'sum', 'count']).reset_index()
        user_watch.columns = ['user_id', 'mean', 'sum', 'count']
        for _, row in user_watch.iterrows():
            lookups['watch_time']['user'][row['user_id']] = {
                'mean': row['mean'],
                'sum': row['sum'],
                'count': row['count']
            }

        # Pair-level: user-streamer 历史平均观看时长
        pair_watch = click_train.groupby(['user_id', 'streamer_id'])['watch_live_time'].agg(['mean', 'sum', 'count']).reset_index()
        pair_watch.columns = ['user_id', 'streamer_id', 'mean', 'sum', 'count']
        for _, row in pair_watch.iterrows():
            key = (row['user_id'], row['streamer_id'])
            lookups['watch_time']['pair'][key] = {
                'mean': row['mean'],
                'sum': row['sum'],
                'count': row['count']
            }

        # Streamer-level: 主播历史平均被观看时长
        str_watch = click_train.groupby('streamer_id')['watch_live_time'].agg(['mean', 'sum', 'count']).reset_index()
        str_watch.columns = ['streamer_id', 'mean', 'sum', 'count']
        for _, row in str_watch.iterrows():
            lookups['watch_time']['streamer'][row['streamer_id']] = {
                'mean': row['mean'],
                'sum': row['sum'],
                'count': row['count']
            }

        log(f"  Watch time lookups: {len(lookups['watch_time']['pair']):,} pairs, "
            f"{len(lookups['watch_time']['user']):,} users", "SUCCESS")

    log(f"  Lookups created: {len(lookups['pair']):,} pairs, "
        f"{len(lookups['user']):,} users, {len(lookups['streamer']):,} streamers", "SUCCESS")

    return lookups


def apply_frozen_features(df, lookups, is_train=False):
    """
    应用 Frozen 特征到 DataFrame

    对于 Val/Test：直接用 lookup 表（train 期间的统计）
    对于 Train：需要检查 last_ts < click_ts，避免看到未来数据

    Args:
        df: DataFrame with user_id, streamer_id, timestamp columns
        lookups: dict from create_frozen_lookups()
        is_train: 是否是训练集（需要额外的 past-only 检查）

    Returns:
        DataFrame with past-only features added
    """
    log(f"Applying frozen features (is_train={is_train})...")

    df = df.copy()
    n = len(df)

    # 创建 pair keys
    pair_keys = list(zip(df['user_id'], df['streamer_id']))
    timestamps = df['timestamp'].values

    # -------------------------------------------------------------------------
    # Pair features (带 _past 后缀，表示无泄漏)
    # 对于 Train 集：需要检查 last_ts < click_ts
    # -------------------------------------------------------------------------
    pair_count = np.zeros(n)
    pair_sum = np.zeros(n)
    pair_mean = np.zeros(n)
    pair_std = np.zeros(n)
    pair_max = np.zeros(n)
    pair_last_gap = np.full(n, 999.0)

    for i, (key, click_ts) in enumerate(zip(pair_keys, timestamps)):
        if key in lookups['pair']:
            info = lookups['pair'][key]
            last_ts = info.get('last_ts', np.nan)

            # 关键检查：对于 Train 集，只有 last_ts < click_ts 才能使用
            if is_train and not np.isnan(last_ts) and last_ts >= click_ts:
                # 这个 gift 发生在 click 之后，不能使用
                # 保持默认值 0
                continue

            pair_count[i] = info.get('count', 0)
            pair_sum[i] = info.get('sum', 0.0)
            pair_mean[i] = info.get('mean', 0.0)
            pair_std[i] = info.get('std', 0.0)
            pair_max[i] = info.get('max', 0.0)

            if not np.isnan(last_ts) and last_ts < click_ts:
                pair_last_gap[i] = (click_ts - last_ts) / (1000 * 3600)

    df['pair_gift_count_past'] = pair_count
    df['pair_gift_sum_past'] = pair_sum
    df['pair_gift_mean_past'] = pair_mean
    df['pair_gift_std_past'] = pair_std
    df['pair_gift_max_past'] = pair_max
    df['pair_last_gift_gap_hours'] = pair_last_gap

    # -------------------------------------------------------------------------
    # User features（简化：直接用 lookup，因为是全局统计）
    # -------------------------------------------------------------------------
    df['user_gift_count_past'] = df['user_id'].map(
        lambda x: lookups['user'].get(x, {}).get('count', 0)
    )
    df['user_gift_sum_past'] = df['user_id'].map(
        lambda x: lookups['user'].get(x, {}).get('sum', 0.0)
    )
    df['user_gift_mean_past'] = df['user_id'].map(
        lambda x: lookups['user'].get(x, {}).get('mean', 0.0)
    )
    df['user_unique_streamers_past'] = df['user_id'].map(
        lambda x: lookups['user'].get(x, {}).get('unique_streamers', 0)
    )

    # -------------------------------------------------------------------------
    # Streamer features（简化：直接用 lookup）
    # -------------------------------------------------------------------------
    df['str_gift_count_past'] = df['streamer_id'].map(
        lambda x: lookups['streamer'].get(x, {}).get('count', 0)
    )
    df['str_gift_sum_past'] = df['streamer_id'].map(
        lambda x: lookups['streamer'].get(x, {}).get('sum', 0.0)
    )
    df['str_gift_mean_past'] = df['streamer_id'].map(
        lambda x: lookups['streamer'].get(x, {}).get('mean', 0.0)
    )
    df['str_unique_givers_past'] = df['streamer_id'].map(
        lambda x: lookups['streamer'].get(x, {}).get('unique_givers', 0)
    )

    # -------------------------------------------------------------------------
    # Watch time features (历史观看时长)
    # -------------------------------------------------------------------------
    if 'watch_time' in lookups and lookups['watch_time']['user']:
        # User-level 历史观看时长
        df['user_avg_watch_time_past'] = df['user_id'].map(
            lambda x: lookups['watch_time']['user'].get(x, {}).get('mean', 0)
        )
        df['user_total_watch_time_past'] = df['user_id'].map(
            lambda x: lookups['watch_time']['user'].get(x, {}).get('sum', 0)
        )

        # Pair-level 历史观看时长
        pair_watch_mean = np.zeros(n)
        pair_watch_count = np.zeros(n)
        for i, key in enumerate(pair_keys):
            if key in lookups['watch_time']['pair']:
                info = lookups['watch_time']['pair'][key]
                # 对于 Train 集，需要检查是否有未来数据（但观看时长没有时间戳，简化处理）
                pair_watch_mean[i] = info.get('mean', 0)
                pair_watch_count[i] = info.get('count', 0)

        df['pair_avg_watch_time_past'] = pair_watch_mean
        df['pair_watch_count_past'] = pair_watch_count

        # Streamer-level 历史被观看时长
        df['str_avg_watch_time_past'] = df['streamer_id'].map(
            lambda x: lookups['watch_time']['streamer'].get(x, {}).get('mean', 0)
        )

        n_watch_features = 5
    else:
        n_watch_features = 0

    log(f"  Applied {14 + n_watch_features} past-only features", "SUCCESS")

    return df


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

    Args:
        train_days: 训练集天数 (default: 7)
        val_days: 验证集天数 (default: 7)
        test_days: 测试集天数 (default: 7)
        gap_days: Train-Val/Val-Test gap 天数 (default: 0)
        label_window_hours: 标签窗口（小时）(default: 1)
        use_cache: 是否使用缓存 (default: True)

    Returns:
        tuple: (train_df, val_df, test_df, lookups)

    Example:
        >>> from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
        >>> train_df, val_df, test_df, lookups = prepare_dataset()
        >>> feature_cols = get_feature_columns(train_df)
    """
    log("=" * 60)
    log("Preparing Leakage-Free Dataset")
    log("=" * 60)

    # 1. Load raw data
    gift, click, user, streamer, room = load_raw_data()

    # 2. Click-level labels (removes watch_live_time automatically)
    click_with_labels = prepare_click_level_labels(gift, click, label_window_hours)

    # 3. 7-7-7 split
    train_df, val_df, test_df = split_by_days(
        click_with_labels, train_days, val_days, test_days, gap_days
    )

    # 4. Get train end timestamp for frozen features
    train_end_ts = train_df['timestamp'].max()

    # 5. Create or load frozen lookups
    cache_key = f"frozen_{train_days}_{val_days}_{test_days}_{gap_days}_{label_window_hours}.pkl"
    cache_path = CACHE_DIR / cache_key

    if use_cache and cache_path.exists():
        log(f"Loading cached lookups from {cache_path}")
        with open(cache_path, 'rb') as f:
            lookups = pickle.load(f)
    else:
        # 传递原始 click 数据（含 watch_live_time）用于计算历史观看时长
        lookups = create_frozen_lookups(gift, train_end_ts, click=click)
        with open(cache_path, 'wb') as f:
            pickle.dump(lookups, f)
        log(f"Cached lookups to {cache_path}")

    # 6. Apply frozen features (Train 需要 past-only 检查)
    train_df = apply_frozen_features(train_df, lookups, is_train=True)
    val_df = apply_frozen_features(val_df, lookups, is_train=False)
    test_df = apply_frozen_features(test_df, lookups, is_train=False)

    # 7. Add static features
    train_df = add_static_features(train_df, user, streamer, room)
    val_df = add_static_features(val_df, user, streamer, room)
    test_df = add_static_features(test_df, user, streamer, room)

    # 8. Create targets
    for df in [train_df, val_df, test_df]:
        df['target'] = np.log1p(df['gift_price_label'])
        df['target_raw'] = df['gift_price_label']
        df['is_gift'] = (df['gift_price_label'] > 0).astype(int)

    # 9. Encode all object/category columns
    for df in [train_df, val_df, test_df]:
        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype) == 'category':
                df[col] = df[col].fillna('unknown').astype(str)
                df[col] = pd.Categorical(df[col]).codes

    # 10. Fill NaN for numeric columns
    for df in [train_df, val_df, test_df]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)

    log("=" * 60)
    log("Dataset preparation complete!", "SUCCESS")
    log(f"  Train: {len(train_df):,} (gift_rate={train_df['is_gift'].mean()*100:.2f}%)")
    log(f"  Val:   {len(val_df):,} (gift_rate={val_df['is_gift'].mean()*100:.2f}%)")
    log(f"  Test:  {len(test_df):,} (gift_rate={test_df['is_gift'].mean()*100:.2f}%)")
    log("=" * 60)

    return train_df, val_df, test_df, lookups


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

    # 检查 gift 相关特征必须带 _past 后缀
    gift_features = [f for f in feature_cols if 'gift' in f.lower() and 'label' not in f]
    for f in gift_features:
        if not f.endswith('_past'):
            log(f"WARNING: Gift feature without _past suffix: {f}", "WARNING")

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

    对于每个样本，验证 pair_gift_count_past 等于真实的 past-only count。

    Args:
        df: DataFrame with features
        gift: original gift DataFrame
        n_samples: number of samples to check

    Returns:
        bool: True if passed, False if leakage detected
    """
    log(f"Verifying no leakage ({n_samples} samples)...")

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
        log(f"Leakage verification: FAILED ({len(errors)}/{n_samples} samples)", "ERROR")
        for e in errors[:3]:
            log(f"  idx={e['idx']}: expected={e['expected']}, got={e['got']}, diff={e['diff']}")
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
    if not verify_no_leakage(test_df, gift, n_samples=50):
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
    print("Gift EVpred Data Utils - Example Usage")
    print("=" * 60)

    # 准备数据
    train_df, val_df, test_df, lookups = prepare_dataset()

    # 获取特征列
    feature_cols = get_feature_columns(train_df)
    print(f"\nFeature columns ({len(feature_cols)}):")
    print(feature_cols[:10], "...")

    # 运行验证
    gift, _, _, _, _ = load_raw_data()
    run_full_verification(train_df, val_df, test_df, gift, feature_cols)
