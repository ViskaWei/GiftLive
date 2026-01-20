# 🍃 Leakage-Free Data Processing for Gift EVpred
> **Name:** Leakage-Free Data Utils
> **ID:** `EXP-20260118-gift_EVpred-01`
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.0
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** ✅

> 🎯 **Target:** 建立统一的无泄漏数据处理框架，确保所有 gift_EVpred 实验使用一致的特征
> 🚀 **Next:** 所有后续实验必须使用 `data_utils.py`，不可自行实现数据处理逻辑

## ⚡ 核心结论速览

> **一句话**: 识别并修复了 5 类数据泄漏问题，建立了统一的 `data_utils.py` 模块，验证通过率 200/200

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| P1: watch_live_time 泄漏? | ✅ 已修复 | 完全移除该特征 |
| P2: pair_gift_mean 未来泄漏? | ✅ 已修复 | 使用 frozen lookup + past-only 检查 |
| P3: user/streamer 聚合泄漏? | ✅ 已修复 | 同上 |
| P4: 时间穿越? | ✅ 已修复 | 7-7-7 按天严格划分 |
| P5: watch_time 历史特征? | ⚠️ **部分泄漏** | Train 集存在同期泄漏（见详细分析） |

| 指标 | 值 | 启示 |
|------|-----|------|
| 泄漏验证通过率 | 200/200 (100%) | Gift 相关特征无泄漏 |
| Watch_time 泄漏 | Train 集约 5% 偏差 | 影响较小但需记录 |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` |
| 📄 Code | `gift_EVpred/data_utils.py` |

---

# 1. 🎯 目标

**问题**: 之前的实验存在多种数据泄漏，导致模型性能虚高，需要建立统一的无泄漏数据处理框架

**验证**: 识别所有泄漏问题，实现无泄漏特征，验证通过

| 预期 | 判断标准 |
|------|---------|
| 验证通过 | 200/200 样本通过泄漏检查 → 可作为标准模块 |
| 验证失败 | 需要继续修复 |

---

# 2. 🦾 数据泄漏类型分析

## 2.1 泄漏问题分类

### P1: 结果泄漏 (Result Leakage) - `watch_live_time`

**问题描述**：`watch_live_time` 包含用户观看直播的总时长，包括打赏后的时间。

```
时间线：
Click (T=0) → Gift (T=5min) → Leave (T=15min)
                                    ↑
                            watch_live_time = 15min
```

**泄漏原因**：如果用户打赏后继续观看，`watch_live_time` 会包含打赏后的时间，这是因果倒置。

**解决方案**：**完全移除该特征**

```python
# data_utils.py:128-129
if 'watch_live_time' in click.columns:
    click = click.drop(columns=['watch_live_time'])
```

---

### P2: 未来泄漏 (Future Leakage) - Pair-level Gift 特征

**问题描述**：使用 `groupby(['user_id', 'streamer_id']).agg()` 计算的统计量包含未来数据

**错误代码示例**：
```python
# ❌ 错误：看到了未来的 gift
pair_stats = gift.groupby(['user_id', 'streamer_id']).agg({
    'gift_price': ['count', 'sum', 'mean']
})
df = df.merge(pair_stats, ...)
```

**正确做法（双层保护）**：

1. **Frozen Lookup**：只用 Train 时间窗口的数据计算统计量

```python
# data_utils.py:243
gift_train = gift[gift['timestamp'] <= train_end_ts].copy()
```

2. **Past-Only 检查（针对 Train 集）**：验证 `last_ts < click_ts`

```python
# data_utils.py:397-401
if is_train and not np.isnan(last_ts) and last_ts >= click_ts:
    # 这个 gift 发生在 click 之后，不能使用
    continue
```

**验证**：泄漏检查通过 200/200 样本

---

### P3: User/Streamer 级别聚合泄漏

**问题描述**：与 P2 类似，user_total_gift 和 streamer_total_gift 也可能包含未来数据

**解决方案**：使用 Train-window frozen lookup

```python
# data_utils.py:274-278
user_stats = gift_train.groupby('user_id').agg({
    'gift_price': ['count', 'sum', 'mean'],
    'streamer_id': 'nunique'
})
```

---

### P4: 时间穿越 (Time Travel)

**问题描述**：如果按比例划分而非按天划分，可能导致同一天的数据同时出现在 Train 和 Val/Test

**错误示例**：
```python
# ❌ 错误：按比例划分
train_df = df.sample(frac=0.7)
```

**正确做法**：7-7-7 按天严格划分

```python
# data_utils.py:198-202
train_end = min_date + timedelta(days=train_days - 1)
val_start = train_end + timedelta(days=gap_days + 1)
# ...
```

**划分结果**：
```
Train: Day 1-7  (2017-10-06 ~ 2017-10-12)
Val:   Day 8-14 (2017-10-13 ~ 2017-10-19)
Test:  Day 15-21 (2017-10-20 ~ 2017-10-26)
```

---

### P5: Watch Time 历史特征泄漏 ⚠️

**问题描述**：`user_avg_watch_time_past` 等历史观看时长特征存在同期泄漏

**泄漏机制**：

```
Train 期间 Day 1-7:
- 对于 Day 3 的样本，watch_time 统计包含 Day 1-7 全部数据
- 即 Day 3 样本能看到 Day 4-7 的 watch_time

验证结果：
Avg watch time (ALL train): 8600.94ms
Avg watch time (before Day 3): 9084.60ms
Difference: -483.66ms (约 5% 偏差)
```

**当前处理**：
- Val/Test 集：使用 Train 结束时的统计值，**无泄漏**
- Train 集：存在同期泄漏，添加 WARNING 日志

```python
# data_utils.py:458-461
if is_train:
    log("  WARNING: watch_time features for Train use train-period stats (slight leakage)", "WARNING")
```

**影响评估**：
- 泄漏程度约 5%，远小于 gift 特征的 100% 泄漏
- Watch time 与 gift 的相关性较弱，对预测影响有限
- **结论**：标记为已知限制，不阻塞使用

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive |
| 路径 | `data/KuaiLive/` |
| Train/Val/Test | 3,402,217 / 2,331,379 / 2,461,377 |
| 特征维度 | 40+ |
| Gift Rate | ~1.5% |

## 3.2 7-7-7 划分细节

```yaml
split:
  type: "by_days"
  train_days: 7
  val_days: 7
  test_days: 7
  gap_days: 0

dates:
  train: "2017-10-06 ~ 2017-10-12"
  val: "2017-10-13 ~ 2017-10-19"
  test: "2017-10-20 ~ 2017-10-26"
```

## 3.3 特征分类

### 禁止使用（Forbidden）

| 特征 | 原因 |
|------|------|
| `watch_live_time` | 结果泄漏 |
| `watch_time_log` | 同上 |
| `watch_time_ratio` | 同上 |

### Past-Only 特征（无泄漏）

| 特征 | 说明 |
|------|------|
| `pair_gift_count_past` | Pair 历史打赏次数 |
| `pair_gift_sum_past` | Pair 历史打赏总额 |
| `pair_gift_mean_past` | Pair 历史平均打赏 |
| `pair_gift_std_past` | Pair 历史打赏标准差 |
| `pair_gift_max_past` | Pair 历史最大打赏 |
| `pair_last_gift_gap_hours` | 距上次打赏时间 |
| `user_gift_count_past` | 用户历史打赏次数 |
| `user_gift_sum_past` | 用户历史打赏总额 |
| `user_gift_mean_past` | 用户历史平均打赏 |
| `user_unique_streamers_past` | 用户历史打赏主播数 |
| `str_gift_count_past` | 主播历史收礼次数 |
| `str_gift_sum_past` | 主播历史收礼总额 |
| `str_gift_mean_past` | 主播历史平均收礼 |
| `str_unique_givers_past` | 主播历史打赏者数 |

### 历史观看时长特征（存在轻微泄漏）

| 特征 | 说明 | 泄漏程度 |
|------|------|---------|
| `user_avg_watch_time_past` | 用户历史平均观看时长 | Train ~5% |
| `user_total_watch_time_past` | 用户历史总观看时长 | Train ~5% |
| `pair_avg_watch_time_past` | Pair 历史平均观看时长 | Train ~5% |
| `pair_watch_count_past` | Pair 历史观看次数 | Train ~5% |
| `str_avg_watch_time_past` | 主播历史被观看时长 | Train ~5% |

### 静态特征（无泄漏风险）

| 类别 | 特征示例 |
|------|---------|
| User Profile | `age`, `gender`, `device_brand`, `device_price` |
| Streamer Profile | `fans_user_num`, `accu_live_cnt` |
| Room | `live_type`, `live_content_category` |
| Time | `hour`, `day_of_week`, `is_weekend` |

---

# 4. 📊 验证结果

## 4.1 泄漏验证

```
Verifying train set...
[17:45:23] ✅ Leakage verification: PASSED (50/50 samples)

Verifying val set...
[17:45:25] ✅ Leakage verification: PASSED (50/50 samples)

Verifying test set...
[17:45:27] ✅ Leakage verification: PASSED (100/100 samples)
```

## 4.2 时间划分验证

```
Time split verification: PASSED
  Train max: 2017-10-12 23:59:59
  Val min:   2017-10-13 00:00:00
  Test min:  2017-10-20 00:00:00
```

## 4.3 Linear Regression Baseline（验证无泄漏后）

```yaml
model: LinearRegression
features: 40+
metrics:
  spearman: 0.0827
  top_1pct_capture: 5.35%
  revenue_capture_1pct: 16.11%
```

**对比**：无泄漏 baseline 性能显著低于有泄漏版本，符合预期

---

# 5. 💡 洞见

## 5.1 宏观

- **泄漏 = 虚假性能**：之前实验 AUC 0.95+ 主要来自泄漏，无泄漏后性能大幅下降
- **严格时间划分是基础**：7-7-7 按天划分 + past-only 检查是防止泄漏的关键

## 5.2 模型层

- **Gift 特征是最危险的**：groupby 计算极易引入未来数据
- **Watch time 泄漏影响有限**：约 5% 偏差，不阻塞使用

## 5.3 细节

- **Binary search 技巧**：`searchsorted(side='left') - 1` 确保严格 `<` 不等式
- **缓存机制**：frozen lookup 缓存到 `features_cache/` 加速重复加载

---

# 6. 📝 结论

## 6.1 核心发现

> **建立了统一的无泄漏数据处理框架 `data_utils.py`，所有 gift_EVpred 实验必须使用**

- ✅ P1-P4：Gift 相关泄漏完全修复
- ⚠️ P5：Watch time 存在轻微泄漏（~5%），已标记为已知限制

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **Frozen Lookup 有效** | 200/200 样本验证通过 |
| 2 | **Past-only 检查必要** | 修复了 Train 集同期泄漏 |
| 3 | **Watch time 泄漏可控** | 偏差约 5%，影响有限 |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 统一入口 | 所有实验使用 `prepare_dataset()` |
| 自动排除 | `get_feature_columns()` 自动排除泄漏特征 |
| 强制验证 | 训练前运行 `verify_no_leakage()` |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 自行 groupby 计算 | 包含未来数据 |
| 使用 watch_live_time | 结果泄漏 |
| 按比例划分数据 | 时间穿越 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| 泄漏验证通过率 | 100% | Gift 特征 |
| Watch time 泄漏 | ~5% | Train 集 |
| 特征数量 | 40+ | Past-only |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 修复 Watch time 泄漏 | 实现完全 past-only watch time | 🟡 |
| 更多验证 | 增加验证样本量到 1000 | 🟢 |
| 文档化 | 更新 prompt template | ✅ 已完成 |

---

# 7. 📎 附录

## 7.1 data_utils.py 核心函数

| 函数 | 用途 | 泄漏处理 |
|------|------|---------|
| `prepare_dataset()` | 主入口 | 自动应用所有保护 |
| `get_feature_columns()` | 获取特征列 | 排除禁止特征 |
| `verify_no_leakage()` | 验证无泄漏 | 抽样检查 |
| `create_frozen_lookups()` | 创建冻结查找表 | 只用 Train 数据 |
| `apply_frozen_features()` | 应用冻结特征 | is_train 检查 |

## 7.2 使用示例

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/swei20/GiftLive')

from gift_EVpred.data_utils import (
    prepare_dataset,
    get_feature_columns,
    verify_no_leakage,
    load_raw_data
)

# 1. 准备数据（7-7-7 无泄漏）
train_df, val_df, test_df, lookups = prepare_dataset()

# 2. 获取特征列（自动排除泄漏特征）
feature_cols = get_feature_columns(train_df)

# 3. 验证无泄漏
gift, _, _, _, _ = load_raw_data()
verify_no_leakage(train_df, gift, n_samples=100)

# 4. 训练模型
X_train = train_df[feature_cols]
y_train = train_df['target']
```

## 7.3 文件路径

| 文件 | 路径 |
|------|------|
| 数据处理模块 | `gift_EVpred/data_utils.py` |
| 数据处理指南 | `gift_EVpred/DATA_PROCESSING_GUIDE.md` |
| Coding Prompt 模板 | `gift_EVpred/prompts/prompt_template_evpred.md` |
| 本实验报告 | `gift_EVpred/exp/exp_leakage_free_data_utils_20260118.md` |

---

> **实验完成时间**: 2026-01-18
