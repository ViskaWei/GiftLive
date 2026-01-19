# 🎯 Baseline 7-7-7 Day Split 配置说明

> **Date:** 2026-01-18  
> **Author:** Viska Wei  
> **Script:** `scripts/train_baseline_7days.py`  
> **Status:** 🚀 **运行中**

---

## 📋 关键改进

### 1. 时间切分：7-7-7 天

**之前**：70-15-15 比例切分（按样本数）
**现在**：7-7-7 天切分（按日期）

```python
# 切分逻辑
test_start = max_date - timedelta(days=6)   # 最后 7 天
val_start = test_start - timedelta(days=7)  # 测试前 7 天
train_end = val_start - timedelta(days=1)   # 训练结束

train = df[df['date'] < val_start]          # 训练集：前 N 天
val = df[(df['date'] >= val_start) & (df['date'] < test_start)]  # 验证集：中间 7 天
test = df[df['date'] >= test_start]          # 测试集：最后 7 天
```

**优势**：
- ✅ 更符合实际业务场景（按天切分）
- ✅ 避免时间重叠
- ✅ 更容易理解和复现

### 2. Precompute Frozen Features（确保无泄漏）

**核心原则**：
1. **只用训练集窗口统计**：所有 frozen features 只从 `train_df` 的时间窗口内计算
2. **保存为 Lookup 表**：统计结果保存为字典（pair/user/streamer lookup）
3. **所有数据共享同一个 Lookup**：train/val/test 都用同一个 lookup 表（但 lookup 只从训练集计算）

**实现细节**：
```python
# Step 1: 只用训练集窗口计算 lookup
lookups = create_frozen_features(gift, click, click_train)
# → 只使用 gift_train 窗口内的数据

# Step 2: 为所有数据应用同一个 lookup
train_df = prepare_features(..., click_train, click_train, lookups)  # train 用 train lookup
val_df = prepare_features(..., click_val, click_train, lookups)      # val 用 train lookup
test_df = prepare_features(..., click_test, click_train, lookups)   # test 用 train lookup
```

**无泄漏保证**：
- ✅ `create_frozen_features()` 只使用 `train_df` 的时间窗口
- ✅ val/test 数据只查 lookup 表，不重新计算
- ✅ 代码中有验证逻辑，检查时间窗口不重叠

### 3. 验证无泄漏

**代码中的验证**：
```python
# 验证时间窗口
train_min_ts = click_train['timestamp'].min()
train_max_ts = click_train['timestamp'].max()
val_min_ts = click_val['timestamp'].min()
test_min_ts = click_test['timestamp'].min()

# 检查无重叠
if train_max_ts >= val_min_ts or val_max_ts >= test_min_ts:
    log_message("⚠️ WARNING: Time overlap detected!", "WARNING")
else:
    log_message("✅ No time overlap (correct split)", "SUCCESS")
```

---

## 🔧 配置参数

```python
CONFIG = {
    'label_window_hours': 0.25,  # 15 minutes
    'train_days': 7,              # Train: 7 days
    'val_days': 7,                # Val: 7 days
    'test_days': 7,               # Test: 7 days
    'feature_version': 'frozen',  # Frozen (strict no-leakage)
    'use_optimized': True,        # Use optimized version
    'cache_lookups': True,        # Cache lookups for reuse
}
```

---

## 📊 预期结果

### 数据量

假设数据时间跨度约 21 天：
- **Train**: 前 7 天
- **Val**: 中间 7 天
- **Test**: 最后 7 天

### 性能指标

预期与之前的 baseline 类似：
- **Spearman**: ~0.09-0.10
- **Top-1% Capture**: ~11-12%
- **Revenue Capture@1%**: ~21-22%
- **AUC**: ~0.56-0.57

---

## 🚀 运行状态

**启动命令**：
```bash
cd /home/swei20/GiftLive
source init.sh
nohup python scripts/train_baseline_7days.py > logs/baseline_7days_$(date +%Y%m%d).log 2>&1 &
```

**监控**：
```bash
# 查看进程
ps aux | grep train_baseline_7days

# 查看日志
tail -f logs/baseline_7days_20260118.log
```

**输出文件**：
- 结果 JSON: `gift_EVpred/results/baseline_7days_20260118.json`
- 模型文件: `gift_EVpred/models/baseline_7days_20260118.pkl`
- Lookup 缓存: `gift_EVpred/features_cache/frozen_lookups_7days_*.pkl`

---

## ✅ 关键保证

1. **无泄漏**：
   - ✅ 所有 frozen features 只用训练集窗口计算
   - ✅ val/test 只查 lookup 表，不重新计算
   - ✅ 代码中有时间窗口验证

2. **Precompute**：
   - ✅ 所有数据（train/val/test）使用同一个 precomputed lookup
   - ✅ Lookup 缓存到磁盘，可复用

3. **7-7-7 切分**：
   - ✅ 按日期切分，不是按样本数
   - ✅ 确保时间不重叠
   - ✅ 更符合实际业务场景

---

> **状态**: 🚀 运行中  
> **最后更新**: 2026-01-18 16:05
