# apply_frozen_features 优化总结

> **Date:** 2026-01-18  
> **Author:** Viska Wei

---

## 问题

原始 `apply_frozen_features` 函数使用 `iterrows()` 逐行处理，对大数据集（490 万条记录）非常慢，预计需要数小时。

**原始实现**（`scripts/train_leakage_free_baseline.py`）：
```python
for idx, row in df.iterrows():  # 慢！
    key = (row['user_id'], row['streamer_id'])
    if key in lookups['pair']:
        # ... 逐行赋值
```

---

## 优化方案

### 1. 使用 merge() 代替 iterrows()

**Pair 特征**：将 lookup 表转换为 DataFrame，使用 `merge()` 匹配
**Streamer 特征**：同样使用 `merge()` 代替 `iterrows()`
**User 特征**：已使用 `map()`，无需优化

### 2. 向量化时间间隔计算

使用 pandas 的向量化操作计算 `pair_last_gift_time_gap_past`

### 3. 实现 Lookup 表缓存

支持保存/加载 frozen lookups，避免重复计算

---

## 优化结果

**测试环境**：100K 样本

| 版本 | 时间 | 速度提升 |
|------|------|---------|
| 原始版本 | 7.29s | - |
| 优化版本 | 0.07s | **107.83x** |

**验证**：所有特征值完全匹配（差异 < 1e-6）

---

## 实现

**优化函数**：`scripts/optimize_apply_frozen_features.py`

**关键函数**：
- `apply_frozen_features_optimized()`：优化版本
- `save_frozen_lookups()`：保存 lookup 表
- `load_frozen_lookups()`：加载 lookup 表

**缓存位置**：`gift_EVpred/features_cache/frozen_lookups_*.pkl`

---

## 使用建议

1. **大数据集（>1M 行）**：使用优化版本 `apply_frozen_features_optimized()`
2. **相同 train window**：可复用保存的 lookup 表，避免重复计算
3. **小数据集（<100K 行）**：原始版本也可接受

---

## 性能估算

对于 490 万条记录：
- 原始版本：预计 ~6 小时（基于 100K 样本外推）
- 优化版本：预计 ~3.5 分钟（基于 100K 样本外推）

**建议**：所有后续实验使用优化版本。
