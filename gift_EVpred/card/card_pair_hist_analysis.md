# 知识卡片: pair_gift_cnt_hist 特征分析

> **ID**: KC-20260119-pair_hist
> **作者**: Viska Wei
> **日期**: 2026-01-19
> **标签**: 特征工程, 泄漏验证, 首次打赏

---

## 1. 核心问题

`pair_gift_cnt_hist`（用户-主播历史打赏次数）是模型最强特征（系数 1.35，贡献 29.5%），**是否存在数据泄漏？**

---

## 2. 泄漏验证

### 2.1 验证方法

```python
# Day-Frozen 设计：只用 day < click_day 的历史
click_with_pair = pd.merge_asof(
    click_sorted,
    pair_day[...],
    on='day',
    by=['user_id', 'streamer_id'],
    direction='backward',
    allow_exact_matches=False  # 严格 < 当前天
)
```

### 2.2 验证结果

| 检查项 | 结果 |
|--------|------|
| 随机抽样 10 个打赏样本 | ✅ 10/10 通过 |
| feature_count == true_past_days | ✅ 完全一致 |
| 同一天的 gift 是否被计入 | ✅ 未计入 |

**结论：没有数据泄漏**

---

## 3. 特征价值分析

### 3.1 消融实验

| 指标 | 完整模型 | 去掉 pair_* | 差异 |
|------|----------|-------------|------|
| RevCap@1% | 51.4% | 38.4% | **-12.9pp** |
| Whale Recall@1% | 33.5% | 18.4% | **-15.1pp** |
| Whale Precision@1% | 5.1% | 2.8% | -2.3pp |
| Avg Revenue@1% | 64.1 | 48.0 | -16.1 |
| Gift Rate@1% | 9.2% | 4.8% | -4.4pp |
| CV (稳定性) | 9.4% | 26.6% | **+17.2pp** |

**`pair_*` 特征贡献了 ~13pp 的 RevCap 提升**

### 3.2 为什么这个特征这么强？

| 分组 | 样本占比 | 打赏率 | 比值 |
|------|---------|--------|------|
| 有历史 (pair_hist > 0) | 4.2% | 11.05% | 基准 |
| 无历史 (pair_hist = 0) | 95.8% | 1.11% | 1/10 |

**如果用户之前给这个主播打赏过，再次打赏的概率是 10 倍**

---

## 4. 关键发现：首次打赏问题

### 4.1 收入分布

| 分组 | 样本占比 | 收入占比 | RevCap@1% | Whale Recall@1% |
|------|---------|---------|-----------|-----------------|
| 有历史 (复购) | 4.2% | 43.9% | 35.1% | 18.3% |
| 无历史 (首次) | **95.8%** | **56.1%** | **29.6%** | 9.8% |

### 4.2 问题分析

1. **56% 的收入来自"首次打赏"用户**
2. 但模型对首次打赏的预测能力弱（RevCap 29.6% vs 复购 35.1%）
3. 模型过度依赖"复购"信号

### 4.3 Whale 阈值差异

| 分组 | Whale 阈值 (P90) | Whale 数量 |
|------|-----------------|-----------|
| 有历史 | 199 元 | 671 |
| 无历史 | 62 元 | 1,516 |

**首次打赏的 whale 门槛更低**（62 元 vs 199 元），但数量更多

---

## 5. 结论

### 5.1 泄漏问题

| 结论 | 说明 |
|------|------|
| ✅ **无泄漏** | Day-Frozen 设计正确，10/10 验证通过 |
| ✅ **合理信号** | 历史打赏是复购预测的核心信号 |

### 5.2 局限性

| 问题 | 影响 |
|------|------|
| 过度依赖复购信号 | 56% 收入来自首次打赏，预测能力不足 |
| 冷启动问题 | 新 pair 无历史，只能依赖 user/streamer 级特征 |

### 5.3 优化方向

| 方向 | 预期收益 |
|------|----------|
| 增强 user_gift_*_hist | 用户级复购信号 |
| 增强 str_gift_*_hist | 主播吸金能力信号 |
| 用户-主播匹配特征 | 相似用户历史行为 |
| 首次打赏专属模型 | 针对 cold-start 场景 |

---

## 6. 数据支撑

### 6.1 完整模型指标

```
RevCap@1%: 51.4%
Whale Recall@1%: 33.5%
Whale Precision@1%: 5.1%
Avg Revenue@1%: 64.1 yuan
Gift Rate@1%: 9.2%
CV: 9.4%
95% CI: [47.3%, 54.3%]
```

### 6.2 首次打赏分组指标

```
无历史 pair (95.8% 样本, 56.1% 收入):
  RevCap@1%: 29.6%
  Whale Recall@1%: 9.8%
  Whale Precision@1%: 1.1%
  打赏率: 1.11%
```

---

## 7. 附录：验证脚本

```python
# 验证 pair_gift_cnt_hist 无泄漏
for idx, row in test_with_gift.iterrows():
    click_day = pd.to_datetime(row['timestamp'], unit='ms').normalize()

    # 计算真实的 day < click_day 的历史
    true_past = gift[
        (gift['user_id'] == row['user_id']) &
        (gift['streamer_id'] == row['streamer_id']) &
        (gift['day'] < click_day)
    ]

    assert row['pair_gift_cnt_hist'] == len(true_past)  # ✅ 通过
```
