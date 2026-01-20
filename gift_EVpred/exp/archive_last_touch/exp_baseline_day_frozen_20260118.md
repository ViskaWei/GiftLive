# 🍃 Day-Frozen Baseline: Direct vs Two-Stage
> **Name:** Day-Frozen Baseline Comparison
> **ID:** `EXP-20260118-gift_EVpred-02`
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.0
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** ✅

> 🎯 **Target:** 建立无泄漏 baseline，对比 Direct Regression 和 Two-Stage 两种建模方案
> 🚀 **Next:** Two-Stage 在 Revenue Capture @1% 上提升 17.8%，推荐作为后续 baseline

## ⚡ 核心结论速览

> **一句话**: Two-Stage 模型在 Revenue Capture @1% 上达到 44.42%，比 Direct Regression 提升 17.8%

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| 无泄漏验证? | ✅ 200/200 通过 | Day-Frozen 特征无泄漏 |
| Direct vs Two-Stage? | Two-Stage +17.8% Rev@1% | Two-Stage 更适合高价值用户挖掘 |

| 指标 | Direct | Two-Stage | 提升 |
|------|--------|-----------|------|
| Spearman | 0.1061 | 0.0854 | -19.5% |
| Top-1% Gift Rate | 24.05% | 17.50% | -27.3% |
| **Revenue Capture @1%** | 37.73% | **44.42%** | **+17.8%** |
| Revenue Capture @5% | 59.84% | 57.54% | -3.8% |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` |
| 📄 Data Utils | `gift_EVpred/data_utils.py` |

---

# 1. 🎯 目标

**问题**: 建立无泄漏的 EV 预测 baseline，对比不同建模方案

**验证**:
- H1: Day-Frozen 特征是否真的无泄漏？
- H2: Two-Stage 是否优于 Direct Regression？

| 预期 | 判断标准 |
|------|---------|
| 无泄漏 | 验证函数 200/200 通过 |
| Two-Stage 更优 | Revenue Capture 提升 > 10% |

---

# 2. 🦾 算法

## 2.1 Direct Regression

直接预测 EV：

$$\hat{y} = f(x) \approx \log(1 + \text{gift\_value})$$

## 2.2 Two-Stage Model

分两阶段预测：

$$\text{EV} = P(\text{gift}>0) \times E[\text{gift} | \text{gift}>0]$$

- **Stage 1**: Logistic Regression 预测 $P(\text{gift}>0)$
- **Stage 2**: Ridge Regression（仅在 gifters 上训练）预测 $E[\text{gift} | \text{gift}>0]$

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive |
| 路径 | `data/KuaiLive/` |
| Train/Val/Test | 1,629,415 / 1,717,199 / 1,409,533 |
| 特征维度 | 31 |
| Gift Rate | Train 1.40%, Test 1.68% |

## 3.2 数据划分 (7-7-7)

```yaml
split:
  type: "by_days"
  train: "2025-05-04 ~ 2025-05-10" (Day 1-7)
  val: "2025-05-11 ~ 2025-05-17" (Day 8-14)
  test: "2025-05-18 ~ 2025-05-24" (Day 15-21)
  gap: 0
```

## 3.3 特征 (Day-Frozen, 无泄漏)

| 类别 | 特征 | 数量 |
|------|------|------|
| Pair History | `pair_gift_cnt/sum/mean_hist` | 3 |
| User History | `user_gift_cnt/sum/mean_hist` | 3 |
| Streamer History | `str_gift_cnt/sum/mean_hist` | 3 |
| User Profile | `age, gender, fans_num, ...` | 10 |
| Streamer Profile | `str_fans_user_num, ...` | 7 |
| Room | `live_type, live_content_category` | 2 |
| Time | `hour, day_of_week, is_weekend` | 3 |

**核心设计**：
- 历史特征只用 `day < current_day` 的数据
- 使用 `pd.merge_asof(..., allow_exact_matches=False)` 实现
- Val/Test 用 Train 结束时的统计（Frozen）

## 3.4 模型

| 模型 | 参数 |
|------|------|
| Direct (Ridge) | alpha=1.0 |
| Two-Stage Clf (LogisticRegression) | C=1.0, max_iter=1000 |
| Two-Stage Reg (Ridge) | alpha=1.0 |

## 3.5 训练

| 参数 | 值 |
|------|-----|
| seed | 42 |
| 标签窗口 | click 后 1h |
| 评估集 | Test (Day 15-21) |

---

# 4. 📊 结果

## 4.1 泄漏验证

```
✅ Time split verification: PASSED
✅ Feature column verification: PASSED
✅ Train leakage verification: PASSED (50/50)
✅ Val leakage verification: PASSED (50/50)
✅ Test leakage verification: PASSED (100/100)
```

## 4.2 模型对比

| 指标 | Direct | Two-Stage | 说明 |
|------|--------|-----------|------|
| Spearman | 0.1061 | 0.0854 | 排序相关性 |
| Top-1% Gift Rate | 24.05% | 17.50% | Top-1%中真实有gift的比例 |
| Top-5% Gift Rate | 11.05% | - | - |
| **Revenue Capture @1%** | 37.73% | **44.42%** | **关键指标** |
| Revenue Capture @5% | 59.84% | 57.54% | - |
| Oracle Rev @1% | 99.54% | 99.54% | 理论上界 |

## 4.3 Two-Stage 分类器性能

| 指标 | Train | Test |
|------|-------|------|
| AUC | 0.6819 | 0.7336 |
| AP | - | 0.0716 |

## 4.4 特征重要性 (Direct)

| 特征 | 系数 |
|------|------|
| pair_gift_cnt_hist | 0.194 |
| live_content_category | -0.012 |
| fans_num | 0.008 |
| gender | -0.006 |
| user_gift_cnt_hist | 0.003 |

---

# 5. 💡 洞见

## 5.1 宏观

- **无泄漏 = 真实性能**：Spearman 只有 0.1，远低于有泄漏时的 0.5+
- **Two-Stage 更适合高价值用户挖掘**：虽然 Spearman 低，但 Revenue Capture 更高

## 5.2 模型层

- **Direct 擅长识别 gifters**：Top-1% Gift Rate 高（24%）
- **Two-Stage 擅长识别 whales**：Revenue Capture 高（44%）

## 5.3 细节

- `pair_gift_cnt_hist` 是最强特征：历史打赏次数是最强信号
- 分类 AUC=0.73 说明"谁会打赏"是可预测的

---

# 6. 📝 结论

## 6.1 核心发现

> **Two-Stage 模型在 Revenue Capture @1% 上达到 44.42%，比 Direct 提升 17.8%**

- ✅ H1: Day-Frozen 特征无泄漏（200/200 通过）
- ✅ H2: Two-Stage 在 Revenue Capture 上优于 Direct

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **Two-Stage 更适合业务** | Rev@1% 44.42% vs 37.73% |
| 2 | **历史打赏是最强信号** | pair_gift_cnt_hist 系数最大 |
| 3 | **无泄漏性能合理** | Spearman ~0.1 符合预期 |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 建模方案 | 推荐 Two-Stage（业务导向） |
| 评估指标 | 使用 Revenue Capture（非 Spearman）|
| 特征工程 | 重点挖掘 pair-level 历史特征 |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 只看 Spearman | 忽略高价值用户捕获能力 |
| 只看 Gift Rate | 忽略金额分布 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Best Revenue Capture @1% | 44.42% | Two-Stage |
| Best Top-1% Gift Rate | 24.05% | Direct |
| 泄漏验证通过率 | 100% | 200 samples |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 模型升级 | LightGBM Two-Stage | 🔴 |
| 特征工程 | 添加历史 watch_time 特征 | 🟡 |
| 评估完善 | 分 slice 分析（新用户/老用户） | 🟡 |

---

# 7. 📎 附录

## 7.1 数值结果

| 配置 | Spearman | Rev@1% | Rev@5% |
|------|----------|--------|--------|
| Direct (Ridge) | 0.1061 | 37.73% | 59.84% |
| Two-Stage (LR+Ridge) | 0.0854 | 44.42% | 57.54% |

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 数据处理 | `gift_EVpred/data_utils.py` |
| 结果文件 | `gift_EVpred/results/baseline_comparison_20260118.json` |
| 缓存文件 | `gift_EVpred/features_cache/day_frozen_features_lw1h.parquet` |

```python
# 使用示例
from gift_EVpred.data_utils import prepare_dataset, get_feature_columns
train_df, val_df, test_df = prepare_dataset()
feature_cols = get_feature_columns(train_df)
```

---

> **实验完成时间**: 2026-01-18
