# Strict vs Benchmark 特征模式对比实验

> **实验 ID**: strict_vs_benchmark_20260119
> **作者**: Viska Wei
> **日期**: 2026-01-19
> **状态**: Completed

---

## 1. 背景与动机

在 `data_utils.py` 的 Strict Mode 改造中，我们移除了 11 个来自 KuaiLive 快照的特征（`fans_num`, `follow_num`, `accu_*` 等）。这些特征存在 **时间泄漏风险**：

- KuaiLive 数据集的用户/主播表是 **2024-05-25 的快照**
- 但 click/gift 数据跨越多天，使用 May 25 的 `fans_num` 预测 May 20 的 gift 会造成 look-ahead bias

**问题**：移除这些特征后，RevCap@1% 几乎没有变化（51.4% → 51.4%），为什么？

**本实验目标**：
1. 对比 Strict（20 特征）vs Benchmark（31 特征）的模型性能
2. 分析快照特征的实际贡献度
3. 解释为什么移除快照特征不影响预测效果

---

## 2. 方法

### 2.1 实验配置

| 配置项 | Strict Mode | Benchmark Mode |
|--------|-------------|----------------|
| 模型 | Ridge (alpha=1.0) | Ridge (alpha=1.0) |
| 目标 | raw_y (原始金额) | raw_y (原始金额) |
| 归因窗口 | 1 分钟 | 1 分钟 |
| 特征数 | 20 | 31 |
| 快照特征 | ❌ 不包含 | ✅ 包含 |

### 2.2 特征差异

**Benchmark Mode 额外包含的 11 个快照特征**：

| 来源 | 特征名 | 说明 |
|------|--------|------|
| User | `fans_num` | 用户粉丝数（May 25 快照） |
| User | `follow_num` | 用户关注数 |
| User | `accu_watch_live_cnt` | 累计观看直播次数 |
| User | `accu_watch_live_duration` | 累计观看直播时长 |
| Streamer | `str_fans_user_num` | 主播粉丝数 |
| Streamer | `str_follow_user_num` | 主播关注数 |
| Streamer | `str_fans_group_fans_num` | 主播粉丝团人数 |
| Streamer | `str_accu_live_cnt` | 主播累计开播次数 |
| Streamer | `str_accu_live_duration` | 主播累计开播时长 |
| Streamer | `str_accu_play_cnt` | 主播累计播放次数 |
| Streamer | `str_accu_play_duration` | 主播累计播放时长 |

**⚠️ 泄漏风险说明**：
- 这些特征来自 May 25 的快照
- 用于预测 May 18-24 的 click，存在 1-7 天的 look-ahead bias
- 例如：用 May 25 的 `fans_num=1000` 预测 May 20 的 click（当时可能只有 800 粉丝）

---

## 3. 实验结果

### 3.1 模型性能对比

| 指标 | Strict (20 features) | Benchmark (31 features) | 差异 |
|------|---------------------|------------------------|------|
| **RevCap@1%** | **51.36%** | **51.36%** | **0.00pp** |
| RevCap@0.5% | 43.35% | 43.35% | 0.00pp |
| RevCap@2% | 56.33% | 56.33% | 0.00pp |
| **CV (稳定性)** | **9.38%** | **9.41%** | **-0.03pp** |

### 3.2 快照特征重要性分析

从 Benchmark 模型中提取快照特征的系数排名：

| 特征 | 排名 | |coef| | 贡献度 |
|------|------|-------|--------|
| `accu_watch_live_duration` | 6 | 0.216 | 4.7% |
| `fans_num` | 7 | 0.204 | 4.5% |
| `str_accu_live_duration` | 9 | 0.172 | 3.8% |
| `accu_watch_live_cnt` | 13 | 0.099 | 2.2% |
| `str_follow_user_num` | 14 | 0.099 | 2.2% |
| `str_accu_live_cnt` | 17 | 0.059 | 1.3% |
| `follow_num` | 18 | 0.051 | 1.1% |
| `str_accu_play_cnt` | 20 | 0.040 | 0.9% |
| `str_fans_user_num` | 21 | 0.038 | 0.8% |
| `str_fans_group_fans_num` | 22 | 0.030 | 0.7% |
| `str_accu_play_duration` | 30 | 0.001 | 0.0% |

**快照特征总贡献**：22.6% of Σ|coef|

### 3.3 Top 5 特征（Benchmark 模型）

| 排名 | 特征 | |coef| | 类型 |
|------|------|-------|------|
| 1 | `pair_gift_cnt_hist` | 1.35 | Historical |
| 2 | `str_gift_mean_hist` | 0.50 | Historical |
| 3 | `pair_gift_sum_hist` | 0.36 | Historical |
| 4 | `pair_gift_mean_hist` | 0.33 | Historical |
| 5 | `user_gift_sum_hist` | 0.22 | Historical |

---

## 4. 结论

### 4.1 为什么移除快照特征不影响 RevCap？

1. **快照特征贡献度低**：11 个快照特征合计只占 22.6% 的系数权重
2. **历史打赏特征主导**：Top 5 特征全部是 `*_hist` 历史特征，贡献 60%+ 权重
3. **预测信号高度集中**：`pair_gift_cnt_hist`（用户-主播历史打赏次数）单个特征就贡献 29.5%

### 4.2 快照特征的实际价值

虽然快照特征有泄漏风险，但即使包含它们：
- RevCap 没有提升（51.36% vs 51.36%）
- 稳定性没有改善（CV 9.38% vs 9.41%）

这说明：
- **快照特征不是预测关键**
- **用户-主播历史交互才是核心信号**
- 安心使用 Strict Mode，无需担心损失预测能力

### 4.3 推荐配置

| 场景 | 推荐模式 | 理由 |
|------|----------|------|
| **生产环境** | Strict Mode | 避免时间泄漏，保证公平性 |
| **学术研究** | Strict Mode | 消除 look-ahead bias |
| **快速实验** | Benchmark Mode | 可用于快速验证想法（需标注有泄漏风险） |

---

## 5. 数据文件

```
结果文件: gift_EVpred/results/strict_vs_benchmark_20260119.json
Baseline 文件: gift_EVpred/results/baseline_ridge_final_20260119.json
```

---

## 6. 附录：完整特征列表

### Strict Mode (20 features)

```
Historical: pair_gift_cnt_hist, pair_gift_sum_hist, pair_gift_mean_hist,
            user_gift_cnt_hist, user_gift_sum_hist, user_gift_mean_hist,
            str_gift_cnt_hist, str_gift_sum_hist, str_gift_mean_hist
User:       age, gender, device_brand, device_price, is_live_streamer, is_photo_author
Streamer:   live_type, live_content_category
Time:       hour, day_of_week, is_weekend
```

### Benchmark Mode (31 features)

```
= Strict Mode (20) + Snapshot Features (11):
User:       fans_num, follow_num, accu_watch_live_cnt, accu_watch_live_duration
Streamer:   str_fans_user_num, str_follow_user_num, str_fans_group_fans_num,
            str_accu_live_cnt, str_accu_live_duration, str_accu_play_cnt, str_accu_play_duration
```
