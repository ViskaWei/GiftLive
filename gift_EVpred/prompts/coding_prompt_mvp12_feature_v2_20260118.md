# 🤖 Coding Prompt: Feature Engineering V2

> **Experiment ID:** `EXP-20260118-gift_EVpred-04`
> **MVP:** MVP-1.2
> **Date:** 2026-01-18
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：探索新特征信号源（序列特征、实时特征、内容匹配度、冷启动泛化），提升无泄漏 baseline 的预测能力

**验证假设**：
- H2.1: 序列特征（最近 N 次打赏金额/间隔）能提升预测力
- H2.2: 实时上下文（watch_time, session 互动）能提升预测力
- H2.3: 内容匹配度（user 偏好 vs streamer 内容）能提升预测力
- H2.4: 组合特征有叠加效果

**预期结果**：
- 若 Top-1% > 30%（vs Baseline 11.5%）→ 新特征方向有效，继续深化
- 若 Top-1% < 20% → 特征信号仍不足，考虑任务降级（二分类）

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  sample_unit: "Click-level（含 0）"
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  split_method: "时间切分（前 70% / 中间 15% / 最后 15%）"
  gift_rate: "~1.5%"
  features_baseline: ~70
  features_target: ~100-120
```

### 2.2 模型

```yaml
model:
  name: "LightGBM"
  architecture: "Direct Regression"
  params:
    objective: regression
    metric: mae
    num_leaves: 31
    learning_rate: 0.05
    feature_fraction: 0.8
    bagging_fraction: 0.8
    bagging_freq: 5
    num_boost_round: 500
    early_stopping_rounds: 50
```

### 2.3 训练

```yaml
training:
  target: "log(1+gift_price_label)"
  feature_version: "frozen"  # 严格无泄漏
  epochs: 500
  seed: 42
  n_jobs: -1
```

### 2.4 扫描参数

```yaml
sweep:
  feature_groups:
    - baseline  # 现有 past-only 特征
    - baseline + sequence  # 添加序列特征
    - baseline + realtime  # 添加实时特征
    - baseline + content_match  # 添加内容匹配特征
    - baseline + coldstart  # 添加冷启动泛化特征
    - baseline + all  # 全部新特征
  fixed:
    model: LightGBM
    feature_version: frozen
```

---

## 3. 🦾 新特征设计规格

### 3.1 序列特征（Sequence Features）

```yaml
sequence_features:
  user_last_n_gift_amounts:
    description: "用户最近 N 次打赏金额列表"
    window: [5, 10]
    aggregations: [mean, std, max, min, trend]
    note: "trend = (last_3 - first_3) / first_3"

  user_last_n_gift_intervals:
    description: "用户最近 N 次打赏间隔（小时）"
    window: [5, 10]
    aggregations: [mean, std, min]
    note: "log 变换处理长尾"

  pair_last_n_gift_amounts:
    description: "该 pair 最近 N 次打赏金额"
    window: [3, 5]
    aggregations: [mean, std, max]

  user_gift_trend_7d:
    description: "最近 7 天 vs 前 7 天的打赏金额变化率"
    formula: "(sum_last7d - sum_prev7d) / (sum_prev7d + 1)"

  user_active_hours:
    description: "用户活跃时段分布（0-23 小时）"
    output: "24 维向量或 top-3 活跃时段"

  user_weekday_pattern:
    description: "工作日 vs 周末的打赏比例"
    formula: "weekday_gift_count / total_gift_count"
```

### 3.2 实时上下文特征（Real-time Context Features）

```yaml
realtime_features:
  watch_time_current:
    description: "当前 session 观看时长"
    source: "click.csv watch_live_time"
    transform: "log1p"

  watch_time_ratio:
    description: "当前观看时长 / 用户平均观看时长"
    formula: "watch_live_time / user_avg_watch_time"

  session_click_count:
    description: "当前 session 点击次数"
    note: "需要定义 session（如 30min 无活动则新 session）"

  session_unique_streamers:
    description: "当前 session 观看的主播数"
    note: "session 内 nunique(streamer_id)"

  hour_of_day:
    description: "一天中的小时"
    source: "click.csv timestamp"
    note: "已有，可直接用"

  is_peak_hour:
    description: "是否高峰时段（18-22 点）"
    formula: "1 if hour in [18, 19, 20, 21, 22] else 0"
```

### 3.3 内容匹配度特征（Content Matching Features）

```yaml
content_features:
  user_streamer_category_match:
    description: "用户偏好类目 vs 主播类目"
    method: "cosine similarity"
    user_pref: "历史观看类目分布"
    streamer_content: "live_content_category"

  user_price_tier_match:
    description: "用户消费档次 vs 主播收礼档次"
    user_tier: "quartile(user_total_gift_past)"
    streamer_tier: "quartile(streamer_recent_revenue_past)"
    output: "tier_diff = abs(user_tier - streamer_tier)"

  follow_relationship:
    description: "用户是否关注该主播"
    source: "未知，需确认数据可用性"
    fallback: "若无数据则跳过"

  historical_interaction_depth:
    description: "历史互动深度（打赏/评论/点赞）"
    formula: "weighted_sum(gift_count*3, like_count*1, ...)"
```

### 3.4 冷启动泛化特征（Cold-start Generalization Features）

```yaml
coldstart_features:
  user_gift_tier:
    description: "用户消费档次"
    tiers: ["low", "mid", "high", "whale"]
    boundaries: [p25, p50, p75, p99]
    purpose: "无 pair 历史时泛化"

  streamer_quality_tier:
    description: "主播质量档次"
    tiers: ["low", "mid", "high", "top"]
    boundaries: [p25, p50, p75, p99]
    purpose: "无 pair 历史时泛化"

  user_avg_gift_per_streamer:
    description: "用户平均每主播打赏额"
    formula: "user_total_gift_past / user_unique_streamers_past"

  streamer_avg_gift_per_user:
    description: "主播平均每用户收礼额"
    formula: "streamer_total_gift_past / streamer_unique_givers_past"

  user_streamer_tier_interaction:
    description: "用户档次 × 主播档次 交互项"
    output: "4x4 = 16 个 one-hot"
```

---

## 4. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar (grouped) | 特征组合 | Top-1% Capture (%) | `gift_EVpred/img/feature_v2_ablation.png` |
| Fig2 | bar (grouped) | 特征组合 | Revenue Capture@1% (%) | `gift_EVpred/img/feature_v2_revenue.png` |
| Fig3 | bar (horizontal) | 特征名 | Importance (Gain) | `gift_EVpred/img/feature_v2_importance.png` |
| Fig4 | bar (grouped) | Slice (All/Cold-pair/Cold-streamer) | Performance (%) | `gift_EVpred/img/feature_v2_coldstart.png` |
| Fig5 | line | 特征数量 | Top-1% Capture (%) | `gift_EVpred/img/feature_v2_learning_curve.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则（必须遵守）**：
  - 单张图：`figsize=(6, 5)` 锁死
  - 多张图（subplot）：按 6:5 比例扩增，如 `(12, 5)` for 1×2, `(12, 10)` for 2×2

---

## 5. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_leakage_free_baseline.py` | `load_data()`, `prepare_click_level_data()`, `temporal_split()`, `train_direct_model()`, `evaluate_model()`, `compute_revenue_capture_at_k()` | 添加新特征工程函数 |
| `scripts/train_leakage_free_baseline.py` | `create_past_only_features_frozen()` | 扩展为 `create_sequence_features()`, `create_realtime_features()`, `create_content_features()`, `create_coldstart_features()` |
| `scripts/train_leakage_free_baseline.py` | `plot_feature_importance_comparison()`, `plot_slice_analysis()` | 修改为对比多组特征 |
| `gift_EVpred/exp/exp_feature_engineering_v2_20260118.md` | 实验设计规格 | 按此规格实现 |
| `gift_EVpred/exp/exp_leakage_free_baseline_20260118.md` | Baseline 结果（Top-1%=11.5%） | 用于对比 |

---

## 6. 📝 最终交付物

### 6.1 实验报告
- **路径**: `gift_EVpred/exp/exp_feature_engineering_v2_20260118.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（假设验证 + 设计启示）
  - 🦾 特征消融分析（每组特征的边际贡献）

### 6.2 图表文件
- **路径**: `gift_EVpred/img/`
- **命名**: `feature_v2_*.png`

### 6.3 数值结果
- **格式**: JSON
- **路径**: `gift_EVpred/results/feature_v2_eval_20260118.json`

### 6.4 训练脚本
- **路径**: `scripts/train_feature_v2.py`
- **要求**: 可重复运行，包含完整特征工程流程

---

## 7. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP-1.2 状态 + 结论快照 | §2.1, §4.3 |
| `gift_EVpred_hub.md` | 假设验证状态（H2.1-H2.4）+ 洞见 | §1（假设树）, §4（洞见） |

---

## 8. ⚠️ 注意事项

- [ ] 所有特征必须是 **past-only**（frozen 版本），严禁数据泄漏
- [ ] 序列特征只能用 `t < t_impression` 之前的历史
- [ ] 代码中添加 seed=42 固定随机性
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/feature_v2_20260118.log`
- [ ] 检查数据可用性（如 `follow_relationship` 可能无数据）
- [ ] 对比 Baseline（无新特征）确认改进有效

---

## 9. 📋 验收标准

| 指标 | Baseline (Frozen) | 目标 | 状态 |
|------|-------------------|------|------|
| Top-1% Capture | 11.5% | **> 30%** | 若达标 → 特征方向有效 |
| Revenue Capture@1% | 21.3% | **> 40%** | 若达标 → 特征方向有效 |
| Spearman | 0.103 | **> 0.3** | 排序相关性 |
| 冷启动 slice | [较差] | **显著优于 baseline** | 泛化能力 |

**决策规则**：
- 若 Top-1% > 30% 且 Revenue Capture@1% > 40% → 特征方向有效，继续深化最有效的特征组
- 若 20% < Top-1% < 30% → 特征有边际贡献，但需继续探索其他信号源
- 若 Top-1% < 20% → 当前特征体系信号不足，转向 MVP-1.3（二分类任务降级）

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 所有新特征必须是 past-only（frozen 版本）
7. ✅ 检查数据可用性，若某些特征无数据则跳过并记录
-->
