# 🤖 Coding Prompt: Leakage-Free Baseline

> **Experiment ID:** `EXP-20260118-gift_EVpred-01`  
> **MVP:** MVP-1.0  
> **Date:** 2026-01-18  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：修复 baseline 的数据泄漏问题，从 gift-only 回归转为 click-level EV 预测（含 0），建立可信的对比基准

**验证假设**：
- **H1.1**: Past-only 特征（冻结版 + 在线滚动版）能消除泄漏，Top-1% Capture 仍 > 40%
- **H1.2**: Click-level EV 预测（含 0）的 Revenue Capture@K 优于 gift-only baseline
- **H1.3**: Direct vs Two-Stage 在无泄漏版本上的相对差距与公平对比实验一致（Direct 仍占优）

**预期结果**：
- 若 Past-only 特征 Top-1% > 40%，Revenue Capture@1% > 50% → 确认特征体系可信，进入后续优化
- 若 Top-1% < 30% 或 Spearman 下降 > 0.2 → 需要重新设计特征工程策略

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  base_table: "click.csv"  # 从 gift-only 改为 click-level（含 0）
  label_window: "1h"  # click 后 1 小时内发生的 gift 总额
  train_split: 0.70  # 前 70% 天
  val_split: 0.15    # 中间 15% 天
  test_split: 0.15   # 最后 15% 天
  time_range: "2025-05-04 to 2025-05-25"
  features: ~70  # past-only 版本
```

**关键变更**：
- ✅ 从 `gift.csv` 改为 `click.csv`（或 click-gift join，含未送礼的 click）
- ✅ Label：click 后 H=1h 内的 gift 总额（0 或正数）
- ✅ 时间切分：严格按 timestamp，避免未来信息

### 2.2 特征工程（核心）

```yaml
feature_engineering:
  version: ["frozen", "rolling"]  # 两种版本都要实现
  frozen:
    method: "train_window_only"
    description: "所有聚合特征只用 train 窗口统计，val/test 只查 lookup 表"
  rolling:
    method: "cumsum_with_shift"
    description: "按 timestamp 排序，groupby + cumsum，shift(1) 排除当前样本"
  
  past_only_features:
    pair_features:
      - pair_gift_sum_past
      - pair_gift_mean_past
      - pair_gift_count_past
      - pair_last_gift_time_gap_past
    user_features:
      - user_total_gift_7d_past
      - user_budget_proxy_past  # 最近 7 天总额
    streamer_features:
      - streamer_recent_revenue_past
      - streamer_recent_unique_givers_past
      - streamer_overload_proxy_past  # 最近 1h/1d 观看人数
  
  removed_features:  # 确认泄漏，必须移除
    - pair_gift_mean  # 全量聚合回填
    - pair_gift_sum    # 全量聚合回填
    - 任何用 test 数据统计的特征
```

### 2.3 模型

```yaml
model:
  architectures: ["direct", "two_stage"]
  
  direct:
    name: "LightGBM"
    objective: "regression"
    target: "log(1+Y)"  # 或 raw Y
    params:
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 500
      early_stopping_rounds: 50
      feature_fraction: 0.8
      bagging_fraction: 0.8
      seed: 42
  
  two_stage:
    stage1:
      name: "LightGBM"
      objective: "binary"
      target: "Y > 0"
    stage2:
      name: "LightGBM"
      objective: "regression"
      target: "raw Y | Y > 0"  # 关键：预测 raw amount，确保 p×m 量纲正确
    params:
      num_leaves: 31
      learning_rate: 0.05
      n_estimators: 500
      early_stopping_rounds: 50
      feature_fraction: 0.8
      bagging_fraction: 0.8
      seed: 42
```

### 2.4 训练

```yaml
training:
  data_split: "temporal"  # 严格按时间切分
  split_ratio: [0.70, 0.15, 0.15]
  feature_versions: ["frozen", "rolling"]  # 分别训练
  early_stopping: 50
  seed: 42
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar | Feature Name | Importance | `gift_EVpred/img/leakage_free_feature_importance.png` |
| Fig2 | line | Top-K% (1%, 5%, 10%, ...) | Revenue Capture@K | `gift_EVpred/img/leakage_free_revenue_capture.png` |
| Fig3 | line | Prediction Decile | Actual vs Predicted | `gift_EVpred/img/leakage_free_calibration.png` |
| Fig4 | bar | Slice (Cold/Warm, Top1%/Top10%/Long-tail) | Performance Metric | `gift_EVpred/img/leakage_free_slice_analysis.png` |
| Fig5 | bar | Model (Direct Frozen/Rolling, Two-Stage Frozen/Rolling) | Top-1% Capture / Revenue Capture@1% | `gift_EVpred/img/leakage_free_direct_vs_twostage.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则（必须遵守）**：
  - 单张图：`figsize=(6, 5)` 锁死
  - 多张图（subplot）：按 6:5 比例扩增，如 `(12, 5)` for 1×2, `(12, 10)` for 2×2

**Fig1 要求**：对比 Frozen vs Rolling vs Original baseline 的特征重要性（Top 20）

**Fig2 要求**：对比 Frozen vs Rolling 的 Revenue Capture@K 曲线，同时标注 Top-K% Capture 作为对比

**Fig3 要求**：分桶校准曲线（10 个分桶），显示预测分位 vs 实际分位，计算 ECE

**Fig4 要求**：切片评估，包括：
- 冷启动 pair（train 中 `pair_gift_count=0`）
- 冷启动 streamer（历史收礼=0）
- Top-1% 用户、Top-10% 用户、长尾用户

**Fig5 要求**：对比 Direct vs Two-Stage 在 Frozen 和 Rolling 版本上的表现

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

### 4.1 数据加载和预处理

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_fair_comparison.py` | `load_data()`, `prepare_features()` 中的 click-base 构造逻辑 | 修改 label 构造：从 `gift_agg` 改为 click 后 1h 窗口内的 gift 总额 |
| `scripts/train_two_stage.py` | `prepare_features()` 中的 click-base 构造 | 同上，label 窗口改为 1h |

### 4.2 特征工程（核心）

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_baseline_lgb.py` | `create_user_features()`, `create_streamer_features()`, `create_interaction_features()` | **关键**：修改为 past-only 版本（冻结版和滚动版） |
| `scripts/train_fair_comparison.py` | `create_user_features()`, `create_streamer_features()`, `create_interaction_features()` | 同上，实现两种版本 |

**实现要求**：
- **冻结版**：在 `prepare_features()` 中，先对 train 数据做 groupby 统计，保存为 lookup 字典/DataFrame，val/test 只查表
- **滚动版**：在 `prepare_features()` 中，按 timestamp 排序，对每个 group（user_id, streamer_id 等）做 cumsum，然后 shift(1) 排除当前样本

### 4.3 模型训练

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_baseline_lgb.py` | `train_model()`, LightGBM 训练循环 | 修改为支持两种特征版本（frozen/rolling）分别训练 |
| `scripts/train_two_stage.py` | `train_stage1()`, `train_stage2()` | 修改 Stage2 目标为 raw Y（而非 log），确保 p×m 量纲正确 |

### 4.4 评估指标（新增）

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_recall_rerank.py` | `compute_top_k_capture()` | **新增**：实现 `compute_revenue_capture_at_k()`，计算收入占比而非集合重叠 |
| `scripts/train_baseline_lgb.py` | `plot_calibration()` | 复用校准曲线绘制，计算 ECE |
| `scripts/train_fair_comparison.py` | 评估指标计算逻辑 | **新增**：切片评估（冷启动 pair/streamer、用户分位） |

**Revenue Capture@K 实现**：
```python
# 伪代码（不要直接复制，理解逻辑后实现）
def compute_revenue_capture_at_k(y_true, y_pred, k_pct=0.01):
    # 1. 按 y_pred 排序
    # 2. 取 Top K% 样本
    # 3. 计算这些样本的 y_true 总和
    # 4. 除以全部样本的 y_true 总和
    # 返回：收入占比（0-1）
```

### 4.5 可视化

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_baseline_lgb.py` | `plot_topk_capture()`, `plot_calibration()`, `plot_feature_importance()` | 修改为支持对比 Frozen vs Rolling vs Original |
| `scripts/train_fair_comparison.py` | 图表绘制函数 | 新增切片分析图表 |

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_EVpred/exp/exp_leakage_free_baseline_20260118.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字）
  - 📊 实验图表（所有 5 张图 + 观察）
  - 📝 结论（假设验证 + 设计启示）
  - 📎 数值结果表（Frozen vs Rolling vs Baseline 对比）

### 5.2 图表文件
- **路径**: `gift_EVpred/img/`
- **命名**: 
  - `leakage_free_feature_importance.png`
  - `leakage_free_revenue_capture.png`
  - `leakage_free_calibration.png`
  - `leakage_free_slice_analysis.png`
  - `leakage_free_direct_vs_twostage.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `gift_EVpred/results/leakage_free_eval_20260118.json`
- **必须包含**:
  - Direct (Frozen/Rolling) 的所有指标
  - Two-Stage (Frozen/Rolling) 的所有指标
  - 切片评估结果
  - vs Baseline (gift-only) 的对比

### 5.4 模型文件
- **路径**: `gift_EVpred/models/`
- **命名**:
  - `direct_frozen_20260118.pkl`
  - `direct_rolling_20260118.pkl`
  - `twostage_frozen_20260118.pkl`
  - `twostage_rolling_20260118.pkl`

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP-1.0 状态更新 + 结论快照 | §2.1, §4.3 |
| `gift_EVpred_hub.md` | 假设验证状态（H1.1, H1.2, H1.3）+ 洞见 | §1, §4 |

**Hub 同步内容**：
- H1.1 验证结果 → Q1.1a 状态更新
- H1.2 验证结果 → Q1.1b 状态更新
- H1.3 验证结果 → Q1.1 状态更新（复验结论）
- Revenue Capture@K 指标验证 → Q2.1a 状态更新

---

## 7. ⚠️ 注意事项

### 7.1 数据泄漏检查（必须）

- [ ] **冻结版**：确保 val/test 的特征只来自 train 窗口的 lookup 表
- [ ] **滚动版**：确保每个样本的特征计算时，shift(1) 正确排除当前样本
- [ ] **时间切分**：严格按 timestamp 排序后切分，不允许随机切分
- [ ] **Label 构造**：click 后 1h 窗口内的 gift 总额，必须基于 timestamp 计算，不能包含未来信息

### 7.2 特征工程实现细节

**冻结版实现步骤**：
1. 在 train 数据上，对每个 (user_id, streamer_id) 做 groupby 统计
2. 保存为 lookup 字典或 DataFrame（key: (user_id, streamer_id), value: 统计值）
3. 对 val/test 数据，只查 lookup 表，缺失值填充 0 或默认值

**滚动版实现步骤**：
1. 将全量数据按 timestamp 排序
2. 对每个 group（如 user_id, streamer_id），计算 cumsum / expanding mean 等
3. 使用 `shift(1)` 将特征值向下移动一行（排除当前样本）
4. 然后按时间切分 train/val/test

### 7.3 评估指标实现

**Revenue Capture@K**（关键新指标）：
- 公式：$\text{RevShare@K} = \frac{\sum_{i \in \text{TopK}(\hat{v})} Y_i}{\sum_i Y_i}$
- 与 Top-K% Capture 的区别：Top-K% Capture 是集合重叠比例，Revenue Capture@K 是收入占比
- 实现时注意：分母是全部样本的 y_true 总和，分子是 Top-K% 样本的 y_true 总和

**切片评估**：
- 冷启动 pair：train 中 `pair_gift_count=0` 的样本
- 冷启动 streamer：历史收礼总额=0 的主播
- 用户分位：按用户历史打赏总额分位（Top-1%, Top-10%, 长尾）

### 7.4 其他

- [ ] 代码中添加 seed 固定随机性
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/leakage_free_baseline_20260118.log`
- [ ] 长时间任务使用 nohup 后台运行
- [ ] Two-Stage 的 Stage2 必须预测 raw Y（确保 p×m 量纲正确）

---

## 8. 🔍 关键实现检查清单

### 8.1 数据准备
- [ ] 从 click.csv 构造 click-level 数据集（含 0 值）
- [ ] Label：click 后 1h 窗口内的 gift 总额
- [ ] 严格按时间切分（前 70% / 中间 15% / 最后 15% 天）

### 8.2 特征工程
- [ ] 实现冻结版 past-only 特征（train lookup 表）
- [ ] 实现在线滚动版 past-only 特征（cumsum + shift）
- [ ] 移除泄漏特征（pair_gift_mean/sum 全量聚合版本）
- [ ] 验证特征无泄漏（检查 val/test 特征是否包含未来信息）

### 8.3 模型训练
- [ ] Direct Regression（Frozen 版本）
- [ ] Direct Regression（Rolling 版本）
- [ ] Two-Stage（Frozen 版本，Stage2 预测 raw Y）
- [ ] Two-Stage（Rolling 版本，Stage2 预测 raw Y）

### 8.4 评估
- [ ] 实现 Revenue Capture@K 指标
- [ ] 计算分桶校准（ECE）
- [ ] 切片评估（冷启动、用户分位）
- [ ] 对比分析：vs Baseline (gift-only)

### 8.5 可视化
- [ ] Fig1: 特征重要性对比
- [ ] Fig2: Revenue Capture@K 曲线
- [ ] Fig3: 校准曲线
- [ ] Fig4: 切片分析
- [ ] Fig5: Direct vs Two-Stage 对比

### 8.6 报告
- [ ] 填写 exp.md 报告（按模板）
- [ ] 更新 roadmap.md
- [ ] 更新 hub.md

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 重点实现 past-only 特征（冻结版和滚动版）
7. ✅ 实现 Revenue Capture@K 指标（收入占比，非集合重叠）
8. ✅ 验证数据无泄漏（检查 val/test 特征）
-->
