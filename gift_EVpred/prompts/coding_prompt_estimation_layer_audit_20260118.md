# 🤖 Coding Prompt: Estimation Layer Audit

> **Experiment ID:** `EXP-20260118-gift_EVpred-08`  
> **MVP:** MVP-1.6  
> **Date:** 2026-01-18  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：系统性地审计估计层的每个环节（预测目标、IO口径、时间切分、标签窗口、特征泄漏），确保问题定义正确、无泄漏、可服务在线分配，并用最简单的 Logistic/Linear Regression 跑通 baseline。

**验证假设**：
- H6.1: 预测目标（r_rev/r_usr/r_eco）口径清晰 → 必须明确 action 映射
- H6.2: 样本单位（click/session/impression）定义正确 → click-level 含 0，其他写清楚
- H6.3: 时间切分边界可审计 → 输出具体边界，验证是否 Week1/2/3
- H6.4: 标签窗口 vs 观看时长截断一致 → 对比固定窗口 vs cap，给出结论
- H6.5: 特征严格 past-only → Frozen 必做，Rolling 可选但严格
- H6.6: 简单模型（Logistic/Linear）能跑通 → 确认 IO 正确、指标合理

**预期结果**：
- 若所有审计项通过 → 问题定义清晰，可服务在线分配，进入特征工程优化阶段
- 若发现定义问题 → 修正定义（H、cap、样本单位、特征）后重新审计

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive 数据集"
  path: "data/KuaiLive"
  sample_unit: "click-level（含 0）"
  train_split: 0.70  # 前 70% 天
  val_split: 0.15    # 中间 15% 天
  test_split: 0.15   # 最后 15% 天
  label_windows: [10min, 30min, 1h]  # 至少跑通 1h + 10min
  label_methods: ["fixed_window", "watch_time_cap"]  # 必须对比
```

**必须输出**（按 click.timestamp 排序）：
- Train/Val/Test 各自：min/max 时间戳、样本数、unique user/streamer/live 数
- (z=1) 正样本率、(Y) 的分布（p50/p90/p99）
- watch_time 的分布（p50/p90/p99），验证"p50 是否≈4s"

### 2.2 模型

```yaml
model:
  - name: "Logistic Regression"
    task: "binary_classification"
    target: "z = 1[Y>0]"
    metrics: ["PR-AUC", "LogLoss", "ECE"]
    library: "sklearn.linear_model.LogisticRegression"
  
  - name: "Linear/Ridge Regression"
    task: "regression"
    target: "y = log(1+Y)"
    metrics: ["MAE_log", "RMSE_log", "Spearman", "RevShare@1%"]
    library: "sklearn.linear_model.Ridge"
```

**特征集**：
- Set-0：时间上下文特征（hour, dow, is_weekend）
- Set-1：Set-0 + past-only 聚合特征（Frozen 版本）

**重要检查**：
- Train 指标应 > Test（否则特征没信息或 pipeline 有 bug）
- 任一指标接近完美（如 AUC≈0.999）立刻报警：疑似泄漏，并打印 top 相关特征

### 2.3 训练

```yaml
training:
  seed: 42
  feature_sets: ["Set-0", "Set-1"]  # 逐步增加
  cv: false  # 使用预定义 Train/Val/Test 切分
  normalize: true  # StandardScaler for Linear models
```

### 2.4 扫描参数

```yaml
sweep:
  label_window_hours: [10min, 30min, 1h]  # 至少跑通 1h + 10min
  label_method: ["fixed_window", "watch_time_cap"]  # 必须对比
  feature_set: ["Set-0", "Set-1"]  # 逐步增加
  fixed:
    seed: 42
    model: ["Logistic", "Linear"]
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar + line | Train/Val/Test | min/max timestamp, sample count | `gift_EVpred/img/estimation_audit_time_split.png` |
| Fig2 | scatter + histogram | watch_time bucket | label difference ratio r | `gift_EVpred/img/estimation_audit_label_window.png` |
| Fig3 | bar | feature name | importance score | `gift_EVpred/img/estimation_audit_feature_leakage.png` |
| Fig4 | bar + line | model + feature_set | metrics (PR-AUC, MAE_log, etc.) | `gift_EVpred/img/estimation_audit_baseline_results.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则（必须遵守）**：
  - 单张图：`figsize=(6, 5)` 锁死
  - 多张图（subplot）：按 6:5 比例扩增，如 `(12, 5)` for 1×2, `(12, 10)` for 2×2

**Fig 1 详细要求**：
- 子图1：Train/Val/Test 时间边界（min/max timestamp）柱状图
- 子图2：样本数分布（Train/Val/Test）
- 子图3：正样本率分布（Train/Val/Test）
- 子图4：Y 分布（p50/p90/p99）箱线图
- 子图5：watch_time 分布（p50/p90/p99）箱线图

**Fig 2 详细要求**：
- 子图1：按 watch_time 分桶（<5s、5–30s、30–300s、>300s）的差异占比 r
- 子图2：Y^{(1h)} vs Y^{(cap)} 散点图（log scale）
- 子图3：差异占比 r 的分布直方图

**Fig 3 详细要求**：
- Top 20 特征重要性（Frozen vs Rolling 对比，如果有）
- 标注特征重要性比（目标 < 2x）

**Fig 4 详细要求**：
- 子图1：Logistic Regression 指标（PR-AUC, LogLoss, ECE）Train vs Test
- 子图2：Linear Regression 指标（MAE_log, RMSE_log, Spearman）Train vs Test
- 子图3：RevShare@1% 对比（所有配置）

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_leakage_free_baseline.py` | `prepare_click_level_data()`, `build_frozen_features()`, `build_rolling_features()` | 修改为只使用 Frozen 版本，添加时间切分审计 |
| `scripts/test_watch_time_truncation.py` | `prepare_click_level_data_original()`, `prepare_click_level_data_fixed()` | 复用标签窗口对比逻辑 |
| `scripts/diagnose_rolling_leakage.py` | `verify_rolling_features_no_leakage()` | 复用特征泄漏检查逻辑 |
| `scripts/evaluate_calibration.py` | `compute_ece()`, `plot_reliability_curve()` | 复用校准评估代码 |
| `scripts/evaluate_slices.py` | 数据统计函数 | 复用分布统计代码 |

**关键函数参考**：

1. **数据准备**：
   - `scripts/train_leakage_free_baseline.py` → `prepare_click_level_data()`（修改为支持多种标签窗口和截断方式）
   - `scripts/test_watch_time_truncation.py` → `prepare_click_level_data_original()` 和 `prepare_click_level_data_fixed()`

2. **特征工程**：
   - `scripts/train_leakage_free_baseline.py` → `build_frozen_features()`（只使用 Frozen 版本）
   - `scripts/train_leakage_free_baseline.py` → `build_rolling_features()`（可选，但必须严格 past-only）

3. **时间切分**：
   - `scripts/train_leakage_free_baseline.py` → `split_by_time()`（修改为输出详细统计）

4. **模型训练**：
   - 使用 sklearn 的 `LogisticRegression` 和 `Ridge`
   - 参考 `scripts/train_binary_classification.py` 的训练流程

5. **评估指标**：
   - `scripts/train_leakage_free_baseline.py` → `compute_revenue_capture()`（Revenue Capture@K）
   - `scripts/evaluate_calibration.py` → `compute_ece()`（校准误差）

6. **审计报告生成**：
   - 参考 `scripts/test_watch_time_truncation.py` 的报告格式
   - 输出 Markdown 格式的 `audit_estimation_layer.md`

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_EVpred/exp/exp_estimation_layer_audit_20260118.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（假设验证 + 设计启示）

### 5.2 审计报告（新增）
- **路径**: `gift_EVpred/audit_estimation_layer.md`
- **必须包含**：
  1. **估计层目标**（r_rev/r_usr/r_eco）口径与本轮范围
  2. **样本单位定义**（click/session/impression）与本轮采用的 click-level 的 (X,Y,y,z)
  3. **Train/Val/Test 时间切分边界**（含"Week1/2/3 是否属实"）
  4. **watch_time 与 label window 的一致性结论**（1h vs cap vs 10min/30min）
  5. **Frozen past-only baseline**（Logistic + Linear）结果与 sanity check
  6. **明确结论**：当前定义是否"能服务在线分配"？下一步改什么（H、cap、样本单位、特征）

### 5.3 图表文件
- **路径**: `gift_EVpred/img/`
- **命名**: 
  - `estimation_audit_time_split.png`
  - `estimation_audit_label_window.png`
  - `estimation_audit_feature_leakage.png`
  - `estimation_audit_baseline_results.png`

### 5.4 数值结果
- **格式**: JSON
- **路径**: `gift_EVpred/results/estimation_audit_20260118.json`
- **必须包含**：
  - 时间切分统计（Train/Val/Test 边界、样本数、分布）
  - 标签窗口对比结果（固定 vs 截断，各 watch_time 分桶的差异占比 r）
  - 特征重要性（Top 20，Frozen vs Rolling）
  - 模型指标（Logistic: PR-AUC, LogLoss, ECE; Linear: MAE_log, RMSE_log, Spearman, RevShare@1%）

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP-1.6 状态 + 结论快照 | §2.1, §4.3 |
| `gift_EVpred_hub.md` | Q0.5 假设验证状态 + 洞见 | §1 (Q0.5), §4 |

---

## 7. ⚠️ 注意事项

### 7.1 必须实现的审计项

**A. 预测目标定义**：
- 明确 r_rev/r_usr/r_eco 口径
- 明确 action (a) 在 KuaiLive 数据中如何映射（通常 action≈"把 u 暴露/引导到 (s, live_id)"）
- 输出到 `audit_estimation_layer.md` §1

**B. 样本单位定义**：
- 必须实现 click-level（含 0）
- Session-level 和 Impression-level 写清楚定义（不强制实现）
- 输出到 `audit_estimation_layer.md` §2

**C. 时间切分审计**：
- 验证是否真的是 Week1/Week2/Week3（Train/Val/Test）
- 如果按比例切（70/15/15），输出真实边界
- 输出：min/max 时间戳、样本数、unique user/streamer/live 数、正样本率、Y 分布、watch_time 分布
- 输出到 `audit_estimation_layer.md` §3

**D. 标签窗口 vs 观看时长截断**：
- 对比固定窗口 Y^{(1h)} vs 按观看时长截断 Y^{(cap)}
- 计算差异占比 r = sum(Y^{(1h)} - Y^{(cap)}) / sum(Y^{(1h)})
- 按 watch_time 分桶（<5s、5–30s、30–300s、>300s）分别输出 r
- 给出明确结论：是否需要改成 cap，或改 H（10min/30min）
- 输出到 `audit_estimation_layer.md` §4

**E. 无泄漏特征体系**：
- Frozen（必须实现）：所有聚合特征只用 train window 统计
- Rolling（可选实现，但若实现必须严格 past-only）：cumsum + shift(1)
- 最小特征集：pair_gift_*, user_total_gift_*, streamer_recent_revenue_*
- 输出到 `audit_estimation_layer.md` §5

**F. 最小baseline模型**：
- Logistic Regression：预测 z=1[Y>0]，输出 PR-AUC、LogLoss、ECE
- Linear/Ridge Regression：预测 y=log(1+Y)，输出 MAE_log、RMSE_log、Spearman、RevShare@1%
- 重要检查：Train 指标应 > Test，任一指标接近完美（AUC≈0.999）立刻报警
- 输出到 `audit_estimation_layer.md` §6

### 7.2 代码规范

- [ ] 代码中添加 seed=42 固定随机性
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/estimation_audit_20260118.log`
- [ ] 长时间任务使用 nohup 后台运行
- [ ] 所有函数添加 docstring
- [ ] 关键步骤添加 log_message() 输出

### 7.3 关键检查点

| 检查项 | 要求 | 验证方法 |
|--------|------|---------|
| 预测目标定义 | 明确 r_rev/r_usr/r_eco，action 映射 | 文档输出 |
| 样本单位 | click-level 含 0，其他写清楚 | 数据统计 |
| 时间切分 | 输出具体边界，验证是否 Week1/2/3 | 时间戳分析 |
| 标签窗口 | 对比固定 vs 截断 vs 更短 H | 差异分析 |
| 特征泄漏 | Frozen past-only，Rolling 严格 | 特征重要性比 < 2x |
| 简单模型 | Logistic/Linear 指标合理 | Train > Test，无完美分数 |

---

## 8. 🔍 执行步骤建议

1. **数据准备**：
   - 读取 KuaiLive 数据（click, gift）
   - 实现 click-level 标签构造（支持固定窗口和截断两种方式）
   - 实现时间切分（70/15/15），输出详细统计

2. **特征工程**：
   - 实现 Frozen past-only 特征（必须）
   - 可选实现 Rolling past-only 特征（严格验证无泄漏）

3. **模型训练**：
   - Logistic Regression（Set-0 → Set-1）
   - Linear/Ridge Regression（Set-0 → Set-1）

4. **评估与审计**：
   - 计算所有指标
   - 检查 Train vs Test 差异
   - 检查特征重要性比
   - 生成图表

5. **报告生成**：
   - 更新 `exp_estimation_layer_audit_20260118.md`
   - 生成 `audit_estimation_layer.md`
   - 同步更新 roadmap.md 和 hub.md

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 必须生成 audit_estimation_layer.md 审计报告
7. ✅ 所有审计项（A-F）必须完成
-->
