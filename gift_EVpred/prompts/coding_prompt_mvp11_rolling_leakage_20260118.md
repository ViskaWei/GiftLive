# 🤖 Coding Prompt: Rolling Leakage Diagnosis

> **Experiment ID:** `EXP-20260118-gift_EVpred-03`
> **MVP:** MVP-1.1
> **Date:** 2026-01-18
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：诊断 Rolling 版本（cumsum+shift）为何性能异常高（81.1% vs Frozen 11.5%），确认是否存在时间泄漏

**验证假设**：
- H1.1: cumsum+shift 实现有泄漏 → 特征值包含当前样本信息
- H1.2: 泄漏发生在 pair 特征 → 移除 pair 特征后性能大幅下降
- H1.3: 修复后 Rolling ≈ Frozen → 差距 < 5pp

**预期结果**：
- 若确认泄漏 → 修复 `create_past_only_features_rolling()` 函数，重新评估
- 若无泄漏 → 分析 Rolling 优于 Frozen 的原因（更多历史信息？）

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  files:
    - gift.csv
    - click.csv
    - user.csv
    - streamer.csv
    - room.csv
  sample_unit: "click-level"
  label_window: "1h after click"
```

### 2.2 诊断任务

```yaml
diagnosis:
  checks:
    - name: "timestamp_sorting"
      description: "检查数据是否按时间排序"
      expected: "timestamp 严格升序"

    - name: "duplicate_timestamp"
      description: "统计同一时间戳的记录数"
      expected: "记录数量统计"

    - name: "first_gift_check"
      description: "检查 pair 首次打赏的样本，pair_gift_count_past 是否为 0"
      expected: "应该全为 0，否则有泄漏"

    - name: "time_travel_check"
      description: "抽样 100 条，对比 Rolling 特征 vs 真实 past-only 特征"
      expected: "完全一致"

    - name: "feature_isolation"
      description: "分别移除 pair/user/streamer 特征后重训"
      expected: "定位泄漏来源"
```

### 2.3 特征隔离测试

```yaml
ablation_tests:
  - name: "Test A: Remove pair_* features"
    features_removed:
      - pair_gift_count_past
      - pair_gift_sum_past
      - pair_gift_mean_past
      - pair_last_gift_time_gap_past
    expected: "若泄漏在 pair，性能大幅下降"

  - name: "Test B: Remove user_* features"
    features_removed:
      - user_total_gift_7d_past
      - user_budget_proxy_past
    expected: "若泄漏在 user，性能下降"

  - name: "Test C: Remove streamer_* features"
    features_removed:
      - streamer_recent_revenue_past
      - streamer_recent_unique_givers_past
    expected: "若泄漏在 streamer，性能下降"
```

### 2.4 修复方案（若确认泄漏）

```yaml
fix_strategy:
  issue: "cumsum+shift 可能包含当前样本或未来数据"

  fix_options:
    - name: "Option A: 严格 timestamp < current"
      description: "对每条样本，只统计 timestamp 严格小于当前的历史"

    - name: "Option B: 先切分再 cumsum"
      description: "在 train/val/test 切分后，分别对每个 split 做 cumsum"

    - name: "Option C: 使用 Frozen 方法"
      description: "放弃 Rolling，统一使用 Frozen（train window lookup）"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | scatter + line | 样本索引 (sample 100) | pair_gift_count_past | `gift_EVpred/img/rolling_vs_frozen_features.png` |
| Fig2 | bar | 特征组 (All / No-Pair / No-User / No-Streamer) | Top-1% Capture | `gift_EVpred/img/feature_isolation_test.png` |
| Fig3 | heatmap | 样本类型 | 泄漏率 (%) | `gift_EVpred/img/leakage_diagnosis_matrix.png` |
| Fig4 | bar | Model Version | Top-1% Capture | `gift_EVpred/img/rolling_fixed_vs_frozen.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则（必须遵守）**：
  - 单张图：`figsize=(6, 5)` 锁死
  - 多张图（subplot）：按 6:5 比例扩增，如 `(12, 5)` for 1×2, `(12, 10)` for 2×2

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改/新增 |
|---------|--------|------------|
| `scripts/train_leakage_free_baseline.py` | `load_data()`, `prepare_click_level_data()`, `temporal_split()`, `train_direct_model()`, `evaluate_model()`, `compute_revenue_capture_at_k()` | 读取并分析 `create_past_only_features_rolling()` 函数，找出泄漏点 |
| `scripts/train_leakage_free_baseline.py` | `create_past_only_features_frozen()`, `apply_frozen_features()` | 作为正确实现的参考 |
| `scripts/train_leakage_free_baseline.py` | `get_feature_columns()` | 用于特征隔离测试 |
| `gift_EVpred/models/` | 已保存的模型文件 | 可加载 Rolling 模型检查特征重要性 |

**关键代码位置**（需重点审查）：
- `create_past_only_features_rolling()`: 第 245-338 行
- 检查 `cumsum` 和 `shift` 的使用是否正确排除当前样本
- 检查 `merge` 操作是否引入未来数据

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_EVpred/exp/exp_rolling_leakage_diagnosis_20260118.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 泄漏确认/否定）
  - 📊 诊断图表（4 张图 + 观察）
  - 📝 结论（H1.1/H1.2/H1.3 验证结果 + 修复建议）

### 5.2 图表文件
- **路径**: `gift_EVpred/img/`
- **命名**:
  - `rolling_vs_frozen_features.png`
  - `feature_isolation_test.png`
  - `leakage_diagnosis_matrix.png`
  - `rolling_fixed_vs_frozen.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `gift_EVpred/results/rolling_leakage_diagnosis_20260118.json`
- **内容**:
  - 各诊断检查的结果
  - 特征隔离测试的性能数据
  - 修复前后的对比指标

### 5.4 修复后的脚本（若有泄漏）
- **路径**: `scripts/train_leakage_free_baseline_v2.py`
- **内容**: 修复后的 `create_past_only_features_rolling()` 函数

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred/gift_EVpred_roadmap.md` | MVP-1.1 状态 → ✅ 完成 + 结论快照 | §2.1 实验总览 |
| `gift_EVpred/gift_EVpred_hub.md` | Q1.5 假设验证状态 + 洞见 | §1 假设树, §4 洞见 |

---

## 7. ⚠️ 注意事项

- [ ] 诊断脚本使用 seed=42 固定随机性
- [ ] 抽样检查至少 100 条样本
- [ ] 记录每个检查点的具体数值（不要只说"通过/失败"）
- [ ] 保存完整日志到 `logs/rolling_leakage_diagnosis_20260118.log`
- [ ] 若发现泄漏，先完成诊断报告，再修复代码

---

## 8. 🔍 诊断步骤详细说明

### Step 1: 数据排序检查
- 读取 gift.csv 和 click.csv
- 验证 timestamp 是否严格升序
- 输出：排序状态 + 是否有乱序记录

### Step 2: 重复时间戳检查
- 统计同一 timestamp 下的记录数分布
- 输出：max / mean / 分位数

### Step 3: 首次打赏样本检查（关键）
- 找出每个 (user_id, streamer_id) pair 的第一条 gift 记录
- 检查这些记录的 `pair_gift_count_past` 是否为 0
- **预期**：应该全为 0
- **若不为 0**：确认泄漏，记录非零比例和典型案例

### Step 4: 时间穿越抽样检查（关键）
- 随机抽取 100 条 test set 样本
- 对每条样本，用原始数据重新计算"真实的 past-only 特征"
- 对比 Rolling 版本的特征值
- **预期**：完全一致
- **若不一致**：记录差异样本数、差异大小

### Step 5: 特征隔离测试
- 分别移除 pair / user / streamer 特征组
- 重新训练 Direct 模型
- 记录每组的 Top-1% Capture 和 Revenue Capture@1%
- 分析哪组特征对高性能贡献最大

### Step 6: 修复实现（若确认泄漏）
- 根据诊断结果修复 `create_past_only_features_rolling()`
- 重新训练并评估
- **目标**：修复后 Rolling 与 Frozen 差距 < 5pp

### Step 7: 重新评估
- 对比修复前后的性能
- 更新实验报告

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 诊断优先：先完成诊断，确认问题后再修复
-->
