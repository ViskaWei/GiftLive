# 🗺️ Gift_evmodel Roadmap
> **Name:** Gift EV Model Roadmap | **ID:** `EXP-20260118-gift_EVpred-roadmap`
> **Topic:** `gift_EVpred` | **Phase:** 0 → 1 (修复基础设施)
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** 🔄

```
💡 当前阶段目标：修复数据泄漏和任务定义问题
Gate-0：验证 Past-only + Click-level 无泄漏 Baseline 的有效性
```

---

## 🔴 Gate-0 结果（2026-01-18）

> **Gate-0 失败：泄漏已消除，但性能不达标**。需要重新设计特征工程策略。

| 问题 | 状态 | 解决方案 | 结果 |
|------|------|----------|------|
| **数据泄漏** | ✅ 已解决 | MVP-1.0: Past-only 特征 | 特征重要性比 1.23 < 2x |
| **任务不匹配** | ✅ 已解决 | MVP-1.0: Click-level（含 0） | Y=0 占比 98.50% |
| **评估偏差** | ✅ 已解决 | MVP-1.0: Revenue Capture@K | 已实现 |
| **性能达标** | ❌ 失败 | Gate-0 验证 | Top-1%=11.6%（目标>40%） |

---

## 🔗 Related Files
| Type | File |
|------|------|
| 🧠 Hub | `gift_EVpred_hub.md` |
| 📋 Kanban | `status/kanban.md` |
| 📗 Experiments | `exp_*.md` |

---

# 1. 🚦 Decision Gates

> Roadmap 定义怎么验证，Hub 做战略分析

## 1.1 战略路线 (来自Hub)

| Route | 名称 | Hub推荐 | 验证Gate |
|-------|------|---------|----------|
| **0** | **Fix Foundation（修复基础设施）** | 🔴 **最高优先级** | **Gate-0** |
| 1 | Direct vs Two-Stage 复验 | 🟡 待复验 | Gate-1 |
| 2 | 召回-精排分工 | ⏳ 暂缓 | Gate-2 |
| 3 | 分配优化 + 模拟器 | ⏳ 远期 | Gate-3 |

## 1.2 Gate定义

### Gate-0: 无泄漏 Baseline 验证 【❌ 已完成 - 失败】

| 项 | 内容 |
|----|------|
| **验证** | Past-only 特征 + Click-level EV 能否消除泄漏且保持性能 |
| **MVP** | MVP-1.0 |
| **若通过** | Top-1% > 40% 且 Spearman 下降 < 0.2 → 特征体系可信，进入 Gate-1 |
| **若失败** | Top-1% < 30% → 需重新设计特征工程策略 |
| **状态** | ❌ **已完成 - 失败** |

**实际结果**：
| 指标 | 目标 | 实际 | 状态 |
|------|------|------|------|
| 特征重要性比 | < 2x | **1.23** | ✅ 泄漏消除 |
| Top-1% Capture | > 40% | **11.6%** | ❌ |
| Revenue Capture@1% | > 50% | **21.6%** | ❌ |
| Spearman | 下降 < 0.2 | **0.14**（-0.75） | ❌ |

**结论**：泄漏已消除，但性能大幅下降，揭示原 baseline 性能几乎全部来自数据泄漏。需要重新设计特征工程策略。

### Gate-1: Direct vs Two-Stage 复验

| 项 | 内容 |
|----|------|
| **验证** | 在无泄漏版本上，Direct 是否仍优于 Two-Stage |
| **MVP** | MVP-1.0 (与 Gate-0 同时完成) |
| **若 Direct 仍优** | 差距 > 10pp → 结论稳定，Direct 为召回主力 |
| **若差距缩小** | 差距 < 5pp → 需重新评估架构选择 |
| **状态** | ⏳ 等待 Gate-0 |

### Gate-2: 召回-精排分工策略

| 项 | 内容 |
|----|------|
| **验证** | Direct 召回 + Two-Stage 精排是否优于纯 Direct |
| **MVP** | MVP-1.5 |
| **若分工更优** | Top-1% > Direct 单独 → 采用分工策略 |
| **若无显著提升** | 差距 < 2pp → 保持简单架构 |
| **状态** | ⏳ 等待 Gate-1 |

### Gate-3: 长期目标 + 模拟器

| 项 | 内容 |
|----|------|
| **验证** | 模拟器能否用于策略评估（OPE/分配优化） |
| **MVP** | MVP-2.0 |
| **若模拟有效** | 矩匹配误差 < 10% → 可用于策略迭代 |
| **若模拟偏差大** | 误差 > 30% → 仅用离线数据 |
| **状态** | ⏳ 远期 |

## 1.3 本周重点

| 优先级 | MVP | Gate | 状态 | 说明 |
|--------|-----|------|------|------|
| 🔴 P0 | MVP-1.0 | Gate-0 | ❌ 完成-失败 | 无泄漏 Baseline（泄漏消除，性能不达标） |
| 🔴 P0 | MVP-1.1 | - | ✅ 完成 | Rolling 泄漏诊断（确认泄漏，放弃 Rolling） |
| 🔴 P0 | MVP-1.2 | - | ✅ 完成 | 重构特征工程（watch_time 是最强信号，Top-1%=19.8%） |
| 🔴 P0 | MVP-1.3 | - | ✅ 完成 | 任务降级验证（AUC=0.61，Prec@1%=19.24%） |
| 🔴 P0 | MVP-1.6 | - | ✅ 完成 | 估计层审计（预测目标+IO口径+无泄漏验证，发现标签窗口问题） |
| 🟡 P1 | MVP-1.4 | - | ✅ 完成 | 切片评估（冷启动是致命瓶颈） |
| 🟡 P1 | MVP-1.5 | - | ✅ 完成 | 校准评估（回归严重低估 1500x） |
| 🟡 P1 | MVP-2.0 | Gate-3 | ⏳ 计划 | 模拟器设计 |

---

# 2. 📋 MVP列表

## 2.1 总览

| MVP | 名称 | Phase | Gate | 状态 | exp_id | 报告 |
|-----|------|-------|------|------|--------|------|
| ~~0.2~~ | ~~Baseline (gift-only)~~ | ~~0~~ | ~~-~~ | ❌ **泄漏无效** | `EXP-20260108-gift-allocation-02` | [Link](./exp/exp_baseline_20260108.md) |
| **1.0** | **Leakage-Free Baseline** | **1** | **Gate-0** | **❌ 完成-失败** | `EXP-20260118-gift_EVpred-01` | [Link](./exp/exp_leakage_free_baseline_20260118.md) |
| **1.1** | **Rolling Leakage Diagnosis & Fix** | **1** | **-** | **✅ 完成** | `EXP-20260118-gift_EVpred-02/03` | [Fix](./exp/exp_rolling_leakage_fix_20260118.md), [Diagnosis](./exp/exp_rolling_leakage_diagnosis_20260118.md) |
| **1.2** | **Feature Engineering V2** | **1** | **-** | **✅ 完成** | `EXP-20260118-gift_EVpred-04` | [Link](./exp/exp_feature_engineering_v2_20260118.md) |
| **1.3** | **Binary Classification** | **1** | **-** | **✅ 完成** | `EXP-20260118-gift_EVpred-05` | [Link](./exp/exp_binary_classification_20260118.md) |
| **1.4** | **Slice Evaluation** | **1** | **-** | **✅ 完成** | `EXP-20260118-gift_EVpred-06` | [Link](./exp/exp_slice_evaluation_20260118.md) |
| **1.5** | **Calibration Evaluation** | **1** | **-** | **✅ 完成** | `EXP-20260118-gift_EVpred-07` | [Link](./exp/exp_calibration_evaluation_20260118.md) |
| **1.6** | **Estimation Layer Audit** | **1** | **-** | **✅ 完成** | `EXP-20260118-gift_EVpred-08` | [Link](./exp/exp_estimation_layer_audit_20260118.md) |
| 1.7 | 召回-精排分工验证 | 1 | Gate-2 | ⏳ 暂缓 | - | - |
| **2.0** | **模拟器 V1** | **2** | **Gate-3** | **🟡 计划** | - | [Link](./exp/exp_simulator_v1_20260118.md) |

**状态**: ⏳计划 | 🔴就绪 | 🚀运行 | ✅完成 | ❌废弃

## 2.2 配置速查

| MVP | 数据量 | 特征 | 模型 | 关键变量 | 状态 |
|-----|--------|------|------|---------|------|
| ~~0.2~~ | ~~72,646 (gift-only)~~ | ~~68 (有泄漏)~~ | ~~LightGBM~~ | ~~log(1+Y)~~ | ❌ 废弃 |
| **1.0** | click-level（含 0）~4.9M | ~70 (past-only) | LightGBM | Direct + Two-Stage | 🔴 就绪 |
| **1.6** | click-level（含 0）~4.9M | past-only (Frozen) | Logistic + Ridge | 审计 IO 口径 | ✅ 完成 |
| 1.7 | click-level | past-only | Direct → Two-Stage | 分工策略 | ⏳ 计划 |
| 2.0 | 模拟数据 | 参数化 | - | 策略评估 | 🟡 计划 |

---

# 3. 🔧 MVP规格

## Phase 0: Baseline【❌ 已废弃】

### ~~MVP-0.2: Baseline (gift-only)~~ 【❌ 泄漏无效】

| 项 | 配置 | 问题 |
|----|------|------|
| 目标 | ~~建立直接回归 baseline，预测打赏金额~~ | - |
| 数据 | ~~gift-only, 72,646 样本~~ | ❌ 任务不匹配：学的是 E[gift\|已送礼]，不是 EV |
| 特征 | ~~68 特征（pair_gift_mean/sum 等）~~ | ❌ 数据泄漏：特征=答案 |
| 模型 | ~~LightGBM, log(1+Y)~~ | - |
| 结果 | ~~Top-1%=56.2%, Spearman=0.891~~ | ❌ **虚高，不可信** |

> **教训**：`pair_gift_mean` 重要性=328,571，远超其他 3 倍 → 泄漏特征的典型信号

---

## Phase 1: 修复基础设施【🔴 当前重点】

### MVP-1.0: Leakage-Free Baseline 【🔴 就绪待执行】

| 项 | 配置 |
|----|------|
| **目标** | 修复数据泄漏 + 任务定义，建立可信的对比基准 |
| **Gate** | Gate-0（无泄漏验证）+ Gate-1（架构复验） |
| **数据** | Click-level（含 0），~4.9M 样本，按时间切分（前 70% / 中间 15% / 最后 15%） |
| **特征** | ~70 个 **Past-only 特征**（冻结版 + 滚动版） |
| **模型** | LightGBM (Direct + Two-Stage) |
| **主指标** | **Revenue Capture@K**（收入占比，非集合重叠） |
| **验收** | Top-1% > 40%, Revenue Capture@1% > 50%, Spearman 下降 < 0.2 |

**关键改进**：
1. ✅ Past-only 特征消除泄漏（冻结版 + 滚动版）
2. ✅ Click-level EV（含 0）修正任务定义
3. ✅ Revenue Capture@K 修正评估指标
4. ✅ 分桶校准（ECE）+ 切片评估（冷启动/Top-1%/长尾）
5. ✅ Direct vs Two-Stage 无泄漏版本复验

**预期结果**：
- Top-1% 下降到 40-50%（去除泄漏后的真实性能）
- Direct vs Two-Stage 差距可能缩小或变化
- 冷启动 pair/streamer 性能显著下降（符合预期）

### MVP-1.6: Estimation Layer Audit 【🟡 计划】

| 项 | 配置 |
|----|------|
| **目标** | 系统性地审计估计层的每个环节，确保问题定义正确、IO口径清晰、无泄漏，并用最简单的模型验证基础设施正确性 |
| **Gate** | -（审计类实验，为后续实验提供基础） |
| **前置** | MVP-1.0 完成（发现性能不达标，需要审计） |
| **数据** | Click-level（含 0），KuaiLive 数据集 |
| **模型** | **Logistic Regression**（预测 z=1[Y>0]）+ **Linear/Ridge Regression**（预测 y=log(1+Y)） |
| **验收** | 输出 `audit_estimation_layer.md`，明确当前定义是否"能服务在线分配" |

**审计内容**：

1. **A. 预测目标定义**（必须输出清楚）
   - 明确 r_rev/r_usr/r_eco 口径与本轮范围
   - action (a) 在 KuaiLive 数据中如何映射到字段

2. **B. 样本单位定义**（必须实现 click-level，其他写清楚）
   - Click-level（必须）：一条 click = 一次机会（decision proxy）
   - Session-level（写清楚定义）：session = 同一 live_id 的连续观看段
   - Impression-level（写清楚数据需求）：需要曝光/未点击日志

3. **C. 时间切分审计**（必须做审计 + 输出具体边界）
   - 验证是否真的是 Week1/Week2/Week3（Train/Val/Test）
   - 如果按比例切（70/15/15），输出真实边界
   - 输出：min/max 时间戳、样本数、unique user/streamer/live 数、正样本率、Y 分布、watch_time 分布

4. **D. 标签窗口 vs 观看时长截断**（必须做对照实验）
   - 对比固定窗口 Y^{(1h)} vs 按观看时长截断 Y^{(cap)}
   - 输出差异占比 r，按 watch_time 分桶分析
   - 给出明确结论：是否需要改成 cap，或改 H（10min/30min）

5. **E. 无泄漏特征体系**（必须只用 past-only）
   - E1) Frozen（必须实现）：所有聚合特征只用 train window 统计
   - E2) Rolling（可选实现，但若实现必须严格 past-only）：cumsum + shift(1)
   - 最小特征集：pair_gift_*, user_total_gift_*, streamer_recent_revenue_*

6. **F. 最小baseline模型**（必须跑通，越简单越好）
   - F1) Logistic Regression：预测 z=1[Y>0]，输出 PR-AUC、LogLoss、ECE
   - F2) Linear/Ridge Regression：预测 y=log(1+Y)，输出 MAE_log、RMSE_log、Spearman、RevShare@1%
   - 重要检查：Train 指标应 > Test，任一指标接近完美（AUC≈0.999）立刻报警

**输出物**：
- `audit_estimation_layer.md`：包含 A-F 所有审计结果和明确结论
- 图表：时间切分边界、标签窗口对比、特征泄漏验证、baseline 结果

### MVP-1.7: 召回-精排分工验证

| 项 | 配置 |
|----|------|
| **目标** | 验证 Direct 召回 + Two-Stage 精排是否优于纯 Direct |
| **Gate** | Gate-2 |
| **前置** | MVP-1.0 通过 Gate-0 和 Gate-1 |
| **数据** | 同 MVP-1.0 |
| **模型** | Direct 召回 Top-K → Two-Stage 精排 |
| **验收** | 分工策略 Top-1% > Direct 单独 |

---

## Phase 2: 长期目标 + 模拟器

### MVP-2.0: 模拟器 V1 【🟡 计划】

| 项 | 配置 |
|----|------|
| **目标** | 构建可用于策略评估的模拟环境，填补数据不足（长期留存/因果 uplift） |
| **Gate** | Gate-3 |
| **前置** | MVP-1.0 完成 |

**模拟器最小闭环**：

| 组件 | 设计 |
|------|------|
| **实体** | 用户 u、主播 s、时间 t |
| **用户状态** | 财富/预算、偏好向量、疲劳度、近期满意度 |
| **主播状态** | 质量、容量上限、近期曝光/收入、生态健康分 |
| **行动** | 给每个用户在 t 时刻分配一个主播（或 Top-K） |
| **观测** | click/watch/comment/like/gift（带延迟） |
| **奖励** | $R_t = \alpha \cdot \text{Revenue}_t + \beta \cdot \text{UserSat}_t - \gamma \cdot \text{ConcentrationPenalty}_t$ |

**校准到 KuaiLive（矩匹配）**：

| 统计量 | KuaiLive 真值 | 模拟目标 |
|--------|---------------|----------|
| gift 稀疏率 | ~1.48% (per-click) | ±0.5% |
| 金额分布 | P50=2, P99≈1488（重尾） | 相似分布 |
| 用户/主播集中度 | Gini ~0.7-0.8 | ±0.1 |
| pair 绑定强度 | pair_* 特征主导 | 复现绑定关系 |
| 延迟分布 | Weibull + 0 质量点 | 相似分布 |

**验收**：
- 关键统计量矩匹配误差 < 10%
- 策略 A vs B 的相对排序与离线 OPE 一致

---

# 4. 📊 进度追踪

## 4.1 看板

```
⏳计划        🔴就绪        🚀运行        ✅完成        ❌废弃
MVP-1.5       MVP-1.0                                   MVP-0.2
MVP-2.0
```

## 4.2 Gate进度

| Gate | MVP | 状态 | 结果 | 说明 |
|------|-----|------|------|------|
| **Gate-0** | **MVP-1.0** | **🔴 就绪** | - | 无泄漏 Baseline 验证 |
| Gate-1 | MVP-1.0 | ⏳ 等待 | - | Direct vs Two-Stage 复验 |
| Gate-2 | MVP-1.5 | ⏳ 计划 | - | 召回-精排分工 |
| Gate-3 | MVP-2.0 | ⏳ 远期 | - | 模拟器验证 |

## 4.3 结论快照

| MVP | 结论 | 关键指标 | 同步Hub | 状态 |
|-----|------|---------|---------|------|
| ~~0.2~~ | ~~Baseline 性能超预期~~ → **发现数据泄漏，结果无效** | ~~Top-1%=56.2%~~（虚高） | ✅ §2.1 | ❌ 废弃 |
| **1.0** | ❌ 完成-失败：泄漏已消除但性能不达标（Top-1%=11.6%，目标>40%） | Top-1%=11.6%, RevCap@1%=21.6%, Spearman=0.14 | ✅ | ❌ 完成-失败 |
| **1.2** | ✅ 完成：实时特征（watch_time）是最强信号，Top-1%=19.8%（+94% vs baseline），Spearman=0.405 | Top-1%=19.8% (baseline+rt), RevCap@1%=17.3%, Spearman=0.4051 | ✅ | ✅ 完成 |
| **1.3** | ✅ 完成：二分类 AUC=0.61（<0.70），但 Prec@1%=19.24%（>5%），可用于召回；Two-Stage 改进版 RevCap@1%=25.9% | AUC=0.6087, Prec@1%=19.24%, Two-Stage RevCap@1%=25.9% | ✅ | ✅ 完成 |
| **1.4** | ✅ 完成：冷启动是致命瓶颈（61.5% cold-pair），RevCap@1%=3.2%（仅为基线 16%） | cold-pair=3.2%, warm-pair=31.1% | ✅ | ✅ 完成 |
| **1.5** | ✅ 完成：回归严重低估 1500x（pred=0.0008 vs actual=1.22），需 Two-Stage 或校准层 | 分类 ECE=0, 回归 ECE=1.22 | ✅ | ✅ 完成 |
| **1.6** | ✅ 完成：估计层审计通过，发现标签窗口差异显著（16.51%），建议 watch_time 截断 | RevCap@1%=25.9%（Set-1），当前定义可服务在线分配 | ✅ | ✅ 完成 |
| 2.0 | 计划中：模拟器用于策略评估 | 矩匹配误差 < 10% | - | 🟡 计划 |

## 4.4 时间线

| 日期 | 事件 | 影响 |
|------|------|------|
| 2026-01-08 | MVP-0.2 完成（Baseline gift-only） | - |
| **2026-01-18** | **发现 MVP-0.2 存在三个致命问题（泄漏/任务/评估）** | **所有结论需复验** |
| 2026-01-18 | MVP-1.0 立项（Leakage-Free Baseline） | 最高优先级 |
| 2026-01-18 | MVP-1.6 立项（Estimation Layer Audit） | 系统性审计估计层 |
| 2026-01-18 | MVP-1.6 完成（Estimation Layer Audit） | ✅ 审计通过，发现标签窗口问题，RevCap@1%=25.9% |
| 2026-01-18 | MVP-1.4 完成（Slice Evaluation） | ✅ 冷启动是致命瓶颈（61.5% cold-pair，RevCap@1%=3.2%） |
| 2026-01-18 | MVP-1.5 完成（Calibration Evaluation） | ✅ 回归严重低估 1500x，需 Two-Stage 或校准层 |
| 2026-01-18 | MVP-2.0 计划（模拟器 V1） | 远期规划 |

---

# 5. 🔗 文件索引

## 5.1 实验索引

| exp_id | topic | 状态 | MVP | 报告 |
|--------|-------|------|-----|------|
| `EXP-20260108-gift-allocation-02` | Baseline (gift-only) | ❌ 废弃 | ~~MVP-0.2~~ | [exp_baseline_20260108.md](./exp/exp_baseline_20260108.md) |
| `EXP-20260118-gift_EVpred-01` | Leakage-Free Baseline | ❌ 完成-失败 | MVP-1.0 | [exp_leakage_free_baseline_20260118.md](./exp/exp_leakage_free_baseline_20260118.md) |
| `EXP-20260118-gift_EVpred-02` | Rolling Leakage Fix | ✅ 完成 | MVP-1.1 | [exp_rolling_leakage_fix_20260118.md](./exp/exp_rolling_leakage_fix_20260118.md) |
| `EXP-20260118-gift_EVpred-03` | Rolling Leakage Diagnosis | ✅ 完成 | MVP-1.1 | [exp_rolling_leakage_diagnosis_20260118.md](./exp/exp_rolling_leakage_diagnosis_20260118.md) |
| `EXP-20260118-gift_EVpred-04` | Feature Engineering V2 | ✅ 完成 | MVP-1.2 | [exp_feature_engineering_v2_20260118.md](./exp/exp_feature_engineering_v2_20260118.md) |
| `EXP-20260118-gift_EVpred-05` | Binary Classification | ✅ 完成 | MVP-1.3 | [exp_binary_classification_20260118.md](./exp/exp_binary_classification_20260118.md) |
| `EXP-20260118-gift_EVpred-06` | Slice Evaluation | ✅ 完成 | MVP-1.4 | [exp_slice_evaluation_20260118.md](./exp/exp_slice_evaluation_20260118.md) |
| `EXP-20260118-gift_EVpred-07` | Calibration Evaluation | ✅ 完成 | MVP-1.5 | [exp_calibration_evaluation_20260118.md](./exp/exp_calibration_evaluation_20260118.md) |
| `EXP-20260118-gift_EVpred-08` | Estimation Layer Audit | ✅ 完成 | MVP-1.6 | [exp_estimation_layer_audit_20260118.md](./exp/exp_estimation_layer_audit_20260118.md) |
| - | 模拟器 V1 | 🟡 计划 | MVP-2.0 | [exp_simulator_v1_20260118.md](./exp/exp_simulator_v1_20260118.md) |

## 5.2 运行路径

| MVP | 脚本 | 配置 | 输出 |
|-----|------|------|------|
| ~~0.2~~ | ~~`scripts/train_baseline_lgb.py`~~ | - | ❌ 废弃 |
| 1.0 | `scripts/train_leakage_free_baseline.py` | `configs/leakage_free.yaml` | `results/leakage_free/` |
| 2.0 | `scripts/simulator_v1.py` | `configs/simulator.yaml` | `results/simulator/` |

---

# 6. 📎 附录

## 6.1 数值汇总

| MVP | 配置 | Top-1% Capture | Revenue Capture@1% | Spearman | 状态 |
|-----|------|---------------|-------------------|----------|------|
| ~~0.2~~ | ~~gift-only, 有泄漏~~ | ~~56.2%~~ | ~~[未计算]~~ | ~~0.891~~ | ❌ 泄漏无效 |
| **1.0** | click-level, past-only | **11.6%** | **21.6%** | **0.14** | ❌ 完成-失败 |
| **1.2** | click-level, past-only + 实时特征 (baseline+rt) | **19.8%** | **17.3%** | **0.4051** | ✅ 完成 |
| **1.3** | click-level, past-only (二分类 + Two-Stage 改进) | - | **25.9%** (Two-Stage) | - | ✅ 完成 |
| **1.4** | click-level, past-only (切片评估) | - | **20.8%** (全量), **3.2%** (cold), **31.1%** (warm) | **0.096** | ✅ 完成 |
| **1.5** | click-level, past-only (校准评估) | - | - | 分类 ECE=0, 回归 ECE=1.22 | ✅ 完成 |
| **1.6** | click-level, past-only (Frozen), Logistic+Ridge | - | **25.9%** (Set-1 Linear) | **0.034** (Set-1 Linear) | ✅ 完成 |
| 目标 | - | **> 40%** | **> 50%** | **下降 < 0.2** | 验收标准 |

## 6.2 文件索引

| 类型 | 路径 | 说明 |
|------|------|------|
| Roadmap | `gift_EVpred_roadmap.md` | 本文件 |
| Hub | `gift_EVpred_hub.md` | 战略分析 |
| 实验 | `exp/exp_*.md` | 单实验报告 |
| 图表 | `img/` | 可视化 |
| 提示词 | `prompts/coding_prompt_*.md` | Coding Prompt |
| 模型 | `models/` | 训练产物 |
| 结果 | `results/` | 评估结果 |

## 6.3 更新日志

| 日期 | 变更 | 章节 | 影响 |
|------|------|------|------|
| 2026-01-18 | 创建 | - | - |
| 2026-01-18 | 添加 MVP-1.0 (Leakage-Free Baseline) | §2.1, §3 | - |
| 2026-01-18 | 重大更新：发现 MVP-0.2 三个致命问题 | 全文 | 所有结论需复验 |
| | 废弃 MVP-0.2（数据泄漏/任务不匹配/评估偏差） | §3 | Baseline 无效 |
| | 新增 Gate-0（无泄漏验证）为最高优先级 | §1.2 | 阻塞后续实验 |
| | 新增 MVP-2.0（模拟器 V1）计划 | §3 | 远期规划 |
| | 新增问题诊断和解决方案 | §开头 | 关键阻塞 |
| **2026-01-18** | **Gate-0 完成但失败** | §1.2, §2.1 | **需重构特征工程** |
| | MVP-1.0 泄漏已消除（特征重要性比 1.23） | §1.2 | 基础设施正确 |
| | 性能大幅下降（Top-1%=11.6%，目标>40%） | §1.2 | 当前特征预测力极弱 |
| | 新增 MVP-1.1（特征重构）和 MVP-1.2（任务降级）| §1.3 | 最高优先级 |
| **2026-01-18** | **新增评估类实验（MVP-1.4/1.5）** | §2.1, §5.1 | **补全 §2.5 评估体系** |
| | MVP-1.4: Slice Evaluation（切片评估） | §2.1 | 冷启动/头部/长尾分析 |
| | MVP-1.5: Calibration Evaluation（校准评估） | §2.1 | ECE/reliability curve |
| **2026-01-18** | **MVP-1.6 完成（Estimation Layer Audit）** | §1.3, §2.1, §3, §4.3, §6.1 | **审计通过，发现标签窗口问题** |
| | MVP-1.6: 审计通过，RevCap@1%=25.9% | §2.1, §4.3 | 当前定义可服务在线分配 |
| | 发现标签窗口差异显著（16.51%） | §3 | 建议使用 watch_time 截断或更短 H |
| | apply_frozen_features 优化（107x 加速） | - | 性能优化，实现 lookup 缓存 |
| **2026-01-18** | **MVP-1.4/1.5 完成** | §4.3, §6.1 | **切片评估 + 校准评估** |
| | MVP-1.4: 冷启动 pair RevCap@1%=3.2%（仅为基线 16%） | §4.3 | 61.5% 的 test 是冷启动，致命瓶颈 |
| | MVP-1.5: 回归 ECE=1.22，低估 1500x | §4.3 | 分类 ECE=0，回归需校准层或 Two-Stage |
| | 更新数值汇总表和结论快照 | §4.3, §6.1 | 同步 Hub |
| **2026-01-18** | **MVP-1.2/1.3 完成** | §1.3, §2.1, §4.3, §6.1 | **特征工程 + 二分类验证** |
| | MVP-1.2: watch_time 是最强信号，Top-1%=19.8%（+94% vs baseline） | §4.3 | 实时特征有效，但未达 30% 目标 |
| | MVP-1.3: 二分类 AUC=0.61，Prec@1%=19.24%，Two-Stage 改进版 RevCap@1%=25.9% | §4.3 | 二分类可用于召回，Two-Stage 有小幅提升 |
| | 更新数值汇总表：添加 MVP-1.2/1.3 结果 | §6.1 | 同步 Hub |
