# 🗺️ Gift Allocation Roadmap

> **Status:** ✅ Phase 0-3 完成 (12/13 MVP) | 📝 Phase 4 立项 (7 MVP) | **Data:** KuaiLive | **Date:** 2026-01-09

## 📋 MVP 总览

| MVP | 名称 | 状态 | 关键结论 |
|-----|------|------|---------|
| 0.1 | KuaiLive EDA | ✅ | Gini=0.94, 打赏率=1.48% |
| 0.2 | Baseline | ✅ | Top-1%=56.2%, Spearman=0.89 |
| 0.3 | Simulator V1 | ✅ | Gini误差<5%, Greedy 3x Random |
| 1.1 | Two-Stage | ✅ | 与Baseline不可直接对比 |
| 1.1-fair | 公平对比 | ✅ | **Direct胜出** (54.5% vs 35.7%) |
| 1.2 | 延迟建模 | ✅ | 延迟不是问题，DG2关闭 |
| 1.2-audit | 延迟审计 | ✅ | 代码bug修复，86.3%立即发生 |
| 1.3 | 多任务 | ✅ | 无优势 (Δ=-1.76pp)，DG5关闭 |
| 1.4 | 诊断拆解 | ✅ | Stage1分类不足，Stage2精排有优势 |
| 1.5 | 召回-精排 | ⏳ | 可选优化 |
| 2.1 | 凹收益 | ✅ | Δ=-1.17%无优势，DG3关闭 |
| 2.2 | 冷启动 | ✅ | 软约束+32%收益，Gate-2关闭 |
| 3.1 | OPE验证 | ✅ | SNIPS RelErr<10%，Gate-3关闭 |

## ✅ Gate 状态

| Gate | MVP | 状态 | 结论 |
|------|-----|------|------|
| Gate-1 | 1.1~1.4, 1.2-audit, 1.3 | ✅ 完成 | 直接回归胜出；延迟/多任务无效 |
| Gate-2 | 2.1, 2.2 | ✅ 关闭 | Greedy + 软约束冷启动 (λ=0.5) |
| Gate-3 | 3.1 | ✅ 关闭 | SNIPS可用 (ε≥0.3, N≥5000) |

## 📊 核心数值

| 类别 | 指标 | 值 | MVP |
|------|------|-----|-----|
| **数据特征** | 打赏率 | 1.48% | 0.1 |
| | User Gini | 0.942 | 0.1 |
| | 矩阵密度 | 0.0064% | 0.1 |
| **预测性能** | Direct Top-1% | 54.5% | 1.1-fair |
| | Two-Stage Top-1% | 35.7% | 1.1-fair |
| | Two-Stage NDCG@100 | 0.359 | 1.1-fair |
| | Stage2 gift子集Spearman | 0.892 | 1.4 |
| | Single-Task PR-AUC | 0.182 | 1.3 |
| | Multi-Task PR-AUC | 0.165 | 1.3 |
| **分配性能** | Greedy/Random收益比 | 2.94x | 0.3 |
| | 软约束收益提升 | +32% | 2.2 |
| | 软约束成功率提升 | +263% | 2.2 |
| | 凹收益 vs Greedy | -1.17% | 2.1 |
| **评估性能** | SNIPS RelErr (Softmax) | 0.57% | 3.1 |
| | SNIPS RelErr (Greedy) | 9.97% | 3.1 |
| | Simulator Gini误差 | <5% | 0.3 |

## 🏗️ 推荐架构

```
估计层: Direct Regression (LightGBM, log(1+Y))
    → Top-1% Capture = 54.5%
    → 交互历史特征最重要 (pair_gift_mean)
    ↓
分配层: Greedy + 软约束冷启动
    → λ=0.5, min_alloc=10
    → 收益+32%, 成功率+263%
    ↓
评估层: SNIPS OPE
    → 探索率ε≥0.3, 日志量N≥5000
    → RelErr < 10%
```

## 📁 实验索引

| exp_id | 名称 | MVP | 报告 |
|--------|------|-----|------|
| EXP-01 | KuaiLive EDA | 0.1 | [exp_kuailive_eda_20260108.md](./exp/exp_kuailive_eda_20260108.md) |
| EXP-02 | Baseline | 0.2 | [exp_baseline_20260108.md](./exp/exp_baseline_20260108.md) |
| EXP-03 | Two-Stage | 1.1 | [exp_two_stage_20260108.md](./exp/exp_two_stage_20260108.md) |
| EXP-04 | Fair Comparison | 1.1-fair | [exp_fair_comparison_20260108.md](./exp/exp_fair_comparison_20260108.md) |
| EXP-05 | Delay Modeling | 1.2 | [exp_delay_modeling_20260108.md](./exp/exp_delay_modeling_20260108.md) |
| EXP-06 | Multi-Task | 1.3 | [exp_multitask_20260108.md](./exp/exp_multitask_20260108.md) |
| EXP-07 | Simulator V1 | 0.3 | [exp_simulator_v1_20260108.md](./exp/exp_simulator_v1_20260108.md) |
| EXP-08 | Concave Allocation | 2.1 | [exp_concave_allocation_20260108.md](./exp/exp_concave_allocation_20260108.md) |
| EXP-09 | Coldstart Constraint | 2.2 | [exp_coldstart_constraint_20260108.md](./exp/exp_coldstart_constraint_20260108.md) |
| EXP-10 | OPE Validation | 3.1 | [exp_ope_validation_20260108.md](./exp/exp_ope_validation_20260108.md) |
| EXP-11 | Two-Stage Diagnosis | 1.4 | [exp_two_stage_diagnosis_20260108.md](./exp/exp_two_stage_diagnosis_20260108.md) |
| EXP-13 | Delay Audit | 1.2-audit | [exp_delay_audit_20260108.md](./exp/exp_delay_audit_20260108.md) |
| EXP-14 | Simulator V2 Enhanced | 4.1+ | [exp_simulator_v2_20260109.md](./exp/exp_simulator_v2_20260109.md) |

---

## 🚀 Phase 4: 资源约束与生态健康 (立项中)

> **Status:** 📝 立项 | **Date:** 2026-01-09 | **目标:** 解决真实场景的资源约束问题

### Phase 4 核心命题

Phase 0-3 在**仿真器**中验证了分配策略，但仍缺乏对真实场景的建模：
1. 主播并发承载上限与拥挤外部性
2. 鲸鱼（大哥）的互斥分配与分散
3. 长期生态健康（留存/满意度）
4. 金额分布的准确校准

### Phase 4 MVP 列表

| MVP | 名称 | 优先级 | 状态 | 验收标准 |
|-----|------|--------|------|----------|
| 4.1+ | Simulator V2 - 增强版 | 🔴 P0 | 🔄 立项 | 离散档位+预算+社交+时序, P50/P90误差<30% |
| 4.2 | Simulator V2 - 并发容量 | 🔴 P0 | ⏳ | 拥挤边际递减可观测 |
| 4.3 | 召回-精排分工 | 🟡 P1 | ⏳ | Top-1% ≥56% |
| 4.4 | 供需匹配/影子价格 | 🟡 P1 | ⏳ | 收益+5% vs Greedy |
| 4.5 | 鲸鱼分散 (b-matching) | 🟢 P2 | ⏳ | 超载率<10% |
| 4.6 | 不确定性排序 (UCB) | 🟢 P2 | ⏳ | CVaR 改善 |
| 4.7 | 多目标生态调度 | 🟢 P2 | ⏳ | Pareto 前沿 |

### Phase 4 决策门

| Gate | 条件 | 通过动作 |
|------|------|----------|
| Gate-4A | Simulator V2 校准误差<30% | 继续 4.4/4.5 |
| Gate-4B | 召回-精排 Top-1%≥56% | 采用分工架构 |
| Gate-4C | 影子价格收益≥+5% | 替换 Greedy |

**详见**: [Phase 4 立项书](./gift_allocation_phase4_charter.md)

---

## 🔗 导航

| 文件 | 用途 |
|------|------|
| [Hub](./gift_allocation_hub.md) | 核心结论、问题树 |
| [Phase 4 立项](./gift_allocation_phase4_charter.md) | 下一阶段规划 |
| [exp/](./exp/) | 12个实验报告 |
| [results/](./results/) | 数值结果JSON |
