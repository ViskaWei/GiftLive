# 📋 Kanban

> **最后更新**: 2026-01-09

---

## 看板

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   ⏳ 计划    │   🔴 就绪    │   🚀 运行    │   ✅ 完成    │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ MVP-4.5     │ MVP-4.1     │             │ Phase 0-3   │
│ MVP-4.6     │ MVP-4.2     │             │ (12 MVP)    │
│ MVP-4.7     │ MVP-4.3     │             │             │
│             │ MVP-4.4     │             │ Phase 4     │
│             │             │             │ 立项 ✓      │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

---

## 状态说明

| 状态 | 含义 |
|------|------|
| ⏳ 计划 | 已规划，待执行 |
| 🔴 就绪 | 准备就绪，等待启动 |
| 🚀 运行 | 正在执行中 |
| ✅ 完成 | 已完成 |
| ❌ 取消 | 已取消 |

---

## 按主题分类

### Gift Allocation - Phase 4

| MVP | 名称 | 状态 | 优先级 | 链接 |
|-----|------|------|--------|------|
| 4.1 | Simulator V2 - 金额校准 | 🔴 就绪 | P0 | [立项书](../gift_allocation/gift_allocation_phase4_charter.md#41-mvp-41-simulator-v2---金额校准) |
| 4.2 | Simulator V2 - 并发容量 | 🔴 就绪 | P0 | [立项书](../gift_allocation/gift_allocation_phase4_charter.md#42-mvp-42-simulator-v2---并发容量) |
| 4.3 | 召回-精排分工 | 🔴 就绪 | P1 | [立项书](../gift_allocation/gift_allocation_phase4_charter.md#43-mvp-43-召回-精排分工) |
| 4.4 | 供需匹配/影子价格 | 🔴 就绪 | P1 | [立项书](../gift_allocation/gift_allocation_phase4_charter.md#44-mvp-44-供需匹配影子价格贪心) |
| 4.5 | 鲸鱼分散 (b-matching) | ⏳ 计划 | P2 | [立项书](../gift_allocation/gift_allocation_phase4_charter.md#45-mvp-45-鲸鱼分散-b-matching) |
| 4.6 | 不确定性排序 (UCB) | ⏳ 计划 | P2 | [立项书](../gift_allocation/gift_allocation_phase4_charter.md#46-mvp-46-不确定性排序-ucb) |
| 4.7 | 多目标生态调度 | ⏳ 计划 | P2 | [立项书](../gift_allocation/gift_allocation_phase4_charter.md#47-mvp-47-多目标生态调度) |

### Gift Allocation - Phase 0-3 (已完成)

| MVP | 名称 | 状态 | 核心结论 |
|-----|------|------|---------|
| 0.1 | KuaiLive EDA | ✅ | Gini=0.94, 打赏率=1.48% |
| 0.2 | Baseline | ✅ | Top-1%=56.2%, Spearman=0.89 |
| 0.3 | Simulator V1 | ✅ | Gini误差<5%, Greedy 3x Random |
| 1.1-fair | 公平对比 | ✅ | Direct胜出 (54.5% vs 35.7%) |
| 1.2+audit | 延迟建模 | ✅ | 延迟不是问题，DG2关闭 |
| 1.3 | 多任务 | ✅ | 无优势，DG5关闭 |
| 1.4 | 诊断拆解 | ✅ | Stage1分类不足，Stage2精排有优势 |
| 2.1 | 凹收益 | ✅ | Δ=-1.17%无优势，DG3关闭 |
| 2.2 | 冷启动 | ✅ | 软约束+32%收益，Gate-2关闭 |
| 3.1 | OPE验证 | ✅ | SNIPS RelErr<10%，Gate-3关闭 |

---

## 本周重点

| 优先级 | 任务 | 状态 | 验收标准 |
|--------|------|------|----------|
| 🔴 P0 | MVP-4.1 金额校准 | 🔴 就绪 | P50/P90误差<30% |
| 🔴 P0 | MVP-4.2 并发容量 | 🔴 就绪 | 拥挤边际递减可观测 |
| 🟡 P1 | MVP-4.3 召回-精排 | 🔴 就绪 | Top-1% ≥56% |
| 🟡 P1 | MVP-4.4 供需匹配 | 🔴 就绪 | 收益+5% |
