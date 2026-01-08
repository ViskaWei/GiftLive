# 📌 Next Steps

> **最后更新**: 2026-01-09

---

## 🔴 P0 - 必须完成

| # | 任务 | 状态 | 验收标准 | 备注 |
|---|------|------|----------|------|
| 1 | **MVP-4.1 Simulator V2 - 金额校准** | 🔴 就绪 | P50/P90误差<30% | 混合 Lognormal+Pareto 分位匹配 |
| 2 | **MVP-4.2 Simulator V2 - 并发容量** | 🔴 就绪 | 拥挤边际递减可观测 | 加入 C_s 与惩罚函数 |
| 3 | 编写 `scripts/simulator/simulator_v2.py` | ⏳ | 通过校准测试 | 继承 V1，加入新机制 |

---

## 🟡 P1 - 应该完成

| # | 任务 | 状态 | 验收标准 | 备注 |
|---|------|------|----------|------|
| 1 | **MVP-4.3 召回-精排分工** | 🔴 就绪 | Top-1% ≥56% | Direct召回 + Stage2精排 |
| 2 | **MVP-4.4 供需匹配/影子价格** | 🔴 就绪 | 收益+5% vs Greedy | 拉格朗日对偶分配 |
| 3 | 编写 `scripts/train_recall_rerank.py` | ⏳ | 实验报告 | 召回-精排流水线 |
| 4 | 编写 `scripts/simulator/policies_v2.py` | ⏳ | 包含影子价格策略 | 新分配策略实现 |

---

## 🟢 P2 - 可选完成

| # | 任务 | 状态 | 验收标准 | 备注 |
|---|------|------|----------|------|
| 1 | MVP-4.5 鲸鱼分散 (b-matching) | ⏳ | 超载率<10% | 依赖 Simulator V2 |
| 2 | MVP-4.6 不确定性排序 (UCB) | ⏳ | CVaR 改善 | 分位数回归或 ensemble |
| 3 | MVP-4.7 多目标生态调度 | ⏳ | Pareto 前沿 | 权重扫描实验 |

---

## ✅ 已完成

| # | 任务 | 完成日期 | 产出 |
|---|------|---------|------|
| **Phase 0-3 (12 MVP)** |||
| 1 | MVP-0.1 KuaiLive EDA | 2026-01-08 | Gini=0.94, Top1%=60% |
| 2 | MVP-0.2 Baseline | 2026-01-08 | Top-1%=56.2%, Spearman=0.891 |
| 3 | MVP-0.3 Simulator V1 | 2026-01-08 | Gini误差<5%, Greedy 3x Random |
| 4 | MVP-1.1 两段式建模 | 2026-01-08 | 揭示与Baseline不可直接对比 |
| 5 | MVP-1.1-fair 公平对比 | 2026-01-08 | Direct胜出 (54.5% vs 35.7%) |
| 6 | MVP-1.2 延迟建模 | 2026-01-08 | 延迟不是问题，DG2关闭 |
| 7 | MVP-1.2-audit 延迟审计 | 2026-01-08 | 代码bug修复，86.3%立即发生 |
| 8 | MVP-1.3 多任务学习 | 2026-01-08 | 无优势 (Δ=-1.76pp)，DG5关闭 |
| 9 | MVP-1.4 诊断拆解 | 2026-01-08 | Stage1分类不足，Stage2精排有优势 |
| 10 | MVP-2.1 凹收益分配 | 2026-01-08 | Δ=-1.17%无优势，DG3关闭 |
| 11 | MVP-2.2 冷启动约束 | 2026-01-08 | 软约束+32%收益，Gate-2关闭 |
| 12 | MVP-3.1 OPE验证 | 2026-01-08 | SNIPS RelErr<10%，Gate-3关闭 |
| **Phase 4 立项** |||
| 13 | Phase 4 立项书 | 2026-01-09 | [立项书](../gift_allocation/gift_allocation_phase4_charter.md) |
| 14 | Hub/Roadmap 更新 | 2026-01-09 | 加入 Phase 4 预览 |

---

## 📊 Phase 4 决策树

```
MVP-4.1 + 4.2 (Simulator V2)
    ├── If 金额校准误差<30% && 并发可观测 → Gate-4A 通过
    │   ├── → MVP-4.4 供需匹配
    │   └── → MVP-4.5 鲸鱼分散
    └── Else → 调参迭代

MVP-4.3 (召回-精排分工) [独立]
    ├── If Top-1% ≥56% → Gate-4B 通过 → 采用分工架构
    └── Else → 保留 Direct-only

MVP-4.4 (供需匹配)
    ├── If 收益≥+5% vs Greedy → Gate-4C 通过 → 替换 Greedy
    └── Else → 保留 Greedy + 软约束
```

---

## 📅 里程碑

| 周 | 任务 | 产出 | 状态 |
|----|------|------|------|
| W1 | MVP-4.1 金额校准 | Simulator V2 (金额) | ⏳ |
| W1-2 | MVP-4.2 并发容量 | Simulator V2 (完整) | ⏳ |
| W2 | MVP-4.3 召回-精排 | Top-1%≥56% | ⏳ |
| W3 | MVP-4.4 供需匹配 | 影子价格分配策略 | ⏳ |
| W3-4 | MVP-4.5/4.6 鲸鱼+UCB | 辅助策略 | ⏳ |
| W4 | MVP-4.7 多目标 | Pareto 前沿 | ⏳ |
| W4 | Hub/Roadmap 更新 | Phase 4 闭环 | ⏳ |

---

## 📝 快速命令

### 查看立项书
```bash
cat gift_allocation/gift_allocation_phase4_charter.md
```

### 查看当前状态
```bash
cat status/kanban.md
```

### 开始 MVP-4.1
```bash
# 1. 创建 coding prompt
# 2. 编写 simulator_v2.py
# 3. 运行校准测试
# 4. 更新实验报告
```
