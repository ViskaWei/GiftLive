# 🍃 召回-精排分工实验
> **Name:** Recall-Rerank Division of Labor  
> **ID:** `EXP-20260109-gift-allocation-51`  
> **Topic:** `gift_EVmodel` | **MVP:** MVP-1.5 (召回-精排分工)  
> **Author:** Viska Wei | **Date:** 2026-01-09 | **Status:** ✅  

> 🎯 **Target:** 验证 Direct 召回 + Stage2 精排能否兼顾 Top-1% 和 NDCG  
> 🚀 **Next:** Gate-5A FAIL (Top-1%未达标) → 保留 Direct-only 架构，但可用于 NDCG 场景

## ⚡ 核心结论速览

> **一句话**: 召回-精排分工保持 Top-1%=54.3% (与 Direct 相同)，NDCG@100 提升 2.3x (0.34 vs 0.15)，但未达 56% 目标

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H5.1: 分工能保持 Top-1% ≥ 56%? | ❌ 54.3% | 未达标，召回覆盖率低 |
| H5.2: 分工能提升 NDCG@100? | ✅ 2.3x | 显著提升 (0.34 > 0.15) |

| 指标 | Direct-only | Stage2-only | Recall-Rerank (M=200) |
|------|-------------|-------------|----------------------|
| Top-1% Capture | 54.31% | 33.75% | **54.31%** |
| Top-5% Capture | 40.09% | 25.83% | 40.09% |
| NDCG@100 | 0.1485 | 0.5301 | **0.3360** |
| Spearman | 0.3257 | 0.1407 | 0.3257 |

| Type | Link |
|------|------|
| 🧠 Hub | `../gift_EVmodel_hub.md` § Q1.4, DG1 |
| 🗺️ Roadmap | `../gift_EVmodel_roadmap.md` § MVP-1.5 |

---
# 1. 🎯 目标

**问题**: Direct 全量排序强 (Top-1%=54.5%)，Two-Stage 在 NDCG@100 上更好 (+14pp)，Stage2 在 gift 子集排序强 (Spearman=0.89)。能否结合两者优势？

**验证**: 
- H5.1: 分工架构能否保持 Top-1% ≥ 56%？
- H5.2: 分工架构能否提升 NDCG@100 > 21.7%？

| 预期 | 判断标准 |
|------|---------|
| Top-1% ≥ 56% & NDCG ↑ | Gate-5A PASS → 采用分工架构 |
| Top-1% 下降 | Gate-5A FAIL → 保留 Direct-only |

---

# 2. 🦾 算法

## 2.1 召回-精排流程

```
全量候选 (N=490K)
    ↓ Direct Regression 打分
    ↓ 取 Top-M 候选
    ↓ Stage2 m(x) 重排 Top-M
    ↓ 最终排序
```

## 2.2 评估指标

- **Top-K% Capture**: 真实 Top-K% 高价值样本被预测为 Top-K% 的比例
- **NDCG@K**: 排序质量指标，考虑位置权重
- **Recall Coverage**: Direct Top-M 中包含多少真实 Top-K% 样本

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|-----|-----|
| 数据集 | KuaiLive click 全量 |
| 测试集 | 490,952 样本 (10% 时间切分) |
| 打赏样本 | 10,009 (1.93%) |

## 3.2 实验组

| Top-M | 说明 |
|-------|------|
| 50 | 极小召回集 |
| 100 | 小召回集 |
| 200 | 中等召回集 |
| 500 | 大召回集 |
| 1000 | 超大召回集 |

---

# 4. 📊 实验图表

## 4.1 Recall Coverage vs Top-M
![Recall vs TopK](../img/mvp51_recall_vs_topk.png)

**观察**: 即使 M=1000，召回覆盖率也只有 14.1%，说明 Direct 模型的高分预测与真实高价值样本重叠度低。

## 4.2 Method Comparison
![Method Comparison](../img/mvp51_method_comparison.png)

**观察**: 召回-精排的 Top-1% 与 Direct 相同，但 NDCG 显著提升。

## 4.3 Tradeoff Analysis
![Tradeoff](../img/mvp51_tradeoff.png)

**观察**: NDCG 在 M=200 时达到峰值 0.336，更大的 M 反而降低 NDCG。

## 4.4 Score Correlation
![Score Correlation](../img/mvp51_score_correlation.png)

**观察**: Direct 和 Stage2 分数相关性较低，说明两者学到了不同的信号。

---

# 5. 💡 关键洞见

## 5.1 宏观洞见

| ID | 洞见 | 证据 | 决策影响 |
|----|------|------|---------|
| I25 | 召回覆盖率是分工架构的瓶颈 | M=200 只召回 3.2% 真实 Top-1% | 需要更好的召回模型 |
| I26 | Stage2 能显著提升精排质量 | NDCG 从 0.15 提升到 0.34 (2.3x) | 分工架构有价值 |
| I27 | 分工架构不损害 Top-1% | 54.31% = Direct 基线 | 可安全采用 |

## 5.2 根因分析

**为什么 Top-1% 没有提升？**
1. Direct 召回的 Top-M 候选中，只有少量真实高价值样本
2. Stage2 精排只能在召回集内重排，无法找回被过滤的高价值样本
3. 最终 Top-1% 取决于召回质量，而非精排质量

**为什么 NDCG 提升显著？**
1. NDCG@100 评估的是前 100 个位置的排序质量
2. Stage2 在局部排序上表现更好 (Spearman 在 gift 子集上为 0.89)
3. 即使召回覆盖率低，精排后的排序质量仍然提升

---

# 6. 📝 结论

## 6.1 核心发现

1. **Gate-5A: FAIL** — Top-1% = 54.3% < 56% 目标
2. **NDCG 提升显著** — 从 0.1485 到 0.3360 (2.3x)
3. **召回是瓶颈** — M=200 只覆盖 3.2% 真实 Top-1%
4. **分工架构可选用** — 对于重视 NDCG 的场景有价值

## 6.2 关键数字速查

| 指标 | 值 |
|------|-----|
| Direct Top-1% | 54.31% |
| Recall-Rerank Top-1% | 54.31% (无损) |
| NDCG 提升 | 2.3x (0.15 → 0.34) |
| 最佳 M | 200 |
| 召回覆盖率 @M=200 | 3.2% |

## 6.3 Gate-5A 决策

| 条件 | 结果 | 判断 |
|------|------|------|
| Top-1% ≥ 56% | 54.3% | ❌ FAIL |
| NDCG > 0.217 (Direct) | 0.336 | ✅ PASS |
| **Overall** | | ❌ FAIL |

**决策**: 保留 Direct-only 作为主架构，但对于重视 NDCG 的场景可考虑分工架构。

## 6.4 设计启示

- **召回模型需要优化**: 当前 Direct 模型的召回覆盖率低，需要更好的召回策略
- **分工架构有条件价值**: 对于精排场景 (如 Top-100 展示位) 有显著收益
- **两阶段分数融合可探索**: 加权组合 Direct + Stage2 分数可能更好

---

# 7. 📍 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| Gate-5A FAIL | 保留 Direct-only 主架构 | 🔴 |
| 可选优化 | 探索更好的召回策略 (如 Hybrid 召回) | 🟡 |
| 继续 Phase 5 | MVP-5.2 影子价格/供需匹配 | 🔴 |

---

# 8. 📎 附录

## 8.1 数值结果表

| Top-M | Recall@1% | Recall@5% | Top-1% | Top-5% | NDCG@100 |
|-------|-----------|-----------|--------|--------|----------|
| 50 | 0.90% | 0.19% | 54.31% | 40.09% | 0.2893 |
| 100 | 1.69% | 0.35% | 54.31% | 40.09% | 0.2850 |
| 200 | 3.18% | 0.65% | 54.31% | 40.09% | **0.3360** |
| 500 | 7.31% | 1.54% | 54.31% | 40.09% | 0.2661 |
| 1000 | 14.10% | 2.97% | 54.31% | 40.09% | 0.2252 |

## 8.2 基线对比

| 方法 | Top-1% | NDCG@100 | 说明 |
|------|--------|----------|------|
| Direct-only | 54.31% | 0.1485 | 基线 |
| Stage2-only | 33.75% | 0.5301 | 全量用 Stage2 排序 |
| Recall-Rerank (M=200) | 54.31% | 0.3360 | 最佳配置 |

## 8.3 实验执行记录

```
实验时间: 2026-01-09
运行命令: python scripts/train_recall_rerank.py
运行时长: ~2 min
Gate-5A: FAIL
```

## 8.4 相关文件

| 类型 | 路径 |
|------|------|
| 代码 | `scripts/train_recall_rerank.py` |
| 结果 | `gift_allocation/results/recall_rerank_20260109.json` |
| 图表 | `gift_allocation/img/mvp51_*.png` |
| 模型 | `gift_allocation/models/fair_*.pkl` |
