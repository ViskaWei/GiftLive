# 🍃 Two-Stage Diagnosis: Error Decomposition

> **Name:** Two-Stage Diagnostic Decomposition  
> **ID:** `EXP-20260108-gift-allocation-11`  
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.4  
> **Author:** Viska Wei | **Date:** 2026-01-08 | **Status:** ✅

> 🎯 **Target:** 诊断Two-Stage输的主因：Stage2数据不足 / 乘法噪声 / OOD外推  
> 🚀 **Next:** ✅ 主因是Stage1分类不足 → 推荐**召回-精排分工**策略

## ⚡ 核心结论速览

> **一句话**: ✅ **DG1.1 关闭**：Two-Stage 输的主因是 **Stage1 分类能力不足**（Oracle_p 增益 +54.9pp >> Oracle_m +3.1pp），但 **Stage2 在 gift 子集内表现极好**（Spearman=0.89 > Direct=0.74），推荐**召回-精排分工**策略。

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| H1: Stage2数据量不足是主因？ | ❌ | Oracle_m 增益仅 +3.1pp，不是主因 |
| H2: p×m乘法放大误差是主因？ | ❌ | Stage1-only (34.3%) ≈ Two-Stage (35.8%)，乘法影响不大 |
| H3: Stage2 OOD预测问题？ | ❌ | Stage2 在 gift 子集 Spearman=0.89，表现很好 |
| H4: 特征/实验口径存在问题？ | ❌ | 无证据，指标正常 |
| **真正主因** | ✅ | **Stage1 分类不足**：Oracle_p 增益 +54.9pp |

| 指标 | 值 | 启示 |
|------|-----|------|
| Oracle p 上界 | **90.7%** | Stage1 若完美，Top-1% 可达 90% |
| Oracle m 上界 | 38.9% | Stage2 完美仅提升 3pp |
| Stage1-only Top-1% | 34.3% | 与 Two-Stage 35.8% 接近 |
| Stage2 gift子集 Spearman | **0.892** | 远超 Direct 的 0.737，精排优势明显 |

| Type | Link |
|------|------|
| 🧠 Hub | `../gift_EVpred_hub.md` § Q1.1, Q1.4 |
| 🗺️ Roadmap | `../gift_EVpred_roadmap.md` § MVP-1.4 |
| 📗 Prior Exp | `exp_fair_comparison_20260108.md` |

---

# 1. 🎯 目标

**问题**: MVP-1.1-fair 显示 Two-Stage (35.7%) 远落后于 Direct Reg (54.5%)，但 NDCG@100 反而更优 (+14.2pp)。需要拆解误差来源，明确是哪个组件导致整体性能下降。

**验证**: DG1.1 - Two-Stage输的主因是什么？

**假设列表**:
- **H1**: Stage2 数据量不足（34k gift vs 1.87M click）
- **H2**: p×m 乘法在排序上放大误差/引入噪声
- **H3**: Stage2 的 OOD 预测问题（只见 gift 样本分布）
- **H4**: 实验口径/特征可能存在隐藏问题

| 预期 | 判断标准 |
|------|---------|
| H1 成立 | Oracle m >> 实际 Two-Stage，且 Stage2 在 gift 子集表现好 |
| H2 成立 | Oracle p 后 Two-Stage 接近 Direct，但 Stage1-only 已落后 |
| H3 成立 | Stage2 在 gift 子集表现好，但整体外推差 |
| H4 成立 | 审计发现特征泄漏或口径问题 |

---

# 2. 🦾 算法

**误差分解框架**：

$$v(x) = p(x) \cdot m(x)$$

**Stage-wise 评估**：
1. **Stage1-only**: 只用 $p(x)$ 做排序，不乘 $m(x)$
2. **Stage2 on gift subset**: 在 test 中 $Y>0$ 的样本上评估 $m(x)$ 的排序能力

**Oracle 分解**：
1. **Oracle p**: 用真实 $\mathbb{1}(Y>0)$ 替换 $p(x)$，计算 Top-K
2. **Oracle m**: 用真实 $\log(1+Y)$ 替换 $m(x)$ (仅 gift 样本)，计算 Top-K

**理论上界**：
- 若 Oracle p + 实际 m 接近完美 → 问题在 Stage1
- 若实际 p + Oracle m 接近完美 → 问题在 Stage2
- 若两者都需要 Oracle 才好 → 乘法组合本身有问题

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive（复用 MVP-1.1-fair） |
| 路径 | `data/KuaiLive/` |
| Train/Val/Test | 1,872,509 / 1,701,600 / 1,335,406 |
| Test gift 样本 | ~25,757 (1.93%) |
| 时间切分 | 按天，最后7天test |

## 3.2 实验列表

| 实验 | 描述 | 输入 | 输出 |
|------|------|------|------|
| Exp1 | Stage1-only 排序 | p(x) | Top-K, NDCG, Spearman |
| Exp2 | Stage2 gift子集评估 | m(x) on Y>0 | Spearman_gift, NDCG_gift |
| Exp3 | Oracle p 分解 | 1(Y>0) × m(x) | Top-K 上界 |
| Exp4 | Oracle m 分解 | p(x) × log(1+Y) | Top-K 上界 |
| Exp5 | 特征一致性审计 | 特征构造代码 | 有无泄漏 |

## 3.3 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| 排序 | Top-1%/5%/10% Capture | 命中率 |
| 排序 | Spearman | 全局排名相关 |
| 排序 | NDCG@100 | Top-100 精细排序 |
| 子集 | Spearman_gift | 仅 gift 样本排序 |

## 3.4 通过门槛

| 分解结果 | 结论 | 下一步 |
|----------|------|--------|
| Stage1-only 接近 Direct | Stage2/乘法是主因 | → MVP-1.5 改进 Stage2 |
| Stage1-only 远差 | Stage1 也有问题 | → 检查 Stage1 训练 |
| Oracle m >> 实际 | Stage2 数据不足 | → Stage2 正则化 |
| Oracle p >> 实际 | Stage1 预测不准 | → 优化分类器 |
| Stage2 gift子集好，整体差 | OOD 外推问题 | → 考虑分工策略 |

---

# 4. 📊 图表

### Fig 1: Stage-wise Performance Comparison
![](../img/two_stage_diagnosis.png)

**观察**:
- **左图**：Top-1% 对比 - Direct(54.5%) > Two-Stage(35.8%) ≈ Stage1-only(34.3%)
- Stage1-only 与 Two-Stage 差距极小，说明 **m(x) 对整体排序贡献有限**
- Oracle_p 可达 90.7%，说明 Stage1 分类是瓶颈
- **中图**：Oracle 增益 - Oracle_p(+54.9pp) >> Oracle_m(+3.1pp)
- **右图**：Gift 子集 Spearman - Stage2(0.89) > Direct(0.74)，精排能力更强

---

# 5. 💡 洞见

## 5.1 宏观
- **Two-Stage 更适合精排而非全量排序**：Stage2 在 gift 子集内表现极好（Spearman=0.89），但整体排序受限于 Stage1 分类能力
- **Stage1 分类瓶颈**：Oracle_p 增益 +54.9pp 说明"完美分类"可大幅提升 Top-1%，但实际 Stage1 分类能力不足

## 5.2 模型层
- **Stage1 问题**：分类器 Top-1% = 34.3%，低于 Direct 的 54.5%，说明 p(x) 排序能力弱
- **Stage2 优势**：在已知 gift 的样本内，m(x) 能很好地区分金额大小（Spearman=0.89）
- **乘法组合影响小**：Stage1-only(34.3%) ≈ Two-Stage(35.8%)，p×m 不是问题

## 5.3 细节
- **NDCG@100 悖论**：Two-Stage NDCG@100=0.37 > Direct=0.22，说明在 Top-100 内排序更准
- **Binary 上限**：完美二分类 Top-1%=45%，而 Direct 达到 54.5%，说明 Direct 学到了金额信息
- **Stage2 OOD 不是问题**：Stage2 在 gift 子集表现好，说明回归模型本身没问题

---

# 6. 📝 结论

## 6.1 核心发现

> **Two-Stage 输的主因是 Stage1 分类不足，而非 Stage2 或乘法组合。Stage2 在 gift 子集内表现极好，推荐召回-精排分工策略。**

- ✅ **DG1.1 关闭**：主因已明确 = Stage1 分类能力不足
- ✅ **推荐策略**：Direct 做召回，Two-Stage(m(x)) 做精排

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **Stage1 分类是瓶颈** | Oracle_p 增益 +54.9pp >> Oracle_m +3.1pp |
| 2 | **Stage2 在精排场景有优势** | Gift 子集 Spearman: 0.89 > 0.74 |
| 3 | **乘法组合影响小** | Stage1-only (34.3%) ≈ Two-Stage (35.8%) |
| 4 | **Direct 学到了金额信息** | Top-1% 54.5% > Binary上限 45% |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| 召回-精排分工 | Direct 做 Top-M 召回，Two-Stage 做 re-rank |
| Stage1 优化 | 提升分类器 PR-AUC/Top-K 精度 |
| Stage2 复用 | m(x) 可用于精排场景，无需改进 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Direct Top-1% | **54.5%** | 全量测试集 |
| Two-Stage Top-1% | 35.8% | 全量测试集 |
| Stage1-only Top-1% | 34.3% | 只用 p(x) |
| Oracle_p Top-1% | 90.7% | 完美分类 × m(x) |
| Oracle_m Top-1% | 38.9% | p(x) × 完美回归 |
| Stage2 Spearman (gift) | **0.892** | gift 子集 |
| Direct Spearman (gift) | 0.737 | gift 子集 |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| ✅ 采用 | **召回-精排分工**：Direct 召回 + Two-Stage 精排 | 🟡 MVP-1.5 |
| 🟡 可选 | 优化 Stage1 分类器（更多特征/更深模型） | P2 |
| ❌ 关闭 | 全量 Two-Stage 替代 Direct | - |

---

# 7. 📎 附录

## 7.1 实验设计详细说明

### Exp1: Stage1-only 排序能力
```python
# 只用 p(x) 做排序，不乘 m(x)
score = p_pred  # Stage1 预测的打赏概率
# 计算 Top-K Capture, NDCG, Spearman
```

### Exp2: Stage2 在 gift 子集的能力
```python
# 筛选 test 中 Y > 0 的样本
gift_mask = y_test > 0
m_gift = m_pred[gift_mask]
y_gift = y_test[gift_mask]
# 计算 Spearman(m_gift, y_gift)
```

### Exp3: Oracle p 分解
```python
# 用真实 1(Y>0) 替换 p(x)
oracle_p = (y_test > 0).astype(float)
score = oracle_p * m_pred
# 计算 Top-K Capture
```

### Exp4: Oracle m 分解
```python
# 用真实 log(1+Y) 替换 m(x)（仅 gift 样本）
oracle_m = np.where(y_test > 0, np.log1p(y_test), m_pred)
score = p_pred * oracle_m
# 计算 Top-K Capture
```

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/diagnose_two_stage.py` |
| 依赖数据 | MVP-1.1-fair 模型输出 |
| Output | `gift_allocation/results/two_stage_diagnosis_20260108.json` |
| 测试集大小 | 1,335,406 样本, 29,349 gifts (2.20%) |
| 运行时间 | 112.4s |

```bash
# 运行实验
source init.sh
python scripts/diagnose_two_stage.py
```

## 7.3 数值结果完整表

| 配置 | Top-1% | Top-5% | NDCG@100 | Spearman |
|------|--------|--------|----------|----------|
| Direct Regression | **54.5%** | 41.4% | 21.7% | 0.331 |
| Two-Stage (p×m) | 35.8% | 40.4% | **37.1%** | 0.247 |
| Stage1-only (p) | 34.3% | 42.2% | 0.1% | **0.585** |
| Oracle p (×m) | **90.7%** | 100% | 70.6% | - |
| Oracle m (p×) | 38.9% | 38.4% | 85.1% | - |
| Binary 上限 | 45.0% | - | - | - |

---

> **实验完成时间**: 2026-01-08
