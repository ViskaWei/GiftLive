# 📗 实验报告: Simulator V2 - 并发容量建模

> **Experiment ID:** `EXP-15`  
> **MVP:** MVP-4.2  
> **作者:** Viska Wei  
> **日期:** 2026-01-09  
> **状态:** ⚠️ 部分验证

---

## 🔗 上游追溯

| 来源 | 链接 |
|------|------|
| Coding Prompt | [coding_prompt_mvp42_concurrency_20260109.md](../prompts/coding_prompt_mvp42_concurrency_20260109.md) |
| Roadmap | [gift_allocation_roadmap.md](../gift_allocation_roadmap.md) |
| Phase Charter | [gift_allocation_phase4_charter.md](../gift_allocation_phase4_charter.md) |

---

## ⚡ 核心结论速览

### 一句话总结
**当前配置下容量约束未触发 (overcrowded_ratio < 1%)，收益随并发线性增长，无边际递减现象**

### 假设验证
| 假设 | 结果 | 备注 |
|------|------|------|
| H4.2: 并发容量影响收益边际 | ⚠️ 条件未满足 | 容量设置过高，未触发瓶颈 |

### 关键数字

| 指标 | 结果 |
|------|------|
| Revenue Diff (有/无容量) | **0%** |
| Marginal Decreasing | **❌ 未观测到** |
| Max Overcrowded Ratio | **0.02%** (@800 users) |
| Gate-4A Extended | **⚠️ CONDITIONAL** |

---

## 1. 🎯 目标

**实验目的**: 验证主播并发容量约束对收益的边际递减效应

**预期结果**:
- 若观测到边际递减 → 继续 MVP-4.3
- 若无效应 → 调整参数或重新设计

---

## 2. 🧪 实验设计

### 2.1 容量配置

```yaml
capacity_by_tier:
  top_10%: 100 users/streamer
  middle_40%: 50 users/streamer  
  tail_50%: 20 users/streamer

crowding_penalty:
  type: inverse
  formula: 1 / (1 + beta * overflow_ratio)
  beta: 0.5
```

### 2.2 实验矩阵

| 实验 | 变量 | 范围 |
|------|------|------|
| Exp 1 | 容量开关 | off / on |
| Exp 2 | 并发用户数 | 50, 100, 200, 400, 800 |
| Exp 3 | 惩罚强度 Beta | 0.1, 0.3, 0.5, 1.0, 2.0 |
| Exp 4 | 容量倍数 | 0.5x, 1.0x, 2.0x |

### 2.3 基础配置

```yaml
n_users: 10000
n_streamers: 100
amount_version: 3  # V2+ discrete tiers
n_rounds: 50
n_simulations: 50
```

---

## 3. 📊 实验图表

### Fig 1: 收益 vs 并发

![Revenue vs Concurrency](../img/mvp42_revenue_vs_concurrency.png)

**观察**: 收益随并发**线性增长**，无饱和迹象

### Fig 2: 边际收益

![Marginal Revenue](../img/mvp42_marginal_revenue.png)

**观察**: 边际收益**递增**而非递减，表明容量未达瓶颈

### Fig 3: 容量效果对比

![Capacity Comparison](../img/mvp42_capacity_comparison.png)

**观察**: 有/无容量约束的收益和 Gini **完全相同**

### Fig 4: Beta 敏感性

![Beta Sweep](../img/mvp42_param_heatmap.png)

**观察**: 惩罚强度 Beta 变化对收益**无影响**

### Fig 5: 用户效率

![Revenue per User](../img/mvp42_streamer_revenue_by_tier.png)

**观察**: 每用户收益略有上升趋势

---

## 4. 💡 关键洞见

### 4.1 宏观发现

1. **容量未触发**: 当前配置下 overcrowded_ratio ≤ 0.02%
2. **线性增长**: 收益 ∝ 并发用户数（斜率 ≈ 20）
3. **配置不当**: 容量设置过高相对于用户数

### 4.2 根因分析

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 无拥挤 | 平均负载 = 800/100 = 8 << 最低容量 20 | 降低容量或增加用户 |
| 无差异 | Greedy Policy 分散分配 | 尝试 Popularity-weighted 分配 |
| 无边际递减 | 远离饱和区 | 需制造人为拥挤场景 |

### 4.3 数值证据

```
Concurrency Sweep:
  50 users  → overcrowded: 0.00%, revenue_per_user: 0.46
  100 users → overcrowded: 0.00%, revenue_per_user: 0.44
  200 users → overcrowded: 0.00%, revenue_per_user: 0.48
  400 users → overcrowded: 0.00%, revenue_per_user: 0.48
  800 users → overcrowded: 0.02%, revenue_per_user: 0.50
```

---

## 5. 📝 结论

### 5.1 核心发现

1. ✅ **容量机制已实现**: 代码逻辑正确，惩罚函数已集成
2. ⚠️ **实验配置不当**: 容量远高于实际负载
3. ❌ **未观测到边际递减**: 需调整实验设计

### 5.2 设计启示

| 编号 | 原则 | 依据 |
|------|------|------|
| D-4.2.1 | 容量设置需贴近真实负载 | overcrowded_ratio → 0 |
| D-4.2.2 | 需测试极端拥挤场景 | 当前配置过于保守 |
| D-4.2.3 | 分配策略影响拥挤程度 | Greedy 分散分配避免拥挤 |

### 5.3 下一步建议

**选项 A**: 重新设计实验
- 降低容量: top_10%=20, middle=10, tail=5
- 增加并发: users_per_round = 1000+
- 使用 Popularity-weighted Policy 制造头部拥挤

**选项 B**: 接受当前结论
- 结论: "在合理配置下，容量不是瓶颈"
- 继续 MVP-4.3

**建议**: 选项 A - 补充拥挤场景验证

---

## 6. 📎 附录

### 6.1 数值结果表

| 实验 | 配置 | Revenue | Gini |
|------|------|---------|------|
| Baseline-off | enable_capacity=False | 4816 | 0.868 |
| Baseline-on | enable_capacity=True | 4816 | 0.868 |
| Concurrency-50 | upr=50 | 1160 | - |
| Concurrency-800 | upr=800 | 19983 | - |
| Beta-0.1 | beta=0.1 | 9677 | 0.838 |
| Beta-2.0 | beta=2.0 | 9677 | 0.838 |
| Scale-0.5x | capacity*0.5 | 9675 | 0.838 |
| Scale-2.0x | capacity*2.0 | 9677 | 0.838 |

### 6.2 实验流程

```bash
source init.sh
python scripts/run_simulator_experiments.py --mvp 4.2 --n_sim 50
```

### 6.3 相关文件

| 类型 | 路径 |
|------|------|
| 结果 JSON | `gift_allocation/results/concurrency_capacity_20260109.json` |
| 图表 | `gift_allocation/img/mvp42_*.png` |
| 代码 | `scripts/run_simulator_experiments.py` |
| 模拟器 | `scripts/simulator/simulator.py` |

---

## 7. 📝 后续实验建议

### MVP-4.2b: 拥挤场景验证 (建议)

```yaml
experiment:
  name: "Crowding Stress Test"
  changes:
    capacity_top10: 20  # 降低 5x
    capacity_middle: 10
    capacity_tail: 5
    users_per_round: [200, 400, 600, 800, 1000]
  policy: PopularityWeightedPolicy  # 制造头部拥挤
  expected: overcrowded_ratio > 10%
```
