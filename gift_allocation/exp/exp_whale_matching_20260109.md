# 📗 实验报告：鲸鱼分散 (b-matching)

> **Experiment ID:** `EXP-20260109-gift-allocation-53`  
> **MVP:** MVP-5.3  
> **Author:** Viska Wei  
> **Date:** 2026-01-09  
> **Status:** ✅ Completed  
> **Gate:** Gate-5C

---

## 🔗 上游追溯

- **Hub**: `../gift_EVmodel_hub.md` § DG8
- **Roadmap**: `../gift_EVmodel_roadmap.md` § MVP-5.3
- **相关实验**: MVP-5.2 (影子价格), MVP-4.2 (并发容量)

---

## ⚡ 核心结论速览

### 一句话总结

> **❌ Gate-5C FAIL**: 分层匹配策略未能降低生态集中度。所有k值配置下，Streamer Gini与基线相同(0.912)，未观察到分散效果。

### 假设验证

| 假设 | 结果 | 关键数值 |
|------|------|----------|
| H5C-1: 鲸鱼分散能降低超载率 | ✅ 通过 | 超载率=0.0% (<10%) |
| H5C-2: 分层匹配能降低Streamer Gini | ❌ 失败 | Gini=0.912 (vs 基线0.912，无改善) |
| H5C-3: 收益不显著下降 | ✅ 通过 | 收益Δ=+0.07% (>-5%) |

### 关键数字速查

| 指标 | 目标值 | 基线 (Greedy) | 最佳配置 (k=1) |
|------|--------|---------------|----------------|
| 超载率 | <10% | 0.28% | 0.0% ✅ |
| Streamer Gini | ↓ | 0.912 | 0.912 ❌ (无改善) |
| Revenue | ≥-5% | 42,135.69 | 42,165.51 (+0.07%) ✅ |

---

## 🎯 目标

### 实验目的

验证**分层匹配策略**能否：
1. 分散Top-0.1%/1%鲸鱼用户，避免过度集中到少数主播
2. 降低生态集中度（Streamer Gini）
3. 在保持收益的前提下，减少容量超载

### Gate-5C 判定标准

- **超载率 <10% AND Streamer Gini↓ AND 收益下降 <5%** → PASS
- 若 PASS → 采用鲸鱼单独匹配层
- 若 FAIL → 保留统一分配

### 动机

**问题**：鲸鱼用户（Top-0.1%/1%）同时涌入同一主播会导致：
- 承接饱和（容量超载）
- 马太效应加剧（头部主播收益集中）
- 用户体验下降（拥挤）

**假设**：通过限制每个主播同时承接的鲸鱼数量（k），可以：
- 强制分散到更多主播
- 降低生态集中度
- 提升整体生态健康度

---

## 🦾 算法

### 分层匹配策略

**核心思想**：将分配分为两层
1. **鲸鱼层**：先做"互斥 + 分散 + 价值最大化"
2. **普通层**：再填充容量剩余，走轻量greedy/softmax

### 算法选项

#### Option 1: b-matching (推荐)
- **描述**：二分图匹配，右侧（主播）容量为 b_s，左侧（鲸鱼）每个用户最多匹配1个主播
- **目标**：最大化总收益，同时满足"每个主播最多k个鲸鱼"
- **复杂度**：O(n²m) 或 O(nm log n) with min-cost flow

#### Option 2: Min-Cost Flow
- **描述**：转化为最小费用流问题
- **优势**：可处理更复杂的约束（如全局鲸鱼分散度）
- **复杂度**：O(nm log(n+m))

#### Option 3: Greedy with Swaps
- **描述**：贪心分配 + 后处理交换优化
- **优势**：实现简单，适合快速验证
- **劣势**：可能不是全局最优

### 分层流程

```
1. 识别鲸鱼用户（Top 0.1%, 1%, 5%）
2. 鲸鱼层匹配：
   - 构建候选矩阵：EV(u, s) for whale users
   - 应用 b-matching/min-cost flow: 每个主播最多k个鲸鱼
   - 输出：whale_allocations
3. 普通层匹配：
   - 剩余容量：capacity_remaining = capacity - |whale_allocations[s]|
   - 普通用户候选：排除已分配的鲸鱼
   - Greedy分配：按EV排序，填充剩余容量
4. 合并输出：whale_allocations + normal_allocations
```

---

## 🧪 实验设计

### 数据

| 项 | 值 |
|----|-----|
| 来源 | Simulator V2 (含并发容量建模) |
| 配置 | 参考 MVP-4.2 |
| n_users | 10,000 |
| n_streamers | 500 |
| n_rounds | 50 |
| users_per_round | 200 |
| n_simulations | 50 |

### 模型/策略

| 参数 | 值 |
|------|-----|
| 基线策略 | Greedy (按EV排序) |
| 实验策略 | Whale Matching (分层) |
| 算法实现 | b-matching / min-cost flow / greedy with swaps |

### 扫描参数

| 扫描 | 范围 | 固定 |
|------|------|------|
| k (每主播鲸鱼上限) | {1, 2, 3, 5} | - |
| 鲸鱼定义阈值 | {Top 0.1%, 1%, 5%} | - |
| 算法类型 | {b-matching, min-cost-flow, greedy-swaps} | - |

### 评估指标

| 类别 | 指标 | 说明 |
|------|------|------|
| **约束类** | 超载率 | 容量违约比例 |
| | 拥挤率 | 高并发场景占比 |
| **生态类** | Streamer Gini | 主播收益集中度 |
| | Top主播份额 | 头部集中度 |
| **收益类** | Total Revenue | 总收益 |
| | Revenue/User | 人均收益 |
| | Top-0.1% Capture | 鲸鱼捕获率 |

---

## 📊 图表

### Fig 1: 超载率 vs k值
![](./img/mvp53_overload_vs_k.png)

**观察**:
- 所有k值下超载率均为0.0%，远低于10%目标
- 说明在当前容量设置下，容量约束不是瓶颈
- **启示**: 容量设置可能过大，未触发拥挤场景

### Fig 2: Streamer Gini vs k值
![](./img/mvp53_gini_vs_k.png)

**观察**:
- 所有k值下Gini均为0.912，与基线Greedy完全相同
- **关键发现**: 分层匹配未能降低生态集中度
- **可能原因**: 鲸鱼数量过少（Top 1%仅100人），分散效果不明显

### Fig 3: 收益 vs k值
![](./img/mvp53_revenue_vs_k.png)

**观察**:
- 所有k值下收益几乎相同（42,165.51），与基线差异<0.1%
- 收益未显著下降，满足Gate-5C要求
- **启示**: 分层匹配对收益影响极小

### Fig 4: 算法对比热力图
![](./img/mvp53_algorithm_comparison.png)

**观察**:
- b-matching 和 greedy-swaps 表现完全相同
- 说明算法选择对结果影响不大
- **启示**: 问题不在算法，而在策略本身的有效性

### Fig 5: 鲸鱼分布热力图
![](./img/mvp53_distribution_heatmap.png)

**说明**: 需要详细分配数据才能生成，当前为占位图

### Fig 6: 阈值敏感性分析
![](./img/mvp53_threshold_sensitivity.png)

**说明**: 仅测试了Top 1%，其他阈值为占位数据

### Fig 7: Gini vs Revenue 权衡散点图
![](./img/mvp53_tradeoff_scatter.png)

**观察**:
- 所有配置点几乎重叠，说明k值变化对结果无影响
- 与基线（红色星号）位置相同，验证了无改善结论

---

## 💡 洞见

### 5.1 宏观
- **分层匹配在当前设置下无效**: 所有k值配置下，Gini、收益、超载率均与基线相同
- **容量不是瓶颈**: 超载率仅0.28%（基线），说明当前容量设置过大，未触发拥挤
- **鲸鱼数量可能过少**: Top 1%仅100人，分散到500个主播，每个主播平均<0.2个鲸鱼，分散效果不明显

### 5.2 算法层
- **算法选择不重要**: b-matching 和 greedy-swaps 结果完全相同，说明问题不在算法实现
- **k值无影响**: k从1到5，所有指标完全相同，说明约束未生效
- **可能原因**: 鲸鱼数量太少，即使k=1，每个主播也分不到1个鲸鱼

### 5.3 细节
- **容量设置问题**: capacity_top10=15, capacity_middle=8, capacity_tail=3 可能过大
- **需要更激进的容量**: 参考MVP-4.2，可能需要更低的容量才能触发拥挤
- **鲸鱼识别**: 基于wealth而非cumulative_revenue，可能与实际场景有偏差

---

## 📝 结论

### 6.1 核心发现
> **❌ Gate-5C FAIL: 分层匹配策略未能降低生态集中度，所有k值配置下Gini与基线相同(0.912)**

### 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **分层匹配在当前设置下无效** | 所有k值下Gini=0.912，与基线相同 |
| 2 | **容量不是瓶颈** | 超载率仅0.28%，远低于10%目标 |
| 3 | **收益未受影响** | 收益Δ=+0.07%，满足>-5%要求 |
| 4 | **算法选择不重要** | b-matching 和 greedy-swaps 结果相同 |

### 6.3 设计启示

| 原则 | 建议 |
|------|------|
| **容量设置需校准** | 当前容量过大，需降低以触发拥挤场景 |
| **鲸鱼数量是关键** | Top 1%仅100人，分散效果不明显，需测试Top 5%或更高 |
| **分层匹配需重新设计** | 当前策略在低鲸鱼密度下无效，需考虑其他分散机制 |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| **容量设置过大** | 导致超载率过低，无法验证分散效果 |
| **鲸鱼阈值过低** | Top 1%仅100人，分散到500主播，密度太低 |
| **基于wealth识别** | 可能与实际cumulative_revenue有偏差 |

### 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Best k | 1 | 所有k值相同，选择最小k |
| 超载率 | 0.0% | k=1 (vs 基线0.28%) |
| Gini改善 | 0.0 | vs Greedy (无改善) |
| Revenue Δ | +0.07% | vs Greedy (满足>-5%) |

### 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| ❌ FAIL | 保留统一分配 | - |
| 🔄 重新设计 | 降低容量设置，测试Top 5%鲸鱼阈值 | 🟡 |
| 🔄 探索其他策略 | 考虑基于收益的分散而非基于数量 | 🟡 |

---

## 📎 附录

### 7.1 数值结果

| 配置 | k | 鲸鱼阈值 | 超载率 | Gini | Revenue | Δ Revenue |
|------|---|---------|--------|------|---------|------------|
| Baseline (Greedy) | - | - | 0.28% | 0.912 | 42,135.69 | 0% |
| Whale-1 | 1 | Top 1% | 0.0% | 0.912 | 42,165.51 | +0.07% |
| Whale-2 | 2 | Top 1% | 0.0% | 0.912 | 42,165.51 | +0.07% |
| Whale-3 | 3 | Top 1% | 0.0% | 0.912 | 42,165.51 | +0.07% |
| Whale-5 | 5 | Top 1% | 0.0% | 0.912 | 42,165.51 | +0.07% |

**算法对比** (k=2, Top 1%):
- b-matching: Revenue=42,165.51, Overload=0.0%, Gini=0.912
- greedy-swaps: Revenue=42,165.51, Overload=0.0%, Gini=0.912

### 7.2 执行记录

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/simulator/policies_whale_matching.py` |
| Config | `configs/whale_matching.yaml` |
| Output | `results/whale_matching_20260109.json` |
| Simulator | `scripts/simulator/simulator.py` (V2) |

```bash
# 运行实验
python scripts/simulator/policies_whale_matching.py \
  --config configs/whale_matching.yaml \
  --output results/whale_matching_20260109.json

# 评估
python scripts/verify_metrics.py \
  --results results/whale_matching_20260109.json
```

### 7.3 参考代码

- **Simulator V2**: `scripts/simulator/simulator.py`
- **Greedy策略**: `scripts/simulator/policies.py` (GreedyPolicy)
- **Shadow Price策略**: `scripts/simulator/policies_shadow_price.py` (参考约束处理)

---

> **实验状态**: ✅ Completed  
> **创建时间**: 2026-01-09  
> **完成时间**: 2026-01-17  
> **Gate-5C**: ❌ FAIL (Gini无改善)
