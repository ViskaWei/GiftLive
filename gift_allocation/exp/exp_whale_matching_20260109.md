# 📗 实验报告：鲸鱼分散 (b-matching)

> **Experiment ID:** `EXP-20260109-gift-allocation-53`  
> **MVP:** MVP-5.3  
> **Author:** Viska Wei  
> **Date:** 2026-01-09  
> **Status:** 🔄 Planning  
> **Gate:** Gate-5C

---

## 🔗 上游追溯

- **Hub**: `gift_allocation/gift_allocation_hub.md` § DG8
- **Roadmap**: `gift_allocation/gift_allocation_roadmap.md` § MVP-5.3
- **相关实验**: MVP-5.2 (影子价格), MVP-4.2 (并发容量)

---

## ⚡ 核心结论速览

### 一句话总结

> **⏳ 待执行**: 验证分层匹配（鲸鱼层+普通层）能否降低生态集中度，同时保持收益不显著下降。

### 假设验证

| 假设 | 待验证 | 判断标准 |
|------|--------|----------|
| H5C-1: 鲸鱼分散能降低超载率 | ⏳ | 超载率 <10% |
| H5C-2: 分层匹配能降低Streamer Gini | ⏳ | Gini↓ vs Greedy |
| H5C-3: 收益不显著下降 | ⏳ | 收益下降 <5% |

### 关键数字速查

| 指标 | 目标值 | 基线 (Greedy) |
|------|--------|---------------|
| 超载率 | <10% | TBD |
| Streamer Gini | ↓ | TBD |
| Revenue | ≥-5% | TBD |

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

## 📊 图表（待生成）

### Fig 1: 超载率 vs k值
![](./img/whale_matching_overload_vs_k.png)

**预期观察**:
- k越小，超载率越低
- 找到k值使得超载率<10%

### Fig 2: Streamer Gini vs k值
![](./img/whale_matching_gini_vs_k.png)

**预期观察**:
- k越小，Gini越低（分散效果越好）
- 与基线Greedy对比

### Fig 3: 收益 vs k值
![](./img/whale_matching_revenue_vs_k.png)

**预期观察**:
- k越小，收益可能下降（分散导致次优匹配）
- 验证收益下降<5%

### Fig 4: 鲸鱼分布热力图
![](./img/whale_matching_distribution_heatmap.png)

**预期观察**:
- Greedy: 鲸鱼集中在少数主播
- Whale Matching: 鲸鱼分散到更多主播

---

## 💡 洞见（待补充）

### 5.1 宏观
- [待实验后补充]

### 5.2 算法层
- [待实验后补充]

### 5.3 细节
- [待实验后补充]

---

## 📝 结论（待补充）

### 6.1 核心发现
> **[待实验后补充]**

### 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | [待补充] | [数据] |

### 6.3 设计启示

| 原则 | 建议 |
|------|------|
| [待补充] | [做法] |

### 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Best k | TBD | TBD |
| 超载率 | TBD | k=TBD |
| Gini改善 | TBD | vs Greedy |

### 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 若PASS | 集成到主分配流程 | 🔴 |
| 若FAIL | 探索其他分散策略 | 🟡 |

---

## 📎 附录

### 7.1 数值结果（待补充）

| 配置 | k | 鲸鱼阈值 | 超载率 | Gini | Revenue | Δ Revenue |
|------|---|---------|--------|------|---------|------------|
| Baseline | - | - | TBD | TBD | TBD | 0% |
| Whale-1 | 1 | Top 0.1% | TBD | TBD | TBD | TBD |
| Whale-2 | 2 | Top 0.1% | TBD | TBD | TBD | TBD |
| Whale-3 | 3 | Top 0.1% | TBD | TBD | TBD | TBD |
| Whale-5 | 5 | Top 0.1% | TBD | TBD | TBD | TBD |

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

> **实验状态**: 🔄 Planning  
> **创建时间**: 2026-01-09  
> **预计完成**: TBD
