# 🤖 Coding Prompt: 鲸鱼分散 (b-matching)

> **Experiment ID:** `EXP-20260109-gift-allocation-53`  
> **MVP:** MVP-5.3  
> **Date:** 2026-01-09  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证分层匹配策略（鲸鱼层+普通层）能否降低生态集中度，同时保持收益不显著下降

**验证Gate**：Gate-5C - 鲸鱼分散能否降低生态集中度

**预期结果**：
- 若 **超载率 <10% AND Streamer Gini↓ AND 收益下降 <5%** → Gate-5C PASS → 采用鲸鱼单独匹配层
- 若 超载率仍高 或 收益下降>5% → Gate-5C FAIL → 保留统一分配

**背景**：
- **问题**：鲸鱼用户（Top-0.1%/1%）同时涌入同一主播导致容量超载、马太效应加剧
- **假设**：通过限制每个主播同时承接的鲸鱼数量（k），强制分散到更多主播，降低生态集中度
- **策略**：分层匹配 - 先做鲸鱼层（互斥+分散+价值最大化），再填充普通层（轻量greedy）

---

## 2. 🧪 实验设定

### 2.1 依赖

```yaml
dependency:
  - MVP-4.1+: Simulator V2+ (金额校准) ✅ 已完成
  - MVP-4.2: 并发容量建模 ✅ 已完成
  - MVP-5.2: Shadow Price约束处理 ✅ 已完成（参考WhaleSpreadConstraint设计）
```

### 2.2 核心算法

**分层匹配流程**：

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

**算法选项**：

| 算法 | 描述 | 复杂度 | 推荐度 |
|------|------|--------|--------|
| **b-matching** | 二分图匹配，右侧容量为k | O(n²m) | ⭐⭐⭐ 推荐 |
| **min-cost flow** | 最小费用流 | O(nm log(n+m)) | ⭐⭐ 复杂约束 |
| **greedy with swaps** | 贪心+后处理交换 | O(nm) | ⭐ 快速验证 |

### 2.3 策略配置

```yaml
policies:
  # 基线策略
  - name: "greedy"
    description: "纯贪心分配（基线）"
    formula: "argmax_s EV(u,s)"
    
  # 实验策略
  - name: "whale_matching_bmatching"
    description: "分层匹配 - b-matching算法"
    algorithm: "b-matching"
    whale_threshold: "Top 0.1%" | "Top 1%" | "Top 5%"
    k: 1 | 2 | 3 | 5  # 每主播鲸鱼上限
    
  - name: "whale_matching_mincost"
    description: "分层匹配 - min-cost flow算法"
    algorithm: "min-cost-flow"
    whale_threshold: "Top 0.1%" | "Top 1%" | "Top 5%"
    k: 1 | 2 | 3 | 5
    
  - name: "whale_matching_greedy"
    description: "分层匹配 - greedy with swaps"
    algorithm: "greedy-swaps"
    whale_threshold: "Top 0.1%" | "Top 1%" | "Top 5%"
    k: 1 | 2 | 3 | 5
```

### 2.4 实验配置

```yaml
experiments:
  # 实验1: 算法对比（固定k=2, Top 1%）
  - name: "algorithm_comparison"
    whale_threshold: "Top 1%"
    k: 2
    algorithms: ["b-matching", "min-cost-flow", "greedy-swaps"]
    n_simulations: 50
    baseline: "greedy"
    
  # 实验2: k值扫描（固定算法=b-matching, Top 1%）
  - name: "k_sweep"
    algorithm: "b-matching"
    whale_threshold: "Top 1%"
    k_values: [1, 2, 3, 5]
    n_simulations: 50
    
  # 实验3: 鲸鱼阈值扫描（固定算法=b-matching, k=2）
  - name: "whale_threshold_sweep"
    algorithm: "b-matching"
    k: 2
    whale_thresholds: ["Top 0.1%", "Top 1%", "Top 5%"]
    n_simulations: 50
    
  # 实验4: 完整网格扫描
  - name: "full_grid"
    algorithms: ["b-matching", "greedy-swaps"]
    k_values: [1, 2, 3, 5]
    whale_thresholds: ["Top 0.1%", "Top 1%"]
    n_simulations: 30  # 减少以节省时间
```

### 2.5 Simulator配置

```yaml
simulation:
  n_users: 10000
  n_streamers: 500
  n_rounds: 50
  users_per_round: 200
  n_simulations: 50
  seed: 42
  
  # 使用V2+金额模型
  amount_version: 3  # V2+ discrete tiers
  
  # 使用并发容量建模（MVP-4.2）
  capacity_per_streamer: 15  # 参考MVP-4.2校准值
  crowding_penalty_beta: 0.5
```

### 2.6 评估指标

```yaml
metrics:
  # 约束类（Gate-5C核心指标）
  overload_rate:
    formula: "n_overload / n_total_allocations"
    gate_threshold: "< 0.10"  # <10%
    description: "容量违约比例"
    
  crowding_rate:
    formula: "n_crowded / n_rounds"
    description: "高并发场景占比"
    
  # 生态类（Gate-5C核心指标）
  streamer_gini:
    formula: "gini(streamer_revenues)"
    gate_threshold: "↓ vs greedy"  # 必须降低
    description: "主播收益集中度"
    
  top_10_share:
    formula: "sum(top_10% streamers) / total_revenue"
    description: "头部集中度"
    
  # 收益类（Gate-5C核心指标）
  total_revenue:
    formula: "sum(all gifts)"
    gate_threshold: ">= greedy * 0.95"  # 下降<5%
    
  revenue_per_user:
    formula: "total_revenue / n_users"
    
  top_01pct_capture:
    formula: "revenue_from_top_01pct / total_revenue"
    description: "鲸鱼捕获率"
    
  # 分散度指标
  whale_distribution_entropy:
    formula: "entropy(whale_counts_per_streamer)"
    description: "鲸鱼分布熵（越高越分散）"
    
  n_streamers_with_whales:
    formula: "count(streamers with whale > 0)"
    description: "承接鲸鱼的主播数量"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | line (multi) | k值 | Overload Rate (per algorithm) | `../img/mvp53_overload_vs_k.png` |
| Fig2 | line (multi) | k值 | Streamer Gini (per algorithm) | `../img/mvp53_gini_vs_k.png` |
| Fig3 | line (multi) | k值 | Revenue Δ% vs Greedy (per algorithm) | `../img/mvp53_revenue_vs_k.png` |
| Fig4 | heatmap | Algorithm | k值 | Overload Rate (color) | `../img/mvp53_algorithm_comparison.png` |
| Fig5 | heatmap | Streamer ID | User ID (whale only) | Allocation Matrix | `../img/mvp53_distribution_heatmap.png` |
| Fig6 | bar (grouped) | Whale Threshold | Metrics (Overload/Gini/Revenue) | `../img/mvp53_threshold_sensitivity.png` |
| Fig7 | scatter | Streamer Gini | Revenue (per k) | `../img/mvp53_tradeoff_scatter.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize**: 单张 `(6, 5)`，多张按 6:5 比例扩增

**特殊图表说明**：
- **Fig5 热力图**：对比 Greedy vs Whale Matching 的鲸鱼分布模式
  - Greedy: 鲸鱼集中在少数主播（列）
  - Whale Matching: 鲸鱼分散到更多主播（列）
- **Fig7 散点图**：展示 Gini vs Revenue 权衡，每个点代表一个k值配置

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/simulator/simulator.py` | `GiftLiveSimulator`, `AllocationPolicy`接口 | - |
| `scripts/simulator/policies.py` | `GreedyPolicy`, `ConstrainedAllocationPolicy` | 扩展为分层匹配 |
| `scripts/simulator/policies_shadow_price.py` | `WhaleSpreadConstraint`（识别鲸鱼逻辑） | 提取鲸鱼识别函数 |
| `scripts/simulator/calibration.py` | 校准工具 | - |
| `../results/concurrency_capacity_20260109.json` | 容量参数参考 | - |
| `../results/shadow_price_20260109.json` | 基线收益参考 | - |

**新增代码模块**：

1. **`WhaleMatchingPolicy` 类**:
   - 继承 `AllocationPolicy`
   - 实现分层匹配逻辑
   - 方法：
     - `identify_whales(users, threshold)` → Set[user_id]
     - `whale_layer_matching(whales, k, algorithm)` → Dict[streamer_id, List[user_id]]
     - `normal_layer_matching(normal_users, remaining_capacity)` → List[allocation]
     - `allocate(users, simulator)` → List[streamer_id]

2. **b-matching 实现**:
   - 文件：`scripts/simulator/matching_algorithms.py` (新建)
   - 函数：`b_matching(ev_matrix, k_per_streamer)` → allocations
   - 可选库：`networkx` 或 `scipy.optimize`

3. **min-cost flow 实现**:
   - 文件：`scripts/simulator/matching_algorithms.py`
   - 函数：`min_cost_flow_matching(ev_matrix, k_per_streamer)` → allocations
   - 可选库：`ortools` 或 `networkx`

4. **greedy with swaps 实现**:
   - 文件：`scripts/simulator/matching_algorithms.py`
   - 函数：`greedy_with_swaps(ev_matrix, k_per_streamer)` → allocations
   - 简单实现：贪心分配 + 后处理交换优化

5. **评估函数**:
   - `compute_overload_rate(allocations, capacity)` → float
   - `compute_whale_distribution(allocations, whale_set)` → Dict[streamer_id, count]
   - `compute_whale_entropy(whale_distribution)` → float

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `exp/exp_whale_matching_20260109.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（超载率、Gini改善、收益Δ%）
  - 📊 实验图表（所有7张图 + 观察）
  - 📝 结论（Gate-5C 判定 + 设计建议）

### 5.2 代码文件
- **路径**: `scripts/simulator/policies_whale_matching.py`
- **内容**: `WhaleMatchingPolicy` 类
- **路径**: `scripts/simulator/matching_algorithms.py` (新建)
- **内容**: b-matching, min-cost-flow, greedy-swaps 实现

### 5.3 数值结果
- **路径**: `../results/whale_matching_20260109.json`
- **内容**: 
  ```json
  {
    "experiment_id": "EXP-20260109-gift-allocation-53",
    "baseline": {
      "greedy": {
        "revenue": ...,
        "overload_rate": ...,
        "streamer_gini": ...,
        ...
      }
    },
    "whale_matching": {
      "b-matching": {
        "k=1": {"revenue": ..., "overload_rate": ..., "gini": ..., ...},
        "k=2": {...},
        "k=3": {...},
        "k=5": {...}
      },
      "greedy-swaps": {...},
      "min-cost-flow": {...}
    },
    "best_config": {
      "algorithm": "b-matching",
      "k": 2,
      "whale_threshold": "Top 1%",
      "overload_rate": ...,
      "gini_improvement": ...,
      "revenue_delta_pct": ...
    },
    "gate5c": "PASS" | "FAIL"
  }
  ```

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVmodel_roadmap.md` | MVP-5.3 状态 + 结论快照 | §2.1, §6.3 |
| `gift_EVmodel_hub.md` | Gate-5C 结果 + DG8 关闭 + 洞见 | §1.2, §0 核心结论, §4 洞见汇合 |

---

## 7. ⚠️ 注意事项

- [ ] 运行前先 `source init.sh`
- [ ] seed=42 固定随机性
- [ ] 鲸鱼识别基于用户历史累计收益（需在simulator中维护）
- [ ] b-matching 实现需处理"每个主播最多k个鲸鱼"约束
- [ ] 普通层分配时需排除已分配的鲸鱼用户
- [ ] 多次模拟取均值 ± 标准差
- [ ] 长时间任务使用 nohup 后台运行
- [ ] 图表文字全英文

---

## 8. 📐 核心算法细节

### 8.1 鲸鱼识别

```python
def identify_whales(users: List[User], threshold: str) -> Set[int]:
    """
    识别鲸鱼用户
    
    threshold: "Top 0.1%" | "Top 1%" | "Top 5%"
    基于用户历史累计收益排序
    """
    # 按累计收益排序
    # 取Top X%作为鲸鱼
    pass
```

### 8.2 b-matching 算法

**问题定义**：
- 左侧节点：鲸鱼用户（每个最多匹配1个主播）
- 右侧节点：主播（每个最多匹配k个鲸鱼）
- 边权重：EV(u, s)
- 目标：最大化总权重

**实现思路**：
1. 构建二分图
2. 应用匈牙利算法变种（带容量约束）
3. 或转化为最小费用流问题

### 8.3 分层匹配伪代码

```python
def allocate(users, simulator):
    # 1. 识别鲸鱼
    whales = identify_whales(users, threshold="Top 1%")
    normal_users = [u for u in users if u.id not in whales]
    
    # 2. 鲸鱼层匹配
    ev_matrix_whale = simulator.get_expected_values(whales)
    whale_allocations = b_matching(ev_matrix_whale, k=2)
    # whale_allocations: Dict[streamer_id, List[whale_user_id]]
    
    # 3. 计算剩余容量
    remaining_capacity = {}
    for s in simulator.streamer_pool.streamers:
        whale_count = len(whale_allocations.get(s.id, []))
        remaining_capacity[s.id] = s.capacity - whale_count
    
    # 4. 普通层匹配（Greedy）
    normal_allocations = greedy_allocate(
        normal_users, 
        remaining_capacity,
        simulator
    )
    
    # 5. 合并
    return merge_allocations(whale_allocations, normal_allocations)
```

---

## 9. 🔗 接口设计

```
WhaleMatchingPolicy:
  
  输入:
    - users: List[User]
    - simulator: GiftLiveSimulator
  
  输出:
    - allocations: List[int]  # streamer_id for each user
  
  参数:
    - algorithm: "b-matching" | "min-cost-flow" | "greedy-swaps"
    - k: int  # 每主播鲸鱼上限
    - whale_threshold: "Top 0.1%" | "Top 1%" | "Top 5%"
  
  方法:
    - identify_whales(users, threshold) -> Set[user_id]
    - whale_layer_matching(whales, k, algorithm) -> Dict[streamer_id, List[user_id]]
    - normal_layer_matching(normal_users, remaining_capacity) -> List[allocation]
```

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件（尤其是 policies.py, policies_shadow_price.py）
3. ✅ 理解现有 AllocationPolicy 接口和 Simulator 工作流程
4. ✅ 复用现有代码结构，新增 WhaleMatchingPolicy 类
5. ✅ b-matching 算法可参考网络流算法库（networkx, ortools）
6. ✅ 鲸鱼识别逻辑可参考 policies_shadow_price.py 中的 WhaleSpreadConstraint
7. ✅ 按模板输出 exp.md 报告
8. ✅ 运行前先 source init.sh
9. ✅ 图表文字全英文，figsize=(6,5) 或按比例扩增
-->
