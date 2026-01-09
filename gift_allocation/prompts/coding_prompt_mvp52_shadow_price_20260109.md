# 🤖 Coding Prompt: 影子价格/供需匹配

> **Experiment ID:** `EXP-20260109-gift-allocation-16`  
> **MVP:** MVP-5.2  
> **Date:** 2026-01-09  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证Primal-Dual影子价格框架能否统一处理多约束分配并提升收益

**验证Gate**：Gate-5B - 影子价格能否统一处理多约束且提升收益

**预期结果**：
- 若 收益 ≥ Greedy+5% 且 约束满足率 >90% → Gate-5B PASS → 替换Greedy+规则
- 若 收益无提升 或 约束违约率高 → Gate-5B FAIL → 保留Greedy+软约束

**背景**：
- 当前分配难点：稀缺大哥资源 + 主播承接上限 + 生态约束
- 现有方案：Greedy + 多个独立规则（频控、冷启动bonus）
- 影子价格优势：统一框架处理所有约束，通过对偶变量自动学习约束价值

---

## 2. 🧪 实验设定

### 2.1 依赖

```yaml
dependency:
  - MVP-4.1+: Simulator V2+ (金额校准) ✅ 已完成
  - MVP-4.2: 并发容量建模 ✅ 已完成
  - MVP-2.2: 冷启动约束 ✅ 已完成（参考ConstraintConfig设计）
```

### 2.2 核心公式

**分配决策**：

$$s^* = \arg\max_s \left[ \widehat{EV}(u,s) - \sum_c \lambda_c \cdot \Delta\text{violation}_c(u \to s) \right]$$

**对偶变量更新（Primal-Dual）**：

$$\lambda_c \gets \lambda_c + \eta \cdot (\text{usage}_c - \text{budget}_c)$$

- 超载时 $\lambda_c$ 涨价（减少该约束资源消耗）
- 未满时 $\lambda_c$ 降价（鼓励使用）

### 2.3 约束类型

```yaml
constraints:
  # C1: 并发容量约束
  capacity:
    description: "主播同时服务用户上限"
    threshold: "capacity_s"  # 每主播容量
    violation: "当前并发 / capacity_s"
    lambda_init: 0.1
    
  # C2: 冷启动覆盖约束
  cold_start:
    description: "新主播最低曝光保障"
    threshold: "min_alloc_per_new = 10"
    violation: "max(0, min_alloc - current_alloc) / min_alloc"
    lambda_init: 0.5  # 来自MVP-2.2最优值
    
  # C3: 头部占比上限
  head_cap:
    description: "Top-10%主播收益占比上限"
    threshold: "max_share = 0.5"
    violation: "max(0, top10_share - max_share)"
    lambda_init: 0.1
    
  # C4: 鲸鱼分散约束
  whale_spread:
    description: "每主播同时承接鲸鱼数上限"
    threshold: "max_whale_per_streamer = 2"
    violation: "当前鲸鱼数 / max_whale"
    lambda_init: 0.2
    
  # C5: 频控约束
  frequency:
    description: "用户对同一主播重复曝光限制"
    threshold: "max_freq_per_pair = 3"
    violation: "1 if freq > max else 0"
    lambda_init: 0.3
```

### 2.4 策略配置

```yaml
policies:
  # 基线策略
  - name: "greedy"
    description: "纯贪心分配"
    formula: "argmax_s EV(u,s)"
    
  - name: "greedy_with_rules"
    description: "贪心 + 现有规则（频控+冷启动bonus）"
    formula: "greedy + cold_start_bonus + freq_penalty"
    lambda_cold_start: 0.5
    
  # 影子价格策略
  - name: "shadow_price_all"
    description: "影子价格 - 所有约束"
    constraints: ["capacity", "cold_start", "head_cap", "whale_spread", "frequency"]
    
  - name: "shadow_price_core"
    description: "影子价格 - 核心约束（容量+冷启动+头部）"
    constraints: ["capacity", "cold_start", "head_cap"]
    
  - name: "shadow_price_light"
    description: "影子价格 - 轻量版（容量+冷启动）"
    constraints: ["capacity", "cold_start"]
```

### 2.5 实验配置

```yaml
experiments:
  # 实验1: 策略对比
  - name: "policy_comparison"
    policies: ["greedy", "greedy_with_rules", "shadow_price_core", "shadow_price_all"]
    n_simulations: 100
    n_users: 10000
    n_streamers: 500
    n_rounds: 50
    users_per_round: 200
    seed: 42
    
  # 实验2: 学习率敏感度
  - name: "lr_sweep"
    policy: "shadow_price_core"
    sweep_param: "eta"
    values: [0.001, 0.01, 0.05, 0.1, 0.2]
    n_simulations: 50
    
  # 实验3: 约束组合消融
  - name: "constraint_ablation"
    base_constraints: ["capacity"]
    add_constraints:
      - ["cold_start"]
      - ["cold_start", "head_cap"]
      - ["cold_start", "head_cap", "whale_spread"]
      - ["cold_start", "head_cap", "whale_spread", "frequency"]
    n_simulations: 50
    
  # 实验4: Lambda收敛分析
  - name: "lambda_convergence"
    policy: "shadow_price_all"
    track_lambda_history: true
    n_rounds: 100
    n_simulations: 10
```

### 2.6 评估指标

```yaml
metrics:
  # 收益指标
  total_revenue:
    formula: "sum(all gifts)"
    gate_threshold: ">= greedy * 1.05"  # +5%
    
  revenue_per_user:
    formula: "total_revenue / n_users"
    
  # 约束满足指标
  capacity_satisfy_rate:
    formula: "n_satisfy / n_total"
    gate_threshold: ">= 0.90"
    
  cold_start_success_rate:
    formula: "n_new_with_gift / n_new"
    gate_threshold: ">= 0.30"
    
  head_share_within_cap:
    formula: "top10_share <= 0.5"
    gate_threshold: ">= 0.90"
    
  whale_spread_rate:
    formula: "n_streamers_under_whale_cap / n_streamers"
    gate_threshold: ">= 0.90"
    
  # 公平性指标
  streamer_gini:
    formula: "gini(streamer_revenues)"
    
  top_10_share:
    formula: "sum(top_10% streamers) / total_revenue"
    
  # 效率指标
  lambda_stability:
    formula: "std(lambda) / mean(lambda) 在后50轮"
    description: "对偶变量收敛稳定性"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar (grouped) | Policy | Revenue / Revenue_vs_Greedy% | `gift_allocation/img/mvp52_revenue_comparison.png` |
| Fig2 | heatmap | Constraint | Satisfaction Rate (per policy) | `gift_allocation/img/mvp52_constraint_satisfaction.png` |
| Fig3 | line (multi) | Round | Lambda Value (per constraint) | `gift_allocation/img/mvp52_lambda_convergence.png` |
| Fig4 | line | Learning Rate η | Revenue / Constraint Satisfy Rate | `gift_allocation/img/mvp52_lr_sensitivity.png` |
| Fig5 | bar (stacked) | Constraint Combo | Revenue Δ vs Greedy | `gift_allocation/img/mvp52_constraint_ablation.png` |
| Fig6 | scatter | Revenue | Constraint Satisfy (avg) | `gift_allocation/img/mvp52_pareto_frontier.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize**: 单张 `(6, 5)`，多张按 6:5 比例扩增

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/simulator/__init__.py` | 导入接口 | 添加新策略导出 |
| `scripts/simulator/policies.py` | `ConstraintConfig`, `ConstrainedAllocationPolicy` | 扩展为 Primal-Dual 版本 |
| `scripts/simulator/calibration.py` | 校准工具 | - |
| `gift_allocation/results/concurrency_capacity_20260109.json` | 容量参数参考 | - |
| `gift_allocation/results/coldstart_constraint_20260108.json` | 冷启动λ=0.5 | - |

**新增代码模块**：

1. **`ShadowPriceAllocator` 类**:
   - 输入：candidates, capacity, constraints
   - 输出：allocations, lambda_values, metrics
   - 实现 Primal-Dual 更新逻辑

2. **约束接口**:
   ```
   class Constraint:
       def compute_violation(state) -> float
       def compute_penalty(user, streamer) -> float
   ```

3. **评估函数**:
   - `evaluate_constraint_satisfaction()`
   - `track_lambda_history()`

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_allocation/exp/exp_shadow_price_20260109.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（收益Δ% + 约束满足率）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（Gate-5B 判定 + 设计建议）

### 5.2 代码文件
- **路径**: `scripts/simulator/policies_shadow_price.py`
- **内容**: `ShadowPriceAllocator`, 约束类, 评估工具

### 5.3 数值结果
- **路径**: `gift_allocation/results/shadow_price_20260109.json`
- **内容**: 
  ```json
  {
    "policy_comparison": {
      "greedy": {"revenue": ..., "gini": ...},
      "greedy_with_rules": {"revenue": ..., "constraints": {...}},
      "shadow_price_core": {"revenue": ..., "constraints": {...}},
      "shadow_price_all": {"revenue": ..., "constraints": {...}}
    },
    "delta_revenue_pct": ...,
    "avg_constraint_satisfy_rate": ...,
    "lambda_final": {"capacity": ..., "cold_start": ..., ...},
    "gate5b": "PASS" | "FAIL"
  }
  ```

---

## 6. 📤 报告抄送

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_allocation_roadmap.md` | MVP-5.2 状态 + 结论快照 | §2.1, §6.3 |
| `gift_allocation_hub.md` | Gate-5B 结果 + DG7 关闭 + 洞见 | §1.2, §0 核心结论 |

---

## 7. ⚠️ 注意事项

- [ ] 运行前先 `source init.sh`
- [ ] seed=42 固定随机性
- [ ] 对偶变量 λ 需要 clip 到 [0, λ_max] 避免爆炸
- [ ] λ 更新在每轮结束后进行（不是每个分配后）
- [ ] 多次模拟取均值 ± 标准差
- [ ] 长时间任务使用 nohup 后台运行

---

## 8. 📐 核心公式

### 8.1 Primal-Dual 分配框架

**原始问题（Primal）**：
$$\max \sum_{u,s} x_{u,s} \cdot EV(u,s) \quad \text{s.t.} \quad g_c(x) \leq b_c, \forall c$$

**对偶问题（Dual）**：
$$\min_{\lambda \geq 0} \max_x \mathcal{L}(x, \lambda) = \sum_{u,s} x_{u,s} \cdot EV(u,s) - \sum_c \lambda_c (g_c(x) - b_c)$$

**在线分配决策**：
$$s^*(u) = \arg\max_s \left[ EV(u,s) - \sum_c \lambda_c \cdot \Delta g_c(u \to s) \right]$$

### 8.2 对偶变量更新

$$\lambda_c^{(t+1)} = \left[ \lambda_c^{(t)} + \eta \cdot (g_c^{(t)} - b_c) \right]_+$$

其中 $[\cdot]_+$ 表示投影到非负区间。

### 8.3 约束违约惩罚（示例）

| 约束 | $\Delta g_c(u \to s)$ |
|------|----------------------|
| 容量 | $\mathbb{1}[\text{load}_s \geq C_s]$ |
| 冷启动 | $-\mathbb{1}[s \in \text{new\_set}]$ (bonus) |
| 头部cap | $\mathbb{1}[s \in \text{top10}] \cdot \text{current\_share}$ |
| 鲸鱼分散 | $\mathbb{1}[u \in \text{whale}] \cdot \text{whale\_count}_s / k$ |
| 频控 | $\mathbb{1}[\text{freq}_{u,s} \geq \text{max}]$ |

---

## 9. 🔗 影子价格分配器接口

```
ShadowPriceAllocator:
  
  输入:
    - candidates: List[(user_id, streamer_id, EV)]
    - capacity: Dict[streamer_id, int]
    - coldstart_set: Set[streamer_id]
    - whale_set: Set[user_id]
    - constraints: Dict[str, ConstraintConfig]
  
  输出:
    - allocations: List[(user_id, streamer_id)]
    - lambda_values: Dict[str, float]
    - metrics: Dict[str, float]
  
  方法:
    - allocate_batch(users) -> allocations
    - update_dual_variables() -> violations
    - get_constraint_status() -> Dict[str, satisfy_rate]
```

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件（尤其是 policies.py）
3. ✅ 复用现有 ConstrainedAllocationPolicy 设计，扩展为 Primal-Dual
4. ✅ 每个约束实现为独立类，方便组合和消融
5. ✅ Lambda 历史需要记录用于收敛分析
6. ✅ 按模板输出 exp.md 报告
7. ✅ 运行前先 source init.sh
-->
