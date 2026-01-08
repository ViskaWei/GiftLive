# 🤖 Coding Prompt: 冷启动/公平约束

> **Experiment ID:** `EXP-20260108-gift-allocation-09`  
> **MVP:** MVP-2.2  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证冷启动约束和公平约束对生态健康的影响，在可控收益损失下提升新主播成功率

**验证假设**：Hub Q2.3 - 冷启动/公平约束如何嵌入分配层？

**预期结果**：
- 若 收益损失 < 5% 且新主播成功率提升 > 20% → 约束值得引入
- 若 收益损失 ≥ 5% 或成功率无显著提升 → 简化约束或放弃

**背景**：
- 92.2% 主播无打赏（冷启动严重）
- 头部 1% 主播获得 53.4% 收益（马太效应）
- 需平衡短期收益与长期生态健康

---

## 2. 🧪 实验设定

### 2.1 依赖

```yaml
dependency:
  - MVP-0.3: Simulator V1 必须完成
  - MVP-2.1: 凹收益分配（作为基线策略）
```

### 2.2 约束设计

```yaml
constraints:
  # 冷启动约束
  cold_start:
    type: "minimum_exposure"
    description: "新主播最低曝光/分配量"
    params:
      min_allocation_per_new: 10   # 每个新主播至少分配10个用户
      new_streamer_window: 7       # 7天内算新主播
      enforce: "soft"              # soft=拉格朗日，hard=硬约束
  
  # 头部上限约束
  head_cap:
    type: "maximum_share"
    description: "头部主播收益上限"
    params:
      max_share_top10: 0.5         # Top 10% 主播最多获得 50% 收益
      enforce: "soft"
  
  # 多样性约束
  diversity:
    type: "minimum_coverage"
    description: "最低主播覆盖率"
    params:
      min_coverage: 0.3            # 至少 30% 主播有收益
      enforce: "soft"
```

### 2.3 优化方法

```yaml
optimization:
  # 拉格朗日方法
  lagrangian:
    objective: "sum(g(V_s))"
    constraints:
      - "allocation(s) >= min_per_new for new streamers"
      - "sum(top_10% revenue) <= 0.5 * total"
    lambda_init: [0.1, 0.1]
    lambda_lr: 0.01
    dual_update: "subgradient"
  
  # 对比：硬约束
  hard_constraint:
    method: "reserve_and_allocate"
    reserve_for_cold: 0.1          # 预留10%流量给冷启动
```

### 2.4 实验配置

```yaml
experiments:
  # 实验1: 约束效果对比
  - name: "constraint_comparison"
    policies:
      - name: "no_constraint"      # 纯凹收益分配
      - name: "soft_cold_start"    # 软约束：冷启动
      - name: "soft_head_cap"      # 软约束：头部上限
      - name: "soft_all"           # 软约束：全部
      - name: "hard_cold_start"    # 硬约束：预留流量
    n_simulations: 100
    n_rounds: 50
    metrics: ["revenue", "gini", "cold_start_success", "coverage"]
  
  # 实验2: 约束强度敏感度
  - name: "constraint_strength_sweep"
    policy: "soft_cold_start"
    sweep_param: "min_allocation_per_new"
    values: [5, 10, 20, 50, 100]
    n_simulations: 50
  
  # 实验3: 长期模拟（用户留存代理）
  - name: "long_term_simulation"
    n_rounds: 200
    metrics: ["new_streamer_survival_rate", "ecosystem_diversity"]
```

### 2.5 评估指标

```yaml
metrics:
  # 效率
  revenue:
    formula: "sum(all gifts)"
  revenue_loss_pct:
    formula: "(baseline_revenue - constrained_revenue) / baseline_revenue * 100"
  
  # 冷启动
  cold_start_success_rate:
    formula: "n_new_streamers_with_gift / n_new_streamers"
    threshold: 0.1  # 成功=获得至少1次打赏
  cold_start_revenue:
    formula: "sum(new_streamer_revenue)"
  
  # 公平性
  coverage:
    formula: "n_streamers_with_revenue / n_streamers"
  head_share:
    formula: "top_10%_revenue / total_revenue"
  
  # 长期
  ecosystem_diversity:
    formula: "entropy(streamer_revenue_distribution)"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar (grouped) | Policy | Revenue / Cold Start Success | `gift_allocation/img/mvp22_constraint_effect.png` |
| Fig2 | scatter | Revenue Loss % | Cold Start Success Δ | `gift_allocation/img/mvp22_tradeoff_scatter.png` |
| Fig3 | line | Min Allocation per New | Revenue Loss / Success Rate | `gift_allocation/img/mvp22_constraint_strength.png` |
| Fig4 | line (multi) | Round | Cumulative Coverage | `gift_allocation/img/mvp22_coverage_over_time.png` |
| Fig5 | bar | Policy | Ecosystem Diversity (Entropy) | `gift_allocation/img/mvp22_ecosystem_diversity.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize**: 单张 `(6, 5)`

---

## 4. 📁 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/simulator/simulator.py` | `GiftLiveSimulator` | 添加新主播标记 |
| `scripts/simulator/policies.py` | `ConcaveAllocationPolicy` | 添加约束版本 |

**新增代码模块**：
1. `ConstrainedAllocationPolicy`: 带约束的分配策略
2. `LagrangianOptimizer`: 拉格朗日对偶更新
3. `ColdStartTracker`: 冷启动主播追踪

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_allocation/exp/exp_coldstart_constraint_20260108.md`
- **模板**: `_backend/template/exp.md`

### 5.2 数值结果
- **路径**: `gift_allocation/results/coldstart_constraint_20260108.json`
- **内容**: 
  ```json
  {
    "constraint_comparison": {
      "no_constraint": {"revenue": ..., "cold_start_success": ...},
      "soft_cold_start": {"revenue": ..., "cold_start_success": ...},
      "revenue_loss_pct": ...,
      "success_improvement_pct": ...
    },
    "decision": "adopt_constraint" | "reject_constraint"
  }
  ```

---

## 6. 📤 报告抄送

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_allocation_roadmap.md` | MVP-2.2 状态 + Gate-2 综合结果 | §1.2, §2.1, §4.2 |
| `gift_allocation_hub.md` | Q2.3 答案 + P6 原则验证 | §1, §6.2 |

---

## 7. ⚠️ 注意事项

- [ ] 依赖 MVP-0.3, MVP-2.1 完成后才能执行
- [ ] seed=42 固定随机性
- [ ] 拉格朗日对偶变量需正确更新（次梯度法）
- [ ] 注意约束违反惩罚的数值稳定性

---

## 8. 📐 核心公式

### 8.1 拉格朗日目标

$$\mathcal{L} = \sum_s g(V_s) + \sum_c \lambda_c \cdot \text{slack}_c$$

其中 $\text{slack}_c$ 是约束 $c$ 的松弛变量

### 8.2 对偶更新

$$\lambda_c^{(t+1)} = \max\left(0, \lambda_c^{(t)} + \eta \cdot \text{violation}_c^{(t)}\right)$$

### 8.3 冷启动约束

$$\sum_u \mathbb{1}[u \to s] \geq \text{min\_alloc}, \quad \forall s \in \text{NewStreamers}$$

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 确认 MVP-0.3, MVP-2.1 已完成
3. ✅ 复用 MVP-2.1 的分配策略作为基线
4. ✅ 按模板输出 exp.md 报告
-->
