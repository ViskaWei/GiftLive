# 🤖 Coding Prompt: 凹收益分配层

> **Experiment ID:** `EXP-20260108-gift-allocation-08`  
> **MVP:** MVP-2.1  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证凹收益分配（边际递减）是否优于贪心分配，关闭 DG3

**验证假设**：DG3 - 凹收益分配 vs 贪心的总收益差异有多大？

**预期结果**：
- 若 总收益 Δ ≥ 10% 且 Gini 改善 → 确认分配层设计，采用凹收益分配
- 若 Δ < 10% → 分配层收益不显著，简化为贪心+硬约束

**背景**：
- 贪心分配：argmax $v_{u,s}$，将用户分配给期望收益最高的主播
- 凹收益分配：argmax $g'(V_s) \cdot v_{u,s}$，考虑边际递减
- 理论上凹收益可避免流量过度堆叠，提升总福利和公平性

---

## 2. 🧪 实验设定

### 2.1 依赖

```yaml
dependency:
  - MVP-0.3: Simulator V1 必须完成
  - 需要: GiftLiveSimulator, AllocationPolicy 接口
```

### 2.2 分配策略

```yaml
policies:
  # 基线策略
  - name: "random"
    description: "随机分配"
    formula: "uniform(streamers)"
  
  - name: "greedy"
    description: "贪心分配"
    formula: "argmax_s v_{u,s}"
  
  # 凹收益策略
  - name: "concave_log"
    description: "对数凹收益"
    formula: "argmax_s g'(V_s) * v_{u,s}"
    g_function: "log(1 + V)"
    g_derivative: "1 / (1 + V)"
  
  - name: "concave_exp"
    description: "饱和型凹收益"
    formula: "argmax_s g'(V_s) * v_{u,s}"
    g_function: "B * (1 - exp(-V/B))"
    g_derivative: "exp(-V/B)"
    B: 1000  # 饱和阈值
  
  # 带折扣的贪心
  - name: "greedy_with_cap"
    description: "贪心 + 头部上限"
    formula: "argmax_s v_{u,s} if V_s < cap else random"
    cap: 10000
```

### 2.3 实验配置

```yaml
experiments:
  # 实验1: 策略对比
  - name: "policy_comparison"
    policies: ["random", "greedy", "concave_log", "concave_exp", "greedy_with_cap"]
    n_simulations: 100
    n_users_per_sim: 10000
    n_streamers: 500
    n_rounds: 50  # 每轮分配一批用户
    metrics:
      - total_revenue        # 总收益
      - streamer_gini        # 主播收益 Gini
      - top_10_share         # Top 10% 主播收益占比
      - cold_start_coverage  # 冷启动主播覆盖率
  
  # 实验2: 凹函数参数敏感度
  - name: "concave_param_sweep"
    policy: "concave_exp"
    sweep_param: "B"
    values: [100, 500, 1000, 5000, 10000]
    n_simulations: 50
  
  # 实验3: 用户量规模效应
  - name: "scale_effect"
    policy: ["greedy", "concave_log"]
    n_users_per_sim: [1000, 5000, 10000, 50000]
    n_simulations: 30
```

### 2.4 评估指标

```yaml
metrics:
  # 效率指标
  total_revenue:
    formula: "sum(all gifts)"
    higher_better: true
  
  avg_revenue_per_user:
    formula: "total_revenue / n_users"
    higher_better: true
  
  # 公平性指标
  streamer_gini:
    formula: "gini(streamer_revenues)"
    lower_better: true  # 更公平
  
  top_10_share:
    formula: "sum(top_10% streamers) / total_revenue"
    lower_better: true
  
  # 覆盖指标
  cold_start_coverage:
    formula: "n_streamers_with_revenue / n_streamers"
    higher_better: true
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar (grouped) | Policy | Total Revenue | `../img/mvp21_revenue_comparison.png` |
| Fig2 | bar (grouped) | Policy | Gini / Top-10% Share | `../img/mvp21_fairness_comparison.png` |
| Fig3 | scatter | Total Revenue | Gini (lower=fairer) | `../img/mvp21_pareto_frontier.png` |
| Fig4 | line | B (saturation param) | Revenue / Gini | `../img/mvp21_param_sensitivity.png` |
| Fig5 | line | N Users | Revenue Δ (Concave - Greedy) | `../img/mvp21_scale_effect.png` |
| Fig6 | lorenz (multi) | Cumulative % Streamers | Cumulative % Revenue | `../img/mvp21_lorenz_by_policy.png` |

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
| `scripts/simulator/simulator.py` | `GiftLiveSimulator` | 添加分配策略接口 |
| `scripts/simulator/policies.py` | `AllocationPolicy` 基类 | 实现凹收益策略 |
| `scripts/eda_kuailive.py` | `calculate_gini()` | - |

**新增代码模块**：
1. `ConcaveAllocationPolicy`: 凹收益分配策略
2. `AllocationExperiment`: 分配实验框架
3. `evaluate_allocation()`: 评估函数

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `exp/exp_concave_allocation_20260108.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（凹收益 vs 贪心的 Δ收益、Δ公平性）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（DG3 验证结果 + Gate-2 判定）

### 5.2 数值结果
- **路径**: `../results/concave_allocation_20260108.json`
- **内容**: 
  ```json
  {
    "policy_comparison": {
      "greedy": {"revenue": ..., "gini": ..., "top_10_share": ...},
      "concave_log": {"revenue": ..., "gini": ...},
      "delta_revenue_pct": ...,
      "delta_gini": ...
    },
    "dg3_result": "closed" | "open",
    "gate2_decision": "confirm_allocation" | "simplify_to_greedy"
  }
  ```

---

## 6. 📤 报告抄送

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVmodel_roadmap.md` | MVP-2.1 状态 + Gate-2 结果 | §1.2, §2.1, §4.2, §4.3 |
| `gift_EVmodel_hub.md` | DG3 验证 + Q2.1 答案 + 洞见 | §1, §5, §4 |

---

## 7. ⚠️ 注意事项

- [ ] 依赖 MVP-0.3 完成后才能执行
- [ ] seed=42 固定随机性
- [ ] 多次模拟取均值 ± 标准差
- [ ] 注意累计收益 $V_s$ 的实时更新（每分配一个用户需更新）

---

## 8. 📐 核心公式

### 8.1 贪心分配

$$s^* = \arg\max_s v_{u,s}$$

### 8.2 凹收益分配

$$s^* = \arg\max_s g'(V_s) \cdot v_{u,s}$$

### 8.3 常用凹函数

| 名称 | $g(V)$ | $g'(V)$ |
|------|--------|---------|
| 对数型 | $\log(1+V)$ | $\frac{1}{1+V}$ |
| 饱和型 | $B(1-e^{-V/B})$ | $e^{-V/B}$ |
| 幂次型 | $V^\alpha, \alpha<1$ | $\alpha V^{\alpha-1}$ |

### 8.4 边际增益决策

$$\Delta(u \to s) = g(V_s + v_{u,s}) - g(V_s) \approx g'(V_s) \cdot v_{u,s}$$

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 确认 MVP-0.3 Simulator 已完成
3. ✅ 先读取 Simulator 代码理解接口
4. ✅ 复用 Simulator 的 AllocationPolicy 接口
5. ✅ 按模板输出 exp.md 报告
6. ✅ 运行前先 source init.sh
-->
