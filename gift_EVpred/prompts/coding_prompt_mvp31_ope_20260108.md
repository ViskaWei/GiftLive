# 🤖 Coding Prompt: OPE 验证

> **Experiment ID:** `EXP-20260108-gift-allocation-10`  
> **MVP:** MVP-3.1  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证 Off-Policy Evaluation (OPE) 方法在高方差打赏场景下的有效性，关闭 Gate-3

**验证假设**：Hub Q3.1 - OPE(IPS/DR)在高方差场景的有效性如何？

**预期结果**：
- 若 OPE 估计与 Simulator 真值误差 < 10% → OPE 可用于线上策略对比
- 若 误差 ≥ 10% → 需依赖 Simulator 或小流量 A/B 测试

**背景**：
- 打赏金额重尾（Gini=0.94），方差极大
- IPS 在高方差场景容易爆炸
- DR（Doubly Robust）可能更稳定
- 需要带 propensity 的日志数据

---

## 2. 🧪 实验设定

### 2.1 依赖

```yaml
dependency:
  - MVP-0.3: Simulator V1（需支持日志生成+propensity记录）
  - Simulator 需输出: (user, streamer, action, reward, propensity)
```

### 2.2 OPE 方法

```yaml
ope_methods:
  # 基线
  - name: "on_policy"
    description: "直接在目标策略上采样（真值）"
    use_for: "ground_truth"
  
  # IPS 系列
  - name: "ips"
    description: "Inverse Propensity Scoring"
    formula: "mean(reward * pi(a|x) / mu(a|x))"
    clip: false
  
  - name: "ips_clipped"
    description: "Clipped IPS"
    formula: "mean(reward * min(pi/mu, M))"
    clip_M: 10
  
  - name: "snips"
    description: "Self-Normalized IPS"
    formula: "sum(w * reward) / sum(w)"
    normalize: true
  
  # Doubly Robust
  - name: "dr"
    description: "Doubly Robust"
    formula: "mean(q(x,a) + w * (r - q(x,a)))"
    q_model: "LightGBM"
  
  - name: "dr_clipped"
    description: "Clipped DR"
    clip_M: 10
```

### 2.3 实验配置

```yaml
experiments:
  # 实验1: OPE 方法对比
  - name: "ope_comparison"
    behavior_policy: "epsilon_greedy"  # 日志采集策略
    epsilon: 0.3                        # 30% 随机探索
    target_policies:
      - "greedy"
      - "concave_log"
      - "soft_cold_start"
    n_logs: 100000                     # 日志样本数
    n_repeats: 100                     # 重复次数估计方差
    metrics:
      - bias                           # E[估计] - 真值
      - variance                       # Var[估计]
      - mse                            # 均方误差
      - relative_error                 # |估计-真值| / 真值
  
  # 实验2: 日志量敏感度
  - name: "sample_size_sweep"
    ope_method: ["ips", "snips", "dr"]
    n_logs: [1000, 5000, 10000, 50000, 100000, 500000]
    n_repeats: 50
  
  # 实验3: 探索率敏感度
  - name: "epsilon_sweep"
    behavior_policy: "epsilon_greedy"
    epsilon: [0.1, 0.2, 0.3, 0.5, 0.8]
    ope_method: ["snips", "dr"]
    n_logs: 50000
  
  # 实验4: 策略差异敏感度
  - name: "policy_gap_sweep"
    description: "目标策略与行为策略差异越大，OPE越难"
    target_policies:
      - "near_behavior"    # 与行为策略相似
      - "moderate_diff"
      - "very_different"
```

### 2.4 评估指标

```yaml
metrics:
  # 核心
  relative_error:
    formula: "abs(estimate - truth) / truth"
    threshold: 0.1   # < 10% 算可接受
  
  bias:
    formula: "mean(estimate) - truth"
  
  variance:
    formula: "var(estimate)"
  
  mse:
    formula: "bias^2 + variance"
  
  # 辅助
  effective_sample_size:
    formula: "(sum(w))^2 / sum(w^2)"
    description: "有效样本量，越大越好"
  
  max_weight:
    formula: "max(pi/mu)"
    description: "最大重要性权重，过大说明 IPS 不稳定"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar (grouped) | OPE Method | Relative Error (mean ± std) | `../img/mvp31_ope_comparison.png` |
| Fig2 | bar (grouped) | OPE Method | Bias / Variance / MSE | `../img/mvp31_bias_variance.png` |
| Fig3 | line (multi) | N Logs | Relative Error | `../img/mvp31_sample_size_effect.png` |
| Fig4 | line (multi) | Epsilon (exploration) | Relative Error | `../img/mvp31_epsilon_effect.png` |
| Fig5 | box | OPE Method | Estimate Distribution | `../img/mvp31_estimate_distribution.png` |
| Fig6 | heatmap | Target Policy × OPE Method | Relative Error | `../img/mvp31_policy_ope_heatmap.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize**: 单张 `(6, 5)`

---

## 4. 📁 参考代码

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/simulator/simulator.py` | `GiftLiveSimulator` | 添加 propensity 日志输出 |
| `scripts/simulator/policies.py` | 各种策略 | 添加 `get_propensity()` 方法 |

**新增代码模块**：
1. `IPSEstimator`: IPS / Clipped IPS / SNIPS
2. `DoublyRobustEstimator`: DR 估计器
3. `RewardModel`: Q 函数估计（用于 DR）
4. `OPEExperiment`: OPE 实验框架
5. `LogGenerator`: 带 propensity 的日志生成器

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `exp/exp_ope_validation_20260108.md`
- **模板**: `_backend/template/exp.md`

### 5.2 数值结果
- **路径**: `../results/ope_validation_20260108.json`
- **内容**: 
  ```json
  {
    "ope_comparison": {
      "ips": {"bias": ..., "variance": ..., "mse": ..., "rel_error": ...},
      "snips": {...},
      "dr": {...}
    },
    "best_method": "dr",
    "best_rel_error": 0.08,
    "gate3_result": "closed" | "open",
    "recommendation": "dr_for_policy_comparison" | "use_simulator"
  }
  ```

---

## 6. 📤 报告抄送

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP-3.1 状态 + Gate-3 结果 | §1.2, §2.1, §4.2 |
| `gift_EVpred_hub.md` | Q3.1 答案 + OPE 设计原则 | §1, §6.1 |

---

## 7. ⚠️ 注意事项

- [ ] 依赖 MVP-0.3 Simulator 完成
- [ ] propensity 必须在日志生成时记录（不能事后估计）
- [ ] IPS 权重可能爆炸，务必监控 max_weight
- [ ] DR 需要额外训练 Q 函数，注意过拟合
- [ ] seed=42 固定随机性

---

## 8. 📐 核心公式

### 8.1 Inverse Propensity Scoring (IPS)

$$\hat{V}^{\text{IPS}}(\pi) = \frac{1}{n} \sum_{i=1}^{n} \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)} r_i$$

### 8.2 Self-Normalized IPS (SNIPS)

$$\hat{V}^{\text{SNIPS}}(\pi) = \frac{\sum_{i=1}^{n} w_i r_i}{\sum_{i=1}^{n} w_i}, \quad w_i = \frac{\pi(a_i|x_i)}{\mu(a_i|x_i)}$$

### 8.3 Doubly Robust (DR)

$$\hat{V}^{\text{DR}}(\pi) = \frac{1}{n} \sum_{i=1}^{n} \left[ \hat{Q}(x_i, \pi) + w_i \left(r_i - \hat{Q}(x_i, a_i)\right) \right]$$

其中 $\hat{Q}(x,a)$ 是奖励模型

### 8.4 有效样本量

$$n_{\text{eff}} = \frac{\left(\sum_i w_i\right)^2}{\sum_i w_i^2}$$

---

## 9. 🔬 技术细节

### 9.1 Propensity 记录

```python
# 在 Simulator 中
log_entry = {
    "user_id": u,
    "streamer_id": s,
    "action": a,
    "reward": r,
    "propensity": mu(a | x)  # 行为策略概率
}
```

### 9.2 IPS 稳定化

- **Clipping**: $w' = \min(w, M)$, 常用 $M \in [5, 20]$
- **Self-normalization**: 除以权重和，减少方差
- **Weight capping**: 极端权重截断

### 9.3 DR 的 Q 函数训练

```yaml
q_model:
  type: LightGBM
  features: [user_features, streamer_features, context]
  target: reward
  train_on: behavior_logs
```

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 确认 MVP-0.3 Simulator 已完成且支持 propensity
3. ✅ 复用 Simulator 的策略和日志生成
4. ✅ 按模板输出 exp.md 报告
-->
