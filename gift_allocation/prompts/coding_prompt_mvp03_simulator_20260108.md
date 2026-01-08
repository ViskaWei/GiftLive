# 🤖 Coding Prompt: Simulator V1 构建

> **Experiment ID:** `EXP-20260108-gift-allocation-07`  
> **MVP:** MVP-0.3  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：构建可控的直播打赏模拟器，支持后续分配层(MVP-2.1/2.2)和评估层(MVP-3.1)验证

**验证假设**：Simulator 能否复现 KuaiLive 真实数据的关键统计特性（打赏率、Gini系数、金额分布）

**预期结果**：
- 若 Simulator 统计特性与真实数据误差 < 20% → Simulator 可用于分配策略评估
- 若 误差 > 20% → 需调优参数或增加模型复杂度

**动机**：
- Gate-2（分配层验证）需要可控环境测试凹收益分配
- Gate-3（OPE验证）需要带 propensity 的日志数据
- 真实 A/B 测试成本高，Simulator 提供快速迭代能力

---

## 2. 🧪 实验设定

### 2.1 Simulator 设计

```yaml
simulator:
  name: "GiftLiveSimulator V1"
  components:
    # 1. 用户池
    user_pool:
      n_users: 10000
      wealth_dist:
        type: mixture
        components:
          - dist: lognormal
            mean: 3.0
            std: 1.0
            weight: 0.95   # 普通用户
          - dist: pareto
            alpha: 1.5
            min: 100
            weight: 0.05   # 大R玩家
      preference_dim: 16   # 用户兴趣向量维度
    
    # 2. 主播池
    streamer_pool:
      n_streamers: 500
      popularity_dist: pareto  # alpha=1.2，头部效应
      content_dim: 16          # 内容向量维度
    
    # 3. 打赏概率模型
    gift_prob:
      formula: "sigmoid(theta0 + theta1*log_wealth + theta2*match + theta3*engage + theta4*crowd)"
      params:
        theta0: -5.0      # 基础打赏率很低
        theta1: 0.5       # 财富系数
        theta2: 1.0       # 匹配度系数 (match = user_pref · streamer_content)
        theta3: 0.3       # 互动系数
        theta4: -0.1      # 拥挤效应（边际递减）
    
    # 4. 打赏金额模型
    gift_amount:
      formula: "exp(mu0 + mu1*log_wealth + mu2*match + epsilon)"
      params:
        mu0: 2.0
        mu1: 0.8
        mu2: 0.5
        sigma: 0.5        # 噪声标准差
    
    # 5. 延迟模型（保留接口，但基于 MVP-1.2 结论简化）
    delay:
      dist: none          # 打赏立即发生（中位数=0s）
      # 备选: weibull, shape=1.5, scale=10 (分钟)
    
    # 6. 观看时长模型
    watch_duration:
      dist: lognormal
      mean: 2.0           # log(分钟)
      std: 1.0
    
    # 7. 外部性/边际递减
    externality:
      type: diminishing_returns
      formula: "Y / (1 + gamma * N_big)"
      gamma: 0.05         # 边际递减系数
```

### 2.2 校准目标

```yaml
calibration_targets:
  # 来自 MVP-0.1 的真实数据统计
  gift_rate: 0.0148           # 1.48% 打赏率
  user_gini: 0.942            # 用户打赏额 Gini
  streamer_gini: 0.930        # 主播收益 Gini
  top_1pct_user_share: 0.599  # Top 1% 用户贡献 60%
  amount_p50: 2
  amount_p90: 88
  amount_p99: 1488
  amount_mean: 82.7
```

### 2.3 实验配置

```yaml
experiments:
  # 实验1: 校准验证
  - name: "calibration_check"
    n_simulations: 100
    n_interactions_per_sim: 100000
    metrics: ["gift_rate", "user_gini", "streamer_gini", "amount_distribution"]
  
  # 实验2: 分配策略对比（预览，为 MVP-2.1 铺路）
  - name: "allocation_preview"
    policies:
      - name: "greedy"           # 贪心：argmax v_us
      - name: "uniform"          # 均匀随机
      - name: "round_robin"      # 轮流分配
    n_simulations: 50
    metrics: ["total_revenue", "streamer_gini", "top_10_share"]
  
  # 实验3: 外部性敏感度分析
  - name: "externality_sweep"
    sweep_param: "gamma"
    values: [0, 0.01, 0.05, 0.1, 0.2]
    n_simulations: 50
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar (grouped) | Metric | Value (Real vs Simulated) | `gift_allocation/img/mvp03_calibration_comparison.png` |
| Fig2 | histogram (overlay) | Gift Amount (log scale) | Frequency | `gift_allocation/img/mvp03_amount_distribution.png` |
| Fig3 | lorenz curve | Cumulative % Users | Cumulative % Revenue | `gift_allocation/img/mvp03_lorenz_curve.png` |
| Fig4 | bar (grouped) | Policy | Total Revenue / Gini | `gift_allocation/img/mvp03_policy_preview.png` |
| Fig5 | line | Gamma (externality) | Total Revenue / Fairness | `gift_allocation/img/mvp03_externality_sweep.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则（必须遵守）**：
  - 单张图：`figsize=(6, 5)` 锁死
  - 多张图（subplot）：按 6:5 比例扩增，如 `(12, 5)` for 1×2, `(12, 10)` for 2×2

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/eda_kuailive.py` | Gini 计算 `calculate_gini()`、Lorenz 曲线绘制 | - |
| `gift_allocation/results/eda_stats_20260108.json` | 真实数据统计作为校准目标 | - |
| `_backend/template/exp.md` | 报告模板 | - |

**新增代码模块**：
1. `GiftLiveSimulator`: 核心模拟器类
2. `UserPool`, `StreamerPool`: 用户/主播生成
3. `GiftModel`: 打赏概率+金额模型
4. `AllocationPolicy`: 分配策略接口（贪心、均匀等）
5. `SimulatorCalibrator`: 参数校准工具

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_allocation/exp/exp_simulator_v1_20260108.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（Simulator 是否校准成功 + 关键指标误差）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（Simulator 可用性判定 + 后续 MVP 铺垫）

### 5.2 图表文件
- **路径**: `gift_allocation/img/`
- **命名**: `mvp03_*.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `gift_allocation/results/simulator_v1_20260108.json`
- **内容**: 
  ```json
  {
    "calibration": {
      "gift_rate": {"real": 0.0148, "sim": ..., "error_pct": ...},
      "user_gini": {"real": 0.942, "sim": ..., "error_pct": ...},
      "streamer_gini": {"real": 0.930, "sim": ..., "error_pct": ...}
    },
    "policy_preview": {
      "greedy": {"revenue": ..., "gini": ...},
      "uniform": {"revenue": ..., "gini": ...}
    },
    "externality_sweep": {...}
  }
  ```

### 5.4 Simulator 代码
- **路径**: `scripts/simulator/` (新目录)
  - `simulator.py`: 核心模拟器
  - `policies.py`: 分配策略
  - `calibration.py`: 校准工具

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_allocation_roadmap.md` | MVP-0.3 状态 → ✅ + 结论快照 | §2.1, §4.3 |
| `gift_allocation_hub.md` | Q3.2 验证状态 + Simulator 参数 | §1, §6.2 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中添加 seed=42 固定随机性 (numpy, random)
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/mvp03_simulator_20260108.log`
- [ ] Simulator 参数可通过 YAML 配置，便于后续调优
- [ ] 保留扩展接口（如延迟模型、羊群效应）
- [ ] 长时间模拟使用 nohup 后台运行

---

## 8. 📐 核心公式

### 8.1 打赏概率

$$P(\text{gift}|u,s) = \sigma\left(\theta_0 + \theta_1 \log(w_u) + \theta_2 (p_u \cdot q_s) + \theta_3 e_{us} + \theta_4 N_s\right)$$

其中：
- $w_u$: 用户财富
- $p_u, q_s$: 用户偏好/主播内容向量
- $e_{us}$: 互动指标
- $N_s$: 当前直播间金主数（拥挤效应）

### 8.2 打赏金额

$$Y|(\text{gift}=1) = \exp\left(\mu_0 + \mu_1 \log(w_u) + \mu_2 (p_u \cdot q_s) + \epsilon\right), \quad \epsilon \sim N(0, \sigma^2)$$

### 8.3 边际递减

$$\tilde{Y}_{us} = \frac{Y_{us}}{1 + \gamma \cdot N_{\text{big},s}}$$

其中 $N_{\text{big},s}$ 是直播间已有大额打赏数

### 8.4 Gini 系数

$$G = 1 - 2 \int_0^1 L(p) dp$$

其中 $L(p)$ 是 Lorenz 曲线

---

## 9. 🔗 后续 MVP 衔接

| 后续 MVP | 依赖 Simulator 的功能 |
|---------|---------------------|
| MVP-2.1 凹收益分配 | 需要 `AllocationPolicy` 接口 + 可控环境测试 |
| MVP-2.2 冷启动约束 | 需要 `StreamerPool` 支持新主播标记 |
| MVP-3.1 OPE 验证 | 需要带 `propensity` 的日志数据生成 |

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数（如 Gini 计算），不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 运行前先 source init.sh
7. ✅ Simulator 设计需模块化，便于后续 MVP 复用
-->
