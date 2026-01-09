# 🤖 Coding Prompt: Simulator V2 - 并发容量建模

> **Experiment ID:** `EXP-15`  
> **MVP:** MVP-4.2  
> **Date:** 2026-01-09  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：为 Simulator 添加主播并发容量约束，验证拥挤外部性对收益的边际递减效应

**验证假设**：H4.2 - 并发容量影响收益边际

**背景问题**：V1 Simulator 无主播承载上限，无法模拟"过度集中导致体验损伤"

**预期结果**：
- 若观测到收益随并发边际递减 → 继续 MVP-4.4/4.5
- 若无明显效应 → 调整拥挤惩罚函数参数

---

## 2. 🧪 实验设定

### 2.1 并发容量模型

```yaml
concurrency_model:
  # 主播容量分层
  capacity_by_popularity:
    top_10%: 100    # 头部主播容量高
    middle_40%: 50  # 中部主播
    tail_50%: 20    # 长尾主播

  # 拥挤惩罚函数
  crowding_penalty:
    type: "inverse"  # 1 / (1 + beta * overflow_ratio)
    beta: 0.5        # 惩罚强度

  # 溢出处理
  overflow_mode: "degrade"  # degrade (降级体验) 或 reject (拒绝)
```

### 2.2 实验设计

```yaml
experiments:
  # 实验1: 基准对比 (有/无容量约束)
  baseline_comparison:
    configs:
      - name: "no_capacity"
        enable_capacity: false
      - name: "with_capacity"
        enable_capacity: true
    metrics: [revenue, gini, top_10_share]

  # 实验2: 并发压力测试
  concurrency_sweep:
    users_per_round: [50, 100, 200, 400, 800]
    fixed:
      n_streamers: 100
      enable_capacity: true
    observe: "revenue vs concurrency curve"

  # 实验3: 容量敏感性
  capacity_sensitivity:
    capacity_scale: [0.5, 1.0, 2.0]  # 相对默认容量的倍数
    fixed:
      users_per_round: 400

  # 实验4: 惩罚强度扫描
  beta_sweep:
    beta: [0.1, 0.3, 0.5, 1.0, 2.0]
    fixed:
      enable_capacity: true
      users_per_round: 400
```

### 2.3 验收标准

```yaml
acceptance:
  # Gate-4A 附加条件
  marginal_decreasing:
    condition: "revenue growth rate decreases as concurrency increases"
    evidence: "second derivative of revenue curve < 0"
  
  saturation_visible:
    condition: "top streamers show revenue saturation"
    evidence: "top-10% streamer revenue flattens at high concurrency"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | 收益曲线 (line) | 并发用户数 | 总收益 | `gift_allocation/img/mvp42_revenue_vs_concurrency.png` |
| Fig2 | 边际收益 (line) | 并发用户数 | 边际收益 | `gift_allocation/img/mvp42_marginal_revenue.png` |
| Fig3 | 对比柱状图 (bar) | 配置 (有/无容量) | 收益/Gini | `gift_allocation/img/mvp42_capacity_comparison.png` |
| Fig4 | 热力图 (heatmap) | Beta | Capacity Scale | `gift_allocation/img/mvp42_param_heatmap.png` |
| Fig5 | 主播收益分布 (box) | 主播类型 | 收益 | `gift_allocation/img/mvp42_streamer_revenue_by_tier.png` |

**图表要求**：
- 所有文字必须英文
- figsize: 单张 `(6, 5)`，多张按比例扩增

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！只写路径，让 Agent 自己读取**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/simulator/simulator.py` | `SimConfig`, `Streamer`, `GiftModel` | 添加容量和拥挤惩罚 |
| `scripts/run_simulator_experiments.py` | 实验框架、绘图函数 | 添加 MVP-4.2 实验函数 |

**修改要点**：
1. 在 `SimConfig` 中添加容量参数
2. 在 `Streamer` 中添加 `capacity` 和 `current_load` 属性
3. 在 `GiftModel.apply_diminishing_returns()` 中添加拥挤惩罚
4. 添加 `run_mvp42_concurrency()` 函数
5. 添加 `plot_mvp42_figures()` 函数

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `gift_allocation/exp/exp_concurrency_capacity_20260109.md`
- **模板**: `_backend/template/exp.md`

### 5.2 图表文件
- **路径**: `gift_allocation/img/mvp42_*.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `gift_allocation/results/concurrency_capacity_20260109.json`

---

## 6. 📤 报告抄送

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_allocation_roadmap.md` | MVP-4.2 状态 + 结论 | §Phase 4 MVP 列表 |
| `gift_allocation_phase4_charter.md` | H4.2 验证结果 | §6.1 新假设 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中添加 seed 固定随机性
- [ ] 图表文字全英文
- [ ] 使用 `nohup` 后台运行

---

## 8. 🔑 技术关键点

### 8.1 容量分配逻辑

```
1. 根据主播 popularity 分配容量
   - top 10%: capacity = 100
   - middle 40%: capacity = 50
   - tail 50%: capacity = 20

2. 每轮开始前重置 current_load = 0

3. 分配用户时更新 current_load
```

### 8.2 拥挤惩罚函数

```
crowding_penalty(n_current, capacity):
    if n_current <= capacity:
        return 1.0
    else:
        overflow_ratio = (n_current - capacity) / capacity
        return 1.0 / (1 + beta * overflow_ratio)
```

### 8.3 验收检查

| 检查项 | 方法 | 通过条件 |
|--------|------|----------|
| 边际递减 | 计算收益二阶导 | < 0 在高并发区域 |
| 头部饱和 | Top-10% 收益曲线 | 斜率趋近 0 |
| 差异显著 | 有/无容量对比 | 差异 > 5% |

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 运行前必须 `source init.sh`
-->
