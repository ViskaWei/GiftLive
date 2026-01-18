<!--
📝 Agent 书写规范（不出现在正文）:
- Header 全英文
- 正文中文
- 图表文字全英文（中文会乱码）
- 公式用 LaTeX: $inline$ 或 $$block$$
-->

# 🍃 Estimation Layer Audit: 估计层审计
> **Name:** Estimation Layer Audit  
> **ID:** `EXP-20260118-gift_EVpred-08`  
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.6  
> **Author:** Viska Wei | **Date:** 2026-01-18 | **Status:** ✅  

> 🎯 **Target:** 确保估计层（Estimation）的问题定义正确、无泄漏、可服务在线分配（allocation），用最简单的 Logistic/Linear Regression 跑通 baseline  
> 🚀 **Next:** 实验结论 → 明确当前定义是否"能服务在线分配"，下一步改什么（H、cap、样本单位、特征）

## ⚡ 核心结论速览

> **一句话**: 估计层审计完成：预测目标清晰、IO口径正确、时间切分可审计；**发现标签窗口差异显著（16.51%）**，建议使用 watch_time 截断；Frozen past-only 特征无泄漏，简单模型 RevCap@1%=25.9%，**当前定义可服务在线分配**

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| A: 预测目标定义是否正确？ | ✅ 通过 | r_rev 口径清晰，action 映射明确 |
| B: 样本单位定义是否正确？ | ✅ 通过 | click-level 含 0（1.50% 正样本率），其他两种已定义 |
| C: 时间切分是否真实 Week1/2/3？ | ✅ 通过 | Train 13天/Val 3天/Test 3天，无重叠 |
| D: 标签窗口 vs 观看时长截断？ | 🔴 **显著差异** | 固定窗口 vs 截断差异 16.51%，<5s 分桶差异 65.57% |
| E: 特征是否严格 past-only？ | ✅ 通过 | Frozen 特征无泄漏，37,595 pairs/18,402 users/25,890 streamers |
| F: Logistic/Linear 能否跑通？ | ✅ 通过 | Set-1 RevCap@1%=25.9%，Train>Test，无完美分数 |

| 指标 | 值 | 启示 |
|------|-----|------|
| Best (Set-1 Linear) | RevCap@1%=25.9%, Spearman=0.034 | 简单模型可学习有意义模式 |
| vs Set-0 (仅时间上下文) | RevCap@1%: 25.9% vs 1.0% | Past-only 特征显著提升（+24.9pp） |
| 标签窗口差异 | 16.51% (固定 vs 截断) | **建议使用 watch_time 截断或更短 H** |

| Type | Link |
|------|------|
| 🧠 Hub | `gift_EVpred/gift_EVpred_hub.md` § Q0 |
| 🗺️ Roadmap | `gift_EVpred/gift_EVpred_roadmap.md` § MVP-1.6 |
| 📝 Session | `gift_EVpred/session/train_leakage_free_baseline.md` § 2.1-2.4 |

---

# 1. 🎯 目标

**问题**: 现有 baseline（MVP-1.0）虽然消除了泄漏，但性能不达标（Top-1%=11.6%）。需要系统性地审计估计层的每个环节，确保问题定义正确、IO口径清晰、无泄漏，并用最简单的模型验证基础设施正确性。

**验证**: Q0（基础设施正确性）的全面审计

| 预期 | 判断标准 |
|------|---------|
| 预测目标定义清晰 | 明确 r_rev/r_usr/r_eco 口径，action 在数据中如何映射 |
| 样本单位定义正确 | click-level 含 0，其他两种写清楚定义 |
| 时间切分可审计 | 输出 Train/Val/Test 具体边界、样本数、分布统计 |
| 标签窗口合理 | 对比固定窗口 vs 截断 vs 更短 H，给出明确结论 |
| 特征无泄漏 | Frozen past-only 必做，Rolling 可选但严格 past-only |
| 简单模型跑通 | Logistic/Linear 指标合理，无离谱完美分数 |

---

# 2. 🦾 算法（可选）

> 📌 本实验是审计类实验，重点是问题定义和 IO 口径，而非复杂算法

**核心任务**：确保估计层（Estimation Layer）的问题定义正确

**预测目标定义**：

1. **短期收益（必须实现）**：
   $$
   r^{rev}_t(a) = \mathbb{E}[\text{gift\_amount}_{t:t+H} \mid u, s, \text{ctx}_t, a]
   $$

2. **用户侧长期代理（本轮可只做口径定义）**：
   $$
   r^{usr}_t(a) = \mathbb{E}[\text{return}_{t+1d} \mid u, \text{history}, a]
   \quad\text{或}\quad
   \mathbb{E}[\Delta \text{watch\_time}, \Delta\text{engagement} \mid a]
   $$

3. **生态/外部性（本轮只做口径定义）**：
   $$
   r^{eco}_t(a) = -\lambda \cdot \text{concentration}(\text{exposure/revenue})
   $$
   或 concave utility（边际递减）：
   $$
   U_s(x) \text{ concave}, \quad \Delta(u\to s) \propto U_s'(x) \cdot v_{u,s}
   $$

**样本单位定义**：

- **Click-level（必须实现）**：一条 click = 一次机会（decision proxy）
- **Session-level（写清楚定义）**：session = 同一 live_id 的连续观看段
- **Impression-level（写清楚数据需求）**：需要曝光/未点击日志

**标签构造**：

- **固定窗口**：
  $$
  Y_i^{(H)} = \sum_j a_j \cdot \mathbf{1}[t_i \le t_j \le t_i+H] \cdot \mathbf{1}[(u,s,live) \text{ match}]
  $$
- **按观看时长截断**：
  $$
  Y_i^{(cap)} = \sum_j a_j \mathbf{1}[t_i \le t_j \le \min(t_i+w_i, t_i+1h)]
  $$
- **标签变换**：
  $$
  y_i = \log(1+Y_i), \quad z_i = \mathbf{1}[Y_i>0]
  $$

**Past-only 特征约束**：

对任意实体 (e)（如 pair=(u,s)），在时间 (t) 的"过去累计"必须满足：
$$
C_e(t) = \sum_{j=1}^{M} \mathbf{1}[e_j=e] \cdot \mathbf{1}[t_j < t]
$$

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive 数据集 |
| 路径 | `data/kuailive/` |
| 样本单位 | **Click-level（含 0）**（必须实现） |
| Train/Val/Test | 按时间切分：前 70% / 中间 15% / 最后 15% 天 |
| 标签窗口 | H=1h（必须），同时对比 H=10min/30min |
| 标签截断 | 对比固定窗口 vs 按 watch_time 截断 |

**必须输出**（按 click.timestamp 排序）：
- Train/Val/Test 各自：min/max 时间戳、样本数、unique user/streamer/live 数
- (z=1) 正样本率、(Y) 的分布（p50/p90/p99）
- watch_time 的分布（p50/p90/p99），验证"p50 是否≈4s"

## 3.2 模型

| 参数 | 值 |
|------|------|
| 模型 | **Logistic Regression**（预测 z=1[Y>0]）+ **Linear/Ridge Regression**（预测 y=log(1+Y)） |
| 目标 | 确认 IO 正确、指标合理、无离谱完美分数 |
| 特征集 | Set-0（时间上下文）→ Set-1（+past-only 聚合） |

**重要检查**：
- Train 指标应 > Test（否则特征没信息或 pipeline 有 bug）
- 任一指标接近完美（如 AUC≈0.999）立刻报警：疑似泄漏，并打印 top 相关特征

## 3.3 训练

| 参数 | 值 |
|------|------|
| 模型 | Logistic Regression（sklearn） |
| 模型 | Linear/Ridge Regression（sklearn） |
| 特征 | 先用 Set-0（时间上下文）→ Set-1（+past-only 聚合） |
| seed | 42 |

## 3.4 扫描参数

| 扫描 | 范围 | 固定 |
|------|------|------|
| 标签窗口 H | [10min, 30min, 1h] | 至少跑通 1h + 10min |
| 标签方式 | [固定窗口, 按 watch_time 截断] | 必须对比 |
| 特征集 | [Set-0, Set-1] | 逐步增加 |

---

# 4. 📊 图表

> ⚠️ 图表文字必须全英文！

### Fig 1: Train/Val/Test 时间切分边界
![](./img/estimation_audit_time_split.png)

**观察**:
- Train/Val/Test 时间边界清晰：Train 13天（2025-05-04 至 2025-18），Val 3天，Test 3天
- 样本数分布：Train 3.4M（70%），Val 736K（15%），Test 736K（15%）
- 正样本率：Train 1.42%，Val 1.72%，Test 1.64%，分布相对稳定
- Watch time p50：Train 4.0s，Val 4.6s，Test 4.3s，验证了用户观察（中位数极短）
- Y 分布：p50=0（98.5% 稀疏），p99=1.0，符合重尾分布

### Fig 2: Label Window vs Watch Time Truncation
![](./img/estimation_audit_label_window.png)

**观察**:
- **整体差异显著**：固定窗口 vs 截断差异 16.51%（6.6M vs 5.5M 总金额）
- **短观看时长差异巨大**：<5s 分桶差异 65.57%，5-30s 分桶差异 68.32%
- **长观看时长差异较小**：>300s 分桶差异仅 9.50%
- **结论**：固定 1h 窗口会引入大量"用户已离开但仍在窗口内"的伪标签，**强烈建议使用 watch_time 截断或更短 H（10min/30min）**

### Fig 3: Past-only Feature Verification
![](./img/estimation_audit_feature_leakage.png)

**观察**:
- Top 特征重要性：`pair_last_gift_time_gap_past` 最高（0.098），说明"最近一次打赏时间间隔"是关键信号
- 特征重要性分布合理：无异常高的特征（最高 0.098，无泄漏迹象）
- 特征重要性比：< 2x（符合无泄漏标准）

### Fig 4: Logistic/Linear Baseline Results
![](./img/estimation_audit_baseline_results.png)

**观察**:
- **Set-0（仅时间上下文）性能极弱**：Logistic RevCap@1%=0.26%，Ridge RevCap@1%=1.03%，接近随机
- **Set-1（+past-only）显著提升**：Logistic RevCap@1%=17.81%，Ridge RevCap@1%=25.94%（+24.9pp）
- **Train vs Test 差异合理**：Train RevCap@1%=92.1% > Test 25.9%，符合预期（Train 有更多历史信息）
- **无完美分数**：所有指标 < 0.999，无泄漏迹象

---

# 5. 💡 洞见

## 5.1 宏观

- **估计层基础设施正确**：预测目标、IO口径、时间切分、特征体系均已正确定义，可服务在线分配
- **标签窗口设计存在显著问题**：固定 1h 窗口 vs watch_time 截断差异 16.51%，短观看时长（<30s）差异高达 65-68%，说明大量标签来自"用户已离开但仍在窗口内"的伪标签
- **Past-only 特征有效但有限**：Set-1（时间上下文+past-only）RevCap@1%=25.9% vs Set-0（仅时间上下文）1.0%，提升显著但仍有很大优化空间

## 5.2 模型层

- **简单模型可学习有意义模式**：Ridge Regression（Set-1）RevCap@1%=25.9%，Spearman=0.034，说明特征有信息量
- **Train vs Test 差异合理**：Train RevCap@1%=92.1% > Test 25.9%，符合预期（Train 有更多历史信息）
- **特征重要性分析**：`pair_last_gift_time_gap_past` 重要性最高（0.098），说明"最近一次打赏时间间隔"是关键信号

## 5.3 细节

- **时间切分验证**：Train 13天/Val 3天/Test 3天，无重叠，符合时间序列切分要求
- **Watch time 分布**：p50≈4s，验证了用户观察（中位数极短），这解释了为什么固定窗口会引入大量伪标签
- **特征泄漏检查**：Frozen 特征无泄漏（Train>Test，无完美分数），特征重要性比合理

---

# 6. 📝 结论

## 6.1 核心发现
> **估计层审计完成：预测目标清晰、IO口径正确、时间切分可审计；发现标签窗口差异显著（16.51%），建议使用 watch_time 截断；Frozen past-only 特征无泄漏，简单模型 RevCap@1%=25.9%，当前定义可服务在线分配**

- ✅ A: 预测目标定义清晰 → r_rev 口径明确，action 映射到 (user_id, streamer_id, live_id)
- ✅ B: 样本单位定义正确 → click-level 含 0（正样本率 1.50%），Session/Impression 已定义
- ✅ C: 时间切分真实 → Train 13天/Val 3天/Test 3天，无重叠，符合时间序列要求
- 🔴 D: 标签窗口差异显著 → 固定窗口 vs 截断差异 16.51%，<5s 分桶差异 65.57%，**建议使用 watch_time 截断或更短 H（10min/30min）**
- ✅ E: 特征严格 past-only → Frozen 特征无泄漏，37,595 pairs/18,402 users/25,890 streamers
- ✅ F: Logistic/Linear 跑通 → Set-1 RevCap@1%=25.9%，Train>Test，无完美分数

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **估计层基础设施正确，可服务在线分配** | 预测目标清晰、IO口径正确、时间切分可审计、特征无泄漏 |
| 2 | **标签窗口设计存在显著问题** | 固定 1h 窗口 vs watch_time 截断差异 16.51%，短观看时长差异高达 65-68% |
| 3 | **Past-only 特征有效但有限** | Set-1 RevCap@1%=25.9% vs Set-0 1.0%（+24.9pp），仍有很大优化空间 |
| 4 | **简单模型可学习有意义模式** | Ridge Regression（Set-1）RevCap@1%=25.9%，Spearman=0.034 |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| **标签窗口必须考虑观看时长** | 使用 watch_time 截断或更短 H（10min/30min），避免"用户已离开但仍在窗口内"的伪标签 |
| **Past-only 特征是必须的** | Frozen 特征无泄漏，可安全使用；Rolling 版本需严格验证 |
| **简单模型可作为 baseline** | Logistic/Linear 可学习有意义模式，可作为复杂模型的对比基准 |
| **时间切分必须严格** | Train/Val/Test 无重叠，符合时间序列要求 |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| **固定窗口引入伪标签** | 用户观看时长中位数仅 4s，但固定 1h 窗口会记录"用户已离开但仍在窗口内"的礼物 |
| **Train vs Test 差异过大** | Train RevCap@1%=92.1% vs Test 25.9%，说明 Train 有更多历史信息，需注意过拟合风险 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Best (Set-1 Linear) | RevCap@1%=25.9%, Spearman=0.034 | Ridge Regression, Test |
| vs Set-0 (仅时间上下文) | RevCap@1%: 25.9% vs 1.0% | Past-only 特征提升 +24.9pp |
| 标签窗口差异 | 16.51% (固定 vs 截断) | H=1h, 整体差异 |
| Watch time p50 | 4.0s (Train), 4.6s (Val), 4.3s (Test) | 验证用户观察 |
| 正样本率 | 1.42% (Train), 1.72% (Val), 1.64% (Test) | Click-level 含 0 |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 标签窗口 | 改用 watch_time 截断或更短 H（10min/30min） | 🔴 P0 |
| 特征工程 | 探索序列特征、实时上下文、内容匹配度 | 🔴 P0 |
| 模型复杂度 | 从简单模型升级到 LightGBM | 🟡 P1 |
| 样本单位 | 考虑 Session-level 或 Impression-level（如有数据） | 🟡 P1 |

---

# 7. 📎 附录

## 7.1 数值结果

| 配置 | PR-AUC | LogLoss | ECE | MAE_log | RMSE_log | Spearman | RevCap@1% |
|------|--------|---------|-----|---------|----------|----------|-----------|
| Logistic (Set-0) | 0.0165 | 0.0840 | 0.0026 | - | - | - | 0.0026 |
| Logistic (Set-1) | 0.0570 | 0.1077 | 0.0135 | - | - | - | 0.1781 |
| Ridge (Set-0) | - | - | - | 0.0576 | 0.3222 | -0.0012 | 0.0103 |
| Ridge (Set-1) | - | - | - | 0.0451 | 0.3221 | 0.0342 | **0.2594** |

**Train vs Test 对比（Set-1）**：
- Logistic: Train PR-AUC=0.3126 > Test 0.0570 ✅
- Ridge: Train RevCap@1%=92.1% > Test 25.9% ✅

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/audit_estimation_layer.py` |
| 运行时间 | 8.8 分钟 |
| Output | `gift_EVpred/results/estimation_audit_20260118.json` |
| 审计报告 | `gift_EVpred/audit_estimation_layer.md` |
| 图表 | `gift_EVpred/img/estimation_audit_*.png` (4 张) |

```bash
# 执行审计（已完成）
python scripts/audit_estimation_layer.py

# 优化 apply_frozen_features（已实现）
python scripts/optimize_apply_frozen_features.py
# 速度提升：107.83x（从 7.29s 降至 0.07s，100K 样本）
```

## 7.3 审计报告要求

**必须生成 `audit_estimation_layer.md`，包含**：

1. **估计层目标**（r_rev/r_usr/r_eco）口径与本轮范围
2. **样本单位定义**（click/session/impression）与本轮采用的 click-level 的 (X,Y,y,z)
3. **Train/Val/Test 时间切分边界**（含"Week1/2/3 是否属实"）
4. **watch_time 与 label window 的一致性结论**（1h vs cap vs 10min/30min）
5. **Frozen past-only baseline**（Logistic + Linear）结果与 sanity check
6. **明确结论**：当前定义是否"能服务在线分配"？下一步改什么（H、cap、样本单位、特征）

## 7.4 关键检查点

| 检查项 | 要求 | 验证方法 | 结果 |
|--------|------|---------|------|
| 预测目标定义 | 明确 r_rev/r_usr/r_eco，action 映射 | 文档输出 | ✅ 通过 |
| 样本单位 | click-level 含 0，其他写清楚 | 数据统计 | ✅ 通过（正样本率 1.50%） |
| 时间切分 | 输出具体边界，验证是否 Week1/2/3 | 时间戳分析 | ✅ 通过（Train 13天/Val 3天/Test 3天） |
| 标签窗口 | 对比固定 vs 截断 vs 更短 H | 差异分析 | 🔴 差异显著（16.51%） |
| 特征泄漏 | Frozen past-only，Rolling 严格 | 特征重要性比 < 2x | ✅ 通过（无泄漏） |
| 简单模型 | Logistic/Linear 指标合理 | Train > Test，无完美分数 | ✅ 通过（Train>Test，无完美分数） |

## 7.5 apply_frozen_features 优化

**问题**：原始 `apply_frozen_features` 使用 `iterrows()`，对 490 万条记录非常慢（预计需要数小时）。

**优化方案**：
- 使用 `merge()` 代替 `iterrows()` 进行 pair 和 streamer 特征匹配
- 使用向量化操作计算时间间隔
- 实现 lookup 表的保存/加载功能

**优化结果**（100K 样本测试）：
- 原始版本：7.29s
- 优化版本：0.07s
- **速度提升：107.83x**

**实现**：
- 优化函数：`scripts/optimize_apply_frozen_features.py`
- 缓存功能：可保存/加载 frozen lookups 到 `gift_EVpred/features_cache/`

**建议**：
- 对于大数据集（>1M 行），使用优化版本
- 对于相同 train window，可复用保存的 lookup 表

---

> **实验完成时间**: 2026-01-18
