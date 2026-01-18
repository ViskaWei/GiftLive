# 🍃 Multi-Task Learning for Gift Prediction
> **Name:** Multi-Task Learning  
> **ID:** `EXP-20260108-gift-allocation-06`  
> **Topic:** `gift_EVpred` | **MVP:** MVP-1.3  
> **Author:** Viska Wei | **Date:** 2026-01-08 | **Status:** ✅  

> 🎯 **Target:** 验证多任务学习（用密集信号 watch/comment 扶起稀疏打赏信号）能否提升打赏预测性能  
> 🚀 **Next:** DG5 已关闭 → 多任务学习在当前场景无显著收益，保留单任务方案

## ⚡ 核心结论速览

> **一句话**: ❌ **多任务学习未能提升打赏预测**，PR-AUC 反降 1.76pp (0.182 → 0.165)，密集信号迁移效果不显著，DG5 关闭

| 验证问题 | 结果 | 结论 |
|---------|------|------|
| DG5: 密集信号对打赏预测的迁移效果？ | ❌ Δ PR-AUC = -1.76pp | 多任务无优势，保留单任务 |

| 指标 | Single-Task | Multi-Task | Δ | 结论 |
|------|-------------|------------|---|------|
| **PR-AUC** | **0.182** | 0.165 | **-1.76pp** | 单任务更优 |
| ROC-AUC | 0.900 | **0.905** | +0.5pp | 多任务略优 |
| Top-1% Capture | **15.3%** | 12.0% | **-3.3pp** | 单任务更优 |
| Top-5% Capture | 13.3% | **16.4%** | +3.1pp | 多任务更优 |
| Spearman | 0.188 | **0.199** | +1.1pp | 多任务更优 |
| ECE | 0.112 | **0.107** | -0.5pp | 多任务校准更好 |

| Type | Link |
|------|------|
| 🧠 Hub | `../gift_EVpred_hub.md` § Q1.2 |
| 🗺️ Roadmap | `../gift_EVpred_roadmap.md` § MVP-1.3 |

---
# 1. 🎯 目标

**问题**: 多任务学习（联合训练 watch_time/comment/gift）能否利用密集行为信号提升稀疏打赏预测？

**验证**: DG5 - 密集信号对打赏预测的迁移效果

| 预期 | 判断标准 |
|------|---------|
| Multi-Task PR-AUC ≥ Single-Task + 3pp | 通过 → 采用多任务架构 |
| Δ < 3pp 或负 | 拒绝 → 保留单任务，DG5 关闭 |

**背景**：
- 打赏率极稀疏 (1.82%)，正负样本严重不均衡
- 评论率 (4.88%) 和观看时长是更密集的信号
- 假设：共享表征可以让稀疏任务借助密集任务的监督信号

---

# 2. 🦾 算法

**Multi-Task MLP 架构**：

```
Input Features (68 dims)
        │
   ┌────▼────┐
   │ Shared  │  [256 → 128 → 64] + BN + ReLU + Dropout(0.2)
   │ Tower   │
   └────┬────┘
        │
   ┌────┼────┬────┬────┐
   ▼    ▼    ▼    ▼    ▼
 Head1 Head2 Head3 Head4
  [32]  [32]  [32]  [32]
   │     │     │     │
watch  comment gift  amount
(MSE)  (BCE)  (Focal)(MSE)
```

**Focal Loss**（应对 1.82% 正例）：
$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$
$\alpha=0.25, \gamma=2.0$

**多任务损失加权**：
$$\mathcal{L} = 0.3 \cdot \mathcal{L}_{\text{watch}} + 0.2 \cdot \mathcal{L}_{\text{comment}} + 0.3 \cdot \mathcal{L}_{\text{gift}} + 0.2 \cdot \mathcal{L}_{\text{amount}}$$

---

# 3. 🧪 实验设计

## 3.1 数据

| 项 | 值 |
|----|-----|
| 来源 | KuaiLive (click 全量) |
| 路径 | `data/KuaiLive/` |
| Train/Val/Test | 1,872,509 / 1,701,600 / 1,335,406 |
| 特征维度 | 68 |
| 打赏率 (Train) | 1.82% |
| 评论率 (Train) | 4.88% |

## 3.2 模型

| 参数 | Single-Task | Multi-Task |
|------|-------------|------------|
| 架构 | MLP [256,128,64] + 2 heads | MLP [256,128,64] + 4 heads |
| 任务 | has_gift + gift_amount | watch_time + has_comment + has_gift + gift_amount |
| 损失权重 | {gift:0.8, amount:0.2} | {watch:0.3, comment:0.2, gift:0.3, amount:0.2} |

## 3.3 训练

| 参数 | 值 |
|------|-----|
| epochs | 50 (early stop) |
| batch_size | 4096 |
| lr | 1e-3 |
| optimizer | Adam |
| scheduler | ReduceLROnPlateau |
| patience | 10 |
| seed | 42 |
| device | CUDA |

## 3.4 对比实验

| 实验 | 任务 | 说明 |
|------|------|------|
| Single-Task | has_gift + gift_amount | 只训练打赏相关任务 |
| Multi-Task | 4 tasks 全部 | 联合训练所有任务 |

---

# 4. 📊 图表

### Fig 1: PR-AUC Comparison
![](./img/mvp13_pr_auc_comparison.png)

**观察**:
- Single-Task PR-AUC = 0.182，Multi-Task = 0.165
- **多任务反而降低了 PR-AUC**，Δ = -1.76pp
- 决策阈值 ≥3pp 未达成

### Fig 2: Training Curves
![](./img/mvp13_training_curves.png)

**观察**:
- Single-Task 训练 50 epochs，Multi-Task 在 21 epochs 早停
- Multi-Task 损失下降更快（多任务正则化效应），但验证集 PR-AUC 未能超越
- Multi-Task 最佳 epoch 仅为 11，说明模型容量被非目标任务占用

### Fig 3: Top-K% Capture Comparison
![](./img/mvp13_topk_capture.png)

**观察**:
- **Top-1% Capture**：Single 15.3% > Multi 12.0% (-3.3pp) ❌
- **Top-5% Capture**：Single 13.3% < Multi 16.4% (+3.1pp) ✅
- 多任务在精细排序上略有优势，但顶部金主识别能力下降

### Fig 4: All Metrics Comparison
![](./img/mvp13_all_metrics.png)

**观察**:
- ROC-AUC: Multi 略优 (90.5% vs 90.0%)
- Spearman: Multi 略优 (19.9% vs 18.8%)
- PR-AUC: Single 明显更优

### Fig 5: Multi-Task Per-Task Loss Evolution
![](./img/mvp13_task_loss_evolution.png)

**观察**:
- watch_time 损失最高，主导梯度
- has_gift 损失下降最慢，可能被其他任务挤压
- 任务间可能存在负迁移

---

# 5. 💡 洞见

## 5.1 宏观
- **密集信号未能有效扶起稀疏信号**：watch/comment 任务的信息未有效迁移到打赏预测
- **任务异质性高**：观看时长是回归任务（连续值），打赏是极稀疏二分类，目标函数差异大

## 5.2 模型层
- **共享表征被非目标任务主导**：watch_time 样本量最大、损失值最高，可能挤占 has_gift 的表征空间
- **早停过早**：Multi-Task 在 epoch 21 早停（best=11），说明 has_gift 任务未能充分收敛

## 5.3 细节
- **任务权重需重新调优**：当前固定权重 {0.3, 0.2, 0.3, 0.2} 可能不是最优
- **GradNorm 等动态权重方法可能有帮助**：自动平衡梯度冲突
- **ROC-AUC vs PR-AUC 矛盾**：Multi-Task ROC-AUC 更好说明整体分类能力强，但 PR-AUC 差说明对正例的精准识别不足

---

# 6. 📝 结论

## 6.1 核心发现
> **多任务学习在当前极稀疏打赏场景下无显著收益，密集信号（watch/comment）未能有效迁移**

- ❌ DG5: 密集信号迁移效果不显著，Δ PR-AUC = -1.76pp < 3pp 阈值
- ❌ 假设"共享表征可扶起稀疏信号"在本场景不成立

## 6.2 关键结论

| # | 结论 | 证据 |
|---|------|------|
| 1 | **多任务未提升 PR-AUC** | 0.165 vs 0.182, Δ = -1.76pp |
| 2 | **Top-1% 金主识别能力下降** | 12.0% vs 15.3%, Δ = -3.3pp |
| 3 | **多任务在宽松阈值(Top-5%)略有优势** | 16.4% vs 13.3%, Δ = +3.1pp |
| 4 | **校准(ECE)和 Spearman 多任务略优** | ECE: 0.107 vs 0.112 |

## 6.3 设计启示

| 原则 | 建议 |
|------|------|
| P9: 极稀疏场景慎用多任务 | 若正例 <2%，多任务可能被密集任务主导 |
| P10: 任务异质性大时需 GradNorm | 自动权重平衡可缓解负迁移 |

| ⚠️ 陷阱 | 原因 |
|---------|------|
| 固定任务权重 | 回归任务(watch)梯度可能压制分类任务(gift) |
| 早停按 PR-AUC | 若其他任务未收敛，可能影响共享表征质量 |

## 6.4 关键数字

| 指标 | 值 | 条件 |
|------|-----|------|
| Single-Task PR-AUC | **0.182** | has_gift + gift_amount |
| Multi-Task PR-AUC | 0.165 | 4 tasks |
| Δ PR-AUC | **-1.76pp** | < 3pp 阈值 |
| Single Top-1% | **15.3%** | 金主识别 |
| Multi Top-1% | 12.0% | 金主识别 |
| Single 训练时间 | 2389s | 50 epochs |
| Multi 训练时间 | 1113s | 21 epochs (early stop) |

## 6.5 下一步

| 方向 | 任务 | 优先级 |
|------|------|--------|
| 分配层验证 | MVP-2.1: 凹收益分配 vs 贪心 | 🔴 |
| Simulator 构建 | MVP-0.3: 为分配层提供可控环境 | 🟡 |
| ~~GradNorm 动态权重~~ | ~~多任务优化~~ | 🟢 (低优先级，DG5 已关闭) |

---

# 7. 📎 附录

## 7.1 数值结果

| 模型 | PR-AUC | ROC-AUC | ECE | Top-1% | Top-5% | Top-10% | Spearman |
|------|--------|---------|-----|--------|--------|---------|----------|
| Single-Task | **0.182** | 0.900 | 0.112 | **15.3%** | 13.3% | 16.3% | 0.188 |
| Multi-Task | 0.165 | **0.905** | **0.107** | 12.0% | **16.4%** | **17.7%** | **0.199** |
| Δ | -1.76pp | +0.5pp | -0.5pp | -3.3pp | +3.1pp | +1.4pp | +1.1pp |

## 7.2 执行记录

| 项 | 值 |
|----|-----|
| 脚本 | `scripts/train_multitask.py` |
| 日志 | `logs/mvp13_multitask_20260108.log` |
| 结果 | `../results/multitask_results_20260108.json` |
| 图表 | `../img/mvp13_*.png` |

```bash
# 训练
source init.sh
PYTHONUNBUFFERED=1 python scripts/train_multitask.py > logs/mvp13_multitask_20260108.log 2>&1

# 查看结果
cat gift_allocation/results/multitask_results_20260108.json
```

## 7.3 调试

| 问题 | 解决 |
|------|------|
| nohup 日志无输出 | 添加 PYTHONUNBUFFERED=1 |
| Multi-Task 早停过早 | 正常行为，反映任务冲突 |

---

> **实验完成时间**: 2026-01-08 17:04
