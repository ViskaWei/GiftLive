# 🤖 Coding Prompt: 多任务学习

> **Experiment ID:** `EXP-20260108-gift-allocation-06`  
> **MVP:** MVP-1.3  
> **Date:** 2026-01-08  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证多任务学习（用密集信号click/watch/comment扶起稀疏打赏信号）是否能提升打赏预测性能

**验证假设**：DG5 - 密集信号对打赏预测的迁移效果如何？

**预期结果**：
- 若 多任务模型 PR-AUC > 单任务 PR-AUC (Δ≥3%) → 确认多任务学习有效，采用联合训练
- 若 Δ<3% 或无提升 → 多任务收益不显著，保留单任务直接回归

**背景**：
- 打赏率极稀疏 (1.48%，即 98% Y=0)
- 直接回归已有较好性能 (Top-1% Capture=54.5%)
- 密集行为信号 (click, watch_time, comment, like) 可能提供辅助监督

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  files:
    - click.csv      # 主表，含 watch_live_time
    - comment.csv    # 评论行为
    - like.csv       # 点赞行为
    - gift.csv       # 打赏行为
    - user.csv       # 用户画像
    - streamer.csv   # 主播画像
    - room.csv       # 直播间信息
  train_ratio: 0.7   # 按天切分
  val_ratio: 0.15
  test_ratio: 0.15
  features: ~70      # 复用 baseline 特征工程
```

### 2.2 模型

```yaml
model:
  name: "Multi-Task MLP"
  architecture:
    shared_tower:
      hidden_dims: [256, 128, 64]
      activation: "ReLU"
      dropout: 0.2
      batch_norm: true
    task_heads:
      - name: "watch_time"        # Task 1: 观看时长 (回归, MSE)
        head_dims: [32]
        output: 1
        loss: "mse"
        weight: 0.3
      - name: "has_comment"       # Task 2: 是否评论 (二分类, BCE)
        head_dims: [32]
        output: 1
        loss: "bce"
        weight: 0.2
      - name: "has_gift"          # Task 3: 是否打赏 (二分类, Focal)
        head_dims: [32]
        output: 1
        loss: "focal"
        focal_alpha: 0.25
        focal_gamma: 2.0
        weight: 0.3
      - name: "gift_amount"       # Task 4: 打赏金额 (回归, MSE, 条件于has_gift)
        head_dims: [32]
        output: 1
        loss: "mse"
        weight: 0.2
        conditional: true         # 只在 has_gift=1 时计算
```

### 2.3 训练

```yaml
training:
  epochs: 50
  batch_size: 4096
  lr: 1e-3
  optimizer: Adam
  scheduler: ReduceLROnPlateau
  patience: 10
  seed: 42
  device: cuda
  early_stop_metric: "val_has_gift_pr_auc"
```

### 2.4 对照实验

```yaml
experiments:
  - name: "single_task_gift"      # 对照组：单任务打赏预测
    tasks: ["has_gift"]
    description: "只训练打赏二分类"
  
  - name: "multi_task_all"        # 实验组1：全任务
    tasks: ["watch_time", "has_comment", "has_gift", "gift_amount"]
    description: "四任务联合训练"
  
  - name: "multi_task_no_amount"  # 实验组2：去除金额任务
    tasks: ["watch_time", "has_comment", "has_gift"]
    description: "三任务联合，不含金额"
  
  - name: "multi_task_gradient_blend"  # 实验组3：梯度混合
    tasks: ["watch_time", "has_comment", "has_gift", "gift_amount"]
    gradient_blend: true
    description: "使用 GradNorm 动态权重"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | bar | Model (Single vs Multi) | PR-AUC (has_gift) | `../img/mvp13_pr_auc_comparison.png` |
| Fig2 | line (multi-line) | Epoch | Loss (per task) | `../img/mvp13_training_curves.png` |
| Fig3 | bar grouped | Model | Top-1%/5%/10% Capture | `../img/mvp13_topk_capture.png` |
| Fig4 | heatmap | Task | Task (gradient conflict) | `../img/mvp13_gradient_conflict.png` |
| Fig5 | bar | Feature/Embedding dim | Importance (via ablation) | `../img/mvp13_shared_repr_analysis.png` |

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
| `scripts/train_baseline_lgb.py` | `load_data()`, `create_*_features()`, `temporal_split()`, `evaluate_model()` | 改为 PyTorch DataLoader |
| `scripts/train_fair_comparison.py` | 特征工程部分 | 添加多任务标签 |
| `scripts/eda_kuailive.py` | 数据探索逻辑 | - |

**新增代码模块**：
1. `MultiTaskMLP`: 共享塔 + 多头网络
2. `FocalLoss`: 处理极稀疏 has_gift
3. `MultiTaskDataset`: 整合多个标签
4. `GradNorm` (可选): 动态任务权重

---

## 5. 📝 最终交付物

### 5.1 实验报告
- **路径**: `exp/exp_multitask_20260108.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（一句话 + 关键数字）
  - 📊 实验图表（所有图 + 观察）
  - 📝 结论（DG5 验证结果 + 设计启示）

### 5.2 图表文件
- **路径**: `../img/`
- **命名**: `mvp13_*.png`

### 5.3 数值结果
- **格式**: JSON
- **路径**: `../results/multitask_results_20260108.json`
- **内容**: 
  ```json
  {
    "single_task": {"pr_auc": ..., "top_1pct": ..., "top_5pct": ...},
    "multi_task_all": {"pr_auc": ..., "top_1pct": ..., "top_5pct": ...},
    "delta": {"pr_auc": ..., "top_1pct": ...}
  }
  ```

### 5.4 训练脚本
- **路径**: `scripts/train_multitask.py`

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVpred_roadmap.md` | MVP-1.3 状态 → ✅ + 结论快照 | §2.1, §4.3, §6.1 |
| `gift_EVpred_hub.md` | DG5 验证结果 + 洞见 I10 | §5, §4 |

---

## 7. ⚠️ 注意事项

- [ ] 代码中添加 seed=42 固定随机性 (torch, numpy, random)
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/mvp13_multitask_20260108.log`
- [ ] 长时间任务使用 nohup 后台运行
- [ ] 确保 CUDA 可用 (`torch.cuda.is_available()`)
- [ ] 多任务损失加权需实验调优

---

## 8. 📐 评估指标定义

### 8.1 核心指标（对齐 DG5 决策）

| 指标 | 定义 | 决策阈值 |
|------|------|---------|
| **PR-AUC (has_gift)** | 打赏二分类精确率-召回率曲线下面积 | Δ≥3% → 多任务有效 |
| Top-1% Capture | 预测 Top-1% 中捕获多少真正 Top-1% 金主 | Δ>0 → bonus |
| ECE (has_gift) | 期望校准误差 | 越低越好 |

### 8.2 辅助指标

| 指标 | 定义 | 用途 |
|------|------|------|
| Task Loss Ratio | 各任务损失占比 | 检查任务冲突 |
| Gradient Cosine | 任务间梯度余弦相似度 | 分析迁移方向 |

---

## 9. 🔬 技术细节

### 9.1 Focal Loss 实现

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

其中 $\alpha=0.25$, $\gamma=2.0$（应对 1.48% 正例率）

### 9.2 多任务损失

$$\mathcal{L} = \sum_{t} w_t \mathcal{L}_t$$

- 固定权重: $w = [0.3, 0.2, 0.3, 0.2]$
- GradNorm 动态权重（实验组3）

### 9.3 条件金额预测

`gift_amount` 任务只在 `has_gift=1` 样本上计算损失，避免 0 值污染

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 运行前先 source init.sh
7. ✅ 长时间训练用 nohup 后台运行
-->
