# 🤖 Coding Prompt: 召回-精排分工

> **Experiment ID:** `EXP-20260109-gift-allocation-51`  
> **MVP:** MVP-5.1  
> **Date:** 2026-01-09  
> **Author:** Viska Wei

---

## 1. 📌 实验目标

**一句话**：验证 Direct 召回 + Two-Stage Stage2 精排的分工架构，能否同时保住 Top-1% 和提升 NDCG。

**验证决策空白**：DG6 - 召回-精排分工能否同时保住 Top-1% 和提升 NDCG？

**Gate**：Gate-5A

**动机**：
- Direct Regression 全量排序强（Top-1%=54.5%），但 NDCG@100 较弱（21.7%）
- Two-Stage 在 NDCG@100 上更优（+14.2pp），且 Stage2 在 gift 子集 Spearman=0.89
- 推测：Direct 负责召回粗排，Stage2 负责头部精排，可取长补短

**预期结果**：
- 若 Top-1% ≥ 56% & NDCG@100 ↑ → Gate-5A PASS → 采用分工架构
- 若 Top-1% 下降 → Gate-5A FAIL → 保留 Direct-only

---

## 2. 🧪 实验设定

### 2.1 数据

```yaml
data:
  source: "KuaiLive"
  path: "data/KuaiLive/"
  train_samples: 1,872,509
  val_samples: 1,701,600
  test_samples: 1,335,406
  positive_rate: 1.82%
  split: "按天，最后7天test"
```

### 2.2 模型

```yaml
recall_model:
  name: "Direct Regression (LightGBM)"
  task: "regression"
  objective: "mae"
  target: "log(1+Y)"
  note: "复用 MVP-1.1-fair 已训练模型"

rerank_model:
  name: "Two-Stage Stage2 (LightGBM)"
  task: "regression"
  objective: "mae"
  target: "log(1+Y) | Y>0"
  note: "复用 MVP-1.1-fair 已训练模型"
```

### 2.3 召回-精排流程

```yaml
pipeline:
  step1_recall:
    model: "Direct Regression"
    input: "test_samples (全量 1.3M)"
    output: "Top-M candidates per query"
    
  step2_rerank:
    model: "Stage2 回归"
    input: "Top-M candidates from step1"
    output: "精排后的 Top-K 列表"
    
  step3_evaluate:
    metrics:
      - Top-1% Capture (vs 全量 ground truth)
      - Top-5% Capture
      - NDCG@100
      - Spearman (全量)
```

### 2.4 扫描参数

```yaml
sweep:
  Top_M: [50, 100, 200, 500, 1000]  # 召回候选数
  
baseline:
  Direct_only: "Direct Regression 全量排序"
  Two_Stage_only: "p(x) × m(x) 全量排序"

ablation:
  - "Direct_only (Top-1%=54.5%, 参照基线)"
  - "Recall-Rerank @ Top-M"
  - "分析：Top-M 对 recall coverage 的影响"
```

---

## 3. 📊 要画的图

| 图号 | 图表类型 | X轴 | Y轴 | 保存路径 |
|------|---------|-----|-----|---------|
| Fig1 | line plot | Top-M | Top-1% Capture | `../img/mvp51_recall_vs_topk.png` |
| Fig2 | grouped bar | Method | Top-1% / Top-5% / NDCG@100 | `../img/mvp51_method_comparison.png` |
| Fig3 | line plot (dual y) | Top-M | Top-1% Capture / Recall Coverage | `../img/mvp51_tradeoff.png` |
| Fig4 | scatter | Direct Score Rank | Stage2 Score Rank | `../img/mvp51_score_correlation.png` |

**图表要求**：
- 所有文字必须英文
- 包含 legend、title、axis labels
- 分辨率 ≥ 300 dpi
- **figsize 规则**：
  - 单张图：`figsize=(6, 5)`
  - 多张图（subplot）：按 6:5 比例扩增

---

## 4. 📁 参考代码

> ⚠️ **不要在这里写代码！先读取以下文件，理解后再开发**

| 参考脚本 | 可复用 | 需修改 |
|---------|--------|--------|
| `scripts/train_fair_comparison.py` | 数据加载、特征构造、模型定义 | 添加召回-精排流程 |
| `scripts/train_two_stage.py` | Stage2 模型训练逻辑 | - |
| `scripts/diagnose_two_stage.py` | Stage2 在 gift 子集评估逻辑 | 复用评估函数 |
| `exp/exp_fair_comparison_20260108.md` | 基线数值参考 | - |
| `exp/exp_two_stage_diagnosis_20260108.md` | Stage2 精排优势分析 | - |

**关键复用点**：
1. **Direct 模型**：`train_fair_comparison.py` 中 `train_direct_regression()` 函数
2. **Stage2 模型**：`train_fair_comparison.py` 中 Stage2 训练逻辑
3. **评估函数**：`compute_top_k_capture()`, `compute_ndcg()`, `compute_spearman()`
4. **特征构造**：`create_*_features()` 系列函数

**输出脚本位置**：`scripts/train_recall_rerank.py`

---

## 5. 📝 最终交付物

### 5.1 实验报告

- **路径**: `exp/exp_recall_rerank_20260109.md`
- **模板**: `_backend/template/exp.md`
- **必须包含**:
  - ⚡ 核心结论速览（Gate-5A 是否 PASS + 关键数字）
  - 📊 实验图表（4张图 + 每张图的观察）
  - 💡 关键洞见（最优 Top-M、召回覆盖率分析）
  - 📝 结论（假设验证 + 推荐架构）

### 5.2 图表文件

- **路径**: `../img/`
- **命名**: `mvp51_*.png`

### 5.3 数值结果

- **格式**: JSON
- **路径**: `../results/recall_rerank_20260109.json`
- **内容**:

```yaml
required_fields:
  - Top_M: [50, 100, 200, 500, 1000]
  - Top1_Capture: [...]  # 每个 Top-M 的 Top-1% Capture
  - Top5_Capture: [...]
  - NDCG_100: [...]
  - Recall_Coverage: [...]  # 召回集覆盖的真实 Top-1% 比例
  - baseline_direct: {...}  # Direct-only 基线
  - baseline_two_stage: {...}  # Two-Stage-only 基线
```

---

## 6. 📤 报告抄送

完成后需同步更新：

| 目标文件 | 更新内容 | 章节 |
|---------|---------|------|
| `gift_EVmodel_roadmap.md` | MVP-5.1 状态 + 结论快照 | §2.1 总览, §6.3 结论快照 |
| `gift_EVmodel_hub.md` | DG6 验证结果 + 洞见 | §决策空白, §洞见汇合 |
| `status/kanban.md` | MVP-5.1 状态更新 | Phase 5 |

---

## 7. ⚠️ 注意事项

- [ ] 复用已有模型：优先加载 `gift_allocation/models/` 中的预训练模型
- [ ] 若无预训练模型，先运行 `train_fair_comparison.py` 获取基线模型
- [ ] 代码中添加 `SEED = 42` 固定随机性
- [ ] 图表文字全英文
- [ ] 保存完整日志到 `logs/mvp51_recall_rerank.log`
- [ ] 长时间任务使用 `nohup python scripts/train_recall_rerank.py > logs/mvp51_recall_rerank.log 2>&1 &`

---

## 8. 🔑 关键评估逻辑（伪代码描述，需 Agent 实现）

### 8.1 Recall Coverage 计算

```
定义：Top-M 召回集能覆盖多少比例的真实 Top-1% 用户

步骤：
1. 用 Direct 模型对全量 test 打分
2. 取 Direct 打分 Top-M 作为召回集
3. 计算真实 ground truth Top-1% 样本有多少在召回集中
4. Recall Coverage = 交集数量 / Top-1% 样本数
```

### 8.2 Recall-Rerank 评估

```
步骤：
1. Direct 打分全量 → 取 Top-M 候选
2. Stage2 对 Top-M 候选重新打分
3. 按 Stage2 分数排序得到最终排名
4. 计算 Top-1% Capture（基于全量 ground truth）
```

### 8.3 对比方法

```
| 方法 | 说明 |
|------|------|
| Direct-only | Direct 全量打分后直接取 Top-K |
| Two-Stage-only | p(x)×m(x) 全量打分后取 Top-K |
| Recall-Rerank@M | Direct Top-M → Stage2 精排 → Top-K |
```

---

## 9. 🎯 成功标准

| 指标 | 基线 (Direct-only) | 目标 | Gate 判定 |
|------|-------------------|------|----------|
| Top-1% Capture | 54.5% | ≥ 56% | PASS 条件 1 |
| NDCG@100 | 21.7% | ↑ (高于基线) | PASS 条件 2 |
| Recall Coverage | - | ≥ 80% @最优Top-M | 架构可行性 |

**Gate-5A 判定规则**：
- **PASS**: Top-1% ≥ 56% **且** NDCG@100 > 21.7%
- **FAIL**: 任一条件不满足

---

<!--
📌 Prompt 执行规则（Agent 必读）：

1. ❌ 不要在这个 Prompt 里写代码
2. ✅ 先读取"参考代码"中列出的文件
3. ✅ 理解现有代码逻辑后再修改
4. ✅ 复用已有函数，不要重复造轮子
5. ✅ 按模板输出 exp.md 报告
6. ✅ 确保环境初始化：source init.sh
-->
