# Coding Prompt: MVP-1.4 Two-Stage Diagnostic Decomposition

## 实验元数据
| 项 | 值 |
|---|---|
| MVP | MVP-1.4 |
| exp_id | EXP-20260108-gift-allocation-11 |
| 关闭 Gate | DG1.1 |
| 优先级 | 🔴 P0 |
| 预计时间 | 10-15 分钟 |

## 1. 实验目标

诊断 Two-Stage 输给 Direct Regression 的主因：
- **H1**: Stage2 数据量不足（34k gift vs 1.87M click）
- **H2**: p×m 乘法在排序上放大误差
- **H3**: Stage2 的 OOD 预测问题
- **H4**: 实验口径/特征可能存在问题

## 2. 实验设计

### 2.1 数据
复用 MVP-1.1-fair 的数据和模型输出：
- 参考代码路径: `scripts/train_fair_comparison.py`
- 已有模型: `gift_allocation/models/fair_direct_reg_20260108.pkl`
- 已有结果: `gift_allocation/results/fair_comparison_20260108.json`

### 2.2 实验列表

| 实验 | 描述 | 指标 |
|------|------|------|
| Exp1 | Stage1-only 排序 | Top-1%, NDCG@100, Spearman |
| Exp2 | Stage2 gift子集评估 | Spearman_gift, NDCG_gift |
| Exp3 | Oracle p 分解 | Top-1% 上界 |
| Exp4 | Oracle m 分解 | Top-1% 上界 |

### 2.3 实验详细说明

**Exp1: Stage1-only 排序能力**
```
score = p(x)  # 只用分类概率，不乘 m(x)
计算: Top-1%/5%/10% Capture, NDCG@100, Spearman
对比: Direct Reg 的 54.5%
如果接近 → 问题在 Stage2/乘法
如果远差 → Stage1 也有问题
```

**Exp2: Stage2 在 gift 子集的能力**
```
筛选: test 中 Y > 0 的样本 (~25k)
评估: m(x) 对 log(1+Y) 的 Spearman
对比: Direct Reg 在同样子集的 Spearman
如果 Stage2 更好 → 解释 NDCG@100 优势
如果 Stage2 更差 → Stage2 训练有问题
```

**Exp3: Oracle p 分解**
```
用真实 1(Y>0) 替换 p(x)
score = oracle_p × m(x)
计算: Top-1% Capture
这是 "完美分类 + 实际回归" 的上界
```

**Exp4: Oracle m 分解**
```
用真实 log(1+Y) 替换 m(x)（仅 gift 样本，非 gift 保持原 m(x)）
score = p(x) × oracle_m
计算: Top-1% Capture
这是 "实际分类 + 完美回归" 的上界
```

## 3. 参考代码路径

请阅读以下已有代码，复用数据加载和特征工程逻辑：

| 文件 | 用途 |
|------|------|
| `scripts/train_fair_comparison.py` | 数据加载、特征构造、模型训练 |
| `scripts/train_two_stage.py` | Two-Stage 模型结构 |
| `scripts/diagnose_two_stage.py` | 已有诊断脚本（如存在，可复用） |

## 4. 输出要求

### 4.1 结果 JSON
保存到: `gift_allocation/results/two_stage_diagnosis_20260108.json`

```json
{
  "experiment_id": "EXP-20260108-gift-allocation-11",
  "mvp": "MVP-1.4",
  "timestamp": "...",
  "exp1_stage1_only": {
    "top_1pct_capture": ...,
    "top_5pct_capture": ...,
    "ndcg_100": ...,
    "spearman": ...
  },
  "exp2_stage2_gift_subset": {
    "n_gift_samples": ...,
    "spearman_stage2": ...,
    "spearman_direct": ...,
    "stage2_better": true/false
  },
  "exp3_oracle_p": {
    "top_1pct_capture": ...,
    "description": "Oracle p + actual m"
  },
  "exp4_oracle_m": {
    "top_1pct_capture": ...,
    "description": "Actual p + oracle m"
  },
  "reference": {
    "direct_reg_top_1pct": 0.545,
    "two_stage_top_1pct": 0.357
  },
  "diagnosis": {
    "primary_cause": "stage2_data_insufficient / multiplication_noise / ood / stage1_issue",
    "evidence": "...",
    "recommendation": "..."
  }
}
```

### 4.2 图表
保存到: `gift_allocation/img/`

1. `two_stage_diagnosis_stagewise.png` - Stage-wise 性能对比柱状图
2. `two_stage_diagnosis_oracle.png` - Oracle 分解上界对比
3. `two_stage_diagnosis_gift_subset.png` - Gift 子集 Spearman 对比

### 4.3 实验报告
更新: `gift_allocation/exp/exp_two_stage_diagnosis_20260108.md`
- 填写 §4 图表
- 填写 §5 洞见
- 填写 §6 结论

## 5. 决策规则

| 分解结果 | 结论 | 下一步 |
|----------|------|--------|
| Stage1-only ≈ Direct (±5pp) | Stage2/乘法是主因 | → MVP-1.5 |
| Oracle m >> 实际 Two-Stage | Stage2 数据不足 | → Stage2 正则化 |
| Oracle p >> Oracle m | Stage1 更关键 | → 优化分类器 |
| Stage2 gift子集 > Direct | Two-Stage 适合精排 | → 召回-精排分工 |

## 6. 运行命令

```bash
source init.sh
nohup python scripts/diagnose_two_stage.py > logs/two_stage_diagnosis_20260108.log 2>&1 &
echo $! > logs/two_stage_diagnosis_20260108.pid

# 查看日志
tail -f logs/two_stage_diagnosis_20260108.log
```

## 7. 同步更新

实验完成后，根据结论更新：
- `gift_allocation/gift_allocation_hub.md` § H1-H4 状态
- `gift_allocation/gift_allocation_roadmap.md` § MVP-1.4 状态
