# 📌 Next Steps

> **最后更新**: 2026-01-08

---

## 🔴 P0 - 必须完成

| # | 任务 | 状态 | 截止日期 | 备注 |
|---|------|------|---------|------|
| 1 | **MVP-1.1-fair 两段式公平对比** | 🔴 就绪 | - | click全量上对比Two-Stage vs Direct Reg，关闭DG1 |
| 2 | 编写 `train_fair_comparison.py` 脚本 | ⏳ | - | 参考已有 train_two_stage.py |

---

## 🟡 P1 - 应该完成

| # | 任务 | 状态 | 截止日期 | 备注 |
|---|------|------|---------|------|
| 1 | **MVP-1.2 延迟反馈建模** | 🔴 就绪 | - | Chapelle/生存分析，关闭DG2 |
| 2 | MVP-0.3 Simulator V1构建 | ⏳ | - | 凹收益分配验证的基石 |
| 3 | 编写 `train_delay_modeling.py` 脚本 | ⏳ | - | 延迟分布估计+软标签 |

---

## 🟢 P2 - 可选完成

| # | 任务 | 状态 | 截止日期 | 备注 |
|---|------|------|---------|------|
| 1 | MVP-1.3 多任务学习 | ⏳ | - | 密集信号扶起稀疏 |
| 2 | MVP-2.1 凹收益分配层 | ⏳ | - | 依赖Simulator |
| 3 | 阅读Delayed Conversion论文(Chapelle 2014) | ⏳ | - | 延迟反馈建模参考 |
| 4 | 阅读Online Matching with Concave Returns | ⏳ | - | 分配层理论基础 |

---

## ✅ 已完成

| # | 任务 | 完成日期 | 产出 |
|---|------|---------|------|
| 1 | 项目初始化 | 2026-01-08 | 基础目录结构 |
| 2 | Gift Allocation项目立项 | 2026-01-08 | Hub + Roadmap |
| 3 | MVP-0.1 KuaiLive数据探索 | 2026-01-08 | Gini=0.94, Top1%=60% |
| 4 | MVP-0.2 Baseline(直接回归) | 2026-01-08 | Top-1%=56.2%, Spearman=0.891 |
| 5 | MVP-1.1 两段式建模 | 2026-01-08 | 揭示与Baseline不可直接对比 |
| 6 | 下载KuaiLive数据集 | 2026-01-08 | data/KuaiLive/ |
| 7 | MVP-1.1-fair 立项 | 2026-01-08 | exp_fair_comparison_20260108.md |
| 8 | MVP-1.2 立项 | 2026-01-08 | exp_delay_modeling_20260108.md |

---

## 📊 优先级决策树

```
MVP-1.1-fair (公平对比)
    ├── If Top-1% 提升 ≥5% → 确认两段式，进入 MVP-1.2
    └── Else → 保留Baseline，考虑端到端或简化

MVP-1.2 (延迟反馈)
    ├── If ECE 改善 ≥0.02 → 纳入主模型
    └── Else → 简单窗口截断

MVP-0.3 (Simulator)
    └── 完成后 → MVP-2.1 凹收益分配验证
```

---

## 📝 使用说明

### 添加任务
```
next add P0 [任务描述]
next add P1 [任务描述]
```

### 完成任务
```
next done 1       # 完成第 1 个任务
next done [描述]  # 按描述匹配完成
```

### 智能推荐
```
next plan         # AI 分析并推荐下一步
```
