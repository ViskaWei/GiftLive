# Skill: research-update

## Description
更新实验文档。触发词：u, U, update, 更新

## Variants
- `u [experiment_id]` - 完整更新指定实验（补全+同步hub/roadmap+git push）
- `u hub [topic]` - 根据 prompt 模板重写/优化 Hub
- `u [关键词/描述]` - 智能匹配并追加内容

## Arguments
- `[experiment_id]` - 实验 ID 或文件名，如 `EXP-20250108-model-01` 或 `exp_baseline_20250108`
- `[topic]` - topic 名称，用于 hub 更新模式
- `[关键词/描述]` - 要追加的内容描述

---

## Mode 1: `u [experiment_id]` — 完整实验更新流程

### Step 1: 定位实验报告
根据 experiment_id 查找：
- 在 `experiments/[topic]/exp/` 或 `gift_allocation/exp/` 中搜索
- 支持模糊匹配

### Step 2: 审查实验报告完整性
检查以下章节：
| 章节 | 检查项 |
|------|--------|
| ⚡ 核心结论速览 | 一句话总结、假设验证、关键数字 |
| §1 目标 | 问题、验证假设 |
| §3 实验设计 | 数据、模型、训练配置 |
| §4 图表 | 图表是否存在、描述是否完整 |
| §5 洞见 | 宏观/模型层/细节 |
| §6 结论 | 核心发现、设计启示、关键数字 |
| §7.2 执行记录 | 脚本路径、配置、输出 |

### Step 3: 补全遗漏内容
- 如果 §7.2 缺少代码引用 → 搜索相关脚本并补充
- 如果缺少数值结果 → 从 results/ 或日志中提取
- 如果图表描述不完整 → 补充观察点

### Step 4: 提炼假设与洞见 → 同步到 hub.md
**hub.md 同步内容映射**：
| exp.md 来源 | hub.md 目标 |
|-------------|-------------|
| 新共识 | §0 共识表 (K1-Kn) |
| 假设验证结果 | §1 假设树状态更新 |
| 关键数字 | §0 权威数字 |
| 设计启示 | §6.1 已确认原则 |
| 失败结论 | §6.4 已关闭方向 |

### Step 5: 同步数值结果 → roadmap.md
**roadmap.md 同步内容映射**：
| exp.md 来源 | roadmap.md 目标 |
|-------------|-----------------|
| Header 状态 | §2.1 实验总览（状态列） |
| 一句话总结 | §4.3 结论快照 |
| 关键数字 | §6.1 数值汇总表 |

### Step 6: Git Commit + Push
```bash
git add -A
git commit -m "docs: 更新实验 [exp_id] 并同步 hub/roadmap"
git push
```

### Output Format (Mode 1)
```
📝 完整更新实验报告...

📖 Step 1: 定位实验
   ✅ 找到: [path_to_exp.md]

📋 Step 2: 审查报告完整性
| 章节 | 状态 |
|------|------|
| ⚡ 核心结论速览 | ✅ |
| §6.2 执行记录 | ❌ 缺少代码引用 |

🔧 Step 3: 补全遗漏内容
   ✅ 已补充到 §6.2 执行记录

📤 Step 4: 同步到 hub.md
   ✅ §1 假设树: H1.1 状态 → ✅
   ✅ §6.1 设计原则: 添加 P[X]

📤 Step 5: 同步到 roadmap.md
   ✅ §2.1 实验总览: MVP-1.1 状态 → ✅
   ✅ §4.3 结论快照: 添加一句话总结

📦 Step 6: Git Commit + Push
   ✅ 完成

✅ 实验更新完成！
```

---

## Mode 2: `u hub [topic]` — Hub 重写/优化

### Workflow
1. 读取 `[topic]_hub.md`
2. 根据 `_backend/template/hub.md` 模板检查结构
3. 优化/重写内容，确保符合模板规范
4. 保存并 Git Push

---

## Mode 3: `u [关键词/描述]` — 智能追加内容

### Step 1: 解析内容类型
根据关键词判断内容类型：

| 内容类型 | 关键词示例 | 目标位置 |
|---------|-----------|----------|
| 新发现/洞见 | "发现", "观察到", "意外" | hub.md §4 洞见汇合 |
| 设计建议 | "建议", "应该", "原则" | hub.md §6 设计原则 |
| 数值结果 | "结果", "指标", "数值" | roadmap.md §6.1 数值结果表 |
| 假设验证 | "验证", "假设", "确认" | hub.md §1 假设树 |
| 战略方向 | "战略", "路线", "方向" | hub.md §3 战略推荐 |

### Step 2: 定位目标文件
- 如果用户指定了 topic → 使用该 topic 的 hub/roadmap
- 否则 → 根据内容智能匹配最相关的 topic

### Step 3: 追加内容
在目标章节追加新内容，保持格式一致

### Step 4: Git Push
```bash
git add -A
git commit -m "docs: 更新 [topic] - [内容摘要]"
git push
```

### Output Format (Mode 3)
```
📝 更新文档...

🔍 Step 1: 解析内容
   类型: [洞见/设计建议/数值结果/...]
   内容: [用户输入摘要]

📁 Step 2: 匹配目标
   Topic: [topic]
   文件: [hub.md/roadmap.md]
   章节: §[X] [章节名]

✏️ Step 3: 追加内容
   ✅ 已添加到 [path] §[X]

📦 Step 4: Git Push
   ✅ 完成
```

---

## Important Notes
- 更新后始终执行 Git Push
- 保持文档格式一致性
- 同步更新相关文件（hub ↔ roadmap ↔ exp）

## Template Reference
- exp.md: `_backend/template/exp.md`
- hub.md: `_backend/template/hub.md`
- roadmap.md: `_backend/template/roadmap.md`
