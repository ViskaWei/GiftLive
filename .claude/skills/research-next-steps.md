# Skill: research-next-steps

## Description
管理下一步计划和待办清单。触发词：next, 下一步, 计划

## Variants
- `next` - 查看当前计划
- `next add P0/P1 [描述]` - 添加任务
- `next done [序号/描述]` - 完成任务
- `next plan` - AI 智能推荐下一步

## Arguments
- `add P0 [描述]` - 添加 P0 优先级任务
- `add P1 [描述]` - 添加 P1 优先级任务
- `done [N]` - 标记第 N 个任务完成
- `done [描述]` - 模糊匹配并标记完成
- `plan` - AI 分析并推荐下一步

## Workflow

### Mode: `next` — 查看当前计划
1. 读取 `status/next_steps.md`
2. 显示所有 P0/P1 任务

### Mode: `next add P0/P1 [描述]` — 添加任务
1. 解析优先级和描述
2. 追加到 `status/next_steps.md`
3. Git Commit

### Mode: `next done [N/描述]` — 完成任务
1. 定位任务（序号或模糊匹配）
2. 标记为完成（添加 ✅ 前缀或移动到已完成区）
3. Git Commit

### Mode: `next plan` — AI 智能推荐
1. 扫描最近实验报告
2. 分析 hub.md 中的 Decision Gaps (§5)
3. 检查 roadmap.md 中的待验证 Gate
4. 生成推荐的下一步任务
5. 询问用户是否添加

## Output Format

### 查看计划
```
📋 当前计划

🔴 P0 (必须完成):
1. [任务1]
2. [任务2]

🟡 P1 (应该做):
1. [任务3]
2. [任务4]

✅ 已完成 (最近):
- [任务5] (完成于 YYYY-MM-DD)
```

### 添加任务
```
➕ 添加任务

🔴 P0: [描述]
✅ 已添加到 status/next_steps.md
```

### 完成任务
```
✅ 完成任务

任务: [描述]
状态: 🔴 P0 → ✅ 完成
```

### AI 推荐
```
🧠 智能推荐

📊 分析来源:
- 最近实验: exp_xxx.md, exp_yyy.md
- Decision Gaps: hub.md §5
- 待验证 Gate: roadmap.md §1.2

💡 建议的下一步:
🔴 P0: [推荐任务1] - [理由]
🟡 P1: [推荐任务2] - [理由]

是否添加到计划？(y/n/选择序号)
```

## File Location
- `status/next_steps.md`

## File Format (next_steps.md)
```markdown
# 📌 下一步计划

## 🔴 P0 (必须完成)
- [ ] [任务描述]
- [ ] [任务描述]

## 🟡 P1 (应该做)
- [ ] [任务描述]

## ✅ 已完成
- [x] [任务描述] (YYYY-MM-DD)
```
