# Skill: research-session

## Description
将 GPT/Claude 会话归档为结构化实验计划。触发词：session, 会话, gpt

## Variants
- `session new [topic]` - 创建新会话归档
- `session list` - 列出最近会话

## Arguments
- `new [topic]` - 创建新会话，指定 topic
- `list` - 列出最近会话

## Workflow

### Mode: `session new [topic]` — 创建新会话

#### Step 1: 创建会话文件
使用模板 `_backend/template/session.md`
文件路径: `[topic_dir]/sessions/session_[YYYYMMDD].md`

#### Step 2: 填写会话信息
- Header: 会话 ID、日期、参与者
- §1 起点: 问题 & 动机
- §2 GPT 对话摘录: 关键对话内容

#### Step 3: 结构化 MVP 候选
从对话中提取实验候选：
```markdown
## MVP 候选列表

| # | 名称 | 目的 | 优先级 |
|---|------|------|--------|
| 1 | [MVP名称] | [目的] | P0/P1 |
| 2 | [MVP名称] | [目的] | P0/P1 |
```

#### Step 4: 选择要执行的实验
标记选中的实验，分配 experiment_id

#### Step 5: Git Commit
```bash
git add -A
git commit -m "docs: 归档 GPT 会话 [session_id]"
```

### Mode: `session list` — 列出会话
扫描 `[topic_dir]/sessions/` 目录，显示最近会话

## Output Format

### 创建新会话
```
💬 创建 GPT 会话归档...

📁 Step 1: 创建会话文件
   ✅ [topic]/sessions/session_[YYYYMMDD].md

📝 Step 2: 填写会话信息
   会话 ID: SESSION-[YYYYMMDD]-[topic]-01
   日期: YYYY-MM-DD
   参与者: [User], GPT-4/Claude

📋 Step 3: MVP 候选
   提取了 [N] 个 MVP 候选

🎯 Step 4: 选择执行
   请在会话文件中标记要执行的实验

📦 Step 5: Git Commit
   ✅ 完成
```

### 列出会话
```
💬 最近会话

| 日期 | 会话 ID | Topic | MVP 数 |
|------|---------|-------|--------|
| YYYY-MM-DD | SESSION-xxx | [topic] | N |
| YYYY-MM-DD | SESSION-xxx | [topic] | N |
```

## Template Reference
- `_backend/template/session.md`

## File Location
- `experiments/[topic]/sessions/session_[YYYYMMDD].md`
- `gift_allocation/sessions/session_[YYYYMMDD].md`
