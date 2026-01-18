# Skill: research-grow-topic

## Description
子节点生长命令。当一个子节点/子topic需要更深入理解和实验时，单独长出一个节点，生成配套文件结构，并移动相关文件。触发词：grow, 生长, 新建节点

## Arguments
- `[new_topic]` - 新 topic 名称
- `[parent_topic]` - 父 topic 名称
- `--keyword [关键词]` - 用于匹配文件的关键词（可选，默认使用 new_topic）
- `--insights [内容]` - Hub 文件中的 insights 内容（可选）
- `--dry-run` - 预览模式，不实际执行

## Workflow

### Step 1: 解析参数
提取：
- 新 topic 名称
- 父 topic 名称
- 关键词（默认 = new_topic）

### Step 2: 查找相关文件
在父 topic 下查找包含关键词的：
- 实验文件 (`exp/*.md`)
- Prompt 文件 (`prompts/*.md`)

### Step 3: 创建目录结构
创建新 topic 目录：
```
[new_topic]/
├── exp/
├── prompts/
├── img/
├── results/
├── models/
└── sessions/
```

### Step 4: 生成核心文件
基于模板创建：
- `[new_topic]_hub.md` (from `_backend/template/hub.md`)
- `[new_topic]_roadmap.md` (from `_backend/template/roadmap.md`)

### Step 5: 移动文件
将相关实验和 prompt 文件移动到新 topic 目录

### Step 6: 更新链接
更新所有受影响的超链接：
- 父 topic 的 hub、roadmap
- 被移动文件中的链接

### Step 7: Git Commit
```bash
git add -A
git commit -m "feat: grow [new_topic] topic and reorganize files"
```

## Output Format
```
🌱 Grow Topic: [new_topic] (from [parent_topic])

📋 Step 1: 解析参数
   New Topic: [new_topic]
   Parent: [parent_topic]
   Keyword: [keyword]

🔍 Step 2: 查找相关文件
   找到 [N] 个实验文件
     - exp_xxx.md
     - exp_yyy.md
   找到 [N] 个 prompt 文件
     - coding_prompt_xxx.md

📁 Step 3: 创建目录结构
   ✅ 已创建: [new_topic]/
   ✅ 子目录: exp/, prompts/, img/, results/, models/

📝 Step 4: 创建核心文件
   ✅ [new_topic]_hub.md
   ✅ [new_topic]_roadmap.md

📦 Step 5: 移动文件
   ✅ exp_xxx.md → [new_topic]/exp/
   ✅ exp_yyy.md → [new_topic]/exp/
   ✅ coding_prompt_xxx.md → [new_topic]/prompts/

🔗 Step 6: 更新链接
   ✅ [parent]_hub.md
   ✅ [parent]_roadmap.md
   ✅ [new_topic]/exp/exp_xxx.md

📦 Step 7: Git Commit
   ✅ 完成

✅ 新 topic '[new_topic]' 已创建！
```

## Notes
- 新 topic 可以是顶层目录（如 `gift_allocation`）或 `experiments/` 下的子目录
- 关键词默认为新 topic 名称，可用于匹配实验文件名或内容
- 支持 `--dry-run` 模式预览操作
- 移动文件后会自动更新所有相关链接

## Script Reference
- `_backend/scripts/grow_topic.py`

## Template Reference
- hub: `_backend/template/hub.md`
- roadmap: `_backend/template/roadmap.md`
