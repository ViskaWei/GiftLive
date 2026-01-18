# Skill: research-card

## Description
从多个实验中创建知识卡片。触发词：card, 卡片, kc

## Arguments
- `[关键词/主题]` - 知识卡片主题

## Definition
Card 是**可复用的阶段性知识**，跨多个实验的结构性认知
- ✅ 做：理论依据、可指导决策的结论、关键证据
- ❌ 不做：指导下一步实验（这是 hub 的职责）

## Workflow

### Step 1: 确定 Card 位置
根据主题范围：
- 单主题 → `experiments/[topic]/card/` 或 `gift_allocation/card/`
- 跨主题 → `experiments/card/`

### Step 2: 检索相关实验
根据关键词搜索：
- 扫描 `exp/*.md` 文件
- 匹配标题、结论、洞见章节

### Step 3: 提取关键信息
从每个相关实验中提取：
- 核心结论
- 关键数字
- 设计原则
- 证据链接

### Step 4: 生成知识卡片
使用模板 `_backend/template/card.md`

### Step 5: 保存 + Git Commit
```bash
git add -A
git commit -m "docs: 创建知识卡片 [card_name]"
```

## Output Format
```
📇 创建知识卡片...

📁 Step 1: 确定位置
   范围: [单主题/跨主题]
   路径: [card_dir]

🔍 Step 2: 检索相关实验
   关键词: [keyword]
   找到 [N] 个相关实验:
   - exp_xxx.md
   - exp_yyy.md

📝 Step 3: 提取关键信息
   - 结论: [N] 条
   - 数字: [N] 个
   - 原则: [N] 条

📄 Step 4: 生成卡片
   ✅ [card_dir]/card_[name]_[YYYYMMDD].md

📦 Step 5: Git Commit
   ✅ 完成
```

## Template Reference
- `_backend/template/card.md`

## File Location
- 单主题: `experiments/[topic]/card/card_[name]_[YYYYMMDD].md`
- 跨主题: `experiments/card/card_[name]_[YYYYMMDD].md`
