# Skill: research-new-experiment

## Description
创建新实验计划。触发词：n, N, new, 新建, 立项

## Arguments
- `[实验描述]` - 实验主题、研究问题、验证假设、设计思路

## Workflow

### Step 1: 解析用户输入
提取以下信息：
- 实验主题 (topic)
- 研究问题 (question)
- 验证假设 (hypothesis)
- 实验设计思路 (design)

### Step 2: 定位目标目录
根据 topic 确定目录：
- `gift_allocation` → `~/GiftLive/gift_allocation/`
- 其他 topic → `~/GiftLive/experiments/[topic]/`

### Step 3: 创建/更新 hub.md（如果涉及新问题/假设）
- 新研究问题 → 添加到 hub.md §1 核心问题树
- 新假设 → 添加到 hub.md §1 假设树

### Step 4: 创建 exp.md（只填写实验前部分）
使用模板 `_backend/template/exp.md`，填写：
- Header: Name, ID (`EXP-[YYYYMMDD]-[topic]-[##]`), Topic, MVP, Author, Date, Status
- §1 目标: 问题、验证假设、预期结果
- §3 实验设计: 数据、模型、训练配置（如已知）

文件命名：`exp_[name]_[YYYYMMDD].md`
保存位置：`[topic_dir]/exp/`

### Step 5: 更新 roadmap.md
- §2.1 实验总览：添加新条目
- §3 MVP 详细设计：添加规格（如需要）

### Step 6: Git Commit
```bash
git add -A
git commit -m "feat: 创建新实验计划 [exp_name]"
```

## Output Format
```
📝 创建实验计划...

📋 Step 1: 解析输入
   Topic: [topic]
   问题: [question]
   假设: [hypothesis]

📁 Step 2: 目标目录
   [target_dir]

🧠 Step 3: 更新 hub.md
   ✅ 已添加到 §1 核心假设树: Q[X.X]

📗 Step 4: 创建 exp.md
   ✅ 已创建: [path_to_exp.md]
   - Header ✅
   - §1 目标 ✅
   - §3 实验设计 [✅/⏳ 待补充]

🗺️ Step 5: 更新 roadmap.md
   ✅ 已添加到 §2.1 实验总览: MVP-X.X

📦 Step 6: Git Commit
   ✅ 完成

✅ 实验计划创建完成！
```

## Important Notes
- ❌ 不生成任何代码
- ❌ 不执行实验
- ✅ 只创建/更新文档文件（.md）
- ✅ 如果 topic 目录不存在，先创建目录结构

## Template Reference
- exp.md: `_backend/template/exp.md`
- hub.md: `_backend/template/hub.md`
- roadmap.md: `_backend/template/roadmap.md`

## File Naming Convention
- exp: `exp_[name]_[YYYYMMDD].md`
- ID: `EXP-[YYYYMMDD]-[topic]-[##]`
