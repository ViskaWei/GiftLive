# Skill: research-coding-prompt

## Description
从 MVP 规格生成可执行的 Coding Prompt。触发词：p, P, prompt, 生成prompt

## Arguments
- `[MVP描述/实验描述]` - 要生成 Prompt 的实验内容

## Workflow

### Step 1: 读取模板
读取 `_backend/template/prompt_coding.md`

### Step 2: 确定实验信息
从用户输入或相关 exp.md 中提取：
- 实验 ID 与元数据
- 数据配置
- 模型配置
- 训练配置

### Step 3: 填写实验规格
使用 YAML 格式填写：
```yaml
experiment:
  id: MVP-X.X
  name: [实验名称]
  topic: [topic]

data:
  source: [数据来源]
  path: data/xxx
  split: train/val/test

model:
  name: [模型名称]
  config: [关键配置]

training:
  epochs: N
  batch_size: N
  lr: 1e-4
  optimizer: Adam
```

### Step 4: 列出要画的图
```yaml
figures:
  - name: fig_1_[description]
    type: [line/bar/heatmap/...]
    x_axis: [X轴含义]
    y_axis: [Y轴含义]
    save_path: [topic]/img/[filename].png
```

### Step 5: 列出参考代码路径
**⚠️ 强制规则：只写路径，不写代码**
```yaml
reference_code:
  - path: scripts/train.py
    purpose: 训练流程参考
  - path: scripts/eval.py
    purpose: 评估流程参考
  - path: src/models/xxx.py
    purpose: 模型定义参考
```

### Step 6: 指定交付物
```yaml
deliverables:
  report:
    path: [topic]/exp/exp_[name]_[YYYYMMDD].md
    template: _backend/template/exp.md
  figures:
    dir: [topic]/img/
  sync:
    - [topic]_roadmap.md §2.1, §4.3
    - [topic]_hub.md §1 (如有重要发现)
```

### Step 7: 保存 Prompt
保存到: `[topic]/prompts/coding_prompt_[name]_YYYYMMDD.md`

### Step 8: Git Commit
```bash
git add -A
git commit -m "feat: 生成 Coding Prompt [mvp_name]"
```

## Output Format
```
📝 生成 Coding Prompt...

📖 Step 1: 读取模板
   ✅ _backend/template/prompt_coding.md

📋 Step 2-4: 填写实验规格
   ID: MVP-X.X
   Topic: [topic]
   数据: [data_config]
   模型: [model_config]
   图表: [N] 张

📁 Step 5: 参考代码路径
   - scripts/train.py
   - scripts/eval.py
   - src/models/xxx.py

📦 Step 6: 交付物
   - exp.md: [topic]/exp/exp_[name]_[YYYYMMDD].md
   - 图表: [topic]/img/

💾 Step 7: 保存
   ✅ [topic]/prompts/coding_prompt_[name]_YYYYMMDD.md

📦 Step 8: Git Commit
   ✅ 完成
```

## Critical Rules
- ❌ **绝对禁止**：在 Coding Prompt 中写任何代码块
- ✅ **必须**：只提供参考代码路径
- ✅ **必须**：确保参考代码路径存在
- 💡 **原因**：写代码骨架容易与已有代码不一致

## Template Reference
- `_backend/template/prompt_coding.md`

## Output Location
- 通用：`experiments/[topic]/prompts/coding_prompt_[name]_YYYYMMDD.md`
- gift_allocation：`gift_allocation/prompts/coding_prompt_[name]_YYYYMMDD.md`
