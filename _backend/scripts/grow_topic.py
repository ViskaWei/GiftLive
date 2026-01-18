#!/usr/bin/env python3
"""
Grow Topic - 子节点生长命令

当一个子节点/子topic（比如 kuailive eda）需要更深入理解和实验时，
单独长出一个节点，生成配套文件结构，并移动相关文件。

用法:
    python _backend/scripts/grow_topic.py <new_topic> <parent_topic> [--dry-run]
    
示例:
    python _backend/scripts/grow_topic.py kuailive gift_allocation
    python _backend/scripts/grow_topic.py kuailive gift_allocation --dry-run
"""

import os
import sys
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "_backend" / "template"

# 专题目录映射（gift_allocation 是顶层目录）
TOPIC_DIR_MAPPING = {
    "gift_allocation": PROJECT_ROOT / "gift_allocation",
    # 可以添加其他专题
}

def find_experiments_by_keyword(parent_topic: str, keyword: str) -> List[Path]:
    """查找包含关键词的实验文件"""
    if parent_topic in TOPIC_DIR_MAPPING:
        exp_dir = TOPIC_DIR_MAPPING[parent_topic] / "exp"
    else:
        exp_dir = PROJECT_ROOT / "experiments" / parent_topic / "exp"
    
    if not exp_dir.exists():
        return []
    
    experiments = []
    for exp_file in exp_dir.glob("exp_*.md"):
        content = exp_file.read_text(encoding="utf-8")
        # 检查文件名或内容中是否包含关键词
        if keyword.lower() in exp_file.stem.lower() or keyword.lower() in content.lower():
            experiments.append(exp_file)
    
    return experiments

def find_prompts_by_keyword(parent_topic: str, keyword: str) -> List[Path]:
    """查找包含关键词的 prompt 文件"""
    if parent_topic in TOPIC_DIR_MAPPING:
        prompts_dir = TOPIC_DIR_MAPPING[parent_topic] / "prompts"
    else:
        prompts_dir = PROJECT_ROOT / "experiments" / parent_topic / "prompts"
    
    if not prompts_dir.exists():
        return []
    
    prompts = []
    for prompt_file in prompts_dir.glob("*.md"):
        content = prompt_file.read_text(encoding="utf-8")
        if keyword.lower() in prompt_file.stem.lower() or keyword.lower() in content.lower():
            prompts.append(prompt_file)
    
    return prompts

def create_directory_structure(new_topic: str, parent_topic: str, dry_run: bool = False) -> Path:
    """创建新 topic 的目录结构"""
    # 确定新 topic 的路径
    if new_topic in TOPIC_DIR_MAPPING:
        new_topic_dir = TOPIC_DIR_MAPPING[new_topic]
    else:
        new_topic_dir = PROJECT_ROOT / "experiments" / new_topic
    
    dirs_to_create = [
        new_topic_dir,
        new_topic_dir / "exp",
        new_topic_dir / "prompts",
        new_topic_dir / "img",
        new_topic_dir / "results",
        new_topic_dir / "models",
    ]
    
    if not dry_run:
        for d in dirs_to_create:
            d.mkdir(parents=True, exist_ok=True)
        print(f"✅ 已创建目录结构: {new_topic_dir}")
    else:
        print(f"🔍 [DRY-RUN] 将创建目录结构: {new_topic_dir}")
    
    return new_topic_dir

def create_hub_file(new_topic: str, new_topic_dir: Path, insights: Optional[str] = None, dry_run: bool = False) -> Path:
    """创建 hub.md 文件"""
    hub_file = new_topic_dir / f"{new_topic}_hub.md"
    template_file = TEMPLATE_DIR / "hub.md"
    
    if not template_file.exists():
        print(f"⚠️  模板文件不存在: {template_file}")
        return hub_file
    
    template_content = template_file.read_text(encoding="utf-8")
    
    # 替换模板变量
    today = datetime.now().strftime("%Y-%m-%d")
    hub_content = template_content.replace("[topic]", new_topic)
    hub_content = hub_content.replace("YYYY-MM-DD", today)
    hub_content = hub_content.replace("EXP-YYYYMMDD-topic-hub", f"EXP-{datetime.now().strftime('%Y%m%d')}-{new_topic}-hub")
    
    # 如果有 insights，添加到洞见汇合部分
    if insights:
        # 在 §4 洞见汇合部分添加内容
        insights_section = f"""
## 4) 洞见汇合（多实验 → 共识）

{insights}

"""
        # 简单插入到洞见汇合部分（如果模板中有占位符）
        if "## 4) 洞见汇合" in hub_content:
            # 在洞见汇合表格后插入
            hub_content = hub_content.replace(
                "## 4) 洞见汇合（多实验 → 共识）",
                f"## 4) 洞见汇合（多实验 → 共识）\n\n{insights}"
            )
    
    if not dry_run:
        hub_file.write_text(hub_content, encoding="utf-8")
        print(f"✅ 已创建: {hub_file}")
    else:
        print(f"🔍 [DRY-RUN] 将创建: {hub_file}")
    
    return hub_file

def create_roadmap_file(new_topic: str, new_topic_dir: Path, dry_run: bool = False) -> Path:
    """创建 roadmap.md 文件"""
    roadmap_file = new_topic_dir / f"{new_topic}_roadmap.md"
    template_file = TEMPLATE_DIR / "roadmap.md"
    
    if not template_file.exists():
        print(f"⚠️  模板文件不存在: {template_file}")
        return roadmap_file
    
    template_content = template_file.read_text(encoding="utf-8")
    
    # 替换模板变量
    today = datetime.now().strftime("%Y-%m-%d")
    roadmap_content = template_content.replace("<TOPIC>", new_topic.capitalize())
    roadmap_content = roadmap_content.replace("<topic>", new_topic)
    roadmap_content = roadmap_content.replace("YYYY-MM-DD", today)
    roadmap_content = roadmap_content.replace("EXP-[YYYYMMDD]-[topic]-roadmap", f"EXP-{datetime.now().strftime('%Y%m%d')}-{new_topic}-roadmap")
    
    # 更新相关文件链接
    roadmap_content = roadmap_content.replace(
        "`[topic]_hub.md`",
        f"`{new_topic}_hub.md`"
    )
    
    if not dry_run:
        roadmap_file.write_text(roadmap_content, encoding="utf-8")
        print(f"✅ 已创建: {roadmap_file}")
    else:
        print(f"🔍 [DRY-RUN] 将创建: {roadmap_file}")
    
    return roadmap_file

def move_files(files: List[Path], target_dir: Path, dry_run: bool = False) -> List[Path]:
    """移动文件到目标目录"""
    moved_files = []
    for file in files:
        target_file = target_dir / file.name
        if not dry_run:
            shutil.move(str(file), str(target_file))
            moved_files.append(target_file)
            print(f"✅ 已移动: {file.name} → {target_file}")
        else:
            print(f"🔍 [DRY-RUN] 将移动: {file.name} → {target_file}")
            moved_files.append(target_file)
    
    return moved_files

def update_links_in_file(file_path: Path, old_topic: str, new_topic: str, dry_run: bool = False):
    """更新文件中的链接"""
    if not file_path.exists():
        return
    
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    
    # 判断文件位置，决定相对路径
    file_dir = file_path.parent
    is_in_new_topic = new_topic.lower() in str(file_dir).lower()
    
    # 更新各种链接模式
    patterns = []
    
    if is_in_new_topic:
        # 在新 topic 目录下的文件，需要指向同级或父级
        patterns = [
            # Hub 链接（指向同级）
            (rf"`{old_topic}/{old_topic}_hub\.md`", f"`../{new_topic}_hub.md`"),
            (rf"`{old_topic}_hub\.md`", f"`{new_topic}_hub.md`"),
            (rf"`experiments/{old_topic}/{old_topic}_hub\.md`", f"`{new_topic}_hub.md`"),
            (rf"`gift_allocation/gift_allocation_hub\.md`", f"`../../gift_allocation/gift_allocation_hub.md`"),
            
            # Roadmap 链接（指向同级）
            (rf"`{old_topic}/{old_topic}_roadmap\.md`", f"`../{new_topic}_roadmap.md`"),
            (rf"`{old_topic}_roadmap\.md`", f"`{new_topic}_roadmap.md`"),
            (rf"`experiments/{old_topic}/{old_topic}_roadmap\.md`", f"`{new_topic}_roadmap.md`"),
            
            # Exp 链接（指向同级 exp 目录）
            (rf"`{old_topic}/exp/", f"`exp/"),
            (rf"`exp/exp_", f"`exp/exp_"),
            
            # 图片路径（指向同级 img 目录）
            (rf"`{old_topic}/img/", f"`../img/"),
            (rf"`\.\./img/", f"`../img/"),  # 保持相对路径
            
            # Results 路径（指向同级 results 目录）
            (rf"`{old_topic}/results/", f"`../results/"),
            (rf"`\.\./results/", f"`../results/"),  # 保持相对路径
        ]
    else:
        # 在父 topic 或其他目录下的文件，需要指向新 topic
        patterns = [
            # Hub 链接
            (rf"`{old_topic}/{old_topic}_hub\.md`", f"`../{new_topic}/{new_topic}_hub.md`"),
            (rf"`{old_topic}_hub\.md`", f"`../{new_topic}/{new_topic}_hub.md`"),
            (rf"`experiments/{old_topic}/{old_topic}_hub\.md`", f"`../{new_topic}/{new_topic}_hub.md`"),
            
            # Roadmap 链接
            (rf"`{old_topic}/{old_topic}_roadmap\.md`", f"`../{new_topic}/{new_topic}_roadmap.md`"),
            (rf"`{old_topic}_roadmap\.md`", f"`../{new_topic}/{new_topic}_roadmap.md`"),
            (rf"`experiments/{old_topic}/{old_topic}_roadmap\.md`", f"`../{new_topic}/{new_topic}_roadmap.md`"),
            
            # Exp 链接（指向新 topic 的 exp 目录）
            (rf"`{old_topic}/exp/exp_", f"`../{new_topic}/exp/exp_"),
            (rf"`exp/exp_kuailive", f"`../{new_topic}/exp/exp_kuailive"),
        ]
    
    # Topic 字段更新（通用）
    patterns.extend([
        (rf"Topic:` `{old_topic}`", f"Topic:` `{new_topic}`"),
        (rf"Topic: `{old_topic}`", f"Topic: `{new_topic}`"),
    ])
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    # 更新实验 ID 中的 topic（仅在新 topic 目录下的文件）
    if is_in_new_topic:
        content = re.sub(
            rf"EXP-\d+-{old_topic}-(\d+)",
            rf"EXP-{datetime.now().strftime('%Y%m%d')}-{new_topic}-\1",
            content
        )
    
    if content != original_content:
        if not dry_run:
            file_path.write_text(content, encoding="utf-8")
            print(f"✅ 已更新链接: {file_path}")
        else:
            print(f"🔍 [DRY-RUN] 将更新链接: {file_path}")

def update_all_affected_links(new_topic: str, old_topic: str, keyword: str, dry_run: bool = False):
    """更新所有受影响的文件链接"""
    # 更新新 topic 目录下的文件
    if new_topic in TOPIC_DIR_MAPPING:
        new_topic_dir = TOPIC_DIR_MAPPING[new_topic]
    else:
        new_topic_dir = PROJECT_ROOT / "experiments" / new_topic
    
    for md_file in new_topic_dir.rglob("*.md"):
        update_links_in_file(md_file, old_topic, new_topic, dry_run)
    
    # 更新父 topic 的 roadmap 和 hub
    if old_topic in TOPIC_DIR_MAPPING:
        parent_dir = TOPIC_DIR_MAPPING[old_topic]
    else:
        parent_dir = PROJECT_ROOT / "experiments" / old_topic
    
    for hub_file in [parent_dir / f"{old_topic}_hub.md", parent_dir / f"{old_topic}_roadmap.md"]:
        if hub_file.exists():
            update_links_in_file(hub_file, old_topic, new_topic, dry_run)
    
    # 更新 README.md
    readme_file = PROJECT_ROOT / "README.md"
    if readme_file.exists():
        update_links_in_file(readme_file, old_topic, new_topic, dry_run)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Grow Topic - 子节点生长命令",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python _backend/scripts/grow_topic.py kuailive gift_allocation
  python _backend/scripts/grow_topic.py kuailive gift_allocation --dry-run
        """
    )
    
    parser.add_argument("new_topic", help="新 topic 名称（如 kuailive）")
    parser.add_argument("parent_topic", help="父 topic 名称（如 gift_allocation）")
    parser.add_argument("--keyword", help="用于匹配实验文件的关键词（默认使用 new_topic）")
    parser.add_argument("--insights", help="Hub 文件中的 insights 内容（可选）")
    parser.add_argument("--dry-run", action="store_true", help="只显示将要执行的操作，不实际执行")
    
    args = parser.parse_args()
    
    keyword = args.keyword or args.new_topic
    
    print(f"\n{'='*60}")
    print(f"🌱 Grow Topic: {args.new_topic} (from {args.parent_topic})")
    print(f"{'='*60}\n")
    
    if args.dry_run:
        print("🔍 DRY-RUN 模式：只显示操作，不实际执行\n")
    
    # 1. 查找相关文件
    print("📋 Step 1: 查找相关文件...")
    experiments = find_experiments_by_keyword(args.parent_topic, keyword)
    prompts = find_prompts_by_keyword(args.parent_topic, keyword)
    
    print(f"   找到 {len(experiments)} 个实验文件")
    for exp in experiments:
        print(f"     - {exp.name}")
    print(f"   找到 {len(prompts)} 个 prompt 文件")
    for prompt in prompts:
        print(f"     - {prompt.name}")
    
    # 2. 创建目录结构
    print(f"\n📁 Step 2: 创建目录结构...")
    new_topic_dir = create_directory_structure(args.new_topic, args.parent_topic, args.dry_run)
    
    # 3. 创建 hub.md
    print(f"\n📝 Step 3: 创建 hub.md...")
    create_hub_file(args.new_topic, new_topic_dir, args.insights, args.dry_run)
    
    # 4. 创建 roadmap.md
    print(f"\n📝 Step 4: 创建 roadmap.md...")
    create_roadmap_file(args.new_topic, new_topic_dir, args.dry_run)
    
    # 5. 移动实验文件
    if experiments:
        print(f"\n📦 Step 5: 移动实验文件...")
        exp_dir = new_topic_dir / "exp"
        move_files(experiments, exp_dir, args.dry_run)
    
    # 6. 移动 prompt 文件
    if prompts:
        print(f"\n📦 Step 6: 移动 prompt 文件...")
        prompts_dir = new_topic_dir / "prompts"
        move_files(prompts, prompts_dir, args.dry_run)
    
    # 7. 更新所有链接
    print(f"\n🔗 Step 7: 更新所有受影响的链接...")
    update_all_affected_links(args.new_topic, args.parent_topic, keyword, args.dry_run)
    
    print(f"\n{'='*60}")
    print(f"✅ 完成！新 topic '{args.new_topic}' 已创建")
    print(f"{'='*60}\n")
    
    if not args.dry_run:
        print(f"📂 新 topic 位置: {new_topic_dir}")
        print(f"📝 Hub: {new_topic_dir / f'{args.new_topic}_hub.md'}")
        print(f"📝 Roadmap: {new_topic_dir / f'{args.new_topic}_roadmap.md'}")
        print(f"\n💡 提示: 请检查并完善 hub.md 中的内容，特别是洞见汇合部分")

if __name__ == "__main__":
    main()
