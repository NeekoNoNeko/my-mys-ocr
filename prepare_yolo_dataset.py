import argparse
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Set

"""
根据已有的 YOLO 标签文件，生成符合 YOLO 训练要求的数据集。

脚本会扫描源目录（默认：tools/split_character/labels）中的图片与同名 .txt 标签文件，
并按给定比例划分为训练集/验证集，最终复制到以下目录：

- data/train/images
- data/train/labels
- data/valid/images
- data/valid/labels

说明：
- 支持多行图片（如 line_01_img.jpg、line_02_img.jpg …）。
- 保留原始文件名。如果不同子目录下出现同名文件，会在复制时自动追加编号（_1、_2…）
  以避免覆盖。也可通过 --keep-subdirs 参数保留原始子目录结构，进一步降低冲突概率。
- 仅当图片与对应标签同时存在时才被纳入数据集。
- 支持的图片扩展名：.jpg、.jpeg、.png、.bmp、.tif、.tiff

使用示例：
  python tools/prepare_yolo_dataset.py \
      --src tools/split_character/labels \
      --out data \
      --val-ratio 0.1 \
      --seed 42 \
      --clean

运行结束后，生成的 tools/train.yaml 中应包含：
  train: data/train/images
  val:   data/valid/images
即可直接用于 YOLO 训练。
"""

# ===========================================
# 配置区域 - 用户可在此处修改默认参数
# ===========================================

# 默认路径
DEFAULT_SOURCE_PATH = str(Path('tools') / 'split_character' / 'labels')
DEFAULT_OUTPUT_PATH = 'data'

# 数据集划分比例
DEFAULT_VAL_RATIO = 0.2  # 验证集占比
DEFAULT_SEED = 42        # 随机种子

# 支持的图片后缀
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# ===========================================
# 配置区域结束
# ===========================================


def find_pairs(src_root: Path, keep_subdirs: bool = False) -> List[Tuple[Path, Path, Path]]:
    """
    查找 (图片路径, 标签路径, 相对目标目录) 三元组列表。
    若 keep_subdirs 为 True，则保持原始子目录结构。
    """
    pairs: List[Tuple[Path, Path, Path]] = []
    for root, dirs, files in os.walk(src_root):
        root_path = Path(root)
        for fname in files:
            fpath = root_path / fname
            if fpath.suffix.lower() in IMG_EXTS:
                stem = fpath.stem
                label_path = root_path / f"{stem}.txt"
                if not label_path.is_file():
                    # 跳过无标签的图片
                    continue
                if keep_subdirs:
                    rel_dir = fpath.parent.relative_to(src_root)
                else:
                    rel_dir = Path('.')
                pairs.append((fpath, label_path, rel_dir))
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path, Path]], val_ratio: float, seed: int) -> Tuple[List, List]:
    random.seed(seed)
    shuffled = pairs[:]
    random.shuffle(shuffled)
    n_total = len(shuffled)
    n_val = int(round(n_total * val_ratio))
    val_pairs = shuffled[:n_val]
    train_pairs = shuffled[n_val:]
    return train_pairs, val_pairs


def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dst))


def list_existing_stems(dir_path: Path) -> Set[str]:
    stems: Set[str] = set()
    if dir_path.exists():
        for p in dir_path.glob('*'):
            if p.is_file():
                stems.add(p.stem)
    return stems


def unique_stem(desired: str, used: Set[str]) -> str:
    if desired not in used:
        used.add(desired)
        return desired
    idx = 1
    while True:
        candidate = f"{desired}_{idx}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        idx += 1


def prepare_dataset(src: Path, out_root: Path, val_ratio: float, seed: int, clean: bool, keep_subdirs: bool) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"未找到源目录：{src}")

    train_img_dir = out_root / 'train' / 'images'
    train_lbl_dir = out_root / 'train' / 'labels'
    val_img_dir = out_root / 'valid' / 'images'
    val_lbl_dir = out_root / 'valid' / 'labels'

    if clean:
        # 清空已有目录，防止旧文件干扰
        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            if d.exists():
                shutil.rmtree(d)

    pairs = find_pairs(src, keep_subdirs=keep_subdirs)
    n_pairs = len(pairs)
    if n_pairs == 0:
        print(f"在 {src} 中未找到任何图片/标签对。")
        return

    # 统计多行图片信息
    line_stats = {}
    for img_path, _, _ in pairs:
        filename = img_path.name
        if filename.startswith('line_') and filename.endswith('_img.jpg'):
            try:
                line_part = filename.split('_')[1]
                line_num = int(line_part)
                line_stats[line_num] = line_stats.get(line_num, 0) + 1
            except (IndexError, ValueError):
                pass

    train_pairs, val_pairs = split_pairs(pairs, val_ratio, seed)

    # 为每个相对子目录维护已用 stem 集合，确保图片与标签同步命名
    used_train: Dict[str, Set[str]] = {}
    used_val: Dict[str, Set[str]] = {}

    def ensure_used_map(base_img_dir: Path, base_lbl_dir: Path, rel: Path, used_map: Dict[str, Set[str]]):
        key = str(rel).replace('\\', '/')
        if key not in used_map:
            stems: Set[str] = set()
            stems |= list_existing_stems(base_img_dir / rel)
            stems |= list_existing_stems(base_lbl_dir / rel)
            used_map[key] = stems
        return used_map[key]

    # 复制训练集
    for (img_path, lbl_path, rel) in train_pairs:
        used = ensure_used_map(train_img_dir, train_lbl_dir, rel, used_train)
        stem = img_path.stem
        new_stem = unique_stem(stem, used)
        dest_img = (train_img_dir / rel / f"{new_stem}{img_path.suffix}")
        dest_lbl = (train_lbl_dir / rel / f"{new_stem}.txt")
        safe_copy(img_path, dest_img)
        safe_copy(lbl_path, dest_lbl)

    # 复制验证集
    for (img_path, lbl_path, rel) in val_pairs:
        used = ensure_used_map(val_img_dir, val_lbl_dir, rel, used_val)
        stem = img_path.stem
        new_stem = unique_stem(stem, used)
        dest_img = (val_img_dir / rel / f"{new_stem}{img_path.suffix}")
        dest_lbl = (val_lbl_dir / rel / f"{new_stem}.txt")
        safe_copy(img_path, dest_img)
        safe_copy(lbl_path, dest_lbl)

    print("YOLO 数据集准备完毕：")
    print(f"  源目录：{src}")
    print(f"  总对数：{n_pairs}")

    # 打印多行图片统计
    if line_stats:
        print(f"  检测到多行图片：")
        for line_num in sorted(line_stats.keys()):
            print(f"    行 {line_num:02d}: {line_stats[line_num]} 张")

    print(f"  训练集：{len(train_pairs)} 张 -> {train_img_dir}")
    print(f"  验证集：{len(val_pairs)} 张 -> {val_img_dir}")


def main():
    parser = argparse.ArgumentParser(description='根据 YOLO 标签文件生成训练/验证数据集，支持多行图片。')
    parser.add_argument('--src', type=str, default=DEFAULT_SOURCE_PATH, help='源根目录，包含图片与 .txt 标签')
    parser.add_argument('--out', type=str, default=DEFAULT_OUTPUT_PATH, help='输出根目录（默认：data）')
    parser.add_argument('--val-ratio', type=float, default=DEFAULT_VAL_RATIO, help='验证集比例（默认：0.1）')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='划分随机种子')
    parser.add_argument('--clean', action='store_true', help='复制前清空输出目录')
    parser.add_argument('--keep-subdirs', action='store_true', help='在输出目录中保留原始子目录结构')
    args = parser.parse_args()

    src = Path(args.src)
    out_root = Path(args.out)

    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError('验证集比例 val-ratio 必须在 [0, 1) 之间。')

    prepare_dataset(src, out_root, args.val_ratio, args.seed, args.clean, args.keep_subdirs)


if __name__ == '__main__':
    main()