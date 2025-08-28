#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO数据集分割工具

功能：
- 自动扫描指定文件夹下的图片和对应的txt标签文件
- 按指定比例分割成训练集(train)和验证集(valid)
- 生成符合YOLO训练标准的目录结构

使用方法：
python yolo_dataset_splitter.py --input_dir /path/to/dataset --output_dir /path/to/output --train_ratio 0.8

目录结构：
output_dir/
├── train/
│   ├── images/
│   └── labels/
└── valid/
    ├── images/
    └── labels/
"""

import os
import random
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Set

# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


class YOLODatasetSplitter:
    def __init__(self, input_dir: str, output_dir: str, train_ratio: float = 0.8, random_seed: int = 42):
        """
        初始化YOLO数据集分割器

        Args:
            input_dir: 输入目录，包含图片和标签文件
            output_dir: 输出目录
            train_ratio: 训练集比例 (0-1之间)
            random_seed: 随机种子，确保结果可重现
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.train_ratio = train_ratio
        self.valid_ratio = 1.0 - train_ratio
        self.random_seed = random_seed

        # 设置随机种子
        random.seed(random_seed)

        # 创建输出目录结构
        self.train_images_dir = self.output_dir / 'train' / 'images'
        self.train_labels_dir = self.output_dir / 'train' / 'labels'
        self.valid_images_dir = self.output_dir / 'valid' / 'images'
        self.valid_labels_dir = self.output_dir / 'valid' / 'labels'

    def find_image_label_pairs(self) -> List[Tuple[Path, Path]]:
        """
        查找输入目录中的图片和对应标签文件对

        Returns:
            图片路径和标签路径的元组列表
        """
        pairs = []

        print(f"扫描目录: {self.input_dir}")

        # 遍历输入目录（包括子目录）
        for root, dirs, files in os.walk(self.input_dir):
            root_path = Path(root)

            for file in files:
                file_path = root_path / file

                # 检查是否为图片文件
                if file_path.suffix.lower() in IMAGE_EXTENSIONS:
                    # 查找对应的标签文件
                    label_file = file_path.with_suffix('.txt')

                    if label_file.exists():
                        pairs.append((file_path, label_file))
                    else:
                        print(f"警告: 找不到对应标签文件 {label_file}")

        print(f"找到 {len(pairs)} 个图片-标签对")
        return pairs

    def create_output_dirs(self):
        """创建输出目录结构"""
        dirs_to_create = [
            self.train_images_dir,
            self.train_labels_dir,
            self.valid_images_dir,
            self.valid_labels_dir
        ]

        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"创建目录: {dir_path}")

    def split_dataset(self, pairs: List[Tuple[Path, Path]]) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
        """
        按比例分割数据集

        Args:
            pairs: 图片标签对列表

        Returns:
            (训练集对, 验证集对)
        """
        # 随机打乱数据
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)

        # 计算分割点
        total_count = len(shuffled_pairs)
        train_count = int(total_count * self.train_ratio)

        train_pairs = shuffled_pairs[:train_count]
        valid_pairs = shuffled_pairs[train_count:]

        print(f"数据集分割完成:")
        print(f"  训练集: {len(train_pairs)} 对 ({len(train_pairs)/total_count*100:.1f}%)")
        print(f"  验证集: {len(valid_pairs)} 对 ({len(valid_pairs)/total_count*100:.1f}%)")

        return train_pairs, valid_pairs

    def copy_files(self, pairs: List[Tuple[Path, Path]], images_dir: Path, labels_dir: Path, dataset_type: str):
        """
        复制文件到指定目录

        Args:
            pairs: 图片标签对列表
            images_dir: 图片输出目录
            labels_dir: 标签输出目录
            dataset_type: 数据集类型（用于显示）
        """
        print(f"正在复制{dataset_type}文件...")

        # 用于处理文件名冲突
        used_names: Set[str] = set()

        for i, (image_path, label_path) in enumerate(pairs):
            # 生成唯一文件名
            base_name = image_path.stem
            image_ext = image_path.suffix

            # 处理文件名冲突
            final_name = base_name
            counter = 1
            while final_name in used_names:
                final_name = f"{base_name}_{counter}"
                counter += 1
            used_names.add(final_name)

            # 目标文件路径
            target_image = images_dir / f"{final_name}{image_ext}"
            target_label = labels_dir / f"{final_name}.txt"

            # 复制文件
            try:
                shutil.copy2(image_path, target_image)
                shutil.copy2(label_path, target_label)

                if (i + 1) % 100 == 0 or (i + 1) == len(pairs):
                    print(f"  已复制 {i + 1}/{len(pairs)} 个文件")

            except Exception as e:
                print(f"复制文件失败 {image_path} -> {target_image}: {e}")

    def generate_yaml_config(self):
        """生成YOLO训练配置文件"""
        yaml_content = f"""# YOLO数据集配置文件
# 由yolo_dataset_splitter.py自动生成

# 数据集路径
train: {self.train_images_dir.absolute()}
val: {self.valid_images_dir.absolute()}

# 类别数量（请根据实际情况修改）
nc: 1

# 类别名称（请根据实际情况修改）
names:
  0: 'object'

# 数据集统计信息
# 训练集比例: {self.train_ratio}
# 验证集比例: {self.valid_ratio}
# 随机种子: {self.random_seed}
"""

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(yaml_content)

        print(f"生成配置文件: {yaml_path}")

    def run(self, clean_output: bool = False):
        """
        执行数据集分割

        Args:
            clean_output: 是否清理输出目录
        """
        print("=" * 60)
        print("YOLO数据集分割工具")
        print("=" * 60)

        # 检查输入目录
        if not self.input_dir.exists():
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")

        # 清理输出目录（如果需要）
        if clean_output and self.output_dir.exists():
            print(f"清理输出目录: {self.output_dir}")
            shutil.rmtree(self.output_dir)

        # 创建输出目录
        self.create_output_dirs()

        # 查找图片标签对
        pairs = self.find_image_label_pairs()

        if not pairs:
            print("错误: 未找到任何图片-标签对")
            return

        # 分割数据集
        train_pairs, valid_pairs = self.split_dataset(pairs)

        # 复制文件
        self.copy_files(train_pairs, self.train_images_dir, self.train_labels_dir, "训练集")
        self.copy_files(valid_pairs, self.valid_images_dir, self.valid_labels_dir, "验证集")

        # 生成配置文件
        self.generate_yaml_config()

        print("\n" + "=" * 60)
        print("数据集分割完成！")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='YOLO数据集分割工具')

    parser.add_argument('--input_dir', '-i', type=str, required=True,
                       help='输入目录路径，包含图片和标签文件')

    parser.add_argument('--output_dir', '-o', type=str, default='./yolo_dataset',
                       help='输出目录路径（默认：./yolo_dataset）')

    parser.add_argument('--train_ratio', '-r', type=float, default=0.8,
                       help='训练集比例，0-1之间（默认：0.8）')

    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='随机种子（默认：42）')

    parser.add_argument('--clean', '-c', action='store_true',
                       help='清理输出目录')

    args = parser.parse_args()

    # 参数验证
    if not 0 < args.train_ratio < 1:
        raise ValueError("训练集比例必须在0-1之间")

    # 创建分割器并运行
    splitter = YOLODatasetSplitter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        random_seed=args.seed
    )

    try:
        splitter.run(clean_output=args.clean)
    except Exception as e:
        print(f"错误: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
