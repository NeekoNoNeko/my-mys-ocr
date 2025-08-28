import os
import csv
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import re

# ==================== 默认参数配置 ====================

# 默认输出文件名
DEFAULT_OUTPUT_FILE = "files_with_labels.csv"

# 默认预览文件数量
DEFAULT_PREVIEW_LIMIT = 10

# 支持的文件扩展名分类
SUPPORTED_EXTENSIONS = {
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp', '.svg', '.ico'],
    'document': ['.txt', '.doc', '.docx', '.pdf', '.rtf', '.odt', '.pages'],
    'video': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm', '.m4v'],
    'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'],
    'archive': ['.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz'],
    'code': ['.py', '.java', '.cpp', '.c', '.js', '.html', '.css', '.php', '.rb', '.go'],
    # 'data': ['.json', '.xml', '.csv', '.xlsx', '.xls', '.yaml', '.yml'],
'   data': ['.json', '.xml', '.xlsx', '.xls', '.yaml', '.yml'],
    'executable': ['.exe', '.msi', '.deb', '.rpm', '.dmg', '.app']
}

# 默认标签（当文件类型未知时使用）
DEFAULT_LABEL = 'other'

# 统一标签（如果设置了此变量，所有文件都使用此标签，为None时使用自动标签）
UNIFIED_LABEL = ">MgAI9Zn1(A)<E3"

# CSV文件编码
CSV_ENCODING = 'utf-8'

# 预设的标签规则示例（可以在交互模式中参考）
EXAMPLE_LABEL_RULES = {
    r'IMG_\d+': 'camera_photo',
    r'screenshot.*': 'screenshot',
    r'.*test.*': 'test_file',
    r'backup.*': 'backup',
    r'temp.*': 'temporary',
    r'.*\.(log|tmp)$': 'log_temp'
}

# ====================================================

class FileCSVGenerator:
    """文件名提取和CSV生成工具类"""

    def __init__(self):
        self.supported_extensions = SUPPORTED_EXTENSIONS

    def get_file_type_label(self, filename: str) -> str:
        """根据文件扩展名自动生成标签"""
        ext = Path(filename).suffix.lower()

        for file_type, extensions in self.supported_extensions.items():
            if ext in extensions:
                return file_type

        return DEFAULT_LABEL

    def get_custom_label(self, filename: str, label_rules: Dict[str, str] = None, unified_label: str = None) -> str:
        """根据自定义规则生成标签"""
        # 优先使用统一标签
        if unified_label:
            return unified_label

        if not label_rules:
            return self.get_file_type_label(filename)

        filename_lower = filename.lower()

        # 检查是否匹配自定义规则
        for pattern, label in label_rules.items():
            if re.search(pattern, filename_lower):
                return label

        # 如果没有匹配到自定义规则，使用默认的文件类型标签
        return self.get_file_type_label(filename)

    def scan_directory(self, directory_path: str, recursive: bool = False) -> List[Tuple[str, str]]:
        """
        扫描目录获取所有文件

        Args:
            directory_path: 目录路径
            recursive: 是否递归扫描子目录

        Returns:
            List[Tuple[str, str]]: (文件名, 相对路径) 的列表
        """
        files_info = []
        directory = Path(directory_path)

        if not directory.exists():
            print(f"错误: 目录 '{directory_path}' 不存在")
            return files_info

        if not directory.is_dir():
            print(f"错误: '{directory_path}' 不是一个目录")
            return files_info

        # 根据是否递归选择不同的扫描方式
        if recursive:
            pattern = "**/*"
            files = directory.glob(pattern)
        else:
            files = directory.iterdir()

        for file_path in files:
            if file_path.is_file():
                filename = file_path.name
                relative_path = str(file_path.relative_to(directory))
                files_info.append((filename, relative_path))

        return sorted(files_info)

    def generate_csv(self, directory_path: str, output_file: str = DEFAULT_OUTPUT_FILE, 
                    recursive: bool = False, label_rules: Dict[str, str] = None,
                    custom_labels: Dict[str, str] = None, unified_label: str = None) -> bool:
        """
        生成包含文件名和标签的CSV文件

        Args:
            directory_path: 扫描的目录路径
            output_file: 输出CSV文件名
            recursive: 是否递归扫描
            label_rules: 标签规则字典 {正则表达式: 标签}
            custom_labels: 自定义标签字典 {文件名: 标签}

        Returns:
            bool: 是否成功生成
        """
        print(f"正在扫描目录: {directory_path}")
        files_info = self.scan_directory(directory_path, recursive)

        if not files_info:
            print("未找到任何文件")
            return False

        print(f"找到 {len(files_info)} 个文件")

        # 如果输出文件不是绝对路径，则将其放在目标文件夹中
        output_path = Path(output_file)
        if not output_path.is_absolute():
            target_dir = Path(directory_path)
            output_file = str(target_dir / output_file)

        print(f"输出文件路径: {output_file}")

        # 生成CSV数据
        csv_data = []
        for filename, relative_path in files_info:
            # 优先使用用户自定义标签
            if custom_labels and filename in custom_labels:
                label = custom_labels[filename]
            else:
                # 使用规则生成标签（包括统一标签）
                label = self.get_custom_label(filename, label_rules, unified_label)

            csv_data.append({
                '文件名': filename,
                '标签': label,
                '相对路径': relative_path if recursive else ''
            })

        # 写入CSV文件
        try:
            with open(output_file, 'w', newline='', encoding=CSV_ENCODING) as csvfile:
                fieldnames = ['文件名', '标签']
                if recursive:
                    fieldnames.append('相对路径')

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for row in csv_data:
                    if not recursive:
                        # 如果不是递归模式，不包含相对路径列
                        row = {k: v for k, v in row.items() if k != '相对路径'}
                    writer.writerow(row)

            print(f"CSV文件已生成: {output_file}")
            return True

        except Exception as e:
            print(f"生成CSV文件时出错: {e}")
            return False

    def print_preview(self, directory_path: str, limit: int = DEFAULT_PREVIEW_LIMIT, 
                     recursive: bool = False, label_rules: Dict[str, str] = None, unified_label: str = None):
        """预览前几个文件的标签结果"""
        files_info = self.scan_directory(directory_path, recursive)

        if not files_info:
            print("未找到任何文件")
            return

        print(f"\n预览前 {min(limit, len(files_info))} 个文件:")
        print("-" * 60)
        print(f"{'文件名':<30} {'标签':<15} {'路径'}")
        print("-" * 60)

        for i, (filename, relative_path) in enumerate(files_info[:limit]):
            label = self.get_custom_label(filename, label_rules, unified_label)
            path_display = relative_path if recursive else "."
            print(f"{filename:<30} {label:<15} {path_display}")

        if len(files_info) > limit:
            print(f"... 还有 {len(files_info) - limit} 个文件")


def create_label_rules_interactive() -> Dict[str, str]:
    """交互式创建标签规则"""
    rules = {}
    print("\n=== 创建标签规则 ===")
    print("输入正则表达式模式和对应标签，空行结束")
    print("预设示例规则:")
    for pattern, label in list(EXAMPLE_LABEL_RULES.items())[:3]:
        print(f"  '{pattern}' -> '{label}'")
    print()

    while True:
        pattern = input("正则表达式模式 (空行结束): ").strip()
        if not pattern:
            break

        label = input(f"标签 (对应 '{pattern}'): ").strip()
        if label:
            rules[pattern] = label
            print(f"已添加规则: '{pattern}' -> '{label}'")

    return rules


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成文件名和标签的CSV文件')
    parser.add_argument('directory', help='要扫描的目录路径')
    parser.add_argument('-o', '--output', default=DEFAULT_OUTPUT_FILE, 
                       help=f'输出CSV文件名 (默认: {DEFAULT_OUTPUT_FILE})')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='递归扫描子目录')
    parser.add_argument('-p', '--preview', type=int, default=0,
                       help='预览前N个文件 (不生成CSV)')
    parser.add_argument('--interactive', action='store_true',
                       help='交互式设置标签规则')
    parser.add_argument('-l', '--label', type=str,
                       help='为所有文件设置统一标签')

    args = parser.parse_args()

    generator = FileCSVGenerator()

    # 检查目录是否存在
    if not os.path.exists(args.directory):
        print(f"错误: 目录 '{args.directory}' 不存在")
        return

    # 交互式设置标签规则
    label_rules = None
    if args.interactive:
        label_rules = create_label_rules_interactive()

    # 获取统一标签（命令行参数优先于配置文件）
    unified_label = args.label or UNIFIED_LABEL

    # 预览模式
    if args.preview > 0:
        generator.print_preview(args.directory, args.preview, args.recursive, label_rules, unified_label)
        return

    # 生成CSV
    success = generator.generate_csv(
        directory_path=args.directory,
        output_file=args.output,
        recursive=args.recursive,
        label_rules=label_rules,
        unified_label=unified_label
    )

    if success:
        print("\n任务完成!")
        print(f"扫描目录: {args.directory}")
        print(f"输出文件: {args.output}")
        if args.recursive:
            print("模式: 递归扫描")
        if unified_label:
            print(f"统一标签: {unified_label}")
    else:
        print("任务失败!")


if __name__ == "__main__":
    main()
