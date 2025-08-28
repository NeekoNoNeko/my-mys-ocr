import os
import argparse
from typing import List

# 支持的图片扩展名（全部小写）
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


def is_image_file(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMG_EXTS


def decode_path_to_char(token: str) -> str:
    """
    Decode a single token back to its original character.
    - If token matches 'U+XXXX' pattern, decode it back to the original character.
    - Otherwise, return the token as-is.
    """
    if token.startswith('U+') and len(token) == 6:
        try:
            # Extract hex code and convert to character
            hex_code = token[2:]
            char_code = int(hex_code, 16)
            return chr(char_code)
        except (ValueError, OverflowError):
            # If decoding fails, return as-is
            pass
    return token


def decode_filename(filename: str) -> str:
    """
    Decode a filename containing U+XXXX encoded characters back to original characters.
    Split by U+XXXX pattern and decode each matching part.
    """
    import re

    # Split filename by U+XXXX pattern while preserving the matches
    pattern = r'(U\+[0-9A-F]{4})'
    parts = re.split(pattern, filename)

    decoded_parts = []
    for part in parts:
        if part.startswith('U+') and len(part) == 6:
            decoded_parts.append(decode_path_to_char(part))
        else:
            decoded_parts.append(part)

    return ''.join(decoded_parts)


def remove_suffix_pattern(filename: str) -> str:
    """
    Remove _xxx suffix pattern from filename.
    For example: AA_1 -> AA, BB_test -> BB
    """
    import re

    # 匹配 _xxx 模式（下划线后跟任意字符）
    pattern = r'_.*$'
    return re.sub(pattern, '', filename)


def list_images(directory: str, recursive: bool = False) -> List[str]:
    """
    列出目录中的所有图片文件。
    返回相对目录的文件路径（相对于directory）。
    """
    results: List[str] = []
    if recursive:
        for root, _, files in os.walk(directory):
            for f in files:
                if is_image_file(f):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, directory)
                    results.append(rel_path)
    else:
        try:
            for f in os.listdir(directory):
                if is_image_file(f):
                    results.append(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"目录不存在: {directory}")
    results.sort(key=lambda p: p.lower())
    return results




def main():
    parser = argparse.ArgumentParser(
        description="为combination中的所有图片生成标签txt，其中标签为图片名（不含扩展名）"
    )
    default_dir = os.path.join('tools', 'split_character', 'combination')
    parser.add_argument('--img-dir', '-i', default=default_dir, help='图片所在目录（默认: tools/split_character/combination）')
    parser.add_argument('--out', '-o', default=None, help='输出labels.txt路径（默认写入到图片目录下的labels.txt）')
    parser.add_argument('--recursive', '-r', action='store_true', help='递归扫描子目录中的图片')

    args = parser.parse_args()

    img_dir = args.img_dir
    out_path = args.out if args.out else os.path.join(img_dir, 'labels.txt')

    if not os.path.isdir(img_dir):
        print(f"[ERROR] 图片目录不存在: {img_dir}")
        return 1

    images = list_images(img_dir, recursive=args.recursive)

    if not images:
        print(f"[WARN] 在目录中未找到图片: {img_dir}")

    # 写入标签文件
    # 按照 gen_label_txt.py 的格式：<文件名>\t<标签>\n
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        for rel_path in images:
            # 标签为图片的文件名（去除扩展名）
            base_name = os.path.basename(rel_path)
            name_wo_ext, _ = os.path.splitext(base_name)
            # 移除_xxx后缀模式
            name_without_suffix = remove_suffix_pattern(name_wo_ext)
            # 解码文件名中的U+XXXX格式字符
            decoded_label = decode_filename(name_without_suffix)
            # 输出第一列为相对img_dir的路径（若recursive，则包含子目录），与gen_label_txt相似（其为文件名）
            f.write(f"{rel_path}\t{decoded_label}\n")

    print(f"已生成标签文件: {out_path}，共 {len(images)} 条")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
