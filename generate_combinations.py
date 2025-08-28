import os
import sys
import argparse
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import cv2
import numpy as np


def is_image_file(name: str) -> bool:
    lower = name.lower()
    return lower.endswith('.jpg') or lower.endswith('.jpeg') or lower.endswith('.png') or lower.endswith('.bmp')

def sanitize_char_for_path(ch: str) -> str:
    """
    Sanitize a single character for safe use in Windows folder/file names.
    - For disallowed characters < > : " / \\ | ? * and special single names like '.' and '..',
      return a reversible token 'U+XXXX' (uppercase hex code point).
    - Otherwise, return the character as-is.
    """
    if len(ch) != 1:
        # Fallback: encode the whole string
        return 'U+' + ''.join(f"{ord(c):04X}" for c in ch)
    invalid = set('<>:"/\\|?*')
    if ch in invalid or ch in {'.', ' '}:
        return f"U+{ord(ch):04X}"
    # Also protect control chars
    if ord(ch) < 32:
        return f"U+{ord(ch):04X}"
    return ch


def sanitize_filename_unicode(text: str) -> str:
    """
    Sanitize a text string for safe use as filename by processing each character
    using sanitize_char_for_path function.
    """
    if not text:
        return "empty"
    result = ''.join(sanitize_char_for_path(ch) for ch in text)
    return result if result else "empty"

def decode_path_to_char(folder_name: str) -> str:
    """
    Decode a folder name back to its original character.
    - If folder_name matches 'U+XXXX' pattern, decode it back to the original character.
    - Otherwise, return the folder_name as-is (assuming it's already a valid character).
    """
    if folder_name.startswith('U+') and len(folder_name) == 6:
        try:
            # Extract hex code and convert to character
            hex_code = folder_name[2:]
            char_code = int(hex_code, 16)
            return chr(char_code)
        except (ValueError, OverflowError):
            # If decoding fails, treat as regular folder name
            pass
    return folder_name




def scan_split_root(split_root: str) -> Dict[str, List[str]]:
    """
    Walk tools/split_character/split and build a mapping: character -> list of image file paths
    Expected structure:
      split/<subfolder>/<char>/<char>-1.jpg, ...
    Also supports encoded folder names like U+003C for special characters.
    """
    char_to_paths: Dict[str, List[str]] = {}
    if not os.path.isdir(split_root):
        raise FileNotFoundError(f"Split root not found: {split_root}")

    for sub in sorted(os.listdir(split_root)):
        sub_path = os.path.join(split_root, sub)
        if not os.path.isdir(sub_path):
            continue
        # inside sub_path there are character directories (single-char or encoded)
        for ch_name in sorted(os.listdir(sub_path)):
            ch_dir = os.path.join(sub_path, ch_name)
            if not os.path.isdir(ch_dir):
                continue

            # Decode folder name to get the actual character
            ch_label = decode_path_to_char(ch_name)

            # Process all image files in this character directory
            for fname in sorted(os.listdir(ch_dir)):
                if not is_image_file(fname):
                    continue
                full_path = os.path.join(ch_dir, fname)
                char_to_paths.setdefault(ch_label, []).append(full_path)

    return char_to_paths


def read_image_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def pad_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Pad a grayscale image to target_h with constant (white) padding, centered vertically."""
    h, w = img.shape[:2]
    if h == target_h:
        return img
    if h > target_h:
        # If taller, resize down preserving aspect ratio
        new_w = max(1, int(round(w * target_h / h)))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)
    # pad
    pad_total = target_h - h
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top
    # assume white background (255)
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=255)


def hconcat_images(images: List[np.ndarray]) -> np.ndarray:
    """Horizontally concatenate a list of grayscale images by first padding each to the max height."""
    if not images:
        raise ValueError("No images to concatenate")
    heights = [im.shape[0] for im in images]
    target_h = max(heights)
    padded = [pad_to_height(im, target_h) for im in images]
    return cv2.hconcat(padded)


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Invalid image size for resizing")
    if h == height:
        return img
    new_w = max(1, int(round(w * height / h)))
    return cv2.resize(img, (new_w, height), interpolation=cv2.INTER_AREA if height < h else cv2.INTER_CUBIC)


def ensure_unique_path(base_dir: str, base_name: str, ext: str = '.jpg') -> str:
    """
    Ensure a unique file path in base_dir with given base_name and extension.
    First try base_name.ext, then base_name_1.ext, base_name_2.ext, ...
    """
    path = os.path.join(base_dir, base_name + ext)
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        candidate = os.path.join(base_dir, f"{base_name}_{idx}{ext}")
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def generate_one(char_to_paths: Dict[str, List[str]], min_len: int, max_len: int, rng: random.Random) -> Tuple[str, np.ndarray]:
    if not char_to_paths:
        raise ValueError("No character images found under split directory")
    chars = list(char_to_paths.keys())
    L = rng.randint(min_len, max_len)
    text_chars: List[str] = []
    imgs: List[np.ndarray] = []
    for _ in range(L):
        ch = rng.choice(chars)
        paths = char_to_paths[ch]
        if not paths:
            continue
        path = rng.choice(paths)
        img = read_image_grayscale(path)
        # 将单字符图片等比例缩放到高度32像素
        img_resized = resize_to_height(img, 32)
        text_chars.append(ch)
        imgs.append(img_resized)
    if not imgs:
        raise ValueError("Failed to sample any images for this combination")
    text = ''.join(text_chars)
    combined = hconcat_images(imgs)
    return text, combined


def save_image(path: str, img: np.ndarray) -> None:
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save as JPEG with reasonable quality
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def show_progress(current: int, total: int, bar_length: int = 50) -> None:
    """显示进度条"""
    progress = current / total
    filled_length = int(bar_length * progress)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    percent = progress * 100
    print(f'\r进度: |{bar}| {current}/{total} ({percent:.1f}%)', end='', flush=True)
    if current == total:
        print()  # 完成时换行


def generate_image_worker(worker_id: int, mapping: Dict[str, List[str]], min_len: int, max_len: int, 
                         out_dir: str, generated_texts: set, lock: threading.Lock, 
                         seed: int = None, verbose: bool = False) -> Tuple[str, bool]:
    """线程工作函数，生成单个图片"""
    # 为每个线程创建独立的随机数生成器
    worker_rng = random.Random(seed + worker_id if seed is not None else None)

    max_attempts = 100  # 单个工作线程的最大尝试次数
    for _ in range(max_attempts):
        try:
            text, img = generate_one(mapping, min_len, max_len, worker_rng)

            # 线程安全地检查和添加文本
            with lock:
                if text in generated_texts:
                    continue  # 文本重复，继续尝试
                generated_texts.add(text)

            # 保存图片
            fname_base = sanitize_filename_unicode(text)
            target_path = ensure_unique_path(out_dir, fname_base, ext='.jpg')
            save_image(target_path, img)
            return text, True

        except Exception as e:
            if verbose:
                print(f"\n[WARN] Worker {worker_id} skipping one due to: {e}")
            continue

    return "", False  # 失败


def main():
    parser = argparse.ArgumentParser(description="Generate concatenated string images from split character images")
    parser.add_argument('--split-dir', '-s', default=os.path.join('tools', 'split_character', 'split'), help='Root directory containing per-subfolder character images')
    parser.add_argument('--out-dir', '-o', default=os.path.join('tools', 'split_character', 'combination'), help='Output directory to save concatenated images')
    parser.add_argument('--min-len', type=int, default=1, help='Minimum number of characters per generated image')
    parser.add_argument('--max-len', type=int, default=21, help='Maximum number of characters per generated image')
    parser.add_argument('--count', '-c', type=int, default=50000, help='Number of images to generate')
    parser.add_argument('--threads', '-t', type=int, default=4, help='Number of threads to use for parallel processing')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress information')

    args = parser.parse_args()

    if args.min_len < 1 or args.max_len < args.min_len:
        print("Invalid min/max length settings", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)

    try:
        mapping = scan_split_root(args.split_dir)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    # Filter out characters with no images
    mapping = {ch: paths for ch, paths in mapping.items() if paths}
    if not mapping:
        print(f"[ERROR] No character images found in: {args.split_dir}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    total = args.count
    generated = 0
    generated_texts = set()  # 用于跟踪已生成的文本，确保不重复
    lock = threading.Lock()  # 线程锁保护共享资源

    print(f"开始使用 {args.threads} 个线程生成 {total} 个不重复的图片...")
    show_progress(0, total)  # 显示初始进度

    # 使用线程池执行并行处理
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # 提交任务
        future_to_worker = {}
        for i in range(total):
            future = executor.submit(
                generate_image_worker, 
                i % args.threads,  # 工作线程ID
                mapping, 
                args.min_len, 
                args.max_len, 
                args.out_dir,
                generated_texts,
                lock,
                args.seed,
                args.verbose
            )
            future_to_worker[future] = i

        # 处理完成的任务
        for future in as_completed(future_to_worker):
            try:
                text, success = future.result()
                if success:
                    generated += 1
                    show_progress(generated, total)

                    # 如果已经生成足够数量，取消剩余任务
                    if generated >= total:
                        for remaining_future in future_to_worker:
                            if not remaining_future.done():
                                remaining_future.cancel()
                        break

            except Exception as e:
                if args.verbose:
                    print(f"\n[WARN] Task failed: {e}")
                    show_progress(generated, total)

    if generated < total:
        print(f"\n注意：只生成了 {generated}/{total} 个图片，可能已达到可能组合的上限")
    else:
        print(f"完成！已生成 {generated} 个不重复的图片到: {args.out_dir}")


if __name__ == '__main__':
    main()
