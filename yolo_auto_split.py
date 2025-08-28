import os
import sys
import json
import argparse
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
import threading

import cv2
import numpy as np

try:
    # ultralytics >=8.x provides YOLO class for yolo11 models
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

# ===========================================
# 配置区域 - 用户可修改的参数
# ===========================================

# 默认路径配置
DEFAULT_INPUT_PATH = os.path.join('tools', 'split_character', 'output')
DEFAULT_OUTPUT_PATH = os.path.join('tools', 'split_character', 'split')
DEFAULT_WEIGHTS_PATH = os.path.join('tools', 'runs', 'detect', 'train2', 'weights', 'best.pt')

# 默认文本配置 - 按行顺序对应 line_01_img.jpg, line_02_img.jpg, line_03_img.jpg...
DEFAULT_TEXTS = [
    "M2204063429",  # line_01
    "AZ91D",        # line_02
    "FT-1"          # line_03
]

# YOLO推理参数配置
DEFAULT_CONF_THRESHOLD = 0.7  # 置信度阈值
DEFAULT_IOU_THRESHOLD = 0.45   # NMS IoU阈值

# 多线程配置
DEFAULT_THREADS = 4

# 其他配置
PROGRESS_FILENAME = "progress.json"
SAVE_INFERENCE_RESULTS = True  # 是否默认保存推理结果图片

# ===========================================
# 配置区域结束
# ===========================================


def load_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {path}")
    return img


def list_subfolders(root: str) -> List[str]:
    result = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            result.append(name)
    return result


def find_line_images(folder_path: str) -> List[Tuple[int, str]]:
    """
    Find all line_XX_img.jpg files in the folder and return sorted list of (line_number, filename).
    """
    import re
    pattern = re.compile(r'^line_(\d+)_img\.jpg$')
    results = []

    if not os.path.isdir(folder_path):
        return results

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            line_num = int(match.group(1))
            results.append((line_num, filename))

    return sorted(results)


def crop_by_splits(image: np.ndarray, xs: List[int]) -> List[np.ndarray]:
    h, w = image.shape[:2]
    xs_sorted = sorted(set([x for x in xs if 0 < x < w]))
    # add boundaries
    edges = [0] + xs_sorted + [w]
    crops = []
    for i in range(len(edges) - 1):
        x0, x1 = edges[i], edges[i + 1]
        if x1 <= x0:
            continue
        crop = image[:, x0:x1]
        crops.append(crop)
    return crops


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


def ensure_char_dirs(base_dir: str, text: str):
    os.makedirs(base_dir, exist_ok=True)
    for ch in text:
        safe = sanitize_char_for_path(ch)
        ch_dir = os.path.join(base_dir, safe)
        os.makedirs(ch_dir, exist_ok=True)


def next_filename_for_char(ch_dir: str, safe_ch: str) -> str:
    prefix = f"{safe_ch}-"
    max_idx = 0
    if os.path.isdir(ch_dir):
        for name in os.listdir(ch_dir):
            if not name.lower().endswith('.jpg'):
                continue
            if not name.startswith(prefix):
                continue
            base = os.path.splitext(name)[0]
            parts = base.split('-')
            if len(parts) >= 2:
                try:
                    idx = int(parts[-1])
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    pass
    return os.path.join(ch_dir, f"{safe_ch}-{max_idx + 1}.jpg")


def save_crops(output_root: str, folder_name: str, crops: List[np.ndarray], text: str):
    base_dir = os.path.join(output_root, folder_name)
    ensure_char_dirs(base_dir, text)
    if len(crops) != len(text):
        raise ValueError(f"Number of crops ({len(crops)}) does not match text length ({len(text)}).")
    for i, ch in enumerate(text):
        safe = sanitize_char_for_path(ch)
        ch_dir = os.path.join(base_dir, safe)
        target_path = next_filename_for_char(ch_dir, safe)
        cv2.imwrite(target_path, crops[i])


def compute_splits_from_boxes(w: int, boxes_xyxy: np.ndarray) -> List[int]:
    """
    Given sorted (by x) boxes in xyxy format and image width w, compute vertical split x positions
    located between adjacent boxes. This mirrors manual_split behavior by producing full-height
    slices using edges [0] + splits + [w].
    """
    if boxes_xyxy.shape[0] <= 1:
        return []
    xs = []
    # sort by x-center just in case
    centers = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    order = np.argsort(centers)
    b = boxes_xyxy[order]
    for i in range(b.shape[0] - 1):
        right_i = b[i, 2]
        left_j = b[i + 1, 0]
        split = int(round((right_i + left_j) / 2.0))
        split = max(1, min(w - 1, split))
        xs.append(split)
    # Ensure ascending and unique
    xs = sorted(set(xs))
    return xs


def load_progress(progress_path: str) -> Dict:
    if os.path.isfile(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(progress_path: str, data: Dict):
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def draw_boxes_and_save(image: np.ndarray, boxes_xyxy: np.ndarray, output_path: str, text: str = ""):
    """
    在图片上绘制检测框并保存推理结果图片
    """
    # 创建图片副本用于绘制
    vis_img = image.copy()

    # 为每个检测框绘制矩形
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box.astype(int)
        # 绘制绿色矩形框
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 保存推理结果图片
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, vis_img)


def save_inference_results(output_root: str, folder_name: str, image: np.ndarray, 
                          boxes_xyxy: np.ndarray, line_type: str, text: str):
    """
    保存YOLO推理结果图片到指定文件夹
    """
    # 直接保存到文件夹下，不创建inference_results子文件夹
    folder_dir = os.path.join(output_root, folder_name)
    os.makedirs(folder_dir, exist_ok=True)

    # 构建文件名
    inference_filename = f"{line_type}_inference.jpg"
    inference_path = os.path.join(folder_dir, inference_filename)

    # 绘制检测框并保存
    draw_boxes_and_save(image, boxes_xyxy, inference_path, text)


def detect_char_boxes(model: "YOLO", image: np.ndarray, conf: float = DEFAULT_CONF_THRESHOLD, iou: float = DEFAULT_IOU_THRESHOLD) -> np.ndarray:
    """
    Run YOLO detection and return an array of N x 4 boxes in xyxy (float) for the character class (assumed class 0).
    """
    results = model.predict(source=image, conf=conf, iou=iou, verbose=False)
    if not results:
        return np.zeros((0, 4), dtype=np.float32)
    res = results[0]
    if not hasattr(res, 'boxes') or res.boxes is None or res.boxes.xyxy is None:
        return np.zeros((0, 4), dtype=np.float32)
    xyxy = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    # If multi-class, filter by class 0 (assumed char class). If no cls, keep all.
    try:
        cls = res.boxes.cls.detach().cpu().numpy().astype(np.int32)
        mask = (cls == 0)
        xyxy = xyxy[mask]
    except Exception:
        pass
    # sort by x-center
    if xyxy.shape[0] > 0:
        centers = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
        order = np.argsort(centers)
        xyxy = xyxy[order]
    return xyxy


def process_image_yolo(model: "YOLO", img_path: str, known_text: str, conf: float = DEFAULT_CONF_THRESHOLD, iou: float = DEFAULT_IOU_THRESHOLD) -> Tuple[bool, List[np.ndarray], np.ndarray, np.ndarray]:
    img = load_image(img_path)
    h, w = img.shape[:2]
    boxes_xyxy = detect_char_boxes(model, img, conf=conf, iou=iou)

    if boxes_xyxy.shape[0] != len(known_text):
        print(f"[WARN] Detected {boxes_xyxy.shape[0]} boxes, but expected {len(known_text)} for '{os.path.basename(img_path)}'. Skipping.")
        return False, [], img, boxes_xyxy

    splits = compute_splits_from_boxes(w, boxes_xyxy)
    # Using manual_split's crop_by_splits to get full-height slices
    crops = crop_by_splits(img, splits)
    if len(crops) != len(known_text):
        print(f"[WARN] Computed {len(crops)} crops after splits, expected {len(known_text)}. Skipping.")
        return False, [], img, boxes_xyxy
    return True, crops, img, boxes_xyxy


# 线程安全的进度保存锁
progress_lock = threading.Lock()


def save_progress_thread_safe(progress_path: str, data: Dict):
    """线程安全的进度保存函数"""
    with progress_lock:
        save_progress(progress_path, data)


def process_single_folder(args_tuple):
    """处理单个文件夹的函数，用于多线程"""
    folder, input_root, output_root, texts, model, conf, iou, save_inference, progress_path, processed = args_tuple

    sub_in = os.path.join(input_root, folder)

    # 找到所有line图片
    line_images = find_line_images(sub_in)

    if not line_images:
        print(f"[WARN] No line_XX_img.jpg files found in {sub_in}, skipping folder.")
        return folder, {}

    # 确保每个文本对应字符的目录存在
    for text in texts:
        ensure_char_dirs(os.path.join(output_root, folder), text)

    # 获取当前状态
    with progress_lock:
        state = processed.get(folder, {})

    # 处理每一行图片
    for line_num, filename in line_images:
        line_key = f"line_{line_num:02d}"

        if state.get(line_key, False):
            continue  # 已经处理过

        image_path = os.path.join(sub_in, filename)

        # 确定使用哪个文本（循环使用可用的文本）
        text_index = (line_num - 1) % len(texts)
        current_text = texts[text_index]

        if not os.path.isfile(image_path):
            print(f"[WARN] Missing {filename} in {sub_in}, skipping this line.")
            state[line_key] = True
            continue

        try:
            ok, crops, img, boxes = process_image_yolo(model, image_path, current_text, conf=conf, iou=iou)
            if ok:
                save_crops(output_root, folder, crops, current_text)
                state[line_key] = True
                print(f"[PROCESSED] {folder}/{filename} with text '{current_text}'")
                # 保存推理结果图片
                if save_inference:
                    line_type = f"line_{line_num:02d}"
                    save_inference_results(output_root, folder, img, boxes, line_type, current_text)
            else:
                # 即使失败也可能要保存推理结果用于调试
                if save_inference and boxes.shape[0] > 0:
                    line_type = f"line_{line_num:02d}"
                    save_inference_results(output_root, folder, img, boxes, line_type, current_text)
                print(f"[INFO] Skipped {filename} for {folder} due to detection/count mismatch.")
        except Exception as e:
            print(f"[ERROR] Failed to process {filename} for {folder}: {e}")

        # 更新进度
        with progress_lock:
            processed[folder] = state
            progress = {"processed": processed}
            save_progress(progress_path, progress)

    # 总结这个文件夹的处理结果
    completed_lines = [k for k, v in state.items() if v and k.startswith('line_')]
    print(f"[DONE] {folder}: completed {len(completed_lines)} lines: {', '.join(completed_lines)}")
    return folder, state


def main():
    parser = argparse.ArgumentParser(description="Automated character splitter using YOLOv11n detections; outputs identical structure to manual_split per-character crops")
    parser.add_argument('--input', '-i', default=DEFAULT_INPUT_PATH, help='Input root directory containing subfolders')
    parser.add_argument('--output', '-o', default=DEFAULT_OUTPUT_PATH, help='Output root directory to save split results (same as manual_split)')
    parser.add_argument('--texts', default=','.join(DEFAULT_TEXTS), help='Comma-separated known texts for line images (e.g., "M220,A291,D7")')
    parser.add_argument('--weights', '-w', default=DEFAULT_WEIGHTS_PATH, help='Path to YOLOv11n weights (best.pt)')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF_THRESHOLD, help='Confidence threshold for YOLO inference')
    parser.add_argument('--iou', type=float, default=DEFAULT_IOU_THRESHOLD, help='NMS IoU threshold for YOLO inference')
    parser.add_argument('--start-folder', default=None, help='Optional: start processing from this subfolder name')
    parser.add_argument('--save-inference', action='store_true', default=SAVE_INFERENCE_RESULTS, help='Save YOLO inference result images with detection boxes')
    parser.add_argument('--threads', '-t', type=int, default=DEFAULT_THREADS, help='Number of threads for parallel processing')
    args = parser.parse_args()

    input_root = args.input
    output_root = args.output
    texts = [text.strip() for text in args.texts.split(',') if text.strip()]
    weights_path = args.weights

    if YOLO is None:
        print("[ERROR] ultralytics is not installed. Please install it (e.g., pip install ultralytics) and retry.")
        sys.exit(1)

    if not os.path.isdir(input_root):
        print(f"Input directory does not exist: {input_root}")
        sys.exit(1)

    if not os.path.isfile(weights_path):
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    # Prepare output dirs: ensure per-char directories exist to match manual_split behavior
    os.makedirs(output_root, exist_ok=True)
    # Progress tracking
    progress_path = os.path.join(output_root, PROGRESS_FILENAME)
    progress = load_progress(progress_path)
    processed: Dict[str, Dict[str, bool]] = progress.get('processed', {})

    # Create model
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model from {weights_path}: {e}")
        sys.exit(1)

    folders = list_subfolders(input_root)
    if args.start_folder and args.start_folder in folders:
        start_index = folders.index(args.start_folder)
        folders = folders[start_index:]

    print(f"Processing {len(folders)} folders using {args.threads} threads...")

    # 准备多线程参数
    thread_args = [
        (folder, input_root, output_root, texts, model, 
         args.conf, args.iou, args.save_inference, progress_path, processed)
        for folder in folders
    ]

    # 使用多线程处理
    completed_folders = 0
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(process_single_folder, args) for args in thread_args]

        for future in futures:
            try:
                folder_name, state = future.result()
                completed_folders += 1
                print(f"[PROGRESS] Completed {completed_folders}/{len(folders)} folders")
            except Exception as e:
                print(f"[ERROR] Thread execution failed: {e}")
                completed_folders += 1

    print("All eligible folders processed.")


if __name__ == '__main__':
    main()
