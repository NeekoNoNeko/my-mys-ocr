import argparse
import cv2
import os
import json
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# ========== 配置参数 ==========
# 默认匹配阈值
DEFAULT_THRESHOLD = 0.7

def load_offsets_config(config_path: str = "offsets_config.json") -> List[dict]:
    """
    从JSON文件加载偏移量配置
    """
    config_file = Path(config_path)
    if not config_file.exists():
        # 如果配置文件不存在，使用默认配置
        print(f"[WARN] 配置文件 {config_path} 不存在，使用默认配置")
        return [
            {
                'name': 'first_line',
                'x0': -63,
                'y0': 1,
                'x1': -157,
                'y1': -38
            },
            {
                'name': 'second_line',
                'x0': -73,
                'y0': 20,
                'x1': -166,
                'y1': -18
            },
            {
                'name': 'third_line',
                'x0': -66,
                'y0': 38,
                'x1': -169,
                'y1': -1
            }
        ]

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = json.load(f)

        # 将JSON格式转换为原有的列表格式
        offsets_list = []
        for name, offsets in config_data.items():
            offset_config = {
                'name': name,
                'x0': offsets['x0'],
                'y0': offsets['y0'],
                'x1': offsets['x1'],
                'y1': offsets['y1']
            }
            offsets_list.append(offset_config)

        print(f"[INFO] 已从 {config_path} 加载 {len(offsets_list)} 个偏移量配置")
        return offsets_list

    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON文件格式错误: {e}")
        print(f"[INFO] 使用默认配置")
        return [
            {
                'name': 'first_line',
                'x0': -63,
                'y0': 1,
                'x1': -157,
                'y1': -38
            },
            {
                'name': 'second_line',
                'x0': -73,
                'y0': 20,
                'x1': -166,
                'y1': -18
            },
            {
                'name': 'third_line',
                'x0': -66,
                'y0': 38,
                'x1': -169,
                'y1': -1
            }
        ]
    except Exception as e:
        print(f"[ERROR] 读取配置文件时发生错误: {e}")
        print(f"[INFO] 使用默认配置")
        return [
            {
                'name': 'first_line',
                'x0': -63,
                'y0': 1,
                'x1': -157,
                'y1': -38
            },
            {
                'name': 'second_line',
                'x0': -73,
                'y0': 20,
                'x1': -166,
                'y1': -18
            },
            {
                'name': 'third_line',
                'x0': -66,
                'y0': 38,
                'x1': -169,
                'y1': -1
            }
        ]

# 多行文本框偏移量配置 (从JSON文件加载)
TEXT_LINES_OFFSETS = load_offsets_config()

# 默认支持的图像文件扩展名
DEFAULT_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "bmp", "tiff", "tga")

# YOLO标签类别ID
DEFAULT_CLASS_ID = 1
"""
0 顺时针90度
1 不变
2 逆时针90度
3 180度
"""

# 最小框尺寸限制
MIN_BOX_SIZE = 1

# 图像旋转设置 (设置为None表示不旋转)
# ROTATION_MODE = None
# ROTATION_MODE = cv2.ROTATE_90_CLOCKWISE
# ROTATION_MODE = cv2.ROTATE_90_COUNTERCLOCKWISE
ROTATION_MODE = cv2.ROTATE_180

# 输出文件命名后缀 (动态生成基于TEXT_LINES_OFFSETS)
OUTPUT_SUFFIXES = {
    line_config['name']: f"_{line_config['name']}.jpg" 
    for line_config in TEXT_LINES_OFFSETS
}
OUTPUT_SUFFIXES.update({
    'visualization': '_viz.jpg',
    'yolo_labels': '.txt'
})

# 输出控制选项 (动态生成基于TEXT_LINES_OFFSETS)
SAVE_OPTIONS = {
    f"save_{line_config['name']}": False  # 默认不保存裁剪图像
    for line_config in TEXT_LINES_OFFSETS
}
SAVE_OPTIONS.update({
    'save_yolo_labels': True,      # 是否保存YOLO标签文件
    'save_visualization': False,   # 是否保存可视化图像
})

# 多线程设置
THREADING_CONFIG = {
    'max_workers': 4,  # 默认线程数，0表示使用CPU核心数
    'enable_progress': True,  # 是否显示进度信息
}

# 命令行参数默认值 (动态生成基于配置)
CMD_DEFAULTS = {
    'input_dir': "..\\828\\E2",  # 输入目录 (必需参数，无默认值)
    'output_dir': "..\\828\\E2",  # 输出目录 (必需参数，无默认值)
    'template': "template.jpg",   # 模板图片路径 (必需参数，无默认值)
    'threshold': DEFAULT_THRESHOLD,  # 匹配阈值
    'extensions': ",".join(DEFAULT_IMAGE_EXTENSIONS),  # 支持的图像扩展名
    'save_yolo_labels': SAVE_OPTIONS['save_yolo_labels'],  # 是否保存YOLO标签
    'save_visualization': SAVE_OPTIONS['save_visualization'],  # 是否保存可视化图像
    'max_workers': THREADING_CONFIG['max_workers'],  # 最大线程数
    'show_progress': THREADING_CONFIG['enable_progress'],  # 是否显示进度
}
# 动态添加每行的保存选项
for line_config in TEXT_LINES_OFFSETS:
    line_name = line_config['name']
    CMD_DEFAULTS[f'save_{line_name}'] = SAVE_OPTIONS[f'save_{line_name}']
# ========== 配置参数结束 ==========


def load_grayscale(path: Path) -> Optional[any]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def match_template(img, template, threshold: float = 0.8) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """
    Returns (top_left, bottom_right, confidence) if found else None
    """
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val >= threshold:
        th, tw = template.shape
        top_left = max_loc
        bottom_right = (max_loc[0] + tw, max_loc[1] + th)
        return top_left, bottom_right, max_val
    return None


def clamp_box(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    # ensure top-left <= bottom-right
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return x0, y0, x1, y1


def compute_line_boxes(top_left: Tuple[int, int], bottom_right: Tuple[int, int], text_lines_offsets=None):
    """使用配置参数计算多行文本框的位置"""
    if text_lines_offsets is None:
        text_lines_offsets = TEXT_LINES_OFFSETS

    tlx, tly = top_left
    brx, bry = bottom_right

    line_boxes = []
    for line_offset in text_lines_offsets:
        line_tl = [tlx + line_offset['x0'], tly + line_offset['y0']]
        line_br = [brx + line_offset['x1'], bry + line_offset['y1']]
        line_boxes.append((line_tl, line_br))

    return line_boxes


def save_yolo_labels(labels: List[str], out_txt: Path):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open('w', encoding='utf-8') as f:
        f.write('\n'.join(labels))


def to_yolo_labels(boxes: List[Tuple[int, int, int, int]], img_w: int, img_h: int, cls: int = DEFAULT_CLASS_ID) -> List[str]:
    labels: List[str] = []
    for (x0, y0, x1, y1) in boxes:
        w = max(MIN_BOX_SIZE, x1 - x0)
        h = max(MIN_BOX_SIZE, y1 - y0)
        x = x0
        y = y0
        x = max(0, min(x, img_w))
        y = max(0, min(y, img_h))
        w = max(MIN_BOX_SIZE, min(w, img_w - x))
        h = max(MIN_BOX_SIZE, min(h, img_h - y))
        xc = x + w / 2
        yc = y + h / 2
        xc_norm = xc / img_w
        yc_norm = yc / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        labels.append(f"{cls} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
    return labels


# 线程安全的打印锁
print_lock = threading.Lock()

def thread_safe_print(message: str):
    """线程安全的打印函数"""
    with print_lock:
        print(message)

def process_image(img_path: Path, template, threshold: float, out_dir: Path, 
                 save_lines: dict = None, save_labels: bool = True, 
                 save_viz: bool = False, verbose: bool = True, 
                 text_lines_offsets=None) -> bool:
    img = load_grayscale(img_path)
    if img is None:
        if verbose:
            thread_safe_print(f"[WARN] Unable to read image: {img_path}")
        return False

    h_img, w_img = img.shape[:2]
    match = match_template(img, template, threshold)
    if match is None:
        if verbose:
            thread_safe_print(f"[INFO] No match (<{threshold}) for: {img_path.name}")
        return False

    top_left, bottom_right, conf = match

    # 如果save_lines为None，默认不保存任何行
    if save_lines is None:
        save_lines = {}

    # 使用传入的offsets配置或默认配置
    if text_lines_offsets is None:
        text_lines_offsets = TEXT_LINES_OFFSETS

    # 生成对应的输出后缀配置
    output_suffixes = {
        line_config['name']: f"_{line_config['name']}.jpg" 
        for line_config in text_lines_offsets
    }
    output_suffixes.update({
        'visualization': '_viz.jpg',
        'yolo_labels': '.txt'
    })

    # compute boxes for all lines
    line_boxes = compute_line_boxes(top_left, bottom_right, text_lines_offsets)

    # process each line
    line_crops = []
    yolo_boxes = []
    saved_files = []

    stem = img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (line_config, (line_tl, line_br)) in enumerate(zip(text_lines_offsets, line_boxes)):
        line_name = line_config['name']

        # clamp to bounds
        x0, y0, x1, y1 = clamp_box(line_tl[0], line_tl[1], line_br[0], line_br[1], w_img, h_img)

        # crop and rotate
        line_crop = img[y0:y1, x0:x1]

        if line_crop.size == 0:
            if verbose:
                thread_safe_print(f"[WARN] Empty crop for {line_name} in: {img_path.name}")
            continue

        # 只有设置了旋转模式才进行旋转
        if ROTATION_MODE is not None:
            line_crop = cv2.rotate(line_crop, ROTATION_MODE)
        line_crops.append((line_name, line_crop))

        # 保存裁剪图像（如果需要）
        if save_lines.get(line_name, False):
            line_out = out_dir / f"{stem}{output_suffixes[line_name]}"
            cv2.imwrite(str(line_out), line_crop)
            saved_files.append(line_out.name)

        # 添加到YOLO框列表
        yolo_boxes.append((x0, y0, x1, y1))

    # 检查是否有有效的裁剪
    if not line_crops:
        if verbose:
            thread_safe_print(f"[WARN] No valid crops for: {img_path.name}")
        return False

    # 可选保存YOLO标签文件
    if save_labels and yolo_boxes:
        labels = to_yolo_labels(yolo_boxes, w_img, h_img, cls=DEFAULT_CLASS_ID)
        yolo_out = out_dir / f"{stem}{output_suffixes['yolo_labels']}"
        save_yolo_labels(labels, yolo_out)
        saved_files.append(yolo_out.name)

    # 可选保存可视化图像
    if save_viz:
        # draw rectangles on original for visualization
        rect_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        # 绘制模板匹配框 (红色)
        cv2.rectangle(rect_img, top_left, bottom_right, (0, 0, 255), 2)

        # 为每一行绘制不同颜色的框
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        for i, (x0, y0, x1, y1) in enumerate(yolo_boxes):
            color = colors[i % len(colors)]
            cv2.rectangle(rect_img, (x0, y0), (x1, y1), color, 1)
            # 添加行标签
            line_name = text_lines_offsets[i]['name']
            cv2.putText(rect_img, line_name, (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        viz_out = out_dir / f"{stem}{output_suffixes['visualization']}"
        cv2.imwrite(str(viz_out), rect_img)
        saved_files.append(viz_out.name)

    if verbose:
        if saved_files:
            thread_safe_print(f"[OK] {img_path.name}: conf={conf:.4f} -> saved {', '.join(saved_files)}")
        else:
            thread_safe_print(f"[OK] {img_path.name}: conf={conf:.4f} -> no files saved (all options disabled)")
    return True


def find_images(input_dir: Path, exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(input_dir.rglob(f"*.{ext}"))
        files.extend(input_dir.rglob(f"*.{ext.upper()}"))
    # unique and sort
    files = sorted(set(files))
    return files


def main():
    parser = argparse.ArgumentParser(description="Batch template match, crop two lines, and export YOLO labels.")

    # 添加配置文件参数
    parser.add_argument("--config", type=str, default="offsets_config.json", help="Path to offsets configuration JSON file (default: offsets_config.json)")

    # 使用配置参数设置命令行参数默认值
    if CMD_DEFAULTS['input_dir'] is None:
        parser.add_argument("--input-dir", type=str, required=True, help="Input directory containing images")
    else:
        parser.add_argument("--input-dir", type=str, default=CMD_DEFAULTS['input_dir'], help=f"Input directory containing images (default: {CMD_DEFAULTS['input_dir']})")

    if CMD_DEFAULTS['output_dir'] is None:
        parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated images and labels (single folder)")
    else:
        parser.add_argument("--output-dir", type=str, default=CMD_DEFAULTS['output_dir'], help=f"Directory to save generated images and labels (default: {CMD_DEFAULTS['output_dir']})")

    if CMD_DEFAULTS['template'] is None:
        parser.add_argument("--template", type=str, required=True, help="Path to template image (grayscale)")
    else:
        parser.add_argument("--template", type=str, default=CMD_DEFAULTS['template'], help=f"Path to template image (default: {CMD_DEFAULTS['template']})")

    parser.add_argument("--threshold", type=float, default=CMD_DEFAULTS['threshold'], help=f"Matching threshold (default: {CMD_DEFAULTS['threshold']})")
    parser.add_argument("--exts", type=str, default=CMD_DEFAULTS['extensions'], help=f"Comma-separated image extensions (default: {CMD_DEFAULTS['extensions']})")

    # 动态生成每行的保存选项参数
    for line_config in TEXT_LINES_OFFSETS:
        line_name = line_config['name']
        default_val = CMD_DEFAULTS[f'save_{line_name}']
        parser.add_argument(f"--save-{line_name.replace('_', '-')}", 
                          action="store_true", default=default_val,
                          help=f"Save {line_name.replace('_', ' ')} cropped images (default: {default_val})")
        parser.add_argument(f"--no-save-{line_name.replace('_', '-')}", 
                          action="store_false", dest=f"save_{line_name}",
                          help=f"Do not save {line_name.replace('_', ' ')} cropped images")

    parser.add_argument("--save-labels", action="store_true", default=CMD_DEFAULTS['save_yolo_labels'], help=f"Save YOLO label files (default: {CMD_DEFAULTS['save_yolo_labels']})")
    parser.add_argument("--no-save-labels", action="store_false", dest="save_labels", help="Do not save YOLO label files")

    if CMD_DEFAULTS['save_visualization']:
        parser.add_argument("--save-viz", action="store_true", default=True, help=f"Also save visualization image with boxes (default: {CMD_DEFAULTS['save_visualization']})")
        parser.add_argument("--no-save-viz", action="store_false", dest="save_viz", help="Do not save visualization image with boxes")
    else:
        parser.add_argument("--save-viz", action="store_true", default=CMD_DEFAULTS['save_visualization'], help="Also save visualization image with boxes")

    # 多线程参数
    parser.add_argument("--workers", type=int, default=CMD_DEFAULTS['max_workers'], help=f"Number of worker threads (0 = auto, default: {CMD_DEFAULTS['max_workers']})")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--progress", action="store_true", default=CMD_DEFAULTS['show_progress'], help=f"Show progress information (default: {CMD_DEFAULTS['show_progress']})")

    args = parser.parse_args()

    # 加载配置（如果指定了不同的配置文件）
    current_offsets = TEXT_LINES_OFFSETS
    if args.config != "offsets_config.json":
        current_offsets = load_offsets_config(args.config)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    template_path = Path(args.template)

    if not input_dir.exists():
        raise SystemExit(f"Input dir not found: {input_dir}")
    if not template_path.exists():
        raise SystemExit(f"Template not found: {template_path}")

    template = load_grayscale(template_path)
    if template is None:
        raise SystemExit(f"Failed to read template as grayscale: {template_path}")

    exts = tuple([e.strip().lstrip('.').lower() for e in args.exts.split(',') if e.strip()])
    images = find_images(input_dir, exts)
    if not images:
        print(f"[INFO] No images found in {input_dir} with extensions: {exts}")
        return

    # 确定线程数
    max_workers = args.workers if args.workers > 0 else None
    verbose_mode = not args.quiet

    print(f"[INFO] Found {len(images)} images. Processing with {max_workers if max_workers else 'auto'} threads...")

    start_time = time.time()
    ok = 0
    total = len(images)

    def process_single_image(img_path):
        """处理单个图像的包装函数"""
        try:
            # 构建保存行配置字典
            save_lines = {}
            for line_config in current_offsets:
                line_name = line_config['name']
                save_lines[line_name] = getattr(args, f'save_{line_name}', False)

            success = process_image(
                img_path, template, args.threshold, output_dir,
                save_lines=save_lines, save_labels=args.save_labels, 
                save_viz=args.save_viz, verbose=verbose_mode,
                text_lines_offsets=current_offsets
            )
            return img_path, success, None
        except Exception as e:
            return img_path, False, e

    # 使用多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_path = {executor.submit(process_single_image, img_path): img_path 
                         for img_path in images}

        # 处理完成的任务
        for i, future in enumerate(as_completed(future_to_path), 1):
            img_path, success, error = future.result()

            if success:
                ok += 1
            elif error:
                thread_safe_print(f"[ERROR] {img_path}: {error}")

            # 显示进度
            if args.progress and (i % 10 == 0 or i == total):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                thread_safe_print(f"[PROGRESS] {i}/{total} ({i/total*100:.1f}%) - "
                                f"Rate: {rate:.1f} img/s - ETA: {eta:.0f}s")

    elapsed_total = time.time() - start_time
    print(f"[DONE] Success: {ok}/{len(images)} in {elapsed_total:.1f}s "
          f"({len(images)/elapsed_total:.1f} img/s). Outputs saved in: {output_dir}")


if __name__ == "__main__":
    main()
