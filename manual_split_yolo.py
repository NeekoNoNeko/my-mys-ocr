import os
import sys
import json
import argparse
import shutil
from typing import List, Tuple, Dict

import cv2
import numpy as np

"""
水平文本图像手工字符切分工具（YOLO 标注器）。

这是一个独立脚本，可通过交互方式在字符之间插入竖直切分线，
并输出 YOLO 格式的 .txt 标签，仅包含一个类别（“char” -> 类别 id 0）。
它不依赖也不修改 manual_split.py。

功能说明：
- 递归遍历输入根目录的所有子目录，每个子目录应包含：
    line_01_img.jpg、line_02_img.jpg、line_03_img.jpg 等。
- 对每张图像，先显示放大后的预览；用户点击竖直切分线，按 Enter 确认。
- 切分线会映射回原图分辨率（而非放大后的预览分辨率）。
- 默认已知文本：
    第 1 行：M220
    第 2 行：A291
    第 3 行：D7
- 生成 YOLO 格式的 .txt 标签，类别为 0（char）。每两条相邻切分线
  （含图像左右边界）之间的区域为一个字符框，框高为整图高度。
- 进度自动保存到输出根目录下的 progress.json，可随时中断后恢复。
- 任何时候都可按 'q' 直接退出，按 'u' 撤销上一次切分线，按 'r' 清空全部切分线，按 Enter 确认。

示例用法：
  python manual_split_yolo.py \
    --input tools/split_character/output \
    --output tools/split_character/labels \
    --texts M220,A291,D7 \
    --scale 2.0

注意事项：
- 程序会自动检测每个子目录中所有符合 line_XX_img.jpg 格式的文件并依次处理。
- 如果点击的切分线数量不等于 len(text)-1，会要求重新标注。
- YOLO .txt 每行格式： "0 x_center y_center width height"（均已归一化到 [0,1]）。
- 窗口会实时显示已放置的切分线；满意后按 Enter 确认即可。
"""

# ===== 默认配置参数 =====
DEFAULT_TEXTS = [
    "M2204063429",
    "AZ91D",
    "FT-1"
]
PROGRESS_FILENAME = "progress.json"

# python .\manual_split_yolo.py --input .\dataset\crnn\all\ --output .\dataset\crnn\split --filelist .\dataset\crnn\all\files_with_labels.txt --scale 8.0
# 命令行参数默认值
DEFAULT_INPUT_PATH = os.path.join('tools', 'split_character', 'output')
DEFAULT_OUTPUT_PATH = os.path.join('tools', 'split_character', 'labels')
DEFAULT_SCALE = 8.0
DEFAULT_TEXTS_STR = ','.join(DEFAULT_TEXTS)
DEFAULT_START_FOLDER = None
DEFAULT_GLOB = None
DEFAULT_FILELIST = None
DEFAULT_NO_COPY = False
DEFAULT_MIRROR_OUTPUT = False


class SplitSession:
    """交互式切分会话类"""
    def __init__(self, image: np.ndarray, display_scale: float = 2.0, window_name: str = "切分"):
        self.image = image
        self.scale = max(0.1, float(display_scale))
        self.window_name = window_name
        self.lines: List[int] = []  # 存储显示坐标系下的 x 坐标
        self.confirmed = False
        self.quit = False

        h, w = image.shape[:2]
        # 等比例缩放，保持宽高比
        self.display_size = (int(w * self.scale), int(h * self.scale))
        self.disp = cv2.resize(self.image, self.display_size, interpolation=cv2.INTER_CUBIC)

    def _draw(self) -> np.ndarray:
        """绘制切分线"""
        canvas = self.disp.copy()
        for x in self.lines:
            cv2.line(canvas, (x, 0), (x, canvas.shape[0] - 1), (0, 255, 255), 1)
        return canvas

    def _mouse(self, event, x, y, flags, param):
        """鼠标回调：左键点击添加竖直线"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.lines.append(x)
            self.lines.sort()

    def run(self) -> Tuple[bool, List[int]]:
        """启动交互窗口并等待用户操作"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse)
        while True:
            canvas = self._draw()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                self.quit = True
                self.confirmed = False
                break
            elif key == ord('u'):
                if self.lines:
                    self.lines.pop()
            elif key == ord('r'):
                self.lines.clear()
            elif key in (13, 10):  # Enter
                self.confirmed = True
                break
        cv2.destroyWindow(self.window_name)
        return self.confirmed and not self.quit, self.lines


def map_display_to_original(xs_disp: List[int], scale: float) -> List[int]:
    """将显示坐标映射回原始图像坐标"""
    return [max(0, int(round(x / scale))) for x in xs_disp]


def yolo_boxes_from_splits(image: np.ndarray, xs: List[int]) -> List[Tuple[float, float, float, float]]:
    """
    根据竖直切分线生成 YOLO 归一化框 (x_center, y_center, width, height)。
    框高始终为整图高度。
    """
    h, w = image.shape[:2]
    xs_sorted = sorted(set([x for x in xs if 0 < x < w]))
    edges = [0] + xs_sorted + [w]
    boxes: List[Tuple[float, float, float, float]] = []
    for i in range(len(edges) - 1):
        x0, x1 = edges[i], edges[i + 1]
        if x1 <= x0:
            continue
        x_center = (x0 + x1) / 2.0
        y_center = h / 2.0
        bw = x1 - x0
        bh = h
        boxes.append((x_center / w, y_center / h, bw / w, bh / h))
    return boxes


def save_yolo_labels(output_root: str, folder_name: str, image_basename: str,
                     boxes: List[Tuple[float, float, float, float]]):
    """保存 YOLO 标签（类别 0）"""
    base_dir = os.path.join(output_root, folder_name)
    os.makedirs(base_dir, exist_ok=True)
    label_path = os.path.join(base_dir, os.path.splitext(image_basename)[0] + '.txt')
    with open(label_path, 'w', encoding='utf-8') as f:
        for (xc, yc, w, h) in boxes:
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def copy_image_to_output(input_image_path: str, output_root: str, folder_name: str, image_basename: str):
    """将原图复制到输出目录（与标签同目录）"""
    base_dir = os.path.join(output_root, folder_name)
    os.makedirs(base_dir, exist_ok=True)
    output_image_path = os.path.join(base_dir, image_basename)
    shutil.copy2(input_image_path, output_image_path)


def load_image(path: str) -> np.ndarray:
    """读取图像"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return img


def list_subfolders(root: str) -> List[str]:
    """列出根目录下的所有子目录"""
    result = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            result.append(name)
    return result


def find_line_images(folder_path: str) -> List[Tuple[int, str]]:
    """查找目录中所有 line_XX_img.jpg 并返回 (行号, 文件名) 排序列表"""
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


def load_progress(progress_path: str) -> Dict:
    """读取进度文件"""
    if os.path.isfile(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(progress_path: str, data: Dict):
    """保存进度文件"""
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ===== 新增工具函数（filelist/glob 模式会用到） =====
from typing import Optional

def read_filelist(filelist_path: str) -> List[Tuple[str, str]]:
    """读取文件清单，每行格式：path,text 或 path\ttext。忽略空行与以 # 开头的注释行。"""
    pairs: List[Tuple[str, str]] = []
    with open(filelist_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            if ',' in s:
                p, t = s.split(',', 1)
            elif '\t' in s:
                p, t = s.split('\t', 1)
            else:
                # 不符合规则的行，跳过
                continue
            pairs.append((p.strip(), t.strip()))
    return pairs


def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_yolo_labels_by_target_path(label_path: str,
                                    boxes: List[Tuple[float, float, float, float]]):
    """直接按指定路径保存 YOLO 标签（类别恒为 0）。会确保父目录存在。"""
    ensure_parent_dir(label_path)
    with open(label_path, 'w', encoding='utf-8') as f:
        for (xc, yc, w, h) in boxes:
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def process_image_interactive(img_path: str, text: str, scale: float,
                              window_title: str) -> Tuple[bool, List[Tuple[float, float, float, float]]]:
    """单张图像交互处理入口"""
    img = load_image(img_path)
    session = SplitSession(img, display_scale=scale, window_name=window_title)
    confirmed, lines_disp = session.run()
    if session.quit:
        print("用户退出程序。")
        sys.exit(0)
    if not confirmed:
        return False, []
    xs = map_display_to_original(lines_disp, session.scale)
    if len(xs) != max(0, len(text) - 1):
        print(f"文本 '{text}' 需要 {len(text)-1} 条切分线，实际 {len(xs)} 条。按任意键继续...")
        cv2.waitKey(0)
        return False, []
    boxes = yolo_boxes_from_splits(img, xs)
    if len(boxes) != len(text):
        print(f"切分得到 {len(boxes)} 个框，期望 {len(text)} 个。按任意键继续...")
        cv2.waitKey(0)
        return False, []
    return True, boxes


def main():
    parser = argparse.ArgumentParser(description="水平文本图像手工切分工具（YOLO 单类别 'char'=0）")
    parser.add_argument('--input', '-i', default=os.path.join('tools', 'split_character', 'output'),
                        help='输入根目录，包含若干子目录')
    parser.add_argument('--output', '-o', default=os.path.join('tools', 'split_character', 'labels'),
                        help='输出根目录，用于存放 YOLO 标签 .txt')
    parser.add_argument('--texts', default=','.join(DEFAULT_TEXTS),
                        help='已知文本，用英文逗号分隔，如 "M220,A291,D7"')
    parser.add_argument('--scale', type=float, default=3.0,
                        help='预览图像的放大倍数')
    parser.add_argument('--start-folder', default=DEFAULT_START_FOLDER,
                        help='可选：从指定子目录名开始处理（目录模式）')
    # 新增：按通配符批量处理文件
    parser.add_argument('--glob', default=DEFAULT_GLOB,
                        help='可选：如 "C:\\data\\imgs\\*.jpg"，按排序批量处理这些文件')
    # 新增：按清单映射 文件路径→文本
    parser.add_argument('--filelist', default=DEFAULT_FILELIST,
                        help='可选：CSV/TXT 文件，每行格式 "relative_or_abs_path,text" 或以制表符分隔')
    # 新增：只写标签，不复制图片
    parser.add_argument('--no-copy', action='store_true', default=DEFAULT_NO_COPY,
                        help='只写标签，不复制图片到输出目录')
    # 新增：在输出目录中镜像输入相对路径
    parser.add_argument('--mirror-output', action='store_true', default=DEFAULT_MIRROR_OUTPUT,
                        help='在输出目录中按输入相对路径镜像创建子目录')
    args = parser.parse_args()

    input_root = args.input
    output_root = args.output
    texts = [t.strip() for t in args.texts.split(',') if t.strip()]
    scale = args.scale

    if not os.path.isdir(input_root):
        print(f"输入目录不存在: {input_root}")
        sys.exit(1)

    os.makedirs(output_root, exist_ok=True)
    progress_path = os.path.join(output_root, PROGRESS_FILENAME)
    progress = load_progress(progress_path)
    processed: Dict[str, Dict[str, bool]] = progress.get('processed', {})

    folders = list_subfolders(input_root)
    if args.start_folder and args.start_folder in folders:
        start_index = folders.index(args.start_folder)
        folders = folders[start_index:]

    print("操作提示：\n"
          " - 在字符之间点击竖直切分线。\n"
          " - 按 Enter 确认并完成切分。\n"
          " - 'u' 撤销上一步；'r' 清空全部；'q' 退出。\n")

    # ========== 新增分支：glob 模式 ==========
    if args.glob:
        import glob as _glob
        files = sorted(_glob.glob(args.glob))
        if not files:
            print(f"[警告] 未匹配到文件: {args.glob}")
            return
        progress_path = os.path.join(output_root, PROGRESS_FILENAME)
        progress = load_progress(progress_path)
        processed_glob = progress.get('processed_glob', {})

        # 作为 mirror-output 的基准目录（glob 的目录部分）
        base_dir = os.path.dirname(args.glob.rstrip('*').rstrip('?'))

        for i, img_path in enumerate(files, start=1):
            if os.path.isdir(base_dir):
                rel_key = os.path.relpath(img_path, base_dir)
            else:
                rel_key = os.path.basename(img_path)
            if processed_glob.get(rel_key, False):
                continue

            current_text = texts[(i - 1) % len(texts)] if texts else ''
            ok, boxes = process_image_interactive(
                img_path, current_text, scale, window_title=os.path.basename(img_path)
            )
            if ok:
                if args.mirror_output and os.path.isdir(base_dir):
                    label_rel = os.path.splitext(rel_key)[0] + '.txt'
                    label_out = os.path.join(output_root, label_rel)
                else:
                    label_out = os.path.join(output_root, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
                save_yolo_labels_by_target_path(label_out, boxes)

                if not args.no_copy:
                    out_img_path = (os.path.join(output_root, rel_key)
                                    if (args.mirror_output and os.path.isdir(base_dir))
                                    else os.path.join(output_root, os.path.basename(img_path)))
                    ensure_parent_dir(out_img_path)
                    shutil.copy2(img_path, out_img_path)

                processed_glob[rel_key] = True
                print(f"[已完成] {rel_key} 文本: '{current_text}'")
            else:
                print(f"[未确认] {rel_key}，下次可继续。")
                processed_glob[rel_key] = False

            progress['processed_glob'] = processed_glob
            save_progress(progress_path, progress)
        print("glob 匹配到的项已完成。")
        return

    # ========== 新增分支：filelist 模式 ==========
    if args.filelist:
        pairs = read_filelist(args.filelist)
        if not pairs:
            print(f"[警告] filelist 为空或不可用: {args.filelist}")
            return
        progress_path = os.path.join(output_root, PROGRESS_FILENAME)
        progress = load_progress(progress_path)
        processed_filelist = progress.get('processed_filelist', {})

        print("操作提示：\n"
              " - 在字符之间点击竖直切分线。\n"
              " - 按 Enter 确认并完成切分。\n"
              " - 'u' 撤销上一步；'r' 清空全部；'q' 退出。\n")

        # 当 filelist 中给出相对路径时的基准目录
        base_dir = input_root if os.path.isdir(input_root) else os.getcwd()

        for idx, (rel_or_abs_path, text) in enumerate(pairs):
            img_path = rel_or_abs_path
            if not os.path.isabs(img_path):
                img_path = os.path.join(base_dir, rel_or_abs_path)
            img_path = os.path.normpath(img_path)

            key = rel_or_abs_path  # 以清单中的原始字符串作为键，便于断点续作
            if processed_filelist.get(key, False):
                continue

            if not os.path.isfile(img_path):
                print(f"[警告] 找不到图像: {img_path}，跳过。")
                processed_filelist[key] = True
                progress['processed_filelist'] = processed_filelist
                save_progress(progress_path, progress)
                continue

            ok, boxes = process_image_interactive(
                img_path, text, scale, window_title=os.path.basename(img_path)
            )
            if ok:
                if args.mirror_output:
                    try:
                        rel_to_base = os.path.relpath(img_path, base_dir)
                    except ValueError:
                        rel_to_base = os.path.basename(img_path)
                    label_rel = os.path.splitext(rel_to_base)[0] + '.txt'
                    label_out = os.path.join(output_root, label_rel)
                else:
                    label_name = os.path.splitext(os.path.basename(img_path))[0] + '.txt'
                    label_out = os.path.join(output_root, label_name)

                save_yolo_labels_by_target_path(label_out, boxes)

                if not args.no_copy:
                    if args.mirror_output:
                        out_img_path = os.path.join(output_root, rel_to_base)
                    else:
                        out_img_path = os.path.join(output_root, os.path.basename(img_path))
                    ensure_parent_dir(out_img_path)
                    shutil.copy2(img_path, out_img_path)

                processed_filelist[key] = True
                print(f"[已完成] {rel_or_abs_path} 文本: '{text}'")
            else:
                print(f"[未确认] {rel_or_abs_path}，下次可继续。")
                processed_filelist[key] = False

            progress['processed_filelist'] = processed_filelist
            save_progress(progress_path, progress)

        print("按清单的所有可处理项已完成。")
        return

    for folder in folders:
        sub_in = os.path.join(input_root, folder)
        line_images = find_line_images(sub_in)
        if not line_images:
            print(f"[警告] {sub_in} 中未找到 line_XX_img.jpg，跳过该目录。")
            continue

        state = processed.get(folder, {})
        should_exit = False
        for line_num, filename in line_images:
            line_key = f"line_{line_num:02d}"
            if state.get(line_key, False):
                continue  # 已处理

            image_path = os.path.join(sub_in, filename)
            text_index = (line_num - 1) % len(texts)
            current_text = texts[text_index]

            if not os.path.isfile(image_path):
                print(f"[警告] 缺少 {filename}，跳过该行。")
                state[line_key] = True
                continue

            ok, boxes = process_image_interactive(
                image_path, current_text, scale,
                window_title=f"{folder} - {filename}"
            )

            if ok:
                save_yolo_labels(output_root, folder, filename, boxes)
                copy_image_to_output(image_path, output_root, folder, filename)
                state[line_key] = True
                print(f"[已完成] {folder}/{filename} 文本: '{current_text}'")
            else:
                print(f"第 {line_num} 行未确认。若按 'q' 则退出，否则下次重试。")
                processed[folder] = state
                progress['processed'] = processed
                save_progress(progress_path, progress)
                should_exit = True
                break

            # 每行处理完后即时保存进度
            processed[folder] = state
            progress['processed'] = processed
            save_progress(progress_path, progress)

        if should_exit:
            return

        completed_lines = [k for k, v in state.items() if v and k.startswith('line_')]
        print(f"[目录完成] {folder}: 共处理 {len(completed_lines)} 行: {', '.join(completed_lines)}")

    print("所有可处理目录已完成。")


if __name__ == '__main__':
    main()