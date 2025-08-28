# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
from typing import List, Tuple, Dict

import cv2
import numpy as np

"""
水平排列文本图像的手动字符切分工具。

功能说明：
- 递归遍历输入根目录下所有子文件夹，每个子文件夹应包含：
    - debug_img.jpg（可选，本脚本不使用）
    - first_line_img.jpg
    - second_line_img.jpg
- 针对每张图像，先以缩放后的尺寸展示给用户；用户点击竖直分割线，按 Enter 确认并完成切分。
- 分割线坐标会映射回原图分辨率，确保裁剪精度。
- 默认已知文本：
    第一行：M2204063429
    第二行：AZ91DE7
- 将每个字符小图保存至输出根目录下的 split/<子文件夹名>/<字符>/<字符-N>.jpg
- 重复字符会递增编号，如 2-1.jpg、2-2.jpg。
- 使用 output_root/progress.json 记录处理进度，可随时中断后恢复。
- 命令行参数可指定输入根目录、输出根目录、文本内容、显示缩放倍数及起始子文件夹。
- 任意时刻可按 'q' 优雅退出；按 'u' 撤销上一条分割线；按 'r' 清空所有分割线。

使用示例：
  python tools/split_character/manual_split.py \
    --input tools/split_character/output \
    --output tools/split_character/split \
    --first-text M2204063429 \
    --second-text AZ91DE7 \
    --scale 2.0

注意：
- 脚本假设每个子文件夹中的两张图均包含与指定文本完全匹配的单行水平文本。
- 如果用户点击的分割线数量不等于 len(text)-1，将提示重试。
- 窗口实时显示操作指引与已放置分割线；确认无误后按 Enter。
"""

DEFAULT_FIRST = "M2204063429"      # 默认第一行文本
DEFAULT_SECOND = "AZ91DE7"         # 默认第二行文本
PROGRESS_FILENAME = "progress.json"  # 进度文件名


class SplitSession:
    """
    单张图像的交互式切分会话
    """
    def __init__(self, image: np.ndarray, display_scale: float = 2.0, window_name: str = "Split"):
        self.image = image              # 原图
        self.scale = max(0.1, float(display_scale))  # 显示倍数，防止过小
        self.window_name = window_name  # 窗口名称
        self.lines: List[int] = []      # 记录显示坐标系下的分割线 x 坐标
        self.confirmed = False          # 是否已确认
        self.quit = False               # 用户是否按 q 退出
        self.reset = False              # 是否重置
        self.undo = False               # 是否撤销

        h, w = image.shape[:2]
        self.display_size = (int(w * self.scale), int(h * self.scale))
        self.disp = cv2.resize(self.image, self.display_size, interpolation=cv2.INTER_CUBIC)

    def _draw(self) -> np.ndarray:
        """在画布上绘制分割线"""
        canvas = self.disp.copy()
        # 画竖线
        for x in self.lines:
            cv2.line(canvas, (x, 0), (x, canvas.shape[0] - 1), (0, 255, 255), 1)
        return canvas

    def _mouse(self, event, x, y, flags, param):
        """鼠标回调：左键点击添加分割线"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.lines.append(x)
            self.lines.sort()

    def run(self) -> Tuple[bool, List[int]]:
        """启动交互窗口，返回 (是否成功, 分割线列表)"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
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
                # 撤销上一条线
                if self.lines:
                    self.lines.pop()
            elif key == ord('r'):
                # 清空所有线
                self.lines.clear()
            elif key in (13, 10):  # Enter
                self.confirmed = True
                break
        cv2.destroyWindow(self.window_name)
        return self.confirmed and not self.quit, self.lines


def map_display_to_original(xs_disp: List[int], scale: float) -> List[int]:
    """将显示坐标系下的 x 坐标映射回原图坐标"""
    return [max(0, int(round(x / scale))) for x in xs_disp]


def crop_by_splits(image: np.ndarray, xs: List[int]) -> List[np.ndarray]:
    """根据分割线坐标，将图像按竖直方向切分成若干小图"""
    h, w = image.shape[:2]
    xs_sorted = sorted(set([x for x in xs if 0 < x < w]))
    edges = [0] + xs_sorted + [w]  # 加入左右边界
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
    将单个字符转义成可在 Windows 路径中安全使用的字符串。
    规则：
    - 把非法字符 (< > : " / \\ | ? *)、英文句点、空格及控制字符
      替换为 'U+XXXX'（Unicode 码位十六进制）。
    - 其余字符保持不变。
    """
    if len(ch) != 1:
        return 'U+' + ''.join(f"{ord(c):04X}" for c in ch)
    invalid = set('<>:"/\\|?*')
    if ch in invalid or ch in {'.', ' '}:
        return f"U+{ord(ch):04X}"
    if ord(ch) < 32:
        return f"U+{ord(ch):04X}"
    return ch


def ensure_char_dirs(base_dir: str, text: str):
    """为文本中每个字符创建对应文件夹"""
    os.makedirs(base_dir, exist_ok=True)
    for ch in text:
        safe = sanitize_char_for_path(ch)
        ch_dir = os.path.join(base_dir, safe)
        os.makedirs(ch_dir, exist_ok=True)


def next_filename_for_char(ch_dir: str, safe_ch: str) -> str:
    """根据目录下已有文件，为当前字符生成下一个编号文件名"""
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
    """将切分好的字符小图保存到对应目录"""
    base_dir = os.path.join(output_root, folder_name)
    ensure_char_dirs(base_dir, text)
    if len(crops) != len(text):
        raise ValueError(f"切分数量({len(crops)})与文本长度({len(text)})不符")
    for i, ch in enumerate(text):
        safe = sanitize_char_for_path(ch)
        ch_dir = os.path.join(base_dir, safe)
        target_path = next_filename_for_char(ch_dir, safe)
        cv2.imwrite(target_path, crops[i])


def load_image(path: str) -> np.ndarray:
    """读取图像，失败则抛出异常"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"无法加载图像: {path}")
    return img


def list_subfolders(root: str) -> List[str]:
    """返回根目录下所有子文件夹名称（排序后）"""
    result = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isdir(full):
            result.append(name)
    return result


def read_txt_label(txt_path: str) -> str:
    """读取txt标签文件，返回第一行的文本内容"""
    if not os.path.isfile(txt_path):
        return ""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            return line
    except Exception:
        return ""


def find_txt_files_in_folder(folder_path: str) -> Dict[str, str]:
    """在指定文件夹中查找所有txt文件，返回文件名到文本内容的映射"""
    txt_mapping = {}
    if not os.path.isdir(folder_path):
        return txt_mapping

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            txt_path = os.path.join(folder_path, filename)
            text_content = read_txt_label(txt_path)
            if text_content:
                # 去掉.txt后缀作为键
                key = os.path.splitext(filename)[0]
                txt_mapping[key] = text_content
    return txt_mapping


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


def load_progress(progress_path: str) -> Dict:
    """读取进度 JSON，若不存在或异常则返回空字典"""
    if os.path.isfile(progress_path):
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_progress(progress_path: str, data: Dict):
    """将进度写入 JSON 文件"""
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_image_interactive(img_path: str, text: str, scale: float, window_title: str) -> Tuple[bool, List[np.ndarray]]:
    """单张图的交互式切分流程，返回 (是否成功, 字符小图列表)"""
    img = load_image(img_path)
    session = SplitSession(img, display_scale=scale, window_name=window_title)
    confirmed, lines_disp = session.run()
    if not confirmed:
        return False, []
    xs = map_display_to_original(lines_disp, session.scale)
    # 检查分割线数量
    if len(xs) != max(0, len(text) - 1):
        print(f"文本 '{text}' 需要 {len(text)-1} 条分割线，实际 {len(xs)} 条。按任意键继续...")
        cv2.waitKey(0)
        return False, []
    crops = crop_by_splits(img, xs)
    if len(crops) != len(text):
        print(f"切分后得到 {len(crops)} 张小图，期望 {len(text)} 张。按任意键继续...")
        cv2.waitKey(0)
        return False, []
    return True, crops


def main():
    parser = argparse.ArgumentParser(description="水平文本图像手动字符切分工具")
    parser.add_argument('--input', '-i', default=os.path.join('tools', 'split_character', 'output'),
                        help='输入根目录，包含若干子文件夹')
    parser.add_argument('--output', '-o', default=os.path.join('tools', 'split_character', 'split'),
                        help='输出根目录，用于保存切分结果')
    parser.add_argument('--first-text', default=DEFAULT_FIRST,
                        help='第一行图像对应的已知文本')
    parser.add_argument('--second-text', default=DEFAULT_SECOND,
                        help='第二行图像对应的已知文本')
    parser.add_argument('--labels-dir', default=None,
                        help='可选：标签文件目录，包含txt文件，文件名对应图像名')
    parser.add_argument('--filelist', default=None,
                        help='可选：CSV/TXT 文件，每行格式 "filename,text" 或以制表符分隔')
    parser.add_argument('--scale', type=float, default=2.0,
                        help='显示缩放倍数，低分辨率图像可放大查看')
    parser.add_argument('--start-folder', default=None,
                        help='可选：从指定子文件夹开始处理')
    args = parser.parse_args()

    input_root = args.input
    output_root = args.output
    first_text = args.first_text
    second_text = args.second_text
    labels_dir = args.labels_dir
    scale = args.scale

    if not os.path.isdir(input_root):
        print(f"输入目录不存在: {input_root}")
        sys.exit(1)

    # 如果指定了标签目录，读取txt标签文件
    txt_mapping = {}
    if labels_dir and os.path.isdir(labels_dir):
        txt_mapping = find_txt_files_in_folder(labels_dir)
        print(f"从 {labels_dir} 加载了 {len(txt_mapping)} 个txt标签文件")

    os.makedirs(output_root, exist_ok=True)
    progress_path = os.path.join(output_root, PROGRESS_FILENAME)
    progress = load_progress(progress_path)
    processed: Dict[str, Dict[str, bool]] = progress.get('processed', {})

    folders = list_subfolders(input_root)
    if args.start_folder and args.start_folder in folders:
        start_index = folders.index(args.start_folder)
        folders = folders[start_index:]

    print("操作指引：\n - 点击字符之间位置添加竖直分割线。\n - 按 Enter 确认并完成切分。\n - 'u' 撤销上一步；'r' 清空所有线；'q' 退出。\n")

    # ========== 新增分支：filelist 模式 ==========
    if args.filelist:
        pairs = read_filelist(args.filelist)
        if not pairs:
            print(f"[警告] filelist 为空或不可用: {args.filelist}")
            return
        progress_path = os.path.join(output_root, PROGRESS_FILENAME)
        progress = load_progress(progress_path)
        processed_filelist = progress.get('processed_filelist', {})

        # 当 filelist 中给出相对路径时的基准目录
        base_dir = input_root if os.path.isdir(input_root) else os.getcwd()

        for idx, (filename, text) in enumerate(pairs):
            img_path = filename
            if not os.path.isabs(img_path):
                img_path = os.path.join(base_dir, filename)
            img_path = os.path.normpath(img_path)

            key = filename  # 以清单中的原始文件名作为键，便于断点续传
            if processed_filelist.get(key, False):
                continue

            if not os.path.isfile(img_path):
                print(f"[警告] 找不到图像: {img_path}，跳过。")
                processed_filelist[key] = True
                progress['processed_filelist'] = processed_filelist
                save_progress(progress_path, progress)
                continue

            print(f"处理图像: {filename}，使用文本: '{text}'")
            ok, crops = process_image_interactive(
                img_path, text, scale, window_title=os.path.basename(img_path)
            )
            if ok:
                # 使用文件名（不包含扩展名）作为文件夹名
                folder_name = os.path.splitext(os.path.basename(filename))[0]
                save_crops(output_root, folder_name, crops, text)
                processed_filelist[key] = True
                print(f"[已完成] {filename} 文本: '{text}'")
            else:
                print(f"[未确认] {filename}，下次可继续。")
                processed_filelist[key] = False

            progress['processed_filelist'] = processed_filelist
            save_progress(progress_path, progress)

        print("按清单的所有可处理项已完成。")
        return

    for folder in folders:
        sub_in = os.path.join(input_root, folder)

        # 根据txt标签确定文本内容，如果没有标签则使用默认值
        current_first_text = txt_mapping.get('first_line_img', first_text)
        current_second_text = txt_mapping.get('second_line_img', second_text)

        # 预先创建两个文本所需的字符目录
        ensure_char_dirs(os.path.join(output_root, folder), current_first_text)
        ensure_char_dirs(os.path.join(output_root, folder), current_second_text)

        state = processed.get(folder, {"first": False, "second": False})

        # 处理第一行
        if not state.get("first", False):
            first_path = os.path.join(sub_in, 'first_line_img.jpg')
            if not os.path.isfile(first_path):
                print(f"[警告] {sub_in} 中缺少 first_line_img.jpg，跳过第一行。")
                state["first"] = True  # 标记为已处理，避免后续重复提示
            else:
                print(f"处理第一行图像，使用文本: '{current_first_text}'")
                ok, crops = process_image_interactive(first_path, current_first_text, scale,
                                                     window_title=f"{folder} - first_line")
                if ok:
                    save_crops(output_root, folder, crops, current_first_text)
                    state["first"] = True
                else:
                    # 用户按 q 退出或其他原因未完成
                    print("第一行未确认。若按了 'q' 将退出，否则下次可重试。")
                    processed[folder] = state
                    progress['processed'] = processed
                    save_progress(progress_path, progress)
                    return

        # 保存中间进度
        processed[folder] = state
        progress['processed'] = processed
        save_progress(progress_path, progress)

        # 处理第二行
        if not state.get("second", False):
            second_path = os.path.join(sub_in, 'second_line_img.jpg')
            if not os.path.isfile(second_path):
                print(f"[警告] {sub_in} 中缺少 second_line_img.jpg，跳过第二行。")
                state["second"] = True
            else:
                print(f"处理第二行图像，使用文本: '{current_second_text}'")
                ok, crops = process_image_interactive(second_path, current_second_text, scale,
                                                     window_title=f"{folder} - second_line")
                if ok:
                    save_crops(output_root, folder, crops, current_second_text)
                    state["second"] = True
                else:
                    print("第二行未确认。将退出以便稍后恢复。")
                    processed[folder] = state
                    progress['processed'] = processed
                    save_progress(progress_path, progress)
                    return

        # 两行都完成后保存最终进度
        processed[folder] = state
        progress['processed'] = processed
        save_progress(progress_path, progress)
        print(f"[完成] {folder}: 第一行={state['first']} 第二行={state['second']}")

    print("所有符合条件的子文件夹已处理完毕。")


if __name__ == '__main__':
    main()