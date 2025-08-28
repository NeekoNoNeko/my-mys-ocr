# -*- coding: utf-8 -*-
"""
YOLO 推理 + 裁剪 + 旋转 一体化脚本

功能：
1. 对指定文件夹(source)下的所有图片进行 YOLO 推理；
2. 自动遍历并处理所有图片；
3. 将每个检测到的目标按边界框裁剪并保存；
4. 根据类别将裁剪图片做对应旋转：
   - 'clockwise90'           -> 顺时针旋转 90 度
   - '0degrees'              -> 不旋转
   - 'counterclockwise90'    -> 逆时针旋转 90 度
   - '180degrees'            -> 旋转 180 度
5. 输出保存到目标文件夹（可与输入相同），文件名包含原图名与检测序号。

使用示例（Windows）：
  python tools/detect_crop_rotate.py \
      --weights checkpoints/best.pt \
      --source test \
      --save_dir test  \
      --conf 0.25 \
      --imgsz 640

说明：
- 需要安装 ultralytics（YOLOv8+）以及其依赖（torch 等）。
- 若 requirements.txt 中 ultralytics 被注释，请自行 pip 安装：
    pip install ultralytics
- 类别名需与数据集训练时一致且包含以下 4 类之一，否则将按“未知类别”处理且不旋转。

作者：项目自动化脚本（Junie）
"""

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError as e:
    raise ImportError(
        "未找到 ultralytics 包，请先安装：pip install ultralytics；"
        "如果没有安装 torch 也需要一并安装。原始错误：%s" % str(e)
    )


pth = "828\\E3"
# ============================================================================
# 全局配置参数 - 可手动修改
# ============================================================================
DEFAULT_WEIGHTS = "..\\models\\allyolo.pt"      # YOLO 权重路径
DEFAULT_SOURCE = "..\\" + pth                       # 待处理图片所在文件夹
DEFAULT_SAVE_DIR = "..\\dataset\\crnn\\" + pth                         # 保存目录（空字符串表示保存到 source 目录）
DEFAULT_CONF = 0.25                          # 置信度阈值
DEFAULT_IMGSZ = 640                          # 推理尺寸
DEFAULT_PATTERN = "*"                        # 通配过滤，例如 *.jpg
DEFAULT_DEVICE = None                        # 设备选择，例如 '0' 或 'cpu'，None 表示自动
# ============================================================================

# 允许的图片扩展名（小写）
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# 类别到旋转角度（PIL 的 rotate 为逆时针为正）
# 为了实现“顺时针 90 度”，使用 angle = -90
CLASS_ROTATE_DEG: Dict[str, int] = {
    'clockwise90': -90,
    '0degrees': 0,
    'counterclockwise90': 90,
    '180degrees': 180,
}


def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )


def iter_images(source: str, pattern: str = '*') -> List[str]:
    """遍历 source 下的所有图片路径。
    :param source: 目标文件夹
    :param pattern: 额外的通配符过滤（例如 '*.jpg'），默认 '*' 表示不过滤
    :return: 图片路径列表（绝对路径）
    """
    p = Path(source)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"source 目录不存在或不是文件夹: {source}")

    files: List[str] = []
    # 允许用户传入更窄的过滤；但仍会再按扩展名过滤一遍
    pattern = pattern if pattern else '*'
    for ext in IMG_EXTS:
        files.extend(glob.glob(str(p / pattern), recursive=False))
        # 如果用户 pattern 已经带扩展，则上面的 ext 循环不必重复拼接
    # 去重并过滤扩展
    uniq: List[str] = []
    seen = set()
    for f in files or glob.glob(str(p / '*')):
        fp = str(Path(f).resolve())
        if fp in seen:
            continue
        if Path(fp).suffix.lower() in IMG_EXTS and Path(fp).is_file():
            uniq.append(fp)
            seen.add(fp)
    uniq.sort()
    return uniq


def clip_box(xyxy: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
    """裁剪边界，确保坐标在图像尺寸内。"""
    x1, y1, x2, y2 = xyxy
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    # 防止空框
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def rotate_by_class(img: Image.Image, cls_name: str) -> Image.Image:
    angle = CLASS_ROTATE_DEG.get(cls_name)
    if angle is None:
        # 未知类别不旋转，仅记录日志
        logging.warning(f"未知类别 '{cls_name}'，不执行旋转。")
        return img
    # expand=True 以避免旋转后裁剪
    return img.rotate(angle, expand=True)


def save_crop(img: Image.Image, save_dir: str, base_name: str, idx: int, cls_name: str, conf: float) -> str:
    os.makedirs(save_dir, exist_ok=True)
    stem = Path(base_name).stem
    ext = '.jpg'  # 统一保存为 JPG，亦可选择与原图相同后缀
    out_name = f"{stem}_det{idx}_{cls_name}_{conf:.2f}{ext}"
    out_path = str(Path(save_dir) / out_name)
    # 防止重名覆盖
    cnt = 1
    while os.path.exists(out_path):
        out_name = f"{stem}_det{idx}_{cls_name}_{conf:.2f}_{cnt}{ext}"
        out_path = str(Path(save_dir) / out_name)
        cnt += 1
    # 保存
    img.save(out_path, quality=95)
    return out_path


def run(weights: str, source: str, save_dir: str, conf: float = 0.25, imgsz: int = 640, pattern: str = '*', device: str = None):
    setup_logger()

    logging.info(f"加载模型: {weights}")
    model = YOLO(weights)

    files = iter_images(source, pattern)
    if not files:
        logging.warning(f"未在 {source} 下找到图片，已结束。")
        return

    # 若保存目录为空，则默认保存到 source（满足“保存到目标文件夹中”的要求）
    save_dir = save_dir if save_dir else source
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    total_imgs = len(files)
    total_dets = 0

    for i, f in enumerate(files, 1):
        logging.info(f"[{i}/{total_imgs}] 推理: {f}")
        # predict 返回一个列表（每张图一个结果）
        results = model.predict(source=f, conf=conf, imgsz=imgsz, verbose=False, device=device)
        if not results:
            logging.warning("模型未返回结果对象，跳过该图。")
            continue
        res = results[0]
        names = res.names  # id->name 映射

        # 使用 PIL 读取以便后续旋转
        with Image.open(f) as pil_im:
            pil_im = pil_im.convert('RGB')
            w, h = pil_im.size

            if res.boxes is None or len(res.boxes) == 0:
                logging.info("无检测，跳过裁剪。")
                continue

            # 提取框、类别与置信度
            boxes_xyxy = res.boxes.xyxy.cpu().numpy()
            clss = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()

            for j, (xyxy, cls_id, c) in enumerate(zip(boxes_xyxy, clss, confs), 1):
                x1, y1, x2, y2 = clip_box(xyxy, w, h)
                crop = pil_im.crop((x1, y1, x2, y2))
                cls_name = names.get(int(cls_id), str(cls_id)) if isinstance(names, dict) else (
                    names[int(cls_id)] if isinstance(names, (list, tuple)) and int(cls_id) < len(names) else str(cls_id)
                )
                crop_rot = rotate_by_class(crop, cls_name)
                out_path = save_crop(crop_rot, save_dir, base_name=os.path.basename(f), idx=j, cls_name=cls_name, conf=float(c))
                logging.info(f"保存: {out_path}")
                total_dets += 1

    logging.info(f"处理完成。图像数: {total_imgs}, 检测并保存裁剪数: {total_dets}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO 推理 + 裁剪 + 旋转')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS, help='YOLO 权重路径，例如 checkpoints\\best.pt')
    parser.add_argument('--source', type=str, default=DEFAULT_SOURCE, help='待处理图片所在文件夹')
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR, help='保存目录（默认保存到 source 目录）')
    parser.add_argument('--conf', type=float, default=DEFAULT_CONF, help='置信度阈值')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_IMGSZ, help='推理尺寸')
    parser.add_argument('--pattern', type=str, default=DEFAULT_PATTERN, help='可选通配过滤，例如 *.jpg')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE, help="设备选择，例如 '0' 或 'cpu'，默认自动")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(
        weights=args.weights,
        source=args.source,
        save_dir=args.save_dir,
        conf=args.conf,
        imgsz=args.imgsz,
        pattern=args.pattern,
        device=args.device,
    )
