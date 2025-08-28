# -*- coding: utf-8 -*-
"""
批量 YOLO 推理 + 裁剪 + 按类别旋转并保存

功能：
1. 对指定文件夹中的所有图片进行 YOLO 推理；
2. 将检测到的目标区域裁剪出来；
3. 根据类别执行指定角度的旋转（'clockwise90', '0degrees', 'counterclockwise90', '180degrees'）；
4. 按类别分类保存到目标输出目录中（默认：<source>\\rotated_crops）。

使用示例（Windows，注意反斜杠路径）：
    python tools\yolo_detect_and_rotate.py \
        --weights checkpoints\best.pt \
        --source test \
        --output test\rotated_crops \
        --conf 0.25 \
        --imgsz 640

参数说明：
- --weights: 模型权重路径（ultralytics YOLO 权重，如 .pt 文件）。
- --source: 输入图片文件夹路径。
- --output: 输出目录（可选，默认 <source>\\rotated_crops）。
- --conf: 置信度阈值（默认 0.25）。
- --imgsz: 推理输入尺寸（默认 640）。
- --device: 设备，如 '0' 或 'cpu'（默认 '0'）。
- --save-vis: 是否保存可视化框图（默认 False）。

备注：
- 旋转规则使用 PIL 的 transpose：
  - 'clockwise90' -> 顺时针 90 度（PIL: ROTATE_270）
  - 'counterclockwise90' -> 逆时针 90 度（PIL: ROTATE_90）
  - '180degrees' -> 180 度（PIL: ROTATE_180）
  - '0degrees' -> 不旋转
- 若模型类别名与上述四类不完全一致，仍会保存裁剪结果（按原类别名建立子目录），未知类别默认不旋转。

依赖：ultralytics, pillow, numpy
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

# 支持的图片后缀
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# 类别到旋转操作的映射（PIL transpose 常量）
from PIL import Image as PILImage
ROTATE_OPS: Dict[str, int | None] = {
    '0degrees': None,  # 不旋转
    'clockwise90': PILImage.ROTATE_270,  # 顺时针 90
    'counterclockwise90': PILImage.ROTATE_90,  # 逆时针 90
    '180degrees': PILImage.ROTATE_180,  # 180 度
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='YOLO 推理+裁剪+按类别旋转并保存')
    parser.add_argument('--weights', type=str, required=True, help='YOLO 权重路径 (.pt)')
    parser.add_argument('--source', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', type=str, default=None, help='输出目录（默认 <source>\\rotated_crops）')
    parser.add_argument('--conf', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--imgsz', type=int, default=640, help='推理输入尺寸')
    parser.add_argument('--device', type=str, default='0', help="设备：GPU 编号如 '0' 或 'cpu'")
    parser.add_argument('--save-vis', action='store_true', help='是否保存可视化检测图')
    return parser.parse_args()


def is_image_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clamp_box(xyxy: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(0, min(x2, w)))
    y2 = int(max(0, min(y2, h)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def rotate_by_class(img: Image.Image, class_name: str) -> Image.Image:
    op = ROTATE_OPS.get(class_name, None)
    if op is None:
        return img
    return img.transpose(op)


def draw_boxes(image: Image.Image, boxes: np.ndarray, labels: List[str]) -> Image.Image:
    try:
        from PIL import ImageDraw, ImageFont
        vis = image.copy()
        draw = ImageDraw.Draw(vis)
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1 + 2, y1 + 2), label, fill=(255, 0, 0))
        return vis
    except Exception:
        return image


def main():
    args = parse_args()

    source_dir = Path(args.source)
    assert source_dir.exists() and source_dir.is_dir(), f'输入目录不存在：{source_dir}'

    output_dir = Path(args.output) if args.output else (source_dir / 'rotated_crops')
    ensure_dir(output_dir)

    # 加载模型
    model = YOLO(args.weights)

    # 收集图片文件
    images: List[Path] = [p for p in source_dir.rglob('*') if is_image_file(p)]
    if not images:
        print(f'[WARN] 未在目录中找到图片：{source_dir}')
        return

    print(f'[INFO] 共发现 {len(images)} 张图片，开始推理...')

    for idx, img_path in enumerate(images, 1):
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'[WARN] 打开图片失败，跳过: {img_path} ({e})')
            continue

        # 推理
        results = model.predict(
            source=np.array(image),
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False
        )
        if not results:
            print(f'[INFO] 无结果：{img_path.name}')
            continue

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            print(f'[INFO] 未检测到目标：{img_path.name}')
            continue

        names = res.names if hasattr(res, 'names') and res.names else getattr(model, 'names', None)
        if isinstance(names, dict):
            id2name = names
        elif isinstance(names, list):
            id2name = {i: n for i, n in enumerate(names)}
        else:
            id2name = {}

        boxes_xyxy = res.boxes.xyxy.cpu().numpy()  # (N,4)
        clses = res.boxes.cls.cpu().numpy().astype(int)  # (N,)
        confs = res.boxes.conf.cpu().numpy()  # (N,)

        W, H = image.size
        vis_boxes: List[Tuple[int, int, int, int]] = []
        vis_labels: List[str] = []

        base_name = img_path.stem
        for i, (xyxy, cid, conf) in enumerate(zip(boxes_xyxy, clses, confs)):
            cls_name = id2name.get(int(cid), str(cid))
            x1, y1, x2, y2 = clamp_box(tuple(xyxy), W, H)
            crop = image.crop((x1, y1, x2, y2))

            # 按类别旋转
            rotated = rotate_by_class(crop, cls_name)

            # 为类别创建子目录
            class_dir = output_dir / cls_name
            ensure_dir(class_dir)

            save_name = f'{base_name}_obj{i}_{cls_name}_{conf:.2f}.png'
            save_path = class_dir / save_name
            try:
                rotated.save(save_path)
            except Exception as e:
                print(f'[WARN] 保存失败：{save_path} ({e})')
                continue

            vis_boxes.append((x1, y1, x2, y2))
            vis_labels.append(f'{cls_name} {conf:.2f}')

        if args.save_vis and vis_boxes:
            vis_img = draw_boxes(image, np.array(vis_boxes), vis_labels)
            vis_dir = output_dir / '_vis'
            ensure_dir(vis_dir)
            try:
                vis_img.save(vis_dir / f'{base_name}_vis.png')
            except Exception:
                pass

        print(f'[OK] 处理完成({idx}/{len(images)}): {img_path.name}')

    print(f'[DONE] 结果已保存至：{output_dir}')


if __name__ == '__main__':
    main()
