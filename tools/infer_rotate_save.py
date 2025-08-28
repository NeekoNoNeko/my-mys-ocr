# -*- coding: utf-8 -*-
"""
使用 ultralytics 的 YOLO 接口对图片/目录进行推理：
- 将检测到的目标框裁剪为单独图片；
- 根据类别名进行旋转；
- 使用 CRNN 模型对裁剪图片进行文本识别；
- 将识别结果保存为 CSV 文件。

旋转规则：
  - 'clockwise90' => 顺时针旋转 90 度
  - '0degrees' => 不变
  - 'counterclockwise90' => 逆时针旋转 90 度
  - '180degrees' => 旋转 180 度

使用示例：
python tools/infer_rotate_save.py \
  --source /path/to/images \
  --yolo_model tools/runs/detect/train4/weights/best.pt \
  --crnn_model code/checkpoints/crnn_best.pth \
  --out_file outputs/results.csv \
  --save_images

说明:
- 支持 --source 为单张图片路径、包含图片的目录、或通配符。
- 一张图片可能包含 2-3 个同类目标，脚本会逐个裁剪、识别并记录。
- 识别结果将保存为 CSV 格式，包含文件名、检测框坐标、置信度、类别和识别文本。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

# 动态导入 CRNN 模型 - 修复路径问题
CRNN = None
model_imported = False

# 获取当前脚本目录和项目根目录
current_dir = Path(__file__).parent
project_root = current_dir.parent

# 尝试多种导入路径
import_paths = [
    project_root / "code",  # ../code
    current_dir / "code",   # ./code  
    project_root,           # 项目根目录
    current_dir,            # 当前目录
]

for path in import_paths:
    if (path / "model.py").exists():
        sys.path.insert(0, str(path))
        try:
            from model import CRNN
            model_imported = True
            print(f"成功从 {path} 导入 CRNN 模型")
            break
        except ImportError as e:
            print(f"尝试从 {path} 导入失败: {e}")
            continue

if not model_imported:
    print("警告：无法导入 CRNN 模型")
    print("请检查以下路径是否存在 model.py 文件：")
    for path in import_paths:
        model_file = path / "model.py"
        status = "存在" if model_file.exists() else "不存在"
        print(f"  {model_file} - {status}")
    CRNN = None

# 预期四类名称（用于兜底映射）
DEFAULT_CLASS_NAMES: List[str] = [
    'clockwise90',        # 0
    '0degrees',           # 1
    'counterclockwise90', # 2
    '180degrees',         # 3
]

# CRNN 相关配置
CHARS = "()-.><0123456789ABDEFIMRTVZgn"
BLANK = '─'  # 使用特殊符号作为填充符，避免与实际字符冲突
CHARS = BLANK + CHARS
NCLASS = len(CHARS)
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

def decode_crnn(preds):
    """CRNN 输出解码函数"""
    preds = preds.argmax(2)
    preds = preds.permute(1, 0)  # (batch, seq)
    texts = []
    for pred in preds:
        char_list = []
        prev_idx = 0
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                char_list.append(IDX2CHAR[idx])
            prev_idx = idx
        texts.append(''.join(char_list))
    return texts

class CRNNInferencer:
    """CRNN 推理器"""
    def __init__(self, model_path: str, device: str = None):
        if CRNN is None:
            raise ImportError(
                "CRNN 模型未能成功导入。请检查：\n"
                "1. model.py 文件是否存在于 code/ 目录中\n"
                "2. model.py 文件是否包含 CRNN 类定义\n"
                "3. 是否有必要的依赖包（torch等）"
            )

        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.imgH = 32
        self.nc = 1
        self.nh = 256

        print(f"使用设备: {self.device}")

        try:
            # 加载模型
            self.model = CRNN(self.imgH, self.nc, NCLASS, self.nh).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"CRNN 模型加载成功: {model_path}")
        except Exception as e:
            raise RuntimeError(f"加载 CRNN 模型失败: {e}")

        # 预处理
        self.transform = transforms.Compose([
            transforms.Resize((self.imgH, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def infer(self, img: np.ndarray) -> str:
        """对图片进行文本识别"""
        # 转换为PIL图像
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(img)

        # 预处理
        img_tensor = self.transform(pil_img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            preds = self.model(img_tensor)
            preds_log_softmax = torch.nn.functional.log_softmax(preds, 2)
            text = decode_crnn(preds_log_softmax)[0]

        return text


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rotate_by_class(img: np.ndarray, class_name: str) -> np.ndarray:
    """根据类别名旋转图像。
    使用 OpenCV 的 cv2.rotate，避免插值差异。
    """
    name = (class_name or '').strip().lower()
    # 兼容可能出现的不同写法/标点
    name = name.replace(" ", "").replace("deg", "degrees").replace("’", "'").replace("`", "'")

    if name in ("clockwise90", "clockwise90'", "90clockwise", "cw90"):
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if name in ("0degrees", "0degree", "0degreess", "0", "none", "nochange"):
        return img
    if name in ("counterclockwise90", "counterclockwise90'", "90counterclockwise", "ccw90"):
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if name in ("180degrees", "180degree", "180"):
        return cv2.rotate(img, cv2.ROTATE_180)

    # 未知类名时不做旋转
    return img


def clip_box(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> List[int]:
    x1i = int(max(0, min(x1, w - 1)))
    y1i = int(max(0, min(y1, h - 1)))
    x2i = int(max(0, min(x2, w)))
    y2i = int(max(0, min(y2, h)))
    # 确保右下角大于左上角
    if x2i <= x1i:
        x2i = min(w, x1i + 1)
    if y2i <= y1i:
        y2i = min(h, y1i + 1)
    return [x1i, y1i, x2i, y2i]


def build_class_names_map(result_names: Dict[int, str] | None) -> Dict[int, str]:
    """构建类 ID -> 名称 的映射。优先使用结果中的 names，其次使用 DEFAULT_CLASS_NAMES。"""
    id2name: Dict[int, str] = {}
    if result_names and len(result_names) > 0:
        # ultralytics result.names 通常是 dict[int->str]
        for k, v in result_names.items():
            id2name[int(k)] = str(v)
        return id2name
    # 兜底：按索引映射
    return {i: n for i, n in enumerate(DEFAULT_CLASS_NAMES)}


def process_image(yolo_model: YOLO, crnn_model: CRNNInferencer, img_path: Path, 
                 conf: float = 0.25, imgsz: int | None = None, 
                 save_images: bool = False, save_dir: Path | None = None) -> List[Dict]:
    """对单张图片推理、裁剪、识别。返回识别结果列表。"""
    # 运行YOLO预测
    kwargs = {"source": str(img_path), "conf": conf}
    if imgsz:
        kwargs["imgsz"] = imgsz
    results = yolo_model.predict(show=False, save=False, **kwargs)

    detection_results = []
    for ri, result in enumerate(results):
        # 原图（BGR）
        orig: np.ndarray = result.orig_img  # H x W x 3
        h, w = orig.shape[:2]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        id2name = build_class_names_map(getattr(result, 'names', None))

        # 遍历每个检测框
        for bi in range(len(boxes)):
            box = boxes[bi]
            xyxy = box.xyxy.squeeze().tolist()  # [x1,y1,x2,y2]
            cls_id = int(box.cls.item()) if hasattr(box, 'cls') else 0
            confv = float(box.conf.item()) if hasattr(box, 'conf') else 0.0
            x1, y1, x2, y2 = xyxy
            x1i, y1i, x2i, y2i = clip_box(x1, y1, x2, y2, w, h)
            crop = orig[y1i:y2i, x1i:x2i].copy()

            # 旋转
            class_name = id2name.get(cls_id, DEFAULT_CLASS_NAMES[cls_id] if 0 <= cls_id < len(DEFAULT_CLASS_NAMES) else str(cls_id))
            crop_rot = rotate_by_class(crop, class_name)

            # 使用CRNN进行文本识别
            try:
                recognized_text = crnn_model.infer(crop_rot)
            except Exception as e:
                print(f"CRNN 识别失败 {img_path.name}[{bi}]: {e}")
                recognized_text = ""

            # 保存裁剪图片（可选）
            saved_image_path = ""
            if save_images and save_dir:
                try:
                    # 创建保存目录结构：save_dir/图片名/
                    stem = img_path.stem
                    img_dir = save_dir / stem
                    ensure_dir(img_dir)

                    # 构造文件名：识别文本_类别_置信度_索引.jpg
                    safe_text = recognized_text.replace("/", "-").replace("\\", "-").replace(":", "-").replace(" ", "_")
                    if not safe_text:
                        safe_text = "unknown"

                    safe_class = class_name.replace("/", "-").replace(" ", "_")
                    save_name = f"{safe_text}_{safe_class}_{confv:.3f}_{bi}.jpg"
                    save_path = img_dir / save_name

                    # 保存图片
                    cv2.imwrite(str(save_path), crop_rot)
                    saved_image_path = str(save_path)

                except Exception as e:
                    print(f"保存图片失败 {img_path.name}[{bi}]: {e}")

            # 记录结果 - 只保留必要字段
            result_dict = {
                'image_path': str(img_path),
                'image_name': img_path.name,
                'recognized_text': recognized_text
            }
            detection_results.append(result_dict)

    return detection_results


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_image_files(source_path: Path, recursive: bool = True) -> List[Path]:
    """收集指定路径下的所有图片文件"""
    image_files = []

    if source_path.is_file():
        if is_image_file(source_path):
            image_files.append(source_path)
        return image_files

    if source_path.is_dir():
        if recursive:
            # 递归遍历所有子目录
            for file_path in source_path.rglob("*"):
                if file_path.is_file() and is_image_file(file_path):
                    image_files.append(file_path)
        else:
            # 只遍历当前目录
            for file_path in source_path.iterdir():
                if file_path.is_file() and is_image_file(file_path):
                    image_files.append(file_path)

    return sorted(image_files)


def print_directory_stats(image_files: List[Path], source_path: Path):
    """打印目录统计信息"""
    if not image_files:
        print(f"在 {source_path} 中没有找到图片文件")
        return

    print(f"在 {source_path} 中找到 {len(image_files)} 个图片文件")

    # 按文件扩展名统计
    ext_count = {}
    for img_file in image_files:
        ext = img_file.suffix.lower()
        ext_count[ext] = ext_count.get(ext, 0) + 1

    print("文件类型统计:")
    for ext, count in sorted(ext_count.items()):
        print(f"  {ext}: {count} 个文件")

    # 按目录层级统计
    dir_count = {}
    for img_file in image_files:
        rel_path = img_file.relative_to(source_path)
        dir_path = str(rel_path.parent) if rel_path.parent != Path('.') else "根目录"
        dir_count[dir_path] = dir_count.get(dir_path, 0) + 1

    if len(dir_count) > 1:
        print("目录分布:")
        for dir_path, count in sorted(dir_count.items()):
            print(f"  {dir_path}: {count} 个文件")

    print("-" * 60)


def save_results_to_csv(results: List[Dict], output_file: Path) -> None:
    """将识别结果保存为CSV文件"""
    if not results:
        print("没有检测结果，不生成CSV文件")
        return

    # 确保输出目录存在
    ensure_dir(output_file.parent)

    # CSV列名 - 简化版本
    fieldnames = ['image_path', 'image_name', 'recognized_text']

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="YOLO + CRNN 推理并保存结果为CSV")
    parser.add_argument("--source", type=str, required=True, help="图片路径、目录或通配符")
    parser.add_argument("--yolo_model", type=str, default=str(Path("tools") / "runs" / "detect" / "train4" / "weights" / "best.pt"), help="YOLO模型权重路径 .pt")
    parser.add_argument("--crnn_model", type=str, required=True, help="CRNN模型权重路径 .pth")
    parser.add_argument("--out_file", type=str, default=str(Path("outputs") / "results.csv"), help="输出CSV文件路径")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO置信度阈值")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO推理输入尺寸，可不填")
    parser.add_argument("--device", type=str, default=None, help="设备选择 (cuda/cpu)，默认自动选择")
    parser.add_argument("--save_images", action="store_true", help="是否保存检测到的对象图片")
    parser.add_argument("--save_dir", type=str, default=str(Path("outputs") / "crops"), help="检测对象图片保存目录")
    parser.add_argument("--recursive", action="store_true", default=True, help="递归遍历子文件夹（默认启用）")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false", help="不递归遍历子文件夹")

    args = parser.parse_args()

    # 检查模型文件
    yolo_model_path = Path(args.yolo_model)
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"未找到YOLO模型权重: {yolo_model_path}")

    crnn_model_path = Path(args.crnn_model)
    if not crnn_model_path.exists():
        raise FileNotFoundError(f"未找到CRNN模型权重: {crnn_model_path}")

    # 初始化模型
    print("加载YOLO模型...")
    yolo_model = YOLO(str(yolo_model_path))

    print("加载CRNN模型...")
    crnn_model = CRNNInferencer(str(crnn_model_path), args.device)

    source = Path(args.source)
    output_file = Path(args.out_file)
    save_dir = Path(args.save_dir) if args.save_images else None

    if args.save_images:
        print(f"检测图片将保存到: {save_dir}")

    # 收集所有要处理的图片文件
    print("正在扫描图片文件...")
    if source.exists() and (source.is_file() or source.is_dir()):
        image_files = collect_image_files(source, args.recursive)
        print_directory_stats(image_files, source)
    else:
        # 通配符处理 - 先用YOLO预测获取文件列表
        print("使用通配符模式扫描文件...")
        try:
            yolo_results = yolo_model.predict(source=str(source), conf=args.conf, show=False, save=False, imgsz=args.imgsz)
            image_files = []
            for result in yolo_results:
                img_path = Path(getattr(result, 'path', ''))
                if img_path and img_path.exists():
                    image_files.append(img_path)
            print(f"通配符匹配到 {len(image_files)} 个图片文件")
        except Exception as e:
            print(f"通配符处理失败: {e}")
            return

    if not image_files:
        print("没有找到可处理的图片文件")
        return

    # 批量处理所有图片
    all_results = []
    total_detections = 0

    print(f"开始处理 {len(image_files)} 个图片文件...")
    print("=" * 60)

    for i, img_path in enumerate(image_files, 1):
        try:
            results = process_image(yolo_model, crnn_model, img_path, conf=args.conf, imgsz=args.imgsz,
                                  save_images=args.save_images, save_dir=save_dir)
            all_results.extend(results)
            total_detections += len(results)

            # 显示进度
            rel_path = img_path.relative_to(source) if source.is_dir() else img_path.name
            print(f"[{i:4d}/{len(image_files)}] {rel_path} -> {len(results)} 个目标")

        except Exception as e:
            print(f"[{i:4d}/{len(image_files)}] {img_path.name} -> 错误: {e}")

    print("=" * 60)
    print(f"批量处理完成:")
    print(f"  处理图片: {len(image_files)} 个")
    print(f"  检测目标: {total_detections} 个")
    print(f"  成功识别: {len([r for r in all_results if r['recognized_text']])} 个")

    # 保存结果
    save_results_to_csv(all_results, output_file)
    print(f"完成，共处理 {len(all_results)} 个检测目标")


if __name__ == "__main__":
    main()
