import cv2
import os
import glob
import argparse
from pathlib import Path

def find_template_match(img, template_path, threshold=0.8):
    """
    在图片中查找模板匹配

    Args:
        img: 输入图片（BGR格式）
        template_path: 模板图片路径
        threshold: 匹配阈值

    Returns:
        匹配信息字典或None
    """
    if not os.path.exists(template_path):
        print(f"警告：模板文件不存在 - {template_path}")
        return None

    # 转换为灰度图
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if template is None:
        print(f"警告：无法读取模板文件 - {template_path}")
        return None

    template_height, template_width = template.shape

    # 执行模板匹配
    result = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        return {
            'location': max_loc,
            'confidence': max_val,
            'top_left': max_loc,
            'bottom_right': (max_loc[0] + template_width, max_loc[1] + template_height)
        }
    else:
        print(f"模板匹配失败，最高置信度: {max_val:.4f} (阈值: {threshold})")
        return None

def calculate_dynamic_boxes(match_info):
    """
    根据模板匹配结果计算动态边界框

    Args:
        match_info: 模板匹配信息

    Returns:
        边界框列表
    """
    top_left = match_info['top_left']
    bottom_right = match_info['bottom_right']

    # 定义偏移量（根据您的notebook）
    first_line_offset = {
        'x0': -28, 'y0': -100,
        'x1': -49, 'y1': -15
    }
    second_line_offset = {
        'x0': -2, 'y0': -78,
        'x1': -25, 'y1': -42
    }

    # 计算第一行边界框
    first_line_top_left = [top_left[0] + first_line_offset['x0'],
                          top_left[1] + first_line_offset['y0']]
    first_line_bottom_right = [bottom_right[0] + first_line_offset['x1'],
                              bottom_right[1] + first_line_offset['y1']]

    # 计算第二行边界框
    second_line_top_left = [top_left[0] + second_line_offset['x0'],
                           top_left[1] + second_line_offset['y0']]
    second_line_bottom_right = [bottom_right[0] + second_line_offset['x1'],
                               bottom_right[1] + second_line_offset['y1']]

    # 转换为boxes格式（左上角坐标 + 宽高）
    boxes = [
        {
            "class": 0,
            "x": first_line_top_left[0],
            "y": first_line_top_left[1],
            "w": first_line_bottom_right[0] - first_line_top_left[0],
            "h": first_line_bottom_right[1] - first_line_top_left[1]
        },
        {
            "class": 0,
            "x": second_line_top_left[0],
            "y": second_line_top_left[1],
            "w": second_line_bottom_right[0] - second_line_top_left[0],
            "h": second_line_bottom_right[1] - second_line_top_left[1]
        }
    ]

    return boxes

def process_single_image(img_path, template_path, output_dir=None):
    """
    处理单张图片：使用模板匹配动态生成YOLO标注文件和裁剪图片

    Args:
        img_path: 图片路径
        template_path: 模板图片路径
        output_dir: 输出目录，如果为None则使用图片所在目录
    """
    try:
        # 1. 读原图，获取宽高
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            return False

        h_img, w_img = img.shape[:2]

        # 2. 模板匹配
        match_info = find_template_match(img, template_path)
        if match_info is None:
            print(f"模板匹配失败，跳过图片: {img_path}")
            return False

        print(f"模板匹配成功，置信度: {match_info['confidence']:.4f}")

        # 3. 计算动态边界框
        boxes = calculate_dynamic_boxes(match_info)

        # 4. 转换为 YOLO 格式
        labels = []
        for box in boxes:
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]

            # 确保边界框在图片范围内
            x = max(0, min(x, w_img))
            y = max(0, min(y, h_img))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))

            xc = x + w / 2
            yc = y + h / 2
            # 归一化
            xc_norm = xc / w_img
            yc_norm = yc / h_img
            w_norm  = w  / w_img
            h_norm  = h  / h_img
            labels.append(f"{box['class']} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

        # 5. 确定输出路径
        if output_dir is None:
            output_dir = os.path.dirname(img_path)
        else:
            os.makedirs(output_dir, exist_ok=True)

        # 获取文件名（不含扩展名）
        img_name = os.path.splitext(os.path.basename(img_path))[0]

        # 6. 保存 YOLO 标注文件
        txt_path = os.path.join(output_dir, f"{img_name}.txt")
        with open(txt_path, 'w') as f:
            f.write('\n'.join(labels))

        # 7. 裁剪并保存区域图片（添加90度顺时针旋转）
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for i, box in enumerate(boxes, 1):
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]

            # 确保裁剪区域在图片范围内
            x = max(0, min(x, w_img))
            y = max(0, min(y, h_img))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))

            # 裁剪区域
            cropped = img_gray[y:y+h, x:x+w]
            # 90度顺时针旋转
            cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

            cropped_path = os.path.join(output_dir, f"{img_name}_line_{i:02d}.jpg")
            cv2.imwrite(cropped_path, cropped)

        print(f"✓ 处理完成: {img_path}")
        print(f"  - YOLO标注: {txt_path}")
        print(f"  - 裁剪图片: {len(boxes)} 个 (已旋转90度)")
        return True

    except Exception as e:
        print(f"✗ 处理失败 {img_path}: {str(e)}")
        return False
