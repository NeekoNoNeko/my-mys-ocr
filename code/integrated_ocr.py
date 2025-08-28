import cv2
import numpy as np
import glob
import os
import torch
from torchvision import transforms
from PIL import Image
from model import CRNN
import argparse
import time

# 字符集（需与训练时一致）
CHARS = "OP12/403ADC@E"  # 使用与训练时相同的字符集
BLANK = '-'
CHARS = BLANK + CHARS
nclass = len(CHARS)
idx2char = {i: c for i, c in enumerate(CHARS)}

def decode(preds):
    """解码CRNN模型输出"""
    preds = preds.argmax(2)
    preds = preds.permute(1, 0)  # (batch, seq)
    texts = []
    for pred in preds:
        char_list = []
        prev_idx = 0
        for idx in pred:
            idx = idx.item()
            if idx != 0 and idx != prev_idx:
                char_list.append(idx2char[idx])
            prev_idx = idx
        texts.append(''.join(char_list))
    return texts

def deskew_image(image):
    """增强的矫正图片倾斜函数（提高0.3°-1.7°范围的精度）"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用直方图均衡化增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # 使用高斯模糊减少噪声
    blurred = cv2.GaussianBlur(enhanced_gray, (5, 5), 0)

    # 使用Canny边缘检测（调整参数以检测更细的边缘）
    edges = cv2.Canny(blurred, 30, 100, apertureSize=3)

    # 使用霍夫变换检测直线（调整参数以提高小角度检测精度）
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, 40, minLineLength=110, maxLineGap=12)

    # 创建调试图像
    debug_img = image.copy()
    height, width = image.shape[:2]
    bottom_start = int(height * 0.6)  # 从高度70%处开始

    # 计算左侧区域边界（图像宽度的70%）
    left_region_boundary = int(width * 0.65)  # 65%宽度处

    # 在调试图像上添加底部参考线和文本
    cv2.line(debug_img, (0, bottom_start), (width, bottom_start), (255, 0, 0), 2)
    cv2.putText(debug_img, "Bottom Reference Line", (10, bottom_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 绘制左侧区域边界
    cv2.line(debug_img, (left_region_boundary, 0), (left_region_boundary, height), (0, 255, 255), 1)
    cv2.putText(debug_img, "Left Region Boundary", (left_region_boundary + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 如果没有检测到直线
    if lines is None:
        cv2.putText(debug_img, f"Skew Angle: 0.00 degrees", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image, debug_img, 0.0

    # 提取图片底部区域的直线（只考虑图片下方30%的区域）
    bottom_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 计算直线中点x坐标
        mid_x = (x1 + x2) / 2

        # 只考虑在底部区域且位于左侧区域的直线
        if (y1 > bottom_start and y2 > bottom_start and
                mid_x < left_region_boundary):
            # 过滤掉大角度的线（角度大于3度）
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 3:  # 忽略近似垂直的线
                bottom_lines.append(line[0])

    # 如果没有合适的底部直线
    if not bottom_lines:
        cv2.putText(debug_img, f"Skew Angle: 0.00 degrees", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image, debug_img, 0.0

    # 计算所有底部直线的角度（更精确的方法）
    angles = []
    weighted_angles = []

    for x1, y1, x2, y2 in bottom_lines:
        # 计算直线长度
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # 计算直线角度（以度为单位）
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # 保存角度
        angles.append(angle)

        # 使用长度作为权重（较长的线权重更大）
        weighted_angles.extend([angle] * int(length / 10))

    # 计算加权平均角度
    if weighted_angles:
        median_angle = np.median(weighted_angles)
    else:
        median_angle = np.median(angles)

    # 在调试图像上绘制检测到的直线
    for x1, y1, x2, y2 in bottom_lines:
        cv2.line(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 添加角度文本到调试图像
    angle_text = f"Detected Skew Angle: {median_angle:.4f} degrees"
    cv2.putText(debug_img, angle_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示检测到的直线数量
    cv2.putText(debug_img, f"Detected Lines: {len(bottom_lines)}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    # 显示检测区域信息
    region_info = f"Detection Region: Bottom {int(100 - height * 0.65 / height * 100)}% + Left {int(left_region_boundary / width * 100)}%"
    cv2.putText(debug_img, region_info, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    # 如果角度绝对值小于0.05度，则不旋转（更小的阈值）
    if abs(median_angle) < 0.05:
        cv2.putText(debug_img, "Correction: Not needed (angle < 0.05 degrees)", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return image, debug_img, median_angle

    # 旋转图像以矫正倾斜（使用更精确的插值方法）
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, rotation_matrix, (width, height),
                              flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_REPLICATE)

    # 添加矫正状态文本
    cv2.putText(debug_img, f"Correction: Applied ({median_angle:.4f} degrees)", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    # 显示旋转前后的对比
    cv2.putText(debug_img, "Before", (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.putText(deskewed, "After", (width - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    return deskewed, debug_img, median_angle

def stitch_images(img_path1, img_path2, output_path):
    """拼接两张图片"""
    # 读取图像并检查是否成功
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None:
        print(f"错误：无法读取图像 '{img_path1}'，请检查路径和文件是否存在")
        return None

    if img2 is None:
        print(f"错误：无法读取图像 '{img_path2}'，请检查路径和文件是否存在")
        return None

    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 确定目标高度（两张图片中较大的高度）
    target_height = max(h1, h2)

    # 缩放第一张图片（如果需要）
    if h1 != target_height:
        scale = target_height / h1
        new_width = int(w1 * scale)
        img1 = cv2.resize(img1, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        print(f"左图已缩放：原始尺寸({w1}x{h1}) -> 新尺寸({new_width}x{target_height})")
    else:
        new_width = w1

    # 缩放第二张图片（如果需要）
    if h2 != target_height:
        scale = target_height / h2
        new_width2 = int(w2 * scale)
        img2 = cv2.resize(img2, (new_width2, target_height), interpolation=cv2.INTER_LANCZOS4)
        print(f"右图已缩放：原始尺寸({w2}x{h2}) -> 新尺寸({new_width2}x{target_height})")
    else:
        new_width2 = w2

    # 水平拼接（现在两张图片高度相同）
    result = np.hstack((img1, img2))

    # 保存结果
    cv2.imwrite(output_path, result)
    print(f"拼接完成！结果已保存至: {output_path}")
    return result

def stitch_images_in_memory(img1, img2):
    """在内存中拼接两张图片（不保存到文件）"""
    if img1 is None or img2 is None:
        print("错误：输入图像为空")
        return None

    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 确定目标高度（两张图片中较大的高度）
    target_height = max(h1, h2)

    # 缩放第一张图片（如果需要）
    if h1 != target_height:
        scale = target_height / h1
        new_width = int(w1 * scale)
        img1 = cv2.resize(img1, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
    else:
        new_width = w1

    # 缩放第二张图片（如果需要）
    if h2 != target_height:
        scale = target_height / h2
        new_width2 = int(w2 * scale)
        img2 = cv2.resize(img2, (new_width2, target_height), interpolation=cv2.INTER_LANCZOS4)
    else:
        new_width2 = w2

    # 水平拼接（现在两张图片高度相同）
    result = np.hstack((img1, img2))
    return result

def ocr_recognize(image, model, device, transform):
    """对图像进行OCR识别"""
    try:
        # 将OpenCV图像转换为PIL图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb).convert('L')
        
        # 转换为张量
        image_tensor = transform(pil_image)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # 进行推理
        with torch.no_grad():
            preds = model(image_tensor)
            preds_log_softmax = torch.nn.functional.log_softmax(preds, 2)
            text = decode(preds_log_softmax)[0]
        
        return text
    except Exception as e:
        print(f"OCR识别错误: {e}")
        return ""

def process_images_with_ocr(input_folder=None, single_image_path=None, model_path=None, template_files=None, output_base_dir="output", 
                          save_debug=True, save_processed=True, cpu_only=False):
    """整合的图片处理和OCR识别函数"""
    
    # 初始化时间统计
    time_stats = {
        'model_loading': 0,
        'image_reading': 0,
        'preprocessing': 0,
        'template_matching': 0,
        'ocr_recognition': 0,
        'file_saving': 0,
        'total_time': 0
    }
    
    total_start_time = time.time()
    
    # 设置设备
    device = torch.device('cpu') if cpu_only else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载OCR模型
    print("正在加载OCR模型...")
    model_start_time = time.time()
    imgH = 32
    nc = 1
    nh = 256
    model = CRNN(imgH, nc, nclass, nh).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model_end_time = time.time()
    time_stats['model_loading'] = (model_end_time - model_start_time) * 1000
    print(f"模型加载完成，耗时: {time_stats['model_loading']:.2f}ms")
    
    # 设置图像变换
    transform = transforms.Compose([
        transforms.Resize((imgH, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 创建输出目录
    output_dirs = {
        'right': os.path.join(output_base_dir, 'single_Sliced_Right'),
        'left': os.path.join(output_base_dir, 'single_Sliced_Left'),
        'combined': os.path.join(output_base_dir, 'single_combined'),
        'fail': os.path.join(output_base_dir, 'fail'),
        'debug': os.path.join(output_base_dir, 'debug'),
        'preprocess_debug': os.path.join(output_base_dir, 'preprocess_debug')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 定义裁剪区域
    x, y, w, h = 750, 1450, 730, 200
    
    # 读取图片文件
    if single_image_path:
        # 处理单张图片
        image_files = [single_image_path]
        print(f"处理单张图片: {single_image_path}")
    else:
        # 处理文件夹中的所有图片
        image_files = glob.glob(os.path.join(input_folder, "*.png"), recursive=False)
        if not image_files:
            print(f"在文件夹 {input_folder} 中没有找到PNG图片文件")
            return
        print(f"共读取 {len(image_files)} 张图片")
    
    # 预处理所有图像：旋转+裁剪+矫正倾斜
    print("开始图像预处理...")
    preprocessing_start_time = time.time()
    processed_images = []
    for i, img_path in enumerate(image_files):
        # 读取图像时间统计
        read_start_time = time.time()
        img = cv2.imread(img_path)
        read_end_time = time.time()
        time_stats['image_reading'] += (read_end_time - read_start_time) * 1000
        
        if img is None:
            print(f"警告：无法读取图片 {img_path}")
            continue
            
        # 图像预处理时间统计
        preprocess_start_time = time.time()
        rotated_img = cv2.rotate(img, cv2.ROTATE_180)
        cropped_img = rotated_img[y:y + h, x:x + w]
        
        # 矫正倾斜
        deskewed_img, debug_img, angle = deskew_image(cropped_img)
        preprocess_end_time = time.time()
        time_stats['preprocessing'] += (preprocess_end_time - preprocess_start_time) * 1000
        
        processed_images.append(deskewed_img)
        
        # 保存调试图像时间统计
        if save_debug:
            save_start_time = time.time()
            # 为单张图片使用更合适的文件名
            if single_image_path:
                base_name = os.path.splitext(os.path.basename(single_image_path))[0]
                debug_filename = f'{base_name}_deskew.png'
            else:
                debug_filename = f'deskew_{i + 1}.png'
            cv2.imwrite(os.path.join(output_dirs['preprocess_debug'], debug_filename), debug_img)
            save_end_time = time.time()
            time_stats['file_saving'] += (save_end_time - save_start_time) * 1000
        print(f"图片 {i + 1}: 检测到倾斜角度 {angle:.4f} 度，已矫正")
    
    preprocessing_end_time = time.time()
    print(f"图像预处理完成，总耗时: {(preprocessing_end_time - preprocessing_start_time) * 1000:.2f}ms")
    
    # 处理每张图片进行模板匹配和OCR识别
    print("开始模板匹配和OCR识别...")
    results = []
    for j, img in enumerate(processed_images):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found = False
        best_match = None
        
        # 模板匹配时间统计
        template_start_time = time.time()
        # 尝试多个模板
        for template_index, template_file in enumerate(template_files):
            template = cv2.imread(template_file, 0)
            if template is None:
                print(f"警告：无法加载模板图像 {template_file}，跳过")
                continue
            
            h0, w0 = template.shape
            
            # 执行模板匹配
            res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.4
            loc = np.where(res >= threshold)
            
            if len(loc[0]) == 0:
                continue
            
            # 收集所有匹配点及其置信度
            matches = []
            for pt in zip(*loc[::-1]):
                confidence = res[pt[1], pt[0]]
                matches.append((pt, confidence))
            
            matches.sort(key=lambda x: x[1], reverse=True)
            best_pt, best_confidence = matches[0]
            
            if best_match is None or best_confidence > best_match[1]:
                best_match = (best_pt, best_confidence, template_index, template)
            
            found = True
        
        template_end_time = time.time()
        time_stats['template_matching'] += (template_end_time - template_start_time) * 1000
        
        # 如果找到匹配点
        if best_match is not None:
            best_pt, best_confidence, template_index, template = best_match
            h0, w0 = template.shape
            
            # 切割匹配区域
            matched_region_right = img[best_pt[1] + 9:best_pt[1] + h0 - 304,
                                     best_pt[0] + 6:best_pt[0] + w0 - 18]
            
            matched_region_left = img[best_pt[1] + 76:best_pt[1] + h0 - 35,
                                    best_pt[0] - 430:best_pt[0] + w0 - 228]
            
            # 保存匹配区域
            if save_processed:
                save_start_time = time.time()
                if single_image_path:
                    base_name = os.path.splitext(os.path.basename(single_image_path))[0]
                    right_filename = f'{base_name}_right.png'
                    left_filename = f'{base_name}_left.png'
                    combined_filename = f'{base_name}_combined.png'
                else:
                    right_filename = f'1-{j + 1}.png'
                    left_filename = f'1-{j + 1}.png'
                    combined_filename = f'1-{j + 1}.png'
                
                cv2.imwrite(os.path.join(output_dirs['right'], right_filename), matched_region_right)
                cv2.imwrite(os.path.join(output_dirs['left'], left_filename), matched_region_left)
                save_end_time = time.time()
                time_stats['file_saving'] += (save_end_time - save_start_time) * 1000
            
            # 拼接左右图像
            combined_region = stitch_images_in_memory(matched_region_left, matched_region_right)
            
            # 保存拼接后的图像
            if save_processed:
                save_start_time = time.time()
                cv2.imwrite(os.path.join(output_dirs['combined'], combined_filename), combined_region)
                save_end_time = time.time()
                time_stats['file_saving'] += (save_end_time - save_start_time) * 1000
            
            # 对拼接后的图像进行OCR识别
            ocr_start_time = time.time()
            combined_text = ocr_recognize(combined_region, model, device, transform)
            ocr_end_time = time.time()
            time_stats['ocr_recognition'] += (ocr_end_time - ocr_start_time) * 1000
            
            # 绘制匹配区域并保存调试图像
            if save_debug:
                save_start_time = time.time()
                img_annotation = img.copy()
                cv2.rectangle(img_annotation,
                              (best_pt[0], best_pt[1]),
                              (best_pt[0] + w0, best_pt[1] + h0),
                              (0, 255, 0), 2)
                
                text = f"Conf: {best_confidence:.4f}"
                cv2.putText(img_annotation, text,
                            (best_pt[0], best_pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if single_image_path:
                    debug_filename = f'{base_name}_debug.png'
                else:
                    debug_filename = f'1-{j + 1}.png'
                cv2.imwrite(os.path.join(output_dirs['debug'], debug_filename), img_annotation)
                save_end_time = time.time()
                time_stats['file_saving'] += (save_end_time - save_start_time) * 1000
            
            print(f"图片{j + 1}: 使用模板{template_index}找到最佳匹配(置信度:{best_confidence:.6f})")
            print(f"  合并区域OCR结果: {combined_text}")
            
            results.append({
                'image_id': j + 1,
                'template_index': template_index,
                'confidence': best_confidence,
                'combined_text': combined_text
            })
            
        else:
            print(f"图片{j + 1}: 所有模板均未找到匹配区域")
            if save_processed:
                save_start_time = time.time()
                if single_image_path:
                    base_name = os.path.splitext(os.path.basename(single_image_path))[0]
                    failed_filename = f'{base_name}_failed.png'
                else:
                    failed_filename = f'failed-{j + 1}.png'
                cv2.imwrite(os.path.join(output_dirs['fail'], failed_filename), img)
                save_end_time = time.time()
                time_stats['file_saving'] += (save_end_time - save_start_time) * 1000
    
    # 保存OCR结果到文件
    save_start_time = time.time()
    output_file = os.path.join(output_base_dir, "ocr_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("OCR识别结果\n")
        f.write("=" * 50 + "\n")
        for result in results:
            f.write(f"图片{result['image_id']}:\n")
            f.write(f"  模板索引: {result['template_index']}\n")
            f.write(f"  置信度: {result['confidence']:.6f}\n")
            f.write(f"  合并区域识别结果: {result['combined_text']}\n")
            f.write("-" * 30 + "\n")
    save_end_time = time.time()
    time_stats['file_saving'] += (save_end_time - save_start_time) * 1000
    
    # 计算总时间
    total_end_time = time.time()
    time_stats['total_time'] = (total_end_time - total_start_time) * 1000
    
    # 打印时间统计
    print(f"\n处理完成！共处理 {len(results)} 张图片")
    print(f"结果已保存到: {output_file}")
    print("\n" + "="*50)
    print("时间统计 (毫秒):")
    print("="*50)
    print(f"模型加载时间:     {time_stats['model_loading']:>10.2f}ms")
    print(f"图像读取时间:     {time_stats['image_reading']:>10.2f}ms")
    print(f"图像预处理时间:   {time_stats['preprocessing']:>10.2f}ms")
    print(f"模板匹配时间:     {time_stats['template_matching']:>10.2f}ms")
    print(f"OCR识别时间:      {time_stats['ocr_recognition']:>10.2f}ms")
    print(f"文件保存时间:     {time_stats['file_saving']:>10.2f}ms")
    print("-"*50)
    print(f"总处理时间:       {time_stats['total_time']:>10.2f}ms")
    print("="*50)
    
    # 保存时间统计到文件
    time_stats_file = os.path.join(output_base_dir, "time_statistics.txt")
    with open(time_stats_file, 'w', encoding='utf-8') as f:
        f.write("时间统计报告 (毫秒)\n")
        f.write("="*50 + "\n")
        f.write(f"模型加载时间:     {time_stats['model_loading']:>10.2f}ms\n")
        f.write(f"图像读取时间:     {time_stats['image_reading']:>10.2f}ms\n")
        f.write(f"图像预处理时间:   {time_stats['preprocessing']:>10.2f}ms\n")
        f.write(f"模板匹配时间:     {time_stats['template_matching']:>10.2f}ms\n")
        f.write(f"OCR识别时间:      {time_stats['ocr_recognition']:>10.2f}ms\n")
        f.write(f"文件保存时间:     {time_stats['file_saving']:>10.2f}ms\n")
        f.write("-"*50 + "\n")
        f.write(f"总处理时间:       {time_stats['total_time']:>10.2f}ms\n")
        f.write("="*50 + "\n")
    
    print(f"时间统计已保存到: {time_stats_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='整合的图片处理和OCR识别程序')
    parser.add_argument('--input-folder', type=str, help='输入图片文件夹路径')
    parser.add_argument('--single-image', type=str, help='单张图片路径')
    parser.add_argument('model_path', type=str, help='OCR模型路径')
    parser.add_argument('--template', type=str, default='template/template_9.png', 
                       help='模板文件路径 (默认: template/template_9.png)')
    parser.add_argument('--output', type=str, default='output', 
                       help='输出目录 (默认: output)')
    parser.add_argument('--cpu', action='store_true', help='只使用CPU进行推理')
    parser.add_argument('--no-debug', action='store_true', help='不保存调试图像')
    parser.add_argument('--no-processed', action='store_true', help='不保存处理后的图像')
    
    args = parser.parse_args()
    
    # 检查输入参数
    if not args.input_folder and not args.single_image:
        print("错误: 必须指定 --input-folder 或 --single-image 参数")
        return
    
    if args.input_folder and args.single_image:
        print("错误: 不能同时指定 --input-folder 和 --single-image 参数")
        return
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件 {args.model_path} 不存在")
        return
    
    # 检查模板文件是否存在
    if not os.path.exists(args.template):
        print(f"错误: 模板文件 {args.template} 不存在")
        return
    
    # 检查输入路径是否存在
    if args.input_folder and not os.path.exists(args.input_folder):
        print(f"错误: 输入文件夹 {args.input_folder} 不存在")
        return
    
    if args.single_image and not os.path.exists(args.single_image):
        print(f"错误: 输入图片 {args.single_image} 不存在")
        return
    
    # 模板文件列表
    template_files = [args.template]
    
    # 处理图片
    results = process_images_with_ocr(
        input_folder=args.input_folder,
        single_image_path=args.single_image,
        model_path=args.model_path,
        template_files=template_files,
        output_base_dir=args.output,
        save_debug=not args.no_debug,
        save_processed=not args.no_processed,
        cpu_only=args.cpu
    )
    
    if results:
        print("\n识别结果汇总:")
        print("-" * 50)
        for result in results:
            print(f"图片{result['image_id']}: {result['combined_text']}")

if __name__ == '__main__':
    main() 