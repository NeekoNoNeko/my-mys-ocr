import cv2
import numpy as np
import glob
import os



# 定义裁剪区域：左上角坐标 (x, y)，宽度 w，高度 h
x, y, w, h = 750, 1450, 730, 200

# 模板文件列表（按优先级顺序）
template_files = [
    'template/template_9.png',
    # 可以继续添加更多模板
]

# 确保输出目录存在
output_dir_Right = 'single_Sliced_Right'
output_dir_Left = 'single_Sliced_Left'
output_dir_fail = 'fail'
os.makedirs(output_dir_Right, exist_ok=True)
os.makedirs(output_dir_Left, exist_ok=True)
os.makedirs(output_dir_fail, exist_ok=True)
os.makedirs('debug', exist_ok=True)  # 确保debug目录存在
os.makedirs('preprocess_debug', exist_ok=True)  # 确保预处理调试目录存在

# 指定目录路径
folder_path = "single_p/"

# 使用glob匹配所有.png文件
image_files = glob.glob(folder_path + "*.png", recursive=False)

# 读取所有匹配的图片
images = []
for img_path in image_files:
    img = cv2.imread(img_path)
    if img is not None:
        images.append(img)
    else:
        print(f"警告：无法读取图片 {img_path}")

print(f"共读取 {len(images)} 张图片")


# 增强的矫正图片倾斜函数（提高0.3°-1.7°范围的精度）
def deskew_image(image):
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, 40, minLineLength=110, maxLineGap=12)  # 更精细的角度分辨率

    # 创建调试图像
    debug_img = image.copy()
    height, width = image.shape[:2]
    bottom_start = int(height * 0.6)  # 从高度70%处开始

    # 计算左侧区域边界（图像宽度的70%）
    left_region_boundary = int(width * 0.65)  # 65%宽度处

    # 在调试图像上添加底部参考线和文本
    cv2.line(debug_img, (0, bottom_start), (width, bottom_start), (255, 0, 0), 2)  # 绘制底部区域参考线
    cv2.putText(debug_img, "Bottom Reference Line", (10, bottom_start - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # 绘制左侧区域边界
    cv2.line(debug_img, (left_region_boundary, 0), (left_region_boundary, height), (0, 255, 255), 1)
    cv2.putText(debug_img, "Left Region Boundary", (left_region_boundary + 5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 如果没有检测到直线
    if lines is None:
        # 添加角度文本（0度）
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
                mid_x < left_region_boundary):  # 只考虑左侧区域的直线
            # 过滤掉大角度的线（角度大于3度）
            angle = np.abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 3:  # 忽略近似垂直的线
                bottom_lines.append(line[0])

    # 如果没有合适的底部直线
    if not bottom_lines:
        # 添加角度文本（0度）
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
        weighted_angles.extend([angle] * int(length / 10))  # 每10像素长度增加一个权重

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
        # 添加矫正状态文本
        cv2.putText(debug_img, "Correction: Not needed (angle < 0.05 degrees)", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return image, debug_img, median_angle

    # 旋转图像以矫正倾斜（使用更精确的插值方法）
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, rotation_matrix, (width, height),
                              flags=cv2.INTER_LANCZOS4,  # 使用更高质量的插值
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


# 预处理所有图像：旋转+裁剪+矫正倾斜
image_list = []
for i, img in enumerate(images):
    rotated_img = cv2.rotate(img, cv2.ROTATE_180)
    cropped_img = rotated_img[y:y + h, x:x + w]

    # 矫正倾斜
    deskewed_img, debug_img, angle = deskew_image(cropped_img)
    image_list.append(deskewed_img)

    # 保存调试图像
    cv2.imwrite(f'preprocess_debug/deskew_{i + 1}.png', debug_img)
    print(f"图片 {i + 1}: 检测到倾斜角度 {angle:.4f} 度，已矫正")

# 处理每张图片
for j, img in enumerate(image_list):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = False  # 标记是否找到匹配点
    best_match = None  # 存储最佳匹配信息

    # 尝试多个模板
    for template_index, template_file in enumerate(template_files):
        # 加载模板图像
        template = cv2.imread(template_file, 0)
        if template is None:
            print(f"警告：无法加载模板图像 {template_file}，跳过")
            continue

        h0, w0 = template.shape

        # 执行模板匹配
        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.4
        loc = np.where(res >= threshold)

        # 如果没有匹配点，继续下一个模板
        if len(loc[0]) == 0:
            continue

        # 收集所有匹配点及其置信度
        matches = []
        for pt in zip(*loc[::-1]):  # pt是(x, y)
            confidence = res[pt[1], pt[0]]  # 获取匹配置信度
            matches.append((pt, confidence))

        # 按置信度降序排序
        matches.sort(key=lambda x: x[1], reverse=True)

        # 选择置信度最高的匹配点
        best_pt, best_confidence = matches[0]

        # 更新全局最佳匹配（选择所有模板中置信度最高的匹配）
        if best_match is None or best_confidence > best_match[1]:
            best_match = (best_pt, best_confidence, template_index, template)

        # 标记已找到匹配
        found = True

    # 如果找到匹配点
    if best_match is not None:
        best_pt, best_confidence, template_index, template = best_match
        h0, w0 = template.shape

        # 切割并保存最佳匹配区域
        matched_region_Right = img[best_pt[1] + 9:best_pt[1] + h0 - 304,
                               best_pt[0] + 6:best_pt[0] + w0 - 18]

        matched_region_Left = img[best_pt[1] + 76:best_pt[1] + h0 - 35,
                              best_pt[0] - 430:best_pt[0] + w0 - 228]

        # 绘制匹配区域并保存调试图像
        img_Annotation = img.copy()
        cv2.rectangle(img_Annotation,
                      (best_pt[0], best_pt[1]),
                      (best_pt[0] + w0, best_pt[1] + h0),
                      (0, 255, 0), 2)

        # 添加置信度文本
        text = f"Conf: {best_confidence:.4f}"  # 提高精度显示
        cv2.putText(img_Annotation, text,
                    (best_pt[0], best_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(f'debug/1-{j + 1}.png', img_Annotation)
        cv2.imwrite(f'{output_dir_Right}/1-{j + 1}.png', matched_region_Right)
        cv2.imwrite(f'{output_dir_Left}/1-{j + 1}.png', matched_region_Left)
        print(f"右图{j + 1}: 使用模板{template_index}找到最佳匹配(置信度:{best_confidence:.6f})")

    # 如果所有模板都尝试后仍未找到匹配点
    else:
        print(f"右图{j + 1}: 所有模板均未找到匹配区域")
        # 保存原始裁剪图像用于调试
        cv2.imwrite(f'{output_dir_fail}/failed-{j + 1}.png', img)


def stitch_images(img_path1, img_path2, output_path):
    # 读取图像并检查是否成功
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None:
        print(f"错误：无法读取图像 '{img_path1}'，请检查路径和文件是否存在")
        return

    if img2 is None:
        print(f"错误：无法读取图像 '{img_path2}'，请检查路径和文件是否存在")
        return

    # 获取图像尺寸
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 确定目标高度（两张图片中较大的高度）
    target_height = max(h1, h2)

    # 缩放第一张图片（如果需要）
    if h1 != target_height:
        # 计算缩放比例（保持宽高比）
        scale = target_height / h1
        new_width = int(w1 * scale)
        img1 = cv2.resize(img1, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)  # 更高质量的缩放
        print(f"左图已缩放：原始尺寸({w1}x{h1}) -> 新尺寸({new_width}x{target_height})")
    else:
        new_width = w1

    # 缩放第二张图片（如果需要）
    if h2 != target_height:
        # 计算缩放比例（保持宽高比）
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


# 确保输出目录存在
os.makedirs('single_combined', exist_ok=True)

# 使用示例（使用绝对路径更可靠）
for con in range(len(image_list)):
    left_path = f'single_Sliced_Left/1-{con + 1}.png'
    right_path = f'single_Sliced_Right/1-{con + 1}.png'
    output_path = f'single_combined/1-{con + 1}.png'

    # 检查文件是否存在
    if os.path.exists(left_path) and os.path.exists(right_path):
        stitch_images(left_path, right_path, output_path)
    else:
        print(f"警告：无法找到文件 {left_path} 或 {right_path}，跳过拼接")

print("处理完成")