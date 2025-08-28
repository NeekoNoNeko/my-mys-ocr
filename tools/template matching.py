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

# 预处理所有图像：旋转+裁剪
image_list = []
for i in images:
    rotated_img = cv2.rotate(i, cv2.ROTATE_180)
    cropped_img = rotated_img[y:y + h, x:x + w]
    image_list.append(cropped_img)

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

        matched_region_Left = img[best_pt[1] + 70:best_pt[1] + h0 - 35,
                              best_pt[0] - 430:best_pt[0] + w0 - 228]

        # 绘制匹配区域并保存调试图像
        img_Annotation = img.copy()
        cv2.rectangle(img_Annotation,
                      (best_pt[0], best_pt[1]),
                      (best_pt[0] + w0, best_pt[1] + h0),
                      (0, 255, 0), 2)

        # 添加置信度文本
        text = f"Conf: {best_confidence:.2f}"
        cv2.putText(img_Annotation, text,
                    (best_pt[0], best_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imwrite(f'debug/1-{j + 1}.png', img_Annotation)
        cv2.imwrite(f'{output_dir_Right}/1-{j + 1}.png', matched_region_Right)
        cv2.imwrite(f'{output_dir_Left}/1-{j + 1}.png', matched_region_Left)
        print(f"右图{j + 1}: 使用模板{template_index}找到最佳匹配(置信度:{best_confidence:.4f})")

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

    # 获取高度并处理差异
    h3, w3 = img1.shape[:2]
    h4, w4 = img2.shape[:2]

    # 选择最大高度
    max_height = max(h3, h4)

    # 创建填充后的新图像（顶部对齐）
    padded1 = np.zeros((max_height, w3, 3), dtype=np.uint8)
    padded1[:h3, :w3] = img1

    padded2 = np.zeros((max_height, w4, 3), dtype=np.uint8)
    padded2[:h4, :w4] = img2

    # 水平拼接
    result = np.hstack((padded1, padded2))

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