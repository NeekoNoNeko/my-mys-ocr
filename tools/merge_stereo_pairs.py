# %%
# 在第一个单元格添加如下代码
import matplotlib.pyplot as plt
import cv2
import os
import re

# 读取图片（假设图片路径为'test/1.png'，如需更换请修改路径）
# 定义图片目录
left_dir = '/root/workspace/data/yolo_out/left/'
right_dir = '/root/workspace/data/yolo_out/right/'

# 匹配文件名的正则表达式
left_pattern = re.compile(r'(\d+)_left_\d+\.png')
right_pattern = re.compile(r'(\d+)_right_\d+\.png')

# 获取所有left和right图片文件
left_files = [f for f in os.listdir(left_dir) if left_pattern.match(f)]
right_files = [f for f in os.listdir(right_dir) if right_pattern.match(f)]

# 构建x到文件名的映射
def build_x_map(files, pattern):
    x_map = {}
    for f in files:
        m = pattern.match(f)
        if m:
            x = m.group(1)
            if x not in x_map:
                x_map[x] = []
            x_map[x].append(f)
    return x_map

left_x_map = build_x_map(left_files, left_pattern)
right_x_map = build_x_map(right_files, right_pattern)

# 找到x相同的对，y和z不影响，按x遍历
output_dir = 'origin_merged_output'
os.makedirs(output_dir, exist_ok=True)
for x in sorted(set(left_x_map.keys()) & set(right_x_map.keys()), key=int):
    # 取每个x下的第一个left和第一个right（如需全部组合可嵌套循环）
    left_img_path = os.path.join(left_dir, left_x_map[x][0])
    right_img_path = os.path.join(right_dir, right_x_map[x][0])

    imgL = cv2.imread(left_img_path)
    # OpenCV读取的图片是BGR格式，需转换为RGB
    imgL_rgb = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    imgR = cv2.imread(right_img_path)
    imgR_rgb = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

    # %%
    from PIL import Image
    import numpy as np

    def crop_image_border_pil_L(img, border=14):
        """
        用PIL裁剪图片四周一圈指定像素
        :param img: numpy.ndarray格式的图片
        :param border: 需要裁剪的像素宽度
        :return: 裁剪后的图片（numpy.ndarray）
        """
        pil_img = Image.fromarray(img)
        width, height = pil_img.size
        box = (border + 15, border, width - border + 15 , height - border)
        cropped_img = pil_img.crop(box)
        return np.array(cropped_img)

    def crop_image_border_pil_R(img, border=14):
        """
        用PIL裁剪图片四周一圈指定像素
        :param img: numpy.ndarray格式的图片
        :param border: 需要裁剪的像素宽度
        :return: 裁剪后的图片（numpy.ndarray）
        """
        pil_img = Image.fromarray(img)
        width, height = pil_img.size
        box = (border + 8, border, width - border , height - border)
        cropped_img = pil_img.crop(box)
        return np.array(cropped_img)

    imgL_cropped = crop_image_border_pil_L(imgL_rgb)

    imgR_cropped = crop_image_border_pil_R(imgR_rgb)

    # %%
    # 假设 imgR_cropped 是 numpy.ndarray 格式的图片
    img_bottom45 = imgR_cropped[-40:, :]

    import cv2
    imgL_rotated = cv2.rotate(imgL_cropped, cv2.ROTATE_180)
    imgR_rotated = cv2.rotate(img_bottom45, cv2.ROTATE_180)

    import numpy as np

    # 获取两张图片的高度
    h1 = imgL_rotated.shape[0]
    h2 = imgR_rotated.shape[0]
    max_height = max(h1, h2)

    # 计算需要填充的高度
    pad_L = max_height - h1
    pad_R = max_height - h2

    # 对较矮的图片在底部填充黑色像素
    if pad_L > 0:
        imgL_padded = np.pad(imgL_rotated, ((0, pad_L), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        imgL_padded = imgL_rotated

    if pad_R > 0:
        imgR_padded = np.pad(imgR_rotated, ((0, pad_R), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        imgR_padded = imgR_rotated

    # 水平拼接
    img_merged = np.concatenate((imgL_padded, imgR_padded), axis=1)

    out_name = os.path.join(output_dir, f'merged_{x}.png')
    cv2.imwrite(out_name, img_merged)


