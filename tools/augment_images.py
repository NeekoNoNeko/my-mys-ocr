import cv2
import os
import numpy as np
import albumentations as A
from tqdm import tqdm
from PIL import Image, ImageOps
import random

# 固定随机种子，保证可复现
random.seed(42)
np.random.seed(42)

input_dir  = "/root/workspace/data/template"
output_dir = "/root/workspace/data/template/augmented"
os.makedirs(output_dir, exist_ok=True)

transform = A.Compose([
    A.Rotate(limit=2, p=0.5),
    A.Affine(translate_percent=(-0.05,0.05), scale=(0.9,1.1), rotate=0, p=0.7),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.RandomGamma(gamma_limit=(80,120), p=0.3),
    A.ISONoise(intensity=(0.05, 0.1), p=0.3)
], additional_targets=None, is_check_shapes=True)

aug_per_image = 10

for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(input_dir, fname)

    # 用 Pillow 打开，保留 EXIF 方向
    img_pil = Image.open(img_path)
    img_pil = ImageOps.exif_transpose(img_pil)
    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # 灰度图转三通道
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for i in range(aug_per_image):
        aug = transform(image=image)['image']
        aug = np.clip(aug, 0, 255).astype(np.uint8)
        out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}_aug_{i}.png")
        cv2.imencode(".png", aug)[1].tofile(out_path)   # 支持中文路径