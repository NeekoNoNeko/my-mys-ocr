import os

# 目标图片文件夹
img_dir = '/root/workspace/data/template/augmented'
# 输出标签文件
label_txt = '/root/workspace/data/template/augmented/labels.txt'
# 固定标签
fixed_label = 'OP122/4021311OADC1@E3'

# 支持的图片扩展名
img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}

img_files = [f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in img_exts]
img_files.sort()

with open(label_txt, 'w', encoding='utf-8') as f:
    for img_name in img_files:
        f.write(f'{img_name}\t{fixed_label}\n')

print(f'已生成标签文件: {label_txt}，共 {len(img_files)} 条') 