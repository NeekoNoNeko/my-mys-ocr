import torch
from torchvision import transforms
from PIL import Image
from model import CRNN
import sys
import os
import glob

# 字符集（需与训练时一致）
# CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/'
# CHARS = "OP12/403ADC@E"
CHARS = "01234679ADEMZ" # 429 E7
# CHARS = "-0123469ADFMTZ" # 429 FT1
# CHARS = "()0123459ADEgIMnR.<>VZ" 552 E2
CHARS = "()-.><0123456789ABDEFIMRTVZgn"
BLANK = '─'  # 使用特殊符号作为填充符，避免与实际字符冲突
CHARS = BLANK + CHARS
nclass = len(CHARS)
idx2char = {i: c for i, c in enumerate(CHARS)}

def decode(preds):
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

def infer_single(img_path, model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgH = 32
    nc = 1
    nh = 256
    model = CRNN(imgH, nc, nclass, nh).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((imgH, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = Image.open(img_path).convert('L')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(image_tensor)
        preds_log_softmax = torch.nn.functional.log_softmax(preds, 2)
        text = decode(preds_log_softmax)[0]
    return text

def infer_folder(folder_path, model_path):
    """识别文件夹下的所有图片"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgH = 32
    nc = 1
    nh = 256
    model = CRNN(imgH, nc, nclass, nh).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((imgH, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"在文件夹 {folder_path} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    print("-" * 60)
    
    results = []
    for i, img_path in enumerate(image_files, 1):
        try:
            image = Image.open(img_path).convert('L')
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds = model(image_tensor)
                preds_log_softmax = torch.nn.functional.log_softmax(preds, 2)
                text = decode(preds_log_softmax)[0]
            
            filename = os.path.basename(img_path)
            print(f"{i:3d}. {filename:<30} -> {text}")
            results.append((filename, text))
            
        except Exception as e:
            print(f"{i:3d}. {os.path.basename(img_path):<30} -> 错误: {e}")
    
    print("-" * 60)
    print(f"识别完成，共处理 {len(results)} 个文件")
    
    # 保存结果到文件
    output_file = os.path.join(folder_path, "ocr_results.txt")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("OCR识别结果\n")
        f.write("=" * 50 + "\n")
        for filename, text in results:
            f.write(f"{filename}\t{text}\n")
    print(f"结果已保存到: {output_file}")

def infer(img_path, model_path):
    """兼容单个文件的推理"""
    text = infer_single(img_path, model_path)
    print(f'图片: {img_path} 识别结果: {text}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='CRNN OCR 推理')
    parser.add_argument('path', type=str, nargs='?', help='图片路径或文件夹路径')
    parser.add_argument('model_path', type=str, nargs='?', help='模型路径')
    parser.add_argument('--cpu', action='store_true', help='只用CPU进行推理')
    args = parser.parse_args()

    if args.path is None or args.model_path is None:
        print('用法:')
        print('  单个文件: python infer.py 图片路径 模型路径 [--cpu]')
        print('  整个文件夹: python infer.py 文件夹路径 模型路径 [--cpu]')
        print('示例:')
        print('  python infer.py crnn/1-1.png checkpoints/crnn_best.pth')
        print('  python infer.py ./test_images/ checkpoints/crnn_best.pth --cpu')
    else:
        # 选择设备
        if args.cpu:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        path = args.path
        model_path = args.model_path

        def infer_single_with_device(img_path, model_path, device):
            imgH = 32
            nc = 1
            nh = 256
            model = CRNN(imgH, nc, nclass, nh).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            transform = transforms.Compose([
                transforms.Resize((imgH, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            image = Image.open(img_path).convert('L')
            image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(image_tensor)
                preds_log_softmax = torch.nn.functional.log_softmax(preds, 2)
                text = decode(preds_log_softmax)[0]
            return text

        def infer_folder_with_device(folder_path, model_path, device):
            imgH = 32
            nc = 1
            nh = 256
            model = CRNN(imgH, nc, nclass, nh).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            transform = transforms.Compose([
                transforms.Resize((imgH, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(folder_path, ext)))
                image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
            if not image_files:
                print(f"在文件夹 {folder_path} 中没有找到图片文件")
                return
            print(f"找到 {len(image_files)} 个图片文件")
            print("-" * 60)
            results = []
            for i, img_path in enumerate(image_files, 1):
                try:
                    image = Image.open(img_path).convert('L')
                    image_tensor = transform(image)
                    image_tensor = image_tensor.unsqueeze(0).to(device)
                    with torch.no_grad():
                        preds = model(image_tensor)
                        preds_log_softmax = torch.nn.functional.log_softmax(preds, 2)
                        text = decode(preds_log_softmax)[0]
                    filename = os.path.basename(img_path)
                    print(f"{i:3d}. {filename:<30} -> {text}")
                    results.append((filename, text))
                except Exception as e:
                    print(f"{i:3d}. {os.path.basename(img_path):<30} -> 错误: {e}")
            print("-" * 60)
            print(f"识别完成，共处理 {len(results)} 个文件")
            output_file = os.path.join(folder_path, "ocr_results.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("OCR识别结果\n")
                f.write("=" * 50 + "\n")
                for filename, text in results:
                    f.write(f"{filename}\t{text}\n")
            print(f"结果已保存到: {output_file}")

        if os.path.isfile(path):
            text = infer_single_with_device(path, model_path, device)
            print(f'图片: {path} 识别结果: {text}')
        elif os.path.isdir(path):
            infer_folder_with_device(path, model_path, device)
        else:
            print(f"错误: {path} 不是有效的文件或文件夹路径") 