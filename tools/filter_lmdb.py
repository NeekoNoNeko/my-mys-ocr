import os
import sys
import lmdb
import six
import numpy as np
from PIL import Image
import io

def check_image_is_valid(image_bin):
    if image_bin is None:
        return False
    try:
        image = Image.open(six.BytesIO(image_bin)).convert('L')
        image_np = np.array(image)
        if image_np.shape[0] == 0 or image_np.shape[1] == 0:
            return False
    except Exception:
        return False
    return True

def filter_lmdb_dataset(input_lmdb_path, output_lmdb_path, valid_chars, map_size=1099511627776):
    """
    过滤lmdb数据集，删除包含无效字符的样本
    
    Args:
        input_lmdb_path: 输入lmdb路径
        output_lmdb_path: 输出lmdb路径
        valid_chars: 有效字符集
        map_size: lmdb映射大小
    """
    print(f'读取原始数据集: {input_lmdb_path}')
    input_env = lmdb.open(
        input_lmdb_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False
    )
    
    if not input_env:
        raise RuntimeError('Cannot open input lmdb dataset', input_lmdb_path)
    
    with input_env.begin(write=False) as txn:
        n_samples = int(txn.get('num-samples'.encode()))
        print(f'原始数据集样本数: {n_samples}')
    
    print(f'创建过滤后的数据集: {output_lmdb_path}')
    output_env = lmdb.open(output_lmdb_path, map_size=map_size)
    
    valid_chars_set = set(valid_chars)
    filtered_count = 0
    total_count = 0
    cache = {}
    
    print('开始过滤数据...')
    for idx in range(1, n_samples + 1):
        if idx % 1000 == 0:
            print(f'已处理 {idx}/{n_samples} 个样本，保留 {filtered_count} 个')
        
        with input_env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % idx
            label_key = 'label-%09d'.encode() % idx
            
            imgbuf = txn.get(img_key)
            label = txn.get(label_key).decode('utf-8')
        
        # 检查图片是否有效
        if not check_image_is_valid(imgbuf):
            print(f'跳过无效图片: 样本 {idx}')
            continue
        
        # 检查标签字符是否都在有效字符集中
        invalid_chars = set(label) - valid_chars_set
        if invalid_chars:
            print(f'跳过包含无效字符的样本 {idx}: "{label}" (无效字符: {invalid_chars})')
            continue
        
        # 保留有效样本
        filtered_count += 1
        new_img_key = 'image-%09d' % filtered_count
        new_label_key = 'label-%09d' % filtered_count
        
        cache[new_img_key.encode()] = imgbuf
        cache[new_label_key.encode()] = label.encode('utf-8')
        
        # 每1000个样本写入一次
        if filtered_count % 1000 == 0:
            with output_env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)
            cache = {}
        
        total_count += 1
    
    # 写入剩余的样本
    if cache:
        with output_env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)
    
    # 写入样本总数
    with output_env.begin(write=True) as txn:
        txn.put('num-samples'.encode(), str(filtered_count).encode())
    
    print(f'过滤完成！')
    print(f'原始样本数: {n_samples}')
    print(f'保留样本数: {filtered_count}')
    print(f'删除样本数: {n_samples - filtered_count}')
    print(f'保留比例: {filtered_count/n_samples*100:.2f}%')
    
    input_env.close()
    output_env.close()

def main():
    if len(sys.argv) != 4:
        print('用法: python filter_lmdb.py 输入lmdb路径 输出lmdb路径 有效字符集')
        print('示例: python filter_lmdb.py data/train.lmdb data/train_filtered.lmdb "0123456789abcdefghijklmnopqrstuvwxyz"')
        sys.exit(1)
    
    input_path, output_path, valid_chars = sys.argv[1:4]
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f'错误: 输入文件不存在: {input_path}')
        sys.exit(1)
    
    # 检查输出目录是否存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f'有效字符集: "{valid_chars}"')
    print(f'字符集长度: {len(valid_chars)}')
    
    filter_lmdb_dataset(input_path, output_path, valid_chars)

if __name__ == '__main__':
    main() 