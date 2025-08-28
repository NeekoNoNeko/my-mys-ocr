import os
import sys
import lmdb
import six
import numpy as np
import random
import argparse
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time

def get_available_disk_space(path):
    """获取指定路径的可用磁盘空间（字节）"""
    try:
        total, used, free = shutil.disk_usage(path)
        return free
    except:
        return None

def calculate_smart_map_size(input_lmdb_path, output_dir, safety_factor=3):
    """
    智能计算map_size

    Args:
        input_lmdb_path: 输入LMDB路径
        output_dir: 输出目录
        safety_factor: 安全系数，建议2-4倍

    Returns:
        适合的map_size值
    """
    try:
        # 获取输入LMDB文件大小
        if os.path.isdir(input_lmdb_path):
            input_size = 0
            for root, dirs, files in os.walk(input_lmdb_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        input_size += os.path.getsize(file_path)
        else:
            input_size = os.path.getsize(input_lmdb_path)

        # 获取可用磁盘空间
        available_space = get_available_disk_space(output_dir)

        if available_space is None:
            # 如果无法获取磁盘空间，使用保守估计
            suggested_size = max(input_size * safety_factor, 1024 * 1024 * 1024)  # 至少1GB
        else:
            # 使用输入大小的safety_factor倍，但不超过可用空间的一半
            suggested_size = min(input_size * safety_factor, available_space // 2)
            # 确保至少有512MB
            suggested_size = max(suggested_size, 512 * 1024 * 1024)

        print(f"输入LMDB大小: {input_size / (1024*1024):.1f} MB")
        if available_space:
            print(f"可用磁盘空间: {available_space / (1024*1024):.1f} MB")
        print(f"建议map_size: {suggested_size / (1024*1024):.1f} MB")

        return suggested_size

    except Exception as e:
        print(f"计算map_size时出错: {e}")
        # 返回一个保守的默认值
        return 1024 * 1024 * 1024  # 1GB

def split_lmdb_dataset(input_lmdb_path, train_lmdb_path, val_lmdb_path, val_ratio=0.2, seed=42, map_size=None, num_threads=4, batch_size=500):
    """
    将一个lmdb数据集划分为训练集和验证集（支持多线程处理）

    Args:
        input_lmdb_path: 输入lmdb路径
        train_lmdb_path: 训练集lmdb输出路径
        val_lmdb_path: 验证集lmdb输出路径
        val_ratio: 验证集比例，默认0.2（20%）
        seed: 随机种子，确保结果可重现
        map_size: lmdb映射大小，如果为None则自动计算
        num_threads: 线程数量，默认4
        batch_size: 批处理大小，默认500
    """
    print(f'读取原始数据集: {input_lmdb_path}')
    print(f'使用线程数: {num_threads}')
    print(f'批处理大小: {batch_size}')

    # 确保输出目录存在
    for output_path in [train_lmdb_path, val_lmdb_path]:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

    # 如果没有指定map_size，则智能计算
    if map_size is None:
        output_dir = os.path.dirname(train_lmdb_path) or '.'
        map_size = calculate_smart_map_size(input_lmdb_path, output_dir)

    print(f'使用map_size: {map_size / (1024*1024):.1f} MB')

    # 打开输入LMDB（允许多个读取器）
    input_env = lmdb.open(
        input_lmdb_path,
        max_readers=num_threads * 2,  # 允许更多读取器
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

    # 设置随机种子
    random.seed(seed)

    # 生成随机索引
    indices = list(range(1, n_samples + 1))
    random.shuffle(indices)

    # 计算分割点
    val_size = int(n_samples * val_ratio)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    print(f'训练集样本数: {len(train_indices)}')
    print(f'验证集样本数: {len(val_indices)}')
    print(f'验证集比例: {val_ratio:.2%}')

    # 创建输出LMDB环境
    try:
        train_env = lmdb.open(train_lmdb_path, map_size=map_size)
        val_env = lmdb.open(val_lmdb_path, map_size=map_size)
    except lmdb.Error as e:
        print(f'创建LMDB失败: {e}')
        print('尝试解决方案:')
        print('1. 清理磁盘空间')
        print('2. 使用更小的map_size参数')
        print('3. 将数据移动到有更多空间的磁盘')

        # 尝试使用更小的map_size
        smaller_map_size = map_size // 2
        print(f'尝试使用更小的map_size: {smaller_map_size / (1024*1024):.1f} MB')
        try:
            train_env = lmdb.open(train_lmdb_path, map_size=smaller_map_size)
            val_env = lmdb.open(val_lmdb_path, map_size=smaller_map_size)
            map_size = smaller_map_size
            print('使用更小的map_size成功创建LMDB')
        except Exception as e2:
            print(f'仍然失败: {e2}')
            input_env.close()
            return
    except Exception as e:
        print(f'创建LMDB时发生未知错误: {e}')
        input_env.close()
        return

    # 使用多线程处理
    start_time = time.time()

    try:
        # 处理训练集
        print('开始处理训练集...')
        train_count = process_dataset_parallel(
            input_env, train_env, train_indices, 
            num_threads, batch_size, "训练集"
        )

        # 处理验证集
        print('开始处理验证集...')
        val_count = process_dataset_parallel(
            input_env, val_env, val_indices, 
            num_threads, batch_size, "验证集"
        )

        # 写入样本总数
        with train_env.begin(write=True) as txn:
            txn.put('num-samples'.encode(), str(train_count).encode())

        with val_env.begin(write=True) as txn:
            txn.put('num-samples'.encode(), str(val_count).encode())

        end_time = time.time()

        print(f'划分完成！耗时: {end_time - start_time:.2f}秒')
        print(f'原始样本数: {n_samples}')
        print(f'训练集样本数: {train_count}')
        print(f'验证集样本数: {val_count}')
        print(f'训练集比例: {train_count/n_samples:.2%}')
        print(f'验证集比例: {val_count/n_samples:.2%}')
        print(f'处理速度: {n_samples/(end_time - start_time):.1f} 样本/秒')

    except Exception as e:
        print(f'处理过程中发生错误: {e}')
    finally:
        input_env.close()
        train_env.close()
        val_env.close()


def read_batch_data(input_env, indices_batch):
    """读取一批数据"""
    batch_data = []

    with input_env.begin(write=False) as txn:
        for idx in indices_batch:
            img_key = 'image-%09d'.encode() % idx
            label_key = 'label-%09d'.encode() % idx

            imgbuf = txn.get(img_key)
            label = txn.get(label_key)

            if imgbuf is not None and label is not None:
                batch_data.append((imgbuf, label))

    return batch_data


def write_batch_data(output_env, batch_data, start_count, write_lock):
    """写入一批数据（线程安全）"""
    cache = {}

    # 准备要写入的数据
    for i, (imgbuf, label) in enumerate(batch_data):
        count = start_count + i + 1
        img_key = ('image-%09d' % count).encode()
        label_key = ('label-%09d' % count).encode()

        cache[img_key] = imgbuf
        cache[label_key] = label

    # 线程安全地写入数据
    with write_lock:
        with output_env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)

    return len(batch_data)


def process_dataset_parallel(input_env, output_env, indices, num_threads, batch_size, dataset_name):
    """并行处理数据集"""
    total_indices = len(indices)
    processed_count = 0
    write_lock = threading.Lock()
    progress_lock = threading.Lock()

    # 将索引分批
    batches = []
    for i in range(0, total_indices, batch_size):
        batch_indices = indices[i:i + batch_size]
        batches.append((batch_indices, i))  # (索引批次, 起始计数)

    print(f'{dataset_name}: 分为 {len(batches)} 批，每批 {batch_size} 个样本')

    def process_batch(batch_info):
        nonlocal processed_count
        batch_indices, start_idx = batch_info

        try:
            # 读取数据
            batch_data = read_batch_data(input_env, batch_indices)

            if not batch_data:
                return 0

            # 写入数据
            written_count = write_batch_data(output_env, batch_data, start_idx, write_lock)

            # 更新进度
            with progress_lock:
                processed_count += written_count
                if processed_count % (batch_size * 5) == 0 or processed_count == total_indices:
                    print(f'{dataset_name}: {processed_count}/{total_indices} '
                          f'({processed_count/total_indices*100:.1f}%)')

            return written_count

        except Exception as e:
            print(f'处理批次时出错: {e}')
            return 0

    # 使用线程池处理
    total_written = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_batch = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                written_count = future.result()
                total_written += written_count
            except Exception as exc:
                print(f'批次 {batch} 处理时产生异常: {exc}')

    return total_written

def main():
    parser = argparse.ArgumentParser(description='Split LMDB dataset into train and validation sets (with multi-threading support)')
    parser.add_argument('input_lmdb', help='输入lmdb路径')
    parser.add_argument('train_lmdb', help='训练集lmdb输出路径')
    parser.add_argument('val_lmdb', help='验证集lmdb输出路径')
    parser.add_argument('--val-ratio', type=float, default=0.2, 
                       help='验证集比例 (默认: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--threads', type=int, default=4,
                       help='线程数量 (默认: 4)')
    parser.add_argument('--batch-size', type=int, default=500,
                       help='批处理大小 (默认: 500)')

    args = parser.parse_args()

    # 检查输入文件是否存在
    if not os.path.exists(args.input_lmdb):
        print(f'错误: 输入文件不存在: {args.input_lmdb}')
        sys.exit(1)

    # 验证参数
    if args.threads < 1:
        print('错误: 线程数量必须至少为1')
        sys.exit(1)

    if args.batch_size < 1:
        print('错误: 批处理大小必须至少为1')
        sys.exit(1)

    # 检查输出目录是否存在
    for output_path in [args.train_lmdb, args.val_lmdb]:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_path):
            os.makedirs(output_dir, exist_ok=True)

    print(f'验证集比例: {args.val_ratio:.2%}')
    print(f'随机种子: {args.seed}')
    print(f'线程数量: {args.threads}')
    print(f'批处理大小: {args.batch_size}')

    split_lmdb_dataset(args.input_lmdb, args.train_lmdb, args.val_lmdb, 
                       args.val_ratio, args.seed, num_threads=args.threads, 
                       batch_size=args.batch_size)

if __name__ == '__main__':
    main() 
    # 基本用法: python tools/split_lmdb.py data/dataset.lmdb data/train.lmdb data/val.lmdb
    # 使用多线程: python tools/split_lmdb.py data/dataset.lmdb data/train.lmdb data/val.lmdb --threads 8 --batch-size 1000
    # 自定义验证集比例: python tools/split_lmdb.py data/dataset.lmdb data/train.lmdb data/val.lmdb --val-ratio 0.3 --threads 6