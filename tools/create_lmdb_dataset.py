import os
import sys
import lmdb
import glob
import six
import numpy as np
from PIL import Image
import io
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time


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

def get_available_disk_space(path):
    """获取指定路径的可用磁盘空间（字节）"""
    try:
        total, used, free = shutil.disk_usage(path)
        return free
    except:
        return None

def process_image_task(task_data):
    """处理单个图片的工作函数"""
    cnt, img_path, label = task_data

    result = {
        'cnt': cnt,
        'success': False,
        'image_key': None,
        'label_key': None,
        'image_bin': None,
        'label_data': None,
        'error': None
    }

    try:
        if not os.path.exists(img_path):
            result['error'] = f'图片不存在: {img_path}'
            return result

        with open(img_path, 'rb') as f_img:
            image_bin = f_img.read()

        if not check_image_is_valid(image_bin):
            result['error'] = f'无效图片: {img_path}'
            return result

        result['success'] = True
        result['image_key'] = 'image-%09d' % cnt
        result['label_key'] = 'label-%09d' % cnt
        result['image_bin'] = image_bin
        result['label_data'] = label.encode('utf-8')

    except Exception as e:
        result['error'] = f'读取图片失败 {img_path}: {e}'

    return result

def estimate_dataset_size(img_dir, label_txt):
    """估算数据集大小"""
    total_size = 0
    try:
        with open(label_txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 取样前100个图片来估算平均大小
        sample_count = 0
        sample_size = 0
        for line in lines[:100]:
            line = line.strip()
            if not line:
                continue
            img_name = line.split('\t', 1)[0]
            img_path = os.path.join(img_dir, img_name)
            if os.path.exists(img_path):
                sample_size += os.path.getsize(img_path)
                sample_count += 1

        if sample_count > 0:
            avg_img_size = sample_size / sample_count
            total_images = len([line for line in lines if line.strip()])
            # 估算总大小（图片 + 标签 + 额外开销），乘以2作为安全系数
            estimated_size = int(total_images * avg_img_size * 2)
            return estimated_size
    except:
        pass

    # 如果估算失败，返回一个保守的默认值（1GB）
    return 1024 * 1024 * 1024

def create_lmdb_dataset(output_path, img_dir, label_txt, map_size=None, num_workers=4):
    print(f'写入LMDB到 {output_path} ...')
    print(f'使用 {num_workers} 个线程进行并行处理')

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f'创建输出目录: {output_dir}')
        except Exception as e:
            print(f'无法创建输出目录 {output_dir}: {e}')
            return

    # 如果没有指定map_size，则自动计算
    if map_size is None:
        estimated_size = estimate_dataset_size(img_dir, label_txt)
        available_space = get_available_disk_space(output_dir or '.')

        if available_space:
            # 使用估算大小和可用空间的较小值，但至少1GB
            map_size = min(estimated_size, available_space // 2)
            map_size = max(map_size, 1024 * 1024 * 1024)  # 至少1GB
            print(f'自动设置map_size: {map_size // (1024*1024)} MB')
        else:
            map_size = 2 * 1024 * 1024 * 1024  # 默认2GB
            print(f'使用默认map_size: {map_size // (1024*1024)} MB')

    # 尝试打开LMDB
    try:
        env = lmdb.open(output_path, map_size=map_size)
    except lmdb.Error as e:
        print(f'LMDB打开失败: {e}')
        print('可能的原因:')
        print('1. 磁盘空间不足')
        print('2. map_size设置过大')
        print('3. 没有写入权限')

        # 尝试使用更小的map_size
        smaller_map_size = 1024 * 1024 * 1024  # 1GB
        print(f'尝试使用更小的map_size: {smaller_map_size // (1024*1024)} MB')
        try:
            env = lmdb.open(output_path, map_size=smaller_map_size)
            print('使用更小的map_size成功打开LMDB')
        except Exception as e2:
            print(f'仍然失败: {e2}')
            return
    except Exception as e:
        print(f'未知错误: {e}')
        return

    try:
        with open(label_txt, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        print(f'无法读取标签文件 {label_txt}: {e}')
        env.close()
        return

    # 准备任务列表
    tasks = []
    cnt = 1

    for line in lines:
        line = line.strip()
        if not line:
            continue

        try:
            img_name, label = line.split('\t', 1)
        except ValueError:
            print(f'标签格式错误，跳过: {line}')
            continue

        img_path = os.path.join(img_dir, img_name)
        tasks.append((cnt, img_path, label))
        cnt += 1

    total_tasks = len(tasks)
    print(f'准备处理 {total_tasks} 个图片文件')

    # 多线程处理图片
    cache = {}
    processed_count = 0
    success_count = 0
    start_time = time.time()

    # 使用线程锁保护缓存操作
    cache_lock = threading.Lock()

    def write_cache_to_db():
        """将缓存写入数据库"""
        nonlocal cache
        if cache:
            try:
                with env.begin(write=True) as txn:
                    for k, v in cache.items():
                        txn.put(k, v)
                cache = {}
                return True
            except Exception as e:
                print(f'写入LMDB失败: {e}')
                return False
        return True

    # 分批处理任务以控制内存使用
    batch_size = 1000

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]

            # 提交批处理任务
            future_to_task = {executor.submit(process_image_task, task): task for task in batch_tasks}

            # 收集结果
            for future in as_completed(future_to_task):
                result = future.result()
                processed_count += 1

                if result['success']:
                    success_count += 1

                    with cache_lock:
                        cache[result['image_key'].encode()] = result['image_bin']
                        cache[result['label_key'].encode()] = result['label_data']

                        # 每1000个成功样本写入一次
                        if success_count % 1000 == 0:
                            if not write_cache_to_db():
                                env.close()
                                return

                            elapsed_time = time.time() - start_time
                            rate = processed_count / elapsed_time if elapsed_time > 0 else 0
                            print(f'已处理 {processed_count}/{total_tasks}, 成功 {success_count}, 速度: {rate:.1f} 图片/秒')
                else:
                    if result['error']:
                        print(result['error'])

    # 写入最后的缓存和样本数量
    with cache_lock:
        cache['num-samples'.encode()] = str(success_count).encode()

        try:
            with env.begin(write=True) as txn:
                for k, v in cache.items():
                    txn.put(k, v)

            elapsed_time = time.time() - start_time
            print(f'完成！共处理 {processed_count} 个文件，成功写入 {success_count} 条样本。')
            print(f'总耗时: {elapsed_time:.2f} 秒，平均速度: {processed_count/elapsed_time:.1f} 图片/秒')

        except Exception as e:
            print(f'最终写入失败: {e}')

    env.close()

if __name__ == '__main__':
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print('用法: python create_lmdb_dataset.py 输出lmdb路径 图片文件夹 标签txt [线程数]')
        print('示例: python create_lmdb_dataset.py data/train.lmdb data/train/images data/train/labels.txt 8')
        print('线程数默认为4，建议根据CPU核心数调整')
        sys.exit(1)

    output_path, img_dir, label_txt = sys.argv[1:4]
    num_workers = int(sys.argv[4]) if len(sys.argv) == 5 else 4

    if num_workers < 1:
        print('线程数必须大于0')
        sys.exit(1)

    create_lmdb_dataset(output_path, img_dir, label_txt, num_workers=num_workers)