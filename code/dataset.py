import os
import lmdb
import six
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import io
import random

# 字符集，需要和 train.py 中保持一致
CHARS = "OP12/403ADC@E"
BLANK = '-'
CHARS = BLANK + CHARS

def augment_label(label):
    if random.random() < 0.1:
        # 随机插入一个字符
        if len(label) < 21: # 仅在长度小于21时插入
            insert_pos = random.randint(0, len(label))
            insert_char = random.choice(CHARS.replace(BLANK, '')) # 不插入空白符
            label = label[:insert_pos] + insert_char + label[insert_pos:]
    elif random.random() < 0.1:
        # 随机删除一个字符
        if len(label) > 1: # 保证标签不为空
            delete_pos = random.randint(0, len(label) - 1)
            label = label[:delete_pos] + label[delete_pos+1:]
    return label

class OCRDataset(Dataset):
    def __init__(self, lmdb_path, transform=None, augment=False):
        self.lmdb_path = lmdb_path
        self.transform = transform
        self.augment = augment
        self._env = None  # 先不打开，惰性初始化
        
        # 只在主进程做一次统计
        with lmdb.open(lmdb_path, readonly=True, lock=False) as tmp_env:
            with tmp_env.begin(write=False) as txn:
                self.n_samples = int(txn.get('num-samples'.encode()))

    @property
    def env(self):
        # 每个子进程第一次调用时重新打开
        if self._env is None:
            self._env = lmdb.open(
                self.lmdb_path,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
        return self._env

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        index = idx + 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)
            buf = six.BytesIO(imgbuf)
            try:
                image = Image.open(buf).convert('L')
            except Exception:
                print(f'Corrupted image for index {index}')
                image = Image.new('L', (100, 32))
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
        
        if self.augment:
            label = augment_label(label)
            
        if self.transform:
            image = self.transform(image)
        return image, label 