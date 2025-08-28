import lmdb
import os

def merge_lmdbs(lmdb_path1, lmdb_path2, output_lmdb_path, map_size=1 << 40):
    # 创建输出目录
    os.makedirs(output_lmdb_path, exist_ok=True)
    # 打开输出lmdb
    env_out = lmdb.open(output_lmdb_path, map_size=map_size)
    
    def copy_env(src_path):
        env = lmdb.open(src_path, readonly=True, lock=False)
        with env.begin() as txn:
            with env_out.begin(write=True) as out_txn:
                cursor = txn.cursor()
                for key, value in cursor:
                    out_txn.put(key, value)
        env.close()
    
    # 先复制第一个
    copy_env(lmdb_path1)
    # 再复制第二个（如有重复key会覆盖）
    copy_env(lmdb_path2)
    env_out.close()

# 用法示例
merge_lmdbs('/root/workspace/databeifen/original/train.lmdb', '/root/workspace/databeifen/original/val.lmdb', '/root/workspace/databeifen/original/lmdb_merged')