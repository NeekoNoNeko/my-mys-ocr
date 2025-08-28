import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CRNN
from dataset import OCRDataset
from torch.utils.tensorboard import SummaryWriter
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import argparse
import re # Added for resuming training

# 字符集（可根据实际数据集修改）
# CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
# CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789/"
# CHARS = "OP12/403ADC@E"
# CHARS = "01234679ADEMZ"
# CHARS = "()0123459ADEgIMnR.<>VZ"
# CHARS = "-0123469ADFMTZ"
CHARS = "()-.><0123456789ABDEFIMRTVZgn"
BLANK = '─'  # 使用特殊符号作为填充符，避免与实际字符冲突
CHARS = BLANK + CHARS
# 检查字符集是否有重复字符
if len(set(CHARS)) != len(CHARS):
    duplicates = set([c for c in CHARS if CHARS.count(c) > 1])
    raise ValueError(f"字符集 CHARS 存在重复字符: {sorted(duplicates)}，请检查并去重！")
nclass = len(CHARS)

# 字符与索引映射s
char2idx = {c: i for i, c in enumerate(CHARS)}
idx2char = {i: c for i, c in enumerate(CHARS)}

def text_to_indices(text):
    return [char2idx[c] for c in text if c in char2idx]

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_indices = [torch.tensor(text_to_indices(label), dtype=torch.long) for label in labels]
    label_lengths = torch.tensor([len(l) for l in label_indices], dtype=torch.long)
    labels_concat = torch.cat(label_indices)
    return images, labels_concat, label_lengths

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

def freeze_model_for_finetuning(model, freeze_backbone=True):
    """
    冻结模型参数，只保留最后的全连接层可训练
    Args:
        model: CRNN模型
        freeze_backbone: 是否冻结backbone（CNN和LSTM）
    """
    if freeze_backbone:
        print("冻结backbone参数（CNN和LSTM）...")
        
        # 冻结CNN部分
        for param in model.cnn.parameters():
            param.requires_grad = False
        print("✓ CNN层已冻结")
        
        # 冻结LSTM部分（除了最后的embedding层）
        for name, param in model.rnn.named_parameters():
            if 'embedding' not in name:  # 只保留embedding层可训练
                param.requires_grad = False
            else:
                print(f"✓ 保持可训练: {name}")
        print("✓ LSTM层已冻结（除embedding层外）")
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable_params:,} / 总参数: {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    else:
        print("所有参数都可训练（正常训练模式）")

def check_dataset_chars(dataset, char2idx):
    """检查数据集中的字符是否与给定字符集匹配"""
    dataset_chars = set()
    print("正在检查数据集字符...")
    for i in range(len(dataset)):
        if i % 1000 == 0:
            print(f"已检查 {i}/{len(dataset)} 个样本")
        _, label = dataset[i]
        for char in label:
            dataset_chars.add(char)
    
    # 检查数据集字符是否都在字符集中
    missing_chars = dataset_chars - set(char2idx.keys())
    extra_chars = set(char2idx.keys()) - dataset_chars - {BLANK}
    
    if missing_chars:
        print(f"错误：数据集中存在字符集未包含的字符：{sorted(missing_chars)}")
        print("建议更新字符集为：")
        all_chars = sorted(list(dataset_chars))
        new_chars = ''.join(all_chars)
        print(f"CHARS = '{new_chars}'")
        raise ValueError(f"数据集中存在字符集未包含的字符：{sorted(missing_chars)}")
    
    if extra_chars:
        print(f"提示：字符集中存在数据集中未出现的字符：{sorted(extra_chars)}")
        print("训练将继续，但建议优化字符集以减少模型复杂度")
    
    print(f"数据集字符检查完成，共发现 {len(dataset_chars)} 个字符")
    return True

def train(check_chars=False, resume_from=None, finetune=False, pretrained_model=None, no_val=False, augment_data=False, 
          num_workers=None, prefetch_factor=2, persistent_workers=False, pin_memory=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    imgH = 32
    nh = 256
    nc = 1
    
    # =====================
    # 超参数分为两套
    # =====================
    if finetune:
        # 微调参数
        batch_size = 128
        n_epoch = 30
        lr = 1e-4
        weight_decay = 1e-4
        scheduler_step = 10
        scheduler_gamma = 0.8
        patience = 5
        print("使用微调超参数：batch_size=128, n_epoch=30, lr=1e-4, weight_decay=1e-4, scheduler_step=10, scheduler_gamma=0.8, patience=5")
    else:
        # 正常训练参数
        batch_size = 64
        n_epoch = 400
        lr = 5e-4
        weight_decay = 1e-4
        scheduler_step = 30
        scheduler_gamma = 0.7
        patience = 10
        print("使用正常训练超参数：batch_size=64, n_epoch=400, lr=5e-4, weight_decay=1e-4, scheduler_step=30, scheduler_gamma=0.7, patience=10")

    # 智能批处理大小调整（基于GPU内存和多线程配置）
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU内存: {gpu_memory_gb:.1f}GB")

        # 基于GPU内存动态调整批处理大小
        if gpu_memory_gb >= 24:  # 高端GPU
            batch_size_multiplier = 2
        elif gpu_memory_gb >= 12:  # 中端GPU
            batch_size_multiplier = 1.5
        elif gpu_memory_gb >= 8:  # 低端GPU
            batch_size_multiplier = 1.2
        else:  # 极低端GPU
            batch_size_multiplier = 0.8

        # 多线程时可以适度增加批处理大小
        if num_workers is not None and num_workers > 0:
            batch_size_multiplier *= 1.2

        original_batch_size = batch_size
        batch_size = int(batch_size * batch_size_multiplier)
        # 确保批处理大小是8的倍数（GPU友好）
        batch_size = (batch_size + 7) // 8 * 8

        if batch_size != original_batch_size:
            print(f"根据硬件配置调整批处理大小: {original_batch_size} -> {batch_size}")
    else:
        print("使用CPU训练，保持原始批处理大小")
    
    writer = SummaryWriter(log_dir='outputs')
    
    best_val_loss = float('inf')
    best_epoch = 0
    early_stop_counter = 0
    start_epoch = 0  # 初始化start_epoch
    
    # 检查训练数据是否存在
    train_lmdb_path = 'data/train.lmdb'
    val_lmdb_path = 'data/val.lmdb'

    print(f"检查训练数据: {train_lmdb_path}")
    if not os.path.exists(train_lmdb_path):
        print(f"""
错误：训练数据不存在: {train_lmdb_path}

请先创建LMDB训练数据集。步骤如下：
1. 准备训练图片文件夹（如：data/train/images/）
2. 准备训练标签文件（如：data/train/labels.txt，格式：图片名称\\t标签）
3. 运行命令：
   python create_lmdb_dataset.py data/train.lmdb data/train/images data/train/labels.txt

当前工作目录: {os.getcwd()}
        """)
        return

    if not no_val:
        print(f"检查验证数据: {val_lmdb_path}")
        if not os.path.exists(val_lmdb_path):
            print(f"""
错误：验证数据不存在: {val_lmdb_path}

请先创建LMDB验证数据集，或使用 --no-val 参数跳过验证。

创建验证数据集步骤：
1. 准备验证图片文件夹（如：data/val/images/）
2. 准备验证标签文件（如：data/val/labels.txt，格式：图片名称\\t标签）
3. 运行命令：
   python create_lmdb_dataset.py data/val.lmdb data/val/images data/val/labels.txt

或者使用 --no-val 参数只使用训练数据：
   python train.py --no-val
            """)
            return

    try:
        train_dataset = OCRDataset(
            lmdb_path=train_lmdb_path,
            transform=transforms.Compose([
                transforms.Resize((imgH, 100)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )
        print(f"✓ 训练数据加载成功，样本数量: {len(train_dataset)}")
    except Exception as e:
        print(f"训练数据加载失败: {str(e)}")
        return

    if not no_val:
        try:
            val_dataset = OCRDataset(
                lmdb_path=val_lmdb_path,
                transform=transforms.Compose([
                    transforms.Resize((imgH, 100)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            )
            print(f"✓ 验证数据加载成功，样本数量: {len(val_dataset)}")
        except Exception as e:
            print(f"验证数据加载失败: {str(e)}")
            print("建议使用 --no-val 参数跳过验证，或检查验证数据是否正确创建")
            return
    
    # 检查数据集字符
    if check_chars:
        print("检查训练集字符...")
        check_dataset_chars(train_dataset, char2idx)
        if not no_val:
            print("检查验证集字符...")
            check_dataset_chars(val_dataset, char2idx)
    else:
        print("跳过字符集检查...")
    
    # 智能多线程配置
    if num_workers is None:
        if os.name == 'nt':  # Windows
            num_workers = 0
            print("Windows系统检测到，使用单线程数据加载 (num_workers=0)")
        else:  # Linux/macOS
            import multiprocessing
            num_workers = min(4, multiprocessing.cpu_count())
            print(f"使用多线程数据加载 (num_workers={num_workers})")
    else:
        print(f"使用用户指定的线程数 (num_workers={num_workers})")

    # 多线程优化提示
    if num_workers > 0:
        print(f"多线程配置:")
        print(f"  - 工作线程数: {num_workers}")
        print(f"  - 预取批次数: {prefetch_factor}")
        print(f"  - 持久化工作线程: {persistent_workers}")
        print(f"  - 内存固定: {pin_memory}")
        if not persistent_workers and num_workers > 0:
            print("  提示: 使用 --persistent-workers 可减少进程创建开销")

    # 构建DataLoader参数
    loader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': pin_memory and torch.cuda.is_available(),
    }

    # 多线程专用参数
    if num_workers > 0:
        loader_kwargs.update({
            'prefetch_factor': prefetch_factor,
            'persistent_workers': persistent_workers,
        })

        # 在非Windows系统上启用更快的多进程启动方法
        if os.name != 'nt':
            try:
                import multiprocessing
                if hasattr(multiprocessing, 'set_start_method'):
                    multiprocessing.set_start_method('spawn', force=True)
                    print("  - 使用spawn启动方法优化多进程性能")
            except RuntimeError:
                pass  # 可能已经设置过了

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    if not no_val:
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = CRNN(imgH, nc, nclass, nh).to(device)
    
    # 微调模式：加载预训练模型
    if finetune and pretrained_model:
        print(f"加载预训练模型: {pretrained_model}")
        if os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✓ 预训练模型加载成功")
        else:
            print(f"警告：预训练模型文件不存在: {pretrained_model}")
            print("将从头开始训练")
    
    # 冻结参数（微调模式）
    if finetune:
        freeze_model_for_finetuning(model, freeze_backbone=True)
    else:
        print("正常训练模式：所有参数都可训练")
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    # 微调时只优化可训练参数
    if finetune:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        # 微调模式使用更保守的betas
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
        print("✓ 优化器仅包含可训练参数，使用微调模式betas=(0.9, 0.95)")
    else:
        # 正常训练模式使用标准betas
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
        print("✓ 使用正常训练模式betas=(0.9, 0.999)")
    
    # 学习率调度器 - 更温和的衰减
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    
    # 断点续训
    if resume_from:
        print(f"从断点恢复训练: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        # 检查是否是新的完整checkpoint格式
        if 'model_state_dict' in checkpoint:
            # 新格式：包含完整训练状态
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint['best_val_loss']
            best_epoch = checkpoint['best_epoch']
            early_stop_counter = checkpoint['early_stop_counter']
            print(f"恢复训练进度: Epoch {start_epoch}, 最佳验证损失: {best_val_loss:.4f}")
        else:
            # 旧格式：只包含模型权重
            model.load_state_dict(checkpoint)
            print("检测到旧格式checkpoint，仅恢复模型权重")
            print("优化器状态和学习率将重新初始化")
            # 尝试从文件名推断epoch
            match = re.search(r'epoch(\d+)', resume_from)
            if match:
                start_epoch = int(match.group(1))
                print(f"从文件名推断起始epoch: {start_epoch}")
            else:
                start_epoch = 0
                print("无法推断起始epoch，从0开始")
    
    scaler = GradScaler('cuda')

    # 性能优化提示
    print("\n=== 训练性能优化信息 ===")
    print(f"设备: {device}")
    print(f"批处理大小: {batch_size}")
    print(f"数据加载线程: {num_workers}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN启用: {torch.backends.cudnn.enabled}")
        print(f"cuDNN基准模式: {torch.backends.cudnn.benchmark}")
        # 启用cuDNN基准模式以优化性能
        torch.backends.cudnn.benchmark = True
        print("已启用cuDNN基准模式以优化性能")
    print("========================\n")

    epoch = start_epoch
    while epoch < n_epoch:
        model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        for images, labels, label_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)
            # 暂时禁用混合精度
            preds = model(images)
            preds_log_softmax = nn.functional.log_softmax(preds, 2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long).to(device)
            loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # 计算acc
            pred_texts = decode(preds_log_softmax)
            batch_size = images.size(0)
            labels_cpu = labels.cpu().numpy()
            label_lengths_cpu = label_lengths.cpu().numpy()
            gt_texts = []
            start = 0
            for l in label_lengths_cpu:
                gt_texts.append(''.join([idx2char[i] for i in labels_cpu[start:start+l]]))
                start += l
            for p, g in zip(pred_texts, gt_texts):
                if p == g:
                    total_correct += 1
            total_samples += batch_size
        train_acc = total_correct / total_samples if total_samples > 0 else 0
        avg_train_loss = total_loss/len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch+1)
        writer.add_scalar('Acc/train', train_acc, epoch+1)
        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # 验证
        if not no_val:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                val_correct = 0
                val_samples = 0
                for images, labels, label_lengths in val_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    label_lengths = label_lengths.to(device)
                    # 暂时禁用混合精度
                    preds = model(images)
                    preds_log_softmax = nn.functional.log_softmax(preds, 2)
                    input_lengths = torch.full(size=(images.size(0),), fill_value=preds.size(0), dtype=torch.long).to(device)
                    loss = criterion(preds_log_softmax, labels, input_lengths, label_lengths)
                    val_loss += loss.item()
                    pred_texts = decode(preds_log_softmax)
                    batch_size = images.size(0)
                    labels_cpu = labels.cpu().numpy()
                    label_lengths_cpu = label_lengths.cpu().numpy()
                    gt_texts = []
                    start = 0
                    for l in label_lengths_cpu:
                        gt_texts.append(''.join([idx2char[i] for i in labels_cpu[start:start+l]]))
                        start += l
                    for p, g in zip(pred_texts, gt_texts):
                        if p == g:
                            val_correct += 1
                    val_samples += batch_size
                val_acc = val_correct / val_samples if val_samples > 0 else 0
                avg_val_loss = val_loss/len(val_loader)
                writer.add_scalar('Loss/val', avg_val_loss, epoch+1)
                writer.add_scalar('Acc/val', val_acc, epoch+1)
                print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 学习率调度器更新
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch+1)
        print(f"Learning Rate: {current_lr:.6f}")
        
        # 早停检查
        if not no_val:
            improved = avg_val_loss < best_val_loss
        else:
            improved = avg_train_loss < best_val_loss
        if improved:
            best_val_loss = avg_val_loss if not no_val else avg_train_loss
            best_epoch = epoch + 1
            early_stop_counter = 0
            # 保存最佳模型
            os.makedirs('checkpoints', exist_ok=True)
            model_suffix = "_finetuned" if finetune else ""
            torch.save(model.state_dict(), f'checkpoints/crnn_best{model_suffix}.pth')
            print(f"保存最佳模型 (Epoch {best_epoch}, {'Val Loss' if not no_val else 'Train Loss'}: {best_val_loss:.4f})")
        else:
            early_stop_counter += 1
            print(f"{'验证' if not no_val else '训练'}损失未改善，早停计数器: {early_stop_counter}/{patience}")
        
        # 早停
        if early_stop_counter >= patience:
            print(f"早停触发！最佳模型在Epoch {best_epoch}，{'验证' if not no_val else '训练'}损失: {best_val_loss:.4f}")
            break
        
        # 保存当前epoch模型（包含完整checkpoint信息）
        os.makedirs('checkpoints', exist_ok=True)
        model_suffix = "_finetuned" if finetune else ""
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
            'early_stop_counter': early_stop_counter,
            'loss': avg_train_loss,
            'val_loss': avg_val_loss if not no_val else None,
            'acc': train_acc,
            'val_acc': val_acc if not no_val else None,
            'finetune': finetune
        }
        torch.save(checkpoint, f'checkpoints/crnn_epoch{epoch+1}{model_suffix}.pth')
        print(f"保存checkpoint: Epoch {epoch+1}")
        
        # 检查是否完成预设的epoch数
        epoch += 1
        if epoch >= n_epoch:
            print(f"\n完成预设的 {n_epoch} 个epoch训练")
            print(f"当前最佳{'验证' if not no_val else '训练'}损失: {best_val_loss:.4f}, 最佳epoch: {best_epoch}")
            print(f"当前{'验证' if not no_val else '训练'}准确率: {val_acc if not no_val else train_acc:.4f}")
            
            # 询问是否继续训练
            try:
                response = input("\n是否继续训练？(y/n): ").strip().lower()
                if response in ['y', 'yes', '是', '继续']:
                    print("继续训练...")
                    n_epoch += 50  # 增加50个epoch
                    print(f"新的训练目标: {n_epoch} 个epoch")
                else:
                    print("训练结束")
                    break
            except KeyboardInterrupt:
                print("\n用户中断，训练结束")
                break
    
    writer.close()
    print(f"训练完成！最佳模型在Epoch {best_epoch}，{'验证' if not no_val else '训练'}损失: {best_val_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRNN OCR Training')
    parser.add_argument('--check-chars', action='store_true', 
                       help='开启字符集检查（默认关闭检查）')
    parser.add_argument('--resume', type=str, default=None,
                       help='从指定checkpoint文件恢复训练')
    parser.add_argument('--finetune', action='store_true',
                       help='启用微调模式，只训练最后的全连接层')
    parser.add_argument('--pretrained-model', type=str, default=None,
                       help='预训练模型路径（微调模式必需）')
    parser.add_argument('--no-val', action='store_true',
                      help='不进行验证，只用训练集')
    parser.add_argument('--augment-data', action='store_true',
                        help='对训练数据进行长度扰动增强')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='数据加载器的工作线程数 (默认: 自动检测，Windows上为0，其他系统为4)')
    parser.add_argument('--prefetch-factor', type=int, default=2,
                        help='每个工作线程预取的批次数 (默认: 2)')
    parser.add_argument('--persistent-workers', action='store_true',
                        help='保持工作线程持久化，减少创建/销毁开销 (默认: False)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='启用内存固定，加速GPU数据传输 (默认: True)')
    args = parser.parse_args()
    
    # 验证微调参数
    if args.finetune and not args.pretrained_model:
        print("错误：微调模式需要指定预训练模型路径")
        print("使用方法: --finetune --pretrained-model path/to/model.pth")
        exit(1)
    
    # 根据参数决定是否检查字符集
    check_chars = args.check_chars
    print(f"字符集检查: {'开启' if check_chars else '关闭'}")
    print(f"微调模式: {'开启' if args.finetune else '关闭'}")
    if args.finetune:
        print(f"预训练模型: {args.pretrained_model}")
    print(f"验证集: {'关闭' if args.no_val else '开启'}")
    print(f"数据增强: {'开启' if args.augment_data else '关闭'}")
    print(f"多线程数据加载: {args.num_workers if args.num_workers is not None else '自动检测'}")

    train(check_chars=check_chars, resume_from=args.resume, 
          finetune=args.finetune, pretrained_model=args.pretrained_model, no_val=args.no_val,
          augment_data=args.augment_data, num_workers=args.num_workers,
          prefetch_factor=args.prefetch_factor, persistent_workers=args.persistent_workers,
          pin_memory=args.pin_memory)

    # tensorboard --logdir=outputs