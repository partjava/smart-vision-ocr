import os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
from src.utils.helpers import DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS
from src.utils.logger import logger

def fix_emnist_orientation(img):
    """
    矫正 EMNIST 图像方向：原始数据存储时旋转了90°并做了镜像翻转。
    正确的矫正方式是：先逆时针旋转90°，再水平翻转。
    """
    img = img.transpose(Image.ROTATE_90)   # 逆时针旋转 90°
    img = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
    return img

def to_three_channels(x):
    """
    把单通道灰度图复制为3通道以适配标准的 CNN 模型输入
    """
    return x.repeat(3, 1, 1)

def get_emnist_transforms():
    """
    定义 EMNIST 的数据预处理和数据增强
    """
    train_transform = transforms.Compose([
        fix_emnist_orientation,
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # 数据增强：模拟真实手写的多变性
        transforms.RandomRotation(15),              # 随机旋转 ±15°
        transforms.RandomHorizontalFlip(p=0.5),    # 水平翻转 (解决 6/9、L/7 等镜像字符混淆)
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),  # 平移+缩放
        transforms.RandomPerspective(distortion_scale=0.15, p=0.3),  # 透视形变
        transforms.ElasticTransform(alpha=30.0, sigma=5.0),  # 弹性形变 (模拟手抖)
        transforms.ToTensor(),
        transforms.Lambda(to_three_channels),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.08)),  # 随机遮挡 (模拟笔迹断裂)
    ])

    val_transform = transforms.Compose([
        fix_emnist_orientation,
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(to_three_channels),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return train_transform, val_transform

def get_emnist_dataloaders(download=True):
    """
    下载/加载 EMNIST 数据集并创建 DataLoader，按 80% / 20% 比例拆分
    """
    train_transform, val_transform = get_emnist_transforms()

    logger.info("正在读取/下载 EMNIST 训练集和测试集...")
    
    full_train_dataset = torchvision.datasets.EMNIST(
        root=DATA_DIR,
        split='balanced',
        train=True,
        download=download,
        transform=train_transform
    )

    test_dataset = torchvision.datasets.EMNIST(
        root=DATA_DIR,
        split='balanced',
        train=False,
        download=download,
        transform=val_transform
    )

    # 拆分训练和验证
    total_len = len(full_train_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_len, val_len], generator=generator
    )

    # 验证集需要用 val_transform，直接访问原始数据绕过 train_transform
    class _TransformSubset(torch.utils.data.Dataset):
        """直接访问底层数据并应用指定 transform，绕过 Subset 继承的 train_transform"""
        def __init__(self, subset, dataset, indices, transform):
            self.dataset = dataset
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, idx):
            img, label = self.dataset.data[self.indices[idx]], int(self.dataset.targets[self.indices[idx]])
            img = Image.fromarray(img.numpy(), mode='L')
            if self.transform:
                img = self.transform(img)
            return img, label

    val_dataset = _TransformSubset(val_dataset, full_train_dataset, val_dataset.indices, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    logger.success(f"EMNIST 数据集加载/划分完成: 训练集 {train_len:,} | 验证集 {val_len:,} | 测试集 {len(test_dataset):,}")
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # 快速运行测试，下载并校验 EMNIST 数据集
    get_emnist_dataloaders(download=True)
