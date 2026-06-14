import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from src.utils.helpers import DATA_DIR, BATCH_SIZE, IMAGE_SIZE, NUM_WORKERS, PLATE_CLASSES
from src.utils.logger import logger

def get_font(font_size=28):
    """
    寻找系统自带的有效中文字体文件
    """
    font_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc",    # 微软雅黑
        "C:\\Windows\\Fonts\\simhei.ttf",   # 黑体
        "C:\\Windows\\Fonts\\simsun.ttc",   # 宋体
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", # Linux Fallback
        "/System/Library/Fonts/PingFang.ttc" # macOS Fallback
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    # 兜底返回默认字体 (注意默认字体可能无法绘制中文，会显示为方框)
    return ImageFont.load_default()

def get_multiple_fonts(font_size=28):
    """
    获取多个系统字体以增加字体多样性
    """
    font_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc",     # 微软雅黑
        "C:\\Windows\\Fonts\\simhei.ttf",    # 黑体
        "C:\\Windows\\Fonts\\simsun.ttc",    # 宋体
        "C:\\Windows\\Fonts\\simfang.ttf",   # 仿宋
        "C:\\Windows\\Fonts\\kaiti.ttf",     # 楷体
        "C:\\Windows\\Fonts\\arial.ttf",     # Arial
        "C:\\Windows\\Fonts\\times.ttf",     # Times New Roman
        "C:\\Windows\\Fonts\\consola.ttf",   # Consolas
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/System/Library/Fonts/PingFang.ttc"
    ]
    fonts = []
    for path in font_paths:
        if os.path.exists(path):
            try:
                fonts.append(ImageFont.truetype(path, font_size))
            except Exception:
                continue
    if not fonts:
        fonts.append(ImageFont.load_default())
    return fonts


def generate_char_image(char, font, size=32):
    """
    动态生成单张车牌字符图像。
    包含蓝、黄、绿、黑背景，并施加旋转、模糊、噪声、透视等增强。
    """
    # 随机选择车牌配色风格（加入颜色抖动）
    bg_style = random.choice(["blue", "yellow", "green", "black"])
    if bg_style == "blue":
        bg_color = (random.randint(0, 30), random.randint(40, 70), random.randint(130, 180))
        text_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))
    elif bg_style == "yellow":
        bg_color = (random.randint(0, 20), random.randint(190, 220), random.randint(230, 255))
        text_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
    elif bg_style == "green":
        bg_color = (random.randint(0, 20), random.randint(130, 180), random.randint(60, 100))
        text_color = (random.randint(0, 30), random.randint(0, 30), random.randint(0, 30))
    else:
        bg_color = (random.randint(0, 20), random.randint(0, 20), random.randint(0, 20))
        text_color = (random.randint(230, 255), random.randint(230, 255), random.randint(230, 255))

    # 创建双倍大小画布以进行高质量旋转和插值
    canvas_size = size * 2
    img = Image.new("RGB", (canvas_size, canvas_size), bg_color)
    draw = ImageDraw.Draw(img)

    # 测量字体尺寸
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except AttributeError:
        tw, th = draw.textsize(char, font=font)

    # 随机轻微偏移字符位置
    x = (canvas_size - tw) // 2 + random.randint(-3, 3)
    y = (canvas_size - th) // 2 - 2 + random.randint(-3, 3)

    # 写入字符
    draw.text((x, y), char, font=font, fill=text_color)

    # 随机旋转（加大角度范围）
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=bg_color)

    # 居中裁剪回原大小
    cx, cy = canvas_size // 2, canvas_size // 2
    half = size // 2
    img = img.crop((cx - half, cy - half, cx + half, cy + half))

    # 随机高斯模糊（模拟拍摄模糊）
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))

    # 转为 numpy 做进一步增强
    img_np = np.array(img, dtype=np.uint8)

    # 随机高斯噪声
    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(2, 12), img_np.shape).astype(np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 随机椒盐噪声
    if random.random() < 0.2:
        salt_pepper = np.random.random(img_np.shape[:2])
        img_np[salt_pepper < 0.02] = 255
        img_np[salt_pepper > 0.98] = 0

    # 随机亮度/对比度变化（模拟光照不均）
    if random.random() < 0.5:
        alpha = random.uniform(0.7, 1.3)  # 对比度
        beta = random.randint(-30, 30)     # 亮度
        img_np = np.clip(alpha * img_np.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # 随机局部遮挡（模拟污损/反光）
    if random.random() < 0.15:
        occlude_x = random.randint(0, size - 8)
        occlude_y = random.randint(0, size - 8)
        occlude_w = random.randint(4, 10)
        occlude_h = random.randint(4, 10)
        occlude_color = random.choice([bg_color, (128, 128, 128)])
        img_np[occlude_y:occlude_y+occlude_h, occlude_x:occlude_x+occlude_w] = occlude_color

    # 随机腐蚀/膨胀（模拟笔画粗细变化）
    if random.random() < 0.2:
        kernel_size = random.choice([2, 3])
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        if random.random() < 0.5:
            img_np = cv2.erode(img_np, kernel, iterations=1)
        else:
            img_np = cv2.dilate(img_np, kernel, iterations=1)

    return Image.fromarray(img_np)

class SyntheticPlateDataset(Dataset):
    """
    合成车牌字符数据集（多字体 + 丰富数据增强）
    """
    def __init__(self, char_list, num_samples_per_class=200, size=32, transform=None):
        self.char_list = char_list
        self.num_samples_per_class = num_samples_per_class
        self.size = size
        self.transform = transform
        # 加载多个字体以增加多样性
        self.fonts = get_multiple_fonts(int(size * 0.8))

        # 预先生成所有样本的定义 (字符, 类别标签)
        self.samples = []
        for idx, char in enumerate(self.char_list):
            for _ in range(self.num_samples_per_class):
                self.samples.append((char, idx))

        # 乱序
        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        char, label = self.samples[idx]
        # 每次随机选一个字体
        font = random.choice(self.fonts)
        img = generate_char_image(char, font, self.size)
        if self.transform:
            img = self.transform(img)
        return img, label

def get_synthetic_transforms():
    """
    预处理操作
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class RealPlateCharDataset(Dataset):
    """
    从磁盘加载真实/预生成的车牌字符图片
    目录结构: data/plate_chars/{char_label}/xxxxx.png
    """
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.samples = []  # [(img_path, label_idx), ...]
        self.class_to_idx = {cls: i for i, cls in enumerate(PLATE_CLASSES)}

        for cls_name in PLATE_CLASSES:
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            label_idx = self.class_to_idx[cls_name]
            for fname in os.listdir(cls_dir):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    self.samples.append((os.path.join(cls_dir, fname), label_idx))

        random.seed(42)
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('L')  # 灰度
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = Image.merge("RGB", (img, img, img))  # 转 3 通道
        if self.transform:
            img = self.transform(img)
        return img, label


def get_synthetic_dataloaders(char_list=PLATE_CLASSES, num_samples_per_class=500, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    加载车牌字符数据集。
    优先使用 data/plate_chars/ 下的真实数据（CCPD 或预生成），
    若不存在则动态生成合成数据。
    """
    real_data_dir = os.path.join(DATA_DIR, 'plate_chars')
    transform = get_synthetic_transforms()

    # 检查是否有预生成的真实数据
    if os.path.exists(real_data_dir) and any(
        os.path.isdir(os.path.join(real_data_dir, d)) for d in os.listdir(real_data_dir)
    ):
        logger.info(f"检测到预生成的车牌字符数据: {real_data_dir}")
        dataset = RealPlateCharDataset(real_data_dir, transform=transform)
        logger.info(f"加载 {len(dataset):,} 张字符图片")
    else:
        logger.info(f"未找到预生成数据，使用动态合成 (每类 {num_samples_per_class} 张)")
        dataset = SyntheticPlateDataset(
            char_list=char_list,
            num_samples_per_class=num_samples_per_class,
            size=IMAGE_SIZE,
            transform=transform
        )
    
    total_len = len(dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    logger.info(f"正在生成/读取合成车牌数据集... 类别数: {len(char_list)} 类 | 每类样本: {num_samples_per_class} 个")
    logger.success(f"合成车牌数据集划分完成: 训练集 {train_len:,} | 验证集 {val_len:,} | 测试集 {test_len:,}")
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def save_sample_images():
    """
    保存部分合成的车牌字符图片到 data/synthetic_samples/ 以便直观查看
    """
    out_dir = os.path.join(DATA_DIR, "synthetic_samples")
    os.makedirs(out_dir, exist_ok=True)
    font = get_font(24)
    
    print(f"[Info] 正在生成车牌合成示例图像并存入 {out_dir} ...")
    # 每个字符生成1张进行可视化展示
    for idx, char in enumerate(PLATE_CLASSES):
        img = generate_char_image(char, font, size=IMAGE_SIZE)
        # 用拼音或索引重命名，防止文件名中文冲突
        img.save(os.path.join(out_dir, f"sample_{idx}.png"))
    print("[Success] 合成示例图像生成完毕。")

if __name__ == '__main__':
    save_sample_images()
