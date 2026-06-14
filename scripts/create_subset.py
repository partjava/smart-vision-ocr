"""
从完整 YOLO 训练集中采样子集，加速训练
用法: python -m scripts.create_subset
"""
import os
import random
import shutil
from src.utils.logger import logger

SRC_DIR = 'data/yolo_plate'
DST_DIR = 'data/yolo_plate_subset'
SUBSET_SIZE = 20000  # 采样 2 万张训练图

def main():
    random.seed(42)

    # 复制 data.yaml
    os.makedirs(DST_DIR, exist_ok=True)
    yaml_content = f"""path: {os.path.abspath(DST_DIR)}
train: images/train
val: images/val

nc: 1
names: ['license_plate']
"""
    with open(os.path.join(DST_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)

    # 处理 train 集 (采样)
    src_train_img = os.path.join(SRC_DIR, 'images', 'train')
    src_train_lbl = os.path.join(SRC_DIR, 'labels', 'train')
    dst_train_img = os.path.join(DST_DIR, 'images', 'train')
    dst_train_lbl = os.path.join(DST_DIR, 'labels', 'train')
    os.makedirs(dst_train_img, exist_ok=True)
    os.makedirs(dst_train_lbl, exist_ok=True)

    all_files = [f for f in os.listdir(src_train_img) if f.endswith('.jpg') or f.endswith('.png')]
    sampled = random.sample(all_files, min(SUBSET_SIZE, len(all_files)))
    logger.info(f"训练集采样: {len(sampled)}/{len(all_files)}")

    for f in sampled:
        shutil.copy2(os.path.join(src_train_img, f), os.path.join(dst_train_img, f))
        lbl = os.path.splitext(f)[0] + '.txt'
        src_lbl = os.path.join(src_train_lbl, lbl)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(dst_train_lbl, lbl))

    # 验证集完整复制
    src_val_img = os.path.join(SRC_DIR, 'images', 'val')
    src_val_lbl = os.path.join(SRC_DIR, 'labels', 'val')
    dst_val_img = os.path.join(DST_DIR, 'images', 'val')
    dst_val_lbl = os.path.join(DST_DIR, 'labels', 'val')
    os.makedirs(dst_val_img, exist_ok=True)
    os.makedirs(dst_val_lbl, exist_ok=True)

    val_files = [f for f in os.listdir(src_val_img) if f.endswith('.jpg') or f.endswith('.png')]
    # 验证集也采样 5000 张加快验证速度
    val_sampled = random.sample(val_files, min(5000, len(val_files)))
    logger.info(f"验证集采样: {len(val_sampled)}/{len(val_files)}")

    for f in val_sampled:
        shutil.copy2(os.path.join(src_val_img, f), os.path.join(dst_val_img, f))
        lbl = os.path.splitext(f)[0] + '.txt'
        src_lbl = os.path.join(src_val_lbl, lbl)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, os.path.join(dst_val_lbl, lbl))

    logger.success(f"子集创建完成: {DST_DIR}")
    logger.info(f"  训练: {len(sampled)} 张")
    logger.info(f"  验证: {len(val_sampled)} 张")


if __name__ == '__main__':
    main()
