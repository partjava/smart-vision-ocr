"""
将 CCPD 数据集转换为 YOLO 格式用于车牌检测训练
用法: python -m scripts.convert_ccpd_to_yolo
"""
import os
import random
import shutil
from pathlib import Path
from src.utils.helpers import DATA_DIR
from src.utils.logger import logger

CCPD_DIR = os.path.join(DATA_DIR, 'CCPD', 'CCPD2019')
OUTPUT_DIR = os.path.join(DATA_DIR, 'yolo_plate')
TRAIN_RATIO = 0.8
SUBSETS = ['ccpd_blur', 'ccpd_challenge', 'ccpd_fn', 'ccpd_rotate', 'ccpd_tilt', 'ccpd_db', 'ccpd_np']


def parse_bbox_from_filename(filename):
    """
    从 CCPD 文件名解析 bounding box
    文件名格式: ID-angle-x1&y1_x2&y2-corners-label-...
    返回: (x1, y1, x2, y2) 像素坐标 或 None
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('-')
    if len(parts) < 3:
        return None

    try:
        bbox_str = parts[2]  # "340&500_404&526"
        top_left, bottom_right = bbox_str.split('_')
        x1, y1 = map(int, top_left.split('&'))
        x2, y2 = map(int, bottom_right.split('&'))
        return (x1, y1, x2, y2)
    except (ValueError, IndexError):
        return None


def convert_to_yolo_format(bbox, img_w, img_h):
    """
    将 (x1, y1, x2, y2) 转为 YOLO 格式: class center_x center_y width height (归一化)
    """
    x1, y1, x2, y2 = bbox
    # 确保坐标有效
    x1 = max(0, min(x1, img_w))
    y1 = max(0, min(y1, img_h))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))

    if x2 <= x1 or y2 <= y1:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h

    # 裁剪到 [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main():
    # 创建输出目录
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    # 收集所有图片路径和标注
    all_items = []
    for subset in SUBSETS:
        subset_dir = os.path.join(CCPD_DIR, subset)
        if not os.path.exists(subset_dir):
            logger.warning(f"子集不存在: {subset_dir}")
            continue
        for f in os.listdir(subset_dir):
            if f.endswith('.jpg') or f.endswith('.png'):
                all_items.append((os.path.join(subset_dir, f), f, subset))

    logger.info(f"共找到 {len(all_items):,} 张 CCPD 图片")

    # 随机打乱
    random.seed(42)
    random.shuffle(all_items)

    # 80/20 划分
    split_idx = int(len(all_items) * TRAIN_RATIO)
    splits = {
        'train': all_items[:split_idx],
        'val': all_items[split_idx:]
    }

    for split_name, items in splits.items():
        logger.info(f"处理 {split_name} 集: {len(items):,} 张...")
        success = 0
        fail = 0

        for idx, (img_path, filename, subset) in enumerate(items):
            if (idx + 1) % 5000 == 0:
                logger.info(f"  {split_name} 进度: {idx+1}/{len(items)} | 成功: {success} | 失败: {fail}")

            # 解析 bounding box
            bbox = parse_bbox_from_filename(filename)
            if bbox is None:
                fail += 1
                continue

            # 获取图片尺寸 (用 PIL 避免中文路径问题)
            from PIL import Image
            try:
                with Image.open(img_path) as img:
                    img_w, img_h = img.size
            except Exception:
                fail += 1
                continue

            # 转 YOLO 格式
            yolo_str = convert_to_yolo_format(bbox, img_w, img_h)
            if yolo_str is None:
                fail += 1
                continue

            # 复制图片和写标注
            new_name = f"{subset}_{filename}"
            dst_img = os.path.join(OUTPUT_DIR, 'images', split_name, new_name)
            dst_lbl = os.path.join(OUTPUT_DIR, 'labels', split_name, os.path.splitext(new_name)[0] + '.txt')

            shutil.copy2(img_path, dst_img)
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.write(yolo_str + '\n')

            success += 1

        logger.info(f"  {split_name} 完成: 成功 {success:,} | 失败 {fail:,}")

    # 写 data.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: images/train
val: images/val

nc: 1
names: ['license_plate']
"""
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    logger.success(f"转换完成! data.yaml: {yaml_path}")
    logger.info(f"  train: {len(splits['train']):,} 张")
    logger.info(f"  val: {len(splits['val']):,} 张")


if __name__ == '__main__':
    main()
