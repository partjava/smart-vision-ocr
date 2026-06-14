"""
从 data/plate_images/ 批量提取单字符到 data/plate_chars/
用法: python scripts/extract_chars_from_plates.py
"""
import os
import cv2
import numpy as np
from PIL import Image
from src.core.traditional.segmenter import segment_plate_characters
from src.utils.helpers import DATA_DIR, PLATE_CLASSES
from src.utils.logger import logger

PLATE_IMG_DIR = os.path.join(DATA_DIR, 'plate_images')
OUTPUT_DIR = os.path.join(DATA_DIR, 'plate_chars')
MAX_PER_CLASS = 500


def parse_plate_filename(filename):
    """从文件名提取车牌文本, 如 'xxx_闽D2315W.jpg' -> '闽D2315W'"""
    basename = os.path.splitext(filename)[0]
    parts = basename.split('_', 1)
    if len(parts) < 2:
        return None
    plate_text = parts[1].replace(' ', '').strip()
    # 只保留车牌合法字符
    if all(c in PLATE_CLASSES for c in plate_text):
        return plate_text
    return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for cls_name in PLATE_CLASSES:
        os.makedirs(os.path.join(OUTPUT_DIR, cls_name), exist_ok=True)

    # 已有计数
    class_counts = {}
    for cls_name in PLATE_CLASSES:
        cls_dir = os.path.join(OUTPUT_DIR, cls_name)
        class_counts[cls_name] = len([f for f in os.listdir(cls_dir) if f.endswith('.png')])

    image_files = [f for f in os.listdir(PLATE_IMG_DIR) if f.endswith('.jpg')]
    logger.info(f"共 {len(image_files)} 张车牌图, 开始切字符...")

    total_extracted = 0
    total_failed = 0

    for idx, filename in enumerate(image_files):
        if (idx + 1) % 2000 == 0:
            done = sum(1 for c in class_counts.values() if c >= MAX_PER_CLASS)
            logger.info(f"  进度: {idx+1}/{len(image_files)} | 已提取: {total_extracted} | 失败: {total_failed} | 类别完成: {done}/{len(PLATE_CLASSES)}")

        if all(c >= MAX_PER_CLASS for c in class_counts.values()):
            logger.info(f"  所有类别已达到 {MAX_PER_CLASS} 张，提前结束。")
            break

        plate_text = parse_plate_filename(filename)
        if plate_text is None or len(plate_text) != 7:
            total_failed += 1
            continue

        img_path = os.path.abspath(os.path.join(PLATE_IMG_DIR, filename))
        pil_img = Image.open(img_path)
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if img is None:
            total_failed += 1
            continue

        img = cv2.resize(img, (220, 70))
        char_images, _ = segment_plate_characters(img)

        if len(char_images) != 7:
            total_failed += 1
            continue

        for char_img, char_label in zip(char_images, plate_text):
            if char_label not in PLATE_CLASSES:
                continue
            if class_counts[char_label] >= MAX_PER_CLASS:
                continue

            save_path = os.path.abspath(os.path.join(OUTPUT_DIR, char_label, f"{class_counts[char_label]:05d}.png"))
            Image.fromarray(cv2.cvtColor(char_img, cv2.COLOR_GRAY2RGB)).save(save_path)
            class_counts[char_label] += 1
            total_extracted += 1

    logger.success(f"完成: 成功 {total_extracted:,} 张 | 失败 {total_failed:,} 张")
    logger.info("各类别数量:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info(f"  {cls_name}: {count}")


if __name__ == '__main__':
    main()
