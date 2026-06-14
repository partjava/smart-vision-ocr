"""
CCPD 真实车牌字符数据集集成模块
- 自动解析 CCPD 文件名获取车牌文本
- 利用垂直投影分割切出单字符
- 生成训练用的字符图片数据集

使用方式：
1. 下载 CCPD 数据集到 data/CCPD/ 目录
   GitHub: https://github.com/detectRecog/CCPD
   或百度网盘等镜像
2. 运行: python -m src.core.deep_learning.plate_dataset
3. 生成的字符图片保存到 data/plate_chars/
"""

import os
import re
import cv2
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from src.utils.helpers import DATA_DIR, PLATE_CLASSES
from src.utils.logger import logger

# CCPD 文件名中的车牌字符映射表（与 CCPD 官方 repo 一致）
# 索引 0-30: 省份简称, 索引 1-24(+31偏移): 字母, 索引 2-34(+31偏移): 字母+数字
CCPD_PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
    "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "京"
]
CCPD_ALPHABETS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "O"
]
CCPD_ADS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "O"
]


def parse_ccpd_filename(filename):
    """
    解析 CCPD 文件名，提取车牌文本和车牌四角坐标。
    CCPD 文件名格式 (6段用 - 分隔):
        段0: 无关ID
        段1: 倾斜角度
        段2: 车牌 bounding box (x1&y1_x2&y2)
        段3: 车牌四角透视坐标 (左上_左下_右下_右上)，格式: x&y_x&y_x&y_x&y
        段4: 字符编码 idx1_idx2_..._idx7
        段5: 其他
    返回: (plate_text, plate_corners) 或 (plate_text, None)
        plate_corners: np.array of shape [4, 2]，顺序为 [左上, 左下, 右下, 右上]
    """
    basename = os.path.splitext(filename)[0]
    parts = basename.split('-')
    if len(parts) < 5:
        return None, None

    # --- 解析车牌文本 (段4) ---
    label_str = parts[4]
    indices = label_str.split('_')
    if len(indices) != 7:
        return None, None

    try:
        idx = [int(i) for i in indices]
        plate_text = (
            CCPD_PROVINCES[idx[0]] +
            CCPD_ALPHABETS[idx[1]] +
            CCPD_ADS[idx[2]] +
            CCPD_ADS[idx[3]] +
            CCPD_ADS[idx[4]] +
            CCPD_ADS[idx[5]] +
            CCPD_ADS[idx[6]]
        )
    except (ValueError, IndexError):
        return None, None

    # --- 解析车牌四角坐标 (段3) ---
    # 格式: "444&547_368&549_364&517_440&515"
    # 对应: 左上_左下_右下_右上
    try:
        corners_str = parts[3]
        corners = []
        for point_str in corners_str.split('_'):
            x, y = point_str.split('&')
            corners.append([int(x), int(y)])
        plate_corners = np.array(corners, dtype=np.float32)
    except (ValueError, IndexError):
        plate_corners = None

    return plate_text, plate_corners


def segment_plate_from_image(plate_img):
    """
    从完整车牌图片中切分出单个字符（复用 segmenter 的逻辑）
    返回: [(char_img, char_label), ...]
    """
    from src.core.traditional.segmenter import segment_plate_characters

    if plate_img is None or plate_img.size == 0:
        return []

    # 确保是 BGR 格式
    if len(plate_img.shape) == 2:
        plate_img = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)

    # 调整到标准尺寸 220x70
    plate_img = cv2.resize(plate_img, (220, 70))

    try:
        char_images, _ = segment_plate_characters(plate_img)
        return char_images
    except Exception:
        return []


def sort_plate_corners(corners):
    """
    将 CCPD 的四角坐标排序为标准顺序: [左上, 右上, 右下, 左下]
    CCPD 文件名中的四角顺序不固定，需要按 y 分上下、x 分左右来重排。
    """
    pts = corners.tolist()
    # 按 y 坐标排序，前两个为上边，后两个为下边
    pts_by_y = sorted(pts, key=lambda p: p[1])
    top = sorted(pts_by_y[:2], key=lambda p: p[0])   # 上边按 x 排: 左→右
    bot = sorted(pts_by_y[2:], key=lambda p: p[0])   # 下边按 x 排: 左→右
    # 返回: 左上, 右上, 右下, 左下
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)


def extract_plate_region(img, corners):
    """
    利用 CCPD 文件名中的四角坐标，通过透视变换将车牌从原图中抠出并拉直。
    参数:
        img: 原始车辆照片 (BGR)
        corners: np.array shape [4, 2]，CCPD 原始四角坐标
    返回:
        拉直后的车牌图像 (BGR, 220x70) 或 None
    """
    if corners is None or len(corners) != 4:
        return None

    # 排序为标准顺序
    src = sort_plate_corners(corners)

    # 定义目标矩形 (标准车牌比例 220x70)
    w, h = 220, 70
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h))
    return warped


def process_ccpd_dataset(ccpd_dir, output_dir, max_images_per_class=500):
    """
    处理 CCPD 数据集，提取单字符图片
    参数:
        ccpd_dir: CCPD 图片目录
        output_dir: 输出字符图片目录
        max_images_per_class: 每类最多保存的图片数
    """
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 初始化每个类别的目录 (使用绝对路径确保 Windows 中文路径兼容)
    for cls_name in PLATE_CLASSES:
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

    # 收集所有 CCPD 图片
    image_files = []
    for root, dirs, files in os.walk(ccpd_dir):
        for f in files:
            if f.endswith('.jpg') or f.endswith('.png'):
                image_files.append(os.path.join(root, f))

    if not image_files:
        logger.warning(f"在 {ccpd_dir} 中未找到任何图片文件！")
        return

    logger.info(f"找到 {len(image_files):,} 张 CCPD 图片，开始处理...")

    # 统计每个类别的计数
    class_counts = {cls: 0 for cls in PLATE_CLASSES}
    total_extracted = 0
    total_failed = 0

    random.shuffle(image_files)
    total_files = len(image_files)

    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)

        # 每处理 500 张输出一次进度
        if (idx + 1) % 500 == 0 or idx == 0:
            done_classes = sum(1 for c in class_counts.values() if c >= max_images_per_class)
            logger.info(f"  进度: {idx+1}/{total_files} | 已提取: {total_extracted} | 失败: {total_failed} | 类别完成: {done_classes}/{len(PLATE_CLASSES)}")

        # 检查是否所有类别都已足够
        if all(c >= max_images_per_class for c in class_counts.values()):
            logger.info(f"  所有类别已达到 {max_images_per_class} 张，提前结束。")
            break

        # 解析文件名获取车牌文本和四角坐标
        plate_text, plate_corners = parse_ccpd_filename(filename)
        if plate_text is None:
            continue

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 利用四角坐标透视变换抠出车牌区域
        plate_img = extract_plate_region(img, plate_corners)
        if plate_img is None:
            total_failed += 1
            continue

        # 对抠出的车牌做字符分割
        char_images = segment_plate_from_image(plate_img)

        if len(char_images) != 7:
            total_failed += 1
            continue

        # 保存每个字符
        for char_img, char_label in zip(char_images, plate_text):
            if char_label not in PLATE_CLASSES:
                continue
            if class_counts[char_label] >= max_images_per_class:
                continue

            # 缩放到 32x32
            char_resized = cv2.resize(char_img, (32, 32), interpolation=cv2.INTER_AREA)

            # 使用绝对路径 + Pillow 保存，避免 Windows 下 cv2 对中文路径编码异常
            save_path = os.path.join(output_dir, char_label, f"{class_counts[char_label]:05d}.png")
            save_path = os.path.abspath(save_path)
            from PIL import Image
            Image.fromarray(cv2.cvtColor(char_resized, cv2.COLOR_GRAY2RGB)).save(save_path)
            class_counts[char_label] += 1
            total_extracted += 1

    logger.success(f"CCPD 字符提取完成: 成功 {total_extracted:,} 张 | 分割失败 {total_failed:,} 张")
    logger.info("各类别数量:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info(f"  {cls_name}: {count}")


def create_enhanced_synthetic(output_dir, num_per_class=500):
    """
    增强版合成数据：多种字体 + 真实车牌配色 + 丰富增强
    当没有 CCPD 数据时使用
    """
    from src.core.deep_learning.dataset_synthetic import get_multiple_fonts, generate_char_image

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for cls_name in PLATE_CLASSES:
        os.makedirs(os.path.join(output_dir, cls_name), exist_ok=True)

    fonts = get_multiple_fonts(28)
    logger.info(f"使用 {len(fonts)} 种字体，每类生成 {num_per_class} 张增强合成字符图...")

    for cls_idx, cls_name in enumerate(PLATE_CLASSES):
        for i in range(num_per_class):
            font = random.choice(fonts)
            img = generate_char_image(cls_name, font, size=32)
            img_np = np.array(img)
            save_path = os.path.join(output_dir, cls_name, f"{i:05d}.png")
            save_path = os.path.abspath(save_path)
            from PIL import Image
            Image.fromarray(img_np).save(save_path)

    total = len(PLATE_CLASSES) * num_per_class
    logger.success(f"增强合成数据生成完成: {total:,} 张")


if __name__ == '__main__':
    ccpd_dir = os.path.join(DATA_DIR, 'CCPD')
    output_dir = os.path.join(DATA_DIR, 'plate_chars')

    # 递归检测子目录中是否有 .jpg 文件 (CCPD2019 结构为 CCPD/CCPD2019/ccpd_base/*.jpg)
    has_ccpd_images = False
    if os.path.exists(ccpd_dir):
        for root, dirs, files in os.walk(ccpd_dir):
            if any(f.endswith('.jpg') for f in files):
                has_ccpd_images = True
                break

    if has_ccpd_images:
        logger.info("检测到 CCPD 数据集，使用真实数据生成训练集...")
        process_ccpd_dataset(ccpd_dir, output_dir, max_images_per_class=500)
    else:
        logger.warning(f"未找到 CCPD 数据集 ({ccpd_dir})")
        logger.info("请下载 CCPD 数据集: https://github.com/detectRecog/CCPD")
        logger.info("下载后将图片放入 data/CCPD/ 目录，重新运行此脚本。")
        logger.info("当前使用增强版合成数据作为替代...")
        create_enhanced_synthetic(output_dir, num_per_class=500)
