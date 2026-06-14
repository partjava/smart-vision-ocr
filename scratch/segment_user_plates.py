import os
import shutil
import cv2
import sys
from tqdm import tqdm

# Ensure src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.helpers import DATA_DIR, PLATE_CLASSES
from src.core.traditional.segmenter import segment_plate_characters

# Paths
src_dir = os.path.join(DATA_DIR, 'plate_images')
dst_dir = os.path.join(DATA_DIR, 'plate_chars')

print("正在清理旧的合成字符数据...")
# 清理旧的子文件夹
if os.path.exists(dst_dir):
    for item in os.listdir(dst_dir):
        item_path = os.path.join(dst_dir, item)
        if os.path.isdir(item_path) and item != ".cache":
            shutil.rmtree(item_path)

# 重新创建类别文件夹
for cls in PLATE_CLASSES:
    os.makedirs(os.path.join(dst_dir, cls), exist_ok=True)

# 列出所有待处理的真实车牌图像
image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
print(f"找到 {len(image_files)} 张真实车牌图片，开始进行字符分割与整理...")

success_count = 0
failed_count = 0

for fname in tqdm(image_files):
    # 解析文件名获取车牌文本
    # 示例: "010120200401082533497_闽D2315W.jpg" -> "闽D2315W"
    try:
        plate_text = os.path.splitext(fname)[0].split('_')[-1]
    except Exception:
        continue
    
    img_path = os.path.join(src_dir, fname)
    # 使用 np.fromfile 从磁盘读取二进制数据，再用 cv2.imdecode 解码，以支持 Windows 下的中文路径
    import numpy as np
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        continue
        
    # 将车牌图像缩放到 220x70（字符切割器要求的标准输入尺寸）
    img_resized = cv2.resize(img, (220, 70))
    
    # 运行垂直投影法切分字符
    try:
        char_images, _ = segment_plate_characters(img_resized)
    except Exception:
        failed_count += 1
        continue
        
    # 校验切割出的字符数量是否与文件名中的车牌字符数一致，一致才保存（防止误标）
    if len(char_images) != len(plate_text):
        failed_count += 1
        continue
        
    # 保存字符图像
    for idx, (c_img, char) in enumerate(zip(char_images, plate_text)):
        if char not in PLATE_CLASSES:
            continue
        
        # 统一缩放到 32x32，并保存为黑底白字 png
        c_resized = cv2.resize(c_img, (32, 32), interpolation=cv2.INTER_AREA)
        
        # 使用 Pillow 保存以解决 Windows 对中文路径的兼容问题
        from PIL import Image
        save_path = os.path.join(dst_dir, char, f"{os.path.splitext(fname)[0]}_{idx}.png")
        Image.fromarray(cv2.cvtColor(c_resized, cv2.COLOR_GRAY2RGB)).save(os.path.abspath(save_path))
        
    success_count += 1

print(f"\n[完成] 成功分割并导出 {success_count} 张车牌，共有 {failed_count} 张车牌分割不匹配或失败被跳过。")
