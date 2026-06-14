"""
CCPD-Base 小数据集下载 + 字符提取一键脚本
CCPD-Base 约 5000 张真实车牌图片，适合快速训练验证。

用法: conda activate pytorch1 && python download_ccpd.py
"""
import os
import sys
import zipfile
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import DATA_DIR
from src.utils.logger import logger

# CCPD-Base 下载地址（GitHub Release）
# 如果下载失败，请手动从以下地址下载：
# https://github.com/detectRecog/CCPD
CCPD_URLS = [
    # 方案1: GitHub 直链（CCPD-Base）
    "https://github.com/detectRecog/CCPD/releases/download/v1.0/CCPD2019.tar.gz",
    # 方案2: 备用链接
    # 如果上面的链接不可用，请手动下载并解压到 data/CCPD/ 目录
]

def download_file(url, dest):
    """下载文件并显示进度"""
    logger.info(f"正在下载: {url}")
    logger.info(f"保存到: {dest}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  下载进度: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=progress)
        print()  # 换行
        return True
    except Exception as e:
        print()
        logger.error(f"下载失败: {e}")
        return False


def extract_archive(archive_path, extract_dir):
    """解压 tar.gz 或 zip 文件"""
    logger.info(f"正在解压: {archive_path}")

    if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        import tarfile
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_dir)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    else:
        logger.error(f"不支持的压缩格式: {archive_path}")
        return False

    logger.info(f"解压完成: {extract_dir}")
    return True


def find_ccpd_images(base_dir):
    """递归查找 CCPD 图片"""
    image_files = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith('.jpg'):
                image_files.append(os.path.join(root, f))
    return image_files


def main():
    ccpd_dir = os.path.join(DATA_DIR, 'CCPD')
    archive_path = os.path.join(DATA_DIR, 'ccpd_base.tar.gz')

    # 检查是否已有 CCPD 数据
    existing_images = find_ccpd_images(ccpd_dir) if os.path.exists(ccpd_dir) else []

    if len(existing_images) > 100:
        logger.info(f"已检测到 {len(existing_images)} 张 CCPD 图片，跳过下载。")
    else:
        logger.info("未检测到 CCPD 数据，开始下载 CCPD-Base...")
        logger.info("")
        logger.info("如果自动下载失败，请手动操作：")
        logger.info("1. 访问 https://github.com/detectRecog/CCPD")
        logger.info("2. 下载 CCPD2019 或 CCPD-Base")
        logger.info("3. 将图片解压到 data/CCPD/ 目录")
        logger.info("")

        # 尝试下载
        success = False
        for url in CCPD_URLS:
            if download_file(url, archive_path):
                success = True
                break

        if not success:
            logger.error("自动下载失败。请手动下载 CCPD 数据集。")
            logger.info("")
            logger.info("手动下载步骤：")
            logger.info("1. 打开 https://github.com/detectRecog/CCPD")
            logger.info("2. 点击 Releases 或直接下载 ZIP")
            logger.info("3. 解压到 data/CCPD/ 目录")
            logger.info("4. 确保 data/CCPD/ 下直接有 .jpg 文件或子文件夹")
            return

        # 解压
        os.makedirs(ccpd_dir, exist_ok=True)
        if not extract_archive(archive_path, ccpd_dir):
            return

        # 清理压缩包
        if os.path.exists(archive_path):
            os.remove(archive_path)

        # 再次检查
        existing_images = find_ccpd_images(ccpd_dir)
        if not existing_images:
            logger.error("解压后未找到图片。请检查 data/CCPD/ 目录结构。")
            return

    logger.info(f"找到 {len(existing_images)} 张 CCPD 图片")

    # 提取字符
    logger.info("开始从 CCPD 图片中提取单字符...")
    from src.core.deep_learning.plate_dataset import process_ccpd_dataset
    output_dir = os.path.join(DATA_DIR, 'plate_chars_ccpd')
    process_ccpd_dataset(ccpd_dir, output_dir, max_images_per_class=500)

    # 替换原有的 plate_chars 目录
    old_dir = os.path.join(DATA_DIR, 'plate_chars')
    if os.path.exists(old_dir):
        import shutil
        backup_dir = os.path.join(DATA_DIR, 'plate_chars_backup')
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)
        os.rename(old_dir, backup_dir)
        logger.info(f"旧数据已备份到: {backup_dir}")

    os.rename(output_dir, old_dir)
    logger.success(f"CCPD 字符数据已保存到: {old_dir}")

    logger.info("")
    logger.info("接下来运行训练：")
    logger.info("  python train_all.py")


if __name__ == '__main__':
    main()
