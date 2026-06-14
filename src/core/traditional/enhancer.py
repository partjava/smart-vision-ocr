import cv2
import numpy as np


def preprocess_for_ocr(gray_img):
    """
    为 EasyOCR 优化的文档预处理流水线：
    1. CLAHE 自适应直方图均衡化（提升低光照/阴影区域对比度）
    2. 非局部均值去噪（保留文字边缘的同时去除拍摄噪点）
    3. 轻微锐化（恢复因拍照模糊的笔画边缘）
    """
    # 1. CLAHE 对比度增强（clipLimit=2.0 避免过度增强，tileGridSize=8x8 局部均衡）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)

    # 2. 非局部均值去噪（h=10 控制去噪强度，templateWindowSize=7 模板窗口）
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10, templateWindowSize=7, searchWindowSize=21)

    # 3. 轻微锐化（unsharp mask）
    blurred = cv2.GaussianBlur(denoised, (0, 0), 3)
    sharpened = cv2.addWeighted(denoised, 1.5, blurred, -0.5, 0)

    return sharpened

def apply_adaptive_threshold(gray_img, block_size=15, C=8):
    """
    自适应高斯二值化处理，有效过滤拍照时留下的反光与大面积背景阴影
    """
    # 确保 block_size 为大于1的奇数
    if block_size <= 1:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, block_size, C
    )

def apply_otsu_threshold(gray_img):
    """
    大津法 (Otsu's Binarization) 自动寻优阈值二值化
    """
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def apply_global_threshold(gray_img, thresh=127):
    """
    全局固定阈值二值化
    """
    _, binary = cv2.threshold(gray_img, thresh, 255, cv2.THRESH_BINARY)
    return binary

def apply_sobel(gray_img, ksize=3):
    """
    Sobel 算子边缘检测（结合 X 和 Y 方向梯度模长）
    """
    # 确保核大小为 1, 3, 5, 7 之一
    if ksize not in [1, 3, 5, 7]:
        ksize = 3
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # 计算模长并在 0~255 范围内截断
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    sobel_combined = np.clip(sobel_combined, 0, 255).astype(np.uint8)
    return sobel_combined

def apply_laplacian(gray_img, ksize=3):
    """
    Laplacian 二阶导数边缘检测算子
    """
    if ksize not in [1, 3, 5, 7]:
        ksize = 3
    lap = cv2.Laplacian(gray_img, cv2.CV_64F, ksize=ksize)
    lap = np.absolute(lap)
    lap = np.clip(lap, 0, 255).astype(np.uint8)
    return lap
