import cv2
import base64
from src.core.traditional.enhancer import (
    apply_adaptive_threshold,
    apply_otsu_threshold,
    apply_global_threshold,
    apply_sobel,
    apply_laplacian
)

def image_to_base64(img):
    """
    将 OpenCV 图像编码为 Base64 字符串
    """
    if img is None:
        return ""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def compare_edges(gray_img, ksize=3, low_thresh=75, high_thresh=200):
    """
    运行并获取 Sobel, Laplacian, Canny 边缘检测算子的对比结果 Base64
    """
    sobel_res = apply_sobel(gray_img, ksize=ksize)
    laplacian_res = apply_laplacian(gray_img, ksize=ksize)
    canny_res = cv2.Canny(gray_img, low_thresh, high_thresh)
    
    return {
        "sobel": image_to_base64(sobel_res),
        "laplacian": image_to_base64(laplacian_res),
        "canny": image_to_base64(canny_res)
    }

def compare_thresholds(gray_img, thresh=127, block_size=15, C=8):
    """
    运行并获取 大津法 (Otsu)、自适应二值化、全局阈值二值化的对比结果 Base64
    """
    otsu_res = apply_otsu_threshold(gray_img)
    adaptive_res = apply_adaptive_threshold(gray_img, block_size=block_size, C=C)
    global_res = apply_global_threshold(gray_img, thresh=thresh)
    
    return {
        "otsu": image_to_base64(otsu_res),
        "adaptive": image_to_base64(adaptive_res),
        "global": image_to_base64(global_res)
    }
