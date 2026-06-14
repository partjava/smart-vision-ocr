import cv2
import numpy as np
from src.core.traditional.base_processor import resize_image_width, get_perspective_transform

def scan_document(img, canny_low=75, canny_high=200, gaussian_k=5):
    """
    使用 OpenCV 传统算法扫描定位文档（边缘检测 + 轮廓拟合 + 透视变换拉平）
    参数:
        img: 输入图像 (BGR)
        canny_low: Canny 算子最低梯度阈值
        canny_high: Canny 算子最高梯度阈值
        gaussian_k: 高斯滤波核尺寸 (必须为奇数)
    返回:
        warped_gray: 拉直后的文档灰度图像 (Grayscale)，若未找到边界则返回中心裁剪兜底
        doc_pts_orig: 原图中的角点坐标 (shape: [4, 2])
        steps: 算法中间处理步骤的图像字典 (用于前端展示)
    """
    h, w = img.shape[:2]
    # 统一等比缩放到宽度 800
    resized, scale = resize_image_width(img, target_width=800)
    orig_resized = resized.copy()
    
    # 1. 灰度化与滤波去噪
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # 确保高斯核大小为奇数
    if gaussian_k % 2 == 0:
        gaussian_k += 1
    blurred = cv2.GaussianBlur(gray, (gaussian_k, gaussian_k), 0)
    
    # 2. Canny 边缘检测
    canny = cv2.Canny(blurred, canny_low, canny_high)
    
    # 3. 膨胀边缘以连接轻微断裂的线条
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)
    
    # 4. 查找最大轮廓并过滤出排名前5的轮廓
    contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    doc_pts = None
    contour_drawn = orig_resized.copy()
    
    for c in contours:
        peri = cv2.arcLength(c, True)
        # 用周长的 2% 拟合多边形
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        
        # 拟合顶点数恰为 4 时，我们假定这就是文档的四边形轮廓
        if len(approx) == 4:
            doc_pts = approx
            break
            
    # 5. 透视变换拉直纸张 (A4纸比例设为 600x840)
    target_w, target_h = 600, 840
    doc_pts_orig = None
    
    if doc_pts is not None:
        doc_pts_orig = (doc_pts.reshape(4, 2) / scale).astype(int)
        cv2.drawContours(contour_drawn, [doc_pts], -1, (0, 0, 255), 2)
        warped_color = get_perspective_transform(resized, doc_pts, target_w, target_h)
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
    else:
        # 兜底操作：若检测失败，给图像中间画框并裁剪
        cv2.putText(contour_drawn, "No boundary detected. Center crop used.", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # 居中裁剪
        crop_x = int(0.1 * resized.shape[1])
        crop_y = int(0.1 * resized.shape[0])
        crop_w = int(0.8 * resized.shape[1])
        crop_h = int(0.8 * resized.shape[0])
        
        fallback_pts = np.array([
            [crop_x, crop_y],
            [crop_x + crop_w, crop_y],
            [crop_x + crop_w, crop_y + crop_h],
            [crop_x, crop_y + crop_h]
        ], dtype="float32")
        
        warped_color = get_perspective_transform(resized, fallback_pts, target_w, target_h)
        warped_gray = cv2.cvtColor(warped_color, cv2.COLOR_BGR2GRAY)
        
    steps = {
        "resized": orig_resized,
        "canny": cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR),
        "contours": contour_drawn,
        "warped_gray": cv2.cvtColor(warped_gray, cv2.COLOR_GRAY2BGR)
    }
    
    return warped_gray, doc_pts_orig, steps
