import cv2
import numpy as np

def sort_points(pts):
    """
    对4个顶点进行排序，顺序为：左上、右上、右下、左下
    """
    pts = pts.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # 左上角的点之和最小，右下角的点之和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上角的点之差最小，左下角的点之差最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def get_perspective_transform(image, pts, target_width, target_height):
    """
    利用透视变换对4个点框出来的区域进行裁剪与拉平矫正
    """
    rect = sort_points(pts)
    dst = np.array([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1]
    ], dtype="float32")

    # 计算透视变换矩阵并应用
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (target_width, target_height))
    return warped

def resize_image_width(image, target_width=800):
    """
    保持宽高比等比缩放图像宽度到指定像素，并返回缩放后的图像与缩放比例
    """
    h, w = image.shape[:2]
    scale = float(target_width) / w
    resized = cv2.resize(image, (target_width, int(h * scale)))
    return resized, scale
