import pytest
import numpy as np
import cv2
from src.core.traditional.base_processor import sort_points, get_perspective_transform

def test_order_points():
    # 构造四个点：左上，右上，右下，左下
    pts = np.array([
        [100, 100],  # A
        [300, 90],   # B
        [290, 280],  # C
        [110, 290]   # D
    ], dtype="float32")
    
    # 随便打乱顺序
    shuffled_pts = pts[[2, 0, 3, 1]]
    
    ordered = sort_points(shuffled_pts)
    
    # 期望顺序：左上，右上，右下，左下
    assert np.allclose(ordered[0], [100, 100])
    assert np.allclose(ordered[1], [300, 90])
    assert np.allclose(ordered[2], [290, 280])
    assert np.allclose(ordered[3], [110, 290])

def test_warp_perspective():
    # 创建一个测试图像
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # 画一个矩形
    cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
    
    pts = np.array([
        [50, 50],
        [250, 50],
        [250, 250],
        [50, 250]
    ], dtype="float32")
    
    warped = get_perspective_transform(img, pts, 200, 200)
    assert warped.shape == (200, 200, 3)
    # 既然在中心，整个warped图应该是纯白色的 (255)
    assert np.mean(warped) > 200

