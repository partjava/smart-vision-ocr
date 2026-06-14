import pytest
import numpy as np
import cv2
from src.core.traditional.plate_locator import locate_license_plate


@pytest.fixture
def blue_plate_image():
    """创建一张带蓝色车牌区域的模拟图"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # 背景色（灰色路面）
    img[:, :] = (80, 80, 80)
    # 画一个蓝色车牌区域 (比例约 3:1)
    cv2.rectangle(img, (150, 180), (450, 280), (153, 51, 0), -1)  # BGR 蓝色
    # 在车牌上加白色文字
    cv2.putText(img, "ABC1234", (170, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return img


@pytest.fixture
def no_plate_image():
    """创建一张没有车牌的纯色图"""
    return np.full((400, 600, 3), (100, 100, 100), dtype=np.uint8)


def test_locate_returns_three_values(blue_plate_image):
    """单目标模式应返回 (warped_plate, bbox, steps)"""
    result = locate_license_plate(blue_plate_image, return_multiple=False)
    assert len(result) == 3


def test_locate_returns_multiple(blue_plate_image):
    """多目标模式应返回 (candidates, steps)"""
    result = locate_license_plate(blue_plate_image, return_multiple=True)
    assert len(result) == 2
    candidates, steps = result
    assert isinstance(candidates, list)
    assert isinstance(steps, dict)


def test_locate_steps_keys(blue_plate_image):
    """steps 字典应包含必要的中间步骤"""
    _, _, steps = locate_license_plate(blue_plate_image, return_multiple=False)
    expected_keys = {'resized', 'mask', 'closed', 'detected'}
    assert set(steps.keys()) == expected_keys


def test_locate_on_no_plate_image(no_plate_image):
    """无车牌图应返回 None"""
    warped, bbox, steps = locate_license_plate(no_plate_image, return_multiple=False)
    assert warped is None
    assert bbox is None


def test_locate_multiple_on_no_plate(no_plate_image):
    """无车牌图多目标模式应返回空列表"""
    candidates, steps = locate_license_plate(no_plate_image, return_multiple=True)
    assert isinstance(candidates, list)
    assert len(candidates) == 0


def test_locate_warped_plate_size(blue_plate_image):
    """裁剪出的车牌应为 220x70"""
    warped, bbox, steps = locate_license_plate(blue_plate_image, return_multiple=False)
    if warped is not None:
        assert warped.shape[:2] == (70, 220)  # (height, width)


def test_locate_bbox_shape(blue_plate_image):
    """bbox 应为 4 个角点"""
    warped, bbox, steps = locate_license_plate(blue_plate_image, return_multiple=False)
    if bbox is not None:
        assert bbox.shape == (4, 2)


def test_locate_multiple_max_candidates(blue_plate_image):
    """多目标模式最多返回 3 个候选"""
    candidates, steps = locate_license_plate(blue_plate_image, return_multiple=True)
    assert len(candidates) <= 3


def test_locate_respects_custom_hsv_range(blue_plate_image):
    """自定义 HSV 范围应不影响函数执行"""
    warped, bbox, steps = locate_license_plate(
        blue_plate_image,
        lower_blue_val=90, upper_blue_val=130,
        return_multiple=False
    )
    # 可能找到也可能找不到，但不应崩溃
    assert steps is not None


def test_steps_images_are_bgr(blue_plate_image):
    """所有 steps 图像应为 BGR 三通道"""
    _, _, steps = locate_license_plate(blue_plate_image, return_multiple=False)
    for key, img in steps.items():
        assert len(img.shape) == 3, f"step '{key}' 不是三通道"
        assert img.shape[2] == 3, f"step '{key}' 通道数不是 3"


def test_steps_images_have_correct_width(blue_plate_image):
    """steps 图像应缩放到 800 宽"""
    _, _, steps = locate_license_plate(blue_plate_image, return_multiple=False)
    for key, img in steps.items():
        assert img.shape[1] == 800, f"step '{key}' 宽度不是 800"


def test_yellow_plate_detection():
    """黄牌检测"""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[:, :] = (60, 60, 60)
    # 黄色车牌区域 (BGR: 0, 204, 255)
    cv2.rectangle(img, (100, 150), (400, 250), (0, 204, 255), -1)
    cv2.putText(img, "B12345", (120, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    warped, bbox, steps = locate_license_plate(img, return_multiple=False)
    # 黄牌也可能被检测到
    assert steps is not None
