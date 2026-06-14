import pytest
import numpy as np
import cv2
from src.core.traditional.segmenter import (
    get_horizontal_projection,
    get_vertical_projection,
    segment_lines_by_projection,
    segment_chars_by_projection,
    segment_plate_characters,
    preprocess_single_char
)


@pytest.fixture
def binary_text_image():
    """创建一张带有多行文字的二值图"""
    img = np.zeros((100, 300), dtype=np.uint8)
    # 第一行
    cv2.putText(img, "ABC", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    # 第二行
    cv2.putText(img, "DEF", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return img


@pytest.fixture
def binary_plate_like():
    """创建一张模拟车牌的 BGR 图 (220x70)"""
    img = np.zeros((70, 220, 3), dtype=np.uint8)
    # 画 7 个字符区域（白色矩形）
    for i in range(7):
        x = 10 + i * 30
        cv2.rectangle(img, (x, 10), (x + 20, 60), (255, 255, 255), -1)
    return img


def test_horizontal_projection_shape(binary_text_image):
    proj = get_horizontal_projection(binary_text_image)
    assert proj.shape == (100,)  # height 个元素


def test_vertical_projection_shape(binary_text_image):
    proj = get_vertical_projection(binary_text_image)
    assert proj.shape == (300,)  # width 个元素


def test_horizontal_projection_detects_lines(binary_text_image):
    proj = get_horizontal_projection(binary_text_image)
    # 应该有两个峰值区域（两行文字）
    # 非零行应该存在
    assert np.max(proj) > 0


def test_segment_lines_finds_two_lines(binary_text_image):
    lines = segment_lines_by_projection(binary_text_image, threshold=2, min_height=5)
    assert len(lines) >= 2  # 应该找到至少两行


def test_segment_chars_on_single_line(binary_text_image):
    """对单行文字做垂直投影分割"""
    lines = segment_lines_by_projection(binary_text_image, threshold=2, min_height=5)
    assert len(lines) >= 1
    y_start, y_end = lines[0]
    line_img = binary_text_image[y_start:y_end, :]
    chars = segment_chars_by_projection(line_img, threshold=1, min_width=3)
    assert len(chars) >= 1  # 至少切出一个字符


def test_segment_chars_returns_valid_ranges():
    """垂直投影分割应返回有效的 (start, end) 区间"""
    img = np.zeros((30, 100), dtype=np.uint8)
    cv2.putText(img, "AB", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    chars = segment_chars_by_projection(img, threshold=1, min_width=3)
    for x_start, x_end in chars:
        assert x_end > x_start
        assert x_start >= 0
        assert x_end <= 100


def test_preprocess_single_char_output_size():
    """preprocess_single_char 应输出 32x32 图像"""
    roi = np.ones((20, 15), dtype=np.uint8) * 255
    result = preprocess_single_char(roi, target_size=32)
    assert result.shape == (32, 32)
    assert result.dtype == np.uint8


def test_preprocess_single_char_empty_roi():
    """空 ROI 应返回全黑图像"""
    roi = np.zeros((0, 10), dtype=np.uint8)
    result = preprocess_single_char(roi, target_size=32)
    assert result.shape == (32, 32)
    assert np.sum(result) == 0


def test_preprocess_single_char_zero_dimension():
    """零尺寸 ROI 应返回全黑图像"""
    roi = np.zeros((10, 0), dtype=np.uint8)
    result = preprocess_single_char(roi, target_size=32)
    assert result.shape == (32, 32)


def test_preprocess_single_char_preserves_content():
    """缩放后字符应居中且保留白色像素"""
    roi = np.zeros((40, 20), dtype=np.uint8)
    cv2.rectangle(roi, (2, 5), (18, 35), 255, -1)
    result = preprocess_single_char(roi, target_size=32)
    # 居中区域应有白色像素
    center_region = result[4:28, 4:28]
    assert np.sum(center_region) > 0


def test_segment_plate_characters_output(binary_plate_like):
    """segment_plate_characters 应返回字符列表和可视化图"""
    char_images, vis_img = segment_plate_characters(binary_plate_like)
    assert isinstance(char_images, list)
    assert len(char_images) >= 1
    # 每个字符图应为 32x32
    for c_img in char_images:
        assert c_img.shape == (32, 32)
    # 可视化图应为彩色
    assert len(vis_img.shape) == 3
    assert vis_img.shape[2] == 3


def test_segment_plate_characters_on_blank_image():
    """空白图不应崩溃"""
    blank = np.zeros((70, 220, 3), dtype=np.uint8)
    char_images, vis_img = segment_plate_characters(blank)
    assert isinstance(char_images, list)
    # 空白图可能切出 0 个或 7 个（均分兜底）
