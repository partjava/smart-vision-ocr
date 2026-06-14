import pytest
import numpy as np
import cv2
from src.core.traditional.enhancer import (
    apply_adaptive_threshold,
    apply_otsu_threshold,
    apply_global_threshold,
    apply_sobel,
    apply_laplacian
)


@pytest.fixture
def sample_gray_image():
    """创建一张带渐变的测试灰度图"""
    img = np.zeros((100, 200), dtype=np.uint8)
    # 左侧暗区
    img[:, :100] = 50
    # 右侧亮区
    img[:, 100:] = 200
    # 加一些文字模拟
    cv2.putText(img, "Hello", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 2)
    return img


def test_adaptive_threshold_output_shape(sample_gray_image):
    result = apply_adaptive_threshold(sample_gray_image)
    assert result.shape == sample_gray_image.shape
    assert result.dtype == np.uint8


def test_adaptive_threshold_binary_output(sample_gray_image):
    result = apply_adaptive_threshold(sample_gray_image)
    unique_values = np.unique(result)
    assert set(unique_values).issubset({0, 255})


def test_adaptive_threshold_even_block_size(sample_gray_image):
    """偶数 block_size 应自动+1变为奇数"""
    result = apply_adaptive_threshold(sample_gray_image, block_size=16)
    assert result.shape == sample_gray_image.shape


def test_adaptive_threshold_small_block_size(sample_gray_image):
    """block_size <= 1 应被修正为 3"""
    result = apply_adaptive_threshold(sample_gray_image, block_size=1)
    assert result.shape == sample_gray_image.shape


def test_otsu_threshold_output(sample_gray_image):
    result = apply_otsu_threshold(sample_gray_image)
    assert result.shape == sample_gray_image.shape
    unique_values = np.unique(result)
    assert set(unique_values).issubset({0, 255})


def test_global_threshold_output(sample_gray_image):
    result = apply_global_threshold(sample_gray_image, thresh=127)
    assert result.shape == sample_gray_image.shape
    unique_values = np.unique(result)
    assert set(unique_values).issubset({0, 255})


def test_sobel_output_shape(sample_gray_image):
    result = apply_sobel(sample_gray_image, ksize=3)
    assert result.shape == sample_gray_image.shape
    assert result.dtype == np.uint8


def test_sobel_invalid_ksize(sample_gray_image):
    """无效 ksize 应回退到 3"""
    result = apply_sobel(sample_gray_image, ksize=4)
    assert result.shape == sample_gray_image.shape


def test_laplacian_output_shape(sample_gray_image):
    result = apply_laplacian(sample_gray_image, ksize=3)
    assert result.shape == sample_gray_image.shape
    assert result.dtype == np.uint8


def test_laplacian_invalid_ksize(sample_gray_image):
    """无效 ksize 应回退到 3"""
    result = apply_laplacian(sample_gray_image, ksize=2)
    assert result.shape == sample_gray_image.shape


def test_sobel_detects_edges(sample_gray_image):
    """Sobel 应该在明暗交界处产生高响应"""
    result = apply_sobel(sample_gray_image, ksize=3)
    # x=100 附近是明暗交界，应有较高梯度值
    assert np.max(result[:, 95:105]) > 50


def test_all_thresholds_on_uniform_image():
    """纯色图上二值化应产生全 0 或全 255"""
    uniform = np.full((50, 50), 128, dtype=np.uint8)
    assert apply_adaptive_threshold(uniform).shape == (50, 50)
    assert apply_otsu_threshold(uniform).shape == (50, 50)
    assert apply_global_threshold(uniform, thresh=127).shape == (50, 50)
