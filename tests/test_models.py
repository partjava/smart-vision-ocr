import pytest
import torch
from src.core.deep_learning.resnet import get_resnet18
from src.core.deep_learning.mobilenet import get_mobilenet_v3
from src.core.deep_learning.custom_cnn import CustomCharCNN

def test_resnet18_forward():
    num_classes = 46  # EMNIST Balanced 实际类别数
    model = get_resnet18(num_classes=num_classes, pretrained=False)
    model.eval()
    
    # EMNIST input size is usually 32x32 (resized in transform)
    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)
    
    assert output.shape == (2, num_classes)

def test_mobilenet_v3_forward():
    num_classes = 67  # 车牌字符集: 31 省份简称 + 36 字母数字
    model = get_mobilenet_v3(num_classes=num_classes, pretrained=False)
    model.eval()
    
    dummy_input = torch.randn(4, 3, 32, 32)
    output = model(dummy_input)
    
    assert output.shape == (4, num_classes)

def test_custom_cnn_forward():
    num_classes = 46  # EMNIST Balanced 实际类别数
    model = CustomCharCNN(num_classes=num_classes)
    model.eval()
    
    dummy_input = torch.randn(3, 3, 32, 32)
    output = model(dummy_input)
    
    assert output.shape == (3, num_classes)
