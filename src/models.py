import warnings
import torch
from src.core.deep_learning.resnet import get_resnet18
from src.core.deep_learning.mobilenet import get_mobilenet_v3 as get_mobilenet_v3_small
from src.core.deep_learning.custom_cnn import CustomCharCNN

# 提示已弃用，建议导入新模块
warnings.warn(
    "src/models.py is deprecated. Please import from src.core.deep_learning instead.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == '__main__':
    # 快速检查网络
    resnet = get_resnet18(47, pretrained=False)
    mobilenet = get_mobilenet_v3_small(47, pretrained=False)
    custom = CustomCharCNN(47)
    print("ResNet18 output shape:", resnet(torch.randn(2, 3, 32, 32)).shape)
    print("MobileNetV3 output shape:", mobilenet(torch.randn(2, 3, 32, 32)).shape)
    print("Custom CNN output shape:", custom(torch.randn(2, 3, 32, 32)).shape)
