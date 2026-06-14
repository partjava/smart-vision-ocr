import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def get_resnet18(num_classes, pretrained=True):
    """
    构建 ResNet18 模型并微调最后一层全连接层。
    输入图像期望为 3 通道。
    """
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18()
        
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
