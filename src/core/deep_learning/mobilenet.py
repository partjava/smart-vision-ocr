import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Small_Weights

def get_mobilenet_v3(num_classes, pretrained=True):
    """
    构建 MobileNetV3-Small 模型并微调其分类器最后一层。
    输入图像期望为 3 通道。
    """
    if pretrained:
        weights = MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    else:
        model = models.mobilenet_v3_small()
        
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
