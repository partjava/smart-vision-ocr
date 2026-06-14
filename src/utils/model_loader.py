import os
import torch
from src.utils.helpers import DEVICE, EMNIST_CLASSES, PLATE_CLASSES, WEIGHTS_DIR
from src.core.deep_learning.resnet import get_resnet18
from src.core.deep_learning.mobilenet import get_mobilenet_v3
from src.core.deep_learning.custom_cnn import CustomCharCNN

# 全局模型字典缓存
models = {
    'emnist': {
        'resnet18': None,
        'mobilenet': None,
        'custom_cnn': None
    },
    'plate': {
        'resnet18': None,
        'mobilenet': None,
        'custom_cnn': None
    }
}

def load_all_models():
    """
    加载所有 PyTorch 模型权重
    """
    global models
    
    # 1. 类别数定义
    emnist_classes_num = len(EMNIST_CLASSES)
    plate_classes_num = len(PLATE_CLASSES)
    
    # --- EMNIST 字符识别模型 ---
    resnet_emnist_path = os.path.join(WEIGHTS_DIR, 'best_resnet18_emnist.pth')
    if os.path.exists(resnet_emnist_path):
        try:
            m = get_resnet18(num_classes=emnist_classes_num, pretrained=False)
            m.load_state_dict(torch.load(resnet_emnist_path, map_location=DEVICE, weights_only=True))
            m.to(DEVICE).eval()
            models['emnist']['resnet18'] = m
            print("[Success] 已加载 ResNet18 EMNIST 权重。")
        except Exception as e:
            print(f"[Error] 加载 ResNet18 EMNIST 权重失败: {e}")
            
    mob_emnist_path = os.path.join(WEIGHTS_DIR, 'best_mobilenet_emnist.pth')
    if os.path.exists(mob_emnist_path):
        try:
            m = get_mobilenet_v3(num_classes=emnist_classes_num, pretrained=False)
            m.load_state_dict(torch.load(mob_emnist_path, map_location=DEVICE, weights_only=True))
            m.to(DEVICE).eval()
            models['emnist']['mobilenet'] = m
            print("[Success] 已加载 MobileNetV3 EMNIST 权重。")
        except Exception as e:
            print(f"[Error] 加载 MobileNetV3 EMNIST 权重失败: {e}")
            
    custom_emnist_path = os.path.join(WEIGHTS_DIR, 'best_custom_cnn_emnist.pth')
    if os.path.exists(custom_emnist_path):
        try:
            m = CustomCharCNN(num_classes=emnist_classes_num)
            m.load_state_dict(torch.load(custom_emnist_path, map_location=DEVICE, weights_only=True))
            m.to(DEVICE).eval()
            models['emnist']['custom_cnn'] = m
            print("[Success] 已加载 CustomCharCNN EMNIST 权重。")
        except Exception as e:
            print(f"[Error] 加载 CustomCharCNN EMNIST 权重失败: {e}")

    # --- 车牌字符识别模型 ---
    resnet_plate_path = os.path.join(WEIGHTS_DIR, 'best_resnet18_plate.pth')
    if os.path.exists(resnet_plate_path):
        try:
            m = get_resnet18(num_classes=plate_classes_num, pretrained=False)
            m.load_state_dict(torch.load(resnet_plate_path, map_location=DEVICE, weights_only=True))
            m.to(DEVICE).eval()
            models['plate']['resnet18'] = m
            print("[Success] 已加载 ResNet18 Plate 权重。")
        except Exception as e:
            print(f"[Error] 加载 ResNet18 Plate 权重失败: {e}")
            
    mob_plate_path = os.path.join(WEIGHTS_DIR, 'best_mobilenet_plate.pth')
    if os.path.exists(mob_plate_path):
        try:
            m = get_mobilenet_v3(num_classes=plate_classes_num, pretrained=False)
            m.load_state_dict(torch.load(mob_plate_path, map_location=DEVICE, weights_only=True))
            m.to(DEVICE).eval()
            models['plate']['mobilenet'] = m
            print("[Success] 已加载 MobileNetV3 Plate 权重。")
        except Exception as e:
            print(f"[Error] 加载 MobileNetV3 Plate 权重失败: {e}")
            
    custom_plate_path = os.path.join(WEIGHTS_DIR, 'best_custom_cnn_plate.pth')
    if os.path.exists(custom_plate_path):
        try:
            m = CustomCharCNN(num_classes=plate_classes_num)
            m.load_state_dict(torch.load(custom_plate_path, map_location=DEVICE, weights_only=True))
            m.to(DEVICE).eval()
            models['plate']['custom_cnn'] = m
            print("[Success] 已加载 CustomCharCNN Plate 权重。")
        except Exception as e:
            print(f"[Error] 加载 CustomCharCNN Plate 权重失败: {e}")
