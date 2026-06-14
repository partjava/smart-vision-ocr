import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """
    通用 GradCAM 实现：对任意 CNN 模型的最后一层卷积特征图生成类激活热力图。
    用于模型可解释性可视化，展示模型"关注的区域"。
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        生成 GradCAM 热力图。
        参数:
            input_tensor: 模型输入张量 (1, C, H, W)
            target_class: 目标类别索引，若为 None 则使用预测概率最高的类别
        返回:
            heatmap: 归一化到 [0, 1] 的热力图 numpy 数组 (H, W)
            predicted_class: 预测类别索引
            confidence: 预测置信度
        """
        self.model.eval()
        output = self.model(input_tensor)

        # 获取预测类别
        prob = F.softmax(output, dim=1).squeeze(0)
        if target_class is None:
            target_class = torch.argmax(prob).item()
        confidence = float(prob[target_class].item())

        # 反向传播目标类别的 score
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        # 计算权重：梯度的全局平均池化
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # 加权组合特征图
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # 只关注正贡献

        # 归一化到 [0, 1]
        cam = cam.squeeze()  # (H, W)
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = torch.zeros_like(cam)

        return cam.cpu().numpy(), target_class, confidence


def get_last_conv_layer(model):
    """
    自动获取模型的最后一层卷积层。
    支持 ResNet、MobileNet、CustomCharCNN。
    """
    # ResNet: layer4 是最后一个残差块
    if hasattr(model, 'layer4'):
        return model.layer4
    # MobileNetV3: features[-1] 是最后的卷积块
    if hasattr(model, 'features'):
        for layer in reversed(model.features):
            if hasattr(layer, 'weight') and len(layer.weight.shape) == 4:
                return layer
            # MobileNetV3 的 InvertedResidual 包含 conv 层
            if hasattr(layer, 'block'):
                for sub in reversed(layer.block):
                    if hasattr(sub, 'weight') and len(sub.weight.shape) == 4:
                        return sub
    # CustomCharCNN: 最后一个 conv 是 conv4
    if hasattr(model, 'conv4'):
        return model.conv4

    # 通用回退：遍历所有模块找最后一个 Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    return last_conv


def overlay_heatmap_on_image(img_np, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    将 GradCAM 热力图叠加到原始图像上。
    参数:
        img_np: 原始图像 numpy 数组 (H, W, 3)，RGB 格式，值域 [0, 255]
        heatmap: 热力图 numpy 数组 (h, w)，值域 [0, 1]
        alpha: 热力图透明度
    返回:
        叠加后的图像 numpy 数组 (H, W, 3)
    """
    h, w = img_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = np.float32(img_np) / 255.0 * (1 - alpha) + np.float32(heatmap_colored) / 255.0 * alpha
    overlay = np.uint8(255 * overlay / overlay.max()) if overlay.max() > 0 else np.uint8(overlay * 255)
    return overlay
