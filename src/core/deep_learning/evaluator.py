import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

# 导入配置与工具
from src.utils.helpers import DEVICE, WEIGHTS_DIR, RESULTS_DIR, EMNIST_CLASSES, PLATE_CLASSES
from src.utils.logger import logger
from src.core.deep_learning.resnet import get_resnet18
from src.core.deep_learning.mobilenet import get_mobilenet_v3
from src.core.deep_learning.custom_cnn import CustomCharCNN
from src.core.deep_learning.dataset_emnist import get_emnist_dataloaders
from src.core.deep_learning.dataset_synthetic import get_synthetic_dataloaders

# Matplotlib 中文支持配置，防止中文车牌省份简称乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def calculate_metrics(y_true, y_pred):
    """
    计算宏平均精确率、召回率和 F1-Score
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    return precision, recall, f1

def get_model_size(model_path):
    """
    计算模型文件大小 (MB)
    """
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0

def count_parameters(model):
    """
    计算模型的可训练参数量 (单位: M)
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def plot_confusion_matrix(y_true, y_pred, classes, model_name, dataset_name):
    """
    绘制并保存归一化的混淆矩阵热图
    """
    cm = confusion_matrix(y_true, y_pred)
    # 归一化，处理单行和为 0 的异常
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    row_sums[row_sums == 0] = 1.0
    cm_normalized = cm.astype('float') / row_sums
    
    # 动态适应画布尺寸
    fig_size = 14 if len(classes) > 50 else 10
    plt.figure(figsize=(fig_size, fig_size - 1))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name.upper()} ({dataset_name}) Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    save_img_name = f"{model_name}_{dataset_name}_confusion.png"
    cm_path = os.path.join(RESULTS_DIR, save_img_name)
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info(f"混淆矩阵热图保存至: {cm_path}")

@torch.no_grad()
def evaluate_model(model_type, dataset_type, dataloader):
    """
    对特定模型与数据集进行指标评估
    """
    logger.info(f"正在评估：模型={model_type}, 数据集={dataset_type}")
    
    # 1. 确定类别
    if dataset_type == 'emnist':
        num_classes = len(EMNIST_CLASSES)
        classes = EMNIST_CLASSES
    else:
        num_classes = len(PLATE_CLASSES)
        classes = PLATE_CLASSES
        
    # 2. 加载模型与权重
    if model_type == 'resnet18':
        model = get_resnet18(num_classes=num_classes, pretrained=False)
    elif model_type == 'mobilenet':
        model = get_mobilenet_v3(num_classes=num_classes, pretrained=False)
    elif model_type == 'custom_cnn':
        model = CustomCharCNN(num_classes=num_classes)
    else:
        raise ValueError(f"未知的模型类型: {model_type}")
        
    weights_filename = f"best_{model_type}_{dataset_type}.pth"
    weights_path = os.path.join(WEIGHTS_DIR, weights_filename)
    
    if not os.path.exists(weights_path):
        logger.warning(f"权重文件不存在: {weights_path}。评估将使用初始随机权重。")
    else:
        model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
        
    model = model.to(DEVICE)
    model.eval()
    
    # 3. 统计模型基本物理指标
    params = count_parameters(model)
    file_size = get_model_size(weights_path)
    
    # 4. 测试集指标评估
    y_true = []
    y_pred = []
    latencies = []
    
    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE)
        
        # 精确测试推理延时
        start_time = time.time()
        outputs = model(inputs)
        batch_time = time.time() - start_time
        
        # 计算单张图片推理延时 (ms)
        latencies.append((batch_time / inputs.size(0)) * 1000)
        
        _, predicted = outputs.max(1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 指标计算
    acc = np.mean(y_true == y_pred)
    precision, recall, f1 = calculate_metrics(y_true, y_pred)
    avg_latency = np.mean(latencies)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
    
    logger.info(f"[{model_type.upper()} on {dataset_type.upper()}] Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%, FPS: {fps:.1f}")
    
    # 5. 绘制混淆矩阵
    plot_confusion_matrix(y_true, y_pred, classes, model_type, dataset_type)
    
    return {
        'model_name': model_type.upper(),
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'file_size_mb': float(file_size),
        'parameters_m': float(params),
        'latency_ms': float(avg_latency),
        'fps': float(fps)
    }

def main():
    # 1. 准备 EMNIST 评估
    logger.info("开始加载 EMNIST 评估测试集...")
    _, _, emnist_test_loader = get_emnist_dataloaders(download=False)
    
    # 2. 准备 车牌字符 评估
    logger.info("开始加载合成车牌评估测试集...")
    _, _, plate_test_loader = get_synthetic_dataloaders()
    
    results = {
        'emnist': {},
        'plate': {}
    }
    
    models = ['resnet18', 'mobilenet', 'custom_cnn']
    
    # 评估 EMNIST
    for m in models:
        try:
            results['emnist'][m] = evaluate_model(m, 'emnist', emnist_test_loader)
        except Exception as e:
            logger.error(f"评估 EMNIST 模型 {m} 失败: {e}", exc_info=True)
            
    # 评估 Plate
    for m in models:
        try:
            results['plate'][m] = evaluate_model(m, 'plate', plate_test_loader)
        except Exception as e:
            logger.error(f"评估 Plate 模型 {m} 失败: {e}", exc_info=True)
            
    # 保存结果 json
    comparison_path = os.path.join(RESULTS_DIR, 'model_comparison.json')
    with open(comparison_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
        
    logger.info(f"所有评估完成。对比数据已写入: {comparison_path}")

if __name__ == '__main__':
    main()
