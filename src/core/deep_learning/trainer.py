import os
import json
import argparse
import re
import unicodedata
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


# 导入配置与工具
from src.utils.helpers import DEVICE, WEIGHTS_DIR, RESULTS_DIR, EMNIST_CLASSES, PLATE_CLASSES, EPOCHS, LR
from src.utils.logger import logger
from src.core.deep_learning.resnet import get_resnet18
from src.core.deep_learning.mobilenet import get_mobilenet_v3
from src.core.deep_learning.custom_cnn import CustomCharCNN
from src.core.deep_learning.dataset_emnist import get_emnist_dataloaders
from src.core.deep_learning.dataset_synthetic import get_synthetic_dataloaders

# 移除所有边框/表格对齐辅助函数，控制台输出改用极简单行横向日志与进度条

class ModelTrainer:
    """
    面向对象设计的通用对比训练管理类
    """
    def __init__(self, model_type, dataset_type, epochs=EPOCHS, lr=LR):
        self.model_type = model_type.lower()
        self.dataset_type = dataset_type.lower()
        self.epochs = epochs
        self.lr = lr
        
        # 确定类别数
        if self.dataset_type == 'emnist':
            self.num_classes = len(EMNIST_CLASSES)
            self.classes = EMNIST_CLASSES
        elif self.dataset_type == 'plate':
            self.num_classes = len(PLATE_CLASSES)
            self.classes = PLATE_CLASSES
        else:
            raise ValueError(f"不支持的数据集类型: {dataset_type}")
            
        # 保存路径
        self.weights_filename = f"best_{self.model_type}_{self.dataset_type}.pth"
        self.save_path = os.path.join(WEIGHTS_DIR, self.weights_filename)
        
        # 初始化模型
        self.model = self._init_model()
        self.model = self.model.to(DEVICE)
        
        # 优化器与损失函数
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
        # 数据加载器缓存
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
    def _init_model(self):
        if self.model_type == 'resnet18':
            return get_resnet18(num_classes=self.num_classes, pretrained=True)
        elif self.model_type == 'mobilenet':
            return get_mobilenet_v3(num_classes=self.num_classes, pretrained=True)
        elif self.model_type == 'custom_cnn':
            return CustomCharCNN(num_classes=self.num_classes)
        else:
            raise ValueError(f"未知的模型类型: {self.model_type}")

    def load_data(self):
        logger.info(f"正在加载 {self.dataset_type} 数据集...")
        if self.dataset_type == 'emnist':
            self.train_loader, self.val_loader, self.test_loader = get_emnist_dataloaders(download=True)
        elif self.dataset_type == 'plate':
            self.train_loader, self.val_loader, self.test_loader = get_synthetic_dataloaders()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1:02d}/{self.epochs:02d}]", leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{correct/total*100:.2f}%"})
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def val_epoch(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False, bar_format="{l_bar}{bar:30}{r_bar}")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'Acc': f"{correct/total*100:.2f}%"})
            
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def plot_history(self, history):
        epochs_range = range(1, len(history['train_loss']) + 1)
        plt.figure(figsize=(12, 5))
        
        # Loss 曲线
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history['train_loss'], 'r-o', label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], 'b-x', label='Val Loss')
        plt.title(f'{self.model_type.upper()} ({self.dataset_type}) Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy 曲线
        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, history['train_acc'], 'r-o', label='Train Acc')
        plt.plot(epochs_range, history['val_acc'], 'b-x', label='Val Acc')
        plt.title(f'{self.model_type.upper()} ({self.dataset_type}) Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(RESULTS_DIR, f'{self.model_type}_{self.dataset_type}_curves.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        logger.success(f"保存训练曲线至: {plot_path}")

    def run(self):
        logger.info(f"开始训练任务 | 模型: {self.model_type.upper()} | 数据集: {self.dataset_type.upper()} | 轮数: {self.epochs} | 学习率: {self.lr:.4f} | 设备: {str(DEVICE).upper()}")

        if self.train_loader is None:
            self.load_data()
            
        best_val_acc = 0.0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        for epoch in range(self.epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.val_epoch()
            self.scheduler.step()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            is_best = False
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.save_path)
                is_best = True
                
            status_str = " | 🏆 最佳模型已保存" if is_best else ""
            logger.info(f"Epoch [{epoch+1:02d}/{self.epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%{status_str}")
        
        # 保存历史 JSON
        history_path = os.path.join(RESULTS_DIR, f'{self.model_type}_{self.dataset_type}_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=4)
            
        self.plot_history(history)
        logger.success(f"训练任务完成! 🏆 最佳验证准确率: {best_val_acc*100:.2f}% | 权重已保存至: {self.save_path}\n")
        return best_val_acc

def main():
    parser = argparse.ArgumentParser(description="Multi-model & Multi-dataset Training")
    parser.add_argument('--model', type=str, default='all', choices=['resnet18', 'mobilenet', 'custom_cnn', 'all'],
                        help="选择训练模型：resnet18, mobilenet, custom_cnn 或者是 all")
    parser.add_argument('--dataset', type=str, default='all', choices=['emnist', 'plate', 'all'],
                        help="选择数据集：emnist, plate 或者是 all")
    parser.add_argument('--epochs', type=int, default=5, help="训练轮数 (Epochs)")
    parser.add_argument('--lr', type=float, default=0.001, help="初始学习率")
    args = parser.parse_args()
    
    # 递归生成所有需要训练的模型与数据集组合
    models_to_train = ['resnet18', 'mobilenet', 'custom_cnn'] if args.model == 'all' else [args.model]
    datasets_to_train = ['emnist', 'plate'] if args.dataset == 'all' else [args.dataset]
    
    for d_type in datasets_to_train:
        for m_type in models_to_train:
            trainer = ModelTrainer(model_type=m_type, dataset_type=d_type, epochs=args.epochs, lr=args.lr)
            trainer.run()

if __name__ == '__main__':
    main()
