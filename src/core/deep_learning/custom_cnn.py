import torch
import torch.nn as nn

class CustomCharCNN(nn.Module):
    """
    自定义的 9 层卷积神经网络 (CustomCharCNN)，用于字符识别。
    输入大小: (B, 3, 32, 32)
    """
    def __init__(self, num_classes):
        super(CustomCharCNN, self).__init__()
        
        # 卷积层组 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32x32 -> 16x16
        self.dropout1 = nn.Dropout2d(0.25)
        
        # 卷积层组 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16x16 -> 8x8
        self.dropout2 = nn.Dropout2d(0.25)
        
        # 全连接层
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.relu_fc1 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # 第一组卷积
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # 第二组卷积
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # 全连接
        x = self.flatten(x)
        x = self.relu_fc1(self.bn_fc1(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    model = CustomCharCNN(num_classes=47)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print("CustomCharCNN Output Shape:", out.shape)  # 期望 [2, 47]
