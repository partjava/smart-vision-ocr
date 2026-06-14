import os
import torch

# 获取项目根目录路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 路径配置
DATA_DIR = os.path.join(BASE_DIR, 'data')
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
STATIC_DIR = os.path.join(BASE_DIR, 'static')
RESULTS_DIR = os.path.join(STATIC_DIR, 'results')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# 确保文件夹存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# 训练配置
BATCH_SIZE = 128
EPOCHS = 10  # 4070显卡上跑10个epoch在EMNIST数据集上仅需几分钟，能快速收敛并达到极高准确率
LR = 0.001
IMAGE_SIZE = 32  # 缩放到32x32适配ResNet18和MobileNetV3的微调
NUM_WORKERS = 4

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 类别映射：EMNIST ByClass 包含 62 类 (0-9, A-Z, a-z)
# EMNIST Balanced 包含 46 类 (合并了大小写易混淆字符，如 c/C, o/O, s/S 等)
# 我们在此使用经典的 Balanced 46类 效果更好也更实用。
EMNIST_CLASSES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]
NUM_CLASSES = len(EMNIST_CLASSES)
