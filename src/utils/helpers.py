import os
import yaml
import torch

# 获取项目根目录 (helpers.py 位于 src/utils/，根目录是它的上上级)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config():
    """
    加载 yaml 配置文件，并转换相对路径为绝对路径
    """
    config_path = os.path.join(BASE_DIR, 'config', 'settings.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 将相对路径转换为绝对路径，并确保其目录存在
    resolved_paths = {}
    for key, rel_path in config['paths'].items():
        abs_path = os.path.join(BASE_DIR, rel_path)
        os.makedirs(abs_path, exist_ok=True)
        resolved_paths[key.upper()] = abs_path
        
    config['resolved_paths'] = resolved_paths
    return config

# 全局导出常用变量
CONFIG = load_config()

# 导出具体路径
DATA_DIR = CONFIG['resolved_paths']['DATA_DIR']
WEIGHTS_DIR = CONFIG['resolved_paths']['WEIGHTS_DIR']
STATIC_DIR = CONFIG['resolved_paths']['STATIC_DIR']
RESULTS_DIR = CONFIG['resolved_paths']['RESULTS_DIR']
TEMPLATES_DIR = CONFIG['resolved_paths']['TEMPLATES_DIR']

# 导出训练参数
BATCH_SIZE = CONFIG['training']['batch_size']
EPOCHS = CONFIG['training']['epochs']
LR = CONFIG['training']['learning_rate']
IMAGE_SIZE = CONFIG['training']['image_size']
NUM_WORKERS = CONFIG['training']['num_workers']

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 导出字符集
EMNIST_CLASSES = CONFIG['char_sets']['emnist_classes']
NUM_CLASSES = len(EMNIST_CLASSES)

PLATE_PROVINCES = CONFIG['char_sets']['plate_provinces']
PLATE_CHARS = CONFIG['char_sets']['plate_chars']
PLATE_CLASSES = PLATE_PROVINCES + PLATE_CHARS
NUM_PLATE_CLASSES = len(PLATE_CLASSES)
