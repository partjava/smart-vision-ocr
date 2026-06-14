import os
import sys
import logging
from logging.handlers import RotatingFileHandler

# 在 Windows 系统上开启 ANSI 虚拟终端序列支持，以正常显示控制台日志颜色
if sys.platform == "win32":
    os.system("")

# 强制 stdout/stderr 使用 UTF-8 编码，防止 Windows 中文与制表符乱码
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 注册自定义的 SUCCESS 日志级别
SUCCESS_LEVEL_NUM = 25
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")

def success(self, message, *args, **kws):
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kws)

logging.Logger.success = success

# 确定日志目录存在
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'app.log')

class ColorConsoleFormatter(logging.Formatter):
    """
    专门针对控制台的彩色极简日志格式化器
    """
    GREY = "\033[90m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_RED = "\033[1;31m"
    RESET = "\033[0m"

    # [时间] [LEVEL] - 消息内容
    FORMAT = "%(grey)s[%(asctime)s]%(reset)s %(color)s[%(levelname)s]%(reset)s - %(message)s"

    def __init__(self, datefmt='%H:%M:%S'):
        super().__init__(datefmt=datefmt)

    def format(self, record):
        level = record.levelno
        if level == logging.DEBUG:
            color = self.GREY
        elif level == logging.INFO:
            color = self.BLUE
        elif level == SUCCESS_LEVEL_NUM:
            color = self.GREEN
        elif level == logging.WARNING:
            color = self.YELLOW
        elif level == logging.ERROR:
            color = self.RED
        elif level == logging.CRITICAL:
            color = self.BOLD_RED
        else:
            color = self.RESET

        # 动态绑定属性
        record.color = color
        record.grey = self.GREY
        record.reset = self.RESET

        formatter = logging.Formatter(self.FORMAT, datefmt=self.datefmt)
        return formatter.format(record)

def get_logger(name="SmartVision"):
    """
    配置并获取全局 logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
        
    logger.setLevel(logging.INFO)
    
    # 1. 控制台输出使用彩色极简格式
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorConsoleFormatter())
    logger.addHandler(console_handler)
    
    # 2. 循环写入文件使用带详细元数据的标准格式
    file_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s:%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger

# 全局默认 logger
logger = get_logger()


