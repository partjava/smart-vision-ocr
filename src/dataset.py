import warnings
from src.core.deep_learning.dataset_emnist import (
    fix_emnist_orientation,
    to_three_channels,
    get_emnist_transforms,
    get_emnist_dataloaders as get_dataloaders
)

# 提示该脚本已弃用，推荐使用新模块
warnings.warn(
    "src/dataset.py is deprecated and will be removed in a future release. "
    "Please import from src.core.deep_learning.dataset_emnist instead.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == '__main__':
    import runpy
    # 转发主程序执行
    runpy.run_module('src.core.deep_learning.dataset_emnist', run_name='__main__')
