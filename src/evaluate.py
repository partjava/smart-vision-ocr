import warnings
import runpy

# 提示已弃用，建议运行新模块
warnings.warn(
    "src/evaluate.py is deprecated. Please run python -m src.core.deep_learning.evaluator instead.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == '__main__':
    # 转发命令行模块运行
    runpy.run_module('src.core.deep_learning.evaluator', run_name='__main__')
