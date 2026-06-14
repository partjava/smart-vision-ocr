"""
一键训练脚本 - 在 pytorch1 环境下直接运行
用法:
  python train_all.py                   # 训练全部 (EMNIST + Plate)
  python train_all.py --dataset emnist  # 只训练 EMNIST 手写字符
  python train_all.py --dataset plate   # 只训练 Plate 车牌字符
"""
import os
import sys
import argparse

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="训练脚本")
    parser.add_argument('--dataset', type=str, default='all', choices=['emnist', 'plate', 'all'],
                        help="选择训练数据集: emnist, plate, 或 all")
    args = parser.parse_args()

    logger.info("=" * 50)
    logger.info("开始训练模型")
    logger.info("=" * 50)

    from src.core.deep_learning.trainer import ModelTrainer

    if args.dataset in ('emnist', 'all'):
        logger.info("\n--- 训练 EMNIST 手写字符模型 ---")
        for model_type in ['resnet18', 'mobilenet', 'custom_cnn']:
            logger.info(f"\n--- 训练 {model_type} on EMNIST ---")
            trainer = ModelTrainer(model_type=model_type, dataset_type='emnist', epochs=30)
            trainer.run()

    if args.dataset in ('plate', 'all'):
        logger.info("\n--- 训练 Plate 车牌字符模型 ---")
        for model_type in ['resnet18', 'mobilenet', 'custom_cnn']:
            logger.info(f"\n--- 训练 {model_type} on Plate ---")
            trainer = ModelTrainer(model_type=model_type, dataset_type='plate', epochs=15)
            trainer.run()

    # 运行评估
    logger.info("\n" + "=" * 50)
    logger.info("训练完成，开始评估...")
    logger.info("=" * 50)
    from src.core.deep_learning.evaluator import main as eval_main
    eval_main()

    logger.success("\n全部完成！重启 python app.py 即可看到效果。")

if __name__ == '__main__':
    main()
