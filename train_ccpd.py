"""从已下载的 CCPD-Base 中提取单字符并训练"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.helpers import DATA_DIR
from src.utils.logger import logger

def main():
    ccpd_base = os.path.join(DATA_DIR, 'CCPD', 'CCPD2019', 'ccpd_base')
    output_dir = os.path.join(DATA_DIR, 'plate_chars')

    if not os.path.exists(ccpd_base):
        logger.error(f"未找到 CCPD-Base: {ccpd_base}")
        return

    # 1. 检查是否已有提取好的字符数据
    if os.path.exists(output_dir) and any(
        os.path.isdir(os.path.join(output_dir, d)) and
        any(f.endswith('.png') for f in os.listdir(os.path.join(output_dir, d)))
        for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
    ):
        # 统计已有数据量
        total_imgs = sum(
            len([f for f in os.listdir(os.path.join(output_dir, d)) if f.endswith('.png')])
            for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))
        )
        logger.info(f"检测到已有字符数据 ({total_imgs} 张)，跳过提取步骤。")
        logger.info(f"如需重新提取，删除 data/plate_chars/ 目录后重新运行。")
    else:
        # 2. 从 CCPD 提取字符
        from src.core.deep_learning.plate_dataset import process_ccpd_dataset
        process_ccpd_dataset(ccpd_base, output_dir, max_images_per_class=2000)

    # 3. 训练
    logger.info("\n开始训练...")
    from src.core.deep_learning.trainer import ModelTrainer
    for model_type in ['resnet18', 'mobilenet', 'custom_cnn']:
        logger.info(f"\n--- 训练 {model_type} on Plate (CCPD) ---")
        trainer = ModelTrainer(model_type=model_type, dataset_type='plate', epochs=15)
        trainer.run()

    # 4. 评估
    logger.info("\n开始评估...")
    from src.core.deep_learning.evaluator import main as eval_main
    eval_main()

    logger.success("\n全部完成！重启 python app.py 即可。")

if __name__ == '__main__':
    main()
