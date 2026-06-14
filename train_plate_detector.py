"""
车牌检测模型训练脚本
基于 YOLOv8n 在 CCPD 数据集上微调，实现车辆照片中车牌区域的定位
用法: python train_plate_detector.py
"""
from ultralytics import YOLO
import os

def main():
    # 加载预训练 YOLOv8n 模型 (nano 版本, 3M 参数, 适合轻量部署)
    model = YOLO('yolov8n.pt')

    # 训练配置
    results = model.train(
        data='data/yolo_plate/data.yaml',   # 14万张子集 (全量太慢)
        epochs=20,                           # 训练轮数
        imgsz=640,                           # 输入图像尺寸
        batch=16,                             # batch size (显存不够可调小)
        workers=2,                           # 数据加载线程数 (2个进程兼顾速度与内存，若仍报错可改为0)
        cache=False,                         # 不缓存到内存 (避免爆内存)
        name='plate_detector',               # 实验名称
        project='weights',                   # 输出目录
        patience=10,                         # 早停: 10 轮无提升则停止
        verbose=True,
    )

    print(f"\n训练完成!")
    print(f"最佳权重: weights/plate_detector/weights/best.pt")
    print(f"mAP@50:   {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP@50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == '__main__':
    main()
