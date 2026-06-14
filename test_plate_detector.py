"""
车牌检测模型测试脚本
加载训练好的 YOLO 模型，在测试图上检测车牌并可视化结果
"""
from ultralytics import YOLO
import cv2
import os
import random

# 加载训练好的模型
model_path = 'weights/plate_detector8/weights/best.pt'
model = YOLO(model_path)

# 从验证集随机取 5 张图测试
val_dir = 'data/yolo_plate/images/val'
val_files = [f for f in os.listdir(val_dir) if f.endswith('.jpg') or f.endswith('.png')]
test_files = random.sample(val_files, min(5, len(val_files)))

os.makedirs('test_results', exist_ok=True)

for i, fname in enumerate(test_files):
    img_path = os.path.join(val_dir, fname)
    results = model(img_path, conf=0.5)

    # 画检测框
    img = cv2.imread(img_path)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{conf:.2f}', (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    save_path = f'test_results/detect_{i}.jpg'
    cv2.imwrite(save_path, img)
    print(f'[{i+1}] {fname} -> {save_path} (detected {len(boxes)} plate(s))')

print(f'\n测试完成! 结果保存在 test_results/')
