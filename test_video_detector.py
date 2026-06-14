"""
车牌检测模型 - 视频测试脚本
读取视频，逐帧检测车牌并实时显示
用法: python test_video_detector.py <视频路径>
"""
import sys
import cv2
from ultralytics import YOLO

# 加载模型
model_path = 'weights/plate_detector8/weights/best.pt'
model = YOLO(model_path)
print(f"模型加载完成: {model_path}")

# 读取视频
video_path = sys.argv[1] if len(sys.argv) > 1 else 'example.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"无法打开视频: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"视频: {w}x{h}, {fps:.1f}fps, 共 {total} 帧")

# 初始化视频写入器
import os
os.makedirs('test_results', exist_ok=True)
output_path = 'test_results/output_example.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
print(f"处理后的视频将保存至: {output_path}")

# 创建可调整大小的窗口，防止大视频溢出屏幕
cv2.namedWindow('License Plate Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('License Plate Detection', 1280, 720)

frame_idx = 0
results = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每 2 帧检测一次 (加速)
    if frame_idx % 2 == 0:
        results = model(frame, imgsz=1920, conf=0.60, verbose=False)

    # 画检测框
    if results:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        # 加载微软雅黑字体
        try:
            font = ImageFont.truetype("msyh.ttc", 20)
        except Exception:
            font = ImageFont.load_default()

        has_boxes = False
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # 在 PIL 上画框和中文文字
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)
                draw.text((x1, y1 - 25), f"车牌 {conf:.2f}", fill=(0, 255, 0), font=font)
                has_boxes = True
        
        if has_boxes:
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 显示帧率和检测数
    num_plates = len(results[0].boxes) if results else 0
    cv2.putText(frame, f'Frame {frame_idx}/{total} | Plates: {num_plates}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 写入帧到输出视频
    out.write(frame)

    # 可视化显示 (在支持 GUI 界面下)
    try:
        cv2.imshow('License Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception:
        pass

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"处理完成: 共 {frame_idx} 帧，输出已保存至 {output_path}")
