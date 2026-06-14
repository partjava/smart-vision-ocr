import cv2
from ultralytics import YOLO

# 加载模型
model_path = 'weights/plate_detector8/weights/best.pt'
model = YOLO(model_path)

# 读取视频的第一帧
cap = cv2.VideoCapture('example.mp4')
ret, frame = cap.read()
cap.release()

if not ret:
    print("无法读取视频帧")
    exit(1)

print(f"原始帧分辨率: {frame.shape[1]}x{frame.shape[0]}")

# 测试不同的推理尺寸 (imgsz) 和置信度阈值 (conf)
configs = [
    {"imgsz": 640, "conf": 0.25},
    {"imgsz": 1280, "conf": 0.25},
    {"imgsz": 1280, "conf": 0.15},
    {"imgsz": 1920, "conf": 0.15},
    {"imgsz": 2240, "conf": 0.10}
]

for cfg in configs:
    imgsz = cfg["imgsz"]
    conf = cfg["conf"]
    results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
    
    boxes = results[0].boxes
    print(f"测试 imgsz={imgsz}, conf={conf} -> 检测到 {len(boxes)} 个车牌")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf[0])
        print(f"  - 车牌 {i+1}: 坐标 [{x1}, {y1}, {x2}, {y2}], 置信度: {score:.4f}, 宽度: {x2-x1}px, 高度: {y2-y1}px")
