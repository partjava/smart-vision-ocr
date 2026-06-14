import cv2
import numpy as np
import os

def main():
    video_path = "example.mp4"
    if not os.path.exists(video_path):
        print("错误：未找到视频文件 example.mp4")
        return
        
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("错误：读取视频帧失败")
        return
        
    h, w = frame.shape[:2]
    print(f"原图分辨率: {w}x{h}")
    
    # 转 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 提取颜色掩膜
    mask_blue = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([140, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([11, 70, 50]), np.array([30, 255, 255]))
    mask_green = cv2.inRange(hsv, np.array([35, 40, 50]), np.array([90, 255, 255]))
    mask = mask_blue | mask_yellow | mask_green
    
    # 因为分辨率从 800 提升到 2230 (放大 2.7875 倍)，核大小也需要等比例放大
    # 17 * 2.7875 = 47.4 -> 47
    # 5 * 2.7875 = 13.9 -> 14
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (47, 14))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, (9, 9))  # 开运算也等比例变大
    
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"共发现 {len(contours)} 个轮廓。以下是符合条件的候选轮廓：")
    
    os.makedirs("scratch", exist_ok=True)
    candidate_idx = 0
    
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        # 面积也等比例放大平方: 150 * (2.7875^2) = 1165.5, 8000 * (2.7875^2) = 62162
        if not (1000 < area < 70000):
            continue
            
        rect = cv2.minAreaRect(c)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
            
        aspect_ratio = max(width, height) / min(width, height)
        if not (2.3 < aspect_ratio < 4.5):
            continue
            
        # 过滤垂直候选区域 (width <= height)
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rw <= rh:
            print(f"过滤掉垂直轮廓: 位置=[x={rx},y={ry},w={rw},h={rh}]")
            continue
            
        candidate_idx += 1
        crop = frame[ry:ry+rh, rx:rx+rw]
        
        # 计算该区域 HSV 均值
        hsv_crop = hsv[ry:ry+rh, rx:rx+rw]
        h_mean = np.mean(hsv_crop[:, :, 0])
        s_mean = np.mean(hsv_crop[:, :, 1])
        v_mean = np.mean(hsv_crop[:, :, 2])
        
        crop_path = f"scratch/orig_cand_{candidate_idx}_ar_{aspect_ratio:.2f}_area_{area:.1f}.png"
        cv2.imwrite(crop_path, crop)
        print(f"候选 {candidate_idx}: 面积={area:.1f}, 宽高比={aspect_ratio:.2f}, 位置=[x={rx},y={ry},w={rw},h={rh}]")
        print(f"  HSV 均值: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
        print(f"  已保存原画质裁剪图到 {crop_path}")

if __name__ == "__main__":
    main()
