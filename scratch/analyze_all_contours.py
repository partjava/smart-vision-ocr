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
    scale = 800.0 / w
    resized = cv2.resize(frame, (800, int(h * scale)))
    
    # 转 HSV
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # 提取颜色掩膜
    mask_blue = cv2.inRange(hsv, np.array([100, 70, 50]), np.array([140, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([11, 70, 50]), np.array([30, 255, 255]))
    mask_green = cv2.inRange(hsv, np.array([35, 40, 50]), np.array([90, 255, 255]))
    mask = mask_blue | mask_yellow | mask_green
    
    # 形态学处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, (3, 3))
    
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"共发现 {len(contours)} 个轮廓。以下是符合宽高比和面积条件的候选轮廓：")
    
    os.makedirs("scratch", exist_ok=True)
    candidate_idx = 0
    
    for idx, c in enumerate(contours):
        area = cv2.contourArea(c)
        if not (150 < area < 8000):
            continue
            
        rect = cv2.minAreaRect(c)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
            
        aspect_ratio = max(width, height) / min(width, height)
        if not (2.3 < aspect_ratio < 4.5):
            continue
            
        candidate_idx += 1
        # 获取最小外接矩形的四个点，截取并保存
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 裁剪出该候选区域并保存
        # 简单使用 bounding rect 裁剪
        rx, ry, rw, rh = cv2.boundingRect(c)
        crop = resized[ry:ry+rh, rx:rx+rw]
        
        # 计算该区域 HSV 均值
        hsv_crop = hsv[ry:ry+rh, rx:rx+rw]
        h_mean = np.mean(hsv_crop[:, :, 0])
        s_mean = np.mean(hsv_crop[:, :, 1])
        v_mean = np.mean(hsv_crop[:, :, 2])
        
        crop_path = f"scratch/cand_{candidate_idx}_ar_{aspect_ratio:.2f}_area_{area:.1f}.png"
        cv2.imwrite(crop_path, crop)
        print(f"候选 {candidate_idx}: 面积={area:.1f}, 宽高比={aspect_ratio:.2f}, 位置=[x={rx},y={ry},w={rw},h={rh}]")
        print(f"  HSV 均值: H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
        print(f"  已保存裁剪图到 {crop_path}")

if __name__ == "__main__":
    main()
