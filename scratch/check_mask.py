import cv2
import numpy as np
import os

def main():
    video_path = "example.mp4"
    if not os.path.exists(video_path):
        print("错误：未找到视频文件 example.mp4")
        return
        
    cap = cv2.VideoCapture(video_path)
    # 定位到第 150 帧（车牌清晰的帧）
    cap.set(cv2.CAP_PROP_POS_FRAMES, 150)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("错误：读取视频帧失败")
        return
        
    # 等比缩放到宽度 800
    h, w = frame.shape[:2]
    scale = 800.0 / w
    resized = cv2.resize(frame, (800, int(h * scale)))
    
    # 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    # 我们知道在 800 宽度的图像中，中间白车（浙B·D16T1）的车牌大致范围在：
    # x: 340 ~ 440, y: 300 ~ 360 之间（大致估算坐标）
    # 我们截取这一块区域，并打印出这块区域的 HSV 统计值，看看它落在哪
    roi_x1, roi_x2 = 340, 440
    roi_y1, roi_y2 = 300, 360
    
    # 保存这个 ROI 区域用于确认是不是车牌
    cv2.imwrite("scratch/roi_plate_debug.png", resized[roi_y1:roi_y2, roi_x1:roi_x2])
    
    plate_hsv_roi = hsv[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # 计算该区域内蓝色像素（即 H 在 100~140 之间）的占比和实际 HSV 值
    h_channel = plate_hsv_roi[:, :, 0]
    s_channel = plate_hsv_roi[:, :, 1]
    v_channel = plate_hsv_roi[:, :, 2]
    
    # 打印该区域的平均 HSV
    print(f"车牌区域平均值：H={np.mean(h_channel):.1f}, S={np.mean(s_channel):.1f}, V={np.mean(v_channel):.1f}")
    print(f"车牌区域最小值：H={np.min(h_channel)}, S={np.min(s_channel)}, V={np.min(v_channel)}")
    print(f"车牌区域最大值：H={np.max(h_channel)}, S={np.max(s_channel)}, V={np.max(v_channel)}")

if __name__ == "__main__":
    main()
