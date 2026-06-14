import sys
import os
import cv2
import numpy as np

# 设置飞桨环境变量，禁用带有 Bug 的 PIR 编译器 API
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.ocr_engine import get_ocr_reader

def main():
    print("正在测试 PaddleOCR 引擎 (已禁用 PIR)...")
    try:
        reader = get_ocr_reader()
        
        # 创建一个 300x100 的单通道灰度测试图像（黑底白字画线）
        img = np.zeros((100, 300), dtype=np.uint8)
        cv2.putText(img, "TEST OCR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
        
        print(f"输入测试图像形状: {img.shape}, 类型: {img.dtype}")
        print("运行 readtext...")
        res = reader.readtext(img)
        print("识别成功! 结果如下:")
        print(res)
    except Exception as e:
        print("识别失败! 报错日志如下:")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
