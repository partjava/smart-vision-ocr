import sys
import os
import cv2
import numpy as np

# 设置飞桨环境变量
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paddleocr import PaddleOCR

def main():
    ocr = PaddleOCR(enable_mkldnn=False)
    img = np.zeros((100, 300), dtype=np.uint8)
    cv2.putText(img, "TEST OCR", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 255, 3)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    print("Running raw ocr...")
    res = ocr.ocr(img)
    print("Type of res:", type(res))
    print("Content of res:")
    import pprint
    pprint.pprint(res)

if __name__ == '__main__':
    main()
