import os
import torch

# 禁用 PaddlePaddle 3.x 包含 Bug 的 PIR 编译器与 oneDNN，以避免 Windows CPU 运行环境下报错
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

_reader = None

class PaddleOCRWrapper:
    def __init__(self, use_gpu):
        import logging
        # 通过 python standard logging 禁用 ppocr 多余的 info 日志
        logging.getLogger("ppocr").setLevel(logging.WARNING)
        from paddleocr import PaddleOCR
        # 使用默认参数启动，自适应 GPU 运算并支持中英文文本识别，显式禁用 mkldnn 以避免 Windows CPU 下 PIR 编译器兼容性 Bug
        self.ocr = PaddleOCR(enable_mkldnn=False)
        
    def readtext(self, img, **kwargs):
        import numpy as np
        import cv2

        # 检查是否为 numpy 数组并对灰度图/单通道图进行 3 通道转换以兼容 PaddleOCR
        if isinstance(img, np.ndarray):
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 运行 PaddleOCR 检测与识别
        result = self.ocr.ocr(img)
        
        # 兼容 EasyOCR 返回格式: [(bbox, text, confidence), ...]
        ocr_results = []
        if result:
            if isinstance(result[0], list):
                # PaddleOCR 2.x 传统列表返回格式
                for line in result[0]:
                    if len(line) >= 2 and len(line[1]) >= 2:
                        bbox = line[0]
                        text = line[1][0]
                        conf = float(line[1][1])
                        ocr_results.append((bbox, text, conf))
            elif isinstance(result[0], dict):
                # PaddleOCR 3.x (PaddleX pipeline) 字典返回格式
                res_dict = result[0]
                rec_polys = res_dict.get('rec_polys') or res_dict.get('dt_polys') or []
                rec_texts = res_dict.get('rec_texts', [])
                rec_scores = res_dict.get('rec_scores', [])
                for i in range(min(len(rec_polys), len(rec_texts), len(rec_scores))):
                    bbox = rec_polys[i]
                    if hasattr(bbox, 'tolist'):
                        bbox = bbox.tolist()
                    text = rec_texts[i]
                    conf = float(rec_scores[i])
                    ocr_results.append((bbox, text, conf))
        return ocr_results

def get_ocr_reader():
    """
    延迟加载 PaddleOCR 单例，避免在导入阶段占用过多显存
    """
    global _reader
    if _reader is None:
        print("[Info] 正在初始化通用 OCR 识别引擎 (PaddleOCR)...")
        gpu_available = torch.cuda.is_available()
        _reader = PaddleOCRWrapper(use_gpu=gpu_available)
        print(f"[Success] PaddleOCR 引擎初始化成功！(GPU = {gpu_available})")
    return _reader
