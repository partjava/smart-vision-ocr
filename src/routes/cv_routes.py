import os
import time
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, Response
from src.utils.helpers import STATIC_DIR, DEVICE, PLATE_CLASSES
from src.utils.ocr_engine import get_ocr_reader
from src.utils.model_loader import models
from src.core.traditional.document_scanner import scan_document
from src.core.traditional.plate_locator import locate_license_plate
from src.core.traditional.segmenter import segment_plate_characters
from src.core.traditional.comparisons import compare_edges, compare_thresholds, image_to_base64
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont

cv_bp = Blueprint('cv', __name__)

# 加载微调的 YOLOv8 检测模型用于高精度车牌定位
yolo_detector = YOLO('weights/plate_detector8/weights/best.pt')

# 统一的 PyTorch 图像预处理变换 (32x32, 3通道, [-1, 1] 归一化)
plate_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# IoU 计算辅助函数
def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

# 批量字符识别辅助函数
def _batched_plate_ocr(char_images):
    """
    高效的批量化字符识别函数（与 test 脚本一致）。
    返回: (best_text, best_conf, best_model,
            ensemble_text, ensemble_conf,
            resnet_text, resnet_conf,
            mobilenet_text, mobilenet_conf,
            custom_text, custom_conf)
    其中 best_xxx 是 ResNet/MobileNet/CustomCNN/Ensemble 四者中置信度最高的那个。
    """
    empty = ("", 0.0, "", "", 0.0, "", 0.0, "", 0.0, "", 0.0)
    if not char_images:
        return empty

    # 批量预处理
    batch_tensors = []
    for c_img in char_images:
        c_rgb = cv2.cvtColor(c_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(c_rgb)
        batch_tensors.append(plate_transform(pil_img))
    input_batch = torch.stack(batch_tensors).to(DEVICE)
    N = len(char_images)

    has_res = models['plate']['resnet18'] is not None
    has_mob = models['plate']['mobilenet'] is not None
    has_cust = models['plate']['custom_cnn'] is not None

    w_res = 0.5 if has_res else 0.0
    w_mob = 0.3 if has_mob else 0.0
    w_cust = 0.2 if has_cust else 0.0
    w_total = w_res + w_mob + w_cust
    if w_total == 0:
        return empty

    # 一次性获取所有模型的 Logits
    with torch.no_grad():
        logits_res = models['plate']['resnet18'](input_batch) if has_res else None
        logits_mob = models['plate']['mobilenet'](input_batch) if has_mob else None
        logits_cust = models['plate']['custom_cnn'](input_batch) if has_cust else None

    # 逐字符应用位置掩码，计算 softmax 概率
    prob_res_all = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE) if has_res else None
    prob_mob_all = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE) if has_mob else None
    prob_cust_all = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE) if has_cust else None

    for i in range(N):
        mask = torch.zeros(len(PLATE_CLASSES), device=DEVICE)
        if i == 0:
            mask[31:] = float('-inf')
        elif i == 1:
            mask[:41] = float('-inf')
        else:
            mask[:31] = float('-inf')

        if has_res:
            prob_res_all[i] = F.softmax(logits_res[i] + mask, dim=0)
        if has_mob:
            prob_mob_all[i] = F.softmax(logits_mob[i] + mask, dim=0)
        if has_cust:
            prob_cust_all[i] = F.softmax(logits_cust[i] + mask, dim=0)

    # 融合概率
    prob_ens = (w_res * (prob_res_all if has_res else torch.zeros(N, len(PLATE_CLASSES), device=DEVICE))
                + w_mob * (prob_mob_all if has_mob else torch.zeros(N, len(PLATE_CLASSES), device=DEVICE))
                + w_cust * (prob_cust_all if has_cust else torch.zeros(N, len(PLATE_CLASSES), device=DEVICE))) / w_total

    # 解码预测结果
    def decode(probs):
        chars, confs = [], []
        for i in range(N):
            pred_idx = torch.argmax(probs[i]).item()
            chars.append(PLATE_CLASSES[pred_idx])
            confs.append(float(probs[i][pred_idx].item()))
        return "".join(chars), float(np.mean(confs)) if confs else 0.0

    ens_text, ens_conf = decode(prob_ens)
    res_text, res_conf = decode(prob_res_all) if has_res else ("", 0.0)
    mob_text, mob_conf = decode(prob_mob_all) if has_mob else ("", 0.0)
    cust_text, cust_conf = decode(prob_cust_all) if has_cust else ("", 0.0)

    # 从三个模型 + Ensemble 中选出置信度最高的
    candidates = []
    if has_res:
        candidates.append((res_text, res_conf, "ResNet18"))
    if has_mob:
        candidates.append((mob_text, mob_conf, "MobileNetV3"))
    if has_cust:
        candidates.append((cust_text, cust_conf, "CustomCNN"))
    candidates.append((ens_text, ens_conf, "Ensemble"))

    best_text, best_conf, best_model = max(candidates, key=lambda x: x[1])

    return best_text, best_conf, best_model, ens_text, ens_conf, res_text, res_conf, mob_text, mob_conf, cust_text, cust_conf

# 定义并创建图片上传目录
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
STEPS_FOLDER = os.path.join(STATIC_DIR, 'uploads', 'steps')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STEPS_FOLDER, exist_ok=True)

# 视频与交互调参全局状态
video_path = None
video_logs = []
last_tuner_filename = None

@cv_bp.route('/api/scan_document', methods=['POST'])
def api_scan_document():
    """
    传统 CV 文档定位裁剪 + PaddleOCR 通用文字识别
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400

        # 保存原图
        filename = f"doc_{int(time.time())}.jpg"
        orig_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(orig_path)

        img = cv2.imread(orig_path)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400

        # 调用 OpenCV 处理
        from src.core.traditional.enhancer import apply_adaptive_threshold, preprocess_for_ocr
        warped_gray, _, steps = scan_document(img)
        enhanced_doc = apply_adaptive_threshold(warped_gray)

        # 保存分步图片
        step_urls = {}
        for key, step_img in steps.items():
            step_filename = f"step_doc_{key}_{filename}"
            step_path = os.path.join(STEPS_FOLDER, step_filename)
            cv2.imwrite(step_path, step_img)
            step_urls[key] = f"/static/uploads/steps/{step_filename}"

        # 保存二值化拉直图
        enhanced_filename = f"enhanced_{filename}"
        enhanced_path = os.path.join(UPLOAD_FOLDER, enhanced_filename)
        cv2.imwrite(enhanced_path, enhanced_doc)

        # 通用 OCR 识别：对灰度图做 CLAHE + 去噪 + 锐化预处理，再送入 PaddleOCR
        ocr_input = preprocess_for_ocr(warped_gray)
        reader = get_ocr_reader()
        start_time = time.time()
        ocr_results = reader.readtext(ocr_input)
        ocr_time = (time.time() - start_time) * 1000

        # 拼接识别文本
        text_lines = [res[1] for res in ocr_results]
        full_text = "\n".join(text_lines)

        return jsonify({
            'original_url': f"/static/uploads/{filename}",
            'enhanced_url': f"/static/uploads/{enhanced_filename}",
            'steps': step_urls,
            'recognized_text': full_text if full_text.strip() else "未识别到文字，请确保纸张边缘清晰且光线充足。",
            'inference_time_ms': ocr_time
        })
    except Exception as e:
        print(f"[Document Scan Error] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'文档处理失败: {str(e)}'}), 500

@cv_bp.route('/api/recognize_plate', methods=['POST'])
def api_recognize_plate():
    """
    传统 CV 车牌物理定位 + EasyOCR 识别 (兼容模式)
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    filename = f"car_{int(time.time())}.jpg"
    orig_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(orig_path)
    
    img = cv2.imread(orig_path)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 400
        
    # 定位车牌 (传统 CV 定位获取步骤图)
    warped_plate_cv, bbox_cv, steps = locate_license_plate(img)
    
    # 使用 YOLOv8 检测器获取高精度裁剪结果
    results = yolo_detector(img, imgsz=1280, conf=0.50, device='cuda', verbose=False)
    
    warped_plate = None
    bbox = None
    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        warped_plate = img[y1:y2, x1:x2]
        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        # 回退到传统定位
        warped_plate = warped_plate_cv
        bbox = bbox_cv
        
    # 保存步骤图
    step_urls = {}
    for key, step_img in steps.items():
        step_filename = f"step_plate_{key}_{filename}"
        step_path = os.path.join(STEPS_FOLDER, step_filename)
        cv2.imwrite(step_path, step_img)
        step_urls[key] = f"/static/uploads/steps/{step_filename}"
        
    if warped_plate is None:
        return jsonify({
            'original_url': f"/static/uploads/{filename}",
            'steps': step_urls,
            'plate_found': False,
            'plate_text': "未能定位到车牌区域（请确保车牌完整且角度不要过偏）"
        })
        
    plate_filename = f"crop_plate_{filename}"
    plate_path = os.path.join(UPLOAD_FOLDER, plate_filename)
    cv2.imwrite(plate_path, warped_plate)
    
    # 车牌 OCR 预处理：放大 + 灰度 + CLAHE + 二值化，让 EasyOCR 能看清字符
    plate_gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)
    # 放大到 3 倍（440x210 → 660x210），EasyOCR 检测模型需要足够大的文字
    plate_upscaled = cv2.resize(plate_gray, (plate_gray.shape[1] * 3, plate_gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    # CLAHE 对比度增强
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    plate_enhanced = clahe.apply(plate_upscaled)
    # OTSU 二值化（车牌文字对比度高，OTSU 效果好）
    _, plate_binary = cv2.threshold(plate_enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 反转：确保白底黑字（EasyOCR 在白底黑字上效果更好）
    white_ratio = np.sum(plate_binary == 255) / plate_binary.size
    if white_ratio < 0.5:
        plate_binary = cv2.bitwise_not(plate_binary)
    # 转为 BGR（EasyOCR 期望 3 通道）
    plate_ocr_input = cv2.cvtColor(plate_binary, cv2.COLOR_GRAY2BGR)

    # 静态 OCR：用预处理后的放大车牌图
    reader = get_ocr_reader()
    start_time = time.time()
    plate_results = reader.readtext(plate_ocr_input, paragraph=False, text_threshold=0.4, low_text=0.2)
    ocr_time = (time.time() - start_time) * 1000
    
    plate_text = ""
    if len(plate_results) > 0:
        # 过滤掉置信度过低的结果和明显不是车牌字符的内容
        valid_texts = []
        for res in plate_results:
            text = res[1].replace(" ", "").strip()
            conf = res[2] if len(res) > 2 else 1.0
            # 过滤：至少有字符、置信度 > 0.3、长度合理
            if text and conf > 0.3 and len(text) >= 2:
                valid_texts.append(text)
        if valid_texts:
            plate_text = "".join(valid_texts).upper()
        else:
            plate_text = "未检测到有效字符"
    else:
        plate_text = "未检测到字符"
        
    return jsonify({
        'original_url': f"/static/uploads/{filename}",
        'plate_url': f"/static/uploads/{plate_filename}",
        'steps': step_urls,
        'plate_found': True,
        'plate_text': plate_text,
        'inference_time_ms': ocr_time
    })

# --- 视频处理 API ---
@cv_bp.route('/api/upload_video', methods=['POST'])
def api_upload_video():
    """
    上传车牌通行视频
    """
    global video_path, video_logs
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    filename = f"traffic_{int(time.time())}.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(video_path)
    
    video_logs = []  # 重置日志
    return jsonify({'success': True, 'filename': filename})

def generate_video_frames():
    """
    YOLOv8 检测 + 三模型字符识别的视频流处理
    与 test_end_to_end_ocr.py 逻辑完全一致
    """
    global video_path, video_logs
    if not video_path or not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 自适应缩放（与 test 脚本一致）
    if orig_w > orig_h:
        w = 1000
        h = int(orig_h * (w / orig_w))
    else:
        h = 720
        w = int(orig_w * (h / orig_h))

    frame_idx = 0
    last_recognized_plates = {}  # track_id -> (last_frame_idx, log_index)

    # 多目标追踪器
    next_track_id = 0
    active_tracks = []
    PALETTE = [(52, 152, 219), (46, 204, 113), (231, 76, 60), (155, 89, 182),
               (241, 196, 15), (230, 126, 34), (26, 188, 156), (243, 156, 18)]

    # PIL 字体（支持中文渲染）
    try:
        pil_font = ImageFont.truetype("msyh.ttc", 20)
    except Exception:
        try:
            pil_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)
        except Exception:
            pil_font = ImageFont.load_default()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 立即缩放到自适应分辨率（与 test 脚本一致）
        frame = cv2.resize(frame, (w, h))

        # 每 2 帧检测一次（与 test 脚本一致）
        if frame_idx % 2 == 0:
            try:
                results = yolo_detector(frame, imgsz=1280, conf=0.60, device='cuda', verbose=False)

                for track in active_tracks:
                    track['frames_since_seen'] += 1

                curr_boxes = []
                if results:
                    for r in results:
                        for box in r.boxes:
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                            curr_boxes.append([bx1, by1, bx2, by2])

                # IoU 追踪关联
                matched_track = set()
                matched_det = set()
                for det_idx, det_box in enumerate(curr_boxes):
                    best_iou, best_t = 0.0, -1
                    for t_idx, track in enumerate(active_tracks):
                        if t_idx in matched_track:
                            continue
                        iou = _iou(det_box, track['box'])
                        if iou > best_iou:
                            best_iou, best_t = iou, t_idx
                    if best_iou > 0.25:
                        matched_track.add(best_t)
                        matched_det.add(det_idx)
                        active_tracks[best_t]['box'] = det_box
                        active_tracks[best_t]['frames_since_seen'] = 0
                    else:
                        color = PALETTE[next_track_id % len(PALETTE)]
                        active_tracks.append({
                            'id': next_track_id, 'box': det_box, 'color': color,
                            'text': '', 'avg_conf': 0.0, 'frames_since_seen': 0,
                            'text_lines': [], 'best_model': ''
                        })
                        next_track_id += 1
                        matched_det.add(det_idx)

                # 清除失联超过 10 帧的对象
                active_tracks = [t for t in active_tracks if t['frames_since_seen'] <= 10]

                # 对可见且置信度不高的目标做 Batch OCR（与 test 脚本一致）
                for track in active_tracks:
                    if track['frames_since_seen'] == 0:
                        # 新目标 或 之前置信度 < 85%，重新识别
                        need_ocr = (track['avg_conf'] == 0.0) or (track['avg_conf'] < 0.85)
                        if need_ocr:
                            x1, y1, x2, y2 = track['box']
                            plate_crop = frame[y1:y2, x1:x2]
                            if plate_crop.size == 0:
                                continue
                            plate_resized = cv2.resize(plate_crop, (220, 70))
                            try:
                                char_images, _ = segment_plate_characters(plate_resized)
                            except Exception:
                                continue
                            if not char_images:
                                continue

                            best_text, best_conf, best_model, ens_text, ens_conf, res_text, res_conf, mob_text, mob_conf, cust_text, cust_conf = _batched_plate_ocr(char_images)

                            # 用置信度最高的模型结果（防止闪烁）
                            if best_conf > track['avg_conf']:
                                track['text'] = best_text
                                track['avg_conf'] = best_conf
                                track['best_model'] = best_model

                            # 日志记录：用 track ID 去重，同一辆车只记录一次
                            if len(track['text']) >= 6:
                                tid = track['id']
                                current_time = time.strftime('%H:%M:%S', time.localtime())
                                if tid not in last_recognized_plates:
                                    # 新车：创建日志条目
                                    crop_fn = f"video_crop_{int(time.time())}_{tid}.jpg"
                                    cv2.imwrite(os.path.join(STEPS_FOLDER, crop_fn), plate_crop)
                                    video_logs.append({
                                        'time': current_time,
                                        'plate': track['text'],
                                        'model': track.get('best_model', ''),
                                        'image_url': f"/static/uploads/steps/{crop_fn}"
                                    })
                                    last_recognized_plates[tid] = (frame_idx, len(video_logs) - 1)
                                    if len(video_logs) > 30:
                                        video_logs.pop(0)
                                elif frame_idx - last_recognized_plates[tid][0] > 60:
                                    # 已有车辆：超过 60 帧后可以更新识别结果
                                    log_idx = last_recognized_plates[tid][1]
                                    if log_idx < len(video_logs):
                                        video_logs[log_idx]['plate'] = track['text']
                                        video_logs[log_idx]['model'] = track.get('best_model', '')
                                    last_recognized_plates[tid] = (frame_idx, log_idx)
            except Exception as e:
                print(f"[Video Error] {e}")

        # 用 PIL 绘制追踪框和中文文字（与 test 脚本一致）
        if active_tracks:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            line_height = 25

            for track in active_tracks:
                if track['frames_since_seen'] <= 1 and track.get('text'):
                    x1, y1, x2, y2 = track['box']
                    color_bgr = track['color']
                    # BGR -> RGB
                    color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

                    draw.rectangle([x1, y1, x2, y2], outline=color_rgb, width=3)

                    text = track['text']
                    try:
                        bbox = draw.textbbox((0, 0), text, font=pil_font)
                        text_w = bbox[2] - bbox[0]
                    except AttributeError:
                        text_w = len(text) * 12

                    # 智能放置文字位置
                    if y1 - 30 < 0:
                        y_offset = y2 + 5
                    else:
                        y_offset = y1 - 30

                    x_pos = max(5, min(x1, w - text_w - 5))
                    draw.text((x_pos, y_offset), text, fill=color_rgb, font=pil_font)

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 流式返回视频帧字节
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

    # 发送最后一帧黑色画面，通知浏览器流已结束，避免闪烁
    black_frame = np.zeros((h, w, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', black_frame)
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@cv_bp.route('/video_feed')
def video_feed():
    """
    流视频源映射
    """
    return Response(generate_video_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@cv_bp.route('/api/video_logs', methods=['GET'])
def get_video_logs():
    """
    拉取最新的车牌警报日志列表
    """
    global video_logs
    return jsonify(video_logs)

# --- 算子交互调参 API ---
@cv_bp.route('/api/tuner/upload', methods=['POST'])
def api_tuner_upload():
    """
    上传待调参的原图并缓存
    """
    global last_tuner_filename
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    last_tuner_filename = f"tuner_{int(time.time())}.jpg"
    file.save(os.path.join(UPLOAD_FOLDER, last_tuner_filename))
    return jsonify({'success': True, 'filename': last_tuner_filename})

@cv_bp.route('/api/tuner/preview', methods=['POST'])
def api_tuner_preview():
    """
    利用滑块参数对缓存的原图进行算子对比与 HSV 掩膜过滤，并返回 Base64 字典
    """
    global last_tuner_filename
    if not last_tuner_filename:
        return jsonify({'error': 'Please upload an image first.'}), 400
        
    img_path = os.path.join(UPLOAD_FOLDER, last_tuner_filename)
    if not os.path.exists(img_path):
        return jsonify({'error': 'Cached image not found.'}), 400
        
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 接收参数
    data = request.get_json() or {}
    
    canny_low = int(data.get('canny_low', 75))
    canny_high = int(data.get('canny_high', 200))
    sobel_ksize = int(data.get('sobel_ksize', 3))
    
    global_thresh = int(data.get('global_thresh', 127))
    adaptive_block = int(data.get('adaptive_block', 15))
    adaptive_c = int(data.get('adaptive_c', 8))
    
    h_min = int(data.get('h_min', 100))
    h_max = int(data.get('h_max', 140))
    s_min = int(data.get('s_min', 70))
    s_max = int(data.get('s_max', 255))
    v_min = int(data.get('v_min', 50))
    v_max = int(data.get('v_max', 255))
    
    # 1. 边缘算子对比
    edge_comp = compare_edges(gray, ksize=sobel_ksize, low_thresh=canny_low, high_thresh=canny_high)
    
    # 2. 二值化算子对比
    thresh_comp = compare_thresholds(gray, thresh=global_thresh, block_size=adaptive_block, C=adaptive_c)
    
    # 3. HSV 掩膜计算
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower, upper)
    
    # 将 mask 转成彩色通道再 base64 化
    mask_b64 = image_to_base64(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
    
    return jsonify({
        'edge_comparison': edge_comp,
        'thresh_comparison': thresh_comp,
        'hsv_mask': mask_b64
    })
