import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# 保证能导入 src 包
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.helpers import DEVICE, PLATE_CLASSES
from src.utils.model_loader import load_all_models, models
from src.core.traditional.segmenter import segment_plate_characters

# 1. 初始化并加载所有分类模型与检测模型
print("正在加载字符分类模型...")
load_all_models()

yolo_model_path = 'weights/plate_detector8/weights/best.pt'
print(f"正在加载 YOLOv8 检测模型: {yolo_model_path}")
detector = YOLO(yolo_model_path)

# 检查分类模型加载状态
has_resnet = models['plate']['resnet18'] is not None
has_mobilenet = models['plate']['mobilenet'] is not None
has_custom = models['plate']['custom_cnn'] is not None
print(f"分类模型状态: ResNet={has_resnet}, MobileNet={has_mobilenet}, CustomCNN={has_custom}")

if not (has_resnet or has_mobilenet or has_custom):
    print("[Error] 没有检测到任何可用的车牌字符分类权重，请确认 weights 目录下的模型。")
    sys.exit(1)

# 2. 统一的 PyTorch 图像预处理变换 (与训练时保持一致: 32x32, RGB, [-1,1]归一化)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def predict_plate_text(char_images, model_net):
    """
    使用指定分类器模型识别字符切片列表，返回完整车牌字符串与平均置信度
    """
    chars = []
    confs = []
    for i, c_img in enumerate(char_images):
        # 将 32x32 灰度图转为 3 通道 RGB 供 PIL 处理
        c_rgb = cv2.cvtColor(c_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(c_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model_net(input_tensor).squeeze(0)
            
            # 构造车牌位置约束规则掩码
            mask = torch.zeros(len(PLATE_CLASSES), device=DEVICE)
            if i == 0:
                mask[31:] = float('-inf')  # 第一位只能是省份汉字 (0-30)
            elif i == 1:
                mask[:41] = float('-inf')  # 第二位只能是大写字母 (41-64)
            else:
                mask[:31] = float('-inf')  # 第三位及以后只能是字母/数字 (31-64)
                
            prob = F.softmax(out + mask, dim=0)
            pred = torch.argmax(prob).item()
            conf = float(prob[pred].item())
            chars.append(PLATE_CLASSES[pred])
            confs.append(conf)
    
    plate_text = "".join(chars)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return plate_text, avg_conf

def predict_plate_text_ensemble(char_images, models_dict, weights=[0.5, 0.3, 0.2]):
    """
    加权集成融合预测，取三个模型的 Softmax 概率均值进行最终决策
    """
    chars = []
    confs = []
    has_res = models_dict['plate']['resnet18'] is not None
    has_mob = models_dict['plate']['mobilenet'] is not None
    has_cust = models_dict['plate']['custom_cnn'] is not None

    w_res = weights[0] if has_res else 0.0
    w_mob = weights[1] if has_mob else 0.0
    w_cust = weights[2] if has_cust else 0.0
    w_total = w_res + w_mob + w_cust

    for i, c_img in enumerate(char_images):
        c_rgb = cv2.cvtColor(c_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(c_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        prob_res = torch.zeros(len(PLATE_CLASSES)).to(DEVICE)
        prob_mob = torch.zeros(len(PLATE_CLASSES)).to(DEVICE)
        prob_cust = torch.zeros(len(PLATE_CLASSES)).to(DEVICE)

        # 构造车牌位置约束规则掩码
        mask = torch.zeros(len(PLATE_CLASSES), device=DEVICE)
        if i == 0:
            mask[31:] = float('-inf')  # 第一位只能是省份汉字 (0-30)
        elif i == 1:
            mask[:41] = float('-inf')  # 第二位只能是大写字母 (41-64)
        else:
            mask[:31] = float('-inf')  # 第三位及以后只能是字母/数字 (31-64)

        with torch.no_grad():
            if has_res:
                logits_res = models_dict['plate']['resnet18'](input_tensor).squeeze(0)
                prob_res = F.softmax(logits_res + mask, dim=0)
            if has_mob:
                logits_mob = models_dict['plate']['mobilenet'](input_tensor).squeeze(0)
                prob_mob = F.softmax(logits_mob + mask, dim=0)
            if has_cust:
                logits_cust = models_dict['plate']['custom_cnn'](input_tensor).squeeze(0)
                prob_cust = F.softmax(logits_cust + mask, dim=0)

        prob_ens = (w_res * prob_res + w_mob * prob_mob + w_cust * prob_cust) / w_total
        pred_idx = torch.argmax(prob_ens).item()
        chars.append(PLATE_CLASSES[pred_idx])
        confs.append(float(prob_ens[pred_idx].item()))

    plate_text = "".join(chars)
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return plate_text, avg_conf

def predict_plate_text_batched(char_images, models_dict, weights=[0.5, 0.3, 0.2]):
    """
    高效的批量化字符识别函数。
    1. 将所有字符图片打成一个 Batch。
    2. 只运行一次 ResNet, MobileNet, CustomCNN 的前向传播（不再重复运行）。
    3. 在 GPU 上计算各模型的 Softmax 概率，并应用位置掩码。
    4. 返回：(Ensemble文本, Ensemble置信度, ResNet文本, ResNet置信度, MobileNet文本, MobileNet置信度, Custom文本, Custom置信度)
    """
    if not char_images:
        return "", 0.0, "", 0.0, "", 0.0, "", 0.0

    # 1. 批量预处理
    batch_tensors = []
    for c_img in char_images:
        c_rgb = cv2.cvtColor(c_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(c_rgb)
        batch_tensors.append(transform(pil_img))
    
    # 堆叠成 batch: (N, 3, 32, 32)
    input_batch = torch.stack(batch_tensors).to(DEVICE)
    N = len(char_images)

    has_res = models_dict['plate']['resnet18'] is not None
    has_mob = models_dict['plate']['mobilenet'] is not None
    has_cust = models_dict['plate']['custom_cnn'] is not None

    w_res = weights[0] if has_res else 0.0
    w_mob = weights[1] if has_mob else 0.0
    w_cust = weights[2] if has_custom else 0.0
    w_total = w_res + w_mob + w_cust

    # 2. 一次性获取所有模型的 Logits
    logits_res = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE)
    logits_mob = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE)
    logits_cust = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE)

    with torch.no_grad():
        if has_res:
            logits_res = models_dict['plate']['resnet18'](input_batch)
        if has_mob:
            logits_mob = models_dict['plate']['mobilenet'](input_batch)
        if has_cust:
            logits_cust = models_dict['plate']['custom_cnn'](input_batch)

    # 3. 逐个字符位置应用位置掩码，并计算 softmax 概率
    prob_res = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE)
    prob_mob = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE)
    prob_cust = torch.zeros(N, len(PLATE_CLASSES), device=DEVICE)

    for i in range(N):
        mask = torch.zeros(len(PLATE_CLASSES), device=DEVICE)
        if i == 0:
            mask[31:] = float('-inf')  # 省份
        elif i == 1:
            mask[:41] = float('-inf')  # 城市字母
        else:
            mask[:31] = float('-inf')  # 数字/字母
        
        if has_res:
            prob_res[i] = F.softmax(logits_res[i] + mask, dim=0)
        if has_mob:
            prob_mob[i] = F.softmax(logits_mob[i] + mask, dim=0)
        if has_cust:
            prob_cust[i] = F.softmax(logits_cust[i] + mask, dim=0)

    # 4. 融合概率
    prob_ens = (w_res * prob_res + w_mob * prob_mob + w_cust * prob_cust) / w_total

    # 5. 解析预测结果
    def decode_predictions(probs):
        chars = []
        confs = []
        for i in range(N):
            pred_idx = torch.argmax(probs[i]).item()
            chars.append(PLATE_CLASSES[pred_idx])
            confs.append(float(probs[i][pred_idx].item()))
        return "".join(chars), float(np.mean(confs)) if confs else 0.0

    res_text, res_conf = decode_predictions(prob_res) if has_res else ("", 0.0)
    mob_text, mob_conf = decode_predictions(prob_mob) if has_mob else ("", 0.0)
    cust_text, cust_conf = decode_predictions(prob_cust) if has_cust else ("", 0.0)
    ens_text, ens_conf = decode_predictions(prob_ens) if (has_res or has_mob or has_cust) else ("", 0.0)

    return ens_text, ens_conf, res_text, res_conf, mob_text, mob_conf, cust_text, cust_conf

def calculate_iou(boxA, boxB):
    """
    计算两个矩形框的 Intersection over Union (IoU)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / float(unionArea)

# 3. 读取视频
video_path = sys.argv[1] if len(sys.argv) > 1 else 'example2.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"无法打开视频: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 根据横向或纵向视频，自适应设定不同的缩放限制
if orig_w > orig_h:
    # 横屏视频：限制宽度为 1000 像素，高度按比例缩小
    w = 1000
    h = int(orig_h * (w / orig_w))
else:
    # 竖屏视频：限制高度为 720 像素，宽度按比例缩小（防止过高超出屏幕）
    h = 720
    w = int(orig_w * (h / orig_h))
print(f"视频原始分辨率: {orig_w}x{orig_h}, 动态自适应调整为: {w}x{h}")
print(f"视频原始帧数: {total} 帧, 正在以自适应分辨率进行推理...")

# 4. 初始化视频写入器
os.makedirs('test_results', exist_ok=True)
output_path = 'test_results/output_ocr.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
print(f"处理后的视频将保存至: {output_path}")

# 创建自适应且可缩放（且保持宽高比）的窗口
cv2.namedWindow('Plate OCR Recognition', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow('Plate OCR Recognition', w, h)

# 5. 初始化极简多目标追踪器与调色板
next_track_id = 0
active_tracks = []
PALETTE = [
    (52, 152, 219),   # 蓝色
    (46, 204, 113),   # 绿色
    (231, 76, 60),    # 红色
    (155, 89, 182),   # 紫色
    (241, 196, 15),   # 黄色
    (230, 126, 34),   # 橙色
    (26, 188, 156),   # 青色
    (243, 156, 18)    # 橘黄
]

frame_idx = 0
results = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 立即缩放到自适应分辨率
    frame = cv2.resize(frame, (w, h))

    # 每 2 帧检测一次 (YOLOv8 强制使用 CUDA)
    if frame_idx % 2 == 0:
        results = detector(frame, imgsz=1280, conf=0.60, device='cuda', verbose=False)

        # 增加追踪对象的失活帧数
        for track in active_tracks:
            track['frames_since_seen'] += 1

        curr_boxes = []
        if results:
            for r in results:
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                    curr_boxes.append([bx1, by1, bx2, by2])

        # 对当前帧检测框进行 IoU 追踪关联
        matched_track_indices = set()
        matched_det_indices = set()

        for det_idx, det_box in enumerate(curr_boxes):
            best_iou = 0.0
            best_track_idx = -1
            for t_idx, track in enumerate(active_tracks):
                if t_idx in matched_track_indices:
                    continue
                iou = calculate_iou(det_box, track['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = t_idx

            if best_iou > 0.25:
                matched_track_indices.add(best_track_idx)
                matched_det_indices.add(det_idx)
                active_tracks[best_track_idx]['box'] = det_box
                active_tracks[best_track_idx]['frames_since_seen'] = 0
            else:
                # 分配专属颜色与ID，新建追踪对象
                color = PALETTE[next_track_id % len(PALETTE)]
                new_track = {
                    'id': next_track_id,
                    'box': det_box,
                    'color': color,
                    'text_lines': [],
                    'avg_conf': 0.0,
                    'frames_since_seen': 0
                }
                next_track_id += 1
                active_tracks.append(new_track)
                matched_det_indices.add(det_idx)

        # 清除失联超过 10 帧的对象
        active_tracks = [t for t in active_tracks if t['frames_since_seen'] <= 10]

        # 对当前可见追踪器且识别置信度不高的对象进行高速 Batched OCR
        for track in active_tracks:
            if track['frames_since_seen'] == 0:
                # 如果是新目标或者之前识别置信度小于 85%，重新运算分类器以优化结果
                if len(track['text_lines']) == 0 or track['avg_conf'] < 0.85:
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

                    # 只运行一次高速 Batch 推理，不再对每个字符独立调用分类器
                    ens_t, ens_c, res_t, res_c, mob_t, mob_c, cust_t, cust_c = predict_plate_text_batched(char_images, models)

                    # 缓存生成的展示行
                    text_lines = []
                    if has_resnet or has_mobilenet or has_custom:
                        text_lines.append(f"Ensemble: {ens_t} ({ens_c:.2%})")
                    if has_resnet:
                        text_lines.append(f"ResNet: {res_t} ({res_c:.2%})")
                    if has_mobilenet:
                        text_lines.append(f"Mobile: {mob_t} ({mob_c:.2%})")
                    if has_custom:
                        text_lines.append(f"Custom: {cust_t} ({cust_c:.2%})")

                    track['text_lines'] = text_lines
                    track['avg_conf'] = ens_c

    # 绘制追踪器中的所有可见框与文字（非检测帧依然渲染，消除频繁闪烁）
    if active_tracks:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        
        font_size = 14 if w < 600 else 20
        line_height = font_size + 5
        try:
            font = ImageFont.truetype("msyh.ttc", font_size)
            small_font = ImageFont.truetype("msyh.ttc", max(10, font_size - 4))
        except Exception:
            font = ImageFont.load_default()
            small_font = ImageFont.load_default()

        has_drawn = False
        for track in active_tracks:
            # 渲染当前帧或上一帧看过的车，避免奇偶帧切换时的画面抖动
            if track['frames_since_seen'] <= 1 and len(track['text_lines']) > 0:
                x1, y1, x2, y2 = track['box']
                color = track['color']
                text_lines = track['text_lines']

                # 绘制彩色包围框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                # 智能计算文本纵向起始高度
                text_height = line_height * len(text_lines)
                if y1 - text_height - 5 < 0:
                    y_offset = y2 + 5
                else:
                    y_offset = y1 - text_height - 5

                # 绘制多行文本标注
                for idx, line in enumerate(text_lines):
                    try:
                        bbox = draw.textbbox((0, 0), line, font=font)
                        text_w = bbox[2] - bbox[0]
                    except AttributeError:
                        text_w = font.getsize(line)[0]

                    # 动态调整横向坐标，防止文字越界
                    x_pos = max(5, min(x1, w - text_w - 5))
                    draw.text((x_pos, y_offset + idx * line_height), line, fill=color, font=font)
                
                has_drawn = True

        if has_drawn:
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 显示信息
    cv2.putText(frame, f'Frame {frame_idx}/{total}', (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # 写入帧
    out_writer.write(frame)

    # 实时展示
    try:
        cv2.imshow('Plate OCR Recognition', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # 空格键暂停/恢复
            print("\n[INFO] 视频播放已暂停。按【空格键】恢复播放，按【Q键】退出程序。")
            while True:
                # 持续刷窗口防卡死，并在暂停期间支持窗口移动/调整
                cv2.imshow('Plate OCR Recognition', frame)
                paused_key = cv2.waitKey(30) & 0xFF
                if paused_key == ord(' '):
                    print("[INFO] 视频恢复播放。")
                    break
                elif paused_key == ord('q'):
                    print("[INFO] 退出视频播放程序。")
                    cap.release()
                    out_writer.release()
                    cv2.destroyAllWindows()
                    sys.exit(0)
    except Exception:
        pass

    frame_idx += 1

cap.release()
out_writer.release()
cv2.destroyAllWindows()
print(f"\n全部处理完成! 输出已保存至: {output_path}")
