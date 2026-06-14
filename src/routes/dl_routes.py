import os
import time
import base64
from io import BytesIO
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from flask import Blueprint, request, jsonify

# 导入工具与配置
from src.utils.helpers import DEVICE, EMNIST_CLASSES, PLATE_CLASSES, STATIC_DIR
from src.utils.model_loader import models
from src.core.traditional.plate_locator import locate_license_plate
from src.core.traditional.segmenter import segment_plate_characters, preprocess_single_char
from src.core.traditional.comparisons import image_to_base64
from src.utils.gradcam import GradCAM, get_last_conv_layer, overlay_heatmap_on_image
from ultralytics import YOLO

dl_bp = Blueprint('dl', __name__)

# 加载微调的 YOLOv8 检测模型用于高精度车牌裁剪
yolo_detector = YOLO('weights/plate_detector8/weights/best.pt')

UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
STEPS_FOLDER = os.path.join(STATIC_DIR, 'uploads', 'steps')

# 统一的 PyTorch 图像预处理变换 (32x32, 3通道, [-1, 1] 归一化)
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@dl_bp.route('/api/predict_canvas', methods=['POST'])
def api_predict_canvas():
    """
    画板手写单个字符预测 (ResNet18 vs MobileNetV3 vs CustomCharCNN 三模型对比)
    """
    # 检查 EMNIST 权重是否加载
    has_resnet = models['emnist']['resnet18'] is not None
    has_mobilenet = models['emnist']['mobilenet'] is not None
    has_custom = models['emnist']['custom_cnn'] is not None
    
    if not (has_resnet or has_mobilenet or has_custom):
        return jsonify({'error': 'EMNIST 字符分类模型未训练或权重缺失，请先运行训练脚本。'}), 500
        
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Invalid request'}), 400
        
    # 解码 Base64 涂鸦数据并转换为 PIL RGB 图像
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img_gray = Image.open(BytesIO(img_bytes)).convert('L')
    # 旋转180°矫正方向（canvas输出与EMNIST训练数据方向相反）
    img_gray = img_gray.rotate(180)
    img_rgb = Image.merge("RGB", (img_gray, img_gray, img_gray))
    
    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    
    response = {}
    
    # 1. ResNet18 推理
    if has_resnet:
        resnet = models['emnist']['resnet18']
        start = time.time()
        with torch.no_grad():
            out = resnet(input_tensor)
            prob = F.softmax(out, dim=1).squeeze(0)
        t_ms = (time.time() - start) * 1000
        top5_vals, top5_idx = torch.topk(prob, 5)
        top5 = [{'char': EMNIST_CLASSES[idx.item()], 'prob': float(val.item())} for val, idx in zip(top5_vals, top5_idx)]
        response['resnet'] = {
            'prediction': top5[0]['char'],
            'confidence': float(top5[0]['prob']),
            'time_ms': t_ms,
            'top5': top5
        }
    else:
        response['resnet'] = {'prediction': '未加载权重', 'confidence': 0, 'time_ms': 0, 'top5': []}
        
    # 2. MobileNetV3 推理
    if has_mobilenet:
        mobilenet = models['emnist']['mobilenet']
        start = time.time()
        with torch.no_grad():
            out = mobilenet(input_tensor)
            prob = F.softmax(out, dim=1).squeeze(0)
        t_ms = (time.time() - start) * 1000
        top5_vals, top5_idx = torch.topk(prob, 5)
        top5 = [{'char': EMNIST_CLASSES[idx.item()], 'prob': float(val.item())} for val, idx in zip(top5_vals, top5_idx)]
        response['mobilenet'] = {
            'prediction': top5[0]['char'],
            'confidence': float(top5[0]['prob']),
            'time_ms': t_ms,
            'top5': top5
        }
    else:
        response['mobilenet'] = {'prediction': '未加载权重', 'confidence': 0, 'time_ms': 0, 'top5': []}
        
    # 3. CustomCharCNN 推理
    if has_custom:
        custom_cnn = models['emnist']['custom_cnn']
        start = time.time()
        with torch.no_grad():
            out = custom_cnn(input_tensor)
            prob = F.softmax(out, dim=1).squeeze(0)
        t_ms = (time.time() - start) * 1000
        top5_vals, top5_idx = torch.topk(prob, 5)
        top5 = [{'char': EMNIST_CLASSES[idx.item()], 'prob': float(val.item())} for val, idx in zip(top5_vals, top5_idx)]
        response['custom_cnn'] = {
            'prediction': top5[0]['char'],
            'confidence': float(top5[0]['prob']),
            'time_ms': t_ms,
            'top5': top5
        }
    else:
        response['custom_cnn'] = {'prediction': '未加载权重', 'confidence': 0, 'time_ms': 0, 'top5': []}
        
    return jsonify(response)

@dl_bp.route('/api/segment_and_predict_plate', methods=['POST'])
def api_segment_and_predict_plate():
    """
    车牌字符垂直投影切分 + CNN 分类器三模型端到端识别接口
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    filename = f"dl_plate_{int(time.time())}.jpg"
    img_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(img_path)
    
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({'error': 'Failed to read image'}), 400
        
    # 1. 传统 CV 车牌物理定位（获取步骤图与回退）
    warped_plate_cv, bbox_cv, _ = locate_license_plate(img)
    
    # 2. 使用微调训练的 YOLOv8 模型进行高精度车牌裁剪
    results = yolo_detector(img, imgsz=1280, conf=0.50, device='cuda', verbose=False)
    
    warped_plate = None
    bbox = None
    if results and len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        warped_plate = img[y1:y2, x1:x2]
        bbox = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    else:
        # 回退到传统定位结果
        warped_plate = warped_plate_cv
        bbox = bbox_cv
        
    if warped_plate is None:
        return jsonify({'plate_found': False, 'message': '未能定位到车牌区域'})
        
    # 2. 缩放到标准尺寸后进行投影法字符分割
    plate_resized = cv2.resize(warped_plate, (220, 70))
    char_images, vis_img = segment_plate_characters(plate_resized)
    vis_b64 = image_to_base64(vis_img)

    # 裁剪的车牌图 (供前端显示)
    plate_crop_b64 = image_to_base64(warped_plate)
    
    # 3. 将分割出的字符切片转换为 base64 列表供前端显示
    char_slices_b64 = []
    for c_img in char_images:
        char_slices_b64.append(image_to_base64(cv2.cvtColor(c_img, cv2.COLOR_GRAY2BGR)))
        
    # 4. 判断并运行三模型 Plate 推理
    has_resnet = models['plate']['resnet18'] is not None
    has_mobilenet = models['plate']['mobilenet'] is not None
    has_custom = models['plate']['custom_cnn'] is not None
    
    # 每个模型的逐字符预测 + 置信度（包括集成融合）
    predictions = {
        'resnet': [],
        'mobilenet': [],
        'custom_cnn': [],
        'ensemble': []
    }
    char_confidences = {
        'resnet': [],
        'mobilenet': [],
        'custom_cnn': [],
        'ensemble': []
    }

    for i, c_img in enumerate(char_images):
        # 将 32x32 灰度图转为 3 通道 RGB 供 PIL 处理
        c_rgb = cv2.cvtColor(c_img, cv2.COLOR_GRAY2RGB)
        pil_img = Image.fromarray(c_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        prob_resnet = torch.zeros(len(PLATE_CLASSES)).to(DEVICE)
        prob_mobilenet = torch.zeros(len(PLATE_CLASSES)).to(DEVICE)
        prob_custom = torch.zeros(len(PLATE_CLASSES)).to(DEVICE)

        # 构造车牌位置约束规则掩码
        mask = torch.zeros(len(PLATE_CLASSES), device=DEVICE)
        if i == 0:
            mask[31:] = float('-inf')  # 第一位只能是省份汉字 (0-30)
        elif i == 1:
            mask[:41] = float('-inf')  # 第二位只能是大写字母 (41-64)
        else:
            mask[:31] = float('-inf')  # 第三位及以后只能是字母/数字 (31-64)

        # ResNet18
        if has_resnet:
            with torch.no_grad():
                out = models['plate']['resnet18'](input_tensor).squeeze(0)
                prob_resnet = F.softmax(out + mask, dim=0)
                pred = torch.argmax(prob_resnet).item()
                predictions['resnet'].append(PLATE_CLASSES[pred])
                char_confidences['resnet'].append(float(prob_resnet[pred].item()))
        else:
            predictions['resnet'].append('?')
            char_confidences['resnet'].append(0.0)

        # MobileNet
        if has_mobilenet:
            with torch.no_grad():
                out = models['plate']['mobilenet'](input_tensor).squeeze(0)
                prob_mobilenet = F.softmax(out + mask, dim=0)
                pred = torch.argmax(prob_mobilenet).item()
                predictions['mobilenet'].append(PLATE_CLASSES[pred])
                char_confidences['mobilenet'].append(float(prob_mobilenet[pred].item()))
        else:
            predictions['mobilenet'].append('?')
            char_confidences['mobilenet'].append(0.0)

        # Custom CNN
        if has_custom:
            with torch.no_grad():
                out = models['plate']['custom_cnn'](input_tensor).squeeze(0)
                prob_custom = F.softmax(out + mask, dim=0)
                pred = torch.argmax(prob_custom).item()
                predictions['custom_cnn'].append(PLATE_CLASSES[pred])
                char_confidences['custom_cnn'].append(float(prob_custom[pred].item()))
        else:
            predictions['custom_cnn'].append('?')
            char_confidences['custom_cnn'].append(0.0)

        # 集成学习融合：加权平均概率。ResNet(0.5) + MobileNet(0.3) + Custom(0.2)
        if has_resnet or has_mobilenet or has_custom:
            w_res = 0.5 if has_resnet else 0.0
            w_mob = 0.3 if has_mobilenet else 0.0
            w_cust = 0.2 if has_custom else 0.0
            w_total = w_res + w_mob + w_cust
            
            prob_ensemble = (w_res * prob_resnet + w_mob * prob_mobilenet + w_cust * prob_custom) / w_total
            pred_ens = torch.argmax(prob_ensemble).item()
            predictions['ensemble'].append(PLATE_CLASSES[pred_ens])
            char_confidences['ensemble'].append(float(prob_ensemble[pred_ens].item()))
        else:
            predictions['ensemble'].append('?')
            char_confidences['ensemble'].append(0.0)

    # 合并为完整车牌字符串 + 平均置信度
    plate_resnet = "".join(predictions['resnet'])
    plate_mobilenet = "".join(predictions['mobilenet'])
    plate_custom = "".join(predictions['custom_cnn'])
    plate_ensemble = "".join(predictions['ensemble'])

    avg_conf_resnet = float(np.mean(char_confidences['resnet'])) if char_confidences['resnet'] else 0.0
    avg_conf_mobilenet = float(np.mean(char_confidences['mobilenet'])) if char_confidences['mobilenet'] else 0.0
    avg_conf_custom = float(np.mean(char_confidences['custom_cnn'])) if char_confidences['custom_cnn'] else 0.0
    avg_conf_ensemble = float(np.mean(char_confidences['ensemble'])) if char_confidences['ensemble'] else 0.0

    return jsonify({
        'plate_found': True,
        'plate_crop': plate_crop_b64,
        'vis_image': vis_b64,
        'char_slices': char_slices_b64,
        'predictions': {
            'resnet': {
                'text': plate_resnet,
                'confidence': avg_conf_resnet,
                'char_confidences': char_confidences['resnet'],
                'status': 'success' if has_resnet else 'weights_missing'
            },
            'mobilenet': {
                'text': plate_mobilenet,
                'confidence': avg_conf_mobilenet,
                'char_confidences': char_confidences['mobilenet'],
                'status': 'success' if has_mobilenet else 'weights_missing'
            },
            'custom_cnn': {
                'text': plate_custom,
                'confidence': avg_conf_custom,
                'char_confidences': char_confidences['custom_cnn'],
                'status': 'success' if has_custom else 'weights_missing'
            },
            'ensemble': {
                'text': plate_ensemble,
                'confidence': avg_conf_ensemble,
                'char_confidences': char_confidences['ensemble'],
                'status': 'success' if (has_resnet or has_mobilenet or has_custom) else 'weights_missing'
            }
        }
    })


@dl_bp.route('/api/gradcam', methods=['POST'])
def api_gradcam():
    """
    对画板手写输入生成三个模型的 GradCAM 可解释性热力图
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Invalid request'}), 400

    # 解码 Base64 图像
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img_gray = Image.open(BytesIO(img_bytes)).convert('L')
    img_rgb = Image.merge("RGB", (img_gray, img_gray, img_gray))

    input_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    input_tensor.requires_grad_(True)

    # 原始图像用于叠加
    orig_np = np.array(img_rgb.resize((280, 280)))

    result = {}

    for model_key, model_name in [('resnet18', 'resnet'), ('mobilenet', 'mobilenet'), ('custom_cnn', 'custom_cnn')]:
        model = models['emnist'][model_key]
        if model is None:
            result[model_name] = None
            continue

        try:
            last_conv = get_last_conv_layer(model)
            if last_conv is None:
                result[model_name] = None
                continue

            gradcam = GradCAM(model, last_conv)
            heatmap, pred_class, confidence = gradcam.generate(input_tensor)

            # 叠加热力图到原图
            overlay = overlay_heatmap_on_image(orig_np, heatmap)
            overlay_b64 = image_to_base64(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            result[model_name] = {
                'heatmap': overlay_b64,
                'predicted_class': EMNIST_CLASSES[pred_class],
                'confidence': confidence
            }
        except Exception as e:
            print(f"[GradCAM Error] {model_name}: {e}")
            result[model_name] = None

    return jsonify(result)
