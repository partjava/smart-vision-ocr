# -*- coding: utf-8 -*-
"""对比 canvas 输出与 EMNIST 训练数据，找出预处理差异"""
import sys, os, torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
sys.path.append('d:/opencv/last')
from src.utils.helpers import DEVICE, EMNIST_CLASSES
from src.utils.model_loader import load_all_models, models
from src.core.deep_learning.dataset_emnist import fix_emnist_orientation

load_all_models()

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---- 1. 加载一张真实 EMNIST 训练图像 ----
emnist_data = torchvision.datasets.EMNIST(root='d:/opencv/last/data', split='balanced', train=True, download=True)
# 找一个 '6' 的样本
for img_pil, label in emnist_data:
    if EMNIST_CLASSES[label] == '6':
        break

print(f"EMNIST 原始 6: size={img_pil.size}, mode={img_pil.mode}")
import numpy as np
arr = np.array(img_pil)
print(f"  像素范围: [{arr.min()}, {arr.max()}], 均值: {arr.mean():.1f}")
print(f"  左上角5x5:\n{arr[:5,:5]}")

# 应用 fix_emnist_orientation
img_fixed = fix_emnist_orientation(img_pil)
arr_fixed = np.array(img_fixed)
print(f"\nfix后 6: size={img_fixed.size}")
print(f"  像素范围: [{arr_fixed.min()}, {arr_fixed.max()}], 均值: {arr_fixed.mean():.1f}")
print(f"  左上角5x5:\n{arr_fixed[:5,:5]}")

# 送入模型
img_rgb_fixed = Image.merge("RGB", (img_fixed, img_fixed, img_fixed))
tensor_fixed = transform(img_rgb_fixed).unsqueeze(0).to(DEVICE)

print("\n=== EMNIST fix后 6 的预测 ===")
for name in ['resnet18', 'mobilenet', 'custom_cnn']:
    m = models['emnist'][name]
    if m is None: continue
    with torch.no_grad():
        out = m(tensor_fixed).squeeze(0)
        prob = torch.softmax(out, dim=0)
        pred_idx = torch.argmax(prob).item()
        print(f"  {name}: {EMNIST_CLASSES[pred_idx]} ({prob[pred_idx]:.0%})")

# ---- 2. 模拟 Canvas 输出 ----
print("\n" + "="*50)
canvas = Image.new('L', (280, 280), 0)
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 200)
except:
    font = ImageFont.load_default()
draw.text((50, 20), '6', fill=255, font=font)

arr_canvas = np.array(canvas)
print(f"Canvas 原始: size={canvas.size}")
print(f"  像素范围: [{arr_canvas.min()}, {arr_canvas.max()}], 均值: {arr_canvas.mean():.1f}")
print(f"  左上角5x5:\n{arr_canvas[:5,:5]}")

# 不加 rotate
img_rgb_no_rotate = Image.merge("RGB", (canvas, canvas, canvas))
tensor_no_rotate = transform(img_rgb_no_rotate).unsqueeze(0).to(DEVICE)

print("\n=== Canvas 不旋转 6 的预测 ===")
for name in ['resnet18', 'mobilenet', 'custom_cnn']:
    m = models['emnist'][name]
    if m is None: continue
    with torch.no_grad():
        out = m(tensor_no_rotate).squeeze(0)
        prob = torch.softmax(out, dim=0)
        pred_idx = torch.argmax(prob).item()
        print(f"  {name}: {EMNIST_CLASSES[pred_idx]} ({prob[pred_idx]:.0%})")

# 加 rotate(180)
canvas_rot = canvas.rotate(180)
arr_rot = np.array(canvas_rot)
print(f"\nCanvas rotate(180)后:")
print(f"  像素范围: [{arr_rot.min()}, {arr_rot.max()}], 均值: {arr_rot.mean():.1f}")
print(f"  左上角5x5:\n{arr_rot[:5,:5]}")

img_rgb_rotate = Image.merge("RGB", (canvas_rot, canvas_rot, canvas_rot))
tensor_rotate = transform(img_rgb_rotate).unsqueeze(0).to(DEVICE)

print("\n=== Canvas rotate(180)后 6 的预测 ===")
for name in ['resnet18', 'mobilenet', 'custom_cnn']:
    m = models['emnist'][name]
    if m is None: continue
    with torch.no_grad():
        out = m(tensor_rotate).squeeze(0)
        prob = torch.softmax(out, dim=0)
        pred_idx = torch.argmax(prob).item()
        print(f"  {name}: {EMNIST_CLASSES[pred_idx]} ({prob[pred_idx]:.0%})")

import torchvision
