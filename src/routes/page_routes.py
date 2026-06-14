import os
import json
from flask import Blueprint, render_template
from src.utils.helpers import RESULTS_DIR

page_bp = Blueprint('page', __name__)

@page_bp.route('/')
def index():
    """
    系统主页
    """
    return render_template('index.html')

@page_bp.route('/canvas')
def canvas_page():
    """
    手写画板对比识别页
    """
    return render_template('canvas.html')

@page_bp.route('/tuner')
def tuner_page():
    """
    传统 CV 算子交互调参页
    """
    return render_template('tuner.html')

@page_bp.route('/dashboard')
def dashboard_page():
    """
    多模型评估指标大盘
    """
    comparison_data = {}
    comp_json_path = os.path.join(RESULTS_DIR, 'model_comparison.json')
    if os.path.exists(comp_json_path):
        try:
            with open(comp_json_path, 'r', encoding='utf-8') as f:
                comparison_data = json.load(f)
        except Exception as e:
            print(f"[Error] 读取 model_comparison.json 失败: {e}")

    # 动态计算每个数据集的最优模型
    best_models = {}
    for dataset_key in ['emnist', 'plate']:
        dataset_results = comparison_data.get(dataset_key, {})
        if not dataset_results:
            continue

        # 找延迟最低的模型
        best_latency_name = '--'
        best_latency = float('inf')
        # 找准确率最高的模型
        best_acc_name = '--'
        best_acc = 0.0
        # 找 F1 最高的模型
        best_f1_name = '--'
        best_f1 = 0.0

        model_display_names = {
            'resnet18': 'ResNet18',
            'mobilenet': 'MobileNetV3-Small',
            'custom_cnn': 'CustomCharCNN'
        }

        for model_key, metrics in dataset_results.items():
            if not isinstance(metrics, dict):
                continue
            name = model_display_names.get(model_key, model_key)

            latency = metrics.get('latency_ms', float('inf'))
            if latency < best_latency:
                best_latency = latency
                best_latency_name = name

            acc = metrics.get('accuracy', 0)
            if acc > best_acc:
                best_acc = acc
                best_acc_name = name

            f1 = metrics.get('f1_score', 0)
            if f1 > best_f1:
                best_f1 = f1
                best_f1_name = name

        best_models[dataset_key] = {
            'latency': best_latency_name,
            'accuracy': best_acc_name,
            'f1': best_f1_name
        }

    return render_template('dashboard.html', comparison=comparison_data, best_models=best_models)
