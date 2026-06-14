import os
from flask import Flask
from src.utils.helpers import STATIC_DIR
from src.utils.model_loader import load_all_models
from src.routes.page_routes import page_bp
from src.routes.cv_routes import cv_bp
from src.routes.dl_routes import dl_bp

# 初始化 Flask
app = Flask(__name__)

# 全局加载模型权重
print("[Info] 正在加载神经网络权重...")
load_all_models()
print("[Success] 神经网络模型初始化完成。")

# 注册蓝图
app.register_blueprint(page_bp)
app.register_blueprint(cv_bp)
app.register_blueprint(dl_bp)

if __name__ == '__main__':
    # 启动 Flask 服务，端口为 5000
    # 注意: PaddlePaddle 导入会触发 watchdog 重载器反复重启，所以不使用 debug 模式
    app.run(host='0.0.0.0', port=5000, debug=False)

