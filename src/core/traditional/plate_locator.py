import cv2
import numpy as np
from src.core.traditional.base_processor import resize_image_width, get_perspective_transform

def locate_license_plate(img, lower_blue_val=100, upper_blue_val=140, morph_w=17, morph_h=5, return_multiple=False):
    """
    使用 OpenCV 传统算法定位车牌（HSV颜色空间过滤 + 形态学连通 + 轮廓筛选）
    支持自适应分辨率（直接在高清原图上处理，避免低画质马赛克影响）
    参数:
        img: 输入图像 (BGR，支持原图分辨率如 2230x1206)
        lower_blue_val: 蓝色色调下界 (0~180)
        upper_blue_val: 蓝色色调上界 (0~180)
        morph_w: 闭运算结构核基准宽度 (对应800宽度下的尺寸)
        morph_h: 闭运算结构核基准高度 (对应800宽度下的尺寸)
        return_multiple: 是否返回多个候选车牌 (若为True返回前3个候选，为False返回最匹配的单个候选)
    返回:
        若 return_multiple=False:
            warped_plate: 拉直后的单个车牌图像 (BGR) 或 None
            bbox: 车牌在原图中的4个角点坐标 (shape: [4, 2]) 或 None
            steps: 算法分步产生的图像字典 (已缩放至800宽用于前端显示)
        若 return_multiple=True:
            candidates: 列表，每个元素为 (warped_plate, bbox)
            steps: 算法分步产生的图像字典
    """
    h, w = img.shape[:2]
    # 计算分辨率缩放因子，以基准 800 宽为基准进行动态自适应缩放
    scale_factor = w / 800.0
    scale = 800.0 / w

    # 准备前端展示的缩放图
    resized = cv2.resize(img, (800, int(h * scale)))
    orig_resized = resized.copy()

    # 1. 转换到 HSV 空间 (在原分辨率上进行处理以获取超清车牌)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. 车牌颜色掩膜提取 (蓝牌、黄牌、绿牌)
    lower_blue = np.array([lower_blue_val, 70, 50])
    upper_blue = np.array([upper_blue_val, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_yellow = np.array([11, 70, 50])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_green = np.array([35, 40, 50])
    upper_green = np.array([90, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # 合并掩膜
    mask = mask_blue | mask_yellow | mask_green

    # 3. 自适应形态学闭运算：等比例放大结构元素以适配高分辨率
    morph_w_scaled = max(1, int(round(morph_w * scale_factor)))
    morph_h_scaled = max(1, int(round(morph_h * scale_factor)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_w_scaled, morph_h_scaled))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 开运算去除细小噪点
    open_size = max(1, int(round(3 * scale_factor)))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, (open_size, open_size))

    # 4. 轮廓提取与筛选
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 动态调整面积筛选区间 (以 scale_factor 的平方放大面积)
    min_area = 150 * (scale_factor ** 2)
    max_area = 8000 * (scale_factor ** 2)

    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if not (min_area < area < max_area):
            continue

        # 获取最小外接矩形以计算长宽比
        rect = cv2.minAreaRect(c)
        width, height = rect[1]

        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)

        # 宽高比筛选 (车牌通常在 2.3 至 4.5 之间)
        if 2.3 < aspect_ratio < 4.5:
            # 过滤垂直候选（车道虚线、直行路标通常高大于宽）
            rx, ry, rw, rh = cv2.boundingRect(c)
            if rw <= rh:
                continue

            # 计算长宽比匹配度打分 (越接近 3.2 越好)
            ratio_diff = abs(aspect_ratio - 3.2)
            score = 1.0 / (1.0 + ratio_diff)

            # 加上微小的面积权重，使得多目标分值接近时大车牌优先
            score += (area / max_area) * 0.1

            candidates.append((score, c))

    # 按打分降序排列
    candidates.sort(key=lambda x: x[0], reverse=True)

    # 5. 执行透视变换提取并拉直车牌
    rect_drawn = orig_resized.copy()
    results = []

    # 提取前 3 个候选车牌 (若 return_multiple 为 False 则只提第 1 个)
    max_candidates = 3 if return_multiple else 1

    for i, (score, c) in enumerate(candidates[:max_candidates]):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # 直接使用高清原图和高清坐标进行透视变换，确保车牌画质清晰
        warped_plate = get_perspective_transform(img, box, 440, 140)
        warped_plate = cv2.resize(warped_plate, (220, 70))

        results.append((warped_plate, box))

        # 绘制检测框到缩放后的展示图上 (绿色为主目标，黄色为副候选目标)
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        # 将原图上的角点 box 缩放到展示图坐标系上以进行绘制
        box_resized = (box * scale).astype(int)
        cv2.drawContours(rect_drawn, [box_resized], -1, color, 2)

    # 汇总中间步骤的图像（转为 BGR 并缩放到 800 宽以便网页高效展示）
    resized_mask = cv2.resize(mask, (800, int(h * scale)))
    resized_closed = cv2.resize(opened, (800, int(h * scale)))

    steps = {
        "resized": orig_resized,
        "mask": cv2.cvtColor(resized_mask, cv2.COLOR_GRAY2BGR),
        "closed": cv2.cvtColor(resized_closed, cv2.COLOR_GRAY2BGR),
        "detected": rect_drawn
    }

    if return_multiple:
        return results, steps
    else:
        if len(results) > 0:
            return results[0][0], results[0][1], steps
        else:
            return None, None, steps
