import cv2
import numpy as np

def get_horizontal_projection(binary_img):
    """
    计算水平方向上的白色像素累加（行累加图）
    """
    return np.sum(binary_img > 0, axis=1)

def get_vertical_projection(binary_img):
    """
    计算垂直方向上的白色像素累加（列累加图）
    """
    return np.sum(binary_img > 0, axis=0)

def segment_lines_by_projection(binary_img, threshold=2, min_height=8):
    """
    利用水平投影将文档二值图分割成单行文字的 y 轴区间
    返回: [(y_start, y_end), ...]
    """
    proj = get_horizontal_projection(binary_img)
    lines = []
    in_line = False
    start_y = 0
    
    for i, val in enumerate(proj):
        if val > threshold and not in_line:
            start_y = i
            in_line = True
        elif val <= threshold and in_line:
            if i - start_y >= min_height:
                lines.append((start_y, i))
            in_line = False
            
    if in_line:
        lines.append((start_y, len(proj)))
        
    return lines

def segment_chars_by_projection(binary_img, threshold=1, min_width=3):
    """
    利用垂直投影将一维单行文字切分为单个字符的 x 轴区间
    返回: [(x_start, x_end), ...]
    """
    proj = get_vertical_projection(binary_img)
    chars = []
    in_char = False
    start_x = 0
    
    for i, val in enumerate(proj):
        if val > threshold and not in_char:
            start_x = i
            in_char = True
        elif val <= threshold and in_char:
            if i - start_x >= min_width:
                chars.append((start_x, i))
            in_char = False
            
    if in_char:
        chars.append((start_x, len(proj)))
        
    return chars

def segment_plate_characters(warped_plate):
    """
    将 220x70 尺寸的车牌图像切分为 7 个单独的字符图像
    返回:
        char_images: 包含 7 张单独字符图像的列表 (均为 32x32 尺寸，黑底白字，供 CNN 直接识别)
        seg_visualization: 绘制了切分边界的车牌彩色可视化图 (用于前端展示)
    """
    gray = cv2.cvtColor(warped_plate, cv2.COLOR_BGR2GRAY)

    # --- 自适应二值化 (比 OTSU 更能应对光照不均) ---
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 25, 10
    )

    # 根据白色像素比例决定是否反转 (确保字符=白, 背景=黑)
    white_ratio = np.sum(binary == 255) / binary.size
    if white_ratio > 0.5:
        binary = cv2.bitwise_not(binary)

    # --- 形态学去噪 ---
    # 开运算去除小于字符笔画的噪点
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    # 闭运算连接断裂笔画
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # --- 去边框: 裁掉上下各 6px、左右各 3px 的边框噪声 ---
    h_b, w_b = binary.shape
    margin_t, margin_b = 6, 6
    margin_l, margin_r = 3, 3
    binary_cropped = binary[margin_t:h_b - margin_b, margin_l:w_b - margin_r]

    # --- 垂直投影分割 ---
    raw_ranges = segment_chars_by_projection(binary_cropped, threshold=3, min_width=6)

    # --- 合并过窄的相邻段 (车牌圆点、噪点导致的误切) ---
    merged_ranges = []
    for x_start, x_end in raw_ranges:
        if merged_ranges and (x_start - merged_ranges[-1][1]) < 4:
            # 间距太小，合并为一段
            merged_ranges[-1] = (merged_ranges[-1][0], x_end)
        else:
            merged_ranges.append((x_start, x_end))

    # --- 启发式筛选: 车牌 7 字符 (新能源 8 个) ---
    char_images = []
    vis_img = warped_plate.copy()

    if 6 <= len(merged_ranges) <= 9:
        for idx, (x_start, x_end) in enumerate(merged_ranges):
            # 映射回裁剪前的原坐标
            xs = x_start + margin_l
            xe = x_end + margin_l
            # 确保不越界
            xs = max(0, xs)
            xe = min(w_b, xe)

            char_roi = binary[margin_t:h_b - margin_b, xs:xe]
            char_img = preprocess_single_char(char_roi)
            char_images.append(char_img)

            cv2.rectangle(vis_img, (xs, 0), (xe, 70), (0, 0, 255), 1)
    else:
        # 均分兜底: 7 字符
        step_w = float(w_b) / 7.0
        for i in range(7):
            xs = int(i * step_w)
            xe = int((i + 1) * step_w)
            char_roi = binary[margin_t:h_b - margin_b, xs:xe]
            char_img = preprocess_single_char(char_roi)
            char_images.append(char_img)
            cv2.rectangle(vis_img, (xs, 0), (xe, 70), (0, 0, 255), 1)

    return char_images, vis_img

def preprocess_single_char(roi, target_size=32):
    """
    对切割出的单字符 ROI 进行图像归一化处理：
    将字符等比缩放并居中放置在 32x32 的黑色背景画板上 (前景色为白色)，适配 CNN 的输入
    """
    h_roi, w_roi = roi.shape
    if h_roi == 0 or w_roi == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)
        
    # 等比缩放，使得最大边长为 target_size - 8 (留出边缘 padding)
    max_side = target_size - 8
    scale = float(max_side) / max(h_roi, w_roi)
    new_w = max(1, int(w_roi * scale))
    new_h = max(1, int(h_roi * scale))
    
    resized_char = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 放入 32x32 黑色画布中心
    canvas = np.zeros((target_size, target_size), dtype=np.uint8)
    dx = (target_size - new_w) // 2
    dy = (target_size - new_h) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized_char
    
    return canvas
