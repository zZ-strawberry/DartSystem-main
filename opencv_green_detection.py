import cv2
import numpy as np
from collections import deque
#识别
# 全局变量声明
GUI_AVAILABLE = False  # 默认为False，将在create_trackbars中尝试创建窗口时更新
DEFAULT_HSV_VALUES = {
    'H min': 24, 'H max': 83, 
    'S min': 31, 'S max': 253,
    'V min': 123, 'V max': 255
}

class KalmanFilter2D:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)  # 4个状态变量(x,y,dx,dy)，2个测量变量(x,y)
        self.kalman.measurementMatrix = np.array([[1., 0., 0., 0.],
                                                [0., 1., 0., 0.]], np.float32)
        self.kalman.transitionMatrix = np.array([[1., 0., 1., 0.],
                                               [0., 1., 0., 1.],
                                               [0., 0., 1., 0.],
                                               [0., 0., 0., 1.]], np.float32)
        self.kalman.processNoiseCov = np.array([[1e-3, 0., 0., 0.],
                                              [0., 1e-3, 0., 0.],
                                              [0., 0., 1e-3, 0.],
                                              [0., 0., 0., 1e-3]], np.float32)
        self.kalman.measurementNoiseCov = np.array([[1e-2, 0.],
                                                  [0., 1e-2]], np.float32)
        self.initialized = False

    def update(self, measurement):
        if not self.initialized:
            if measurement is None:
                return (0, 0)  # 如果未初始化且没有测量值，返回(0,0)
            self.kalman.statePre = np.array([[measurement[0]], [measurement[1]], [0.], [0.]], np.float32)
            self.initialized = True
            return measurement

        prediction = self.kalman.predict()
        if measurement is not None:
            correction = self.kalman.correct(np.array([[measurement[0]], [measurement[1]]], np.float32))
            return (correction[0, 0], correction[1, 0])
        return (prediction[0, 0], prediction[1, 0])

class GreenLightDetector:
    def __init__(self, history_size=3):
        self.offset_history = deque(maxlen=history_size)
        self.kalman_filter = KalmanFilter2D()
        self.last_valid_offset = (0, 0)  # 修改为(x,y)元组
        self.lost_count = 0
        self.MAX_LOST_FRAMES = 5
        self.weight_matrix = None
        self.recent_detections = deque(maxlen=10)
        self.brightness_threshold = 200
        self.last_valid_position = None
        self.hsv_range_samples = []
        self.max_samples = 100
        self.debug_mask = None
        self.debug_result = None

    def create_weight_matrix(self, frame):
        height, width = frame.shape[:2]
        weight_matrix = np.ones((height, width))
        
        # 更加强调上半部分区域
        # 将图像分成四个区域：顶部(0-25%)、上中部(25-50%)、下中部(50-75%)、底部(75-100%)
        top_end = int(height * 0.25)          # 顶部区域结束点
        upper_mid_end = int(height * 0.5)     # 上中部结束点
        lower_mid_end = int(height * 0.75)    # 下中部结束点
        
        # 大幅提高上中部权重，适度提高顶部权重，降低下部权重
        weight_matrix[:top_end, :] *= 1.5              # 顶部权重提高到1.5
        weight_matrix[top_end:upper_mid_end, :] *= 2.0 # 上中部权重提高到2.0（最高优先级）
        weight_matrix[upper_mid_end:lower_mid_end, :] *= 0.5 # 下中部降低到0.5
        weight_matrix[lower_mid_end:, :] *= 0.2        # 底部权重最低0.2
        
        return weight_matrix

    def analyze_hsv_sample(self, hsv_region):
        """分析HSV区域样本，自动优化HSV阈值"""
        if hsv_region is not None and hsv_region.size > 0:
            # 仅保留有限数量的样本，先进先出
            self.hsv_range_samples.append(hsv_region)
            if len(self.hsv_range_samples) > self.max_samples:
                self.hsv_range_samples.pop(0)
            
            # 计算所有样本的统计数据
            if len(self.hsv_range_samples) > 5:  # 至少需要5个样本才开始分析
                all_samples = np.vstack(self.hsv_range_samples)
                
                # 计算H、S、V通道的均值和标准差
                h_mean = np.mean(all_samples[:, 0])
                h_std = np.std(all_samples[:, 0])
                s_mean = np.mean(all_samples[:, 1])
                s_std = np.std(all_samples[:, 1])
                v_mean = np.mean(all_samples[:, 2])
                v_std = np.std(all_samples[:, 2])
                
                # 使用2个标准差覆盖约95%的样本
                h_min = max(0, h_mean - 2 * h_std)
                h_max = min(179, h_mean + 2 * h_std)
                s_min = max(0, s_mean - 2 * s_std)
                s_max = min(255, s_mean + 2 * s_std)
                v_min = max(0, v_mean - 2 * v_std)
                v_max = min(255, v_mean + 2 * v_std)
                
                # 返回优化的HSV范围
                return (
                    np.array([int(h_min), int(s_min), int(v_min)]),
                    np.array([int(h_max), int(s_max), int(v_max)])
                )
        return None, None

    def is_valid_green_target(self, contour, hsv, original_frame):
        """增强的目标有效性验证"""
        # 基础几何特征检查
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 1. 计算圆形度 (4π*面积/周长²)，完美圆形为1
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 2. 拟合椭圆并检查长宽比
        if len(contour) >= 5:  # 至少需要5个点才能拟合椭圆
            ellipse = cv2.fitEllipse(contour)
            (_, _), (width, height), _ = ellipse
            aspect_ratio = min(width, height) / max(width, height) if max(width, height) > 0 else 0
        else:
            aspect_ratio = 1.0  # 点太少，假设是圆形
        
        # 3. 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 检查位置 - 更倾向于上半部分的目标
        img_height = original_frame.shape[0]
        position_score = 1.0
        if y < img_height * 0.5:  # 在上半部分
            position_score = 2.0   # 位置评分加倍
        
        # 4. 在原始帧中提取ROI
        roi = original_frame[y:y+h, x:x+w]
        
        # 5. 分析ROI的亮度特征
        if roi.size > 0:  # 确保ROI不为空
            # 转为灰度
            if len(roi.shape) == 3:
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            else:
                roi_gray = roi
                
            # 计算亮度统计
            mean_brightness = np.mean(roi_gray)
            max_brightness = np.max(roi_gray)
            brightness_std = np.std(roi_gray)
            
            # 6. 创建目标区域的HSV掩码
            mask = np.zeros_like(original_frame[:,:,0])
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # 7. 获取掩码内的HSV值
            hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)
            non_zero = hsv_masked[mask > 0]
            
            if non_zero.size > 0:
                # 保存有效的HSV样本用于自适应
                self.analyze_hsv_sample(non_zero)
                
                # 计算色相的集中度
                h_values = non_zero[:, 0]
                h_std = np.std(h_values)
                
                # 亮度与饱和度比例
                v_values = non_zero[:, 2]
                s_values = non_zero[:, 1]
                sv_ratio = np.mean(s_values) / np.mean(v_values) if np.mean(v_values) > 0 else 0
                
                # 亮度稳定性，真实LED灯更均匀
                brightness_uniformity = 1 - (brightness_std / (mean_brightness + 1e-5))
                
                # 计算绿色指标 - 真实绿灯在HSV中H值应该在绿色范围(约45-75)
                h_mean = np.mean(h_values)
                green_score = 0
                if 30 <= h_mean <= 90:  # 扩大绿色范围
                    # 45-75是最佳绿色范围
                    if 40 <= h_mean <= 80:
                        green_score = 1.0
                    else:
                        green_score = 0.7  # 30-40或80-90是次优绿色
                
                # 放宽判断标准
                is_valid = (
                    circularity > 0.4 and              # 进一步降低圆形度要求
                    aspect_ratio > 0.4 and             # 进一步降低长宽比要求
                    mean_brightness > 70 and           # 进一步降低亮度要求
                    brightness_std < 80 and            # 进一步放宽亮度均匀性要求
                    h_std < 40 and                     # 进一步放宽色相一致性要求
                    sv_ratio > 0.2 and sv_ratio < 4.0 and  # 进一步放宽饱和度与亮度比例要求
                    green_score > 0.3                  # 进一步降低绿色评分要求
                )
                
                # 计算综合得分 - 用于后续权重计算
                score = (
                    circularity * 1.5 +                    # 降低圆形度权重
                    aspect_ratio * 0.8 +                   # 降低长宽比权重
                    brightness_uniformity * 1.2 +          # 降低亮度均匀性权重
                    (1.0 - min(h_std / 40.0, 1.0)) * 0.8 + # 放宽色相一致性要求
                    green_score * 2.0 +                    # 保持绿色评分权重
                    position_score * 2.5                   # 增加位置评分权重
                )
                
                return is_valid, {
                    'circularity': circularity,
                    'aspect_ratio': aspect_ratio,
                    'mean_brightness': mean_brightness,
                    'brightness_std': brightness_std,
                    'h_std': h_std,
                    'h_mean': h_mean,
                    'sv_ratio': sv_ratio,
                    'position_score': position_score,
                    'green_score': green_score,
                    'combined_score': score
                }
        
        return False, {}

    def detect_circles(self, mask, min_radius=5, max_radius=150):
        """
        使用霍夫圆变换检测圆形，调整参数以适应更大的圆
        """
        # 使用霍夫圆变换，降低参数阈值以检测更多圆
        circles = cv2.HoughCircles(
            mask, 
            cv2.HOUGH_GRADIENT, 
            dp=1,               # 累加器分辨率与图像分辨率的比率
            minDist=30,         # 减小检测到的圆的最小距离
            param1=40,          # 降低Canny边缘检测的高阈值
            param2=20,          # 降低累加器阈值，更容易检测到不完美的圆
            minRadius=min_radius,
            maxRadius=max_radius # 增大最大半径
        )
        
        return circles

    def detect_green_light_and_offset(self, frame, lower_hsv, upper_hsv):
        """检测绿光并计算偏移量（先按面积阈值筛选，再进行霍夫圆检测）"""
        # 获取图像中心点
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 创建掩膜
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        self.debug_mask = mask.copy()

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False, 0, None

        # 面积阈值（可按需调整）
        min_area_threshold = 100

        # 选出所有面积>=阈值的候选轮廓（不只取最大）
        candidate_contours = [c for c in contours if cv2.contourArea(c) >= min_area_threshold]
        if not candidate_contours:
            return False, 0, None

        # 基于候选轮廓生成候选区域掩膜
        candidate_mask = np.zeros_like(mask)
        cv2.drawContours(candidate_mask, candidate_contours, -1, 255, thickness=-1)

        # 为霍夫圆做平滑，降低噪声，仅在候选区域检测
        blurred = cv2.GaussianBlur(candidate_mask, (9, 9), 2)

        # 运行霍夫圆检测
        circles = self.detect_circles(blurred)
        if circles is None or len(circles[0]) == 0:
            return False, 0, None

        circles = np.round(circles[0, :]).astype(int)

        # 选择“最可能”的圆：优先候选区域覆盖最多的圆，其次取半径较大者
        def circle_score(c):
            x, y, r = int(c[0]), int(c[1]), int(c[2])
            temp = np.zeros_like(candidate_mask)
            cv2.circle(temp, (x, y), r, 255, thickness=-1)
            overlap = cv2.bitwise_and(temp, candidate_mask)
            return int(np.sum(overlap) // 255), r

        best_circle = max(circles, key=circle_score)
        cx, cy, r = int(best_circle[0]), int(best_circle[1]), int(best_circle[2])

        # 偏移量使用圆心
        horizontal_offset = cx - frame_center_x
        vertical_offset = cy - frame_center_y

        # 选取与圆心最匹配的轮廓用于调试绘制（保持字段不变）
        chosen_contour = None
        chosen_bbox = None
        for cnt in candidate_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if x <= cx <= x + w and y <= cy <= y + h:
                chosen_contour = cnt
                chosen_bbox = (x, y, w, h)
                break
        # 若没有包含圆心的，就用面积最大的候选
        if chosen_contour is None:
            chosen_contour = max(candidate_contours, key=cv2.contourArea)
            chosen_bbox = cv2.boundingRect(chosen_contour)

        # 计算圆面积作为目标面积
        target_area = float(np.pi * (r ** 2))

        # 生成调试结果图
        debug_vis = frame.copy()
        cv2.circle(debug_vis, (cx, cy), r, (0, 255, 255), 2)
        cv2.circle(debug_vis, (cx, cy), 3, (0, 0, 255), -1)
        self.debug_result = debug_vis

        # 返回结果，保持原有字段，并新增circle以便上层需要时使用
        return True, (horizontal_offset, vertical_offset), {
            'contour': chosen_contour,
            'center': (cx, cy),
            'area': target_area,
            'bbox': chosen_bbox,
            'circle': (cx, cy, r)
        }

# 创建检测器实例
detector = GreenLightDetector()

# 修改包装函数
def detect_green_light_and_offset(frame, lower_hsv, upper_hsv):
    """
    包装函数，返回检测结果和偏移量
    返回值: (success, offset, contour_info)
    - success: 布尔值，表示是否检测到目标
    - offset: 元组 (x_offset, y_offset)，表示目标相对于图像中心的偏移
    - contour_info: 字典，包含检测到的目标信息
    """
    return detector.detect_green_light_and_offset(frame, lower_hsv, upper_hsv)

def create_trackbars():
    """
    创建滑动条，用于调节 HSV 阈值
    如果无法创建窗口，将使用默认值
    """
    global GUI_AVAILABLE
    try:
        cv2.namedWindow('HSV Trackbars')
        # 根据图像中的绿光特征调整HSV范围 - 更宽松的范围
        cv2.createTrackbar('H min', 'HSV Trackbars', 30, 179, lambda x: None)  # 进一步降低H下限
        cv2.createTrackbar('H max', 'HSV Trackbars', 90, 179, lambda x: None)  # 进一步提高H上限
        cv2.createTrackbar('S min', 'HSV Trackbars', 30, 255, lambda x: None)  # 进一步降低S下限
        cv2.createTrackbar('S max', 'HSV Trackbars', 255, 255, lambda x: None)
        cv2.createTrackbar('V min', 'HSV Trackbars', 120, 255, lambda x: None)  # 调整V下限
        cv2.createTrackbar('V max', 'HSV Trackbars', 255, 255, lambda x: None)
        GUI_AVAILABLE = True
    except cv2.error:
        print("[INFO] OpenCV GUI不可用，使用默认HSV值")
        GUI_AVAILABLE = False

def get_trackbar_values():
    """
    获取滑动条的 HSV 阈值
    如果GUI不可用，则返回默认值
    :return: lower_hsv, upper_hsv
    """
    if GUI_AVAILABLE:
        try:
            h_min = cv2.getTrackbarPos('H min', 'HSV Trackbars')
            h_max = cv2.getTrackbarPos('H max', 'HSV Trackbars')
            s_min = cv2.getTrackbarPos('S min', 'HSV Trackbars')
            s_max = cv2.getTrackbarPos('S max', 'HSV Trackbars')
            v_min = cv2.getTrackbarPos('V min', 'HSV Trackbars')
            v_max = cv2.getTrackbarPos('V max', 'HSV Trackbars')
        except cv2.error:
            # 如果获取轨迹条失败，使用默认值
            h_min = DEFAULT_HSV_VALUES['H min']
            h_max = DEFAULT_HSV_VALUES['H max']
            s_min = DEFAULT_HSV_VALUES['S min']
            s_max = DEFAULT_HSV_VALUES['S max']
            v_min = DEFAULT_HSV_VALUES['V min']
            v_max = DEFAULT_HSV_VALUES['V max']
    else:
        # 使用默认值
        h_min = DEFAULT_HSV_VALUES['H min']
        h_max = DEFAULT_HSV_VALUES['H max']
        s_min = DEFAULT_HSV_VALUES['S min']
        s_max = DEFAULT_HSV_VALUES['S max']
        v_min = DEFAULT_HSV_VALUES['V min']
        v_max = DEFAULT_HSV_VALUES['V max']
    
    lower_hsv = np.array([h_min, s_min, v_min])
    upper_hsv = np.array([h_max, s_max, v_max])
    
    return lower_hsv, upper_hsv

def get_debug_images():
    """
    获取调试图像
    :return: mask图像, 结果图像
    """
    return detector.debug_mask, detector.debug_result