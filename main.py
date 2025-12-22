import sys
import yaml
import os
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFileDialog, QSlider, QCheckBox
from PyQt6.QtCore import Qt, QTimer, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from ctypes import *
import time
import atexit
import serial
import struct
#赛场 版本
# 调试模式增强：在终端同步输出原本通过串口发送的数据信息
from opencv_green_detection import create_trackbars, get_trackbar_values, detect_green_light_and_offset, detector
from system_recorder import start_system_recording as start_screen_recording
from system_recorder import stop_system_recording as stop_screen_recording

# 根据系统选择正确的库路径
if sys.platform.startswith("win"):
    sys.path.append("./MvImport")
    from MvImport.MvCameraControl_class import *
else:
    sys.path.append("./MvImport_Linux")
    from MvImport_Linux.MvCameraControl_class import *

# 串口通信类
class SerialCommunication(QObject):
    connection_status = pyqtSignal(bool)  # 定义信号，传递连接状态

    def __init__(self, port='/dev/my_stm32', baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = True
        self.connect_serial()

    def connect_serial(self):
        """尝试连接串口，失败时进行重试"""
        try:
            if self.serial is None or not self.serial.is_open:
                self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
                print(f"串口 {self.port} 连接成功")
                self.connection_status.emit(True)  # 连接成功
        except serial.SerialException as e:
            print(f"无法连接串口 {self.port}: {e}")
            self.connection_status.emit(False)  # 连接失败
            self.serial = None

    def stop(self):
        """停止串口读取并关闭连接"""
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
            print(f"串口 {self.port} 已关闭")
            
    def close(self):
        """关闭串口连接"""
        self.stop()

def get_Value(cam, param_type="float_value", node_name=""):
    """获取相机参数"""
    if param_type == "float_value":
        stParam = MVCC_FLOATVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_FLOATVALUE))
        ret = cam.MV_CC_GetFloatValue(node_name, stParam)
        if ret != 0:
            print("获取参数 %s 失败! ret[0x%x]" % (node_name, ret))
            return None
        return stParam.fCurValue
    elif param_type == "enum_value":
        stParam = MVCC_ENUMVALUE()
        memset(byref(stParam), 0, sizeof(MVCC_ENUMVALUE))
        ret = cam.MV_CC_GetEnumValue(node_name, stParam)
        if ret != 0:
            print("获取参数 %s 失败! ret[0x%x]" % (node_name, ret))
            return None
        return stParam.nCurValue

def set_Value(cam, param_type="float_value", node_name="", node_value=0):
    """设置相机参数"""
    if param_type == "float_value":
        ret = cam.MV_CC_SetFloatValue(node_name, node_value)
        if ret != 0:
            print("设置参数 %s 失败! ret[0x%x]" % (node_name, ret))
    elif param_type == "enum_value":
        ret = cam.MV_CC_SetEnumValue(node_name, node_value)
        if ret != 0:
            print("设置参数 %s 失败! ret[0x%x]" % (node_name, ret))

class MainWindow(QMainWindow):
    def __init__(self, camera_config=None, config=None):
        super().__init__()
        
        # 存储相机配置
        self.camera_config = camera_config
        # 保存全局配置引用
        self.config = config
        
        # 视频检测相关属性
        self.video_cap = None
        self.video_path = None
        self.video_fps = 0
        self.video_total_frames = 0
        self.current_frame_idx = 0
        self.is_video_mode = False
        self.is_video_playing = False
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.process_video_frame)
        
        # 初始化变量
        self.camera = None
        self.serial_comm = None  # 串口通信对象
        self.prev_frame_time = 0
        self.status_frame_counter = 0  # 添加状态帧计数器初始化
        self.debug_mode = False  # 调试模式标志

        # 初始化UI
        self.initUI()
        
        # 设置摄像头
        self.setup_camera()
        
        # 设置串口
        self.setup_serial()
        
        # 创建帧率计算定时器
        self.fps_timer = QTimer()
        self.fps_timer.timeout.connect(self.update_camera_frame)
        self.fps_timer.start(30)  # 30ms刷新一次，约为33帧每秒
        
        # 创建串口重新连接定时器
        self.reconnect_timer = QTimer()
        self.reconnect_timer.timeout.connect(self.refresh_connection)
        # 只有在串口启用时才启动重连定时器
        if self.config.get('serial', {}).get('enabled', True):
            self.reconnect_timer.start(5000)  # 每5秒尝试重新连接一次
            print("串口重连定时器已启动")
        else:
            print("串口连接已禁用，重连定时器未启动")

    def setup_camera(self):
        """初始化相机"""
        deviceList = MV_CC_DEVICE_INFO_LIST()
        tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
        ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
        if ret != 0:
            print("枚举设备失败! ret[0x%x]" % ret)
            return

        if deviceList.nDeviceNum == 0:
            print("没有找到设备!")
            self.camera = None
            self.enable_debug_mode()
            return

        print("找到 %d 个设备!" % deviceList.nDeviceNum)

        # 创建相机实例
        self.camera = MvCamera()
        stDeviceList = cast(deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents

        ret = self.camera.MV_CC_CreateHandle(stDeviceList)
        if ret != 0:
            print("创建句柄失败! ret[0x%x]" % ret)
            self.camera = None
            self.enable_debug_mode()
            return

        # 打开设备
        ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            print("打开设备失败! ret[0x%x]" % ret)
            self.camera = None
            self.enable_debug_mode()
            return

        # 设置相机参数，使用配置文件的值
        if self.camera_config and 'exposure' in self.camera_config:
            exposure = self.camera_config['exposure']
            print(f"使用配置文件设置曝光时间: {exposure}")
            set_Value(self.camera, "float_value", "ExposureTime", exposure)
        else:
            # 使用默认值
            print("使用默认曝光时间: 16000")
            set_Value(self.camera, "float_value", "ExposureTime", 16000)

        if self.camera_config and 'gain' in self.camera_config:
            gain = self.camera_config['gain']
            print(f"使用配置文件设置增益值: {gain}")
            set_Value(self.camera, "float_value", "Gain", gain)
        else:
            # 使用默认值
            print("使用默认增益值: 15.9")
            set_Value(self.camera, "float_value", "Gain", 15.9)

        # 开始取流
        ret = self.camera.MV_CC_StartGrabbing()
        if ret != 0:
            print("开始取流失败! ret[0x%x]" % ret)
            self.camera = None
            self.enable_debug_mode()
            return

        # 获取数据包大小
        stParam = MVCC_INTVALUE_EX()
        memset(byref(stParam), 0, sizeof(MVCC_INTVALUE_EX))
        ret = self.camera.MV_CC_GetIntValueEx("PayloadSize", stParam)
        if ret != 0:
            print("获取数据包大小失败! ret[0x%x]" % ret)
            self.camera = None
            self.enable_debug_mode()
            return

        self.data_size = stParam.nCurValue
        self.pData = (c_ubyte * self.data_size)()
        self.stFrameInfo = MV_FRAME_OUT_INFO_EX()
        memset(byref(self.stFrameInfo), 0, sizeof(self.stFrameInfo))
        
        print("相机初始化成功!")

    def enable_debug_mode(self):
        """启用调试模式，允许用户在终端输入信息进行调试"""
        print("\n" + "="*60)
        print("相机初始化失败，已启用调试模式")
        print("="*60)
        print("调试模式功能：")
        print("1. 可以打开视频文件进行检测")
        print("2. 可以在终端输入调试信息")
        print("3. 程序不会因为相机问题而崩溃")
        print("="*60)
        
        # 设置调试模式标志
        self.debug_mode = True
        
        # 在调试模式下禁用一些可能导致问题的UI功能
        if hasattr(self, 'video_slider'):
            self.video_slider.setEnabled(False)
            print("已禁用视频滑块（调试模式）")
        
        # 启动调试输入线程
        self.start_debug_input_thread()

    def start_debug_input_thread(self):
        """启动调试输入线程"""
        import threading
        
        def debug_input_worker():
            while hasattr(self, 'debug_mode') and self.debug_mode:
                try:
                    user_input = input("调试输入 (输入 'help' 查看帮助，'quit' 退出调试): ").strip()
                    if user_input.lower() == 'quit':
                        print("退出调试模式")
                        break
                    elif user_input.lower() == 'help':
                        print("调试命令帮助：")
                        print("  help  - 显示此帮助信息")
                        print("  quit  - 退出调试模式")
                        print("  status - 显示当前状态")
                        print("  video - 提示打开视频文件")
                        print("  其他输入将作为调试信息记录")
                    elif user_input.lower() == 'status':
                        print("当前状态：")
                        print(f"  相机连接: {'已连接' if self.camera else '未连接'}")
                        print(f"  串口连接: {'已连接' if hasattr(self, 'serial_comm') and self.serial_comm else '未连接'}")
                        print(f"  视频模式: {'是' if hasattr(self, 'is_video_mode') and self.is_video_mode else '否'}")
                        print(f"  调试模式: {'是' if hasattr(self, 'debug_mode') and self.debug_mode else '否'}")
                    elif user_input.lower() == 'video':
                        print("提示：请点击界面上的'打开视频'按钮来测试视频检测功能")
                    else:
                        print(f"调试信息: {user_input}")
                except (EOFError, KeyboardInterrupt):
                    print("\n调试输入被中断")
                    break
                except Exception as e:
                    print(f"调试输入错误: {e}")
        
        # 启动调试线程
        self.debug_thread = threading.Thread(target=debug_input_worker, daemon=True)
        self.debug_thread.start()
        print("调试输入线程已启动")

    def setup_serial(self):
        """初始化串口通信，使用多线程处理"""
        # 从配置中获取串口配置
        serial_config = self.config.get('serial', {})
        
        # 检查是否启用串口连接
        if not serial_config.get('enabled', True):
            print("串口连接已禁用，跳过串口初始化")
            self.serial_comm = None
            self.connection_status_label.setText("连接状态: 已禁用")
            return
        
        # 获取主串口，默认为'COM3'
        primary_port = serial_config.get('primary_port', 'COM3')
        # 获取波特率，默认为115200
        baudrate = serial_config.get('baudrate', 115200)
        # 获取备用串口列表，默认为['COM4', 'COM5']
        backup_ports = serial_config.get('backup_ports', ['COM4', 'COM5'])
        
        # 创建要尝试的端口列表，首先是主端口，然后是备用端口
        ports_to_try = [primary_port] + backup_ports
        
        # 移除重复的端口，保持顺序
        unique_ports = []
        for port in ports_to_try:
            if port not in unique_ports:
                unique_ports.append(port)
        
        # 首先检查哪些端口存在
        available_ports = []
        for port in unique_ports:
            try:
                # 尝试打开串口
                test_serial = serial.Serial(port, baudrate, timeout=0.1)
                test_serial.close()
                available_ports.append(port)
            except:
                print(f"端口 {port} 不可用，跳过尝试")
        
        # 如果没有可用端口，尝试查找系统上可能的串口设备
        if not available_ports:
            print("没有找到配置的串口设备，尝试搜索可能的串口设备...")
            # 搜索COM设备
            possible_ports = []
            for i in range(1, 10):  # 检查COM1到COM9
                port = f'COM{i}'
                try:
                    test_serial = serial.Serial(port, baudrate, timeout=0.1)
                    test_serial.close()
                    possible_ports.append(port)
                except:
                    continue
            
            if possible_ports:
                print(f"找到以下可能的串口设备: {possible_ports}")
                available_ports = possible_ports
            else:
                print("没有找到任何可用的串口设备")
        
        # 尝试连接每个可用端口
        for port in available_ports:
            try:
                print(f"尝试连接串口: {port}")
                self.serial_comm = SerialCommunication(port=port, baudrate=baudrate)
                
                # 更新连接状态标签
                self.connection_status_label.setText(f"连接状态: 已连接 ({port})")
                
                # 设置一个定时器，每5秒检查一次串口连接状态
                self.connection_timer = QTimer()
                self.connection_timer.timeout.connect(self.check_serial_connection)
                self.connection_timer.start(5000)  # 5000毫秒 = 5秒
                
                # 成功连接，退出循环
                return
            except Exception as e:
                print(f"串口 {port} 初始化失败: {str(e)}")
                continue
        
        # 所有端口都尝试失败
        print("所有串口尝试都失败，无法初始化串口通信")
        self.serial_comm = None
        self.connection_status_label.setText("连接状态: 未连接")
        
        # 设置一个定时器，每10秒重试一次连接
        self.retry_timer = QTimer()
        self.retry_timer.timeout.connect(self.retry_serial_connection)
        self.retry_timer.start(10000)  # 10000毫秒 = 10秒

    def check_serial_connection(self):
        """检查串口连接状态，如果断开则尝试重新连接"""
        # 检查串口是否被禁用
        if not self.config.get('serial', {}).get('enabled', True):
            return
            
        if self.serial_comm is None or not self.serial_comm.serial or not self.serial_comm.serial.is_open:
            print("串口连接已断开，尝试重新连接...")
            self.connection_status_label.setText("连接状态: 已断开，正在重连...")
            self.retry_serial_connection()
        else:
            # 连接正常，更新状态显示
            port = self.serial_comm.port if hasattr(self.serial_comm, 'port') else "未知"
            self.connection_status_label.setText(f"连接状态: 已连接 ({port})")

    def retry_serial_connection(self):
        """重试串口连接"""
        # 检查串口是否被禁用
        if not self.config.get('serial', {}).get('enabled', True):
            return
            
        print("重试串口连接...")
        # 先关闭现有连接（如果有）
        if self.serial_comm is not None:
            self.serial_comm.stop()
            self.serial_comm = None
        
        # 更新连接状态标签
        self.connection_status_label.setText("连接状态: 正在重新连接...")
        
        # 重新设置串口
        self.setup_serial()

    def update_connection_status(self, connected):
        """更新串口连接状态 - 仅保留兼容性，SerialReceiver不使用"""
        pass

    def read_frame(self):
        """读取一帧图像"""
        # 检查相机是否已连接
        if not hasattr(self, 'camera') or self.camera is None:
            return False, None
            
        ret = self.camera.MV_CC_GetOneFrameTimeout(self.pData, self.data_size, self.stFrameInfo, 1000)
        if ret == 0:
            # 转换图像格式
            data = np.frombuffer(self.pData, dtype=np.uint8)
            
            # 根据像素格式处理图像
            frame = data.reshape((self.stFrameInfo.nHeight, self.stFrameInfo.nWidth))
            
            if self.stFrameInfo.enPixelType == 17301505:  # Mono8
                img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif self.stFrameInfo.enPixelType == 17301513:  # BayerGB8
                img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)  # 使用BG2BGR而不是GB2BGR
            elif self.stFrameInfo.enPixelType == 17301514:  # BayerRG8
                img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            elif self.stFrameInfo.enPixelType == 17301515:  # BayerGR8
                img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            elif self.stFrameInfo.enPixelType == 17301516:  # BayerBG8
                img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
            else:
                try:
                    img = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                except:
                    try:
                        img = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    except:
                        print("无法转换图像格式")
                        return False, None
            
            # 将BGR转换为RGB用于显示
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return True, img
        else:
            print("获取一帧图像失败: ret[0x%x]" % ret)
            return False, None

    def update_camera_frame(self):
        """更新摄像头帧并显示"""
        # 如果是视频模式，不处理相机帧
        if self.is_video_mode:
            return
        
                # 检查相机是否已连接
        if not hasattr(self, 'camera') or self.camera is None:
            # 在调试模式下，显示提示信息
            if hasattr(self, 'debug_mode') and self.debug_mode:
                # 创建一个提示图像
                h, w = 480, 640
                debug_frame = np.ones((h, w, 3), dtype=np.uint8) * 128  # 灰色背景
                
                # 添加文字
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(debug_frame, "Camera Not Connected", (w//2-150, h//2-50), 
                           font, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(debug_frame, "Debug Mode Active", (w//2-120, h//2), 
                           font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(debug_frame, "Use Video Detection", (w//2-130, h//2+50), 
                           font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                
                # 转换为RGB并显示
                debug_frame_rgb = cv2.cvtColor(debug_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = debug_frame_rgb.shape
                bytes_per_line = ch * w
                qt_image = QImage(debug_frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
                
                # 在调试模式下也输出状态信息
                if hasattr(self, 'status_frame_counter'):
                    self.status_frame_counter += 1
                    if self.status_frame_counter % 60 == 0:  # 调试模式下每60帧显示一次
                        print(f"[调试模式] 相机未连接，系统状态：")
                        print(f"  - 调试模式: 已启用")
                        print(f"  - 相机状态: 未连接")
                        print(f"  - 串口状态: {'已连接' if hasattr(self, 'serial_comm') and self.serial_comm and self.serial_comm.serial and self.serial_comm.serial.is_open else '未连接'}")
                        print(f"  - 建议: 使用视频检测功能进行测试")
                        print("-" * 50)
            return
            
        # 计算帧率
        curr_frame_time = time.time()
        fps = 1 / (curr_frame_time - self.prev_frame_time) if hasattr(self, 'prev_frame_time') and self.prev_frame_time != 0 else 0
        self.prev_frame_time = curr_frame_time
        
        ret, frame = self.read_frame()
        if not ret:
            return

        # 获取滑动条的 HSV 阈值
        lower_hsv, upper_hsv = get_trackbar_values()

        # 检测是否存在绿光，并获取偏差值和轮廓信息
        green_detected, offset, contour_info = detect_green_light_and_offset(frame, lower_hsv, upper_hsv)

        # 显示图像和掩膜
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 在原始图像上绘制检测到的区域
        result = frame.copy()
        
        # 始终绘制相机中心点和十字线
        h, w = result.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # 设置文字显示的基础位置和间距
        text_start_x = 10
        text_start_y = 30
        line_spacing = 40
        
        # 显示帧率
        fps_y = text_start_y
        cv2.putText(result, f"FPS: {fps:.1f}", 
                    (text_start_x, fps_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (250, 100, 255), 2)

        # 计算偏移值（整数）
        if green_detected and contour_info:
            x_offset = int(contour_info['center'][0] - center_x)
            y_offset = int(contour_info['center'][1] - center_y)
        else:
            x_offset = 0
            y_offset = 0
        
        # 显示偏移值
        offset_y = fps_y + line_spacing
        cv2.putText(result, f"X_offset: {'+' if x_offset > 0 else ''}{x_offset:d}", 
                    (text_start_x, offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.putText(result, f"Y_offset: {'+' if y_offset > 0 else ''}{y_offset:d}", 
                    (text_start_x, offset_y + line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

        # 如果检测到目标，显示面积
        if green_detected and contour_info:
            area = contour_info['area']
            area_y = offset_y + line_spacing * 2
            cv2.putText(result, f"Area: {area:.0f}", 
                        (text_start_x, area_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
            
        # 修改串口通信部分
        # 计算状态值
        status_int = 1 if green_detected and contour_info else 0
        green_status_int = 1 if green_detected and contour_info else 0
        
        # 构建数据帧
        frame = bytearray()
        frame.append(0xA5)  # 帧头
        frame.append(status_int & 0xFF)  # 状态
        frame.append(green_status_int & 0xFF)  # 绿灯状态
        frame.append((x_offset >> 8) & 0xFF)  # x偏移高字节
        frame.append(x_offset & 0xFF)  # x偏移低字节
        frame.append((y_offset >> 8) & 0xFF)  # y偏移高字节
        frame.append((y_offset & 0xFF))  # y偏移低字节
        
        # 检查串口连接状态并发送数据
        if hasattr(self, 'serial_comm') and self.serial_comm is not None and hasattr(self.serial_comm, 'serial') and self.serial_comm.serial is not None and self.serial_comm.serial.is_open:
            try:
                # 发送数据帧
                self.serial_comm.serial.write(frame)
                
                # 显示发送信息（仅在控制台）
                if hasattr(self, 'status_frame_counter'):
                    self.status_frame_counter += 1
                    if self.status_frame_counter % 30 == 0:  # 每隔30帧显示一次
                        status_text = "检测到目标" if status_int == 1 else "未检测到目标"
                        print(f"[串口已连接] 发送偏移值帧：A5 {status_int:02X} {green_status_int:02X} {(x_offset >> 8) & 0xFF:02X} {x_offset & 0xFF:02X} {(y_offset >> 8) & 0xFF:02X} {y_offset & 0xFF:02X}")
                        print(f"  - 状态: {status_text}")
                        print(f"  - X偏移: {x_offset:d}")
                        print(f"  - Y偏移: {y_offset:d}")
                        print(f"  - 数据已通过串口发送")
            except Exception as e:
                print(f"发送数据时出错: {str(e)}")
        else:
            # 串口未连接时，在终端显示应该发送的数据
            if hasattr(self, 'status_frame_counter'):
                self.status_frame_counter += 1
                if self.status_frame_counter % 30 == 0:  # 每隔30帧显示一次
                    status_text = "检测到目标" if status_int == 1 else "未检测到目标"
                    print(f"[串口未连接] 应该发送的数据帧：A5 {status_int:02X} {green_status_int:02X} {(x_offset >> 8) & 0xFF:02X} {x_offset & 0xFF:02X} {(y_offset >> 8) & 0xFF:02X} {y_offset & 0xFF:02X}")
                    print(f"  - 状态: {status_text}")
                    print(f"  - X偏移: {x_offset:d}")
                    print(f"  - Y偏移: {y_offset:d}")
                    print(f"  - 目标面积: {contour_info['area']:.0f}" if green_detected and contour_info else "  - 目标面积: 无")
                    print(f"  - 注意：串口未连接，数据未实际发送")
                    print("-" * 50)  # 分隔线，便于阅读

        # 绘制中心十字线
        cv2.line(result, (center_x, 0), (center_x, h), (255, 0, 0), 1)  # 垂直线
        cv2.line(result, (0, center_y), (w, center_y), (255, 0, 0), 1)  # 水平线
        cv2.circle(result, (center_x, center_y), 5, (255, 0, 0), -1)  # 中心点
        
        if green_detected and contour_info:
            # 绘制目标边界框
            x, y, w, h = contour_info['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制轮廓
            cv2.drawContours(result, [contour_info['contour']], -1, (0, 255, 0), 2)
            
            # 绘制目标中心点
            target_center = contour_info['center']
            cv2.circle(result, target_center, 3, (0, 0, 255), -1)
            
            # 绘制从中心到目标的连线
            center = (center_x, center_y)
            cv2.line(result, center, target_center, (0, 255, 255), 2)

        # 显示图像
        h, w, ch = result.shape
        bytes_per_line = ch * w
        qt_image = QImage(result.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)

        # 显示掩膜图像
        h, w, ch = mask_colored.shape
        bytes_per_line = ch * w
        qt_mask = QImage(mask_colored.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        mask_pixmap = QPixmap.fromImage(qt_mask)
        scaled_mask = mask_pixmap.scaled(self.mask_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.mask_label.setPixmap(scaled_mask)

    def closeEvent(self, event):
        """窗口关闭事件"""
        print("正在关闭应用...")
        
        # 停止定时器
        if hasattr(self, 'fps_timer'):
            self.fps_timer.stop()
            
        # 停止串口连接检查定时器
        if hasattr(self, 'connection_timer'):
            self.connection_timer.stop()
            
        # 停止串口重试定时器
        if hasattr(self, 'retry_timer'):
            self.retry_timer.stop()
        
        # 关闭相机
        if hasattr(self, 'camera') and self.camera is not None:
            print("关闭相机...")
            self.camera.MV_CC_StopGrabbing()
            self.camera.MV_CC_CloseDevice()
            self.camera.MV_CC_DestroyHandle()
        
        # 停止屏幕录制
        print("停止屏幕录制...")
        stop_screen_recording()
        
        # 关闭视频
        if hasattr(self, 'video_cap') and self.video_cap:
            print("关闭视频...")
            self.video_cap.release()
            self.video_cap = None
        
        # 停止视频定时器
        if hasattr(self, 'video_timer') and self.video_timer.isActive():
            self.video_timer.stop()
        
        # 停止串口通信线程
        if hasattr(self, 'serial_comm') and self.serial_comm is not None:
            print("停止串口接收线程...")
            self.serial_comm.stop()
            self.serial_comm.wait()  # 等待线程结束
            print("串口接收线程已停止")
        elif not self.config.get('serial', {}).get('enabled', True):
            print("串口连接已禁用，无需关闭")
        
        print("应用已关闭")
        event.accept()

    def initUI(self):
        """初始化UI"""
        self.setWindowTitle('相机检测与视频检测')
        self.setGeometry(100, 100, 1400, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout()

        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.camera_label)

        self.mask_label = QLabel()
        self.mask_label.setMinimumSize(640, 480)
        right_layout.addWidget(self.mask_label)

        # 创建连接状态标签
        self.connection_status_label = QLabel("连接状态: 未知")
        self.connection_status_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        
        # 创建刷新按钮
        self.refresh_button = QPushButton("刷新连接")
        self.refresh_button.clicked.connect(self.refresh_connection)
        
        # 视频检测控制区域
        video_control_layout = QVBoxLayout()
        
        # 视频信息标签
        self.video_info_label = QLabel("相机模式")
        self.video_info_label.setStyleSheet("font-size: 14px; font-weight: bold; color: blue;")
        video_control_layout.addWidget(self.video_info_label)
        
        # 视频控制按钮
        video_button_layout = QHBoxLayout()
        
        self.open_video_button = QPushButton("打开视频")
        self.open_video_button.clicked.connect(self.open_video_file)
        video_button_layout.addWidget(self.open_video_button)
        
        self.video_play_button = QPushButton("播放")
        self.video_play_button.clicked.connect(self.toggle_video_playback)
        self.video_play_button.setEnabled(False)  # 初始禁用
        video_button_layout.addWidget(self.video_play_button)
        
        self.camera_mode_button = QPushButton("相机模式")
        self.camera_mode_button.clicked.connect(self.switch_to_camera_mode)
        video_button_layout.addWidget(self.camera_mode_button)
        
        video_control_layout.addLayout(video_button_layout)
        
        # 视频进度滑块
        self.video_slider = QSlider(Qt.Orientation.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.valueChanged.connect(self.on_video_slider_changed)
        video_control_layout.addWidget(self.video_slider)
        
        left_layout.addLayout(video_control_layout)
        
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.connection_status_label)
        control_layout.addWidget(self.refresh_button)
        
        left_layout.addLayout(control_layout)

        layout.addLayout(left_layout)
        layout.addLayout(right_layout)

        main_widget.setLayout(layout)

    def refresh_connection(self):
        """刷新串口连接"""
        # 检查串口是否被禁用
        if not self.config.get('serial', {}).get('enabled', True):
            print("串口连接已禁用，无法刷新")
            return
            
        if hasattr(self, 'serial_comm') and self.serial_comm is not None:
            print("尝试重新连接串口...")
            self.serial_comm.connect_serial()
            
            # 更新连接状态标签
            connection_status = "已连接" if (self.serial_comm.serial and self.serial_comm.serial.is_open) else "未连接"
            self.connection_status_label.setText(f"连接状态: {connection_status}")
            
            # 更新最后连接时间
            if self.serial_comm.serial and self.serial_comm.serial.is_open:
                self.last_serial_update_time = time.time()
        else:
            print("串口通信对象不存在，尝试重新初始化...")
            self.setup_serial()

    def open_video_file(self):
        """打开视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择视频文件", 
            "", 
            "视频文件 (*.mp4 *.avi *.webm *.mkv *.mov);;所有文件 (*.*)"
        )
        
        if file_path:
            self.load_video(file_path)
    
    def load_video(self, video_path):
        """加载视频文件"""
        try:
            # 关闭之前的视频
            if self.video_cap:
                self.video_cap.release()
            
            # 打开新视频
            self.video_cap = cv2.VideoCapture(video_path)
            if not self.video_cap.isOpened():
                print(f"无法打开视频文件: {video_path}")
                return
            
            self.video_path = video_path
            self.video_fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self.video_total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.current_frame_idx = 0
            self.is_video_mode = True
            self.is_video_playing = False

            self.video_play_button.setEnabled(True)
            
            # 更新UI
            self.video_slider.setMaximum(self.video_total_frames - 1)
            self.video_slider.setValue(0)
            self.video_info_label.setText(f"视频: {os.path.basename(video_path)}")
            self.video_play_button.setText("播放")
            
            # 显示第一帧
            self.process_video_frame()
            
            print(f"成功加载视频: {video_path}")
            print(f"总帧数: {self.video_total_frames}, FPS: {self.video_fps:.2f}")
            
        except Exception as e:
            print(f"加载视频时出错: {e}")
    
    def toggle_video_playback(self):
        """切换视频播放/暂停状态"""
        if not self.is_video_mode or not self.video_cap:
            return
        
        if self.is_video_playing:
            self.video_timer.stop()
            self.is_video_playing = False
            self.video_play_button.setText("播放")
        else:
            self.video_timer.start(int(1000 / self.video_fps))  # 根据视频FPS设置定时器
            self.is_video_playing = True
            self.video_play_button.setText("暂停")
    
    def process_video_frame(self):
        """处理视频帧"""
        if not self.is_video_mode or not self.video_cap:
            return
        
        # 读取当前帧
        ret, frame = self.video_cap.read()
        if not ret:
            # 视频结束，重新开始
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self.video_slider.setValue(0)
            return
        
        # 更新当前帧索引
       # 先更新帧索引再处理帧
        self.current_frame_idx = int(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.video_slider.setValue(self.current_frame_idx)
        ret, frame = self.video_cap.read()  # 然后读取帧
        
        # 处理帧（使用与相机相同的检测逻辑）
        self.process_frame_for_detection(frame)
    
    def process_frame_for_detection(self, frame):
        """处理帧进行检测（与相机检测逻辑相同）"""
        # 获取滑动条的 HSV 阈值
        lower_hsv, upper_hsv = get_trackbar_values()

        # 检测是否存在绿光，并获取偏差值和轮廓信息
        green_detected, offset, contour_info = detect_green_light_and_offset(frame, lower_hsv, upper_hsv)

        # 显示图像和掩膜
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # 在原始图像上绘制检测到的区域
        result = frame.copy()
        
        # 始终绘制相机中心点和十字线
        h, w = result.shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        # 设置文字显示的基础位置和间距
        text_start_x = 10
        text_start_y = 30
        line_spacing = 40
        
        # 显示帧信息
        if self.is_video_mode:
            frame_info_y = text_start_y
            cv2.putText(result, f"Frame: {self.current_frame_idx}/{self.video_total_frames}", 
                        (text_start_x, frame_info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (250, 100, 255), 2)
            text_start_y += line_spacing

        # 计算偏移值（整数）
        if green_detected and contour_info:
            x_offset = int(contour_info['center'][0] - center_x)
            y_offset = int(contour_info['center'][1] - center_y)
        else:
            x_offset = 0
            y_offset = 0
        
        # 显示偏移值
        offset_y = text_start_y
        cv2.putText(result, f"X_offset: {'+' if x_offset > 0 else ''}{x_offset:d}", 
                    (text_start_x, offset_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
        cv2.putText(result, f"Y_offset: {'+' if y_offset > 0 else ''}{y_offset:d}", 
                    (text_start_x, offset_y + line_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

        # 如果检测到目标，显示面积
        if green_detected and contour_info:
            area = contour_info['area']
            area_y = offset_y + line_spacing * 2
            cv2.putText(result, f"Area: {area:.0f}", 
                        (text_start_x, area_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        # 在视频模式下输出数据信息到终端
        if self.is_video_mode:
            # 计算状态值
            status_int = 1 if green_detected and contour_info else 0
            green_status_int = 1 if green_detected and contour_info else 0
            
            # 构建数据帧
            frame = bytearray()
            frame.append(0xA5)  # 帧头
            frame.append(status_int & 0xFF)  # 状态
            frame.append(green_status_int & 0xFF)  # 绿灯状态
            frame.append((x_offset >> 8) & 0xFF)  # x偏移高字节
            frame.append(x_offset & 0xFF)  # x偏移低字节
            frame.append((y_offset >> 8) & 0xFF)  # y偏移高字节
            frame.append(y_offset & 0xFF)  # y偏移低字节
            
            # 显示应该发送的数据（视频模式）
            if hasattr(self, 'status_frame_counter'):
                self.status_frame_counter += 1
                if self.status_frame_counter % 2 == 0:  # 每隔2帧显示一次
                    status_text = "检测到目标" if status_int == 1 else "未检测到目标"
                    print(f"[视频模式] 应该发送的数据帧：A5 {status_int:02X} {green_status_int:02X} {(x_offset >> 8) & 0xFF:02X} {x_offset & 0xFF:02X} {(y_offset >> 8) & 0xFF:02X} {y_offset & 0xFF:02X}")
                    print(f"  - 状态: {status_text}")
                    print(f"  - X偏移: {x_offset:d}")
                    print(f"  - Y偏移: {y_offset:d}")
                    print(f"  - 目标面积: {contour_info['area']:.0f}" if green_detected and contour_info else "  - 目标面积: 无")
                    print(f"  - 当前帧: {self.current_frame_idx}/{self.video_total_frames}")
                    print(f"  - 注意：视频模式，数据未通过串口发送")
                    print("-" * 50)  # 分隔线，便于阅读

        # 绘制中心十字线
        cv2.line(result, (center_x, 0), (center_x, h), (255, 0, 0), 1)  # 垂直线
        cv2.line(result, (0, center_y), (w, center_y), (255, 0, 0), 1)  # 水平线
        cv2.circle(result, (center_x, center_y), 5, (255, 0, 0), -1)  # 中心点
        
        if green_detected and contour_info:
            # 绘制目标边界框
            x, y, w, h = contour_info['bbox']
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 绘制轮廓
            cv2.drawContours(result, [contour_info['contour']], -1, (0, 255, 0), 2)
            
            # 绘制目标中心点
            target_center = contour_info['center']
            cv2.circle(result, target_center, 3, (0, 0, 255), -1)
            
            # 绘制从中心到目标的连线
            center = (center_x, center_y)
            cv2.line(result, center, target_center, (0, 255, 255), 2)

        # 显示图像
        h, w, ch = result.shape
        bytes_per_line = ch * w
        qt_image = QImage(result.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_label.setPixmap(scaled_pixmap)

        # 显示掩膜图像
        h, w, ch = mask_colored.shape
        bytes_per_line = ch * w
        qt_mask = QImage(mask_colored.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        mask_pixmap = QPixmap.fromImage(qt_mask)
        scaled_mask = mask_pixmap.scaled(self.mask_label.size(), Qt.AspectRatioMode.KeepAspectRatio)
        self.mask_label.setPixmap(scaled_mask)
    
    def on_video_slider_changed(self, value):
        """视频滑块值改变时的处理"""
        # 在调试模式下禁用此功能
        if hasattr(self, 'debug_mode') and self.debug_mode:
            return
            
        if not self.is_video_mode or not self.video_cap:
            return
        
        # 暂停播放
        if self.is_video_playing:
            self.toggle_video_playback()
        
        # 跳转到指定帧
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, value)
        self.current_frame_idx = value
        
        # 显示当前帧
        self.process_video_frame()
    
    def switch_to_camera_mode(self):
        """切换到相机模式"""
        if self.is_video_mode:
            # 停止视频播放
            if self.video_timer.isActive():
                self.video_timer.stop()
            
            # 关闭视频
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
            
            # 重置视频相关状态
            self.is_video_mode = False
            self.is_video_playing = False
            self.video_path = None
            
            # 更新UI
            self.video_info_label.setText("相机模式")
            self.video_play_button.setText("播放")
            self.video_slider.setValue(0)
            
            # 重新启动相机定时器
            if hasattr(self, 'fps_timer'):
                self.fps_timer.start(33)  # 30 FPS
            
            print("已切换到相机模式")
    
    def switch_to_video_mode(self):
        """切换到视频模式"""
        if not self.is_video_mode:
            # 停止相机定时器
            if hasattr(self, 'fps_timer') and self.fps_timer.isActive():
                self.fps_timer.stop()
                self.fps_timer = None  # 完全取消定时器
            
            self.is_video_mode = True
            print("已切换到视频模式")

             # 确保启用视频播放按钮
            self.video_play_button.setEnabled(True)

def main():
    # 尝试加载config.yaml配置文件
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # 如果找不到配置文件，使用默认参数
        config = {}
    
    # 检查配置是否有效
    if not config:
        config = {}
    
    # 确保camera配置存在
    if 'camera' not in config:
        print("警告：未找到相机配置，将使用默认参数")
        config['camera'] = {
            'exposure': 16000.0,  # 默认曝光时间
            'gain': 15.9         # 默认增益值
        }
    
    # 确保serial配置存在
    if 'serial' not in config:
        print("警告：未找到串口配置，将使用默认参数")
        config['serial'] = {
            'enabled': True, # 默认启用串口
            'primary_port': 'COM3',  # Windows下默认使用COM3
            'baudrate': 115200,
            'backup_ports': ['COM4', 'COM5']  # Windows下可能的备用串口
        }
    
    # 启动屏幕录制
    try:
        # 根据操作系统选择不同的捕获方法
        if sys.platform.startswith("win"):
            start_screen_recording(format="mkv", capture_method="gdigrab")  # Windows使用gdigrab
        else:
            start_screen_recording(format="mkv", capture_method="xcb")  # Linux使用xcb
    except Exception as e:
        print(f"启动屏幕录制失败: {e}")
    
    # 注册程序退出时的回调，确保录制停止
    atexit.register(stop_screen_recording)
    
    app = QApplication(sys.argv)
    window = MainWindow(camera_config=config['camera'], config=config)  # 传递完整的config
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()