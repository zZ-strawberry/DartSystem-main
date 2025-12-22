import serial
import struct
import time
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtWidgets import QApplication
#赛场
class SerialReceiver(QThread):
    data_received = pyqtSignal(float, float)  # 定义信号，传递 angle 和 count

    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.buffer = bytearray()
        self.last_angle = 0
        self.last_count = 0
        self.serial = None
        self.running = True
        self.FRAME_HEADER = 0x05  # 帧头
        self.FRAME_TAIL = 0xA0    # 帧尾
        self.FRAME_LENGTH = 8     # 帧长度：帧头(1) + 角度值(4) + 圈数值(2) + 帧尾(1)
        self.connect_serial()
        self.error_count = 0
        self.max_errors = 50
        self.success_count = 0
        
        # 添加调试模式
        self.debug_mode = True
        # 尝试不同的帧格式
        self.try_alternative_format = False  # 关闭替代格式，因为我们已经知道确切的格式

    def connect_serial(self):
        """连接串口，确保连接成功"""
        try:
            if self.serial is None or not self.serial.is_open:
                self.serial = serial.Serial(self.port, self.baudrate, timeout=0.1)
                print(f"串口 {self.port} 连接成功")
                # 重置错误计数器
                self.error_count = 0
                self.success_count = 0
                # 清空缓冲区
                self.buffer.clear()
                print("串口接收线程已启动，使用端口：" + self.port)
        except serial.SerialException as e:
            print(f"无法连接串口 {self.port}: {e}")
            self.serial = None

    def parse_data(self, frame):
        """解析数据帧 - 根据STM32端的数据格式"""
        try:
            if len(frame) < 8:  # 确保帧长度正确
                return None, None
                
            # 检查帧头帧尾
            if frame[0] != self.FRAME_HEADER or frame[7] != self.FRAME_TAIL:
                return None, None
                
            # 解析角度值 (4字节，小端格式)
            angle_bytes = frame[1:5]
            angle = struct.unpack('<I', angle_bytes)[0]  # 无符号32位整数，小端格式
            
            # 解析圈数值 (2字节，小端格式)
            count_bytes = frame[5:7]
            count = struct.unpack('<h', count_bytes)[0]  # 有符号16位整数，小端格式
            
            if self.debug_mode:
                print(f"[成功] 解析数据: angle={angle}, count={count}")
                
            return angle, count
        except struct.error as e:
            if self.debug_mode:
                print(f"[错误] 解析数据失败: {e}, 帧数据: {frame.hex()}")
            return None, None

    def read_frame(self):
        """读取并解析数据帧"""
        if self.serial and self.serial.is_open:
            try:
                while self.running:
                    if self.serial.in_waiting > 0:
                        new_data = self.serial.read(self.serial.in_waiting)
                        self.buffer.extend(new_data)
                        if self.debug_mode:
                            print(f"[接收到原始字节流] {new_data.hex()}")

                    # 检查是否需要重新连接
                    if self.error_count > self.max_errors:
                        print(f"[警告] 错误次数({self.error_count})超过阈值，尝试重新连接串口")
                        self.serial.close()
                        self.connect_serial()
                        continue

                    # 查找完整帧
                    if len(self.buffer) >= self.FRAME_LENGTH:
                        try:
                            # 寻找帧头
                            start_index = self.buffer.index(self.FRAME_HEADER)
                            
                            # 确保有足够的字节形成一个完整帧
                            if len(self.buffer) - start_index >= self.FRAME_LENGTH:
                                frame = self.buffer[start_index:start_index + self.FRAME_LENGTH]
                                
                                # 检查帧尾
                                if frame[-1] != self.FRAME_TAIL:
                                    if self.debug_mode:
                                        print(f"[警告] 帧尾校验失败: {frame[-1]:02X}")
                                    self.buffer = self.buffer[start_index + 1:]
                                    self.error_count += 1
                                    continue

                                # 从缓冲区移除这一帧
                                self.buffer = self.buffer[start_index + self.FRAME_LENGTH:]
                                if self.debug_mode:
                                    hex_data = ' '.join(f'{b:02X}' for b in frame)
                                    print(f"[找到完整帧] {hex_data}")
                                
                                # 解析数据
                                angle, count = self.parse_data(frame)
                                if angle is None or count is None:
                                    self.error_count += 1
                                    continue
                                
                                # 数据有效性简单检查 (角度为无符号32位整数，圈数为有符号16位整数)
                                if angle < 0 or angle > 0xFFFFFFFF:  # 32位无符号整数范围
                                    if self.debug_mode:
                                        print(f"[警告] 无效的angle值: {angle}")
                                    self.error_count += 1
                                    continue
                                if count < -32768 or count > 32767:  # 16位有符号整数范围
                                    if self.debug_mode:
                                        print(f"[警告] 无效的count值: {count}")
                                    self.error_count += 1
                                    continue

                                print(f"[解析后] angle: {angle}, count: {count}")
                                self.last_angle = angle
                                self.last_count = count
                                
                                # 发送信号
                                self.data_received.emit(float(angle), float(count))
                                self.success_count += 1
                                self.error_count = max(0, self.error_count - 1)  # 成功一次，减少一个错误计数
                                
                                if self.success_count % 100 == 0:
                                    print(f"[信息] 已成功解析 {self.success_count} 帧数据")
                                    
                                return {'angle': angle, 'count': count}
                            else:
                                time.sleep(0.01)
                                continue
                        except ValueError:
                            # 找不到帧头，移除第一个字节并继续
                            if self.debug_mode:
                                print(f"[警告] 未找到帧头，移除第一个字节: {self.buffer.hex()}")
                            self.buffer.pop(0)
                            self.error_count += 1
                            continue
                    else:
                        time.sleep(0.01)
                        continue
            except serial.SerialException as e:
                print(f"[错误] 读取串口数据异常: {e}")
                self.error_count += 10
                self.connect_serial()
            except struct.error as e:
                print(f"[错误] 解析数据结构体异常: {e}, 缓冲区内容: {self.buffer.hex()}")
                self.buffer.clear()
                self.error_count += 5
            except Exception as e:
                print(f"[错误] 未知异常: {e}")
                self.error_count += 5
        else:
            self.connect_serial()

    def run(self):
        """串口数据接收线程"""
        while self.running:
            self.read_frame()

    def stop(self):
        """停止接收线程"""
        self.running = False
        if self.serial and self.serial.is_open:
            self.serial.close()
            print(f"串口 {self.port} 已关闭")

def your_slot_function(angle, count):
    """信号槽函数，处理接收到的数据"""
    print(f"接收到数据 - angle: {angle}, count: {count}")

def main():
    try:
        app = QApplication([])  # 启动 Qt 应用
        receiver = SerialReceiver(port='/dev/ttyACM0', baudrate=115200)
        receiver.data_received.connect(your_slot_function)  # 连接信号槽
        receiver.start()  # 启动接收线程
        print("串口接收器已启动，等待数据...")
        app.exec()  # 进入 Qt 的事件循环
    except KeyboardInterrupt:
        print("\n程序已停止")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        if 'receiver' in locals():
            receiver.stop()  # 停止接收线程
            receiver.wait()  # 等待线程完全结束

if __name__ == "__main__":
    main()
