#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import signal
import time
from datetime import datetime

class SystemRecorder:
    """系统录屏器，使用ffmpeg命令行工具进行屏幕录制"""
    
    def __init__(self, output_folder=None, format="mkv", capture_method="xcb"):
        """初始化录屏器
        
        Args:
            output_folder: 视频保存的文件夹路径，默认为DartSystem/video
            format: 视频格式，可选"mp4"、"avi"、"mkv"或"webm"，默认为 "mkv"
            capture_method: 捕获方法，可选"x11"、"xcb"或"fb"
        """
        self.capture_method = capture_method.lower()  # 捕获方法
        # 设置输出文件夹
        if output_folder is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_folder = os.path.join(script_dir, "video")
            
        self.output_folder = output_folder
        self.recording = False
        self.process = None
        self.output_path = None
        self.format = format.lower()  # 视频格式
        self.capture_method = capture_method.lower()  # 捕获方法
        
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"创建视频保存目录: {output_folder}")
        else:
            print(f"视频将保存到: {output_folder}")
    
    def start_recording(self):
        """开始录制屏幕"""
        if self.recording:
            print("已经在录制中")
            return
        
        try:
            # 检查ffmpeg是否已安装
            try:
                subprocess.run(["which", "ffmpeg"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError:
                print("错误: 未安装ffmpeg。请使用命令安装: sudo apt-get install ffmpeg")
                return
            
            # 生成输出文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 根据选择的格式设置文件扩展名和编码器
            extension = self.format
            video_codec = "libvpx" # 默认为 webm (VP8/VP9)
            ffmpeg_preset = None # 针对特定编码器的预设
            
            if extension == "mp4":
                video_codec = "libx264"
                ffmpeg_preset = "ultrafast" # 使用较快的预设以减少CPU负载
                # 更全面的参数设置，确保断电时文件可恢复
                movflags = [
                    "-movflags", "+frag_keyframe+empty_moov+faststart+default_base_moof", 
                    "-g", "30",  # 每30帧一个关键帧
                    "-keyint_min", "15", # 最小关键帧间隔
                    "-sc_threshold", "0", # 禁用场景变化检测
                    "-strict", "experimental"
                ]
                print("选择 MP4 格式，使用 H.264 编码器 (libx264) 和优化的断电恢复参数")
            elif extension == "mkv":
                video_codec = "libx264"
                ffmpeg_preset = "ultrafast"
                # MKV格式参数，更容易从断电中恢复
                movflags = [
                    "-g", "30",  # 每30帧一个关键帧
                    "-keyint_min", "15", # 最小关键帧间隔
                    "-sc_threshold", "0", # 禁用场景变化检测
                    "-cluster_time_limit", "1000", # 每秒创建新簇，便于恢复
                    "-write_crc32", "1", # 添加CRC32校验，帮助检测损坏
                    "-crf", "25" # 控制质量和大小
                ]
                print("选择 MKV 格式，使用 H.264 编码器 (libx264) 和断电恢复优化参数")
            elif extension not in ["avi", "mkv", "webm", "mp4"]:
                extension = "mkv" # 默认回退到 mkv，更容易恢复
                # 使用相同的恢复参数
                movflags = [
                    "-g", "30",  # 每30帧一个关键帧
                    "-keyint_min", "15", # 最小关键帧间隔
                    "-sc_threshold", "0", # 禁用场景变化检测
                    "-cluster_time_limit", "1000", # 每秒创建新簇，便于恢复
                    "-write_crc32", "1", # 添加CRC32校验，帮助检测损坏
                    "-crf", "25" # 控制质量和大小
                ]
                video_codec = "libx264"
                ffmpeg_preset = "ultrafast"
                print(f"不支持的格式 '{self.format}'，将使用默认的 mkv 格式。")
            else:
                movflags = []
                print(f"选择 {extension} 格式，使用 VP8/VP9 编码器 (libvpx)")

            self.output_path = os.path.join(self.output_folder, f"screen_recording_{timestamp}.{extension}")
            print(f"屏幕录制将保存到: {self.output_path}")
            
            # 基础命令参数 (适用于 X11/XCB)
            base_cmd_x11 = [
                "ffmpeg",
                "-f", "x11grab",
                "-draw_mouse", "1",
                "-s", self._get_screen_resolution(),
                "-i", ":0.0+0,0",
                "-r", "10",                # 帧率10fps
                "-pix_fmt", "yuv420p",     # 像素格式，对 mp4/h264 很重要
                "-y",                      # 覆盖已有文件
            ]
            
            # 添加编码器特定参数
            codec_params = [
                "-c:v", video_codec,
            ]
            if ffmpeg_preset:
                codec_params.extend(["-preset", ffmpeg_preset])
            
            # webm 特有参数 (之前的参数)
            if video_codec == "libvpx":
                 codec_params.extend([
                    "-b:v", "2000k",
                    "-cpu-used", "0",      # VP8/VP9 特定参数
                    "-auto-alt-ref", "1",
                    "-deadline", "good",
                    "-threads", "4",
                 ])
            # mp4 (h264) 通常不需要这么多参数，preset 已经包含很多优化
            # 可以根据需要添加 -crf (Constant Rate Factor) 来控制质量/大小平衡
            # 例如: codec_params.extend(["-crf", "23"]) # 默认值，较低=更好质量

            # 根据捕获方法构建最终命令
            if self.capture_method == "fb":
                # 使用framebuffer捕获方法 (保持不变，但注意可能不兼容所有格式)
                print("使用framebuffer捕获方法")
                cmd = [
                    "ffmpeg",
                    "-f", "fbdev",
                    "-framerate", "10",
                    "-i", "/dev/fb0",
                    "-pix_fmt", "yuv420p", # 确保像素格式
                    # FB可能不支持所有高级编码器选项，这里保持简单
                    "-c:v", video_codec, # 使用选择的编码器
                    "-y",
                    self.output_path
                ]
            elif self.capture_method == "xcb":
                print("使用XCB捕获方法")
                # XCB 和 X11grab 基本参数类似
                cmd = base_cmd_x11 + codec_params + movflags + [self.output_path]
            else: # 默认 x11grab
                print("使用X11捕获方法")
                cmd = base_cmd_x11 + codec_params + movflags + [self.output_path]
            
            # 增加调试输出
            print(f"屏幕录制命令: {' '.join(cmd)}")
            
            # 启动录制进程
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True # 添加这个参数，使输出为文本格式
            )
            
            self.recording = True
            print(f"开始录制屏幕 (进程ID: {self.process.pid})")
        
        except Exception as e:
            print(f"启动录制失败: {e}")
            self.recording = False
            self.process = None
    
    def stop_recording(self):
        """停止录制屏幕"""
        if not self.recording:
            print("没有正在进行的录制")
            return
        
        ffmpeg_stdout = ""
        ffmpeg_stderr = ""

        try:
            print("正在停止录制...")
            
            if self.process:
                # 向进程发送中断信号
                self.process.send_signal(signal.SIGINT)
                
                # 等待进程结束并获取输出
                try:
                    print("等待ffmpeg完成视频文件...")
                    ffmpeg_stdout, ffmpeg_stderr = self.process.communicate(timeout=15) # 增加超时并获取输出
                    print(f"ffmpeg 退出码: {self.process.returncode}")
                    if ffmpeg_stdout:
                        print(f"ffmpeg stdout:\n{ffmpeg_stdout}")
                    if ffmpeg_stderr:
                        print(f"ffmpeg stderr:\n{ffmpeg_stderr}")

                    print(f"录制已停止，视频已保存到: {self.output_path}")
                    print(f"绝对路径: {os.path.abspath(self.output_path)}")
                    
                    self._verify_video_file()
                except subprocess.TimeoutExpired:
                    print("警告: ffmpeg没有及时响应 SIGINT，尝试使用SIGTERM信号")
                    self.process.send_signal(signal.SIGTERM)
                    try:
                        ffmpeg_stdout, ffmpeg_stderr = self.process.communicate(timeout=10) # 再次尝试获取输出
                        print(f"ffmpeg 退出码: {self.process.returncode}")
                        if ffmpeg_stdout:
                            print(f"ffmpeg stdout (after SIGTERM):\n{ffmpeg_stdout}")
                        if ffmpeg_stderr:
                            print(f"ffmpeg stderr (after SIGTERM):\n{ffmpeg_stderr}")
                    except subprocess.TimeoutExpired:
                        print("警告: ffmpeg仍未响应 SIGTERM，强制终止")
                        self.process.kill()
                        ffmpeg_stdout, ffmpeg_stderr = self.process.communicate() # 获取最后输出
                        print(f"ffmpeg 退出码 (after SIGKILL): {self.process.returncode}")
                        if ffmpeg_stdout:
                            print(f"ffmpeg stdout (after SIGKILL):\n{ffmpeg_stdout}")
                        if ffmpeg_stderr:
                            print(f"ffmpeg stderr (after SIGKILL):\n{ffmpeg_stderr}")
            
            # 重置状态
            self.recording = False
            self.process = None
            
        except Exception as e:
            print(f"停止录制时出错: {e}")
            if ffmpeg_stdout: # 即使出错也尝试打印已捕获的输出
                print(f"ffmpeg stdout (on error):\n{ffmpeg_stdout}")
            if ffmpeg_stderr:
                print(f"ffmpeg stderr (on error):\n{ffmpeg_stderr}")
    
    def _verify_video_file(self):
        """验证生成的视频文件是否有效"""
        if not os.path.exists(self.output_path):
            print(f"错误: 视频文件不存在: {self.output_path}")
            return False
        
        # 检查文件大小
        file_size = os.path.getsize(self.output_path)
        if file_size < 1000:  # 小于1KB的文件可能有问题
            print(f"警告: 视频文件过小 ({file_size} 字节)，可能无效")
            return False
        
        # 使用ffprobe验证文件
        try:
            cmd = ["ffprobe", "-v", "error", self.output_path]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"警告: 视频文件可能损坏: {result.stderr}")
                return False
            print("视频文件验证通过")
            return True
        except Exception as e:
            print(f"验证视频文件时出错: {e}")
            return False
    
    def _get_screen_resolution(self):
        """获取屏幕分辨率"""
        try:
            # 使用xdpyinfo命令获取屏幕分辨率
            result = subprocess.run(
                "xdpyinfo | grep dimensions | awk '{print $2}'",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
            
            # 默认分辨率
            return "1920x1080"
        except:
            return "1920x1080"

# 全局录制器实例
recorder = None

def start_system_recording(format="mkv", capture_method="xcb"):
    """启动系统录制
    
    Args:
        format: 视频格式，可选"mp4"、"avi"、"mkv"或"webm"，默认使用mkv
        capture_method: 屏幕捕获方法，可选"x11"、"xcb"或"fb"，默认使用"xcb"解决黑屏问题
    """
    global recorder
    print(f"正在启动系统屏幕录制(格式: {format}, 捕获方法: {capture_method})...")
    try:
        recorder = SystemRecorder(format=format, capture_method=capture_method)
        recorder.start_recording()
        print("系统屏幕录制已成功启动!")
    except Exception as e:
        print(f"启动系统屏幕录制失败: {e}")

def stop_system_recording():
    """停止系统录制"""
    global recorder
    print("正在停止系统屏幕录制...")
    try:
        if recorder:
            recorder.stop_recording()
            print("系统屏幕录制已成功停止!")
        else:
            print("没有正在进行的录制")
    except Exception as e:
        print(f"停止系统屏幕录制失败: {e}")

# 在命令行直接运行此脚本时，提供简单的录制功能
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="系统屏幕录制工具")
    parser.add_argument("action", choices=["start", "stop", "record"], 
                        help="操作: start(开始录制), stop(停止录制), record(录制指定秒数)")
    parser.add_argument("-d", "--duration", type=int, default=10,
                        help="录制时长(秒)，默认10秒，仅当action=record时有效")
    args = parser.parse_args()
    
    if args.action == "start":
        start_system_recording(format="mkv")
    elif args.action == "stop":
        stop_system_recording()
    elif args.action == "record":
        print(f"开始录制，将持续{args.duration}秒...")
        start_system_recording(format="mkv")
        time.sleep(args.duration)
        stop_system_recording()
        print("录制完成！") 