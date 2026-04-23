#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class SystemRecorder:
    """基于 ffmpeg 的可选录屏器。"""

    SUPPORTED_FORMATS = {"avi", "mkv", "mp4", "webm"}

    def __init__(
        self,
        output_folder: str | os.PathLike | None = None,
        format: str = "mkv",
        capture_method: str = "xcb",
        display: str | None = None,
        frame_rate: int = 10,
        resolution: str | None = None,
    ) -> None:
        self.project_root = Path(__file__).resolve().parent
        self.output_folder = Path(output_folder) if output_folder else self.project_root / "video"
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.format = format.lower()
        self.capture_method = capture_method.lower()
        self.display = display or os.environ.get("DISPLAY") or ":0.0"
        self.frame_rate = max(int(frame_rate or 10), 1)
        self.resolution = resolution

        self.recording = False
        self.process: subprocess.Popen | None = None
        self.output_path: Path | None = None

    def is_supported(self) -> tuple[bool, str]:
        if shutil.which("ffmpeg") is None:
            return False, "未找到 ffmpeg，请先安装系统依赖。"

        if self.format not in self.SUPPORTED_FORMATS:
            return False, f"不支持的录制格式: {self.format}"

        if sys.platform.startswith("win"):
            if self.capture_method not in {"gdigrab", "none"}:
                return False, f"Windows 不支持录制方式: {self.capture_method}"
            return True, "ok"

        if sys.platform.startswith("linux"):
            if self.capture_method == "none":
                return False, "录屏已显式禁用。"
            if self.capture_method in {"xcb", "x11"}:
                if "DISPLAY" not in os.environ and not self.display:
                    return False, "未检测到 DISPLAY，当前不是可录制的图形会话。"
                return True, "ok"
            if self.capture_method == "fb":
                if not Path("/dev/fb0").exists():
                    return False, "/dev/fb0 不存在，无法使用 framebuffer 录屏。"
                return True, "ok"
            return False, f"Linux 不支持录制方式: {self.capture_method}"

        return False, f"当前平台不支持录屏: {sys.platform}"

    def start_recording(self) -> bool:
        if self.recording:
            print("录屏已在进行中")
            return True

        supported, reason = self.is_supported()
        if not supported:
            print(f"跳过录屏: {reason}")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = self.output_folder / f"screen_recording_{timestamp}.{self.format}"
        command = self._build_command()
        if command is None:
            print("无法构建录屏命令，已跳过")
            return False

        try:
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            print(f"启动录屏失败: {exc}")
            self.process = None
            return False

        self.recording = True
        print(f"录屏已启动: {self.output_path}")
        return True

    def stop_recording(self) -> None:
        if not self.recording or self.process is None:
            return

        try:
            self.process.send_signal(signal.SIGINT)
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self.recording = False
            self.process = None

        if self.output_path is not None:
            print(f"录屏已停止: {self.output_path}")

    def _build_command(self) -> list[str] | None:
        video_codec = "libx264" if self.format in {"mkv", "mp4"} else "libvpx"
        codec_args = ["-c:v", video_codec]

        if video_codec == "libx264":
            codec_args.extend(["-preset", "ultrafast", "-pix_fmt", "yuv420p"])
        else:
            codec_args.extend(["-b:v", "2000k", "-threads", "4"])

        if sys.platform.startswith("win"):
            if self.capture_method == "none":
                return None

            command = [
                "ffmpeg",
                "-f",
                "gdigrab",
                "-framerate",
                str(self.frame_rate),
                "-i",
                "desktop",
                "-y",
            ]
            if self.resolution:
                command.extend(["-video_size", self.resolution])
            return command + codec_args + [str(self.output_path)]

        if self.capture_method == "fb":
            command = [
                "ffmpeg",
                "-f",
                "fbdev",
                "-framerate",
                str(self.frame_rate),
                "-i",
                "/dev/fb0",
                "-y",
            ]
            return command + codec_args + [str(self.output_path)]

        command = [
            "ffmpeg",
            "-f",
            "x11grab",
            "-draw_mouse",
            "1",
            "-framerate",
            str(self.frame_rate),
            "-i",
            f"{self.display}+0,0",
            "-y",
        ]
        if self.resolution:
            command[1:1] = ["-video_size", self.resolution]
        return command + codec_args + [str(self.output_path)]


recorder: SystemRecorder | None = None


def start_system_recording(
    format: str = "mkv",
    capture_method: str = "xcb",
    output_folder: str | os.PathLike | None = None,
    display: str | None = None,
    frame_rate: int = 10,
    resolution: str | None = None,
) -> bool:
    global recorder
    recorder = SystemRecorder(
        output_folder=output_folder,
        format=format,
        capture_method=capture_method,
        display=display,
        frame_rate=frame_rate,
        resolution=resolution,
    )
    return recorder.start_recording()


def stop_system_recording() -> None:
    global recorder
    if recorder is not None:
        recorder.stop_recording()
        recorder = None
