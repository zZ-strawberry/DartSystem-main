from __future__ import annotations

import copy
import glob
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
IS_WINDOWS = sys.platform.startswith("win")
IS_LINUX = sys.platform.startswith("linux")


def default_capture_method() -> str:
    if IS_WINDOWS:
        return "gdigrab"
    if IS_LINUX:
        return "xcb"
    return "none"


def mv_sdk_python_path() -> Path:
    return PROJECT_ROOT / ("MvImport" if IS_WINDOWS else "MvImport_Linux")


def platform_default_serial_ports() -> list[str]:
    if IS_WINDOWS:
        return ["COM3", "COM4", "COM5"]
    return [
        "/dev/serial/by-id/*",
        "/dev/ttyACM0",
        "/dev/ttyACM1",
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
        "/dev/my_stm32",
    ]


DEFAULT_CONFIG = {
    "runtime": {
        "mode": "hik",
    },
    "ui": {
        "enable_display_window": True,
        "enable_tuning_window": True,
    },
    "test": {
        "source": "video",
        "path": "",
        "auto_start": False,
    },
    "camera": {
        "exposure": 16000.0,
        "gain": 15.9,
    },
    "serial": {
        "enabled": False,
        "primary_port": platform_default_serial_ports()[0],
        "baudrate": 115200,
        "backup_ports": platform_default_serial_ports()[1:],
        "reconnect_interval_ms": 5000,
        "send_every_n_frames": 1,
    },
    "detection": {
        "hsv": {
            "lower": [24, 31, 123],
            "upper": [83, 253, 255],
        },
        "brightness": {
            "min_v": 180,
            "max_std": 90,
        },
        "morphology": {
            "blur_kernel": 5,
            "open_kernel": 3,
            "close_kernel": 5,
            "erode_iterations": 0,
            "dilate_iterations": 1,
        },
        "contour": {
            "min_area": 60,
            "max_area": 20000,
            "far_min_area": 60,
            "near_min_area": 300,
            "min_circularity": 0.55,
            "min_aspect_ratio": 0.55,
            "min_fill_ratio": 0.55,
        },
        "circle": {
            "min_radius": 4,
            "max_radius": 120,
        },
        "scoring": {
            "brightness_weight": 1.4,
            "circularity_weight": 1.2,
            "fill_ratio_weight": 1.0,
            "center_weight": 0.35,
        },
        "debug": {
            "console_log_interval_frames": 20,
            "draw_rejected": False,
        },
    },
    "recording": {
        "enabled": False,
        "format": "mkv",
        "capture_method": default_capture_method(),
        "output_dir": "video",
        "display": None,
        "frame_rate": 10,
        "resolution": None,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict:
    config_file = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    if not config_file.exists():
        return copy.deepcopy(DEFAULT_CONFIG)

    with config_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件格式无效: {config_file}")

    return _deep_merge(DEFAULT_CONFIG, loaded)


def save_config(config: dict, config_path: str | Path | None = None) -> Path:
    config_file = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"
    content = render_config_with_comments(config)
    config_file.write_text(content, encoding="utf-8")
    return config_file


def _yaml_scalar(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return f"\"{value}\""
    return str(value)


def render_config_with_comments(config: dict) -> str:
    runtime = config.get("runtime", {})
    ui = config.get("ui", {})
    test = config.get("test", {})
    camera = config.get("camera", {})
    serial = config.get("serial", {})
    detection = config.get("detection", {})
    hsv = detection.get("hsv", {})
    brightness = detection.get("brightness", {})
    morphology = detection.get("morphology", {})
    contour = detection.get("contour", {})
    circle = detection.get("circle", {})
    scoring = detection.get("scoring", {})
    debug = detection.get("debug", {})
    recording = config.get("recording", {})
    backup_ports = serial.get("backup_ports", [])

    lines = [
        "# 运行模式",
        "# mode: hik 使用海康相机实时检测",
        "# mode: test 使用测试文件，不接相机也能调试",
        "runtime:",
        f"  mode: \"{runtime.get('mode', 'hik')}\"",
        "",
        "# UI 窗口开关",
        "# enable_display_window: 是否显示主画面窗口（原图/掩膜/状态）",
        "# enable_tuning_window: 是否显示实时调参窗口",
        "ui:",
        f"  enable_display_window: {str(bool(ui.get('enable_display_window', True))).lower()}",
        f"  enable_tuning_window: {str(bool(ui.get('enable_tuning_window', True))).lower()}",
        "",
        "# test 模式下的测试源设置",
        "# source: video 或 image",
        "# path: 测试文件路径，支持相对路径和绝对路径",
        "# auto_start: true 时程序启动后自动加载测试文件",
        "test:",
        f"  source: \"{test.get('source', 'video')}\"",
        f"  path: \"{test.get('path', '')}\"",
        f"  auto_start: {str(bool(test.get('auto_start', False))).lower()}",
        "",
        "# 相机参数",
        "camera:",
        f"  exposure: {float(camera.get('exposure', 16000.0))}",
        f"  gain: {float(camera.get('gain', 15.9))}",
        "",
        "# 串口通信参数",
        "serial:",
        f"  enabled: {str(bool(serial.get('enabled', False))).lower()}",
        f"  primary_port: \"{serial.get('primary_port', platform_default_serial_ports()[0])}\"",
        f"  baudrate: {int(serial.get('baudrate', 115200))}",
        f"  reconnect_interval_ms: {int(serial.get('reconnect_interval_ms', 5000))}",
        f"  send_every_n_frames: {int(serial.get('send_every_n_frames', 1))}",
        "  backup_ports:",
    ]
    for port in backup_ports:
        lines.append(f"    - \"{port}\"")
    if not backup_ports:
        lines.append("    - \"\"")

    lines.extend(
        [
            "",
            "# 绿光检测参数",
            "# hsv: 绿光颜色范围",
            "# brightness.min_v: 亮度下限，越高越偏向明亮目标",
            "# contour.min_area/max_area: 总面积范围",
            "# contour.far_min_area: 判定为远目标的最小面积阈值",
            "# contour.near_min_area: 判定为近目标的最小面积阈值",
            "# contour.min_circularity: 越高越要求接近圆",
            "# contour.min_fill_ratio: 越高越要求轮廓填充得更实",
            "# morphology: 掩膜去噪强度",
            "detection:",
            "  hsv:",
            f"    lower: {list(hsv.get('lower', [24, 31, 123]))}",
            f"    upper: {list(hsv.get('upper', [83, 253, 255]))}",
            "  brightness:",
            f"    min_v: {int(brightness.get('min_v', 180))}",
            f"    max_std: {int(brightness.get('max_std', 90))}",
            "  morphology:",
            f"    blur_kernel: {int(morphology.get('blur_kernel', 5))}",
            f"    open_kernel: {int(morphology.get('open_kernel', 3))}",
            f"    close_kernel: {int(morphology.get('close_kernel', 5))}",
            f"    erode_iterations: {int(morphology.get('erode_iterations', 0))}",
            f"    dilate_iterations: {int(morphology.get('dilate_iterations', 1))}",
            "  contour:",
            f"    min_area: {int(contour.get('min_area', 60))}",
            f"    max_area: {int(contour.get('max_area', 20000))}",
            f"    far_min_area: {int(contour.get('far_min_area', 60))}",
            f"    near_min_area: {int(contour.get('near_min_area', 300))}",
            f"    min_circularity: {float(contour.get('min_circularity', 0.55))}",
            f"    min_aspect_ratio: {float(contour.get('min_aspect_ratio', 0.55))}",
            f"    min_fill_ratio: {float(contour.get('min_fill_ratio', 0.55))}",
            "  circle:",
            f"    min_radius: {int(circle.get('min_radius', 4))}",
            f"    max_radius: {int(circle.get('max_radius', 120))}",
            "  scoring:",
            f"    brightness_weight: {float(scoring.get('brightness_weight', 1.4))}",
            f"    circularity_weight: {float(scoring.get('circularity_weight', 1.2))}",
            f"    fill_ratio_weight: {float(scoring.get('fill_ratio_weight', 1.0))}",
            f"    center_weight: {float(scoring.get('center_weight', 0.35))}",
            "  debug:",
            f"    console_log_interval_frames: {int(debug.get('console_log_interval_frames', 20))}",
            f"    draw_rejected: {str(bool(debug.get('draw_rejected', False))).lower()}",
            "",
            "# 录屏参数，默认关闭",
            "recording:",
            f"  enabled: {str(bool(recording.get('enabled', False))).lower()}",
            f"  format: \"{recording.get('format', 'mkv')}\"",
            f"  capture_method: \"{recording.get('capture_method', default_capture_method())}\"",
            f"  output_dir: \"{recording.get('output_dir', 'video')}\"",
            f"  display: {_yaml_scalar(recording.get('display'))}",
            f"  frame_rate: {int(recording.get('frame_rate', 10))}",
            f"  resolution: {_yaml_scalar(recording.get('resolution'))}",
            "",
        ]
    )
    return "\n".join(lines)


def _expand_port_candidate(candidate: str) -> list[str]:
    if "*" in candidate or "?" in candidate:
        return sorted(glob.glob(candidate))
    return [candidate]


def discover_serial_ports(configured_ports: list[str] | None = None) -> list[str]:
    candidates: list[str] = []

    for port in configured_ports or []:
        if port:
            candidates.extend(_expand_port_candidate(port))

    try:
        from serial.tools import list_ports

        discovered = sorted(
            (port.device for port in list_ports.comports() if port.device),
            key=str,
        )
        candidates.extend(discovered)
    except Exception:
        pass

    if IS_LINUX:
        for pattern in ["/dev/serial/by-id/*", "/dev/ttyACM*", "/dev/ttyUSB*"]:
            candidates.extend(sorted(glob.glob(pattern)))
    elif IS_WINDOWS:
        candidates.extend(f"COM{i}" for i in range(1, 21))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)

    return unique
