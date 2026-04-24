# DartSystem

用于检测明亮的圆形绿光，并将识别结果通过 CAN 发送到下位机。

当前主通信路径是：

- Linux
- `drv_ws` 的 `DaMeowCAN`
- CAN FD
- 应用层 payload 保持不变：
  `帧头 + 长度 + 7个float32小端 + CRC16 + 帧尾`

## 功能

- 海康相机实时采集
- 视频/图片测试模式
- 双目标绿光检测
- 近目标 / 远目标面积阈值分类
- 实时调参与保存到 `config.yaml`
- 通过 CAN FD 向下位机发送结果

## 目录

- [main.py](/E:/DartSystem-main/main.py:1)：主程序、GUI、相机、CAN 发送
- [opencv_green_detection.py](/E:/DartSystem-main/opencv_green_detection.py:1)：绿光检测算法
- [app_config.py](/E:/DartSystem-main/app_config.py:1)：默认配置与配置保存
- [config.yaml](/E:/DartSystem-main/config.yaml:1)：主配置文件
- [drv_ws](/E:/DartSystem-main/drv_ws:1)：下位机 CAN 驱动库
- [drv_ws_bridge](/E:/DartSystem-main/drv_ws_bridge:1)：Python 调 `drv_ws` 的桥接层

## 环境

仅推荐在 Linux 上部署 CAN 功能。

系统依赖：

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config libusb-1.0-0-dev
```

Python 环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 编译 drv_ws_bridge

主程序不会直接调用 `drv_ws` 的 C++ 类，而是通过桥接库 `libdrv_ws_bridge.so` 调用。

编译命令：

```bash
cmake -S drv_ws_bridge -B drv_ws_bridge/build
cmake --build drv_ws_bridge/build -j
```

编译完成后，至少应看到：

```bash
drv_ws_bridge/build/libdrv_ws_bridge.so
drv_ws_bridge/build/libdm_device.so
```

## 运行

```bash
source .venv/bin/activate
python main.py
```

如果先做算法联调，建议先用测试模式：

```yaml
runtime:
  mode: "test"

test:
  source: "video"
  path: "test.mp4"
  auto_start: true
```

## CAN 配置

主配置在 [config.yaml](/E:/DartSystem-main/config.yaml:1)。

达妙 CAN 盒推荐起始配置：

```yaml
can:
  enabled: true
  driver: "dameow"
  selector: "0"
  tx_id: "0x123"
  extended_id: false
  bus_mode: "canfd"
  bitrate: 500000
  data_bitrate: 2000000
  bitrate_switch: true
  channel: 0
  device_type: 0
  bridge_library: ""
  reconnect_interval_ms: 5000
  send_every_n_frames: 1
```

参数说明：

- `enabled`：是否启用 CAN 发送
- `driver`：`dameow` 或 `socketcan`。当前主路径用 `dameow`
- `selector`：达妙设备选择器，可填设备索引、SN、别名。单设备时通常先用 `"0"`
- `tx_id`：发送给下位机的 CAN ID
- `extended_id`：是否使用扩展帧
- `bus_mode`：当前必须使用 `canfd`。因为应用层 payload 是 33 字节，Classic CAN 单帧放不下
- `bitrate`：仲裁域波特率
- `data_bitrate`：数据域波特率
- `bitrate_switch`：CAN FD 是否启用 BRS
- `channel`：达妙设备通道号，通常先用 `0`
- `device_type`：设备类型，默认 `0`，即 `DEV_USB2CANFD`
- `bridge_library`：桥接库路径。留空时自动搜索 `drv_ws_bridge/build/libdrv_ws_bridge.so`
- `send_every_n_frames`：每几帧发送一次

如果改用 `socketcan`，还需要配置：

```yaml
can:
  driver: "socketcan"
  interface: "can0"
```

并提前由系统把接口拉起。注意：即使走 `socketcan`，当前 33 字节 payload 依然要求 CAN FD。

## 发送内容

发送给下位机的应用层内容没有改，仍然是 33 字节：

```text
Byte0:    0xA5
Byte1:    Length = 28
Byte2-29: 7个 float32，小端序
Byte30-31: CRC16，小端序
Byte32:   0x5A
```

7 个 `float32` 的含义：

1. `target_count`
2. `near_flag`
3. `near_dx`
4. `near_dy`
5. `far_flag`
6. `far_dx`
7. `far_dy`

说明：

- `dx = target_x - center_x`
- `dy = target_y - center_y`
- 近/远分类由面积阈值决定，不是简单按相对大小排序

## 现场调参

高频调参参数都在 `config.yaml` 的 `detection` 段。

最常调的参数：

- `detection.hsv.lower / upper`
- `detection.brightness.min_v`
- `detection.contour.min_area / max_area`
- `detection.contour.far_min_area / near_min_area`
- `detection.contour.min_circularity`
- `detection.contour.min_fill_ratio`
- `detection.circle.min_radius / max_radius`

界面支持：

- 重载配置
- 实时调参开关
- 保存当前实时参数到配置

## 建议联调顺序

1. 先编译 `drv_ws_bridge`
2. 用 `runtime.mode=test` 跑视频，确认检测结果稳定
3. 打开 `can.enabled=true`
4. 确认下位机监听的 `tx_id`、CAN FD 参数与配置一致
5. 再切到海康相机实时模式

## 常见问题

`CAN: 连接失败 (未找到 drv_ws bridge 库)`

- 说明桥接库没编出来，或者 `bridge_library` 路径不对

`No DaMeow devices found`

- 说明 `drv_ws` 没找到达妙设备，先检查 USB 连接和设备状态

`CANFD requested but selected DaMeow device does not support CANFD`

- 当前设备或配置不支持 CAN FD
- 在“不改发送内容”的前提下，这种情况无法继续，必须让链路支持 CAN FD

`Classic CAN 单帧最多 8 字节`

- 当前应用层包是 33 字节
- 不允许继续用 Classic CAN 单帧发送

## 当前限制

- `drv_ws_bridge` 需要在 Linux 上编译
- 目前没有把 33 字节 payload 拆分成多帧 Classic CAN
- 当前重点支持达妙 CAN 盒；`socketcan` 仅保留兼容路径
