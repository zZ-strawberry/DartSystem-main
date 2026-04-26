# DartSystem

用于检测明亮的圆形绿光，并可在 Linux 上将识别结果通过 **SocketCAN** 发送到下位机。

当前运行/通信边界：

- Linux
- `socketcan`
- Classic CAN（自动多帧分包）或 CAN FD（单帧）
- 应用层 payload 发送水平转角（度）
- Windows 可运行相机/视频检测和调参界面，但自动跳过发送层

## 功能

- 海康相机实时采集
- 视频/图片测试模式
- 双目标绿光检测
- 近目标 / 远目标面积阈值分类
- 实时调参与保存到 `config.yaml`
- Linux 下通过 CAN 向下位机发送结果（Classic 分包或 CAN FD 单帧）

## 文件

- [main.py](/E:/DartSystem-main/main.py:1)：主程序、GUI、相机、CAN 发送
- [opencv_green_detection.py](/E:/DartSystem-main/opencv_green_detection.py:1)：绿光检测算法
- [app_config.py](/E:/DartSystem-main/app_config.py:1)：默认配置与配置保存
- [config.yaml](/E:/DartSystem-main/config.yaml:1)：主配置文件

## Python 环境

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## SocketCAN 准备

程序不会自动配置 `can0`，要先在系统里把接口拉起。

按当前默认参数（`bus_mode: can`）：

- `bitrate = 500000`

执行：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000
sudo ip link set can0 up
```

如果使用 CAN FD（`bus_mode: canfd`），执行：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 500000 dbitrate 2000000 fd on
sudo ip link set can0 up
```

检查：

```bash
ip -details link show can0
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

说明：Linux 下开启 `can.enabled=true` 后，`hik` 和 `test` 两种运行模式都会发送 CAN 数据；Windows 下会自动跳过发送层。

## CAN 配置

[config.yaml](/E:/DartSystem-main/config.yaml:1) 中的推荐配置：

```yaml
can:
  enabled: true
  driver: "socketcan"
  interface: "can0"
  tx_id: "0x123"
  extended_id: false
  bus_mode: "can"
  bitrate: 500000
  data_bitrate: 2000000
  bitrate_switch: true
  reconnect_interval_ms: 5000
  min_send_interval_ms: 120
  send_every_n_frames: 2
```

参数说明：

- `enabled`：是否启用 CAN 发送
- `driver`：当前固定使用 `socketcan`；Windows 会自动跳过发送层
- `interface`：SocketCAN 接口名，如 `can0`
- `tx_id`：发送给下位机的 CAN ID
- `extended_id`：是否使用扩展帧
- `bus_mode`：`can`（Classic CAN 自动分包）或 `canfd`（单帧）
- `bitrate`：仲裁域波特率，仅作为配置记录
- `data_bitrate`：数据域波特率，`canfd` 模式下使用
- `bitrate_switch`：CAN FD 是否启用 BRS
- `min_send_interval_ms`：两次发送之间的最小间隔（毫秒）
- `send_every_n_frames`：每几帧发送一次

## 发送内容

发送给下位机的应用层数据为 25 字节，只包含水平转角：

```text
Byte0:    0xA5
Byte1:    Length = 20
Byte2-21: 5个 float32，小端序
Byte22-23: CRC16，小端序
Byte24:   0x5A
```

5 个 `float32` 的含义：

1. `target_count`
2. `near_flag`
3. `near_angle_deg`
4. `far_flag`
5. `far_angle_deg`

说明：

- 只使用目标中心的 x 轴像素位置：`x_offset = target_x - center_x`
- 标定优先公式：`angle_deg = atan(x_offset / fx) * 180 / pi`
- `calibration.image_width_px` 用于在当前帧宽不同于标定宽度时同步缩放 `fx/cx`
- 如果没有 `calibration.fx`，回退到镜头估算：`angle_deg = atan((x_offset * pixel_size_um / 1000) / focal_length_mm) * 180 / pi`
- 当前默认 `fx=15989.09`，小角度下每像素约 `0.003583°`

### Classic CAN 分包格式（`bus_mode: can`）

当 payload 大于 8 字节时，发送端会自动分包：

- 首帧（最多 8 字节）
  - Byte0: `0xA5`（分包起始标记）
  - Byte1: 总 payload 长度（当前为 25）
  - Byte2: 总帧数
  - Byte3: 序号 `0`
  - Byte4-7: payload 前 4 字节
- 后续帧（每帧最多 8 字节）
  - Byte0: `0x5A`（分包续传标记）
  - Byte1: 序号 `1..N`
  - Byte2-7: payload 数据分片（每帧最多 6 字节）

下位机按序号重组后，得到 25 字节应用层数据（头 `0xA5`、CRC16、尾 `0x5A` 均保持不变）。

## 调参

高频调参参数都在 `config.yaml` 的 `detection` 段。

最常调：

- `detection.hsv.lower / upper`
- `detection.brightness.min_v`
- `detection.contour.min_area / max_area`
- `detection.contour.far_min_area / near_min_area`
- `detection.contour.min_circularity`
- `detection.contour.min_fill_ratio`
- `detection.circle.min_radius / max_radius`
- `calibration.fx / fy / cx / cy`
- `detection.angle.focal_length_mm / pixel_size_um / center_x_px / invert_x`

界面支持：

- 重载配置
- 实时调参开关
- 保存当前实时参数到配置

## 联调顺序

1. 先按 `bus_mode` 对应参数把 `can0` 拉起
2. 用 `runtime.mode=test` 跑视频，确认检测结果稳定
3. 打开 `can.enabled=true`
4. 确认下位机监听的 `tx_id`、波特率与总线模式一致
5. 再切海康相机实时模式

## 常见问题

`CAN: 连接失败 (No such device)`

- 说明 `can0` 不存在

`CAN: 连接失败 (Operation not supported)`

- 如果当前是 `canfd`，说明接口或驱动没有正确支持 CAN FD
- 可切换 `bus_mode: can` 并按 Classic CAN 参数重配 `can0`

`Classic CAN 单帧最多 8 字节`

- 当前应用层包是 25 字节
- 程序已支持自动分包发送；当前上层 payload 为 25 字节水平转角包

`CAN 发送失败: [Errno 105] 没有可用的缓冲区空间`

- 说明发送速率超过当前总线可承载能力或总线 ACK 异常
- 建议先调大 `can.min_send_interval_ms`（如 120~300）或 `can.send_every_n_frames`（如 2~5）
- 可额外提高系统发送队列：`sudo ip link set can0 txqueuelen 1024`
- 检查总线是否有正常应答节点、终端电阻和波特率是否匹配

## 当前限制

- 仅保留 `socketcan` 通信路径
- Classic CAN 采用自定义分包格式，下位机需按文档重组
- Windows 可用于检测/调参，不启用发送层
- Linux 部署时需要系统已正确提供 `can0`
