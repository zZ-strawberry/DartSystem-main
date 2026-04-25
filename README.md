# DartSystem

用于检测明亮的圆形绿光，并将识别结果通过 **SocketCAN + CAN FD** 发送到下位机。

当前通信实现只保留这一条路径：

- Linux
- `socketcan`
- CAN FD
- 应用层 payload 不变

## 功能

- 海康相机实时采集
- 视频/图片测试模式
- 双目标绿光检测
- 近目标 / 远目标面积阈值分类
- 实时调参与保存到 `config.yaml`
- 通过 CAN FD 向下位机发送结果

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

按当前默认参数：

- `bitrate = 500000`
- `data_bitrate = 2000000`
- `fd on`

执行：

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

## CAN 配置

[config.yaml](/E:/DartSystem-main/config.yaml:1) 中的推荐配置：

```yaml
can:
  enabled: true
  driver: "socketcan"
  interface: "can0"
  tx_id: "0x123"
  extended_id: false
  bus_mode: "canfd"
  bitrate: 500000
  data_bitrate: 2000000
  bitrate_switch: true
  reconnect_interval_ms: 5000
  send_every_n_frames: 1
```

参数说明：

- `enabled`：是否启用 CAN 发送
- `driver`：当前固定使用 `socketcan`
- `interface`：SocketCAN 接口名，如 `can0`
- `tx_id`：发送给下位机的 CAN ID
- `extended_id`：是否使用扩展帧
- `bus_mode`：当前必须使用 `canfd`
- `bitrate`：仲裁域波特率，仅作为配置记录
- `data_bitrate`：数据域波特率，仅作为配置记录
- `bitrate_switch`：CAN FD 是否启用 BRS
- `send_every_n_frames`：每几帧发送一次

## 发送内容

发送给下位机的应用层数据没有改，仍然是 33 字节：

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
- 当前发送的是偏移量，不是绝对像素坐标

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

界面支持：

- 重载配置
- 实时调参开关
- 保存当前实时参数到配置

## 联调顺序

1. 先把 `can0` 按 CAN FD 参数拉起
2. 用 `runtime.mode=test` 跑视频，确认检测结果稳定
3. 打开 `can.enabled=true`
4. 确认下位机监听的 `tx_id`、波特率、CAN FD 参数一致
5. 再切海康相机实时模式

## 常见问题

`CAN: 连接失败 (No such device)`

- 说明 `can0` 不存在

`CAN: 连接失败 (Operation not supported)`

- 说明接口或驱动没有正确支持 CAN FD

`Classic CAN 单帧最多 8 字节`

- 当前应用层包是 33 字节
- 如果不改协议，就必须继续使用 `canfd`

## 当前限制

- 仅保留 `socketcan` 通信路径
- 当前 33 字节 payload 没有拆成多帧 Classic CAN
- Linux 部署时需要系统已正确提供 `can0`
