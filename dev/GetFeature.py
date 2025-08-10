import serial
import time
import struct

# 串口配置
port = "COM3"   # 修改为你的端口
baud = 57600
ser = serial.Serial(port, baud)

# 控制开关
debug_packet = True   # True = 打印完整大包
extreme_interval = 10  # 每隔多少秒输出一次极值统计

# 全局统计
extreme_count = 0
last_extreme_report = time.time()

def read_bytes(n):
    """从串口读取 n 个字节"""
    return ser.read(n)

def run():
    global extreme_count, last_extreme_report

    print("[INFO] 开始采集数据...")

    while True:
        a = read_bytes(1)
        if len(a) == 0:
            continue

        if a[0] == 170:  # 0xAA
            a += read_bytes(1)
            if a[1] == 170:
                length_byte = read_bytes(1)
                if len(length_byte) == 0:
                    continue
                payload_length = length_byte[0]

                # 大包调试
                if payload_length == 0x20:
                    payload = list(read_bytes(payload_length))
                    checksum = read_bytes(1)

                    if debug_packet:
                        full_packet = [170, 170, payload_length] + payload + [checksum[0]]
                        print(f"[DEBUG] 大包数据 ({len(full_packet)} bytes): {full_packet}")

                    # Attention / Meditation 读取
                    try:
                        attention = payload[29]
                        meditation = payload[31]
                        signal_quality = payload[1]
                        print(f"[ATT] Attention={attention} Meditation={meditation} SignalQuality={signal_quality}")
                    except IndexError:
                        print("[WARN] 大包长度不足，无法解析 Attention/Meditation")

                # 原始波形数据包
                elif payload_length == 0x04:
                    payload = list(read_bytes(payload_length))
                    checksum = read_bytes(1)

                    # 按 ThinkGear 协议解析 raw 值
                    if payload[0] == 0x80 and payload[1] == 0x02:
                        raw_high = payload[2]
                        raw_low = payload[3]
                        raw_value = struct.unpack('>h', bytes([raw_high, raw_low]))[0]

                        # 极值检测
                        if raw_value in (2047, -2048):
                            extreme_count += 1

                        # 定时输出极值次数
                        now = time.time()
                        if now - last_extreme_report >= extreme_interval:
                            print(f"[INFO] 极值累计出现 {extreme_count} 次")
                            last_extreme_report = now

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\n[INFO] 停止采集")
        ser.close()
