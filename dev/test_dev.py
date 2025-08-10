import serial
import torch
import numpy as np
import time
from Model import EEGCNN
import sys

# ==== 参数配置 ====
SERIAL_PORT = 'COM3'
BAUD_RATE = 57600
SAMPLES_PER_SEGMENT = 128
label_map = {0: 'blink', 1: 'rest', 2: 'left', 3: 'right'}

# ==== 加载模型 ====
model = EEGCNN(num_classes=4)
model.load_state_dict(torch.load("model_dev/model3.pth", map_location='cpu'))
model.eval()

# ==== 初始化串口 ====
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"[✓] 串口连接成功：{SERIAL_PORT}")
except serial.SerialException as e:
    print(f"[×] 串口连接失败: {e}")
    sys.exit()


# ==== 检查校验和 ====
def valid_checksum(packet):
    if len(packet) != 8:
        return False
    sum_check = ((0x80 + 0x02 + packet[5] + packet[6]) ^ 0xFF) & 0xFF
    return sum_check == packet[7]


# ==== 对齐并读取一个合法包 ====
def read_aligned_packet():
    while True:
        head = ser.read(1)
        if head == b'\xaa':
            second = ser.read(1)
            if second == b'\xaa':
                body = ser.read(6)
                if len(body) == 6:
                    packet = head + second + body
                    if packet[2] == 0x04 and packet[3] == 0x80 and packet[4] == 0x02:
                        if valid_checksum(packet):
                            return packet


# ==== 提取原始值 ====
def extract_raw_value(packet):
    high = packet[5]
    low = packet[6]
    val = (high << 8) | low
    if val > 32768:
        val -= 65536
    return val


# ==== 采集片段 ====
def collect_segment():
    buffer = []
    while len(buffer) < SAMPLES_PER_SEGMENT:
        pkt = read_aligned_packet()
        raw_val = extract_raw_value(pkt)
        buffer.append(raw_val)
    return buffer


# ==== 模型预测 ====
def predict(buffer):
    input_tensor = torch.tensor(np.array(buffer, dtype=np.float32).reshape(1, 1, -1))
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        # print()
        return label_map.get(pred_class, "未知")


# ==== 主程序 ====
def main():
    print("开始实时预测（10次，每次1秒）...\n")
    for i in range(10):
        print(f"--- 第 {i + 1} 次 ---")
        segment = collect_segment()
        print("原始数据样本:", segment)
        result = predict(segment)
        print("预测结果：", result)
        time.sleep(1)

    print("\n[✓] 实时预测结束。")
    ser.close()


if __name__ == '__main__':
    main()
