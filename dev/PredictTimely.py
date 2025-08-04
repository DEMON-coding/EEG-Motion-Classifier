import serial
import torch
import numpy as np
import time
from Model import EEGCNN
import sys

import requests
# 机器人 IP 地址
ROBOT_IP = "http://192.168.4.1/"
BASE_URL = f"http://192.168.4.1/controller"

# ==== 参数配置 ====
SERIAL_PORT = 'COM3'
BAUD_RATE = 57600
SAMPLES_PER_SEGMENT = 256  # 更长一段用于滑窗
WINDOW_SIZE = 128
STEP_SIZE = 32
label_map = {0: 'blink', 1: 'frown', 2: 'rest'}

# ==== 加载模型 ====
model = EEGCNN(num_classes=3)
model.load_state_dict(torch.load("models_dict/eeg_cnn_model.pth", map_location='cpu'))
model.eval()

# ==== 初始化串口 ====
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"[✓] 串口连接成功：{SERIAL_PORT}")
except serial.SerialException as e:
    print(f"[×] 串口连接失败: {e}")
    sys.exit()


# ==== 发送指令 ====
def send_preset_command(command_id):
    """发送预设动作指令（如左转、前进等）"""
    try:
        params = {"pm": command_id}  # pm 参数对应动作编号
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()  # 检查请求是否成功
        print(f"指令发送成功，响应：{response.text}")
    except requests.exceptions.RequestException as err:
        print(f"发送指令失败：{err}")


# ==== 校验和 ====
def valid_checksum(packet):
    if len(packet) != 8:
        return False
    sum_check = ((0x80 + 0x02 + packet[5] + packet[6]) ^ 0xFF) & 0xFF
    return sum_check == packet[7]


# ==== 读取一个对齐数据包 ====
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


# ==== 预测单段 ====
def predict(buffer):
    input_tensor = torch.tensor(np.array(buffer, dtype=np.float32).reshape(1, 1, -1))
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        return label_map.get(pred_class, "未知")


# ==== 多窗口预测 + 多数投票 ====
def multi_predict(buffer, window_size=128, step_size=32):
    votes = {lab: 0 for lab in label_map.values()}
    for start in range(0, len(buffer) - window_size + 1, step_size):
        window = buffer[start:start + window_size]
        result = predict(window)
        votes[result] += 1
    final_result = max(votes, key=votes.get)
    return final_result


# ==== 主程序 ====
def main():
    print("开始实时预测（100次，每次约1秒）...\n")
    for i in range(100):
        print(f"\n--- 第 {i + 1} 次 ---")

        segment = []
        while len(segment) < SAMPLES_PER_SEGMENT:
            pkt = read_aligned_packet()
            val = extract_raw_value(pkt)
            segment.append(val)

        print("原始数据样本:", segment)

        result = multi_predict(segment, window_size=WINDOW_SIZE, step_size=STEP_SIZE)

        # 示例：控制机器人前进（对应 pm=2）
        if result == 'rest':
            send_preset_command(2)
        elif result == 'frown':
            send_preset_command(1)
        elif result == 'blink':
            send_preset_command(4)

        print("多数投票预测结果：", result)
        time.sleep(1)

    print("\n[✓] 实时预测结束。")
    ser.close()


if __name__ == '__main__':
    main()
