# -*- coding: utf-8 -*-
import queue
import os
import threading
import serial
import time
from datetime import datetime

data = []
data2 = []
data3 = []
old_data = []
delta_data = []

class EEGThread(threading.Thread):
    def __init__(self, parent=None, debug=False):
        super(EEGThread, self).__init__(parent)
        self.running = True
        self.print = False
        self.filename = os.path.join('data_dev', 'debug.txt')
        self.com = "COM3"
        self.bps = 57600
        self.vaul = []
        self.is_open = False
        self.is_close = True
        self.raw_queue = queue.Queue()
        self.debug = debug
        self.invalid_count = 0  # 记录极值数量

    def print_worker(self):
        write_count = 0  # 初始化计数器
        max_write_count = 8  # 设置最大写入次数

        while True:
            if write_count >= max_write_count:
                print("打印线程已达到最大写入次数，自动停止。")
                self.running = False
                return
            try:
                time.sleep(0.5)  # 每0.5秒写一次缓存数据
                items = []
                while not self.raw_queue.empty():
                    items.append(self.raw_queue.get_nowait())
                if items:
                    if self.print:
                        with open(self.filename, 'a', encoding='utf-8') as f:
                            for t, r in items:
                                f.write(f"{t}\t{r}\n")
                        write_count += 1
                        print(write_count)
            except Exception as e:
                print("打印线程异常:", repr(e))

    def checkList(self, list_, num):
        return sum(1 for i in list_ if i > num)

    def checkEeg(self):
        old_num = sum(1 for old in old_data if self.checkList(old, 200) > 5)
        delta_num = self.checkList(delta_data, 50000)
        return old_num > 3 and delta_num > 4

    def run(self):
        global data, data2, data3, old_data, delta_data
        try:
            t = serial.Serial(self.com, self.bps)
            threading.Thread(target=self.print_worker, daemon=True).start()

            b = t.read(3)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 脑电波设备配对中")
            while b[0] != 170 or b[1] != 170 or b[2] != 4:
                b = t.read(3)

            if b[0] == b[1] == 170 and b[2] == 4:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 配对成功。")
                a = b + t.read(5)

                if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:
                    while self.running:
                        try:
                            a = t.read(8)
                            sum_check = ((0x80 + 0x02 + a[5] + a[6]) ^ 0xffffffff) & 0xff
                            if a[0] == a[1] == 170 and a[2] == 32:
                                y = 1
                            else:
                                y = 0
                            if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:
                                p = 1
                            else:
                                p = 0
                            if sum_check != a[7] and y != 1 and p != 1:
                                b = t.read(3)
                                c, d, e = b[0], b[1], b[2]
                                while c != 170 or d != 170 or e != 4:
                                    c, d, e = d, e, t.read()[0]
                                    if c == 170 and d == 170 and e == 4:
                                        g = t.read(5)
                                        if g[0] == 128 and g[1] == 2:
                                            a = t.read(8)
                                            break

                            if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:
                                high = a[5]
                                low = a[6]
                                rawdata = (high << 8) | low
                                if rawdata > 32768:
                                    rawdata -= 65536
                                sum_check = ((0x80 + 0x02 + high + low) ^ 0xffffffff) & 0xff

                                # 极值屏蔽 + 调试输出
                                if rawdata in (2047, -2048):
                                    self.invalid_count += 1
                                    if self.debug:
                                        now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                        print(f"[DEBUG] 极值rawdata={rawdata}, high={high}, low={low}, "
                                              f"packet={list(a)}, time={now}")
                                    continue

                                if sum_check == a[7]:
                                    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                    self.vaul.append(rawdata)
                                    self.raw_queue.put((timestamp, rawdata))

                            if a[0] == a[1] == 170 and a[2] == 32:
                                c = a + t.read(28)
                                delta = (c[7] << 16) | (c[8] << 8) | c[9]
                                # print(delta)
                                data = self.vaul
                                old_data.append(data[-10:])
                                delta_data.append(delta)
                                old_data[:] = old_data[-10:]
                                delta_data[:] = delta_data[-10:]
                                self.checkEeg()

                                # ===== 新增：解析专注度、冥想度、信号质量 =====
                                signal_quality = c[4]
                                attention = c[32]
                                meditation = c[34]
                                if attention == 0:
                                    self.print = False
                                else:
                                    self.print = True
                                print(
                                    f"[ATT] Attention={attention} Meditation={meditation} SignalQuality={signal_quality}")
                                # =============================================

                                data2.append(c[32])
                                data2[:] = data2[-20:]
                                data3.append(c[34])
                                data3[:] = data3[-20:]
                                self.vaul = []

                        except Exception as e:
                            print(e)
        except Exception as e:
            print(e)
        finally:
            print(f"本次采集中检测到极值 {self.invalid_count} 次。")


if __name__ == '__main__':
    eeg = EEGThread(debug=True)  # 选择是否打开调试模式
    eeg.start()
