# -*- coding: utf-8 -*-
import queue  # 用于线程安全的数据缓存队列
import os

import threading
import serial
import time
data = []
data2 = []
data3 = []
old_data = []
delta_data = []

class EEGThread(threading.Thread):

    def __init__(self, parent=None):
        super(EEGThread, self).__init__(parent)
        # 请在此处将filename修改为实际的文件名
        self.filename = os.path.join('data_dev', 'jox.txt')
        # 请在此处将COM修改为实际脑电设备串口号
        self.com = "COM3"
        self.bps = 57600
        self.vaul = []
        self.is_open = False
        self.is_close = True
        self.raw_queue = queue.Queue()  # 用于异步打印 rawdata 的缓存队列

    def print_worker(self):
        """异步打印 rawdata 的线程任务"""
        write_count = 0  # 初始化计数器
        max_write_count = 100  # 设置最大写入次数

        while True:
            if write_count >= max_write_count:
                print("打印线程已达到最大写入次数，自动停止。")
                break

            try:
                time.sleep(0.5)  # 每0.5秒写一次缓存数据
                items = []
                while not self.raw_queue.empty():
                    items.append(self.raw_queue.get_nowait())
                if items:
                    with open(self.filename, 'a', encoding='utf-8') as f:
                        for t, r in items:
                            f.write(f"{t}\t{r}\n")

            except Exception as e:
                print("打印线程异常:", repr(e))

    def checkList(self,list,num):
        list_num = 0
        for i in list:
            if i > num:
                list_num += 1
        return list_num
    def checkEeg(self):
        old_num = 0
        delta_num = 0
        for old in old_data:
            if self.checkList(old,200)>5:
                old_num += 1

        delta_num =self.checkList(delta_data, 50000)

        if old_num > 3 and delta_num > 4:
            return True
        else:
            return False


    def run(self):
        global data,data2,data3,old_data,delta_data
        try:
            t = serial.Serial(self.com, self.bps)

            threading.Thread(target=self.print_worker, daemon=True).start()  # 启动异步打印线程

            b = t.read(3)
            print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+"脑电波设备配对中")
            while b[0] != 170 or b[1] != 170 \
                    or b[2] != 4:
                b = t.read(3)

            if b[0] == b[1] == 170 and b[2] == 4:
                print(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))+"配对成功。")
                a = b + t.read(5)

                if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:
                    while 1:
                        try:

                            a = t.read(8)
                            sum = ((0x80 + 0x02 + a[5] + a[6]) ^ 0xffffffff) & 0xff
                            if a[0] == a[1] == 170 and a[2] == 32:
                                y = 1
                            else:
                                y = 0
                            if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:
                                p = 1
                            else:
                                p = 0
                            if sum != a[7] and y != 1 and p != 1:
                                b = t.read(3)
                                c = b[0]
                                d = b[1]
                                e = b[2]
                                while c != 170 or d != 170 or e != 4:
                                    c = d
                                    d = e
                                    e = t.read()

                                    if c == (b'\xaa' or 170) and d == (b'\xaa' or 170) and e == b'\x04':
                                        g = t.read(5)
                                        if c == b'\xaa' and d == b'\xaa' and e == b'\x04' and g[0] == 128 and g[1] == 2:
                                            a = t.read(8)
                                            break

                            if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:

                                high = a[5]
                                low = a[6]
                                rawdata = (high << 8) | low
                                if rawdata > 32768:
                                    rawdata = rawdata - 65536
                                sum = ((0x80 + 0x02 + high + low) ^ 0xffffffff) & 0xff
                                if sum == a[7]:
                                    self.vaul.append(rawdata)
                                    timestamp = '0'
                                    self.raw_queue.put((timestamp, rawdata))  # 异步打印写入

                                if sum != a[7]:
                                    b = t.read(3)
                                    c = b[0]
                                    d = b[1]
                                    e = b[2]
                                    while c != 170 or d != 170 or e != 4:
                                        c = d
                                        d = e
                                        e = t.read()
                                        if c == b'\xaa' and d == b'\xaa' and e == b'\x04':
                                            g = t.read(5)
                                            if c == b'\xaa' and d == b'\xaa' and e == b'\x04' and g[0] == 128 and g[
                                                1] == 2:
                                                a = t.read(8)
                                                break
                            if a[0] == a[1] == 170 and a[2] == 32:
                                c = a + t.read(28)
                                delta = (c[7] << 16) | (c[8] << 8) | (c[9])
                                # print(delta)

                                data = self.vaul

                                old_data.append(data)
                                if len(old_data) > 10:
                                    old_data = old_data[-10:]

                                delta_data.append(delta)
                                if len(delta_data) > 10:
                                    delta_data = delta_data[-10:]

                                flag = self.checkEeg()
                                data2.append(c[32])

                                if len(data2) > 20:
                                    data2 = data2[-20:]

                                data3.append(c[34])

                                if len(data3) > 20:
                                    data3 = data3[-20:]

                                self.vaul = []
                        except Exception as e:
                            print(e)
                            sse = 1

        except Exception as e:
            print(e)
            sse = 1


if __name__ == '__main__':
    eeg = EEGThread()
    eeg.start()
