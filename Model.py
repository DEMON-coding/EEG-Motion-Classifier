import torch.nn as nn


class EEGCNN(nn.Module):
    def __init__(self, num_classes):
        super(EEGCNN, self).__init__()
        # 第一个数据是batch_size
        # 输入：[16, 1, 128] 卷积后输出：[16, 16, 128]
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)  # padding 使信号两端填充了 0 值的点
        # 批归一化后 shape 不变：[16, 16, 128]
        self.bn1 = nn.BatchNorm1d(16)
        # 1 池化后长度减半：[16, 16, 64]
        # 2 池化后变为：[16, 32, 32]
        # 最大池化 保留“最强信号”，滤掉噪声
        self.pool = nn.MaxPool1d(2)
        # 输入：[16, 16, 64] 卷积后输出：[16, 32, 64]
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        # 批归一化不变：[16, 32, 64]
        self.bn2 = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(32 * 32, 64)
        self.fc2 = nn.Linear(64, num_classes)
        # ReLU激活也不改变形状
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # 扁平化操作，用于接入全连接层：
        # 原始形状：[batch_size, 32, 32]
        # 展平后形状：[batch_size, 32*32] = [batch_size, 1024]
        x = x.view(x.size(0), -1)
        # 把卷积层提取的特征压缩为 64 个特征
        x = self.relu(self.fc1(x))
        # 把 64 个特征映射为分类概率输出
        return self.fc2(x)
