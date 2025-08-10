import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from Model import EEGCNN

# ============ 1. 加载与处理数据 ============ #
df = pd.read_csv("csv_dev/eeg_data.csv")
df.fillna(df.select_dtypes(include=['float64', 'int64']).mean(), inplace=True)

le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])
print("标签映射：", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop(columns=["label"]).values.astype(np.float32)
y = df["label"].values.astype(np.int64)

# 4:1划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train).unsqueeze(1)
X_test_tensor = torch.tensor(X_test).unsqueeze(1)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ============ 2. 初始化模型 ============ #
device = torch.device("cpu")
model = EEGCNN(num_classes=len(le.classes_)).to(device)
print(len(le.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 随机梯度下降优化器

# ============ 3. 训练模型 ============ #
num_epochs = 32
train_losses = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)  # 这里就是前向传递
        loss = criterion(outputs, labels)

        optimizer.zero_grad()  # 清除旧的梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# ============ 5. 保存模型 ============ #
torch.save(model.state_dict(), "model_dev/model3.pth")
print("模型已保存至 model_dev/model.pth")

# ============ 6. 可视化训练过程 ============ #
plt.figure(figsize=(8, 5))
plt.plot(train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("png/loss_curve3.png")
print("损失曲线图已保存为 png/loss_curve.png")

# ============ 7. 测试评估 ============ #
model.eval()  # 评估模式
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
# 准确率与分类报告
print("测试准确率：", accuracy_score(all_labels, all_preds))
print("分类报告：\n", classification_report(all_labels, all_preds, target_names=le.classes_))

# 混淆矩阵（计数）
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm_raw = confusion_matrix(all_labels, all_preds)
disp_raw = ConfusionMatrixDisplay(confusion_matrix=cm_raw, display_labels=le.classes_)
plt.figure(figsize=(6, 5))
disp_raw.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("png/confusion_matrix3.png")
print("混淆矩阵图已保存")
