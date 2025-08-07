import os
import pandas as pd

# 配置参数
RAW_DATA_DIR = 'data_dev'              # 存放 txt 文件的文件夹
OUTPUT_CSV = 'csv_dev/eeg_data.csv'       # 输出的 CSV 文件
SAMPLES_PER_SEGMENT = 128              # 每个样本用 128 行数据
LABELS = ['blink', 'frown', 'rest']    # 支持的标签

# 确保输出目录存在
os.makedirs('csv_dev', exist_ok=True)

all_samples = []
all_labels = []

# 遍历所有 txt 文件
for filename in os.listdir(RAW_DATA_DIR):
    if not filename.endswith('.txt'):
        continue

    label = filename.split('.')[0].lower()
    if label not in LABELS:
        print(f"跳过未知标签文件: {filename}")
        continue

    file_path = os.path.join(RAW_DATA_DIR, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 跳过表头
    lines = lines[1:] if 'Raw' in lines[0] else lines

    # 提取每行的数值
    values = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            try:
                val = int(parts[1])
                values.append(val)
            except ValueError:
                continue

    # 切分为多个样本（每 128 个为一组）
    for i in range(0, len(values) - SAMPLES_PER_SEGMENT + 1, SAMPLES_PER_SEGMENT):
        segment = values[i:i + SAMPLES_PER_SEGMENT]
        all_samples.append(segment)
        all_labels.append(label)

print(f"总共构建了 {len(all_samples)} 个样本")

# 构建 DataFrame
df = pd.DataFrame(all_samples)
df['label'] = all_labels

# 保存为 CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"已保存数据集到: {OUTPUT_CSV}")
