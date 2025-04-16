
# with open("/mnt/petrelfs/share_data/caizheng/train_ds/tokenized_data/en/refined-web-CC-MAIN-2013-20/train-000773-e51fcf79.bin.meta", "rb") as f:
#     content = f.read()

# print(content)

import numpy as np

meta_path = "/mnt/petrelfs/share_data/caizheng/train_ds/tokenized_data/en/refined-web-CC-MAIN-2013-20/train-000773-e51fcf79.bin.meta"
meta = np.load(meta_path)  # 直接加载 NumPy 数组

# 输出示例：
print(meta.shape)  # (N, 2)
# print(meta[:100])     # 第一行的 (cur, length)

# 定义阈值列表（单位：token 数量）
thresholds = [16000, 32000, 64000, 128000]

# 统计每个阈值的样本数量
counts = {}
for thresh in thresholds:
    mask = meta[:, 1] > thresh  # 第二列是 token 数量
    counts[thresh] = np.sum(mask)

# 输出结果
for thresh, count in counts.items():
    print(f"Token 数量超过 {thresh//1000}k 的样本数: {count}")