

import pandas as pd

# 读取CSV文件
df = pd.read_csv('C:/Users/Administrator/Desktop/person information/申博/modal.csv')  # 将 'path_to_your_file.csv' 替换为你的CSV文件路径

# 假设CSV文件中的列名与示例一致
#algorithms = ['TCN',	'TCN-GMP',	'TCN-GA',	'LSTM',	'BiLSTM',	'BiLSTM_CNN',	'GRU']

#algorithms = ['CNNs', 'MobileNet', 'MobileNetV2', 'ShuffleNetV2']

algorithms = ['TCN-GA','CNNs', 'TrackAISNet']

# 创建一个字典来保存每个算法的均值结果
averages = {algorithm: [] for algorithm in algorithms}

# 对于每个算法，计算每15个连续训练结果的均值
for algorithm in algorithms:
    for i in range(0, len(df), 15):
        # 取出当前块的数据
        block = df[algorithm].iloc[i:i+15]
        # 计算均值并添加到结果列表
        if len(block) == 15:  # 确保我们处理的是完整的15个结果
            avg = block.mean()
            averages[algorithm].append(avg)

# 打印或保存结果
for algorithm, values in averages.items():
    print(f"{algorithm} 平均值: {values}")

# 如果需要将结果保存回CSV文件
result_df = pd.DataFrame(averages)
result_df.to_csv('C:/Users/Administrator/Desktop/person information/申博/smoothed_modal_acc.csv', index=False)



