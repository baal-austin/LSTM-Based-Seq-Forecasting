import pandas as pd

# 假设你的数据在一个CSV文件中
data = pd.read_csv('input_data_season.csv')

# 确保数据按时间排序
data['Datetime'] = pd.to_datetime(data['Datetime'])
# data = data.sort_values('Datetime')

# 创建输入输出数据
input_data = []
output_data = []

for i in range(0, len(data) - 144, 144):
    input_data.append(data.iloc[i:i + 144]["Irradiance"].values.flatten())
    output_data.append(data.iloc[i + 144:i + 288]["Irradiance"].values.flatten())
    print(i)

# 转换为DataFrame
input_df = pd.DataFrame(input_data)
output_df = pd.DataFrame(output_data)

# 保存为CSV文件
input_df.to_csv('input_data1.csv', index=False, header=False)
output_df.to_csv('output_data1.csv', index=False, header=False)
