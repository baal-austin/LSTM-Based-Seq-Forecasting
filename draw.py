import os

import matplotlib.pyplot as plt
import torch
from torch import nn

from test2 import LSTMModel, TimeSeriesDataset, load_model


def plot_results(predicted_output, true_output, day_index):
    # 创建时间轴
    time_steps = list(range(144))

    # 绘制曲线
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, predicted_output, label='Predicted', linestyle='-', color='blue')
    plt.plot(time_steps, true_output, label='True', linestyle='--', color='orange')

    # 添加标题和标签
    # plt.title('Radiation Value Prediction vs True Values')
    plt.xlabel('Time Steps (144 intervals in a day)')
    plt.ylabel('Irradiation(W/m^2)')
    plt.legend()
    plt.grid()
    plt.savefig(f"picture\\{day_index}.png", dpi=300, bbox_inches='tight')  # 保存为 PNG 格式
    # 显示图形
    # plt.show()


# 在预测函数中调用绘图函数
def predict(model, dataset, day_index, device='cpu'):
    input_data = dataset.inputs_scaler
    input_sequence = input_data[day_index].reshape(1, 144, 1)  # (1, 144, 1)

    input_tensor = torch.tensor(input_sequence).to(device)  # 将输入数据移到GPU

    with torch.no_grad():
        predicted_output = model(input_tensor)
    # 反归一化预测结果
    predicted_output = predicted_output.squeeze(-1).cpu().numpy()  # 转换为二维数组

    # 读取真实输出数据
    output_data = dataset.outputs_scaler
    true_output = output_data[day_index].reshape(1, 144, 1)  # (1, 144, 1)
    true_tensor = torch.tensor(true_output).to(device)  # 将真实输出移到GPU

    # 计算损失
    true_tensor = true_tensor.squeeze(-1).cpu().numpy()  # 转换为二维数组
    loss_function = nn.MSELoss()
    loss = loss_function(torch.tensor(predicted_output), torch.tensor(true_tensor))

    predicted_output = dataset.scaler_output.inverse_transform(predicted_output)

    # print(f'Predicted Output: {predicted_output.flatten()}')  # 将数据移回CPU
    # print(f'True Output: {dataset.outputs[day_index]}')
    print(f'Loss: {loss.item():.4f}')
    plot_results(predicted_output.flatten(), dataset.outputs[day_index], day_index)

# 示例用法
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = TimeSeriesDataset('input_data.csv', 'output_data.csv')
    model = LSTMModel().to(device)  # 将模型移到GPU
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    load_model(model, 'param1')  # 尝试加载模型
    # 训练模型
    # train_model('input_data.csv', 'output_data.csv', num_epochs=100000, batch_size=128, device=device)

    for i in range(364):
        # 预测某一天的数据，假设选择第0天的数据
        predict(model, dataset, day_index=i, device=device)


