import os
import re
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# 自定义数据集
class TimeSeriesDataset(Dataset):
    def __init__(self, input_file, output_file):
        self.inputs = pd.read_csv(input_file).values.astype('float32')
        self.outputs = pd.read_csv(output_file).values.astype('float32')

        # 归一化
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.inputs_scaler = self.scaler_input.fit_transform(self.inputs)
        self.outputs_scaler = self.scaler_output.fit_transform(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs_scaler[idx]).unsqueeze(-1), torch.tensor(self.outputs_scaler[idx]).unsqueeze(-1)

# LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out)
        return torch.relu(output)

def load_model(model, model_dir):
    def find_latest_model(model_dir):
        files = os.listdir(model_dir)
        model_files = [f for f in files if f.endswith('.pth')]

        max_epoch = -1
        latest_model = None

        for model_file in model_files:
            match = re.search(r'_(\d+)\.pth', model_file)
            if match:
                epoch = int(match.group(1))
                if epoch > max_epoch:
                    max_epoch = epoch
                    latest_model = model_file

        return latest_model

    latest_model_file = find_latest_model(model_dir)

    if latest_model_file:
        model.load_state_dict(torch.load(os.path.join(model_dir, latest_model_file), map_location='cpu'))  # 加载模型权重
        print(f"成功加载模型参数：{latest_model_file}")
    else:
        print("未找到可加载的模型参数，使用新模型进行训练。")

# 训练模型
def train_model(dataset, num_epochs=100, batch_size=32, device='cpu'):
    # dataset = TimeSeriesDataset(input_file, output_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel().to(device)  # 将模型移到GPU
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    load_model(model, 'param1')  # 尝试加载模型

    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据移到GPU
            optimizer.zero_grad()
            outputs = model(inputs)  # 直接使用inputs
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        if (epoch + 1) % 1000 == 0:
            # 保存模型权重，包含当前epoch
            torch.save(model.state_dict(), os.path.join('param1', f'lstm_model_{epoch+1}.pth'))

# 预测
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

    print(f'Predicted Output: {predicted_output.flatten()}')  # 将数据移回CPU
    print(f'True Output: {dataset.outputs[day_index]}')
    print(f'Loss: {loss.item():.4f}')


# 示例用法
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 检查是否有可用的GPU
    dataset = TimeSeriesDataset('input_data.csv', 'output_data.csv')
    # 训练模型
    # train_model(dataset, num_epochs=100000, batch_size=128, device=device)
    model = LSTMModel().to(device)  # 将模型移到GPU
    load_model(model, 'param1')  # 尝试加载模型
    # model.load_state_dict(torch.load(os.path.join('param1', f'lstm_model_{1000}.pth'), map_location=device))  # 加载模型权重
    model.eval()  # 切换到评估模式
    # 预测某一天的数据，假设选择第0天的数据

    predict(model, dataset, day_index=i, device=device)
    # predict(model, dataset, day_index=100, device=device)
    # predict(model, dataset, day_index=200, device=device)
