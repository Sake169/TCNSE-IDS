import torch
import torch.nn as nn
import torch.nn.utils.parametrizations as param

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.model_name = 'CNNLSTM'
        self.conv1d_1 = param.weight_norm(nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=0))
        self.relu_1 = nn.ReLU(inplace=False)
        self.maxpool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 添加最大池化层
        self.conv1d_2 = param.weight_norm(nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0))
        self.relu_2 = nn.ReLU(inplace=False)
        self.maxpool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 添加最大池化层
        self.dropout = nn.Dropout(0.5)
        self.lstm_1 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        self.lstm_2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        
        # 多分类任务，总共有7个类别
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 7)

        # self.gap1 = nn.AdaptiveAvgPool1d(32)
        # self.gap2 = nn.AdaptiveAvgPool1d(1)
        # self.sigmoid = nn.Sigmoid() # 二分类

    def forward(self, x):
        output = self.conv1d_1(x)
        output = self.relu_1(output)
        output = self.maxpool1d_1(output)  # 添加最大池化层
        output = self.conv1d_2(output)
        output = self.relu_2(output)
        output = self.maxpool1d_2(output)  # 添加最大池化层
        output = self.dropout(output)
        output = output.permute(0, 2, 1)
        self.lstm_1.flatten_parameters()
        output, _ = self.lstm_1(output)
        self.lstm_2.flatten_parameters()
        output, _ = self.lstm_2(output)
        # output = self.gap1(output[:, -1])
        # output = self.gap2(output)
        # output = self.sigmoid(output)
        output = self.fc1(output[:, -1])
        output = self.fc2(output)
        return output

if __name__ == '__main__':
    model = CNNLSTM()
    x = torch.randn(1, 1, 41)
    output = model(x)
    print(output)
    a, b = torch.max(output, 1)
    print(a)
    print("    ")
    print(b)