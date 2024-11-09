import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

# 1.LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=1, dropout=0.1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        output = self.layer_norm(output)

        # LSTM's final hidden state for the last layer (since bidirectional is False, hn shape is (num_layers, batch, hidden_dim))
        hidden = hn[-1, :, :]         #(batch,hidden_dim)

        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)

        out = self.fc2(hidden)

        return out



# 2.BiLSTM hidden用的整体时间序列
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=1, dropout=0.1):
        super(BiLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        output = self.layer_norm(output)

        hidden_forward = hn[-2, :, :]
        hidden_backward = hn[-1, :, :]
        hidden = torch.cat((hidden_forward, hidden_backward), dim=1)

        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)

        out = self.fc2(hidden)

        return out


# 3.GRU
class GRUClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=1, dropout=0.1):
        super(GRUClassifier, self).__init__()
        # 定义单向GRU，取消bidirectional参数
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        # 全连接层：因为是单向GRU，所以不需要 * 2
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, lengths):
        # GRU处理
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hn = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # LayerNorm应用于GRU输出
        output = self.layer_norm(output)

        # 获取GRU的隐藏状态，单向GRU只需使用最后一层的隐藏状态即可
        hidden_gru = hn[-1, :, :]  # 取最后一层的隐藏状态, 得到 (batch_size, hidden_dim)

        # Dropout和全连接层处理
        combined = self.dropout(hidden_gru)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)

        # 输出层
        out = self.fc2(combined)

        return out

###################### 4.BiGRU  #########################################

class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=1, dropout=0.1):
        super(BiGRUClassifier, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 隐藏状态和CNN输出的特征拼接后输入全连接层
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, lengths):
        # GRU处理
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hn = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # LayerNorm应用于GRU输出
        output = self.layer_norm(output)

        # 获取GRU隐藏状态
        hidden_forward = hn[-2, :, :]  # 前向GRU的最后一层隐藏状态
        hidden_backward = hn[-1, :, :]  # 后向GRU的最后一层隐藏状态
        hidden_gru = torch.cat((hidden_forward, hidden_backward),dim=1)  # 合并前向和后向的隐藏状态, 得到 (batch_size, hidden_dim * 2)

        # Dropout和全连接层处理
        combined = self.dropout(hidden_gru)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)

        # 输出层
        out = self.fc2(combined)

        return out


#  5.CCP+BiLSTM
class BiLSTMClassifierWithCNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=64, num_layers=1, dropout=0.1):
        super(BiLSTMClassifierWithCNN, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)  # input_dim 变为输入通道数
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(hidden_dim * 2 + 128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, lengths):
        # LSTM 处理
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # CNN 特征提取
        x_conv = x.permute(0, 2, 1)  # 转换维度：原始形状为 (64, 1440, 9)，转换为 (64, 9, 1440)
        x_conv = F.relu(self.conv1(x_conv))  # 输入尺寸为 (64, 9, 1440)，输出尺寸为 (64, 64, 1440)
        x_conv = F.relu(self.conv2(x_conv))  # 输入尺寸为 (64, 64, 1440)，输出尺寸为 (64, 128, 1440)

        # 对 CNN 的输出进行全局池化，将序列长度 1440 压缩为单一数值，结果为 (64, 128)
        x_conv = x_conv.mean(dim=2)  # 平均池化，沿着时间步长 (sequence_length) 维度取平均

        # 合并 LSTM 和 CNN 特征
        hidden_forward = hn[-2, :, :]  # 前向 LSTM 的最后一层隐藏状态
        hidden_backward = hn[-1, :, :]  # 后向 LSTM 的最后一层隐藏状态
        hidden_lstm = torch.cat((hidden_forward, hidden_backward), dim=1)  # 合并前向和后向的隐藏状态, 得到 (64, hidden_dim * 2)4
        # 合并 LSTM 和 CNN 提取的特征
        combined = torch.cat((hidden_lstm, x_conv), dim=1)  # 合并后的维度为 (64, hidden_dim * 2 + 128)

        combined = self.dropout(combined)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)

        out = self.fc2(combined)  # 最终输出类别预测
        return out


# 6.CCP+BiGRU
class BiGRUClassifierWithCNN(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_layers=1, dropout=0.1):
        super(BiGRUClassifierWithCNN, self).__init__()
        # GRU部分
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # CNN部分
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)  # 输入通道为input_dim
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2 + 128, hidden_dim)  # 隐藏状态和CNN输出的特征拼接后输入全连接层
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, lengths):
        # GRU处理
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hn = self.gru(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # LayerNorm应用于GRU输出
        output = self.layer_norm(output)

        # CNN特征提取
        x_conv = x.permute(0, 2, 1)  # 转换维度：原始形状为 (batch_size, seq_len, input_dim)，转换为 (batch_size, input_dim, seq_len)
        x_conv = F.relu(self.conv1(x_conv))  # 输出尺寸为 (batch_size, 64, seq_len)
        x_conv = F.relu(self.conv2(x_conv))  # 输出尺寸为 (batch_size, 128, seq_len)

        # 对CNN的输出进行全局池化，将序列长度压缩为单一数值，结果为 (batch_size, 128)
        x_conv = x_conv.mean(dim=2)  # 沿着时间步长维度进行平均池化

        # 获取GRU隐藏状态
        hidden_forward = hn[-2, :, :]  # 前向GRU的最后一层隐藏状态
        hidden_backward = hn[-1, :, :]  # 后向GRU的最后一层隐藏状态
        hidden_gru = torch.cat((hidden_forward, hidden_backward),dim=1)  # 合并前向和后向的隐藏状态, 得到 (batch_size, hidden_dim * 2)

        # 合并GRU和CNN提取的特征
        combined = torch.cat((hidden_gru, x_conv), dim=1)  # 合并后的维度为 (batch_size, hidden_dim * 2 + 128)

        # Dropout和全连接层处理
        combined = self.dropout(combined)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)

        # 输出层
        out = self.fc2(combined)

        return out



######################################## 7.TCN Backbone ####################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        # Initialize weights with normal distribution
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


###################### 8.TCN_GA ##########################
class TCNWithGlobalAttention(nn.Module):
    def __init__(self, input_dim, num_classes, num_channels=[64, 128], kernel_size=3, dropout=0.1):
        super(TCNWithGlobalAttention, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1])
        self.fc2 = nn.Linear(num_channels[-1], num_classes)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_channels[-1])

    def forward(self, x, lengths=None):
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        out = self.tcn(x)  # TCN层输出，(batch_size, num_channels, seq_length)

        # 全局注意力机制
        attn_weights = torch.mean(out, dim=1)  # (batch_size, seq_length)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch_size, seq_length)

        # 调整形状以进行矩阵乘法
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_length)

        # 加权输出
        out = torch.bmm(attn_weights, out.transpose(1, 2))  # (batch_size, 1, num_channels)
        out = out.squeeze(1)  # (batch_size, num_channels)

        # Apply dropout and normalization
        out = self.layer_norm(out)
        out = self.dropout(out)

        # 全连接层
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out



#################################### 9.TCN-GA-EfficientNet ######################
from efficientnet_pytorch import EfficientNet


class EfficientNet_FeatureExtractor(nn.Module):
    def __init__(self,hidden_dim=128, dropout=0.1):
        super(EfficientNet_FeatureExtractor, self).__init__()
        # 加载预训练的EfficientNet-B0模型
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0')

        # 替换分类头
        in_features = self.efficientnet._fc.in_features
        self.efficientnet._fc = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, images):
        out = self.efficientnet(images)  # (batch_size, num_classes)
        return out

class TCNWithGlobalAttention_EfficientNet(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        num_channels=[64, 128],
        kernel_size=3,
        dropout=0.1,
        hidden_dim=128,  # EfficientNet 特征维度
        pretrained=True  # 是否使用预训练权重
    ):
        super(TCNWithGlobalAttention_EfficientNet, self).__init__()

        # 序列数据分支 (TCN)
        self.tcn = TemporalConvNet(
            input_dim,
            num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.fc1 = nn.Linear(num_channels[-1], num_channels[-1])
        self.fc2 = nn.Linear(num_channels[-1] + hidden_dim, num_classes)  # 拼接后的维度变化
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_channels[-1])

        # 图像数据分支 (EfficientNet)
        self.efficientnet_extractor = EfficientNet_FeatureExtractor(hidden_dim=hidden_dim, dropout=dropout)
        # 可选：冻结 EfficientNet 参数
        for param in self.efficientnet_extractor.efficientnet.parameters():
            param.requires_grad = False

    def forward(self, x, lengths=None, images=None):
        # 处理序列数据
        x_seq = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_length)
        out = self.tcn(x_seq)  # (batch_size, num_channels[-1], seq_length)

        # 全局注意力机制
        attn_weights = torch.mean(out, dim=1)  # (batch_size, seq_length)
        attn_weights = torch.softmax(attn_weights, dim=-1)  # (batch_size, seq_length)
        attn_weights = attn_weights.unsqueeze(1)  # (batch_size, 1, seq_length)

        # 加权输出
        out = torch.bmm(attn_weights, out.transpose(1, 2))  # (batch_size, 1, num_channels[-1])
        out = out.squeeze(1)  # (batch_size, num_channels[-1])

        # 全连接层
        out = self.layer_norm(out)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)

        # 处理图像数据
        if images is not None:
            img_features = self.efficientnet_extractor(images)  # (batch_size, hidden_dim)

        # 拼接序列特征和图像特征
        combined_features = torch.cat((out, img_features), dim=1)  # (batch_size, num_channels[-1] + hidden_dim)

        # 最终分类
        out = self.fc2(combined_features)  # (batch_size, num_classes)

        return out

