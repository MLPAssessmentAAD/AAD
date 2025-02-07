import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import avg_pool


class ChannelWiseAttention(nn.Module):
    def __init__(self, num_channels, num_features, hidden_size):
        """
        Args:
            num_channels: 通道数（64）
            num_features: 特征维度（例如你希望经过图卷积后的特征数）
            hidden_size: 中间隐含层维度
        """
        super(ChannelWiseAttention, self).__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.hidden_size = hidden_size

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, num_features))
        self.fc1 = nn.Linear(self.num_channels, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.num_channels)

        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x: 输入信号，形状 (batch_size, num_channels, num_features)
        Returns:
            y: 输出信号，形状 (batch_size, num_channels, num_features)
        """
        multiplier = self.avg_pool(x)  # (batch, num_channels, 1)
        multiplier = multiplier.squeeze(-1)  # (batch, num_channels)
        multiplier = self.tanh(self.fc1(multiplier))
        multiplier = self.fc2(multiplier)   # (batch, num_channels)
        multiplier = multiplier.unsqueeze(-1)  # (batch, num_channels, 1)
        multiplier = multiplier.expand(-1, -1, self.num_features)  # (batch, num_channels, num_features)

        y = x * multiplier

        return y

