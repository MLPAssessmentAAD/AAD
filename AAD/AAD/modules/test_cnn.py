import torch.nn as nn
import modules.self_attention as sa
import torch

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 时域卷积
        self.time_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 127), stride=1, padding=(0, 63))
        self.bn0 = nn.BatchNorm2d(1, momentum=0.9)

        self.attn = sa.SelfAttention(input_dim=64, embed_dim=64)

        # 时空卷积
        self.conv = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(64, 17), stride=1, padding=(0, 8))
        self.bn2 = nn.BatchNorm2d(5, momentum=0.9)

        # 线性层
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 128))  # (25, 1, 128) -> (25, 1, 1)
        self.fc1 = nn.Linear(5, 5)  # 将 (25, 1, 1) 展平成 25
        self.fc2 = nn.Linear(5, 2)  # 输出2维（假设二分类任务）

        # 三个激活函数，目前relu没有被使用
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 三路网络，目前后两路不可用
        x = x.unsqueeze(1)
        x = self.time_conv(x)
        x = self.bn0(x)
        x = self.elu(x)

        x = x.squeeze(1)
        x = self.attn(x)
        x = x.unsqueeze(1)

        x = self.conv(x)
        x = self.bn2(x)
        x = self.elu(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.elu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))

        return x