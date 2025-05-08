import torch.nn as nn
import modules.graph_conv_2 as sgc
import torch

class COMPLEX_GCN(nn.Module):
    def __init__(self, out_channel, N, A_base_tensor, inter_mask_tensor):
        super(COMPLEX_GCN, self).__init__()

        # 时域卷积
        self.time_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 127), stride=1, padding=(0, 63))
        self.bn0 = nn.BatchNorm2d(1, momentum=0.9)

        # 图卷积
        self.sgc = sgc.SpectralGraphConv(out_channels=out_channel, A_base=A_base_tensor, inter_mask=inter_mask_tensor)
        self.bn1 = nn.BatchNorm2d(out_channel, momentum=0.9)

        # 时空卷积
        self.conv = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * 5, kernel_size=(N, 17), stride=1, padding=(0, 8))
        self.bn2 = nn.BatchNorm2d(out_channel * 5, momentum=0.9)
        self.avg_pool0 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(0.5)

        # ===== Block 2: 可分离卷积 =====
        # 可分离卷积先用深度卷积（只在时间方向上操作）：
        # 卷积核尺寸 (1, 16)，为保证时间维度不变，padding=(0, 8)
        self.separableConv_depthwise = nn.Conv2d(out_channel * 5, out_channel * 5,
                                                 kernel_size=(1, 16), stride=1,
                                                 padding=(0, 8), groups=out_channel * 5,
                                                 bias=False)
        # 逐点卷积：将深度卷积的输出整合到 F2 个通道中，卷积核尺寸 (1, 1)
        self.separableConv_pointwise = nn.Conv2d(out_channel * 5, 64, kernel_size=(1, 1),
                                                 stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)

        # 线性层
        self.avg_pool1 = nn.AvgPool2d(kernel_size=(1, 8))  # (25, 1, 128) -> (25, 1, 1)
        self.fc1 = nn.Linear(64 * 4, 2)

        # 三个激活函数，目前relu没有被使用
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 三路网络，目前后两路不可用
        # 时域卷积
        x = x.unsqueeze(1)
        x = self.time_conv(x)
        x = self.bn0(x)

        # GCN部分
        x = self.sgc(x)
        x = self.elu(x)
        x = self.bn1(x)

        # 时空卷积
        x = self.elu(self.conv(x))   # (N, 25, 1, 128)
        x = self.bn2(x)
        x = self.avg_pool0(x)       # 时间维度缩小 4 倍
        x = self.dropout1(x)

        # Block 2
        x = self.separableConv_depthwise(x)
        x = self.separableConv_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)

        # 展平，全连接
        x = self.avg_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.sigmoid(self.fc1(x))

        return x
