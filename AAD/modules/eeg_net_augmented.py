import torch
import torch.nn as nn
import modules.spectual_graph_conv as sgc
import modules.channel_wise_attention as cwa
import modules.self_attention as sa

class EEG_NET(nn.Module):
    def __init__(self, F1, D, F2, Chans, Samples, hiddenSize, A, dropoutRate=0.5, nb_classes=2):
        """
        参数说明：
          F1          - 第一个时域卷积的滤波器个数
          D           - 深度卷积中每个 F1 滤波器对应的空间滤波器个数（乘子）
          F2          - 可分离卷积中逐点卷积的输出通道数
          Chans       - EEG 信号的电极（通道）数
          Samples     - EEG 信号的采样点数
          dropoutRate - dropout 概率
          nb_classes  - 分类数
        """
        super(EEG_NET, self).__init__()

        # ===== Block 1: 时域卷积 + 空间深度卷积 =====
        # 时域卷积：在时间方向上滤波，卷积核尺寸 (1, 64)
        # 为保证时间维度不变，这里对时间方向进行 padding=(0,32)
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, 128), stride=1, padding=(0, 64), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.elu = nn.ELU()  # 激活函数

        # 空间深度卷积：滤波器个数为 F1*D，卷积核尺寸 (Chans, 1)，实现对各电极的空间滤波
        # groups=F1 表示对每个时域滤波器单独操作（深度卷积）
        self.depthwiseConv = nn.Conv2d(F1, F1 * D, kernel_size=(Chans, 1),
                                       stride=1, groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)

        # 平均池化和 dropout（Block 1 后）
        self.avgpool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)

        # ===== Block 2: 可分离卷积 =====
        # 可分离卷积先用深度卷积（只在时间方向上操作）：
        # 卷积核尺寸 (1, 16)，为保证时间维度不变，padding=(0, 8)
        self.separableConv_depthwise = nn.Conv2d(F1 * D, F1 * D,
                                                 kernel_size=(1, 16), stride=1,
                                                 padding=(0, 8), groups=F1 * D,
                                                 bias=False)
        # 逐点卷积：将深度卷积的输出整合到 F2 个通道中，卷积核尺寸 (1, 1)
        self.separableConv_pointwise = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1),
                                                 stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)

        # Block 2 后的平均池化和 dropout
        self.avgpool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)

        # ===== 全连接分类器 =====
        # 经过两次池化后，时间维度的变化：
        #   Block1：时间维度从 Samples 变为 Samples/4
        #   Block2：时间维度从 Samples/4 变为 (Samples/4)/8 = Samples/32
        final_time = Samples // 32  # 这里要求 Samples 能被 32 整除，否则可能需要调整

        # 最终特征图的尺寸为 (F2, 1, final_time)，flatten 后的维度为 F2 * final_time
        self.classifier = nn.Linear(F2 * final_time, nb_classes)

        # Add-Ons
        self.sgc = sgc.SpectralGraphConv(out_channels=F1, A=A)
        self.bn4 = nn.BatchNorm2d(F1)
        self.cwa = cwa.ChannelWiseAttention(64, 129, hidden_size=hiddenSize)
        from modules import self_attention as sa
        self.sa = sa.SelfAttention(128, 128)
        self.res = ResidualBlock(F2=F2, num_blocks=15)

    def forward(self, x):
        """
        输入 x 的形状应为: (batch_size, 1, Chans, Samples)
        """
        # Block 1
        x = x.unsqueeze(1)
        x = self.conv1(x)        # -> (batch, F1, Chans, Samples)
        x = self.bn1(x)
        x = self.elu(x)

        # Between Block Add-On
        x = self.cwa(x)
        x = self.sgc(x)
        x = self.bn4(x)
        x = self.elu(x)

        x = self.depthwiseConv(x)  # -> (batch, F1*D, 1, Samples)  (通道维度经过空间卷积后变为1)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)       # 时间维度缩小 4 倍
        x = self.dropout1(x)

        # Block 2
        x = self.separableConv_depthwise(x)
        x = self.separableConv_pointwise(x)
        x = self.bn3(x)
        x = self.elu(x)

        x = self.res(x)

        x = self.avgpool2(x)       # 时间维度进一步缩小 8 倍
        x = self.dropout2(x)

        x = x.squeeze(2)
        x = self.sa(x)

        # Flatten 展平，并通过全连接层
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, F2, num_blocks=3):
        super(ResidualBlock, self).__init__()
        self.num_blocks = num_blocks
        self.res_layers = nn.ModuleList([nn.Conv2d(F2, F2, kernel_size=(1, 1), stride=1, bias=False) for _ in range(num_blocks * 2)])
        self.elu = nn.ELU()

    def forward(self, x):
        res = x
        for i in range(0, self.num_blocks * 2, 2):
            x = self.res_layers[i](x)
            x = self.res_layers[i + 1](x)
            x = x + res  # 残差连接
            x = self.elu(x)
            res = x  # 更新残差分支
        return x