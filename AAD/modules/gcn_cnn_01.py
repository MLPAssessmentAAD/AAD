import torch.nn as nn
import modules.spectual_graph_conv as sgc
import torch

class SIMPLE_GCN(nn.Module):
    def __init__(self, adj_matrix_tensor):
        super(SIMPLE_GCN, self).__init__()

        # 时域卷积
        self.time_conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 127), stride=1, padding=(0, 63))
        self.bn0 = nn.BatchNorm2d(1, momentum=0.9)

        # 图卷积
        self.sgc1 = sgc.SpectralGraphConv(in_features=128, out_features=128, A=adj_matrix_tensor)
        self.sgc2 = sgc.SpectralGraphConv(in_features=128, out_features=128, A=adj_matrix_tensor)
        self.sgc3 = sgc.SpectralGraphConv(in_features=128, out_features=128, A=adj_matrix_tensor)
        self.sgc4 = sgc.SpectralGraphConv(in_features=128, out_features=128, A=adj_matrix_tensor)
        self.sgc5 = sgc.SpectralGraphConv(in_features=128, out_features=128, A=adj_matrix_tensor)
        self.bn1 = nn.BatchNorm2d(5, momentum=0.9)

        # 时空卷积
        self.conv = nn.Conv2d(in_channels=5, out_channels=25, kernel_size=(64, 17), stride=1, padding=(0, 8))
        self.bn2 = nn.BatchNorm2d(25, momentum=0.9)

        # 线性层
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 128))  # (25, 1, 128) -> (25, 1, 1)
        self.fc1 = nn.Linear(25, 10)  # 将 (25, 1, 1) 展平成 25
        self.fc2 = nn.Linear(10, 2)  # 输出2维（假设二分类任务）

        # 三个激活函数，目前relu没有被使用
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        # 另外两路卷积，暂不可用
        self.conv_left = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(15, 17), stride=1, padding=(0, 8))
        self.conv_right = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(15, 17), stride=1, padding=(0, 8))

    def forward(self, x):
        # 三路网络，目前后两路不可用
        x_eeg = x[:, :64, :]
        x_left = x[:, 64:79, :]
        x_right = x[:, 79:, :]

        # 时域卷积
        x_eeg = x_eeg.unsqueeze(1)
        x_eeg = self.time_conv(x_eeg)
        x_eeg = self.bn0(x_eeg)
        x_eeg = x_eeg.squeeze(1)

        # GCN部分
        x_eeg_1 = self.sgc1(x_eeg).unsqueeze(1)
        x_eeg_2 = self.sgc2(x_eeg).unsqueeze(1)
        x_eeg_3 = self.sgc3(x_eeg).unsqueeze(1)
        x_eeg_4 = self.sgc4(x_eeg).unsqueeze(1)
        x_eeg_5 = self.sgc5(x_eeg).unsqueeze(1)
        x_eeg = torch.cat((x_eeg_1, x_eeg_2, x_eeg_3, x_eeg_4, x_eeg_5), dim=1)  # (N, 5, 64, 128)
        x_eeg = self.elu(x_eeg)
        x_eeg = self.bn1(x_eeg)

        # 时空卷积
        x_eeg = self.elu(self.conv(x_eeg))   # (N, 25, 1, 128)
        x_eeg = self.bn2(x_eeg)

        # 展平，全连接
        x_eeg = self.avg_pool(x_eeg)
        x_eeg = x_eeg.view(x_eeg.size(0), -1)
        x_eeg = self.elu(self.fc1(x_eeg))

        # 暂时不可用
        # x_left = x_left.unsqueeze(1)
        # x_right = x_right.unsqueeze(1)
        # x_left = self.elu(self.conv_left(x_left))
        # x_right = self.elu(self.conv_right(x_right))
        # x_left = self.bn1(x_left)
        # x_right= self.bn1(x_right)
        # x_left = self.avg_pool(x_left)
        # x_right = self.avg_pool(x_right)
        # x_left = x_left.view(x_left.size(0), -1)
        # x_right = x_right.view(x_right.size(0), -1)
        # x = torch.cat((x_eeg, x_left, x_right), dim=1)

        x_eeg = self.sigmoid(self.fc2(x_eeg))

        return x_eeg