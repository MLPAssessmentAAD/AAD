import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralGraphConv(nn.Module):
    def __init__(self, out_channels, A):
        """
        Args:
            out_channels
            A: 邻接矩阵，形状 (N, N) ，这里 N=64
        """
        super(SpectralGraphConv, self).__init__()
        self.out_channels = out_channels
        self.A = A  # 邻接矩阵, torch.Tensor, shape: (N, N)
        self.N = A.shape[0]  # 节点数（64）

        # 1. 计算图的拉普拉斯矩阵：L = D - A
        D = torch.diag(A.sum(dim=1))
        L = D - A

        # 2. 对拉普拉斯矩阵做特征分解：L = U Λ U^T
        #    使用 torch.linalg.eigh 因为 L 是对称矩阵
        Lambda, U = torch.linalg.eigh(L)  # U: (N, N), Lambda: (N,)

        # 将 U 和 Lambda 注册为 buffer（不会更新，也不参与梯度计算）
        self.register_buffer('U', U)
        self.register_buffer('Lambda', Lambda)

        # **确保数据类型一致**
        self.register_buffer('U', U.float())
        self.register_buffer('Lambda', Lambda.float())

        # 3. 定义谱域滤波器参数 g_θ(Λ)
        #    这里将 theta 定义为形状 (N, out_channels) 的矩阵，
        #    表示对于每个频率分量（共 N 个），针对每个输出通道都有一个可学习的缩放因子
        self.theta = nn.Parameter(torch.randn(self.N, self.out_channels))

    def forward(self, x):
        """
        Args:
            x: 输入信号，形状 (batch_size, C, N, T=Time)
        Returns:
            y: 输出信号，形状 (batch_size, C, N, T=Time)
        """
        B, C, N, T = x.shape  # B: batch_size, C: in_channels, N: 节点数, T: 时间步数
        # ────────────── ① 图傅里叶变换 ──────────────
        # 将节点域信号转换到频域，即对每个 (B, C, T) 上的节点信号应用 U^T
        # x 的形状为 (B, C, N, T)，这里使用 einsum 完成矩阵乘法：
        #   对于每个 b, c, t，计算 x_hat[b, c, :, t] = U^T @ x[b, c, :, t]
        # 得到 x_hat 的形状为 (B, C, N, T)
        x_hat = torch.einsum('ij, bcjt -> bcit', self.U.t(), x)

        # ────────────── ② 谱域滤波器作用 ──────────────
        # 原来 theta 的形状为 (N,)，现在改为 (N, out_channels)
        # 对于每个输入通道，都要针对每个输出通道进行滤波，即对每个频率分量 n 乘以 theta[n, out_channel]
        # 具体做法：
        #   1. 将 x_hat 扩展一维，变为 (B, C, N, 1, T)
        #   2. 将 theta 扩展为 (1, 1, N, out_channels, 1)
        #   3. 按照节点（频率）维度对应相乘，得到 x_hat_filtered 的形状 (B, C, N, out_channels, T)
        x_hat = x_hat.unsqueeze(3)                           # (B, C, N, 1, T)
        theta = self.theta.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, N, out_channels, 1)
        x_hat_filtered = x_hat * theta                       # (B, C, N, out_channels, T)

        # ────────────── ③ 逆图傅里叶变换 ──────────────
        # 将过滤后的频域信号转换回节点域：
        # 对于每个输入通道和每个输出通道，利用 U 做逆变换，即：
        #   对于每个 b, c, out, t，计算 y[b, c, out, :, t] = U @ x_hat_filtered[b, c, :, out, t]
        # 这里使用 einsum，其中 U 的形状为 (N, N)，x_hat_filtered 的形状为 (B, C, N, out_channels, T)
        # 计算后 y 的形状为 (B, C, out_channels, N, T)
        y = torch.einsum('in, bcnot -> bcoit', self.U, x_hat_filtered)

        # ────────────── ④ 对所有输入通道累加 ──────────────
        # 将 in_channels（C）维度的结果累加，得到输出形状 (B, out_channels, N, T)
        y = y.sum(dim=1)
        return y
