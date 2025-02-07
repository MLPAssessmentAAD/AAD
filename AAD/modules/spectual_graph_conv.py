import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralGraphConv(nn.Module):
    def __init__(self, in_features, out_features, A):
        """
        Args:
            in_features: 输入特征维度（例如128）
            out_features: 输出特征维度（例如你希望经过图卷积后的特征数）
            A: 邻接矩阵，形状 (N, N) ，这里 N=64
        """
        super(SpectralGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
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
        #    这里我们定义一个形状为 (N,) 的向量，每个元素对应一个谱分量的缩放因子，
        #    该参数对所有输入通道均相同，也可以扩展为 (N, out_features) 来实现多个滤波器。
        self.theta = nn.Parameter(torch.randn(self.N))

    def forward(self, x):
        """
        Args:
            x: 输入信号，形状 (batch_size, N, in_features)
        Returns:
            y: 输出信号，形状 (batch_size, N, out_features)
        """
        # ① 图傅里叶变换: 将节点信号 x 从节点域转换到频域
        #    x 的形状为 (batch, N, in_features)，U 的形状为 (N, N)。
        #    这里使用 U^T * x。注意：torch.bmm 为批量矩阵乘法。
        x_hat = torch.bmm(self.U.t().expand(x.shape[0], -1, -1), x)  # (batch, N, in_features)

        # ② 应用谱域滤波器：对每个频率分量乘以对应的可学习参数
        #    self.theta 的形状为 (N,)，扩展后形状 (1, N, 1) 与 x_hat 按元素相乘
        x_hat_filtered = x_hat * self.theta.view(1, self.N, 1)

        # ③ 逆图傅里叶变换：将过滤后的信号转换回节点域
        y = torch.matmul(self.U, x_hat_filtered)  # 形状: (batch, N, in_features)

        return y

