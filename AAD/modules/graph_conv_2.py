import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

class SpectralGraphConv(nn.Module):
    def __init__(self, out_channels, A_base, inter_mask):
        """
        Args:
            out_channels: 输出通道数
            A_base: 固定邻接部分 (Tensor, shape [N, N])
            inter_mask: 区域间连接布尔掩码 (Tensor[bool], shape [N, N])
        """
        super(SpectralGraphConv, self).__init__()
        self.out_channels = out_channels
        self.A_base = A_base  # 区域内连接（不可训练）
        self.inter_mask = inter_mask  # 区域间连接位置

        self.N = A_base.shape[0]

        # 可训练的区域间连接，初始化为0.5
        W_init = torch.full((self.N, self.N), 0.5)
        W_init = W_init * inter_mask  # 非区域间部分为0
        self.W_inter = nn.Parameter(W_init)

        # 初始化时构造 U 和 Lambda
        with torch.no_grad():
            A_full = self.get_full_adj()
            self.update_spectral_basis(A_full)

        # 可学习的谱滤波参数 theta
        self.theta = nn.Parameter(torch.randn(self.N, self.out_channels))

    def get_full_adj(self):
        # 获取对称的区域间连接
        W_sym = (self.W_inter + self.W_inter.t()) / 2.0
        # 将 inter_mask 移动到 W_sym 所在设备
        inter_mask = self.inter_mask.to(W_sym.device)
        # 应用掩码，仅保留区域间连接
        W_sym = W_sym * inter_mask
        # 添加固定区域内连接
        A = self.A_base.to(W_sym.device) + W_sym
        return A

    def update_spectral_basis(self, A):
        # 拉普拉斯矩阵：L = D - A
        D = torch.diag(A.sum(dim=1))
        L = D - A
        Lambda, U = torch.linalg.eigh(L)
        self.register_buffer('U', U.float())
        self.register_buffer('Lambda', Lambda.float())

    def forward(self, x):
        """
        x: 输入 (B, C, N, T)
        返回: (B, out_channels, N, T)
        """
        B, C, N, T = x.shape

        # 动态更新 A 和谱分解（也可以只在训练开始时预处理一次）
        with torch.no_grad():
            A_full = self.get_full_adj()
            self.update_spectral_basis(A_full)

        # ① 图傅里叶变换
        x_hat = torch.einsum('ij,bcjt->bcit', self.U.t(), x)

        # ② 谱域滤波器作用
        x_hat = x_hat.unsqueeze(3)  # (B, C, N, 1, T)
        theta = self.theta.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1,1,N,out,1)
        x_hat_filtered = x_hat * theta

        # ③ 逆图傅里叶变换
        y = torch.einsum('in,bcnot->bcoit', self.U, x_hat_filtered)

        # ④ 累加输入通道
        y = y.sum(dim=1)  # (B, out_channels, N, T)
        return y
