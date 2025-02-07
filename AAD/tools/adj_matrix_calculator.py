import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# 读取通道坐标
def load_channel_coords(file_path):
    df = pd.read_csv(file_path, header=None, names=["Channel", "X", "Y", "Z"])
    coords = df[["X", "Y", "Z"]].values
    return coords

# 根据信道拓扑连接计算邻接矩阵
def get_topological_matrix():
    # 64 通道 EEG 电极标签（国际 10-20 扩展系统）
    electrode_names = [
        "Fp1", "Fpz", "Fp2", "AF3", "AF4", "AF7", "AF8",
        "F1", "Fz", "F2", "F3", "F4", "F5", "F6", "F7", "F8",
        "FT7", "FT8", "FC3", "FC4", "FC5", "FC6", "FC1", "FC2",
        "C1", "Cz", "C2", "C3", "C4", "C5", "C6", "T7", "T8",
        "TP7", "TP8", "CP3", "CP4", "CP5", "CP6", "CP1", "CP2",
        "P1", "Pz", "P2", "P3", "P4", "P5", "P6", "P7", "P8",
        "PO3", "PO4", "PO7", "PO8", "POz", "O1", "O2", "Oz",
    ]

    # 64x64 的邻接矩阵（初始化为 0）
    adj_matrix = np.zeros((64, 64))

    # 定义拓扑连接关系
    connections = [
        ("Fp1", "Fpz"), ("Fp1", "AF3"), ("Fp1", "AF7"),
        ("Fp2", "Fpz"), ("Fp2", "AF4"), ("Fp2", "AF8"),
        ("Fpz", "AF3"), ("Fpz", "AF4"),
        ("AF3", "AF7"), ("AF3", "F3"), ("AF4", "AF8"), ("AF4", "F4"),
        ("F3", "Fz"), ("F3", "F5"), ("F4", "Fz"), ("F4", "F6"),
        ("F5", "F7"), ("F6", "F8"),
        ("F7", "FT7"), ("F8", "FT8"),
        ("FT7", "FC5"), ("FT8", "FC6"),
        ("FC5", "FC3"), ("FC6", "FC4"),
        ("FC3", "FC1"), ("FC4", "FC2"),
        ("FC1", "Fz"), ("FC2", "Fz"),
        ("C3", "Cz"), ("C4", "Cz"),
        ("C3", "C1"), ("C4", "C2"),
        ("C3", "CP3"), ("C4", "CP4"),
        ("CP3", "CP1"), ("CP4", "CP2"),
        ("CP3", "P3"), ("CP4", "P4"),
        ("P3", "Pz"), ("P4", "Pz"),
        ("P3", "P1"), ("P4", "P2"),
        ("P3", "PO3"), ("P4", "PO4"),
        ("PO3", "POz"), ("PO4", "POz"),
        ("PO3", "O1"), ("PO4", "O2"),
        ("O1", "Oz"), ("O2", "Oz"),
        ("T7", "TP7"), ("T8", "TP8"),
        ("TP7", "CP5"), ("TP8", "CP6"),
        ("CP5", "CP3"), ("CP6", "CP4"),
        ("CP5", "P5"), ("CP6", "P6"),
        ("P5", "P7"), ("P6", "P8"),
        ("P7", "PO7"), ("P8", "PO8"),
        ("PO7", "POz"), ("PO8", "POz"),
        ("PO7", "O1"), ("PO8", "O2")
    ]

    # 将连接关系转换为邻接矩阵
    electrode_index = {name: i for i, name in enumerate(electrode_names)}

    for e1, e2 in connections:
        i, j = electrode_index[e1], electrode_index[e2]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # 无向图，双向连接

    # 返回邻接矩阵
    return adj_matrix

# 计算邻接矩阵（使用欧氏距离）
def compute_adjacency_matrix(coords, threshold=5.0):
    n = coords.shape[0]
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < threshold:  # 你可以调整阈值来控制连接的密度
                adj_matrix[i, j] = adj_matrix[j, i] = 1
    return adj_matrix