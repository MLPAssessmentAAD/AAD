import scipy.io
import numpy as np
import os
import torch
import h5py
from torch.utils.data import Dataset, DataLoader

def load_data(filepath, batch_size=128):
    with h5py.File(filepath, 'r') as f:
        # 读取训练数据
        allTrain = f['allTrain']
        train_x = np.array(allTrain['x'])  # shape: (N_train, 64, 128)
        train_y = np.array(allTrain['y']).squeeze() - 1  # shape: (N_train,)
        print(train_x.shape)

        # 读取验证数据
        allVal = f['allVal']
        val_x = np.array(allVal['x'])  # shape: (10944, 64, 128)
        val_y = np.array(allVal['y']).squeeze() - 1  # shape: (10944,)

    # 将 NumPy 数组转换为 PyTorch 张量
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)

    val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
    val_y_tensor = torch.tensor(val_y, dtype=torch.long)

    # 创建 TensorDataset
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader