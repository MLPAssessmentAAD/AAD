import scipy.io
import numpy as np
import os
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, Subset

def load_data(filepath, batch_size=128):
    with h5py.File(filepath, 'r') as f:
        # 读取训练数据
        allTrain = f['allTrain']
        train_x = np.array(allTrain['x'])  # shape: (N_train, 64, 128)
        train_y = np.array(allTrain['y']).squeeze() - 1  # shape: (N_train,)

        # 读取验证数据
        allVal = f['allVal']
        val_x = np.array(allVal['x'])  # shape: (10944, 64, 128)
        val_y = np.array(allVal['y']).squeeze() - 1  # shape: (10944,)

    # 从训练集中抽取 4000 个样本作为测试集
    num_train = len(train_x)
    test_size = 4000
    indices = np.random.permutation(num_train)
    test_indices, train_indices = indices[:test_size], indices[test_size:]

    test_x, test_y = train_x[test_indices], train_y[test_indices]
    train_x, train_y = train_x[train_indices], train_y[train_indices]

    # 将验证集对半分成新的验证集和测试集
    num_val = len(val_x)
    val_size = num_val // 2
    val_x, extra_test_x = val_x[:val_size], val_x[val_size:]
    val_y, extra_test_y = val_y[:val_size], val_y[val_size:]

    # 合并两个测试集
    test_x = np.concatenate([test_x, extra_test_x], axis=0)
    test_y = np.concatenate([test_y, extra_test_y], axis=0)

    # 转换为 PyTorch 张量
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)

    val_x_tensor = torch.tensor(val_x, dtype=torch.float32)
    val_y_tensor = torch.tensor(val_y, dtype=torch.long)

    test_x_tensor = torch.tensor(test_x, dtype=torch.float32)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long)

    # 创建 TensorDataset
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
