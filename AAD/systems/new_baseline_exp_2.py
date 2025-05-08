import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import warnings
import pandas as pd

from sympy.physics.units import momentum

import modules.gcn_cnn_03 as gcn
import modules.eeg_net_augmented as en
import modules.eeg_net as ten
import modules.test_cnn as cnn
import tools.adj_matrix_calculator as amc
from tools.load_data import load_data
from tqdm import tqdm
import time

# —————————————————————————————— S T A R T ———————————————————————————————————
# 构建邻接矩阵
csv_path = r'D:\Coding\AAD\data_kul_processed\64_electrodes.csv'
df_coords = pd.read_csv(csv_path, header=None, names=['ch','x','y','z'])
full_list = df_coords['ch'].str.replace(" (T3)", "", regex=False).tolist()

# 区域划分
auditory_L = ['T7', 'FT7', 'TP7']
auditory_R = ['T8', 'FT8', 'TP8']
language = ['Pz', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6',
            'POz', 'PO3', 'PO4', 'PO7', 'PO8', 'P7', 'P8', 'P9', 'P10']
attention = ['Fpz', 'Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
             'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'Fz']

region_union = list({ch for ch in auditory_L + auditory_R + language + attention})
selected_channels = [ch for ch in full_list if ch in region_union]
selected_indices = [full_list.index(ch) for ch in selected_channels]
N = len(selected_channels)

# 初始化矩阵
A_base = np.zeros((N, N), dtype=float)
inter_mask = np.zeros((N, N), dtype=bool)

def set_bidirectional(mat, i, j, value=1.0):
    mat[i, j] = value
    mat[j, i] = value

# 区域内强连接
for group in [auditory_L, auditory_R, language, attention]:
    for i in group:
        for j in group:
            if i != j and i in selected_channels and j in selected_channels:
                idx_i = selected_channels.index(i)
                idx_j = selected_channels.index(j)
                set_bidirectional(A_base, idx_i, idx_j, 1.0)

# 区域间弱连接掩码（初始化为0.5后可训练）
def mark_inter(group1, group2):
    for i in group1:
        for j in group2:
            if i != j and i in selected_channels and j in selected_channels:
                idx_i = selected_channels.index(i)
                idx_j = selected_channels.index(j)
                inter_mask[idx_i, idx_j] = True
                inter_mask[idx_j, idx_i] = True

mark_inter(auditory_L, language)
mark_inter(auditory_R, language)
mark_inter(language, attention)

# 转为Tensor
A_base_tensor = torch.from_numpy(A_base).float()
inter_mask_tensor = torch.from_numpy(inter_mask)

# —————————————————————————————— H E A D E R —————————————————————————————————
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use GPU:", torch.cuda.is_available())

# 参数设置
batch_size = 128
num_epochs = 200
lr_init = 0.05
momentum = 0.9
best_model_path = r"D:\Coding\AAD\models\best_model.pth"

# 数据加载
data_path = r'D:\Coding\AAD\data_kul_processed\processed_data.mat'
train_loader, val_loader = load_data(data_path, batch_size=batch_size)

# 定义模型：使用39维通道邻接矩阵
model = gcn.COMPLEX_GCN(5, N, A_base_tensor, inter_mask_tensor).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training Epochs")
max_val_acc = 0.0
model_state = None

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(count_parameters(model))

# 训练与验证
for epoch in epoch_bar:
    # --- 训练 ---
    model.train()
    running_loss = 0.0
    train_preds = []
    train_labels = []
    start_time = time.time()

    for inputs, labels in train_loader:
        inputs = inputs.to(device)  # shape: (batch, 1, 64, 128)
        inputs = inputs[:, selected_indices, :]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # 输出 shape: (batch, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(train_labels, train_preds)
    avg_loss = running_loss / len(train_loader.dataset)

    # --- 验证 ---
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            inputs = inputs[:, selected_indices, :]
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_acc = accuracy_score(val_labels, val_preds)
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        model_state = model.state_dict()
        torch.save(model.state_dict(), best_model_path)

    epoch_time = time.time() - start_time

    epoch_bar.set_postfix(loss=f"{avg_loss:.4f}",
                            train_acc=f"{train_acc:.4f}",
                            val_acc=f"{val_acc:.4f}",
                            max_val_acc=f"{max_val_acc:.4f}",
                            training_time=f"{epoch_time:.2f}",)

def load_best_model(model, path=best_model_path):
    """加载验证集最高准确率的模型"""
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model