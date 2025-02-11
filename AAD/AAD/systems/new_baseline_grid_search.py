import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import warnings

from sympy.physics.units import momentum

import modules.gcn_cnn_01 as gcn
import modules.eeg_net_augmented as en
import modules.test_cnn as cnn
import tools.adj_matrix_calculator as amc
from tools.load_data import load_data
from tqdm import tqdm
import time
import itertools

# 头部设置
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use GPU: " + torch.cuda.is_available().__str__())

# 参数
batch_size = 128
num_epochs = 200
lr_init = 0.05
momentum = 0.9
adj_threshold = 40  # 多少距离以内算连接
hidden_size = 32
F2 = 50
best_model_path = r"D:\Coding\AAD\models\best_model.pth"

# 网格搜索参数
batch_size_range = [32, 64, 128, 256]
lr_range = [0.1, 0.05, 0.01]
dropout_range = [0.2, 0.3, 0.4]
momentum_range = [0.95, 0.9, 0.8]
adj_threshold_range = [30, 40, 50]
F1_range = [1, 5, 10, 25]
D_range = [1, 5, 10, 25]
param_combinations = itertools.product(F1_range, D_range)

# 加载数据
mat_file = r'D:\Coding\AAD\data_kul_processed\processed_data.mat'
train_loader, val_loader = load_data(mat_file, batch_size=batch_size)

# 计算图网络邻接矩阵
coords = amc.load_channel_coords(r'D:\Coding\AAD\data_kul_processed\64_electrodes.csv')
# 以下两种方式二选一，默认选第二个
#adj_matrix = gcn.get_topological_matrix()
adj_matrix = amc.compute_adjacency_matrix(coords, threshold=adj_threshold)
edge_index = torch.tensor(np.nonzero(adj_matrix), dtype=torch.long, device=device)
adj_matrix_tensor = torch.from_numpy(adj_matrix)

best_params = None
best_val_acc = 0.0
best_model_state = None
best_optimizer_state = None

for F1, D in param_combinations:
    print(f"Training with F1={F1}, D={D}")

    # 定义模型及训练
    # model = gcn.SIMPLE_GCN(adj_matrix_tensor).to(device)    # 85.7% emm
    model = en.EEG_NET(F1=F1, D=D, F2=F2, Chans=64, Samples=128, hiddenSize=hidden_size, A=adj_matrix_tensor, dropoutRate=0.3,
                       nb_classes=2).to(device)  # 86.4 -> 88.7% GCN -> 90.6% CWA
    # model = cnn.CNN().to(device)    # 78% 原始CNN -> 82% 时域卷积 -> 88% CWA
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training Epochs")
    max_val_acc = 0.0
    no_improve_count = 0
    rollback_count = 0
    model_state = None
    optimizer_state = None

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
            inputs = inputs[:, :64, :]
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
                inputs = inputs[:, :64, :]
                labels = labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            no_improve_count = 0
            model_state = model.state_dict()
            optimizer_state = optimizer.state_dict()
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve_count += 1

        if no_improve_count >= 20:
            if rollback_count < 2:
                print(f"\nRolling back to best model at epoch {epoch}, reducing learning rate.")
                model.load_state_dict(model_state)
                optimizer.load_state_dict(optimizer_state)
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1
                rollback_count += 1
                no_improve_count = 0
            else:
                print("\nEarly stopping as no improvement after two rollbacks.")
                break

        epoch_time = time.time() - start_time
        if epoch_time > 20:
            print(f"\nEpoch {epoch} took too long ({epoch_time:.2f}s), stopping training.")
            break

        epoch_bar.set_postfix(loss=f"{avg_loss:.4f}",
                              train_acc=f"{train_acc:.4f}",
                              val_acc=f"{val_acc:.4f}",
                              max_val_acc=f"{max_val_acc:.4f}",
                              beset_val_acc=f"{best_val_acc:.4f}",
                              training_time=f"{epoch_time:.2f}", )

    # 如果当前超参数组合的验证准确度更好，则更新最优参数
    if max_val_acc > best_val_acc:
        best_val_acc = max_val_acc
        best_params = (F1, D)
        best_model_state = model_state
        best_optimizer_state = optimizer_state
        torch.save(model.state_dict(), best_model_path)

print(f"Best hyperparameters: F1={best_params[0]}, D={best_params[1]}")
print(f"Best validation accuracy: {best_val_acc:.4f}")

def load_best_model(model, path=best_model_path):
    """加载验证集最高准确率的模型"""
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model