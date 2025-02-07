import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import warnings
import modules.gcn_cnn_01 as gcn
import tools.adj_matrix_calculator as amc
from tools.load_data import load_data
from tqdm import tqdm

# 头部设置
warnings.filterwarnings("ignore", category=DeprecationWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use GPU: " + torch.cuda.is_available().__str__())

# params
batch_size = 128
num_epochs = 200
lr_init = 0.05
momentum = 0.9
adj_threshold = 40  # 多少距离以内算连接
best_model_path = r"D:\Coding\AAD\models\best_model.pth"

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

# 定义模型及训练
model = gcn.SIMPLE_GCN(adj_matrix_tensor).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=momentum)
epoch_bar = tqdm(range(1, num_epochs + 1), desc="Training Epochs")
max_val_acc = 0.0

# 训练与验证
for epoch in epoch_bar:
    # --- 训练 ---
    model.train()
    running_loss = 0.0
    train_preds = []
    train_labels = []

    for inputs, labels in train_loader:
        inputs = inputs.to(device)  # shape: (batch, 1, 64, 128)
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
            labels = labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_acc = accuracy_score(val_labels, val_preds)
    if val_acc > max_val_acc:
        max_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

    epoch_bar.set_postfix(loss=f"{avg_loss:.4f}",
                          train_acc=f"{train_acc:.4f}",
                          val_acc=f"{val_acc:.4f}",
                          max_val_acc=f"{max_val_acc:.4f}",)

def load_best_model(model, path=best_model_path):
    """加载验证集最高准确率的模型"""
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()
    return model