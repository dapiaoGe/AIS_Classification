# train_bilstm.py
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import ShipTrajectoryDataset, pad_collate_fn
from model import BiLSTMClassifier,LSTMClassifier,BiGRUClassifier,GRUClassifier
from trajectory_extraction import extract_trajectories
import os
import glob


# 数据加载
def load_data(data_folder='./data/'):
    file_paths = glob.glob(os.path.join(data_folder, '*.csv'))

    all_X, all_Y, all_MMSIs = [], [], []
    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        X, Y, MMSIs = extract_trajectories(file_path)
        all_X.extend(X)
        all_Y.extend(Y)
        all_MMSIs.extend(MMSIs)

    X_train, X_test, Y_train, Y_test, MMSI_train, MMSI_test = train_test_split(all_X, all_Y, all_MMSIs, test_size=0.2,random_state=2024)

    train_dataset = ShipTrajectoryDataset(X_train, Y_train)
    test_dataset = ShipTrajectoryDataset(X_test, Y_test)

    # 统计每个类别的样本数量
    class_counts = np.unique(Y_train, return_counts=True)[1]
    # 计算每个类别的权重
    class_weights = 1.0 / class_counts
    # 归一化权重以确保它们的和为1
    class_weights = class_weights / class_weights.sum()
    # 转换为PyTorch张量
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    return train_dataset, test_dataset, weights


# 模型训练与评估
def train_and_evaluate_model(hidden_dim, num_layers, dropout, learning_rate):
    train_dataset, test_dataset, weights = load_data()  # 自定义的数据加载函数

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)

    model = GRUClassifier(input_dim=9, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout, num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(100):  # 使用合适的epoch以加快超参数搜索
        model.train()
        running_loss = 0.0
        for batch_X, batch_Y, batch_mask, original_lengths in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            output = model(batch_X, original_lengths)
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')

    # 评估模型
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_Y, _, original_lengths in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X, original_lengths)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_Y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 计算加权 F1 分数
    return accuracy, f1


# 超参数搜索
def hyperparameter_search():
    # hidden_dims = [64, 128, 256]
    # num_layers_options = [1, 2, 3]
    # dropout_rates = [0.1, 0.2, 0.5]
    # learning_rates = [0.0001,0.0005, 0.001]

    hidden_dims = [64, 128]
    num_layers_options = [1, 2, 3]
    dropout_rates = [0.1, 0.2]
    learning_rates = [0.0001, 0.0005, 0.001]

    results = []

    for hidden_dim, num_layers, dropout, learning_rate in itertools.product(hidden_dims, num_layers_options,dropout_rates, learning_rates):
        print(f'Testing: hidden_dim={hidden_dim}, num_layers={num_layers}, dropout={dropout}, learning_rate={learning_rate}')
        accuracy, f1 = train_and_evaluate_model(hidden_dim, num_layers, dropout, learning_rate)
        results.append({
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': learning_rate,
            'accuracy': accuracy,
            'f1': f1
        })

    best_result = max(results, key=lambda x: x['f1'])              # accuracy适用于类别较为均衡的情况,f1适用于精确率和召回率的调和平均值，适用于不平衡数据集。
    print("Best Hyperparameters:")
    print(best_result)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparameter_search()