import glob
import os
import torch
import torch.optim as optim
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Import custom modules
from dataset import ShipTrajectoryDataset, pad_collate_fn
from model import LSTMClassifier, BiLSTMClassifier, GRUClassifier, TCNWithGlobalAttention, TCNWithGlobalAttention_EfficientNet, BiLSTMClassifierWithCNN
from trajectory_extraction import extract_trajectories, calculate_class_distribution

# TensorBoard setup
writer = SummaryWriter(log_dir='logs1/TCN_GA')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(data_folder):
    """Load and preprocess data from CSV files."""
    file_paths = glob.glob(os.path.join(data_folder, '*.csv'))
    all_X, all_Y, all_MMSIs = [], [], []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        X, Y, MMSIs = extract_trajectories(file_path)
        all_X.extend(X)
        all_Y.extend(Y)
        all_MMSIs.extend(MMSIs)

    calculate_class_distribution(all_Y)

    return train_test_split(all_X, all_Y, all_MMSIs, test_size=0.2, random_state=2024)


def create_data_loaders(X_train, Y_train, X_test, Y_test):
    """Create DataLoader objects for training and testing."""
    train_dataset = ShipTrajectoryDataset(X_train, Y_train)
    test_dataset = ShipTrajectoryDataset(X_test, Y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, collate_fn=pad_collate_fn)

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, class_names=None, patience=15, delta=0.001):
    """Train the model with early stopping."""
    best_loss = float('inf')  # 初始化最小验证损失为无穷大
    epochs_no_improve = 0  # 用于记录没有改善的轮数

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_X, batch_Y, batch_mask, original_lengths in train_loader:
            batch_X, batch_Y, batch_mask = batch_X.to(device), batch_Y.to(device), batch_mask.to(device)
            lengths = batch_mask.sum(dim=1).long().cpu()

            optimizer.zero_grad()
            output = model(batch_X, lengths)
            loss = criterion(output, batch_Y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {avg_loss:.4f}')

        # Log the average loss for this epoch to TensorBoard
        writer.add_scalar('Training Loss', avg_loss, epoch + 1)

        # Evaluate model after each epoch and log metrics
        if class_names:
            avg_test_loss = evaluate_model(model, test_loader, class_names, epoch)

            # Check if the current validation loss has improved
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss  # 更新最优验证损失
                epochs_no_improve = 0  # 重置无改进的轮数
                print(f"Validation loss improved to {best_loss:.4f}.")
            else:
                epochs_no_improve += 1  # 增加无改进的轮数
                print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

            # If no improvement for 'patience' epochs, stop training
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break  # 早停

def evaluate_model(model, test_loader, class_names, epoch):
    """Evaluate the model and print metrics."""
    model.eval()
    y_true, y_pred, misclassified_samples = [], [], []
    test_loss = 0.0

    with torch.no_grad():
        for i, (batch_X, batch_Y, batch_mask, original_lengths) in enumerate(test_loader):
            batch_X, batch_Y, batch_mask = batch_X.to(device), batch_Y.to(device), batch_mask.to(device)
            lengths = batch_mask.sum(dim=1).long().cpu()

            output = model(batch_X, lengths)
            loss = criterion(output, batch_Y)  # Compute loss
            test_loss += loss.item()  # Accumulate test loss
            _, predicted = torch.max(output.data, 1)

            y_true.extend(batch_Y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute metrics
    avg_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Log metrics to TensorBoard
    writer.add_scalar('Test Loss', avg_test_loss, epoch + 1)  # Log test loss
    writer.add_scalar('Test Accuracy', accuracy, epoch + 1)
    writer.add_scalar('Test Precision', precision, epoch + 1)
    writer.add_scalar('Test Recall', recall, epoch + 1)
    writer.add_scalar('Test F1 Score', f1, epoch + 1)

    return avg_test_loss  # 返回验证集的平均损失



if __name__ == '__main__':
    # Paths and data loading
    data_folder = './data/'
    X_train, X_test, Y_train, Y_test, MMSI_train, MMSI_test = load_data(data_folder)

    # Data loaders
    train_loader, test_loader = create_data_loaders(X_train, Y_train, X_test, Y_test)

    # Model setup
    input_dim = 9
    num_classes = 4  # 类别数，根据任务
    model = TCNWithGlobalAttention(input_dim=input_dim, num_classes=num_classes).to(device)

    # Loss function with class weights
    class_counts = np.unique(Y_train, return_counts=True)[1]
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    class_names = ['Cargo', 'Fishing', 'Tanker', 'Passenger']
    train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=150, class_names=class_names)

    # Close the TensorBoard writer
    writer.close()