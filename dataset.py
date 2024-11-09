import torch
from torch.utils.data import Dataset


# AIS数据加载器类
class ShipTrajectoryDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        trajectory = self.X[idx]
        label = self.Y[idx]
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return trajectory, label

def pad_collate_fn(batch):
    fixed_len = 1440
    trajectories, labels = zip(*batch)
    padded_trajectories = torch.zeros((len(trajectories), fixed_len, trajectories[0].size(1)))
    attention_mask = torch.zeros((len(trajectories), fixed_len))
    original_lengths = []

    for i, traj in enumerate(trajectories):
        end = min(len(traj), fixed_len)
        padded_trajectories[i, :end, :] = traj[:end]
        attention_mask[i, :end] = 1
        original_lengths.append(len(traj))

    labels = torch.tensor(labels, dtype=torch.long)
    original_lengths = torch.tensor(original_lengths, dtype=torch.long)

    return padded_trajectories, labels, attention_mask, original_lengths



# AIS数据和图像数据加载器类
class ShipTrajectoryImageDataset(Dataset):
    def __init__(self, X_trajectory, X_image, Y):
        self.X_trajectory = X_trajectory
        #print(f"Original image shape: {X_image[0].shape}")  # 输出原始图像的 shape
        self.X_image = X_image
        self.Y = Y

    def __len__(self):
        return len(self.X_trajectory)

    def __getitem__(self, idx):
        trajectory = self.X_trajectory[idx]
        image = self.X_image[idx]
        label = self.Y[idx]
        trajectory = torch.tensor(trajectory, dtype=torch.float32)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # 转换为 (C, H, W) 格式
        label = torch.tensor(label, dtype=torch.long)
        return trajectory, image, label

def pad_collate_fn_add_image(batch):
    fixed_len = 1440
    trajectories, images, labels = zip(*batch)  # 加载图像数据
    padded_trajectories = torch.zeros((len(trajectories), fixed_len, trajectories[0].size(1)))
    attention_mask = torch.zeros((len(trajectories), fixed_len))
    original_lengths = []

    for i, traj in enumerate(trajectories):
        end = min(len(traj), fixed_len)
        padded_trajectories[i, :end, :] = traj[:end]
        attention_mask[i, :end] = 1
        original_lengths.append(len(traj))  # 保存原始轨迹长度

    labels = torch.tensor(labels, dtype=torch.long)
    original_lengths = torch.tensor(original_lengths, dtype=torch.long)

    images = torch.stack(images)  # 图像数据堆叠

    return padded_trajectories, images, labels, attention_mask, original_lengths

