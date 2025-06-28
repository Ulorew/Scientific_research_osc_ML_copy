import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# SETTINGS

val_ratio = 0.2  # val size / overall size
device = "cuda"

np.random.seed(174)


# Custom Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label


num_channels = 3
wsz = 256
# Prepare data

data_path = f"../../Data/OscData_raw_{wsz}_{num_channels}"
X_full = torch.load(data_path + "_X.pt").to(device)
y_full = torch.load(data_path + "_y.pt").to(device)

full_size = X_full.shape[0]

perm = np.random.permutation(full_size)

X_full = X_full[perm]
y_full = y_full[perm]

train_size = int(full_size * (1 - val_ratio))

sample_cnt, rec_cnt, wsz = X_full.shape

X_train = X_full[:train_size]
y_train = y_full[:train_size]
X_val = X_full[train_size:]
y_val = y_full[train_size:]

# Create Dataset
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Create DataLoader
batch_size = 4096

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


def data_review():
    print(f"We have {sample_cnt} samples, each one consists of {rec_cnt} recordings with length of {wsz} points.\n")
    print(f"Train / val size: {len(X_train)} / {len(X_val)}")
