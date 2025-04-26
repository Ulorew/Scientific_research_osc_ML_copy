import torch
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# SETTINGS

test_ratio = 0.2  # test size / overall size
device = "cuda"


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


# Prepare data
data_path = "../../Data/OscData2"
X_full = torch.load(data_path + "_X.pt").to(device)
y_full = torch.load(data_path + "_y.pt").to(device)
full_size = X_full.shape[0]
train_size = int(full_size * test_ratio)

_, rec_cnt, feat_cnt, specsz_orig = X_full.shape

specsz = specsz_orig // 2  # length of frequencies prefix to use from spectra
X_full = X_full[:, :, :, : specsz]

print(f"One sample consists of {rec_cnt} recordings, {feat_cnt} features, and {specsz} amplitudes.\n")

X_train = X_full[:train_size]
y_train = y_full[:train_size]
X_val = X_full[train_size:]
y_val = y_full[train_size:]

# Model Settings
spectrum_latent_size = 4
features_latent_size = 8
main_latent_size = 8
batch_size = 256

# Create Dataset
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.fcS1 = nn.Linear(in_features=specsz, out_features=16)
        self.fcS2 = nn.Linear(in_features=16, out_features=spectrum_latent_size)

        self.fcF1 = nn.Linear(in_features=feat_cnt * spectrum_latent_size, out_features=16)
        self.fcF2 = nn.Linear(in_features=16, out_features=features_latent_size)

        self.fcR1 = nn.Linear(in_features=feat_cnt * features_latent_size, out_features=16)
        self.fcR2 = nn.Linear(in_features=16, out_features=main_latent_size)

        self.rfcR1 = nn.Linear(in_features=main_latent_size, out_features=16)
        self.rfcR2 = nn.Linear(in_features=16, out_features=feat_cnt * features_latent_size)

        self.rfcF1 = nn.Linear(in_features=features_latent_size, out_features=16)
        self.rfcF2 = nn.Linear(in_features=16, out_features=feat_cnt * spectrum_latent_size)

        self.rfcS1 = nn.Linear(in_features=spectrum_latent_size, out_features=16)
        self.rfcS2 = nn.Linear(in_features=16, out_features=specsz)

    def encode(self, x):
        cur_batch_size = x.shape[0]
        x = torch.relu(self.fcS1(x))
        x = torch.relu(self.fcS2(x))

        x = x.reshape((cur_batch_size, rec_cnt, feat_cnt * spectrum_latent_size))

        x = torch.relu(self.fcF1(x))
        x = torch.relu(self.fcF2(x))

        x = x.reshape((cur_batch_size, rec_cnt * features_latent_size))

        x = torch.relu(self.fcR1(x))
        x = torch.relu(self.fcR2(x))
        return x

    def decode(self, x):
        cur_batch_size = x.shape[0]
        x = torch.relu(self.rfcR1(x))
        x = torch.relu(self.rfcR2(x))

        x = x.reshape((cur_batch_size, rec_cnt, features_latent_size))

        x = torch.relu(self.rfcF1(x))
        x = torch.relu(self.rfcF2(x))

        x = x.reshape((cur_batch_size, rec_cnt, feat_cnt, spectrum_latent_size))

        x = torch.relu(self.rfcS1(x))
        x = torch.relu(self.rfcS2(x))
        return x

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


model = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
# nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=2, cooldown=15, min_lr=1e-6)
# ExponentialLR(optimizer, gamma=0.995)
# ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, cooldown=15, min_lr=1e-6)

# Training loop
num_epochs = 300

for epoch in (range(num_epochs)):
    train_loss = 0.0
    model.train()

    for batch_data, _ in train_dataloader:
        # Convert numpy arrays to torch tensors
        batch_data = batch_data.float()

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        # original = batch_data.detach().clone()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_data.shape[0]
    train_loss /= len(train_dataloader.dataset)

    scheduler.step(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_data, _ in val_dataloader:
            batch_data = batch_data.float().to(device)
            outputs = model(batch_data)
            loss = criterion(outputs, batch_data)
            val_loss += loss.item() * batch_data.size(0)

        val_loss /= len(val_dataloader.dataset)

    print(f"Epoch [{epoch + 1}/{num_epochs}] "
          f"Train Loss: {train_loss:.6f}  "
          f"Val Loss: {val_loss:.6f}  "
          f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
