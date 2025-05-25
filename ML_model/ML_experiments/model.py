import math
import random

import pandas as pd
import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import plotly.express as px
from tqdm import tqdm

# SETTINGS

val_ratio = 0.2  # val size / overall size
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
data_path = "../../Data/OscData_raw_256_3"
X_full = torch.load(data_path + "_X.pt").to(device)
y_full = torch.load(data_path + "_y.pt").to(device)
full_size = X_full.shape[0]
train_size = int(full_size * (1 - val_ratio))

train_size = int(full_size * (1 - val_ratio))

sample_cnt, rec_cnt, wsz = X_full.shape

print(f"We have {sample_cnt} samples, each one consists of {rec_cnt} recordings with length of {wsz} points.\n")

X_train = X_full[:train_size]
y_train = y_full[:train_size]
X_val = X_full[train_size:]
y_val = y_full[train_size:]

print(f"Train / val size: {len(X_train)} / {len(X_val)}")

# Create Dataset
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Create DataLoader
batch_size = 4096

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

### MODEL

# Spectrum Model Settings
spectrum_latent_size = 8
recording_latent_size = 8
main_latent_size = 32

# Raw Model Settings
latent_dim = 16
base_conv_channels = 4


# class SimpleSpecFC1(nn.Module):
#     model_name = "SimpleSpecFC1"
#
#     def __init__(self):
#         super(SimpleSpecFC1, self).__init__()
#
#         self.fcS1 = nn.Linear(in_features=specsz, out_features=16)
#         self.fcS2 = nn.Linear(in_features=16, out_features=spectrum_latent_size)
#
#         self.fcF1 = nn.Linear(in_features=feat_cnt * spectrum_latent_size, out_features=16)
#         self.fcF2 = nn.Linear(in_features=16, out_features=recording_latent_size)
#
#         self.fcR1 = nn.Linear(in_features=rec_cnt * recording_latent_size, out_features=16)
#         self.fcR2 = nn.Linear(in_features=16, out_features=main_latent_size)
#
#         self.rfcR1 = nn.Linear(in_features=main_latent_size, out_features=16)
#         self.rfcR2 = nn.Linear(in_features=16, out_features=1 * recording_latent_size)
#
#         self.rfcF1 = nn.Linear(in_features=recording_latent_size, out_features=16)
#         self.rfcF2 = nn.Linear(in_features=16, out_features=feat_cnt * spectrum_latent_size)
#
#         self.rfcS1 = nn.Linear(in_features=spectrum_latent_size, out_features=16)
#         self.rfcS2 = nn.Linear(in_features=16, out_features=specsz)
#
#     def encode(self, x):
#         cur_batch_size = x.shape[0]
#         x = torch.relu(self.fcS1(x))
#         x = torch.relu(self.fcS2(x))
#
#         x = x.reshape((cur_batch_size, rec_cnt, feat_cnt * spectrum_latent_size))
#
#         x = torch.relu(self.fcF1(x))
#         x = torch.relu(self.fcF2(x))
#
#         x = x.reshape((cur_batch_size, rec_cnt * recording_latent_size))
#
#         x = torch.relu(self.fcR1(x))
#         x = torch.relu(self.fcR2(x))
#         return x
#
#     def decode(self, x):
#         cur_batch_size = x.shape[0]
#         x = torch.relu(self.rfcR1(x))
#         x = torch.relu(self.rfcR2(x))
#
#         x = x.reshape((cur_batch_size, 1, recording_latent_size))
#
#         x = torch.relu(self.rfcF1(x))
#         x = torch.relu(self.rfcF2(x))
#
#         x = x.reshape((cur_batch_size, 1, feat_cnt, spectrum_latent_size))
#
#         x = torch.relu(self.rfcS1(x))
#         x = torch.relu(self.rfcS2(x))
#         return x
#
#     def forward(self, x):
#         latent = self.encode(x)
#         reconstructed = self.decode(latent)
#         return reconstructed  #
#
#
# class SimpleSpecFC2(nn.Module):  # single spectrum reconstruction
#     model_name = "SimpleSpecFC2"
#     activation_func = torch.nn.SiLU()
#
#     def __init__(self):
#         super(SimpleSpecFC2, self).__init__()
#
#         self.fc1 = nn.Linear(in_features=specsz, out_features=32)
#         self.fc2 = nn.Linear(in_features=32, out_features=32)
#         self.fc3 = nn.Linear(in_features=32, out_features=main_latent_size)
#
#         self.rfc1 = nn.Linear(in_features=main_latent_size, out_features=32)
#         self.rfc2 = nn.Linear(in_features=32, out_features=32)
#         self.rfc3 = nn.Linear(in_features=32, out_features=specsz)
#
#     def encode(self, x):
#         cur_batch_size = x.shape[0]
#         x = self.activation_func(self.fc1(x))
#         x = self.activation_func(self.fc2(x))
#         x = self.activation_func(self.fc3(x))
#         x = x.reshape(cur_batch_size, main_latent_size)
#         return x
#
#     def decode(self, x):
#         cur_batch_size = x.shape[0]
#         x = x.reshape(cur_batch_size, 1, 1, main_latent_size)
#         x = self.activation_func(self.rfc1(x))
#         x = self.activation_func(self.rfc2(x))
#         x = self.activation_func(self.rfc3(x))
#         return x
#
#     def forward(self, x):
#         latent = self.encode(x)
#         reconstructed = self.decode(latent)
#         return reconstructed


class SimpleRaw1(nn.Module):  # single spectrum reconstruction
    activation_func = torch.nn.SiLU()

    def __init__(self, latent_dim):
        super(SimpleRaw1, self).__init__()
        self.model_name = f"SimpleRaw1_lat{latent_dim}_conv{base_conv_channels}"

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=base_conv_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(base_conv_channels),
            nn.ReLU(True),

            nn.Conv1d(in_channels=base_conv_channels, out_channels=base_conv_channels * 2, kernel_size=5, stride=2,
                      padding=2),
            nn.BatchNorm1d(base_conv_channels * 2),
            nn.ReLU(True),

            nn.Conv1d(in_channels=base_conv_channels * 2, out_channels=base_conv_channels * 4, kernel_size=5, stride=2,
                      padding=2),
            nn.BatchNorm1d(base_conv_channels * 4),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten()

        # Compute shape after conv layers to shape the decoder
        dummy = torch.zeros(1, 3, 256)
        conv_out = self.encoder(dummy)
        self._conv_shape = conv_out.shape[1:]  # (channels, length)
        conv_flattened = conv_out.numel() // conv_out.shape[0]

        self.fc_latent = nn.Linear(conv_flattened, latent_dim)
        self.fc_expand = nn.Linear(latent_dim, conv_flattened)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, self._conv_shape),
            nn.Upsample(scale_factor=2, mode='nearest'),  # (32 -> 64)
            nn.Conv1d(base_conv_channels * 4, base_conv_channels * 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_conv_channels * 2),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (64 -> 128)
            nn.Conv1d(base_conv_channels * 2, base_conv_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_conv_channels),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (128 -> 256)
            nn.Conv1d(base_conv_channels, 3, kernel_size=5, padding=2),
            # For regression of raw signal, no activation (linear)
        )

    def encode(self, x):
        z = self.encoder(x)
        z_flat = self.flatten(z)
        latent = self.fc_latent(z_flat)
        return latent

    def decode(self, latent):
        dec_flat = self.fc_expand(latent)
        dec = self.decoder(dec_flat)
        return dec

    def forward(self, x):
        latent = self.encode(x)
        rec = self.decode(latent)
        return rec


### TRAINING

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def train():
    model = SimpleRaw1(latent_dim=latent_dim).to(device)
    model_name = model.model_name
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # criterion=nn.L1Loss()
    # nn.CrossEntropyLoss()

    optimizer = optim.LBFGS(model.parameters(), lr=0.1, max_iter=100)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler1 = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, cooldown=10, min_lr=1e-5,
                                   threshold=0.00001)
    # scheduler2 = ExponentialLR(optimizer, gamma=0.99)
    scheduler2 = ExponentialLR(optimizer, gamma=0.99)

    # Training loop
    num_epochs = 50

    best_val_loss = 1e9
    best_model = model
    early_stopping = EarlyStopping(patience=5, delta=0.0001)

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

            def closure():
                optimizer.zero_grad()
                outputs_ = model(batch_data)
                loss_ = criterion(outputs_, batch_data)
                loss_.backward()
                return loss_

            optimizer.step(closure)
            # optimizer.step()

            train_loss += loss.item() * batch_data.shape[0]

        train_loss /= len(train_dataloader.dataset)

        scheduler1.step(train_loss)
        scheduler2.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_data, _ in val_dataloader:
                batch_data = batch_data.float().to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data)
                val_loss += loss.item() * batch_data.size(0)

            val_loss /= len(val_dataloader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {train_loss:.6f}  "
              f"Val Loss: {val_loss:.6f}  "
              f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if math.isnan(train_loss):
            print("Nan loss, stopping")
            break

        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    save_name = f"{model_name}_{round(best_val_loss, 4)}.pt"
    torch.save(best_model, f"models/{save_name}")
    print(f"Model saved as {save_name}")
    return best_val_loss


def multi_train(num_runs):
    losses = []
    best_losses = []
    best_loss = 1e9
    for run in range(num_runs):
        print(f"Starting run {run + 1}")
        losses.append(train())
        best_loss = min(losses[-1], best_loss)
        best_losses.append(best_loss)
        print(f"Done {run + 1} / {num_runs}. Best loss: {best_loss}")

    losses = torch.tensor(losses)
    save_path = f"stats/losses_{round(best_loss, 5)}.pt"
    torch.save(losses, save_path)
    print(f"Saved losses to {save_path}")
    plt.plot(best_losses)
    np.fil
    plt.ylim((0, np.mean(losses)))
    plt.show()


def multi_train_analytics(filepath: str):
    losses = torch.load(filepath)
    for i in losses:
        if math.isnan(i):
            i = 0

    best_losses = []
    best_loss = 1e9
    for loss in losses:
        best_loss = min(loss, best_loss)
        best_losses.append(best_loss)

    plt.plot(best_losses)
    plt.plot(losses)
    plt.ylim((0, torch.max(losses)))
    plt.show()


#multi_train(10)
#multi_train_analytics("stats/losses_0.0038.pt")

### ANALYTICS

load_model_name = "SimpleRaw1_lat16_conv4_0.004.pt"

model = SimpleRaw1(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load("models/" + load_model_name, weights_only=True))
model.to('cpu')
model.eval()

pt_num = X_val.shape[0]

X_val = X_val.detach().cpu()
y_val = y_val.detach().cpu()


def visualize_spectrum(x, axs=None, draw_feats=[0], color="blue", linestyle='-', label: str = ""):
    if axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    for ax, rec in zip(axs, x):
        for feat in draw_feats:
            ax.plot(rec[feat], color=color, linestyle=linestyle, label=label)
        if label != "":
            ax.legend()

    return axs


def visualize_wave(x, axs=None, draw_feats=[0], color="blue", linestyle='-', label: str = ""):
    if axs is None:
        fig, axs = plt.subplots(1, len(draw_feats), figsize=(12, 8))
    for ax, rec in zip(axs, x[draw_feats]):
        ax.plot(rec, color=color, linestyle=linestyle, label=label)
        if label != "":
            ax.legend()

    return axs


def visualize_reconstruction(x, draw_feats=[0]):
    fig, axs = plt.subplots(1, len(draw_feats), figsize=(12, 8))
    if len(draw_feats) == 1:
        axs = [axs]
    for ax in axs:
        ax.set_ylim([-1.2, 1.2])
    x_rec = model(x.unsqueeze(0)).squeeze()  # treat batch dim as rec dim
    visualize_wave(x_rec.detach().numpy(), axs=axs, draw_feats=draw_feats, color='red',
                   label="rec")
    visualize_wave(x, axs=axs, draw_feats=draw_feats, label="orig", linestyle='--')
    ax.set_title(f"MSE: {((x_rec - x) ** 2).mean()}")


def visualize_2d():
    global X_val, y_val

    model = SimpleRaw1(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load("models/" + load_model_name, weights_only=True))
    model.to('cpu')
    model.eval()

    pt_num = X_val.shape[0]

    X_val = X_val.to('cpu')
    y_val = y_val.to('cpu')

    # X_vis, y_vis = random.choices(list(zip(X_val, y_val)), k=pt_num)
    X_vis, y_vis = X_val[:pt_num], y_val[:pt_num]
    fig, ax = plt.subplots()
    pts_raw = model.encode(X_vis).detach().cpu().numpy()

    pca2D = PCA(n_components=2)
    pts = pca2D.fit_transform(pts_raw)

    ptx = [pt[0] for pt in pts]
    pty = [pt[1] for pt in pts]

    plt.title(f'Группировка точек модели {load_model_name}')
    # plt.set_xlabel('X')
    # plt.set_ylabel('Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.grid(True)

    plt.scatter(ptx, pty, c=y_vis, cmap='viridis')
    plt.colorbar(label='Концентрация событий')
    plt.show()
    # for pt, y in zip(pts, y_vis):
    #     print(pt, y)


def visualize_3d_plotly():
    global X_val, y_val

    model = SimpleRaw1(latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load("models/" + load_model_name, weights_only=True))
    model.to('cpu')
    model.eval()

    pt_num = X_full.shape[0]
    X_vis, y_vis = X_full[:pt_num].to('cpu'), y_full[:pt_num].to('cpu')

    pts_raw = model.encode(X_vis).detach().cpu().numpy()
    pts_3d = PCA(n_components=3).fit_transform(pts_raw)

    df = pd.DataFrame({
        'x': pts_3d[:, 0],
        'y': pts_3d[:, 1],
        'z': pts_3d[:, 2],
        'label': y_vis.numpy()
    })

    fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='label', color_continuous_scale='Viridis',
                        title=f'3D латентное пространство {load_model_name}')
    fig.update_layout(scene=dict(
        xaxis_title='PCA1',
        yaxis_title='PCA2',
        zaxis_title='PCA3'
    ))
    fig.show()


# if __name__ == "__main__":

visualize_3d_plotly()

for l in range(60, 95, 4):
    visualize_reconstruction(X_val[l])
plt.show()
