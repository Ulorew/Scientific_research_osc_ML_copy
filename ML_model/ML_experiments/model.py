import random

import torch
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# SETTINGS

test_ratio = 0.2  # test size / overall size
device = "cuda"
model_name = "simpleFC1"


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

print(f"One sample consists of {rec_cnt} recordings, {feat_cnt} features, and {specsz} / {specsz_orig} amplitudes.\n")

X_train = X_full[:train_size]
y_train = y_full[:train_size]
X_val = X_full[train_size:]
y_val = y_full[train_size:]

print(f"Train / val size: {len(X_train)} / {len(X_val)}")

# Model Settings
spectrum_latent_size = 4
recording_latent_size = 8
main_latent_size = 8
batch_size = 1024

# Create Dataset
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)

# Create DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


### MODEL

class SimpleFC1(nn.Module):
    def __init__(self):
        super(SimpleFC1, self).__init__()

        self.fcS1 = nn.Linear(in_features=specsz, out_features=16)
        self.fcS2 = nn.Linear(in_features=16, out_features=spectrum_latent_size)

        self.fcF1 = nn.Linear(in_features=feat_cnt * spectrum_latent_size, out_features=16)
        self.fcF2 = nn.Linear(in_features=16, out_features=recording_latent_size)

        self.fcR1 = nn.Linear(in_features=rec_cnt * recording_latent_size, out_features=16)
        self.fcR2 = nn.Linear(in_features=16, out_features=main_latent_size)

        self.rfcR1 = nn.Linear(in_features=main_latent_size, out_features=16)
        self.rfcR2 = nn.Linear(in_features=16, out_features=1 * recording_latent_size)

        self.rfcF1 = nn.Linear(in_features=recording_latent_size, out_features=16)
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

        x = x.reshape((cur_batch_size, rec_cnt * recording_latent_size))

        x = torch.relu(self.fcR1(x))
        x = torch.relu(self.fcR2(x))
        return x

    def decode(self, x):
        cur_batch_size = x.shape[0]
        x = torch.relu(self.rfcR1(x))
        x = torch.relu(self.rfcR2(x))

        x = x.reshape((cur_batch_size, 1, recording_latent_size))

        x = torch.relu(self.rfcF1(x))
        x = torch.relu(self.rfcF2(x))

        x = x.reshape((cur_batch_size, 1, feat_cnt, spectrum_latent_size))

        x = torch.relu(self.rfcS1(x))
        x = torch.relu(self.rfcS2(x))
        return x

    def forward(self, x):
        latent = self.encode(x)
        reconstructed = self.decode(latent)
        return reconstructed


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
    model = SimpleFC1().to(device)
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # nn.CrossEntropyLoss()

    # optimizer = optim.LBFGS(model.parameters(), lr=0.001, max_iter=500)
    optimizer = optim.Adam(model.parameters(), lr=0.0015)
    scheduler1 = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=2, cooldown=10, min_lr=1e-5,
                                   threshold=0.0001)
    scheduler2 = ExponentialLR(optimizer, gamma=0.998)
    # scheduler2 = ExponentialLR(optimizer, gamma=0.6)

    # Training loop
    num_epochs = 200

    best_val_loss = 1
    best_model = model
    early_stopping = EarlyStopping(patience=10, delta=0.000001)

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
            loss = criterion(outputs, batch_data[:, 2, :, :])  # compare only the last window

            # Backward pass and optimization
            loss.backward()

            def closure():
                optimizer.zero_grad()
                outputs_ = model(batch_data)
                loss_ = criterion(outputs_, batch_data[:, 2, :, :])
                loss_.backward()
                return loss_

            # optimizer.step(closure)
            optimizer.step()

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

        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    save_name = f"{model_name}_{round(best_val_loss, 4)}.pt"
    torch.save(best_model, f"models/{save_name}")
    print(f"Model saved as {save_name}")


# train()

### ANALYTICS

load_model_name = "simpleFC1_0.0051.pt"

model = SimpleFC1().to(device)
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


def visualize_reconstruction(x, draw_feats=[0]):
    fig, axs = plt.subplots(1, 3, figsize=(12, 8))
    x_rec = model(x.unsqueeze(0)).squeeze(0)  # treat batch dim as rec dim
    visualize_spectrum(x, axs=axs, draw_feats=draw_feats, label="orig")
    visualize_spectrum(x_rec.detach().numpy(), axs=axs[2:3], draw_feats=draw_feats, color='red', linestyle='--',
                       label="rec")


def visualize_2d():
    global X_val, y_val

    model = SimpleFC1().to(device)
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

    plt.title(f'Группировка точек модели {model_name}')
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


# if __name__ == "__main__":

#visualize_2d()
for l in range(200, 220, 5):
    visualize_reconstruction(X_val[l])
plt.show()
