import csv
import itertools
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ML_model.ML_experiments.dataset import *
from ML_model.ML_experiments.model import CustomRaw2

data_review()
# ------------- параметры эксперимента -------------
latent_dims = [16]
# conv_size_patterns = [[4, 8, 4], [4, 4, 4]]
# fc_size_patterns = [[], [1], [1, 1], [2, 1]]
# conv_size_scales = [1, 2]
# fc_size_scales = [1, 2]
conv_size_patterns = [[4, 4, 4]]
fc_size_patterns = [[]]
conv_size_scales = [1]
fc_size_scales = [1]

param_window = (1, 999999)  # допустимый диапазон кол-ва параметров
num_runs = 10
kernel_size = 5
device = 'cuda'  # или 'cpu'

timestamp = time.strftime("%Y%m%d_%H%M%S")
out_csv = f"stats/stats_{timestamp}.csv"


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


# TODO: Custom lr schedule 0.1 0.1 0.1 -> 1->0.995


def train(model, verbose: bool = True):
    model_name = model.model_name
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    time_start = time.time()
    # criterion=nn.L1Loss()
    # nn.CrossEntropyLoss()

    optimizer = optim.LBFGS(model.parameters(), lr=np.random.uniform(0.05, 0.3), max_iter=100)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler1 = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=53535, cooldown=10, min_lr=1e-5,
                                   threshold=0.00001)
    # scheduler2 = ExponentialLR(optimizer, gamma=0.99)
    scheduler2 = ExponentialLR(optimizer, gamma=np.random.uniform(0.96, 0.98))

    # Training loop
    num_epochs = 50

    best_val_loss = 1e9
    best_weights = model
    early_stopping = EarlyStopping(patience=5, delta=0.00005)

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
            best_weights = model.state_dict().copy()

        if verbose:
            print(f"Epoch [{epoch + 1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.6f}  "
                  f"Val Loss: {val_loss:.6f}  "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            if verbose:
                print("Early stopping")
            break
        if math.isnan(train_loss):
            if verbose:
                print("Nan loss, stopping")
            break

        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    save_dir = f"models/{model_name.replace('/', '_')}".replace("_", '/')
    save_name = f"{save_dir}/{round(best_val_loss, 4)}.pt"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, f"{save_name}")
    # if verbose:
    time_end = time.time()
    time_dif = (time_end - time_start)
    print(f"Model saved as {save_name}, done in {time_dif:.2f} seconds")
    return best_val_loss, best_weights


# ------------- вспомогательные функции -------------

def single_config_run(latent_dim, conv_size_pattern, fc_size_pattern, conv_size_scale, fc_size_scale):
    """
    Для одной комбинации параметров:
      - проверяет число параметров
      - если OK, выполняет num_runs запусков train()
      - возвращает список словарей с результатами
    """
    conv_sizes = (np.array(conv_size_pattern) * conv_size_scale).astype(int).tolist()
    fc_sizes = (np.array(fc_size_pattern) * latent_dim * fc_size_scale).astype(int).tolist()

    # считаем кол-во параметров
    model = CustomRaw2(latent_dim=latent_dim,
                       conv_sizes=conv_sizes,
                       fc_sizes=fc_sizes,
                       kernel_size=kernel_size).to(device)
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not (param_window[0] <= param_cnt <= param_window[1]):
        return []  # пропускаем эту конфигурацию

    best_loss = float('inf')
    losses = []

    for run_idx in range(num_runs):
        model = CustomRaw2(latent_dim=latent_dim,
                           conv_sizes=conv_sizes,
                           fc_sizes=fc_sizes,
                           kernel_size=kernel_size).to(device)
        cur_loss, _ = train(model=model, verbose=False)
        if not math.isnan(cur_loss):
            losses.append(cur_loss)
            best_loss = min(best_loss, cur_loss)

    avg_loss = sum(losses) / len(losses) if losses else float('nan')

    return {
        "latent_dim": latent_dim,
        "conv_sizes": conv_sizes,
        "fc_sizes": fc_sizes,
        "param_cnt": param_cnt,
        "best_loss": best_loss,
        "avg_loss": avg_loss
    }


# ------------- main: параллельный запуск -------------

if __name__ == "__main__":
    print(f"Saving stats to {out_csv}")
    # подготавливаем CSV, пишем заголовок
    fieldnames = ["latent_dim", "conv_sizes", "fc_sizes", "param_cnt", "best_loss", "avg_loss"]
    configs = list(
        itertools.product(latent_dims, conv_size_patterns, fc_size_patterns, conv_size_scales, fc_size_scales))
    configs=[conf for conf in configs if (conf[2]!=[] or conf[3]==fc_size_scales[0])]

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for latent_dim, conv_pattern, fc_pattern, conv_scale, fc_scale in tqdm(configs):
            res = single_config_run(latent_dim, conv_pattern, fc_pattern, conv_scale, fc_scale)

            writer.writerow(res)
            # можно сразу сбрасывать на диск
            f.flush()

    print(f"Done! Результаты в {out_csv}")
