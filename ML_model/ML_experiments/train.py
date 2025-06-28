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
latent_dims = [64]

# conv_size_patterns = [[], [4], [4, 4]]
conv_size_patterns = [[], [4, 4, 4], [4, 4, 4, 4]]
conv_size_scales = [1]

fc_size_patterns = [[], [64], [64, 32]]
fc_size_scales = [1]

ker_size = [7]
use_lbfgss=[True]

param_window = (1, 999999)  # допустимый диапазон кол-ва параметров
num_runs = 3
optim_iter = 100
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
        # self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            # self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.best_model_state = model.state_dict()
            self.counter = 0

    # def load_best_model(self, model):
    #     model.load_state_dict(self.best_model_state)


# TODO: Custom lr schedule 0.1 0.1 0.1 -> 1->0.995

def event_weight_corr(ys):
    return torch.ones_like(ys)
    # return torch.sqrt(ys) + 0.1


def train(model, verbose: bool = True, use_lbfgs: bool = True):
    def weightedMSE(X1, X2, weights):
        #l2 = sum(p.pow(2.0).sum() for p in model.parameters() if p.requires_grad)
        pre = (X1 - X2) ** 2
        w2 = weights.unsqueeze(1).unsqueeze(2)
        return torch.sum((pre * w2)) / (X1.numel())

    def set_lr(new_lr):
        for g in optimizer.param_groups:
            g['lr'] = new_lr

    model_name = model.model_name
    # Define loss function and optimizer
    criterion = weightedMSE
    time_start = time.time()
    # criterion=nn.L1Loss()
    # nn.CrossEntropyLoss()

    # Training loop

    best_val_loss = 1e9
    best_weights = model
    if use_lbfgs:
        num_epochs = 150
        # optimizer = optim.LBFGS(model.parameters(), lr=0.01, max_iter=optim_iter, history_size=100000)
        optimizer = optim.LBFGS(model.parameters(), lr=0.2, max_iter=optim_iter,
                                history_size=100, line_search_fn="strong_wolfe")
    else:
        num_epochs = 5000
        optimizer = optim.Adam(model.parameters(), lr=np.random.uniform(0.0001, 0.0050), weight_decay=1e-5)

    if use_lbfgs:
        scheduler1 = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=53535, cooldown=10, min_lr=1e-5,
                                       threshold=0.00001)
        scheduler2 = ExponentialLR(optimizer, gamma=np.random.uniform(1, 1))
        early_stopping = EarlyStopping(patience=10, delta=0.00001)
    else:
        scheduler1 = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=25, cooldown=100, min_lr=1e-5,
                                       threshold=0.003)
        scheduler2 = ExponentialLR(optimizer, gamma=np.random.uniform(0.998, 0.9999))
        early_stopping = EarlyStopping(patience=200, delta=0.00001)

    for epoch in (range(num_epochs)):
        train_loss = 0.0
        val_loss = 0.0

        if epoch >= 3 and use_lbfgs:
            cur_lr = optimizer.param_groups[0]['lr']
            set_lr(min(cur_lr * 1.2, 1))

        model.eval()

        with torch.no_grad():
            for batch_data, ys in val_dataloader:
                batch_data = batch_data.float().to(device)
                outputs = model(batch_data)
                loss = criterion(outputs, batch_data, ys)
                val_loss += loss.item() * batch_data.size(0)

            val_loss /= len(val_dataloader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

        model.train()

        for batch_data, ys in train_dataloader:
            batch_data = batch_data.float()

            optimizer.zero_grad()

            outputs = model(batch_data)
            loss = criterion(outputs, batch_data, ys)

            # Backward pass and optimization
            loss.backward()

            def closure():
                optimizer.zero_grad()
                outputs_ = model(batch_data)
                loss_ = criterion(outputs_, batch_data, ys)
                loss_.backward()
                return loss_

            optimizer.step(closure)
            # optimizer.step()

            train_loss += loss.item() * batch_data.shape[0]

        train_loss /= len(train_dataloader.dataset)

        scheduler1.step(val_loss)
        scheduler2.step()

        if verbose and (use_lbfgs or (epoch % 100 == 0)):
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
    save_name = f"{save_dir}/{best_val_loss:.6f}.pt"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, f"{save_name}")
    # if verbose:
    time_end = time.time()
    time_dif = (time_end - time_start)
    print(f"Model saved as {save_name}, done in {time_dif:.2f} seconds")
    return best_val_loss, best_weights


# ------------- вспомогательные функции -------------

def single_config_run(latent_dim, ker_size, conv_sizes, fc_sizes, use_lbfgs):
    """
    Для одной комбинации параметров:
      - проверяет число параметров
      - если OK, выполняет num_runs запусков train()
      - возвращает список словарей с результатами
    """

    # считаем кол-во параметров
    model = CustomRaw2(latent_dim=latent_dim,
                       conv_sizes=conv_sizes,
                       fc_sizes=fc_sizes,
                       input_channels=num_channels,
                       input_length=wsz,
                       kernel_size=ker_size).to(device)
    param_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if not (param_window[0] <= param_cnt <= param_window[1]):
        return []  # пропускаем эту конфигурацию

    best_loss = float('inf')
    losses = []

    print(f"Starting training of {model.model_name}, params: {param_cnt}")

    for run_idx in range(num_runs):
        model = CustomRaw2(latent_dim=latent_dim,
                           conv_sizes=conv_sizes,
                           fc_sizes=fc_sizes,
                           input_channels=num_channels,
                           input_length=wsz,
                           kernel_size=ker_size).to(device)
        cur_loss, _ = train(model=model, verbose=True, use_lbfgs=use_lbfgs)
        if not math.isnan(cur_loss):
            losses.append(cur_loss)
            best_loss = min(best_loss, cur_loss)

    avg_loss = sum(losses) / len(losses) if losses else float('nan')

    return {
        "ker_size": ker_size,
        "latent_dim": latent_dim,
        "conv_sizes": conv_sizes,
        "fc_sizes": fc_sizes,
        "param_cnt": param_cnt,
        "use_lbfgs":use_lbfgs,
        "best_loss": round(best_loss, 6),
        "avg_loss": round(avg_loss, 6)
    }


# ------------- main: параллельный запуск -------------

if __name__ == "__main__":
    print(f"Saving stats to {out_csv}")
    # подготавливаем CSV, пишем заголовок
    fieldnames = ["latent_dim", "ker_size", "conv_sizes", "fc_sizes", "param_cnt", "use_lbfgs", "best_loss", "avg_loss"]
    configs_pre = list(
        itertools.product(latent_dims, ker_size, conv_size_patterns, fc_size_patterns, conv_size_scales,
                          fc_size_scales, use_lbfgss))
    configs = []
    for latent_dim, ker_sz, conv_size_pattern, fc_size_pattern, conv_size_scale, fc_size_scale, use_lbfgs in configs_pre:
        conv_sizes = (np.array(conv_size_pattern) * conv_size_scale).astype(int).tolist()
        fc_sizes = (np.array(fc_size_pattern) * fc_size_scale).astype(int).tolist()
        cand = (latent_dim, ker_sz, conv_sizes, fc_sizes, use_lbfgs)
        if cand not in configs:
            configs.append(cand)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for latent_dim, ker_size, conv_sizes, fc_sizes, use_lbfgs in tqdm(configs):
            res = single_config_run(latent_dim, ker_size, conv_sizes, fc_sizes, use_lbfgs)

            writer.writerow(res)
            # можно сразу сбрасывать на диск
            f.flush()

    print(f"Done! Результаты в {out_csv}")
