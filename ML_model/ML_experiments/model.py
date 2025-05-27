import math

import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


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





class CustomRaw1(nn.Module):  # single spectrum reconstruction
    def __init__(self, latent_dim, conv_sizes=[8, 16, 32]):
        super(CustomRaw1, self).__init__()
        self.model_name = f"CustomRaw1_lat{latent_dim}_conv{conv_sizes}"

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=conv_sizes[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(conv_sizes[0]),
            nn.ReLU(True),

            nn.Conv1d(in_channels=conv_sizes[0], out_channels=conv_sizes[1], kernel_size=5, stride=2,
                      padding=2),
            nn.BatchNorm1d(conv_sizes[1]),
            nn.ReLU(True),

            nn.Conv1d(in_channels=conv_sizes[1], out_channels=conv_sizes[2], kernel_size=5, stride=2,
                      padding=2),
            nn.BatchNorm1d(conv_sizes[2]),
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
            nn.Conv1d(conv_sizes[2], conv_sizes[1], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_sizes[1]),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (64 -> 128)
            nn.Conv1d(conv_sizes[1], conv_sizes[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_sizes[0]),
            nn.ReLU(True),

            nn.Upsample(scale_factor=2, mode='nearest'),  # (128 -> 256)
            nn.Conv1d(conv_sizes[0], 3, kernel_size=5, padding=2),
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


class CustomRaw2(nn.Module):  # single spectrum reconstruction with dynamic layers
    def __init__(self,
                 latent_dim: int,
                 conv_sizes: list = [8, 16, 32],
                 fc_sizes: list = [],
                 input_channels: int = 3,
                 input_length: int = 256,
                 kernel_size: int = 5):
        super(CustomRaw2, self).__init__()
        self.model_name = f"CustomRaw2_lat{latent_dim}_conv{conv_sizes}_fc{fc_sizes}"

        # --- Build Encoder conv layers dynamically ---
        conv_layers = []
        in_ch = input_channels
        for out_ch in conv_sizes:
            conv_layers += [
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                          kernel_size=kernel_size, stride=2, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(True)
            ]
            in_ch = out_ch
        self.encoder_conv = nn.Sequential(*conv_layers)

        # Flatten and infer conv output shape
        self.flatten = nn.Flatten()
        dummy = torch.zeros(1, input_channels, input_length)
        conv_out = self.encoder_conv(dummy)
        self._conv_shape = conv_out.shape[1:]  # (channels, length)
        conv_flattened = conv_out.numel() // conv_out.shape[0]

        # --- Build Encoder FC layers dynamically ---
        enc_dims = [conv_flattened] + fc_sizes + [latent_dim]
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers.append(nn.Linear(enc_dims[i], enc_dims[i + 1]))
            # add activation for all but last
            if i < len(enc_dims) - 2:
                enc_layers.append(nn.ReLU(True))
        self.encoder_fc = nn.Sequential(*enc_layers)

        # --- Build Decoder FC layers dynamically ---
        dec_dims = [latent_dim] + list(reversed(fc_sizes)) + [conv_flattened]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU(True))
        self.decoder_fc = nn.Sequential(*dec_layers)

        # --- Build Decoder conv layers dynamically ---
        deconv_layers = [nn.Unflatten(1, self._conv_shape)]
        in_ch = self._conv_shape[0]
        rev_conv = list(reversed(conv_sizes))
        for idx, out_ch in enumerate(list(reversed([input_channels] + rev_conv[:-1]))):
            deconv_layers += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                          kernel_size=kernel_size, padding=kernel_size // 2)
            ]
            # add activation+bn except last
            if idx < len(rev_conv) - 1:
                deconv_layers += [
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(True)
                ]
            in_ch = out_ch
        self.decoder_conv = nn.Sequential(*deconv_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder_conv(x)
        z_flat = self.flatten(z)
        latent = self.encoder_fc(z_flat)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        dec = self.decoder_fc(latent)
        dec = self.decoder_conv(dec)
        return dec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        rec = self.decode(latent)
        return rec

    def weights_init(self):
        if isinstance(self, nn.Conv1d):
            init.kaiming_normal_(self.weight, nonlinearity='relu')
            if self.bias is not None:
                init.zeros_(self.bias)
        elif isinstance(self, nn.Linear):
            init.xavier_normal_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)
        elif isinstance(self, nn.BatchNorm1d):
            init.ones_(self.weight)
            init.zeros_(self.bias)


### TRAINING

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



# multi_train_analytics("stats/losses_0.0038.pt")

### ANALYTICS






if __name__ == "__main__":
    multi_train(5, verbose=True)
    load_model_name = "SimpleRaw1_lat16_conv4_0.004.pt"

    model = CustomRaw1(latent_dim=16).to(device)
    model.load_state_dict(torch.load("models/" + load_model_name, weights_only=True))
    model.to('cpu')
    model.eval()

    pt_num = X_val.shape[0]

    X_val = X_val.detach().cpu()
    y_val = y_val.detach().cpu()
    visualize_3d_plotly()

    for l in range(60, 95, 4):
        visualize_reconstruction(X_val[l])
    plt.show()
