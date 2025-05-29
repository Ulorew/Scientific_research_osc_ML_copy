import pandas as pd
import plotly.express as px
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from contextlib import redirect_stdout, redirect_stderr
from sklearn.manifold import TSNE
from sympy import false
from torchmetrics.functional.text import perplexity

from dataset import *

from ML_model.ML_experiments.model import CustomRaw1, CustomRaw2
import plotly.io as pio
import plotly.offline as pyo


# pio.renderers.default = "notebook"
# pyo.init_notebook_mode(connected=True)


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
        ax.set_ylim([-1.5, 1.5])
        ax.grid()
    x_rec = model(x.unsqueeze(0)).squeeze()  # treat batch dim as rec dim
    visualize_wave(x_rec.detach().numpy(), axs=axs, draw_feats=draw_feats, color='red',
                   label="rec")
    visualize_wave(x, axs=axs, draw_feats=draw_feats, label="orig", linestyle='--')
    ax.set_title(f"MSE: {((x_rec - x) ** 2).mean()}")


def visualize_2d():
    global X_val, y_val

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

    plt.title(f'Группировка точек модели {model.model_name}')
    # plt.set_xlabel('X')
    # plt.set_ylabel('Y')
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.grid(True)

    plt.scatter(ptx, pty, c=y_vis)
    plt.colorbar(label='Концентрация событий')
    plt.show()
    # for pt, y in zip(pts, y_vis):
    #     print(pt, y)


def visualize_3d_plotly(show_pca: bool = True, show_sne: bool = False, perplexity: int = 30):
    global X_val, y_val

    pt_num = X_full.shape[0]
    X_vis, y_vis = X_full[:pt_num].to('cpu'), y_full[:pt_num].to('cpu')

    pts_raw = model.encode(X_vis).detach().cpu().numpy()

    if show_sne:
        tsne = TSNE(
            n_components=3,  # число выходных осей (2 или 3)
            perplexity=perplexity,  # «ширина» локального окружения
            learning_rate=100,  # скорость обучения
            max_iter=3000,  # число итераций оптимизации
            random_state=42  # для воспроизводимости
        )
        pts_sne = tsne.fit_transform(pts_raw)

        df_sne = pd.DataFrame({
            'x': pts_sne[:, 0],
            'y': pts_sne[:, 1],
            'z': pts_sne[:, 2],
            'label': y_vis.numpy()
        })
        fig = px.scatter_3d(df_sne, x='x', y='y', z='z',
                            color='label',
                            title=f'Сжатое латентное пространство {model.model_name}, perp = {perplexity}')
        fig.update_layout(scene=
        dict(
            xaxis_title='tSNE1',
            yaxis_title='tSNE2',
            zaxis_title='tSNE3'
        ))
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            fig.show()

    if show_pca:
        pts_pca = PCA(n_components=3).fit_transform(pts_raw)
        df_pca = pd.DataFrame({
            'x': pts_pca[:, 0],
            'y': pts_pca[:, 1],
            'z': pts_pca[:, 2],
            'label': y_vis.numpy()
        })

        fig = px.scatter_3d(df_pca, x='x', y='y', z='z',
                            color='label',
                            title=f'Сжатое латентное пространство {model.model_name}')
        fig.update_layout(scene=
        dict(
            xaxis_title='tPCA1',
            yaxis_title='tPCA2',
            zaxis_title='tPCA3'
        ))
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            fig.show()


conv_size = [4, 4]
fc_size = []
latent_dim = 8
device = "cpu"
load_loss = "0.255"

if __name__ == '__main__':
    model = CustomRaw2(latent_dim=latent_dim, conv_sizes=conv_size, fc_sizes=fc_size, input_channels=num_channels,
                       input_length=wsz).to(device)

    load_dir = f"models/{model.model_name.replace('/', '_')}".replace("_", '/')
    load_name = f"{load_dir}/{load_loss}.pt"
    model.load_state_dict(torch.load(load_name, weights_only=True))
    model.to('cpu')
    model.eval()
    X_full = X_full.cpu()
    y_full = y_full.cpu()
    # plt.hist(y_full, bins=50)
    print(y_full[:100])
    for i in range(0, 100):
        if y_full[i] > 0.32:
            visualize_reconstruction(X_full[i])
    plt.show()
    # lId = 0
    # print(model.encoder_conv[lId * 3])
    # wts = model.encoder_conv[lId * 3].weight.detach().numpy()
    # print(wts)
    # print(wts.shape)
    #
    # for ker in wts:
    #     for i in ker:
    #         plt.plot(i)
    #     plt.show()
    visualize_3d_plotly()
    # for perp in range(30, 51, 10):
    #     visualize_3d_plotly(perp)
