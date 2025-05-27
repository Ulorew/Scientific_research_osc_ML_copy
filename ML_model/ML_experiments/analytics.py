import pandas as pd
import plotly.express as px
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from dataset import *
from ML_model.ML_experiments.model import CustomRaw1, CustomRaw2


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
                        title=f'3D латентное пространство {model.model_name}')
    fig.update_layout(scene=dict(
        xaxis_title='PCA1',
        yaxis_title='PCA2',
        zaxis_title='PCA3'
    ))
    fig.show()

conv_size = [4, 4, 4]
fc_size = []
latent_dim=32
device="cpu"
load_loss="0.002"

if __name__ == '__main__':
    model = CustomRaw2(latent_dim=latent_dim, conv_sizes=conv_size, fc_sizes=fc_size).to(device)

    load_dir = f"models/{model.model_name.replace('/', '_')}".replace("_", '/')
    load_name = f"{load_dir}/{load_loss}.pt"
    model.load_state_dict(torch.load(load_name, weights_only=True))
    model.to('cpu')
    model.eval()

    model.w
    #visualize_3d_plotly()

