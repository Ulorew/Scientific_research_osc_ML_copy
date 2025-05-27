import math
import time
from turtledemo.penrose import start

import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import fft
from tqdm import tqdm
from collections import defaultdict

# SETTINGS


# dataset_path = "../../Data/datset_v1.csv"
period_size = 32
window_period_cnt = 8
wsz = period_size * window_period_cnt
etal_sim_th = -1  # 0.5  # max value of standard deviation with etalon, recommended to leave only abnormal events: 0.5
unmatched_channels_th = 3  # threshold which states the least number of unmatched channels for measure to be considered abnormal

cap_wsz = period_size * 8  # size of window of recordings
capture_step = 13  # step between recordings
adjust_ampl_factor = 2  # features from adjust_ampl_feats have adaptive amplitude scale from 1/fac to fac

feats = ["UA BB", "UB BB", "UC BB"]  # ["IA", "IC", "UA BB", "UB BB", "UC BB"]  # , "UN BB"
adjust_ampl_feats = {"IA", "IC"}  # "UN BB"
x_fft = fft.rfftfreq(wsz, 1.0 / period_size)
specsz = len(x_fft)
etal_weight = np.ones(specsz)
window_func = np.hanning(wsz)
cap_window_func = np.hanning(cap_wsz)
window_func /= np.sqrt(np.average(window_func ** 2))
cap_window_func /= np.sqrt(np.average(cap_window_func ** 2))
cap_feats = ["UA BB", "UB BB", "UC BB"]

dataset_path = "../../Data/OscData2.csv"
save_path = f"../../Data/OscData_raw_{cap_wsz}_{len(cap_feats)}"

channel_ampls = defaultdict(lambda: 1, {
    "IA": 1,
    "IB": 1,
    "IC": 1,
    "UA BB": 84.5,
    "UB BB": 84.5,
    "UC BB": 84.5,
    "UN BB": 1,
})

op_names = ['opr_swch', 'abnorm_evnt', 'emerg_evnt', 'normal']


def norm_fft(seq, window_func=None):
    n = len(seq)
    if window_func is None:
        window_func = np.ones_like(seq)

    res = np.abs(fft.rfft(seq * window_func) / n)
    res[1:] *= 2
    return res


def energy_diff(seq1, seq2, weights=None):  # sqrt(wave power difference)
    if weights is None:
        weights = np.ones(len(seq1))

    return np.sqrt(np.sum((np.abs((seq1 ** 2) - (seq2 ** 2))) * weights))


def etalon_cosine(n=wsz, amplitude=1):
    return norm_fft(amplitude * np.cos(np.linspace(0, (2 * np.pi * n / period_size), n, endpoint=False)),
                    window_func=window_func)


etal0 = np.zeros(specsz)
etal1 = etalon_cosine(wsz, amplitude=1)


def match_etal(lpos, data_track, wsz=wsz, window_func=window_func):
    seq_trim = data_track[lpos:lpos + wsz]
    nomatch = []

    for feat in feats:
        seq = seq_trim[feat].values.copy()
        ampl = channel_ampls[feat]

        if feat in adjust_ampl_feats:
            sig_ampl = np.average(np.abs(seq)) * np.pi / 2
            dif = sig_ampl / ampl
            ampl *= np.clip(dif, 1 / adjust_ampl_factor, adjust_ampl_factor)

        seq /= ampl
        spec = norm_fft(seq, window_func)

        sim0 = energy_diff(etal0, spec, weights=etal_weight)
        sim1 = energy_diff(etal1, spec, weights=etal_weight)

        if min(sim0, sim1) > etal_sim_th:
            nomatch.append(feat)
    return nomatch


def gen_case(starttime, data_track, window_func=cap_window_func, feats=cap_feats, capwsz=cap_wsz):
    measures = []

    lpos = starttime
    rpos = lpos + capwsz
    if lpos < 0 or rpos >= len(data):
        raise IndexError("Wrong time interval for case generation")

    seq_trim = data_track[lpos:lpos + capwsz]

    tracks = []

    for feat in feats:
        seq = seq_trim[feat].values.copy()
        seq /= channel_ampls[feat]
        tracks.append(seq)

    return np.array(tracks)


def process_file(filename: str):
    data_track = data[data["file_name"] == filename]
    X = []
    y = []

    for rpos in range(max(cap_wsz, wsz), len(data_track), capture_step):
        lpos = rpos - wsz
        nomatch = match_etal(lpos, data_track=data_track, window_func=window_func)

        cap_lpos = rpos - cap_wsz

        if len(nomatch) >= unmatched_channels_th:
            X.append(
                gen_case(cap_lpos, data_track=data_track, window_func=cap_window_func, feats=cap_feats,
                         capwsz=cap_wsz))

            events = data_track[op_names[:3]][lpos:rpos]
            y.append(events.mean().mean())

        # print(lpos, nomatch)

    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == "__main__":
    data = pd.read_csv(dataset_path)
    data[op_names] = data[op_names].fillna(value=0)
    # data.fillna({"IB": 0}, inplace=True)
    data.dropna(axis=1, how='any', inplace=True)
    files = np.unique(data["file_name"].values)

    X, y = np.ndarray((0, len(cap_feats), cap_wsz)), np.array([])

    for filename in tqdm(files):
        # filename="b629f3bb07ef79f5845c27daa0a83425_Bus 2 _event N1"
        # print(f"Working with file {filename}")
        tX, ty = process_file(filename)
        if len(tX) == 0:
            continue
        X = np.append(X, tX, axis=0)
        y = np.append(y, ty, axis=0)
        # break

    print(f"Done! Event rate: {y.mean()}")
    print(f"Total cases: {len(y)}")

    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).float()

    torch.save(X, save_path + "_X.pt")
    torch.save(y, save_path + "_y.pt")
    print(f"Saved as {save_path}_*.pt")
