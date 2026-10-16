"""Steps 5 and 6 - ShortFFT_UPEs.m, Plot_spectrogram_UPEs.m, plus the multitaper alternative."""
import os

import numpy as np
import scipy.io as sio
from scipy.signal import detrend as sp_detrend
from scipy.signal.windows import dpss

from matlab_compat import hann, stft_matlab


# ---------------------------------------------------------------- short FFT (validated)
def stft_psd(trace, fs, time_window, overlap):
    window = int(round(time_window * 1000 * fs / 1000))
    ov = int(round(overlap * window))
    s, f, t = stft_matlab(trace, fs, hann(window), ov)
    with np.errstate(divide="ignore"):
        psd = 10 * np.log10(np.abs(s) ** 2)
    return f, t, psd


# ---------------------------------------------------------------- multitaper (NOT validated)
def multitaper_psd(x, fs, freq_range, taper_params, window_params, min_nfft=0,
                   detrend_opt="off", weighting="adapt"):
    """Python version of the Prerau-lab multitaper_spectrogram used in Troubleshooting 1.

    No MATLAB reference output was available, so treat results as unvalidated.
    """
    x = np.asarray(x, dtype=float)
    tw, k = float(taper_params[0]), int(taper_params[1])
    win = int(round(window_params[0] * fs))
    step = int(round(window_params[1] * fs))
    nfft = max(int(2 ** np.ceil(np.log2(win))), int(min_nfft))
    starts = np.arange(0, len(x) - win + 1, step)
    freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
    sel = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    tapers, eig = dpss(win, tw, k, return_ratios=True)
    tapers = tapers * np.sqrt(fs)
    out = np.empty((sel.sum(), len(starts)))
    for j, s in enumerate(starts):
        seg = x[s:s + win]
        if detrend_opt == "linear":
            seg = sp_detrend(seg, type="linear")
        elif detrend_opt == "constant":
            seg = seg - seg.mean()
        spower = np.abs(np.fft.rfft(tapers * seg[None, :], nfft, axis=1)[:, sel]) ** 2
        if weighting == "unity":
            mt = spower.mean(axis=0)
        elif weighting == "eigen":
            wt = (eig / k)[:, None]
            mt = (wt * spower).sum(axis=0)
        else:  # adaptive (Thomson) weights
            sig2 = np.dot(seg, seg) / win
            mt = spower[:2].mean(axis=0)
            for _ in range(100):
                w = (np.sqrt(eig)[:, None] * mt[None, :]) / (eig[:, None] * mt[None, :] + (1 - eig)[:, None] * sig2)
                new = (w ** 2 * spower).sum(axis=0) / (w ** 2).sum(axis=0)
                if np.max(np.abs(new - mt)) <= 5e-4 * sig2 / nfft:
                    mt = new
                    break
                mt = new
        out[:, j] = mt
    with np.errstate(divide="ignore", invalid="ignore"):
        psd = np.where(out > 0, 10 * np.log10(out), np.nan)
    return freqs[sel], (starts + win / 2.0) / fs, psd


# ---------------------------------------------------------------- step 5
def run_stft(fields, params, store, out_dir):
    fs = float(params.get("fs", 20))
    n_subjects = int(params.get("numSubjects", min(a.shape[1] for a in fields.values())))
    method = params.get("method", "stft")
    store.clear()
    for name, X in fields.items():
        for i in range(n_subjects):
            if method == "multitaper":
                f, t, p = multitaper_psd(
                    X[:, i], fs, (float(params.get("minF", 0.001)), fs / 2),
                    (float(params.get("tw", 3)), int(params.get("tapers", 5))),
                    (float(params.get("win", 10)), float(params.get("step", 2))),
                    0, params.get("detrend", "off"), params.get("weighting", "adapt"))
            else:
                f, t, p = stft_psd(X[:, i], fs, float(params.get("timeWindow", 20)),
                                   float(params.get("overlapLength", 0.75)))
            store[f"{name}{i + 1}"] = (f, t, p)
    names = list(store)
    f0, t0, p0 = store[names[0]]
    pos = f0 > 0
    mat = {n: {"power": p, "frequency": f, "time": t} for n, (f, t, p) in store.items()}
    path = os.path.join(out_dir, "Spectrogram.mat")
    sio.savemat(path, {"Spectrogram": mat}, do_compression=True)
    return {"trace_names": names, "shape": list(p0.shape),
            "avg_power": {"name": names[0], "f": f0[pos], "p": np.nanmean(p0[pos], axis=1)},
            "warnings": (["Multitaper output is not validated against MATLAB."] if method == "multitaper" else []),
            "files": [{"label": "Spectrogram.mat", "name": "Spectrogram.mat"}]}


def compare_windows(fields, params):
    """Figure 2 of the paper: the same trace with several window lengths (80% overlap)."""
    fs = float(params.get("fs", 20))
    name = next(iter(fields))
    trace = next(iter(fields.values()))[:, 0]
    items = []
    for w in params.get("windows", [10, 30, 60]):
        win = int(round(float(w) * fs))
        hop = max(1, int(round(0.2 * win)))
        s, f, t = stft_matlab(trace, fs, hann(win), win - hop)
        with np.errstate(divide="ignore"):
            p = 10 * np.log10(np.abs(s) ** 2)
        keep = f >= 2.0 / float(w)
        step = max(1, p.shape[1] // 300)
        items.append({"label": f"{name}1, window={w} s, step={hop / fs:g} s",
                      "f": f[keep], "t": t[::step], "z": p[keep][:, ::step],
                      "avg_f": f[f > 0], "avg_p": p[f > 0].mean(axis=1)})
    return {"compare": items}


# ---------------------------------------------------------------- step 6
def plot_data(store, traces, fs, time_window):
    if not store:
        raise ValueError("Run step 5 first.")
    plots = []
    for name in traces:
        if name not in store:
            raise ValueError(f"Unknown trace '{name}'.")
        f, t, p = store[name]
        min_freq = (1.0 / time_window) * 2
        idx = int(np.argmin(np.abs(f - min_freq)))
        step = max(1, p.shape[1] // 400)
        plots.append({"name": name, "f": f[idx:], "t": t[::step], "z": p[idx:, ::step]})
    return {"plots": plots}
