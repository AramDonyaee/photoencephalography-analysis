"""Step 7 - ComputeDFA_UPE_singleWindow.m / ComputeDFA_UPE_Loop.m (NBT nbt_doDFA, 8-argument version)."""
import os

import numpy as np
import pandas as pd
from scipy.signal import freqz, lfilter, hilbert

from export import write_sheets
from matlab_compat import fir1_bandpass, hann, mround


def design_filter(hp, lp, fs, order):
    return fir1_bandpass(order, [hp / (fs / 2), lp / (fs / 2)], hann(order + 1))


def dfa_curve(sig, fit, calc, overlap, fs, res_logbin=10):
    """Returns exponent, window sizes (samples) and fluctuation F(n)."""
    d1 = int(np.floor(np.log10(calc[0] * fs)))
    d2 = int(np.ceil(np.log10(calc[1] * fs)))
    xt = mround(np.logspace(d1, d2, (d2 - d1) * res_logbin)).astype(int)
    X = xt[(calc[0] * fs <= xt) & (xt <= calc[1] * fs)]
    if len(X) == 0 or X[0] / fs > fit[0] or X[-1] / fs < fit[1]:
        raise ValueError("The CalcInterval is smaller than the FitInterval.")
    y = sig / np.mean(sig)
    y = np.cumsum(y - np.mean(y))
    n = len(y)
    F = np.empty(len(X))
    for i, x in enumerate(X):
        step = int(mround(x * (1 - overlap)))
        starts = np.arange(0, n - x, max(step, 1))
        length = x + 1
        t = np.arange(1, length + 1) / length
        A = np.column_stack([t, np.ones(length)])
        P = A @ np.linalg.pinv(A)
        W = np.stack([y[s:s + length] for s in starts])
        R = W - W @ P.T
        F[i] = np.mean(np.sqrt(np.mean(R ** 2, axis=1)))
    a = np.where(X >= fit[0] * fs)[0].min()
    b = np.where(X <= fit[1] * fs)[0].max()
    design = np.column_stack([np.ones(b - a + 1), np.log10(X[a:b + 1])])
    coef = np.linalg.lstsq(design, np.log10(F[a:b + 1]), rcond=None)[0]
    return float(coef[1]), X, F, (a, b), coef


def trace_dfa(sig, p, b=None):
    sig = np.asarray(sig, dtype=float)
    sig = sig - sig.mean()
    if b is None:
        b = design_filter(p["hp"], p["lp"], p["fs"], p["order"])
    filtered = lfilter(b, [1.0], sig)
    env = np.abs(hilbert(filtered))
    out = dfa_curve(env, p["fit"], p["calc"], p["overlap"], p["fs"])
    return out + (filtered, env)


def read_params(params):
    hp, lp, fs = float(params.get("hp", 0.1)), float(params.get("lp", 1)), float(params.get("fs", 20))
    order = int(round(float(params.get("orderFactor", 50)) / hp))
    return {"hp": hp, "lp": lp, "fs": fs, "order": order,
            "fit": [float(v) for v in params.get("fit", [80, 150])],
            "calc": [float(v) for v in params.get("calc", [20, 200])],
            "overlap": float(params.get("overlap", 0.8))}


def filter_response(params):
    p = read_params(params)
    b = design_filter(p["hp"], p["lp"], p["fs"], p["order"])
    w, h = freqz(b, worN=4096, fs=p["fs"])
    with np.errstate(divide="ignore"):
        mag = 20 * np.log10(np.abs(h))
    return {"filter": {"f": w, "mag_db": np.maximum(mag, -200)}}


def white_noise_check(n_samples, p, n_reps, seed=0):
    b = design_filter(p["hp"], p["lp"], p["fs"], p["order"])
    rng = np.random.default_rng(seed)
    exps, logF = [], []
    for _ in range(n_reps):
        wn = np.sqrt(1e-5) * rng.standard_normal(n_samples)
        env = np.abs(hilbert(lfilter(b, [1.0], wn)))
        e, X, F, _, _ = dfa_curve(env, p["fit"], p["calc"], p["overlap"], p["fs"])
        exps.append(e)
        logF.append(np.log10(F))
    exps = np.array(exps)
    lx = np.log10(X / p["fs"])
    ly = np.mean(logF, axis=0)
    slopes = np.diff(ly) / np.diff(lx)
    return {"mean": float(exps.mean()), "sd": float(exps.std(ddof=1)) if n_reps > 1 else 0.0,
            "n": int(n_reps), "ok": bool(abs(exps.mean() - 0.5) <= 0.03),
            "slopes": {"x": (lx[:-1] + lx[1:]) / 2, "y": slopes}}


def run_single(fields, params):
    p = read_params(params)
    name = params.get("electrode") or next(iter(fields))
    subject = int(params.get("subject", 1))
    X = fields[name]
    e, Xw, F, (a, b), coef, filtered, env = trace_dfa(X[:, subject - 1], p)
    lx = np.log10(Xw / p["fs"])
    stride = max(1, len(env) // 3000)
    result = {"trace": {"exponent": e, "electrode": name, "subject": subject,
                        "x_all": lx, "y_all": np.log10(F), "x_fit": lx[a:b + 1], "y_fit": np.log10(F[a:b + 1]),
                        "line_y": coef[0] + coef[1] * np.log10(Xw[a:b + 1]),
                        "signal": filtered[::stride], "envelope": env[::stride]}}
    n_reps = int(params.get("nWhite", 100))
    if n_reps > 0:
        n_samples = next(iter(fields.values())).shape[0]
        result["white"] = white_noise_check(n_samples, p, n_reps, int(params.get("seed", 0)))
    return result


def run_loop(fields, params, out_dir):
    p = read_params(params)
    n_subjects = int(params.get("numSubjects", min(a.shape[1] for a in fields.values())))
    b = design_filter(p["hp"], p["lp"], p["fs"], p["order"])
    cols = list(fields)
    values = np.full((n_subjects, len(cols)), np.nan)
    for j, name in enumerate(cols):
        for i in range(n_subjects):
            values[i, j] = trace_dfa(fields[name][:, i], p, b)[0]
    inputs = pd.DataFrame([{"hp": p["hp"], "lp": p["lp"], "fs": p["fs"], "filter_order": p["order"],
                            "FitInterval_1": p["fit"][0], "FitInterval_2": p["fit"][1],
                            "CalcInterval_1": p["calc"][0], "CalcInterval_2": p["calc"][1],
                            "overlap": p["overlap"]}])
    fname = write_sheets(os.path.join(out_dir, "DFA Scaling Exponent.xlsx"),
                         {"DFA exp": pd.DataFrame(values, columns=cols), "Input": inputs})
    return {"loop": {"cols": cols, "values": values},
            "files": [{"label": "DFA Scaling Exponent (xlsx)", "name": fname}]}
