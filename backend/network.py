"""Steps 8-11 - correlation, coherence, Granger causality, mutual information (windowed, pairwise).

Windowing, edge rules and estimators reproduce the uploaded MATLAB scripts (validated on R2025b):
  * window step = ceil(width*overlap); the last window is cut at the end of the recording
  * correlation / coherence / MI: |value|, then values < cutoff set to 0
  * Granger (as uploaded): every window runs from its start to the END of the recording
    (a `pval1` typo sends all windows to the catch branch), gctest F statistic with 1 lag
    conditioned on the other electrode(s), and edges are zeroed only when p == 1.
Set params["corrected"]=True for windows of the stated length and the p > 0.05 rule.
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import coherence as sp_coherence, get_window, resample_poly
from scipy.spatial import cKDTree
from scipy.special import digamma
from scipy.stats import f as f_dist

from export import write_sheets

STEP_INFO = {  # frontend step id -> (step number, file stem, directed?)
    "correlation": (8, "Correlation", False), "coherence": (9, "Coherence", False),
    "granger": (10, "GrangerCausality", True), "mutual_info": (11, "MutualInformation", False)}


# ------------------------------------------------------------------ windows
def make_windows(n, width, overlap):
    """1-based inclusive (start, finish) lists exactly as the MATLAB while-loop builds them."""
    step = int(np.ceil(width * overlap))
    starts, finishes = [1], [width]
    while finishes[-1] < n:
        starts.append(starts[-1] + step)
        finishes.append(finishes[-1] + step)
    return starts, finishes


def segment(data, s, f):
    """data: electrodes x time. A window past the end is cut (MATLAB catch branch)."""
    return data[:, s - 1:f] if f <= data.shape[1] else data[:, s - 1:]


# ------------------------------------------------------------------ estimators
def abs_corr(sub):
    return np.abs(np.corrcoef(sub))


def partial_corr_abs(sub):
    P = np.linalg.pinv(np.cov(sub))
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return np.abs(pc)


def mscohere_median(y1, y2):
    n = len(y1)
    L = int(n // 4.5)
    nfft = max(256, 2 ** int(np.ceil(np.log2(L))))
    _, c = sp_coherence(y1, y2, fs=1.0, window=get_window("hamming", L, fftbins=False),
                        noverlap=L // 2, nfft=nfft, detrend=False)
    return float(np.median(c))


def _count_within(v, dk):
    """MATLAB rangesearch_1point on sorted data: points strictly closer than dk, excluding self."""
    order = np.argsort(v, kind="stable")
    sv = v[order]
    n = len(v)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    out = np.zeros(n, dtype=int)
    for i in range(n):
        p0, x0, d, c = rank[i], sv[rank[i]], dk[i], 0
        j = p0 + 1
        while j < n and abs(sv[j] - x0) < d:
            c += 1
            j += 1
        j = p0 - 1
        while j >= 0 and abs(sv[j] - x0) < d:
            c += 1
            j -= 1
        out[i] = c
    return out


def mutual_info(x, y, k=3):
    """mi_cont_cont from the MATLAB script (Kraskov algorithm 1, Chebyshev distance)."""
    x = np.asarray(x, dtype=float).ravel() - np.min(x) + 1e-8
    y = np.asarray(y, dtype=float).ravel() - np.min(y) + 1e-8
    n = len(x)
    pts = np.column_stack([x, y])
    dk = cKDTree(pts).query(pts, k=k + 1, p=np.inf)[0][:, k]
    nx, ny = _count_within(x, dk), _count_within(y, dk)
    return max(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)), 0.0)


def gc_f(y1, y2, others):
    """gctest(y1, y2, others): does y1 Granger-cause y2 given others? 1 lag, F statistic."""
    Y = y2[1:]
    base = [np.ones(len(Y)), y2[:-1]] + [c[:-1] for c in others]
    Ar = np.column_stack(base)
    Au = np.column_stack(base + [y1[:-1]])
    rss = lambda A: np.sum((Y - A @ np.linalg.lstsq(A, Y, rcond=None)[0]) ** 2)
    rr, ru = rss(Ar), rss(Au)
    dof = len(Y) - Au.shape[1]
    F = (rr - ru) / (ru / dof)
    return float(F), float(f_dist.sf(F, 1, dof))


def phase_randomize(sub, rng):
    out = np.empty_like(sub, dtype=float)
    n = sub.shape[1]
    for i, x in enumerate(sub):
        spec = np.fft.rfft(x)
        phase = rng.uniform(0, 2 * np.pi, len(spec))
        phase[0] = 0
        if n % 2 == 0:
            phase[-1] = 0
        out[i] = np.fft.irfft(np.abs(spec) * np.exp(1j * phase), n)
    return out


# ------------------------------------------------------------------ pair naming
def tags(names):
    short = [n[:2] for n in names]                    # MATLAB uses the first two characters
    return short if len(set(short)) == len(short) else list(names)


def pair_list(tag, directed):
    out = []
    for k in range(len(tag) - 1):
        for q in range(k + 1, len(tag)):
            out.append((tag[k] + tag[q], k, q))
            if directed:
                out.append((tag[q] + tag[k], q, k))
    return out


# ------------------------------------------------------------------ main entry
def run(kind, electrodes, params, out_dir):
    """electrodes: {name: array (time x subjects)}, all with the same number of rows."""
    step_no, stem, directed = STEP_INFO[kind]
    names = list(electrodes)
    if len(names) < 2:
        raise ValueError("At least two electrodes are needed.")
    if len({a.shape[0] for a in electrodes.values()}) > 1:
        raise ValueError("Electrodes must have the same number of time points.")
    factor = int(params.get("resampleFactor", 1))
    fs = float(params.get("fs", 20)) / factor
    width = int(round(float(params.get("timeWindow", 60)) * fs))
    overlap = float(params.get("overlap", 0.5))
    cutoff = float(params.get("cutoff", 0))
    n_sub = int(params.get("numSubjects", min(a.shape[1] for a in electrodes.values())))
    corrected = bool(params.get("corrected", False))
    n_surr = int(params.get("nSurrogates", 0))
    do_trgc = bool(params.get("trgc", False))
    rng = np.random.default_rng(int(params.get("seed", 0)))
    tag = tags(names)
    pairs = pair_list(tag, directed)
    warnings = []
    if kind == "granger" and not corrected:
        warnings.append("MATLAB-script compatibility mode: the uploaded script's pval1 typo effectively uses "
                        "expanding windows and does not apply the stated p > 0.05 cutoff. Enable the "
                        "'corrected' option to use fixed-length windows and the paper-described p < 0.05 rule.")
    if kind == "granger" and n_surr > 0 and not corrected:
        warnings.append("Surrogate testing in compatibility mode is slow and not statistically comparable; "
                        "enable corrected windows before using surrogates.")
    if kind == "granger" and corrected:
        warnings.append("Using fixed-length windows and the paper-described p < 0.05 cutoff. Results can differ "
                        "from the uploaded MATLAB script because its window/p-value logic appears inconsistent "
                        "with the written protocol.")
    nE = len(names)
    n_win = None
    traj = {p: [] for p, _, _ in pairs}
    median = {p: [] for p, _, _ in pairs}
    net_sum, net_n = np.zeros((nE, nE)), 0
    trgc_idx = {p: [] for p, _, _ in pairs}

    for subj in range(n_sub):
        data = np.column_stack([electrodes[n][:, subj] for n in names])
        if factor != 1:
            data = resample_poly(data, 1, factor, axis=0)
        data = data.T
        starts, finishes = make_windows(data.shape[1], width, overlap)
        edge_traj = {p: [] for p, _, _ in pairs}
        for s, f in zip(starts, finishes):
            full = data[:, s - 1:] if (kind == "granger" and not corrected) else segment(data, s, f)
            E = np.zeros((nE, nE))
            if kind == "correlation":
                E = partial_corr_abs(full) if params.get("partial") else abs_corr(full)
            elif kind == "coherence":
                for i in range(nE - 1):
                    for j in range(i + 1, nE):
                        E[i, j] = E[j, i] = abs(mscohere_median(full[i], full[j]))
            elif kind == "mutual_info":
                k = int(params.get("K", 3))
                for i in range(nE - 1):
                    for j in range(i + 1, nE):
                        E[i, j] = E[j, i] = abs(mutual_info(full[i], full[j], k))
            if kind != "granger":
                E[E < cutoff] = 0.0
            else:
                p_thr = 0.05 if corrected else 1.0
                null = {}
                if n_surr > 0:
                    for _ in range(n_surr):
                        sur = phase_randomize(full, rng)
                        for _, i, j in pairs:
                            null.setdefault((i, j), []).append(
                                gc_f(sur[i], sur[j], [sur[k] for k in range(nE) if k not in (i, j)])[0])
                for _, i, j in pairs:
                    others = [full[k] for k in range(nE) if k not in (i, j)]
                    F, pval = gc_f(full[i], full[j], others)
                    if n_surr > 0:
                        pval = (1 + np.sum(np.array(null[(i, j)]) >= F)) / (n_surr + 1)
                    keep = pval < p_thr
                    E[i, j] = F if keep else 0.0
                    if do_trgc and keep and F > 0:
                        rev = [o[::-1] for o in others]
                        Fr = gc_f(full[i][::-1], full[j][::-1], rev)[0]
                        trgc_idx[tag[i] + tag[j]].append((F - Fr) / (F + Fr))
            net_sum += E
            net_n += 1
            for p, i, j in pairs:
                edge_traj[p].append(E[i, j])
        for p, _, _ in pairs:
            traj[p].append(edge_traj[p])
            median[p].append(float(np.median(edge_traj[p])))
        n_win = len(starts)

    median_arr = {p: np.array(v) for p, v in median.items()}
    traj_arr = {p: np.array(v) for p, v in traj.items()}
    rows = [dict({"subject": s + 1, "window": w + 1}, **{p: traj_arr[p][s, w] for p in traj_arr})
            for s in range(n_sub) for w in range(n_win)]
    fname = write_sheets(os.path.join(out_dir, f"Results_Step{step_no}_{stem}.xlsx"),
                         {"median": pd.DataFrame(median_arr), "edge_trajectories": pd.DataFrame(rows)})
    result = {"pairs": [p for p, _, _ in pairs], "median": median_arr, "trajectories": traj_arr,
              "mean_network": {"labels": tag, "z": net_sum / net_n}, "n_windows": n_win,
              "warnings": warnings, "files": [{"label": f"{stem} (xlsx)", "name": fname}]}
    if kind == "granger" and do_trgc:
        result["trgc"] = {p: {"index": np.array(v), "fraction_above": float(np.mean(np.array(v) > 0.3)) if v else None}
                          for p, v in trgc_idx.items()}
    return result
