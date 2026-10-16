"""MATLAB behaviours the original PEG scripts rely on.

Each helper was validated against MATLAB R2025b output (see tests/).
"""
import numpy as np

EPS = np.finfo(float).eps  # MATLAB eps


def mround(x):
    """MATLAB round(): half away from zero (NumPy rounds half to even)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def hann(n):
    """MATLAB hann(n): symmetric Hann window with zero end points."""
    return np.hanning(n)


def iqr(x, axis=0):
    """MATLAB iqr(): quantiles use the (k-0.5)/n definition (Hazen)."""
    q = np.percentile(x, [25, 75], axis=axis, method="hazen")
    return q[1] - q[0]


def colon(lo, step, hi):
    """MATLAB lo:step:hi (hi included when reachable)."""
    n = int(np.floor((hi - lo) / step + 1e-10)) + 1
    return lo + np.arange(n) * step


def fir1_bandpass(order, wn, window):
    """MATLAB fir1(order, wn, 'bandpass', window, 'scale'); wn normalised to Nyquist."""
    n = order + 1
    k = np.arange(n) - order / 2.0
    w1, w2 = wn
    safe = np.where(k == 0, 1.0, k)
    h = (np.sin(np.pi * w2 * k) - np.sin(np.pi * w1 * k)) / (np.pi * safe)
    h[k == 0] = w2 - w1
    b = h * window
    f0 = (w1 + w2) / 2.0
    return b / np.abs(np.sum(b * np.exp(-1j * np.pi * f0 * np.arange(n))))


def stft_matlab(x, fs, window, overlap):
    """MATLAB stft(x, fs, 'Window', window, 'OverlapLength', overlap) for a real column.

    Returns S (len(window) x frames), f ascending over (-fs/2, fs/2], t = window centres (s).
    """
    x = np.asarray(x, dtype=float)
    length = len(window)
    hop = length - overlap
    frames = (len(x) - overlap) // hop
    idx = np.arange(frames)[:, None] * hop + np.arange(length)[None, :]
    S = np.fft.fft(x[idx] * window[None, :], axis=1).T
    f = np.fft.fftfreq(length, 1.0 / fs)
    f = np.where(np.isclose(f, -fs / 2), fs / 2, f)
    order = np.argsort(f)
    t = (np.arange(frames) * hop + length / 2.0) / fs
    return S[order], f[order], t
