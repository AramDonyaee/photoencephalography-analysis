"""Step 4 - ShannonEntropyUPEs.m: common Freedman-Diaconis binning, then Shannon entropy."""
import os

import numpy as np

from export import write_sheets
from matlab_compat import EPS, colon, iqr


def fd_widths(fields):
    """Per subject column and per field: ceil(range / (2*IQR*N^(-1/3)))."""
    out = {}
    for name, X in fields.items():
        n = X.shape[0]
        out[name] = np.ceil((X.max(axis=0) - X.min(axis=0)) / (2.0 * iqr(X) * n ** (-1.0 / 3.0)))
    return out


def common_bins(fields):
    widths = fd_widths(fields)
    allw = np.concatenate(list(widths.values()))
    width = float(np.ceil(np.median(allw)))
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Could not determine a valid bin width (is a trace constant?).")
    lo = min(X.min() for X in fields.values())
    hi = max(X.max() for X in fields.values())
    edges = colon(lo, width, hi)
    edges[-1] += EPS                                  # as in the original (no-op for large values)
    return widths, width, edges


def entropy_of(trace, edges):
    counts, _ = np.histogram(trace, bins=edges)      # values beyond the last edge are dropped, like histcounts
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p + EPS))


def run(fields, params, out_dir):
    n_subjects = int(params.get("numSubjects", min(a.shape[1] for a in fields.values())))
    widths, width, edges = common_bins(fields)
    result = {name: np.array([entropy_of(X[:, i], edges) for i in range(n_subjects)])
              for name, X in fields.items()}
    fname = write_sheets(os.path.join(out_dir, "Results_Step4_Entropy.xlsx"), result)
    warnings = []
    top = max(X.max() for X in fields.values())
    if edges[-1] < top:
        warnings.append(f"Highest bin edge ({edges[-1]:.0f}) is below the data maximum ({top:.0f}); "
                        "values above it are not counted (same as the MATLAB script).")
    return {"entropy": result, "bin_width": width, "bin_edges": int(len(edges)),
            "fd_widths": {k: float(np.median(v)) for k, v in widths.items()},
            "warnings": warnings, "files": [{"label": "Entropy (xlsx)", "name": fname}]}
