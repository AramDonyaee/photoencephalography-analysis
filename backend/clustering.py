"""Step 2 - HierarchicalClusteringUPEs.m: clustergram(data,'Cluster','Column','Standardize','Row').

Verified against MATLAB R2025b: 'Cluster','Column' reorders the ROWS (traces) and leaves the time
points in their original order; the cluster tree equals average linkage / Euclidean distance on
row-standardised data. MATLAB's left/right order of sibling branches differs from SciPy's.
"""
import numpy as np
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform


def build_matrix(fields, n_subjects):
    rows, labels = [], []
    for name, X in fields.items():
        for i in range(n_subjects):
            rows.append(X[:, i])
            labels.append(f"{name}_{i + 1}")
    return np.vstack(rows).astype(float), labels


def standardize(data, how):
    if how == "Row":
        return (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, ddof=1, keepdims=True)
    if how == "Column":
        return (data - data.mean(axis=0, keepdims=True)) / data.std(axis=0, ddof=1, keepdims=True)
    return data


def silhouette(D, labels):
    n = len(labels)
    s = np.zeros(n)
    for i in range(n):
        same = (labels == labels[i])
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, labels == c].mean() for c in np.unique(labels) if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def downsample_columns(z, max_cols=600):
    if z.shape[1] <= max_cols:
        return z
    return np.stack([c.mean(axis=1) for c in np.array_split(z, max_cols, axis=1)], axis=1)


def run(fields, params, out_dir=None):
    if params.get("cluster", "Column") != "Column":
        raise ValueError("Only 'Cluster','Column' (as in the paper) has been verified against MATLAB; "
                         "clustering time points is not offered.")
    n_subjects = int(params.get("numSubjects", min(a.shape[1] for a in fields.values())))
    method, metric = params.get("method", "average"), params.get("metric", "euclidean")
    data, labels = build_matrix(fields, n_subjects)
    z = standardize(data, params.get("standardize", "Row"))
    link = linkage(z, method=method, metric="euclidean" if method == "ward" else metric)
    tree = dendrogram(link, no_plot=True)
    order = np.array(tree["leaves"])
    k_cut = int(params.get("nClusters", 3))
    clusters = fcluster(link, k_cut, "maxclust")[order]
    D = squareform(pdist(z, metric="euclidean" if method == "ward" else metric))
    ks = list(range(2, min(10, len(labels) - 1) + 1))
    scores = [silhouette(D, fcluster(link, k, "maxclust")) for k in ks]
    heat = downsample_columns(z[order])
    return {"z": np.round(heat, 3), "row_labels": [labels[i] for i in order],
            "row_order": (order + 1), "clusters": clusters,
            "dendrogram": {"icoord": tree["icoord"], "dcoord": tree["dcoord"]},
            "silhouette": {"k": ks, "score": scores},
            "n_time_points": data.shape[1], "heatmap_columns": heat.shape[1],
            "warnings": (["Heatmap columns are averaged into bins for display; clustering used all time points."]
                         if heat.shape[1] < data.shape[1] else [])}
