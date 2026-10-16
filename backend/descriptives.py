"""Step 3 - DescriptiveStatisticsUPEs.m: mean and coefficient of variation per trace."""
import os

import numpy as np

from export import write_sheets


def compute(fields, n_subjects):
    average, cv = {}, {}
    for name, X in fields.items():
        x = X[:, :n_subjects]
        mean = x.mean(axis=0)
        average[name] = mean
        cv[name] = x.std(axis=0, ddof=1) / mean      # MATLAB std uses N-1
    return average, cv


def run(fields, params, out_dir):
    n_subjects = int(params.get("numSubjects", min(a.shape[1] for a in fields.values())))
    average, cv = compute(fields, n_subjects)
    f1 = write_sheets(os.path.join(out_dir, "Results_Step3_Average.xlsx"), average)
    f2 = write_sheets(os.path.join(out_dir, "Results_Step3_CV.xlsx"), cv)
    return {"average": average, "cv": cv,
            "files": [{"label": "Average (xlsx)", "name": f1}, {"label": "CV (xlsx)", "name": f2}]}
