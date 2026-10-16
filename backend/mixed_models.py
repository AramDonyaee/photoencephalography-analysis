"""Step 12 - MixedEffectModels.R: lmer(Entropy ~ Electrode * Task + (1|Subject) + (1|Subject:Task)).

Two engines:
  * "python" (default): statsmodels MixedLM (REML) + Wald chi-square Type III tests like car::Anova
    + estimated-marginal-mean contrasts with Bonferroni. Post-hoc p-values use the normal
    distribution (emmeans uses Satterthwaite df), so they differ slightly from R.
  * "r": runs the paper's R packages through Rscript when it is installed (exact).
NOT tested against R output yet - compare with your own R run before relying on it.
"""
import itertools
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

CANDIDATES = {
    "Model 1": "{dv} ~ Electrode + Task + (1 | Subject)",
    "Model 2": "{dv} ~ Electrode * Task + (1 | Subject)",
    "Model 3": "{dv} ~ Electrode * Task + (1 | Subject) + (1 | Subject:Electrode)",
    "Model 4": "{dv} ~ Electrode + Task + (1 + Task | Subject)",
    "Model 5": "{dv} ~ Electrode * Task + (1 | Subject) + (1 | Subject:Task)",
}


def _r_levels(series):
    """Level order R's factor() would use: case-insensitive alphabetical."""
    return sorted(series.astype(str).unique(), key=lambda s: (s.lower(), s))


def prepare(df):
    df = df.copy()
    for c in ("Subject", "Electrode", "Task"):
        if c not in df.columns:
            raise ValueError(f"The stats dataset needs a '{c}' column.")
        df[c] = df[c].astype(str)
    for c in ("Electrode", "Task"):
        df[c] = pd.Categorical(df[c], categories=_r_levels(df[c]), ordered=False)
    return df

def _split_top_level(text):
    parts, depth, cur = [], 0, ""
    for ch in text:
        depth += (ch == "(") - (ch == ")")
        if ch == "+" and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    parts.append(cur.strip())
    return [p for p in parts if p]


def parse_formula(formula):
    dv, rhs = [s.strip() for s in formula.split("~", 1)]
    fixed, randoms = [], []
    for term in _split_top_level(rhs):
        m = re.fullmatch(r"\((.+)\|(.+)\)", term)
        if m:
            randoms.append((m.group(1).strip(), m.group(2).strip()))
        else:
            fixed.append(term)
    return dv, " + ".join(fixed) or "1", randoms

class _NumpyFit:
    """REML/ML result for models with random intercepts only; mimics the statsmodels attributes used below."""

    def __init__(self, names, beta, cov, subject_var, vcomp, scale, llf, n_obs):
        self.fe_params = pd.Series(beta, index=names)
        self._cov = pd.DataFrame(cov, index=names, columns=names)
        self.cov_re = np.atleast_2d(np.asarray(subject_var, dtype=float)) if subject_var is not None else np.zeros((0, 0))
        self.vcomp = np.asarray(vcomp, dtype=float)
        self.scale, self.llf, self.nobs = float(scale), float(llf), int(n_obs)

    def cov_params(self):
        return self._cov

    def summary(self):
        lines = ["Fixed effects (GLS estimates)", ""]
        for name, b, se in zip(self.fe_params.index, self.fe_params.values, np.sqrt(np.diag(self._cov.values))):
            lines.append(f"{name:40s} {b:12.5f}  SE {se:10.5f}")
        lines += ["", "Variance components"]
        if self.cov_re.size:
            lines.append(f"{'Subject':40s} {self.cov_re[0, 0]:12.5f}")
        for i, v in enumerate(self.vcomp):
            lines.append(f"{'random intercept ' + str(i + 1):40s} {v:12.5f}")
        lines.append(f"{'Residual':40s} {self.scale:12.5f}")
        return "\n".join(lines)


def _fit_numpy(df, dv, fixed, randoms, reml):
    """REML/ML for random intercepts, (1|Subject:Factor) terms and one unstructured (lhs | Subject) term.

    Subjects are independent, so V is block diagonal (one small block per subject).
    """
    from patsy import dmatrices
    from scipy.optimize import minimize
    y, X = dmatrices(f"{dv} ~ {fixed}", df, return_type="dataframe")
    d = df.loc[y.index]
    yv, Xv, names = y.to_numpy().ravel(), X.to_numpy(), list(X.columns)
    n, p = Xv.shape
    subject_terms = [lhs for lhs, g in randoms if g == "Subject"]
    factors = [g.split(":", 1)[1] for _, g in randoms if g.startswith("Subject:")]
    if len(subject_terms) > 1:
        raise ValueError("Only one (… | Subject) term is supported.")
    if subject_terms:
        Zall = dmatrices(f"{dv} ~ {subject_terms[0]}", df, return_type="dataframe")[1].loc[y.index].to_numpy()
    else:
        Zall = np.zeros((n, 0))
    q = Zall.shape[1]
    tril = np.tril_indices(q)
    subj = d["Subject"].to_numpy()
    groups = [np.where(subj == s)[0] for s in pd.unique(subj)]
    same = [[(d[f].to_numpy()[ix][:, None] == d[f].to_numpy()[ix][None, :]).astype(float) for f in factors]
            for ix in groups]
    n_chol, n_vc = len(tril[0]), len(factors)

    def psi_of(theta):
        L = np.zeros((q, q))
        L[tril] = theta[:n_chol]
        return L @ L.T

    def pieces(theta):
        Psi = psi_of(theta)
        s_vc = np.exp(theta[n_chol:n_chol + n_vc])
        e2 = np.exp(theta[-1])
        A, bvec, yy, logdet = np.zeros((p, p)), np.zeros(p), 0.0, 0.0
        for ix, mats in zip(groups, same):
            Vg = e2 * np.eye(len(ix))
            if q:
                Zg = Zall[ix]
                Vg = Vg + Zg @ Psi @ Zg.T
            for sv, C in zip(s_vc, mats):
                Vg = Vg + sv * C
            Xg, yg = Xv[ix], yv[ix]
            logdet += np.linalg.slogdet(Vg)[1]
            Vx, Vy = np.linalg.solve(Vg, Xg), np.linalg.solve(Vg, yg)
            A += Xg.T @ Vx
            bvec += Xg.T @ Vy
            yy += yg @ Vy
        return Psi, s_vc, e2, A, bvec, yy, logdet

    def crit(theta):
        try:
            Psi, s_vc, e2, A, bvec, yy, logdet = pieces(theta)
            beta = np.linalg.solve(A, bvec)
            val = logdet + (yy - bvec @ beta) + (np.linalg.slogdet(A)[1] if reml else 0.0)
        except np.linalg.LinAlgError:
            return 1e12
        return val if np.isfinite(val) else 1e12

    k = n_chol + n_vc + 1
    base = np.log(max(np.var(yv), 1e-12))
    diag_pos = [i for i, (r_, c_) in enumerate(zip(*tril)) if r_ == c_]
    best = None
    if q > 1:                                        # random slopes: gradient-based, one start plus a polish
        x0 = np.zeros(k)
        x0[diag_pos] = np.sqrt(np.exp(base - 2.5))
        x0[n_chol:] = base - 2.5
        best = minimize(crit, x0, method="BFGS", options={"maxiter": 3000, "gtol": 1e-8})
        for _ in range(2):
            res = minimize(crit, best.x, method="BFGS", options={"maxiter": 3000, "gtol": 1e-9})
            if res.fun < best.fun - 1e-10:
                best = res
            else:
                break
    else:
        for shift in (-1.0, -2.5, -4.0):
            x0 = np.zeros(k)
            x0[diag_pos] = np.sqrt(np.exp(base + shift))
            x0[n_chol:] = base + shift
            res = minimize(crit, x0, method="Nelder-Mead",
                           options={"xatol": 1e-9, "fatol": 1e-11, "maxiter": 8000, "maxfev": 16000})
            if best is None or res.fun < best.fun:
                best = res
    Psi, s_vc, e2, A, bvec, yy, logdet = pieces(best.x)
    cov = np.linalg.inv(A)
    beta = cov @ bvec
    const = (n - p if reml else n) * np.log(2 * np.pi)
    llf = -0.5 * (const + best.fun)
    return _NumpyFit(names, beta, cov, Psi if q else None, s_vc, e2, llf, n)

def _fit(df, formula, reml=True):
    dv, fixed, randoms = parse_formula(formula)
    if randoms and sum(grp == "Subject" for _, grp in randoms) <= 1 and all(
            grp == "Subject" or (grp.startswith("Subject:") and lhs.replace(" ", "") == "1")
            for lhs, grp in randoms):
        return None, _fit_numpy(df, dv, fixed, randoms, reml), dv, fixed
    import statsmodels.formula.api as smf
    re_formula, vc, has_group = None, {}, False
    for lhs, grp in randoms:
        if grp == "Subject":
            has_group = True
            if lhs.replace(" ", "") != "1":
                re_formula = lhs
        elif grp.startswith("Subject:"):
            other = grp.split(":", 1)[1]
            vc[grp] = f"0 + C({other})"
        else:
            raise ValueError(f"Unsupported random-effect grouping '{grp}' (use Subject or Subject:<factor>).")
    if not has_group:
        re_formula = "0"
    elif re_formula is None:
        re_formula = "1"   # explicit Subject random intercept; statsmodels drops it when vc_formula is used
    model = smf.mixedlm(f"{dv} ~ {fixed}", df, groups=df["Subject"], re_formula=re_formula,
                        vc_formula=vc or None)
    return model, model.fit(reml=reml), dv, fixed


def is_singular(res, tol=1e-4):
    scale = float(res.scale)
    if scale <= 0:
        return True
    cov = np.atleast_2d(np.asarray(res.cov_re))
    eig = np.linalg.eigvalsh(cov) if cov.size else np.array([1.0])
    vc = np.asarray(getattr(res, "vcomp", []), dtype=float)
    return bool(np.any(np.sqrt(np.clip(eig, 0, None) / scale) < tol) or np.any(np.sqrt(np.clip(vc, 0, None) / scale) < tol))


def _design(df, dv, fixed):
    from patsy import dmatrices
    _, X = dmatrices(f"{dv} ~ {fixed}", df, return_type="dataframe")
    return X.design_info


def type3_wald(res, info):
    k = len(info.column_names)
    beta = np.asarray(res.fe_params, dtype=float)[:k]
    V = np.asarray(res.cov_params(), dtype=float)[:k, :k]
    rows = []
    for term, sl in info.term_name_slices.items():
        b, v = beta[sl], V[sl, sl]
        chi2 = float(b @ np.linalg.solve(v, b))
        df_ = len(b)
        rows.append({"term": "(Intercept)" if term == "Intercept" else term,
                     "Chisq": chi2, "Df": df_, "Pr(>Chisq)": float(stats.chi2.sf(chi2, df_))})
    return rows


def error_df(df, formula, contrast):
    """Error df R/emmeans (Satterthwaite) reports for the paper's default model, or None when unknown.

    Verified for: Entropy ~ Electrode * Task + (1|Subject) + (1|Subject:Task), contrast Electrode|Task,
    complete balanced data -> (subjects - 1) * (electrodes - 1) * tasks.
    """
    dv, fixed, randoms = parse_formula(formula)
    groups = sorted(g for _, g in randoms)
    intercepts = all(lhs.replace(" ", "") == "1" for lhs, _ in randoms)
    if not (contrast == "Electrode|Task" and intercepts and groups == ["Subject", "Subject:Task"]
            and fixed.replace(" ", "") in ("Electrode*Task", "Task*Electrode")):
        return None
    counts = df.groupby(["Subject", "Electrode", "Task"], observed=True).size()
    n_s, n_e, n_t = df["Subject"].nunique(), df["Electrode"].nunique(), df["Task"].nunique()
    if len(counts) != n_s * n_e * n_t or (counts != 1).any():
        return None
    return (n_s - 1) * (n_e - 1) * n_t


def posthoc(res, info, df, contrast, correction, dof=None):
    from patsy import build_design_matrices
    k = len(info.column_names)
    beta = np.asarray(res.fe_params, dtype=float)[:k]
    V = np.asarray(res.cov_params(), dtype=float)[:k, :k]
    sigma = float(np.sqrt(res.scale))
    factors = {c: list(df[c].cat.categories) for c in ("Electrode", "Task")}
    if contrast not in {"Electrode", "Electrode|Task", "Task|Electrode"}:
        raise ValueError("contrast must be Electrode, Electrode|Task, or Task|Electrode.")
    if correction not in {"bonferroni", "tukey", "holm", "fdr"}:
        raise ValueError("correction must be bonferroni, tukey, holm, or fdr.")
    main, _, by = contrast.partition("|")
    main, by = main.strip(), by.strip()
    other = [c for c in factors if c not in (main, by)]

    # t distribution when the error df is known (matches emmeans), otherwise the normal approximation
    two_sided = (lambda z: float(2 * stats.t.sf(abs(z), dof))) if dof else (lambda z: float(2 * stats.norm.sf(abs(z))))
    crit_for = (lambda tail: float(stats.t.isf(tail, dof))) if dof else (lambda tail: float(stats.norm.isf(tail)))
    df_label = float(dof) if dof else "Inf"
    q_df = float(dof) if dof else np.inf

    def design_row(levels):
        data = pd.DataFrame([{**levels}])
        return np.asarray(build_design_matrices([info], data)[0])[0]

    def emm(main_level, by_level):
        combos = [dict(zip(other, c)) for c in itertools.product(*[factors[o] for o in other])] or [{}]
        base = {main: main_level, **({by: by_level} if by else {})}
        return np.mean([design_row({**base, **c}) for c in combos], axis=0)

    out = []
    for b in (factors[by] if by else [None]):
        comps = list(itertools.combinations(factors[main], 2))
        m = len(comps)
        family = []
        for a1, a2 in comps:
            L = emm(a1, b) - emm(a2, b)
            est, se = float(L @ beta), float(np.sqrt(max(0.0, L @ V @ L)))
            z = est / se if se > 0 else (np.inf if est > 0 else -np.inf if est < 0 else 0.0)
            family.append({"contrast": f"{a1} - {a2}", "estimate": est, "SE": se, "df": df_label,
                           "z.ratio": z, "p_raw": two_sided(z)})

        raw = np.array([r["p_raw"] for r in family], dtype=float)
        if correction == "bonferroni":
            adjusted = np.minimum(1.0, raw * max(m, 1))
            crit = crit_for(0.025 / max(m, 1))
        elif correction == "holm":
            order = np.argsort(raw)
            values = np.empty(m)
            running = 0.0
            for rank, idx in enumerate(order):
                running = max(running, (m - rank) * raw[idx])
                values[idx] = min(1.0, running)
            adjusted = values
            crit = crit_for(0.025 / max(m, 1))            # conservative simultaneous intervals
        elif correction == "fdr":
            order = np.argsort(raw)
            values = np.empty(m)
            running = 1.0
            for rank in range(m - 1, -1, -1):
                idx = order[rank]
                running = min(running, raw[idx] * m / (rank + 1))
                values[idx] = min(1.0, running)
            adjusted = values
            crit = crit_for(0.025)                         # BH-adjusted p-values; pointwise 95% intervals
        else:  # Tukey HSD using the studentized-range distribution
            q = np.array([abs(r["z.ratio"]) * np.sqrt(2.0) for r in family])
            adjusted = np.asarray(stats.studentized_range.sf(q, len(factors[main]), q_df), dtype=float)
            crit = float(stats.studentized_range.ppf(0.95, len(factors[main]), q_df) / np.sqrt(2.0))

        for r, p_adj in zip(family, adjusted):
            row = {k2: v for k2, v in r.items() if k2 != "p_raw"}
            row.update({"lower.CL": r["estimate"] - crit * r["SE"], "upper.CL": r["estimate"] + crit * r["SE"],
                        "p.value": float(p_adj), "Cohen's d": r["estimate"] / sigma})
            if by:
                row = {by: b, **row}
            out.append(row)
    return out


def fit_python(df, params):
    df = prepare(df)
    dv = params.get("dv", "Entropy")
    formula = params.get("formula") or f"{dv} ~ Electrode * Task + (1 | Subject) + (1 | Subject:Task)"
    contrast = params.get("contrast", "Electrode|Task")
    if dv not in df.columns:
        raise ValueError(f"Dependent-variable column '{dv}' was not found in the stats dataset.")
    df[dv] = pd.to_numeric(df[dv], errors="coerce")
    df = df.dropna(subset=[dv])
    if df.empty:
        raise ValueError(f"Dependent-variable column '{dv}' contains no numeric values.")
    model, res, dv, fixed = _fit(df, formula, reml=True)
    info = _design(df, dv, fixed)
    dof = error_df(df, formula, contrast)
    note = ("Post-hoc tests use the t distribution with the error df that R/emmeans reports for this design."
            if dof else
            "Post-hoc tests use a normal approximation (R/emmeans uses Satterthwaite df), so p-values are slightly too small.")
    return {"singular": is_singular(res),
            "summary": f"Linear mixed model fit by REML\nFormula: {formula}\n\n{res.summary()}",
            "anova": type3_wald(res, info),
            "posthoc": posthoc(res, info, df, contrast, params.get("correction", "bonferroni"), dof),
            "warnings": [note, "Type III terms are Wald chi-square tests (car::Anova), not Satterthwaite F tests."]}


def compare_python(df, params):
    df = prepare(df)
    dv = params.get("dv", "Entropy")
    use_reml = bool(params.get("reml", True))      # True = lmer default, as in the paper's script
    rows = []
    for name, tmpl in CANDIDATES.items():
        formula = tmpl.format(dv=dv)
        try:
            model, res, _, fixed = _fit(df, formula, reml=use_reml)
            cov = np.atleast_2d(np.asarray(res.cov_re))
            n_par = len(res.fe_params) + (cov.shape[0] * (cov.shape[0] + 1)) // 2 * (cov.size > 0) \
                + len(getattr(res, "vcomp", [])) + 1
            ll = float(res.llf)
            rows.append({"Model": name, "formula": formula, "AIC": -2 * ll + 2 * n_par,
                         "BIC": -2 * ll + n_par * np.log(len(df)), "LogLik": ll,
                         "note": "singular fit (kept)" if is_singular(res) else ""})
        except Exception as exc:  # noqa: BLE001 - mirrors the R tryCatch
            rows.append({"Model": name, "formula": formula, "AIC": None, "BIC": None,
                         "LogLik": None, "note": f"failed: {exc}"})
    rows.sort(key=lambda r: (r["AIC"] is None, r["AIC"] if r["AIC"] is not None else 0))
    fit_kind = "REML (as in the paper's script)" if use_reml else "maximum likelihood (ML)"
    return {"compare": rows,
            "warnings": [f"Candidate models were fitted by {fit_kind}. Singular fits are kept and flagged "
                         "(the paper's R script drops them)."]}


R_SCRIPT = """
suppressMessages({{library(lme4);library(lmerTest);library(car);library(emmeans)}})
d <- read.csv("data.csv", stringsAsFactors=FALSE)
d$Subject <- as.factor(d$Subject); d$Electrode <- as.factor(d$Electrode); d$Task <- as.factor(d$Task)
m <- lmer({formula}, data=d)
cat(isSingular(m), file="singular.txt")
sink("summary.txt"); print(summary(m)); sink()
a <- as.data.frame(Anova(m, type="III")); a$term <- rownames(a); write.csv(a, "anova.csv", row.names=FALSE)
e <- emmeans(m, ~ {spec})
p <- as.data.frame(summary(contrast(e, method="pairwise", adjust="{adj}", infer=c(TRUE,TRUE))))
p$cohens_d <- p$estimate / sigma(m); write.csv(p, "posthoc.csv", row.names=FALSE)
"""


def fit_r(df, params):
    if not shutil.which("Rscript"):
        raise ValueError("Rscript was not found on the server; use engine 'python'.")
    df = prepare(df)
    dv = params.get("dv", "Entropy")
    formula = params.get("formula") or f"{dv} ~ Electrode * Task + (1 | Subject) + (1 | Subject:Task)"
    with tempfile.TemporaryDirectory() as tmp:
        df.to_csv(os.path.join(tmp, "data.csv"), index=False)
        script = R_SCRIPT.format(formula=formula, spec=params.get("contrast", "Electrode|Task"),
                                 adj=params.get("correction", "bonferroni"))
        done = subprocess.run(["Rscript", "-e", script], cwd=tmp, capture_output=True, text=True, timeout=600)
        if done.returncode != 0:
            raise ValueError("R failed: " + done.stderr[-800:])
        read = lambda n: open(os.path.join(tmp, n)).read()
        return {"singular": read("singular.txt").strip() == "TRUE", "summary": read("summary.txt"),
                "anova": pd.read_csv(os.path.join(tmp, "anova.csv")).to_dict("records"),
                "posthoc": pd.read_csv(os.path.join(tmp, "posthoc.csv")).to_dict("records")}


def run(df, action, params):
    if action == "compare":
        return compare_python(df, params)
    return fit_r(df, params) if params.get("engine") == "r" else fit_python(df, params)
