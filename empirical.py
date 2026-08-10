"""Residual treatment variation in three published staggered designs.

Datasets are the public replication extracts distributed with the `did`
(Callaway and Sant'Anna) and `bacondecomp` (Goodman-Bacon) R packages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import rdata

from download_data import ensure

# Downloads and checksum-verifies the three .rda extracts on first run; a
# no-op once ./data holds valid copies.  Run download_data.py directly if the
# machine needs a proxy or has no outbound network access.
DATA = ensure(quiet=True)


def load(name: str) -> pd.DataFrame:
    df = list(rdata.read_rda(DATA / name).values())[0]
    df.columns = [str(c) for c in df.columns]
    return df


def bound(T: int) -> float:
    return (T**2 - 1.0) / (12.0 * T**2)


def diagnostics(panel: pd.DataFrame, unit: str, time: str, treat: str) -> dict:
    """V by explicit two-way residualization on observed cells."""
    wide = panel.pivot_table(index=unit, columns=time, values=treat)
    D = wide.to_numpy(dtype=float)
    O = ~np.isnan(D)
    N, T = D.shape
    M = int(O.sum())

    # two-way projection on observed cells via SVD (rank deficient by one)
    cells = np.argwhere(O)
    X = np.zeros((len(cells), N + T))
    X[np.arange(len(cells)), cells[:, 0]] = 1.0
    X[np.arange(len(cells)), N + cells[:, 1]] = 1.0
    d = D[O]
    U, sv, _ = np.linalg.svd(X, full_matrices=False)
    keep = sv > sv.max() * 1e-10
    resid = d - U[:, keep] @ (U[:, keep].T @ d)
    V = float(resid @ resid / M)

    return {"N": N, "T": T, "M": M, "balanced": M == N * T, "V": V,
            "bound": bound(T), "pct": 100 * V / bound(T),
            "se_ratio": np.sqrt(bound(T) / V), "n_mult": bound(T) / V,
            "MV": M * V}


rows = []

# ---------------------------------------------------------------- minimum wage
mw = load("mpdta.rda")
mw["D"] = ((mw["first.treat"] > 0) & (mw["year"] >= mw["first.treat"])).astype(float)
rows.append(("Minimum wage, counties (Callaway--Sant'Anna 2021)",
             diagnostics(mw, "countyreal", "year", "D")))
print("minimum wage cohort counts (counties):")
print(mw.groupby("first.treat").countyreal.nunique().to_string(), "\n")

# ------------------------------------------------------------- castle doctrine
cd = load("castle.rda")
cd["D"] = ((cd["effyear"].notna()) & (cd["year"] >= cd["effyear"])).astype(float)
rows.append(("Castle doctrine, states (Cheng--Hoekstra 2013)",
             diagnostics(cd, "state", "year", "D")))
print("castle doctrine adoption years (states):")
print(cd.drop_duplicates("state").effyear.value_counts(dropna=False).sort_index().to_string(), "\n")

# ----------------------------------------------------------- unilateral divorce
dv = load("divorce.rda")
dv = dv[dv.sex == 2].copy()                       # female suicide sample
dv["D"] = (dv["year"] >= dv["divyear"]).astype(float)
rows.append(("Unilateral divorce, states (Goodman-Bacon 2021)",
             diagnostics(dv, "st", "year", "D")))

# same design with the always-treated states removed
always = dv.groupby("st")["D"].min()
dv_no_always = dv[dv.st.isin(always[always == 0].index)]
rows.append(("\\quad excluding always-treated states",
             diagnostics(dv_no_always, "st", "year", "D")))
print("divorce reform years (states):")
print(dv.drop_duplicates("st").divyear.value_counts(dropna=False).sort_index().to_string(), "\n")

# --------------------------------------------------------------------- report
print("=" * 100)
hdr = (f"{'design':<52}{'N':>5}{'T':>4}{'V':>10}{'% bound':>10}"
       f"{'SE ratio':>10}{'N mult.':>10}")
print(hdr)
print("-" * 100)
for name, r in rows:
    tag = "" if r["balanced"] else "*"
    print(f"{name:<52}{r['N']:5d}{r['T']:4d}{r['V']:10.4f}"
          f"{r['pct']:9.1f}%{r['se_ratio']:10.2f}{r['n_mult']:10.2f}{tag}")
print("* unbalanced; V is the per-observation quantity of Proposition 3")

# ------------------------------------------------------------- LaTeX fragment
print("\n" + "=" * 100)
print("LaTeX table body:\n")
for name, r in rows:
    print(f"{name} & {r['N']} & {r['T']} & {r['V']:.4f} & "
          f"{r['pct']:.1f} & {r['n_mult']:.2f} \\\\")
