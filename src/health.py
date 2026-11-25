import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# the four run-to-failure trajectories
RUNS = [("set1", "b3"), ("set1", "b4"), ("set2", "b1"), ("set3", "b3")]


def prepare(df, features, win=15, lo=0.15, hi=0.40, burn_in=0.12):
    runs = {}
    for s, b in RUNS:
        g = df[(df["set"] == s) & (df["bearing"] == b)]
        g = g[g["rms"] > 0.2 * g["rms"].median()]
        u = g.groupby("idx")[features].mean()
        u = u.rolling(win, min_periods=1).median()
        life = np.arange(len(u)) / (len(u) - 1)
        base = u[(life >= lo) & (life < hi)]
        med = base.median()
        sig = (1.4826 * (base - med).abs().median()).replace(0, np.nan).fillna(1.0)
        z = (u - med) / sig
        z["life"] = life
        runs[f"{s}_{b}"] = z[z["life"] >= burn_in]      # discard the run-in segment
    return runs


def squash(z):
    """Log-compress z so a strong-signature run (Set 2) doesn't dominate the metrics."""
    return np.sign(z) * np.log1p(np.abs(z))


def hi_metrics(runs, feature, n_grid=200):
    zs = [squash(r[feature].to_numpy()) for r in runs.values()]
    lives = [r["life"].to_numpy() for r in runs.values()]

    rho = [spearmanr(l, z)[0] for l, z in zip(lives, zs)]
    mono = abs(np.mean(rho))                       # opposite trends across runs cancel out

    grid = np.linspace(max(l[0] for l in lives), 1, n_grid)
    curves = np.array([np.interp(grid, l, z) for l, z in zip(lives, zs)])
    C = np.corrcoef(curves)
    trend = max(0.0, np.min(C[np.triu_indices(len(zs), 1)]))   # signed: opposite curves score 0

    final = np.array([z[-max(5, len(z) // 20):].mean() for z in zs])
    prog = np.exp(-final.std() / (abs(final.mean()) + 1e-9))
    return mono, trend, prog


def alarm_life(r, feature, thr=5.0, n_consec=10):
    """Fraction of life at which the feature has been above thr for n_consec snapshots in a row."""
    hit = (r[feature] > thr).astype(int).rolling(n_consec).sum() == n_consec
    return float(r["life"][hit].iloc[0]) if hit.any() else np.nan


def summarize(runs, features):
    rows = []
    for f in features:
        mono, trend, prog = hi_metrics(runs, f)
        row = {"feature": f, "mono": mono, "trend": trend, "prog": prog}
        for name, r in runs.items():
            row[f"alarm_{name}"] = alarm_life(r, f)
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature").round(2)