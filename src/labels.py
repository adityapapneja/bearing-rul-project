import numpy as np
import pandas as pd
from src.health import RUNS


def run_table(df, s, b, feature="rms", win=15, lo=0.15, hi=0.40):
    """Per-snapshot table for one failed bearing: smoothed HI, ratio to baseline, hours to end of test."""
    g = df[(df["set"] == s) & (df["bearing"] == b)]
    g = g[g["rms"] > 0.2 * g["rms"].median()]                # drop dead/truncated snapshots
    u = g.groupby("idx").agg(timestamp=("timestamp", "first"), value=(feature, "mean"))
    u["value"] = u["value"].rolling(win, min_periods=1).median()   # causal smoothing
    u["life"] = np.arange(len(u)) / (len(u) - 1)
    u["rul_hours"] = (u["timestamp"].iloc[-1] - u["timestamp"]).dt.total_seconds() / 3600
    op = op_hours(u)
    u["rul_op_hours"] = op.iloc[-1] - op
    base = u.loc[(u["life"] >= lo) & (u["life"] < hi), "value"].median()
    u["ratio"] = u["value"] / base
    return u.reset_index()


def find_onset(u, ratio_thr=1.25, n_consec=20, start=0.40):
    """Position of the first snapshot of the first run of n_consec snapshots above ratio_thr."""
    ok = ((u["ratio"] > ratio_thr) & (u["life"] >= start)).astype(int)
    hit = (ok.rolling(n_consec).sum() == n_consec).to_numpy()
    return int(hit.argmax()) - n_consec + 1 if hit.any() else None

def op_hours(u, max_step_min=15):
    """Cumulative operating time: any gap longer than max_step_min counts as max_step_min."""
    dt = u["timestamp"].diff().dt.total_seconds().div(3600).fillna(0)
    return dt.clip(upper=max_step_min / 60).cumsum()