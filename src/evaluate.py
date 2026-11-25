import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TRAIN = (0.15, 0.32)    # healthy data the model is fit on
CAL = (0.32, 0.40)      # healthy data used only to set the alarm threshold; life >= 0.40 is test

BASE = (0.15, 0.40)


def blocked_threshold(X, make_model, q=0.995, k=4, purge=15):
    """Threshold from out-of-sample scores: fit on k-1 blocks, score the held-out block.
    `purge` drops snapshots next to the held-out block (they overlap through the rolling smoothing)."""
    n = len(X); idx = np.arange(n); held = []
    for b in np.array_split(idx, k):
        train = idx[(idx < b[0] - purge) | (idx > b[-1] + purge)]
        held.append(make_model().fit(X[train]).score(X[b]))
    return float(np.quantile(np.concatenate(held), q))


def alarm_blocked(Z, features, make_model, base=BASE, test_from=0.40, q=0.995, n_consec=10, k=4):
    life = Z["life"].to_numpy(); X = Z[features].to_numpy()
    b = (life >= base[0]) & (life < base[1])
    te = life >= test_from
    thr = blocked_threshold(X[b], make_model, q, k)
    s = make_model().fit(X[b]).score(X)
    pos = confirmed_alarm(s[te], thr, n_consec)
    return {
        "alarm_life": np.nan if pos is None else float(life[np.flatnonzero(te)[pos]]),
        "start_above": float(np.mean(s[te][:50] > thr)),   # ~1.0 means "alarm at first opportunity"
    }, s, thr

def confirmed_alarm(score, thr, n_consec=10):
    """Position where an alarm is *confirmed*: the n_consec-th consecutive snapshot above thr.
    A deployed system can't alarm earlier, so lead time is measured from here."""
    above = pd.Series(np.asarray(score) > thr).astype(int)
    hit = (above.rolling(n_consec).sum() == n_consec).to_numpy()
    return int(hit.argmax()) if hit.any() else None


def evaluate_run(Z, features, model, y_tight, y_loose, rul_hours, q=0.995, n_consec=10):
    life = Z["life"].to_numpy()
    X = Z[features].to_numpy()
    tr = (life >= TRAIN[0]) & (life < TRAIN[1])
    ca = (life >= CAL[0]) & (life < CAL[1])
    te = life >= CAL[1]

    model.fit(X[tr])
    s = model.score(X)
    thr = float(np.quantile(s[ca], q))

    out = {"model": model.name}
    for tag, y in (("tight", y_tight), ("loose", y_loose)):
        yy = y.reindex(Z.index).to_numpy()
        ok = 0 < yy[te].sum() < te.sum()
        out[f"auc_{tag}"] = roc_auc_score(yy[te], s[te]) if ok else np.nan

    k = confirmed_alarm(s[te], thr, n_consec)
    if k is None:
        out["alarm_life"], out["lead_hours"] = np.nan, np.nan
    else:
        pos = np.flatnonzero(te)[k]
        out["alarm_life"] = life[pos]
        out["lead_hours"] = rul_hours.reindex(Z.index).to_numpy()[pos]

    pre = te & (y_loose.reindex(Z.index).to_numpy() == 0)     # test stretch before the loose onset
    out["pre_onset_alarm_rate"] = float(np.mean(s[pre] > thr)) if pre.any() else np.nan
    return out, s, thr


# ═══════════════════════════════════════════════════════════════════════
# RUL evaluation metrics & Leave-One-Run-Out (LORO) cross-validation
# ═══════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def phm_score(y_true, y_pred, mean=True):
    """
    PHM08 / NASA C-MAPSS asymmetric scoring function.
    d = y_pred - y_true:
      - early prediction (d < 0): exp(-d / 13) - 1
      - late  prediction (d >= 0): exp( d / 10) - 1  (penalizes late predictions more)
    When `mean=True` (default), returns the per-snapshot mean score so runs of
    different lengths are directly comparable; `mean=False` returns the raw sum.
    """
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    d = np.clip(d, -200.0, 200.0)   # guard against overflow on extreme outliers
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.mean(s) if mean else np.sum(s))


def smooth_predictions(y_pred, win=15):
    """Causal rolling-median smoothing of RUL predictions."""
    return pd.Series(y_pred).rolling(win, min_periods=1).median().to_numpy()


def evaluate_rul_loro(runs_dict, make_model, cap=50.0, late_thr=25.0, smooth_win=None):
    """
    Leave-One-Run-Out (LORO) cross-validation across all run-to-failure bearings.

    runs_dict : dict {run_name: (X, y_rul)} where X is (N, D) and y_rul is (N,)
    make_model: zero-arg callable returning a fresh RUL model with .fit(train_runs) / .predict(X)
    Returns (summary_df, predictions_dict) where predictions_dict[run_name] = (y_true, y_pred).
    """
    rows = []
    preds = {}
    run_names = list(runs_dict.keys())
    for test_name in run_names:
        train_runs = [runs_dict[n] for n in run_names if n != test_name]
        X_test, y_test = runs_dict[test_name]

        model = make_model()
        model.fit(train_runs)
        y_pred = np.clip(model.predict(X_test), 0.0, cap)
        if smooth_win is not None and smooth_win > 1:
            y_pred = smooth_predictions(y_pred, win=smooth_win)

        preds[test_name] = (y_test, y_pred)
        late = y_test <= late_thr
        rows.append({
            "model": getattr(model, "name", type(model).__name__),
            "run": test_name,
            "rmse": rmse(y_test, y_pred),
            "mae": mae(y_test, y_pred),
            "phm_mean": phm_score(y_test, y_pred, mean=True),
            "rmse_late": rmse(y_test[late], y_pred[late]) if late.any() else np.nan,
            "mae_late": mae(y_test[late], y_pred[late]) if late.any() else np.nan,
        })
    return pd.DataFrame(rows), preds