"""
Remaining Useful Life (RUL) prediction models for the NASA/IMS bearing dataset.

All models share a run-aware interface suitable for Leave-One-Run-Out (LORO) CV:
    model.fit(train_runs)  – train_runs is a list of (X_run, y_run) tuples;
                             scalers are fit strictly on training runs only, and
                             sliding windows are built strictly within each run.
    model.predict(X_run)   – returns (N,) predicted RUL array.

Model families:
  Baselines:
    HIBaseline      – naive linear mapping from the primary Health Indicator (rms z-score)
    RidgeRUL        – StandardScaler + Ridge regression

  Classical ML:
    RandomForestRUL – Random Forest Regressor
    XGBoostRUL      – XGBoost Regressor (falls back to HistGradientBoosting if absent)
    LightGBMRUL     – LightGBM Regressor (falls back to HistGradientBoosting if absent)
    SVRRUL          – StandardScaler + RBF Support Vector Regression

  Deep Learning (PyTorch):
    MLPRUL          – Feedforward 3-layer MLP on single snapshots
    CNN1DRUL        – 1-D CNN on causal sliding windows of W=20 snapshots
    LSTMRUL         – 2-layer LSTM on causal sliding windows
    GRURUL          – 2-layer GRU on causal sliding windows
    CNNLSTMRUL      – 1-D Conv front-end + LSTM on causal sliding windows

  Degradation-model based:
    ExpDegradationRUL – Exponential HI growth model with online causal projection
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _stack_runs(train_runs):
    """Concatenate a list of (X, y) run tuples into (X_all, y_all)."""
    Xs = [np.asarray(X, dtype=np.float32) for X, _ in train_runs]
    ys = [np.asarray(y, dtype=np.float32) for _, y in train_runs]
    return np.vstack(Xs), np.concatenate(ys)


def _causal_windows(X: np.ndarray, W: int) -> np.ndarray:
    """
    Build causal sliding windows of length W within a SINGLE run.
    Returns shape (N, W, D); earlier rows are edge-padded with the first snapshot
    so the network never sees artificial zero-steps during early life.
    """
    N, D = X.shape
    out = np.empty((N, W, D), dtype=np.float32)
    for t in range(N):
        start = max(0, t - W + 1)
        seg = X[start : t + 1]
        k = len(seg)
        if k < W:
            out[t, : W - k] = X[0]
        out[t, W - k :] = seg
    return out


def _build_run_windows(train_runs, scaler, W: int):
    """Scale each run using `scaler` and build per-run windows without crossing boundaries."""
    win_list, y_list = [], []
    for X, y in train_runs:
        Xs = scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        win_list.append(_causal_windows(Xs, W))
        y_list.append(np.asarray(y, dtype=np.float32))
    return np.vstack(win_list), np.concatenate(y_list)


# ═══════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════

class HIBaseline:
    """
    Naive Health-Indicator baseline: maps the primary HI (column `hi_col`, default 0 = rms)
    linearly to RUL using log1p(|HI|) * sign(HI) on the training runs.
    """
    name = "hi_baseline"

    def __init__(self, hi_col=0):
        self.hi_col = hi_col

    def _hi_feat(self, X):
        z = np.asarray(X, dtype=np.float64)[:, self.hi_col]
        return (np.sign(z) * np.log1p(np.abs(z))).reshape(-1, 1)

    def fit(self, train_runs):
        X_all, y_all = _stack_runs(train_runs)
        self.reg_ = LinearRegression().fit(self._hi_feat(X_all), y_all)
        return self

    def predict(self, X):
        return self.reg_.predict(self._hi_feat(X))


class RidgeRUL:
    """StandardScaler + Ridge regression baseline."""
    name = "ridge"

    def __init__(self, alpha=50.0):
        self.alpha = alpha

    def fit(self, train_runs):
        X_all, y_all = _stack_runs(train_runs)
        self.sc_ = StandardScaler().fit(X_all)
        self.reg_ = Ridge(alpha=self.alpha).fit(self.sc_.transform(X_all), y_all)
        return self

    def predict(self, X):
        return self.reg_.predict(self.sc_.transform(np.asarray(X, dtype=np.float32)))


# ═══════════════════════════════════════════════════════════════════════
# Classical ML
# ═══════════════════════════════════════════════════════════════════════

class RandomForestRUL:
    """Random Forest Regressor on snapshot features."""
    name = "random_forest"

    def __init__(self, n_estimators=200, max_depth=12, min_samples_leaf=5, seed=0):
        self.m = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
            n_jobs=-1,
        )

    def fit(self, train_runs):
        X_all, y_all = _stack_runs(train_runs)
        self.m.fit(X_all, y_all)
        return self

    def predict(self, X):
        return self.m.predict(np.asarray(X, dtype=np.float32))


class XGBoostRUL:
    """XGBoost Regressor (falls back to HistGradientBoostingRegressor if unavailable)."""
    name = "xgboost"

    def __init__(self, n_estimators=200, max_depth=5, lr=0.05, seed=0):
        if _HAS_XGB:
            self.m = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
                verbosity=0,
            )
        else:
            self.m = HistGradientBoostingRegressor(
                max_iter=n_estimators, max_depth=max_depth,
                learning_rate=lr, random_state=seed,
            )

    def fit(self, train_runs):
        X_all, y_all = _stack_runs(train_runs)
        self.m.fit(X_all, y_all)
        return self

    def predict(self, X):
        return self.m.predict(np.asarray(X, dtype=np.float32))


class LightGBMRUL:
    """LightGBM Regressor (falls back to HistGradientBoostingRegressor if unavailable)."""
    name = "lightgbm"

    def __init__(self, n_estimators=200, max_depth=5, lr=0.05, seed=0):
        if _HAS_LGBM:
            self.m = LGBMRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=lr,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            self.m = HistGradientBoostingRegressor(
                max_iter=n_estimators, max_depth=max_depth,
                learning_rate=lr, random_state=seed,
            )

    def fit(self, train_runs):
        X_all, y_all = _stack_runs(train_runs)
        self.m.fit(X_all, y_all)
        return self

    def predict(self, X):
        return self.m.predict(np.asarray(X, dtype=np.float32))


class SVRRUL:
    """
    StandardScaler + RBF Support Vector Regression.
    Subsamples training snapshots deterministically to `max_train` to keep O(N^2-N^3)
    training fast when Set 3 (~5,500 snapshots) is in the training fold.
    """
    name = "svr"

    def __init__(self, C=10.0, epsilon=1.0, max_train=2500, seed=0):
        self.C = C
        self.epsilon = epsilon
        self.max_train = max_train
        self.seed = seed

    def fit(self, train_runs):
        X_all, y_all = _stack_runs(train_runs)
        self.sc_ = StandardScaler().fit(X_all)
        Xs = self.sc_.transform(X_all)
        if len(Xs) > self.max_train:
            rng = np.random.default_rng(self.seed)
            idx = np.sort(rng.choice(len(Xs), size=self.max_train, replace=False))
            Xs, y_all = Xs[idx], y_all[idx]
        self.m_ = SVR(kernel="rbf", C=self.C, epsilon=self.epsilon)
        self.m_.fit(Xs, y_all)
        return self

    def predict(self, X):
        return self.m_.predict(self.sc_.transform(np.asarray(X, dtype=np.float32)))


# ═══════════════════════════════════════════════════════════════════════
# Degradation-model based
# ═══════════════════════════════════════════════════════════════════════

class ExpDegradationRUL:
    """
    Exponential degradation model on the primary Health Indicator (rms z-score).
    - During fit(), learns the median late-life HI failure level and typical
      exponential growth rate from the training runs.
    - At predict(), slides a causal trailing window over the test run's HI,
      fits a regularized log-linear slope in recent history, and extrapolates
      the remaining snapshots (converted to hours at dt=1/6 h) until the HI
      reaches the learned failure threshold.
    """
    name = "exp_degradation"

    def __init__(self, hi_col=0, win=40, dt_hours=1.0 / 6.0, cap=50.0):
        self.hi_col = hi_col
        self.win = win
        self.dt_hours = dt_hours
        self.cap = cap

    def _to_pos_hi(self, X):
        z = np.asarray(X, dtype=np.float64)[:, self.hi_col]
        # Shift so baseline (z ≈ 0) is ~1.0 and strictly positive for log-fitting
        return np.maximum(1.0 + z, 0.25)

    def fit(self, train_runs):
        fail_levels = []
        for X, y in train_runs:
            hi = self._to_pos_hi(X)
            tail = hi[np.asarray(y) <= 1.0]
            fail_levels.append(np.median(tail) if len(tail) else np.percentile(hi, 98))
        self.hi_fail_ = float(np.median(fail_levels))
        return self

    def predict(self, X):
        hi = self._to_pos_hi(X)
        N = len(hi)
        pred = np.full(N, self.cap, dtype=np.float64)
        log_hi = np.log(hi)
        log_fail = np.log(max(self.hi_fail_, 1.5))

        for t in range(N):
            if hi[t] >= self.hi_fail_:
                pred[t] = 0.0
                continue
            s = max(0, t - self.win + 1)
            seg = log_hi[s : t + 1]
            if len(seg) < 10:
                continue
            x_t = np.arange(len(seg), dtype=np.float64)
            x_c = x_t - x_t.mean()
            denom = np.sum(x_c ** 2) + 1e-9
            slope = np.sum(x_c * (seg - seg.mean())) / denom
            if slope > 1e-4:
                steps_rem = (log_fail - log_hi[t]) / slope
                pred[t] = np.clip(steps_rem * self.dt_hours, 0.0, self.cap)
        return pred


# ═══════════════════════════════════════════════════════════════════════
# Deep Learning (PyTorch)
# ═══════════════════════════════════════════════════════════════════════

if _HAS_TORCH:

    def _train_regressor(net, X_tensor, y_tensor, epochs, lr, batch_size, seed):
        torch.manual_seed(seed)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()
        dl = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                pred = net(xb).squeeze(-1)
                loss_fn(pred, yb).backward()
                opt.step()
        net.eval()

    class _MLPNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return self.net(x)

    class _CNN1DNet(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(32, 1)

        def forward(self, x):
            # x: (B, W, D) -> (B, D, W)
            z = self.conv(x.transpose(1, 2)).squeeze(-1)
            return self.head(z)

    class _RNNNet(nn.Module):
        def __init__(self, in_dim, hidden=32, kind="lstm"):
            super().__init__()
            cls = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = cls(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    class _CNNLSTMNet(nn.Module):
        def __init__(self, in_dim, hidden=32):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_dim, 32, kernel_size=3, padding=1), nn.ReLU(),
            )
            self.lstm = nn.LSTM(32, hidden, num_layers=1, batch_first=True)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            # x: (B, W, D) -> conv (B, 32, W) -> (B, W, 32)
            c = self.conv(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.lstm(c)
            return self.head(out[:, -1, :])

    class MLPRUL:
        """Feedforward 3-layer PyTorch MLP on snapshot features."""
        name = "mlp"

        def __init__(self, epochs=40, lr=1e-3, batch_size=128, seed=0):
            self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed

        def fit(self, train_runs):
            X_all, y_all = _stack_runs(train_runs)
            self.sc_ = StandardScaler().fit(X_all)
            Xs = self.sc_.transform(X_all).astype(np.float32)
            self.net_ = _MLPNet(Xs.shape[1])
            _train_regressor(
                self.net_, torch.from_numpy(Xs), torch.from_numpy(y_all),
                self.epochs, self.lr, self.batch_size, self.seed,
            )
            return self

        @torch.no_grad()
        def predict(self, X):
            Xs = self.sc_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
            return self.net_(torch.from_numpy(Xs)).squeeze(-1).numpy()

    class _BaseWindowRUL:
        """Shared fit/predict logic for causal sliding-window PyTorch regressors."""
        def __init__(self, window=20, epochs=25, lr=1e-3, batch_size=128, seed=0):
            self.W = window
            self.epochs = epochs
            self.lr = lr
            self.batch_size = batch_size
            self.seed = seed

        def _make_net(self, in_dim):
            raise NotImplementedError

        def fit(self, train_runs):
            X_all, _ = _stack_runs(train_runs)
            self.sc_ = StandardScaler().fit(X_all)
            wins, y_all = _build_run_windows(train_runs, self.sc_, self.W)
            self.net_ = self._make_net(wins.shape[2])
            _train_regressor(
                self.net_, torch.from_numpy(wins), torch.from_numpy(y_all),
                self.epochs, self.lr, self.batch_size, self.seed,
            )
            return self

        @torch.no_grad()
        def predict(self, X):
            Xs = self.sc_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
            wins = _causal_windows(Xs, self.W)
            return self.net_(torch.from_numpy(wins)).squeeze(-1).numpy()

    class CNN1DRUL(_BaseWindowRUL):
        """1-D Convolutional network on causal sliding windows."""
        name = "cnn1d"

        def _make_net(self, in_dim):
            return _CNN1DNet(in_dim)

    class LSTMRUL(_BaseWindowRUL):
        """2-layer LSTM on causal sliding windows."""
        name = "lstm"

        def _make_net(self, in_dim):
            return _RNNNet(in_dim, hidden=32, kind="lstm")

    class GRURUL(_BaseWindowRUL):
        """2-layer GRU on causal sliding windows."""
        name = "gru"

        def _make_net(self, in_dim):
            return _RNNNet(in_dim, hidden=32, kind="gru")

    class CNNLSTMRUL(_BaseWindowRUL):
        """1-D Conv front-end + LSTM on causal sliding windows."""
        name = "cnn_lstm"

        def _make_net(self, in_dim):
            return _CNNLSTMNet(in_dim, hidden=32)
