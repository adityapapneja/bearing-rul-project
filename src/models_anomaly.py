"""
Anomaly-detection models for the NASA/IMS bearing dataset.

All models share a minimal interface:
    model.fit(X)      – X: (N, n_features) numpy array of **healthy** snapshots
    model.score(X)    – returns (N,) anomaly score; **higher = more anomalous**

Statistical / classical-ML models (sklearn only):
    Mahalanobis   – squared Mahalanobis distance from the healthy mean
    PCAMonitor    – T² or SPE (Q-statistic) from a PCA on healthy data
    IsoForest     – Isolation Forest (negated sklearn score)
    GMMDetector   – negative log-likelihood under a Gaussian Mixture Model
    OCSVMDetector – negated one-class SVM decision function (RBF kernel)
    LOFDetector   – negated LOF score (novelty mode, can score unseen data)

Deep learning autoencoders (PyTorch; fallback to sklearn MLPRegressor if unavailable):
    DenseAutoencoder   – feedforward AE; scores single snapshots
    LSTMAutoencoder    – sequence AE on sliding windows; causal, scores last step
    CNN1DAutoencoder   – 1-D-convolutional AE on sliding windows; scores mean step
"""

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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

def _causal_windows(X: np.ndarray, W: int) -> np.ndarray:
    """
    Build causal sliding windows of length W.
    Returns array of shape (N, W, D); earlier rows are zero-padded.
    """
    N, D = X.shape
    out = np.zeros((N, W, D), dtype=X.dtype)
    for t in range(N):
        start = max(0, t - W + 1)
        seg = X[start : t + 1]           # (len ≤ W, D)
        out[t, W - len(seg) :] = seg     # right-align; leading rows stay 0
    return out


# ═══════════════════════════════════════════════════════════════════════
# Statistical baselines
# ═══════════════════════════════════════════════════════════════════════

class Mahalanobis:
    """Squared Mahalanobis distance from the healthy mean (LedoitWolf covariance)."""
    name = "mahalanobis"

    def fit(self, X):
        self.est = LedoitWolf().fit(X)
        return self

    def score(self, X):
        return self.est.mahalanobis(X)


class PCAMonitor:
    """PCA-based monitor: T² (inside the PC subspace) or SPE (reconstruction error)."""
    def __init__(self, var=0.95, stat="spe"):
        self.var, self.stat = var, stat
        self.name = f"pca_{stat}"

    def fit(self, X):
        self.pca = PCA(n_components=self.var, svd_solver="full").fit(X)
        return self

    def score(self, X):
        T = self.pca.transform(X)
        if self.stat == "t2":
            return np.sum(T ** 2 / self.pca.explained_variance_, axis=1)
        return np.sum((X - self.pca.inverse_transform(T)) ** 2, axis=1)


# ═══════════════════════════════════════════════════════════════════════
# Classical ML – ensemble / kernel / density
# ═══════════════════════════════════════════════════════════════════════

class IsoForest:
    """Isolation Forest (300 trees).  Negated sklearn score → higher = more anomalous."""
    name = "isolation_forest"

    def __init__(self, seed=0):
        self.m = IsolationForest(n_estimators=300, random_state=seed)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class GMMDetector:
    """
    Gaussian Mixture Model: negative log-likelihood as anomaly score.
    Uses full covariance with mild regularisation.  n_components=3 is enough to
    cover the multi-modal structure of the healthy feature distribution across
    bearings.
    """
    name = "gmm"

    def __init__(self, n_components=3, seed=0):
        self.m = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            reg_covar=1e-4,
            random_state=seed,
        )

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)       # higher = less likely under healthy model


class OCSVMDetector:
    """
    One-Class SVM with RBF kernel.
    nu ≈ upper bound on training outlier fraction (kept small so the support
    covers most healthy data).  Input is z-scored before fitting.
    """
    name = "ocsvm"

    def __init__(self, nu=0.05, kernel="rbf", gamma="scale"):
        self.nu, self.kernel, self.gamma = nu, kernel, gamma

    def fit(self, X):
        self.sc_ = StandardScaler().fit(X)
        self.m_ = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        self.m_.fit(self.sc_.transform(X))
        return self

    def score(self, X):
        return -self.m_.score_samples(self.sc_.transform(X))   # +1=normal → flip


class LOFDetector:
    """
    Local Outlier Factor in novelty mode so unseen test points can be scored.
    n_neighbors=20 gives a neighbourhood covering roughly 10–20 min of data.
    """
    name = "lof"

    def __init__(self, n_neighbors=20):
        self.n_neighbors = n_neighbors

    def fit(self, X):
        self.m_ = LocalOutlierFactor(n_neighbors=self.n_neighbors, novelty=True)
        self.m_.fit(X)
        return self

    def score(self, X):
        return -self.m_.score_samples(X)      # LOF: 1 = normal → flip


# ═══════════════════════════════════════════════════════════════════════
# Deep-learning autoencoders – PyTorch implementations
# ═══════════════════════════════════════════════════════════════════════

if _HAS_TORCH:

    # ── internal PyTorch network definitions ────────────────────────────

    class _DenseAENet(nn.Module):
        """Symmetric feedforward autoencoder: D→32→16→8→16→32→D."""
        def __init__(self, input_dim: int, hidden=(32, 16, 8)):
            super().__init__()
            enc_dims = [input_dim] + list(hidden)
            dec_dims = list(reversed(hidden)) + [input_dim]
            self.encoder = nn.Sequential(
                *[layer for i in range(len(enc_dims) - 1)
                  for layer in (nn.Linear(enc_dims[i], enc_dims[i + 1]), nn.ReLU())]
            )
            self.decoder = nn.Sequential(
                *[layer for i in range(len(dec_dims) - 1)
                  for layer in (nn.Linear(dec_dims[i], dec_dims[i + 1]),
                                (nn.ReLU() if i < len(dec_dims) - 2 else nn.Identity()))]
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class _LSTMAENet(nn.Module):
        """
        LSTM sequence-to-sequence autoencoder.
          Encoder: LSTM over W steps → last hidden state h (hidden_size,)
          Decoder: repeat h W times → LSTM → linear projection → (W, input_dim)
        """
        def __init__(self, input_dim: int, hidden_size: int = 32, num_layers: int = 1):
            super().__init__()
            self.hidden_size = hidden_size
            self.encoder_lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
            self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
            self.output_layer = nn.Linear(hidden_size, input_dim)

        def forward(self, x):
            # x: (B, W, D)
            _, (h, c) = self.encoder_lstm(x)          # h: (1, B, H)
            B, W = x.size(0), x.size(1)
            dec_in = h[-1].unsqueeze(1).expand(-1, W, -1)   # (B, W, H)
            dec_out, _ = self.decoder_lstm(dec_in, (h, c))   # (B, W, H)
            return self.output_layer(dec_out)                 # (B, W, D)

    class _CNN1DAENet(nn.Module):
        """
        1-D convolutional autoencoder.
          Input: (B, D, W) – features as channels, time as spatial dim.
          Encoder: Conv1d stack with stride-2 downsampling → bottleneck
          Decoder: ConvTranspose1d stack back to original length
        """
        def __init__(self, input_dim: int, window: int, mid_ch: int = 16, bot_ch: int = 8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, mid_ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv1d(mid_ch, bot_ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )
            # Compute encoder output length
            L1 = (window + 1) // 2
            L2 = (L1 + 1) // 2
            self._enc_len = L2
            self._enc_ch = bot_ch

            # Decoder mirrors encoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(bot_ch, mid_ch, kernel_size=3, stride=2,
                                   padding=1, output_padding=(window % 2)),
                nn.ReLU(),
                nn.ConvTranspose1d(mid_ch, input_dim, kernel_size=3, stride=2,
                                   padding=1, output_padding=(L1 % 2)),
            )

        def forward(self, x):
            # x: (B, D, W)
            z = self.encoder(x)
            xhat = self.decoder(z)
            return xhat[..., :x.size(-1)]   # trim to exact original length

    # ── utility: train a torch model ────────────────────────────────────

    def _train_torch(net, X_tensor, epochs, lr, batch_size, seed):
        """Mini-batch MSE training loop."""
        torch.manual_seed(seed)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(X_tensor)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for _ in range(epochs):
            for (xb,) in dl:
                opt.zero_grad()
                loss_fn(net(xb), xb).backward()
                opt.step()
        net.eval()

    # ── public model classes (torch path) ───────────────────────────────

    class DenseAutoencoder:
        """
        Feedforward autoencoder (PyTorch).
        Architecture: D → 32 → 16 → 8 → 16 → 32 → D  (ReLU activations).
        Score = per-snapshot mean-squared reconstruction error.
        """
        name = "dense_ae"

        def __init__(self, hidden=(32, 16, 8), epochs=150, lr=1e-3,
                     batch_size=64, seed=0):
            self._hidden = hidden
            self._epochs = epochs
            self._lr = lr
            self._batch = batch_size
            self._seed = seed

        def fit(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.sc_ = StandardScaler().fit(X)
            Xs = self.sc_.transform(X).astype(np.float32)
            self.net_ = _DenseAENet(Xs.shape[1], self._hidden)
            Xt = torch.from_numpy(Xs)
            _train_torch(self.net_, Xt, self._epochs, self._lr, self._batch, self._seed)
            return self

        @torch.no_grad()
        def score(self, X):
            Xs = self.sc_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
            Xt = torch.from_numpy(Xs)
            recon = self.net_(Xt).numpy()
            return np.mean((Xs - recon) ** 2, axis=1)

    class LSTMAutoencoder:
        """
        LSTM sequence-to-sequence autoencoder (PyTorch).
        A causal sliding window of W=15 consecutive feature snapshots is encoded
        by an LSTM to a hidden-state vector, then decoded back by a second LSTM.
        Score for time t = MSE on the LAST step in the window ending at t,
        i.e. the model's ability to predict the current snapshot given its context.
        Architecture: LSTM(D→32) encoder | LSTM(32→32)→Linear(D) decoder.
        """
        name = "lstm_ae"

        def __init__(self, hidden=32, window=15, epochs=80, lr=1e-3,
                     batch_size=64, seed=0):
            self._hidden = hidden
            self._W = window
            self._epochs = epochs
            self._lr = lr
            self._batch = batch_size
            self._seed = seed

        def _windows(self, X):
            """Return (N, W, D) causal windows as a float32 array."""
            return _causal_windows(X.astype(np.float32), self._W)

        def fit(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.sc_ = StandardScaler().fit(X)
            Xs = self.sc_.transform(X).astype(np.float32)
            wins = self._windows(Xs)                          # (N, W, D)
            self.net_ = _LSTMAENet(Xs.shape[1], self._hidden)
            Xt = torch.from_numpy(wins)
            _train_torch(self.net_, Xt, self._epochs, self._lr, self._batch, self._seed)
            return self

        @torch.no_grad()
        def score(self, X):
            Xs = self.sc_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
            wins = self._windows(Xs)                          # (N, W, D)
            Xt = torch.from_numpy(wins)
            recon = self.net_(Xt).numpy()                     # (N, W, D)
            # Score = reconstruction error on the LAST step (index -1)
            err = (wins[:, -1, :] - recon[:, -1, :]) ** 2
            return err.mean(axis=1)

    class CNN1DAutoencoder:
        """
        1-D convolutional autoencoder (PyTorch).
        A sliding window of W=15 feature snapshots is treated as a 1-D signal
        with n_features channels.  Two strided Conv1d layers halve the temporal
        resolution twice, producing a compact latent representation.
        ConvTranspose1d layers decode back to the original length.
        Score = mean MSE across all W reconstructed steps.
        Architecture: Conv1d(D→16, s=2) → Conv1d(16→8, s=2) [encoder]
                      ConvTranspose1d mirror [decoder].
        """
        name = "cnn1d_ae"

        def __init__(self, mid_ch=16, bot_ch=8, window=15, epochs=80, lr=1e-3,
                     batch_size=64, seed=0):
            self._mid = mid_ch
            self._bot = bot_ch
            self._W = window
            self._epochs = epochs
            self._lr = lr
            self._batch = batch_size
            self._seed = seed

        def _windows(self, X):
            return _causal_windows(X.astype(np.float32), self._W)

        def fit(self, X):
            X = np.asarray(X, dtype=np.float32)
            self.sc_ = StandardScaler().fit(X)
            Xs = self.sc_.transform(X).astype(np.float32)
            wins = self._windows(Xs)                          # (N, W, D)
            # CNN needs (N, D, W) – channels first
            wins_cf = wins.transpose(0, 2, 1)
            self.net_ = _CNN1DAENet(Xs.shape[1], self._W, self._mid, self._bot)
            Xt = torch.from_numpy(wins_cf)
            _train_torch(self.net_, Xt, self._epochs, self._lr, self._batch, self._seed)
            return self

        @torch.no_grad()
        def score(self, X):
            Xs = self.sc_.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
            wins = self._windows(Xs)                          # (N, W, D)
            wins_cf = wins.transpose(0, 2, 1)                 # (N, D, W)
            Xt = torch.from_numpy(wins_cf)
            recon = self.net_(Xt).numpy()                     # (N, D, W)
            # Score = mean MSE over all W steps (transpose back for broadcast)
            err = (wins_cf - recon) ** 2                      # (N, D, W)
            return err.mean(axis=(1, 2))                      # (N,)

else:
    # ── Fallback: sklearn-based approximations (no PyTorch) ─────────────

    class DenseAutoencoder:
        """
        Feedforward autoencoder (sklearn MLPRegressor fallback – PyTorch not found).
        Architecture: D → 32 → 16 → 8 → 16 → 32 → D  (ReLU, symmetric).
        Score = mean-squared reconstruction error per snapshot.
        """
        name = "dense_ae"

        def __init__(self, hidden=(32, 16, 8), epochs=300, lr=1e-3, seed=0):
            enc = list(hidden)
            dec = list(reversed(hidden[:-1]))
            self._layers = tuple(enc + dec)
            self._epochs = epochs
            self._lr = lr
            self._seed = seed

        def fit(self, X):
            self.sc_ = StandardScaler().fit(X)
            Xs = self.sc_.transform(X)
            self.ae_ = MLPRegressor(
                hidden_layer_sizes=self._layers,
                activation="relu",
                max_iter=self._epochs,
                learning_rate_init=self._lr,
                random_state=self._seed,
                verbose=False,
            )
            self.ae_.fit(Xs, Xs)
            return self

        def score(self, X):
            Xs = self.sc_.transform(X)
            recon = self.ae_.predict(Xs)
            return np.mean((Xs - recon) ** 2, axis=1)

    class LSTMAutoencoder:
        """
        Sliding-window feedforward autoencoder (sklearn MLPRegressor fallback).
        A causal window of W=15 consecutive snapshots is flattened and reconstructed.
        Score = reconstruction error of the LAST step in the window (causal, as an
        LSTM-AE would produce).  PyTorch not available; this approximation captures
        the same temporal-context philosophy.
        """
        name = "lstm_ae"

        def __init__(self, hidden=(128, 64, 32), window=15, epochs=200,
                     lr=1e-3, seed=0):
            self._W = window
            self._hidden = hidden
            self._epochs = epochs
            self._lr = lr
            self._seed = seed

        def fit(self, X):
            X = np.asarray(X, dtype=np.float64)
            self.sc_ = StandardScaler().fit(X)
            Xs = self.sc_.transform(X)
            wins = _causal_windows(Xs, self._W)              # (N, W, D)
            flat = wins.reshape(len(wins), -1)               # (N, W*D)
            self.ae_ = MLPRegressor(
                hidden_layer_sizes=self._hidden,
                activation="relu",
                max_iter=self._epochs,
                learning_rate_init=self._lr,
                random_state=self._seed,
                verbose=False,
            )
            self.ae_.fit(flat, flat)
            return self

        def score(self, X):
            Xs = self.sc_.transform(np.asarray(X, dtype=np.float64))
            wins = _causal_windows(Xs, self._W)              # (N, W, D)
            flat = wins.reshape(len(wins), -1)
            recon = self.ae_.predict(flat).reshape(wins.shape)
            # error on last step only (index -1) → causal score
            err = (wins[:, -1, :] - recon[:, -1, :]) ** 2
            return err.mean(axis=1)

    class CNN1DAutoencoder:
        """
        Sliding-window autoencoder with temporal pooling (sklearn fallback).
        Window W=15 is split into K=3 non-overlapping segments; each segment is
        average-pooled → input (K*D,) is reconstructed by an MLP.  Score = mean
        MSE over all K reconstructed segments (mirrors 1-D-CNN strided pooling).
        PyTorch not available; this approximation preserves the local-aggregation
        philosophy of a 1-D-CNN-AE.
        """
        name = "cnn1d_ae"

        def __init__(self, hidden=(64, 32), window=15, n_pools=3,
                     epochs=200, lr=1e-3, seed=0):
            self._W = window
            self._K = n_pools
            self._hidden = hidden
            self._epochs = epochs
            self._lr = lr
            self._seed = seed

        def _pool(self, wins):
            """wins: (N, W, D) → pooled: (N, K*D)."""
            N, W, D = wins.shape
            K = self._K
            seg = W // K
            segs = [wins[:, i * seg:(i + 1) * seg, :].mean(axis=1) for i in range(K)]
            return np.concatenate(segs, axis=1)              # (N, K*D)

        def fit(self, X):
            X = np.asarray(X, dtype=np.float64)
            self.sc_ = StandardScaler().fit(X)
            Xs = self.sc_.transform(X)
            wins = _causal_windows(Xs, self._W)
            pooled = self._pool(wins)                        # (N, K*D)
            self.ae_ = MLPRegressor(
                hidden_layer_sizes=self._hidden,
                activation="relu",
                max_iter=self._epochs,
                learning_rate_init=self._lr,
                random_state=self._seed,
                verbose=False,
            )
            self.ae_.fit(pooled, pooled)
            return self

        def score(self, X):
            Xs = self.sc_.transform(np.asarray(X, dtype=np.float64))
            wins = _causal_windows(Xs, self._W)
            pooled = self._pool(wins)
            recon = self.ae_.predict(pooled)
            return np.mean((pooled - recon) ** 2, axis=1)
