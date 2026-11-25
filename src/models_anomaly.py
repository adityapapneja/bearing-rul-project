import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest


class Mahalanobis:
    name = "mahalanobis"

    def fit(self, X):
        self.est = LedoitWolf().fit(X)          # shrinkage keeps the covariance well-conditioned
        return self

    def score(self, X):
        return self.est.mahalanobis(X)          # squared distance to the healthy mean


class PCAMonitor:
    def __init__(self, var=0.95, stat="spe"):
        self.var, self.stat = var, stat
        self.name = f"pca_{stat}"

    def fit(self, X):
        self.pca = PCA(n_components=self.var, svd_solver="full").fit(X)
        return self

    def score(self, X):
        T = self.pca.transform(X)
        if self.stat == "t2":                   # distance inside the retained subspace
            return np.sum(T**2 / self.pca.explained_variance_, axis=1)
        return np.sum((X - self.pca.inverse_transform(T))**2, axis=1)   # SPE: distance outside it


class IsoForest:
    name = "isolation_forest"

    def __init__(self, seed=0):
        self.m = IsolationForest(n_estimators=300, random_state=seed)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)         # sklearn: higher = more normal, so flip the sign