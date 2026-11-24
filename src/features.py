import numpy as np
from scipy.stats import kurtosis, skew


def time_features(x):
    x = np.asarray(x, dtype=np.float64)
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x))
    return {
        "rms": rms,
        "peak": peak,
        "p2p": np.ptp(x),
        "std": np.std(x),
        "kurt": kurtosis(x),
        "skew": skew(x),
        "crest": peak / rms,
        "shape": rms / mean_abs,
        "impulse": peak / mean_abs,
    }