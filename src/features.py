import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import butter, sosfiltfilt, hilbert

FS = 20_000
SPEED_SCALE = 0.979   # calibrated: observed BPFO 231.4 Hz vs 236.4 Hz nominal (Set 2 B1)


# ---------- time domain ----------
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


# ---------- bearing fault frequencies ----------
def fault_frequencies(rpm=2000, n=16, d=0.331, D=2.815, theta_deg=15.17):
    fr = rpm / 60
    c = (d / D) * np.cos(np.radians(theta_deg))
    return {
        "bpfo": n / 2 * fr * (1 - c),
        "bpfi": n / 2 * fr * (1 + c),
        "bsf": (D / (2 * d)) * fr * (1 - c**2),
        "ftf": 0.5 * fr * (1 - c),
    }

FAULT_FREQS = fault_frequencies()


# ---------- spectra ----------
def amplitude_spectrum(x, fs=FS):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    win = np.hanning(len(x))
    X = np.abs(np.fft.rfft(x * win)) * 2 / win.sum()
    f = np.fft.rfftfreq(len(x), 1 / fs)
    return f, X


def envelope_spectrum(x, fs=FS, band=(2000, 9000)):
    """Band-pass around a resonance, take the Hilbert envelope, then FFT."""
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    xf = sosfiltfilt(sos, np.asarray(x, dtype=np.float64))
    env = np.abs(hilbert(xf))
    return amplitude_spectrum(env, fs)


def fault_band_features(f, E, n_harm=3, rel_tol=0.025, prefix="env"):
    """Peak amplitude near each fault frequency and its harmonics.
    Window width scales with the harmonic number (relative tolerance)."""
    noise = np.median(E[(f > 10) & (f < 1000)]) + 1e-12
    out = {}
    for name in ("bpfo", "bpfi", "bsf"):
        base = FAULT_FREQS[name] * SPEED_SCALE
        amps = []
        for h in range(1, n_harm + 1):
            fc = h * base
            tol = rel_tol * fc
            m = (f >= fc - tol) & (f <= fc + tol)
            amps.append(E[m].max())
        out[f"{prefix}_{name}"] = float(np.sum(amps))
        out[f"{prefix}_{name}_snr"] = float(np.sum(amps) / (n_harm * noise))
    return out


def freq_features(x, fs=FS, n_bands=8):
    """Returns (feature dict, envelope spectrum 0-1000 Hz as float32)."""
    f, X = amplitude_spectrum(x, fs)
    P = X**2
    edges = np.linspace(0, fs / 2, n_bands + 1)
    total = P.sum() + 1e-12
    out = {}
    for i in range(n_bands):
        m = (f >= edges[i]) & (f < edges[i + 1])
        out[f"band{i}_frac"] = float(P[m].sum() / total)
    p = P / total
    out["spec_entropy"] = float(-(p * np.log(p + 1e-12)).sum() / np.log(len(p)))
    out["spec_centroid"] = float((f * P).sum() / total)

    fe, E = envelope_spectrum(x, fs)
    out.update(fault_band_features(fe, E))
    return out, E[fe <= 1000].astype(np.float32)

N_SAMPLES = 20_480    # samples per snapshot
SHAFT_HZ = 32.68   # measured from the Set 1 B3 late-life comb (high harmonics: 32.66-32.69 Hz)

def shaft_comb_snr(spec, fs=FS, n_samples=N_SAMPLES, shaft_hz=SHAFT_HZ, n_harm=25, tol=1.5):
    """Summed envelope amplitude at shaft harmonics, relative to the noise floor."""
    freqs = np.arange(spec.shape[1]) * fs / n_samples
    noise = np.median(spec[:, (freqs > 10) & (freqs < 1000)], axis=1) + 1e-12
    amps = []
    for k in range(1, n_harm + 1):
        m = np.abs(freqs - k * shaft_hz) <= tol
        amps.append(spec[:, m].max(axis=1))
    return np.sum(amps, axis=0) / (n_harm * noise)

def add_spectral_features(df, spec, fs=FS, n_samples=N_SAMPLES, rel_tol=0.025, min_tol=1.5):
    """
    Add fault-specific envelope features computed from the saved envelope spectra.

    df   : feature table from build_features (rows must be in the original order)
    spec : array loaded from data/env_spectra.npy, shape (len(df), n_bins), 0-1000 Hz
    Returns a copy of df with these extra columns:
        env_ftf_snr, env_bpfo_snr4, env_bpfi_sb_snr, env_fault_max_snr
    """
    assert len(df) == len(spec), "df and spec are misaligned; don't filter or reorder df first"

    freqs = np.arange(spec.shape[1]) * fs / n_samples       # exact bin frequencies (~0.977 Hz spacing)
    noise = np.median(spec[:, (freqs > 10) & (freqs < 1000)], axis=1) + 1e-12

    def peak_at(fc):
        tol = max(rel_tol * fc, min_tol)    # min_tol keeps at least a few bins in the window at low frequencies
        m = (freqs >= fc - tol) & (freqs <= fc + tol)
        return spec[:, m].max(axis=1)

    def snr(centers):
        return np.sum([peak_at(c) for c in centers], axis=0) / (len(centers) * noise)   

    fr = 2000 / 60 * SPEED_SCALE            # shaft frequency (~32.6 Hz)
    base = {k: v * SPEED_SCALE for k, v in FAULT_FREQS.items()}

    out = df.copy()
    out["env_shaft_comb_snr"] = shaft_comb_snr(spec, fs, n_samples)
    out["env_ftf_snr"] = snr([h * base["ftf"] for h in (1, 2)])
    out["env_bpfo_snr4"] = snr([h * base["bpfo"] for h in (1, 2, 3, 4)])
    out["env_bpfi_sb_snr"] = snr([base["bpfi"] + k * fr for k in (-1, 0, 1)])
    out["env_fault_max_snr"] = out[["env_bpfo_snr", "env_bpfi_snr", "env_bsf_snr"]].max(axis=1)
    return out