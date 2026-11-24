from pathlib import Path
import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
FS = 20_000  # sampling rate (Hz)

DATASETS = {
    "set1": {"path": DATA_ROOT / "1st_test" / "1st_test", "n_channels": 8},
    "set2": {"path": DATA_ROOT / "2nd_test" / "2nd_test", "n_channels": 4},
    "set3": {"path": DATA_ROOT / "3rd_test" / "4th_test" / "txt", "n_channels": 4},
}

# Which bearings actually failed (these are your run-to-failure trajectories)
FAILED = {"set1": ["b3", "b4"], "set2": ["b1"], "set3": ["b3"]}


def channel_map(set_name):
    """Map bearing name -> list of column indices in the raw file."""
    if set_name == "set1":              # 2 accelerometers per bearing
        return {f"b{i+1}": [2 * i, 2 * i + 1] for i in range(4)}
    return {f"b{i+1}": [i] for i in range(4)}   # 1 per bearing


def list_files(set_name):
    """Snapshot files in chronological order (filenames are timestamps)."""
    folder = DATASETS[set_name]["path"]
    return sorted(p for p in folder.iterdir() if p.is_file())


def load_snapshot(path):
    """Return array of shape (20480, n_channels)."""
    return pd.read_csv(path, sep=r"\s+", header=None).to_numpy(dtype=np.float32)


def file_timestamp(path):
    return pd.to_datetime(path.name, format="%Y.%m.%d.%H.%M.%S")