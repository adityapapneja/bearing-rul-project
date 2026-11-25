import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_loader import (
    DATASETS, FAILED, channel_map, list_files, load_snapshot, file_timestamp,
)
from src.features import time_features, freq_features, add_spectral_features

OUT_FEATURES = "data/processed_features.parquet"
OUT_SPECTRA = "data/env_spectra.npy"


def main():
    rows = []       # one dict per (set, bearing, channel, snapshot)
    spectra = []    # envelope spectrum for the same row, same order
    skipped = []

    for set_name in DATASETS:
        cmap = channel_map(set_name)
        files = list_files(set_name)

        for idx, path in enumerate(tqdm(files, desc=set_name)):
            try:
                data = load_snapshot(path)
                ts = file_timestamp(path)
            except Exception as e:                      # corrupt or malformed file
                skipped.append((set_name, path.name, repr(e)))
                continue

            for bearing, cols in cmap.items():
                for ch in cols:
                    x = data[:, ch]
                    feats, env = freq_features(x)
                    spectra.append(env)
                    rows.append({
                        "set": set_name,
                        "bearing": bearing,
                        "channel": ch,
                        "idx": idx,
                        "timestamp": ts,
                        "failed": bearing in FAILED[set_name],
                        **time_features(x),
                        **feats,
                    })

    df = pd.DataFrame(rows)
    spec_arr = np.stack(spectra)

    # row i of the table must match row i of the spectra array
    assert len(df) == len(spec_arr), "feature table and spectra are misaligned"

    df = add_spectral_features(df, spec_arr)
    df.to_parquet(OUT_FEATURES, index=False)
    np.save(OUT_SPECTRA, spec_arr)

    print(f"features: {df.shape} -> {OUT_FEATURES}")
    print(f"spectra:  {spec_arr.shape} -> {OUT_SPECTRA}")
    if skipped:
        print(f"skipped {len(skipped)} files:")
        for s in skipped:
            print("  ", s)


if __name__ == "__main__":
    main()