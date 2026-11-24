import pandas as pd
from tqdm import tqdm
from src.data_loader import DATASETS, FAILED, channel_map, list_files, load_snapshot, file_timestamp
from src.features import time_features

rows = []
for set_name in DATASETS:
    cmap = channel_map(set_name)
    files = list_files(set_name)
    for idx, f in enumerate(tqdm(files, desc=set_name)):
        data = load_snapshot(f)
        ts = file_timestamp(f)
        for bearing, cols in cmap.items():
            for ch in cols:
                rows.append({
                    "set": set_name, "bearing": bearing, "channel": ch,
                    "idx": idx, "timestamp": ts,
                    "failed": bearing in FAILED[set_name],
                    **time_features(data[:, ch]),
                })

df = pd.DataFrame(rows)
df.to_parquet("data/processed_features.parquet", index=False)
print(df.shape)