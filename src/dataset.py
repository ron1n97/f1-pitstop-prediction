import numpy as np
import torch
from torch.utils.data import Dataset
from src.config import SEQ_FEATURES, SEQ_LEN, TARGET


class PitStopDataset(Dataset):
    """
    Vectorised sliding-window dataset — no Python loops over rows.
    Builds index arrays once; __getitem__ slices on demand.
    """
    def __init__(self, df, seq_len: int = SEQ_LEN):
        df = df.sort_values(["race_id", "Driver", "LapNumber"]).reset_index(drop=True)

        self.feats = df[SEQ_FEATURES].values.astype(np.float32)   # (N, F)
        self.labels = df[TARGET].values.astype(np.float32)         # (N,)
        self.seq_len = seq_len

        # For each row i, compute the start index of its window
        # Window = [max(group_start, i-seq_len+1) .. i]
        # Build group boundaries
        group_keys = df["race_id"].astype(str) + "_" + df["Driver"].astype(str)
        group_ids = group_keys.factorize()[0]          # int array 0..G-1

        # group_start[i] = index of first row in the same group as row i
        group_start = np.zeros(len(df), dtype=np.int64)
        boundaries = np.where(np.diff(group_ids, prepend=-1) != 0)[0]
        for b in boundaries:
            # fill forward — will be overwritten by next boundary
            group_start[b:] = b
        # fix: fill correctly
        group_start = np.zeros(len(df), dtype=np.int64)
        cur_start = 0
        for i in range(len(df)):
            if i > 0 and group_ids[i] != group_ids[i - 1]:
                cur_start = i
            group_start[i] = cur_start

        self.window_start = np.maximum(
            np.arange(len(df), dtype=np.int64) - seq_len + 1,
            group_start,
        )
        self.n = len(df)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        s = self.window_start[idx]
        e = idx + 1
        seq = self.feats[s:e]
        pad_len = self.seq_len - len(seq)
        if pad_len > 0:
            pad = np.zeros((pad_len, self.feats.shape[1]), dtype=np.float32)
            seq = np.vstack([pad, seq])
        return torch.from_numpy(seq), torch.tensor(self.labels[idx])
