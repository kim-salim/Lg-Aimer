# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import DATE_COL, FEATURE_COLUMNS, MENU_COL


class Seq2SeqDataset(Dataset):
    """Sliding-window dataset for 28-day input and 7-day target forecasting."""

    def __init__(
        self,
        df,
        menu2idx: dict[str, int],
        seq_len: int = 28,
        horizon: int = 7,
        feature_cols: list[str] | None = None,
    ):
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self.feature_cols = feature_cols or FEATURE_COLUMNS

        for menu, group in df.groupby(MENU_COL):
            group = group.sort_values(DATE_COL)
            if menu not in menu2idx:
                continue

            arr = group[self.feature_cols].to_numpy(dtype=np.float32)
            if len(arr) < seq_len + horizon:
                continue

            menu_idx = menu2idx[menu]
            for start in range(len(arr) - seq_len - horizon + 1):
                seq_x = arr[start : start + seq_len]
                menu_col = np.full((seq_len, 1), menu_idx, dtype=np.float32)
                seq = np.hstack([menu_col, seq_x])
                target = arr[start + seq_len : start + seq_len + horizon, -1]
                self.samples.append((seq, target.astype(np.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq, target = self.samples[idx]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(
            target, dtype=torch.float32
        )

