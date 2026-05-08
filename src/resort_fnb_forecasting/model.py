# -*- coding: utf-8 -*-

from __future__ import annotations

import torch
import torch.nn as nn


class Seq2SeqModel(nn.Module):
    """Menu-aware encoder-decoder LSTM for multi-step sales forecasting."""

    def __init__(
        self,
        n_menus: int,
        emb_size: int = 8,
        hid_size: int = 64,
        horizon: int = 7,
        n_cont_features: int = 6,
    ):
        super().__init__()
        self.horizon = horizon
        self.emb = nn.Embedding(n_menus, emb_size)
        self.encoder = nn.LSTM(
            input_size=emb_size + n_cont_features,
            hidden_size=hid_size,
            batch_first=True,
        )
        self.decoder = nn.LSTM(input_size=1, hidden_size=hid_size, batch_first=True)
        self.fc = nn.Linear(hid_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        menu_idx = x[:, :, 0].long()
        cont = x[:, :, 1:]

        enc_in = torch.cat([self.emb(menu_idx), cont], dim=2)
        _, (h, c) = self.encoder(enc_in)

        dec_in = cont[:, -1:, -1].unsqueeze(2)
        preds = []

        for _ in range(self.horizon):
            out, (h, c) = self.decoder(dec_in, (h, c))
            pred = self.fc(out)
            preds.append(pred.squeeze(2))
            dec_in = pred

        return torch.cat(preds, dim=1)

