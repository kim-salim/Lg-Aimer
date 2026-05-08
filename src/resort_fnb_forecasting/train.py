# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ForecastConfig
from .dataset import Seq2SeqDataset
from .features import DATE_COL, FEATURE_COLUMNS, MENU_COL, add_calendar_features
from .model import Seq2SeqModel
from .utils import ensure_dir, get_device, set_seed


def _build_menu_mapping(df: pd.DataFrame) -> dict[str, int]:
    menus = df[MENU_COL].drop_duplicates().tolist()
    return {menu: idx for idx, menu in enumerate(menus)}


def train_ensemble(config: ForecastConfig) -> list[dict]:
    ensure_dir(config.model_dir)
    device = get_device(config.device)

    df = pd.read_csv(config.train_path, parse_dates=[DATE_COL])
    df = add_calendar_features(df)

    menu2idx = _build_menu_mapping(df)
    dataset = Seq2SeqDataset(
        df,
        menu2idx=menu2idx,
        seq_len=config.seq_len,
        horizon=config.horizon,
        feature_cols=FEATURE_COLUMNS,
    )
    if len(dataset) == 0:
        raise RuntimeError("No training sequences created. Check data date ranges.")

    print(f"Train samples: {len(dataset)} | Menus: {len(menu2idx)} | Device: {device}")

    saved_models = []
    for seed in config.seeds:
        set_seed(seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        drop_last = len(dataset) >= config.batch_size
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=drop_last,
            generator=generator,
        )

        model = Seq2SeqModel(
            n_menus=len(menu2idx),
            emb_size=config.emb_size,
            hid_size=config.hid_size,
            horizon=config.horizon,
            n_cont_features=len(FEATURE_COLUMNS),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        best_path = config.model_dir / f"best_seq2seq_seed{seed}.pth"

        print(f"\n===== Seed {seed} training start =====")
        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss = 0.0
            seen = 0

            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)

                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

                batch_size = xb.size(0)
                total_loss += loss.item() * batch_size
                seen += batch_size

            avg_loss = total_loss / max(seen, 1)
            if epoch == 1 or epoch % 10 == 0:
                print(f"[Seed {seed}] Epoch {epoch}/{config.epochs} - loss {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(
                    {
                        "state": model.state_dict(),
                        "menu2idx": menu2idx,
                        "seed": seed,
                        "feature_cols": FEATURE_COLUMNS,
                        "config": config.to_jsonable(),
                    },
                    best_path,
                )

        print(f"Seed {seed} best loss: {best_loss:.4f} | Saved: {best_path}")
        saved_models.append(
            {"seed": seed, "path": str(best_path), "best_loss": round(best_loss, 6)}
        )

    manifest = {
        "created_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": saved_models,
        "paths": [item["path"] for item in saved_models],
        "seeds": [item["seed"] for item in saved_models],
        "menu2idx": menu2idx,
        "feature_cols": FEATURE_COLUMNS,
        "config": config.to_jsonable(),
    }
    with open(config.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\nEnsemble manifest saved: {config.manifest_path}")
    return saved_models

