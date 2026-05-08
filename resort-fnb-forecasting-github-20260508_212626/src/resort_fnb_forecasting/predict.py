# -*- coding: utf-8 -*-

from __future__ import annotations

import datetime as dt
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ForecastConfig
from .features import DATE_COL, FEATURE_COLUMNS, MENU_COL, add_calendar_features
from .model import Seq2SeqModel
from .utils import ensure_dir, get_device, torch_load


def _resolve_checkpoint_path(path_text: str, model_dir: Path) -> Path | None:
    path = Path(path_text)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(model_dir / path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_ensemble_paths(config: ForecastConfig) -> list[Path]:
    if config.manifest_path.exists():
        with open(config.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        paths = [
            resolved
            for item in manifest.get("paths", [])
            if (resolved := _resolve_checkpoint_path(item, config.model_dir)) is not None
        ]
        if paths:
            print(f"Loaded ensemble from manifest: {len(paths)} model(s)")
            return paths

    paths = sorted(Path(p) for p in glob.glob(str(config.model_dir / "best_seq2seq_seed*.pth")))
    if paths:
        print(f"Loaded ensemble by glob: {len(paths)} model(s)")
        return paths

    single = config.model_dir / "best_seq2seq.pth"
    if single.exists():
        print("Loaded single-model fallback.")
        return [single]

    raise FileNotFoundError(f"No model checkpoints found in {config.model_dir}")


def _make_input_sequence(test_df: pd.DataFrame, menu: str, menu_idx: int, seq_len: int):
    menu_df = test_df[test_df[MENU_COL] == menu].sort_values(DATE_COL)
    if len(menu_df) < seq_len:
        return None

    values = menu_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)[-seq_len:]
    menu_col = np.full((seq_len, 1), menu_idx, dtype=np.float32)
    return np.hstack([menu_col, values])


def predict_ensemble(config: ForecastConfig) -> Path:
    ensure_dir(config.submission_dir)
    device = get_device(config.device)

    checkpoint_paths = load_ensemble_paths(config)
    first_checkpoint = torch_load(checkpoint_paths[0], map_location=device)
    saved_config = first_checkpoint.get("config", {})
    menu2idx = first_checkpoint["menu2idx"]
    menus = list(menu2idx.keys())
    seq_len = int(saved_config.get("seq_len", config.seq_len))
    horizon = int(saved_config.get("horizon", config.horizon))
    emb_size = int(saved_config.get("emb_size", config.emb_size))
    hid_size = int(saved_config.get("hid_size", config.hid_size))

    submission = pd.read_csv(config.sample_submission_path)
    for col in submission.columns[1:]:
        submission[col] = submission[col].astype(float)

    submission["day"] = submission[DATE_COL].str.extract(r"\+(\d+)일")[0].astype(int)
    target_menus = [menu for menu in menus if menu in submission.columns]
    missing_menus = sorted(set(menus) - set(target_menus))
    if missing_menus:
        print(f"Warning: {len(missing_menus)} train menus are missing from submission columns.")

    print(f"Predicting with {len(checkpoint_paths)} model(s) on {device}")

    for test_idx in range(config.num_test_files):
        test_path = config.test_dir / f"TEST_{test_idx:02d}.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test file: {test_path}")

        test_df = pd.read_csv(test_path, parse_dates=[DATE_COL])
        test_df = add_calendar_features(test_df)

        seq_cache = {
            menu: _make_input_sequence(
                test_df=test_df,
                menu=menu,
                menu_idx=menu2idx[menu],
                seq_len=seq_len,
            )
            for menu in target_menus
        }
        preds_sum = {
            menu: np.zeros(horizon, dtype=np.float32) for menu in target_menus
        }

        for checkpoint_path in checkpoint_paths:
            checkpoint = torch_load(checkpoint_path, map_location=device)
            if checkpoint["menu2idx"] != menu2idx:
                raise RuntimeError(f"menu2idx mismatch in checkpoint: {checkpoint_path}")

            model = Seq2SeqModel(
                n_menus=len(menu2idx),
                emb_size=emb_size,
                hid_size=hid_size,
                horizon=horizon,
                n_cont_features=len(FEATURE_COLUMNS),
            ).to(device)
            model.load_state_dict(checkpoint["state"])
            model.eval()

            with torch.no_grad():
                for menu, seq in seq_cache.items():
                    if seq is None:
                        continue
                    x = torch.tensor(seq[None], dtype=torch.float32).to(device)
                    pred = model(x).cpu().numpy().squeeze(0)
                    preds_sum[menu] += pred

            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        preds_avg = {
            menu: np.clip(
                np.round(preds_sum[menu] / len(checkpoint_paths), 4),
                0,
                None,
            )
            for menu in target_menus
        }

        mask = submission[DATE_COL].str.startswith(f"TEST_{test_idx:02d}")
        for menu in target_menus:
            submission.loc[mask, menu] = submission.loc[mask, "day"].map(
                lambda day: preds_avg[menu][day - 1]
            )

    submission = submission.drop(columns=["day"]).round(4)
    out_path = (
        config.submission_dir
        / f"submission_ensemble_{len(checkpoint_paths)}models_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    submission.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")
    return out_path
