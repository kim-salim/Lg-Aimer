# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ForecastConfig
from .predict import predict_ensemble
from .train import train_ensemble


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and predict resort F&B sales with a Seq2Seq LSTM ensemble."
    )
    parser.add_argument("--mode", choices=["train", "predict", "all"], default="all")

    parser.add_argument("--train-path", type=Path, default=Path("data/train.csv"))
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=Path("data/sample_submission.csv"),
    )
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--submission-dir", type=Path, default=Path("submissions"))

    parser.add_argument("--seq-len", type=int, default=28)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--emb-size", type=int, default=8)
    parser.add_argument("--hid-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--num-test-files", type=int, default=10)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override such as cpu, cuda, or cuda:0.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ForecastConfig:
    return ForecastConfig(
        train_path=args.train_path,
        sample_submission_path=args.sample_submission_path,
        test_dir=args.test_dir,
        model_dir=args.model_dir,
        submission_dir=args.submission_dir,
        seq_len=args.seq_len,
        horizon=args.horizon,
        emb_size=args.emb_size,
        hid_size=args.hid_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        ensemble_size=args.ensemble_size,
        base_seed=args.base_seed,
        num_test_files=args.num_test_files,
        device=args.device,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)

    if args.mode in ("train", "all"):
        train_ensemble(config)
    if args.mode in ("predict", "all"):
        predict_ensemble(config)

