# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ForecastConfig:
    train_path: Path = Path("data/train.csv")
    sample_submission_path: Path = Path("data/sample_submission.csv")
    test_dir: Path = Path("data/test")
    model_dir: Path = Path("models")
    submission_dir: Path = Path("submissions")

    seq_len: int = 28
    horizon: int = 7
    emb_size: int = 8
    hid_size: int = 64
    batch_size: int = 64
    epochs: int = 40
    lr: float = 1e-3

    ensemble_size: int = 5
    base_seed: int = 42
    num_test_files: int = 10
    device: Optional[str] = None

    @property
    def seeds(self) -> list[int]:
        return [self.base_seed + i for i in range(self.ensemble_size)]

    @property
    def manifest_path(self) -> Path:
        return self.model_dir / "ensemble_manifest.json"

    def to_jsonable(self) -> dict:
        data = asdict(self)
        for key in (
            "train_path",
            "sample_submission_path",
            "test_dir",
            "model_dir",
            "submission_dir",
        ):
            data[key] = str(data[key])
        data["seeds"] = self.seeds
        data["manifest_path"] = str(self.manifest_path)
        return data

