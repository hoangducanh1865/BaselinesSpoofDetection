"""Shared dataset registry for baseline evaluation adapters."""

from __future__ import annotations

import importlib
import os
from pathlib import Path


DATASET_MODULES = {
    "asvspoof5": ("datasets.asvspoof5", None),
    "asvspoof2019la": ("datasets.asvspoof2019", "LA"),
    "asvspoof2019pa": ("datasets.asvspoof2019", "PA"),
    "asvspoof2021la": ("datasets.asvspoof2021", "LA"),
    "asvspoof2021df": ("datasets.asvspoof2021", "DF"),
    "asvspoof2021pa": ("datasets.asvspoof2021", "PA"),
    "dfadd_test": ("datasets.dfadd", None),
    "fake_or_real": ("datasets.fake_or_real", None),
    "fake_or_real_norm": ("datasets.fake_or_real", "norm"),
    "fake_or_real_2sec": ("datasets.fake_or_real", "2sec"),
    "fake_or_real_original": ("datasets.fake_or_real", "original"),
    "fake_or_real_rerec": ("datasets.fake_or_real", "rerec"),
    "in_the_wild": ("datasets.in_the_wild", None),
    "vlsp2025": ("datasets.vlsp2025", None),
    "vsasv": ("datasets.vsasv", None),
    "vsasv_resplit": ("datasets.vsasv", "resplit"),
}


DATA_ROOTS = {
    "asvspoof5": "/home/user14/anhhd/spoof/datasets/asvspoof5",
    "asvspoof2019la": "/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA",
    "asvspoof2019pa": "/home/user14/anhhd/spoof/datasets/asvspoof2019/PA/PA",
    "asvspoof2021la": "/home/user14/anhhd/spoof/datasets/asvspoof2021",
    "asvspoof2021df": "/home/user14/anhhd/spoof/datasets/asvspoof2021",
    "asvspoof2021pa": "/home/user14/anhhd/spoof/datasets/asvspoof2021",
    "dfadd_test": "/home/user14/anhhd/spoof/datasets/dfadd_test",
    "fake_or_real": "/home/user14/anhhd/spoof/datasets/fake_or_real",
    "fake_or_real_norm": "/home/user14/anhhd/spoof/datasets/fake_or_real",
    "fake_or_real_2sec": "/home/user14/anhhd/spoof/datasets/fake_or_real",
    "fake_or_real_original": "/home/user14/anhhd/spoof/datasets/fake_or_real",
    "fake_or_real_rerec": "/home/user14/anhhd/spoof/datasets/fake_or_real",
    "in_the_wild": "/home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild",
    "vlsp2025": "/home/user14/anhhd/spoof/datasets/vlsp2025",
    "vsasv": "/home/user14/anhhd/spoof/datasets/vsasv",
    "vsasv_resplit": "/home/user14/anhhd/spoof/datasets/vsasv",
}


def dataset_root(dataset: str, config_roots: dict | None = None) -> Path:
    """Resolve the root folder for a dataset name.

    ``SPOOF_DATA_ROOT`` remains a global override for one-off experiments.
    Config roots are used by MoLEx, whose YAML already stores some dataset paths.
    """
    if os.environ.get("SPOOF_DATA_ROOT"):
        return Path(os.environ["SPOOF_DATA_ROOT"])
    if config_roots and config_roots.get(dataset):
        return Path(config_roots[dataset])
    return Path(DATA_ROOTS[dataset])


def ensure_eval_meta(
    dataset: str,
    output_root: Path,
    config_roots: dict | None = None,
    fold: int = 1,
) -> tuple[Path, Path]:
    """Create evaluation metadata and return ``(eval_tsv, wav_scp)``."""
    module_name, track = DATASET_MODULES[dataset]
    mod = importlib.import_module(module_name)
    meta_dir = Path(output_root) / "meta" / dataset
    mod.ensure_meta(
        data_root=dataset_root(dataset, config_roots=config_roots),
        meta_dir=meta_dir,
        fold=fold,
        track=track,
    )
    return meta_dir / f"fold{fold}_evaluation.tsv", meta_dir / "wav.scp"


def ensure_dataset_meta(
    dataset: str,
    meta_dir: Path,
    config_roots: dict | None = None,
    fold: int = 1,
) -> tuple[Path, Path]:
    """Create full metadata under an explicit directory and return ``(meta_dir, wav_scp)``."""
    module_name, track = DATASET_MODULES[dataset]
    mod = importlib.import_module(module_name)
    meta_dir = Path(meta_dir)
    mod.ensure_meta(
        data_root=dataset_root(dataset, config_roots=config_roots),
        meta_dir=meta_dir,
        fold=fold,
        track=track,
    )
    return meta_dir, meta_dir / "wav.scp"
