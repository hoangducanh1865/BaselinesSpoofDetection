"""Unified CLI adapter for the upstream AASIST baseline.

The original AASIST entrypoint is tied to ASVspoof2019 LA config/protocols.
This adapter reuses the model definition and deterministic eval padding, but
loads repo-level dataset metadata so the common ``main.py`` CLI can evaluate
ASVspoof2019 LA, ASVspoof5, and In-The-Wild with the same checkpoint.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
AASIST_DIR = Path(__file__).resolve().parent
DEFAULT_CKPT = Path(
    "/home/user14/anhhd/spoof/pretrained_spoof_models/"
    "trained_on_asvspoof2019la/aasist/aasist_asvspoof2019la.pth"
)

DATASET_MODULES = {
    "asvspoof5": ("datasets.asvspoof5", None),
    "asvspoof2019la": ("datasets.asvspoof2019", "LA"),
    "asvspoof2019pa": ("datasets.asvspoof2019", "PA"),
    "in_the_wild": ("datasets.in_the_wild", None),
}

DATA_ROOTS = {
    "asvspoof5": "/home/user14/anhhd/spoof/datasets/asvspoof5",
    "asvspoof2019la": "/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA",
    "asvspoof2019pa": "/home/user14/anhhd/spoof/datasets/asvspoof2019/PA/PA",
    "in_the_wild": "/home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild",
}


def _output_root() -> Path:
    return REPO_ROOT / "outputs" / "aasist"


def _load_model_config() -> dict:
    with open(AASIST_DIR / "config" / "AASIST.conf") as f:
        return json.load(f)["model_config"]


def _resolve_meta(args) -> tuple[Path, Path]:
    module_name, track = DATASET_MODULES[args.dataset]
    mod = importlib.import_module(module_name)
    data_root = Path(os.environ.get("SPOOF_DATA_ROOT") or DATA_ROOTS[args.dataset])
    meta_dir = _output_root() / "meta" / args.dataset
    fold = 1
    mod.ensure_meta(data_root=data_root, meta_dir=meta_dir, fold=fold, track=track)
    return meta_dir / f"fold{fold}_evaluation.tsv", meta_dir / "wav.scp"


def _load_wav_scp(path: Path) -> dict[str, str]:
    mapping = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1]
    return mapping


def _pad(x: np.ndarray, max_len: int = 64600) -> np.ndarray:
    if x.shape[0] >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x.shape[0]) + 1
    return np.tile(x, num_repeats)[:max_len]


class AASISTEvalDataset(Dataset):
    def __init__(self, meta_path: Path, wav_scp_path: Path):
        df = pd.read_csv(meta_path, sep="\t")
        wav_scp = _load_wav_scp(wav_scp_path)
        id_col, label_col = df.columns[:2]
        rows = []
        for row in df.itertuples(index=False):
            utt_id = str(getattr(row, id_col))
            label = str(getattr(row, label_col)).lower()
            if utt_id in wav_scp:
                rows.append((utt_id, label, wav_scp[utt_id]))
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        utt_id, label, path = self.rows[index]
        audio, _ = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _pad(audio.astype(np.float32))
        y = 1 if label == "bonafide" else 0
        return Tensor(audio), y, utt_id


def _compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    bona_scores = scores[labels == 1]
    spoof_scores = scores[labels == 0]
    if bona_scores.size == 0 or spoof_scores.size == 0:
        raise ValueError("EER requires both bonafide and spoof samples.")

    all_scores = np.concatenate([bona_scores, spoof_scores])
    det_labels = np.concatenate([np.ones(bona_scores.size), np.zeros(spoof_scores.size)])
    indices = np.argsort(all_scores, kind="mergesort")
    det_labels = det_labels[indices]
    n_scores = det_labels.size
    tar_trial_sums = np.cumsum(det_labels)
    nontarget_trial_sums = spoof_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    frr = np.concatenate(([0.0], tar_trial_sums / bona_scores.size))
    far = np.concatenate(([1.0], nontarget_trial_sums / spoof_scores.size))
    thresholds = np.concatenate(([all_scores[indices[0]] - 0.001], all_scores[indices]))
    min_index = np.argmin(np.abs(frr - far))
    return float(np.mean([frr[min_index], far[min_index]]) * 100), float(thresholds[min_index])


def _load_model(ckpt_path: Path, device: torch.device):
    sys.path.insert(0, str(AASIST_DIR))
    module = importlib.import_module("models.AASIST")
    model = module.Model(_load_model_config()).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    checkpoint = {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }
    model.load_state_dict(checkpoint, strict=True)
    return model


def _run_eval(args, compute_eer: bool) -> None:
    if args.config:
        print("[aasist] --config is ignored; using baselines/aasist/config/AASIST.conf model_config.")

    ckpt_path = Path(args.ckpt or os.environ.get("AASIST_CKPT", DEFAULT_CKPT))
    meta_path, wav_scp_path = _resolve_meta(args)
    dataset = AASISTEvalDataset(meta_path, wav_scp_path)
    batch_size = int(os.environ.get("AASIST_EVAL_BATCH_SIZE", 128))
    num_workers = int(os.environ.get("AASIST_EVAL_NUM_WORKERS", 8))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt_path, device)
    model.eval()
    print(
        f"[aasist] Evaluation files: {len(dataset)}, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = _output_root() / "evals" / f"{timestamp}__{ckpt_path.stem}__on__{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / ("eval_output.txt" if compute_eer else "score.txt")

    utt_ids = []
    labels = []
    scores = []
    with torch.inference_mode():
        for batch_x, batch_y, batch_utt in tqdm(loader, desc="Evaluation", dynamic_ncols=True):
            batch_x = batch_x.to(device, non_blocking=True)
            _, batch_out = model(batch_x)
            batch_scores = batch_out[:, 1].detach().cpu().numpy().ravel()
            utt_ids.extend(batch_utt)
            labels.extend(batch_y.numpy().tolist())
            scores.extend(batch_scores.tolist())

    with open(score_path, "w") as f:
        for utt_id, label, score in zip(utt_ids, labels, scores):
            label_text = "bonafide" if label == 1 else "spoof"
            f.write(f"{utt_id}\t{label_text}\t{score}\n")

    with open(output_dir / "eval_config.txt", "w") as f:
        f.write(f"dataset={args.dataset}\n")
        f.write(f"checkpoint={ckpt_path}\n")
        f.write(f"score_higher=bonafide\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"num_workers={num_workers}\n")

    print(f"Scores written to {score_path}")
    if compute_eer:
        eer, threshold = _compute_eer(np.asarray(labels), np.asarray(scores))
        with open(output_dir / "eval_EER.txt", "w") as f:
            f.write(f"EER: {eer:.9f}\n")
            f.write(f"Threshold: {threshold:.9f}\n")
        print(f"EER: {eer:.3f}% (threshold {threshold:.6f})")


def eval(args) -> None:  # noqa: A001 - dispatched by name from main.py
    _run_eval(args, compute_eer=True)


def score(args) -> None:
    _run_eval(args, compute_eer=False)


def train(args) -> None:
    raise NotImplementedError(
        "AASIST training is not wired into the unified CLI. "
        "Use baselines/aasist/main.py directly if training from scratch is needed."
    )
