"""Unified CLI adapter for the Nes2Net baseline.

Nes2Net uses different SSL front-ends in the released experiments:
ASVspoof2019/2021 and In-The-Wild checkpoints use wav2vec2/XLS-R 300M, while
the ASVspoof5 checkpoint uses WavLM-Large. This adapter selects the matching
front-end from the checkpoint path, with ``NES2NET_BACKBONE`` as an override.
"""

from __future__ import annotations

import importlib
import json
import os
from types import SimpleNamespace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.registry import ensure_eval_meta

REPO_ROOT = Path(__file__).resolve().parents[2]
NES2NET_DIR = Path(__file__).resolve().parent
DEFAULT_WAVLM = Path("/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large/WavLM-Large.pt")
DEFAULT_XLSR = Path("/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt")
DEFAULT_CKPT_2019 = Path(
    "/home/user14/anhhd/spoof/pretrained_spoof_models/"
    "trained_on_asvspoof2019la/nes2net/nes2net_asvspoof2019la.pth"
)
DEFAULT_CKPT_ASV5 = Path(
    "/home/user14/anhhd/spoof/pretrained_spoof_models/"
    "trained_on_asvspoof5/nes2net/nes2net_asvspoof5.pth"
)

def _output_root() -> Path:
    return REPO_ROOT / "outputs" / "nes2net"


def _load_model_config() -> dict:
    with open(NES2NET_DIR / "config" / "WavLM_Nes2Net_ASVspoof5.conf") as f:
        return json.load(f)["model_config"]


def _resolve_meta(args) -> tuple[Path, Path]:
    return ensure_eval_meta(args.dataset, _output_root(), fold=1)


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


class Nes2NetEvalDataset(Dataset):
    def __init__(self, meta_path: Path, wav_scp_path: Path, max_len: int | None):
        df = pd.read_csv(meta_path, sep="\t")
        wav_scp = _load_wav_scp(wav_scp_path)
        self.max_len = max_len
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
        audio = audio.astype(np.float32)
        if self.max_len is not None:
            audio = _pad(audio, self.max_len)
        y = 1 if label == "bonafide" else 0
        return Tensor(audio), y, utt_id


def _collate_eval(batch):
    xs, ys, utt_ids = zip(*batch)
    max_len = max(x.numel() for x in xs)
    padded = xs[0].new_zeros((len(xs), max_len))
    for index, x in enumerate(xs):
        padded[index, : x.numel()] = x
    return padded, torch.tensor(ys, dtype=torch.long), list(utt_ids)


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


def _default_ckpt(args) -> Path:
    if args.dataset == "asvspoof5":
        return DEFAULT_CKPT_ASV5
    return DEFAULT_CKPT_2019


def _resolve_backbone(ckpt_path: Path, args) -> str:
    requested = os.environ.get("NES2NET_BACKBONE") or os.environ.get("NES2NET_ARCH")
    if requested:
        requested = requested.lower().strip()
        if requested in {"xlsr", "wav2vec2", "wav2vec"}:
            return "xlsr"
        if requested == "wavlm":
            return "wavlm"
        raise ValueError("NES2NET_BACKBONE must be one of: xlsr, wav2vec2, wavlm.")

    ckpt_name = str(ckpt_path).lower()
    if "asvspoof2019" in ckpt_name or "2019" in ckpt_name:
        return "xlsr"
    if "asvspoof5" in ckpt_name:
        return "wavlm"
    return "wavlm" if args.dataset == "asvspoof5" else "xlsr"


def _load_checkpoint_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("state_dict") or checkpoint.get("model") or checkpoint
    if not hasattr(checkpoint, "items"):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")
    return {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def _load_wavlm_model(ckpt_path: Path, device: torch.device):
    os.environ.setdefault("WAVLM_LARGE_PATH", str(DEFAULT_WAVLM))
    module = importlib.import_module("baselines.nes2net.models.wavlm_nes2net")
    model = module.Model(_load_model_config(), device=device).to(device)
    checkpoint = _load_checkpoint_state_dict(ckpt_path, device)
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load Nes2Net checkpoint {ckpt_path}. "
            "This adapter expects the WavLM-Large Nes2Net-X architecture in "
            "baselines/nes2net/models/wavlm_nes2net.py. If this checkpoint is "
            "from an XLS-R wrapper, its network definition must be added first."
        ) from exc
    return model


def _load_xlsr_model(ckpt_path: Path, device: torch.device):
    os.environ.setdefault("XLSR2_300M_PATH", str(DEFAULT_XLSR))
    try:
        module = importlib.import_module("baselines.nes2net.model_scripts.wav2vec2_Nes2Net_X")
    except ModuleNotFoundError as exc:
        if exc.name == "fairseq":
            raise ModuleNotFoundError(
                "Nes2Net XLS-R backend requires fairseq. On the server, install the "
                "fairseq snapshot used by wav2vec2-AASIST, then rerun eval."
            ) from exc
        raise
    model_args = SimpleNamespace(
        n_output_logits=2,
        dilation=2,
        pool_func="mean",
        Nes_ratio=[8, 8],
        SE_ratio=[1],
    )
    model = module.wav2vec2_Nes2Net_no_Res_w_allT(args=model_args, device=device).to(device)
    checkpoint = _load_checkpoint_state_dict(ckpt_path, device)
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Could not load Nes2Net checkpoint {ckpt_path}. "
            "This adapter expected wav2vec2/XLS-R 300M + Nes2Net-X with "
            "Nes_ratio=[8, 8], pool_func=mean, SE_ratio=1."
        ) from exc
    return model


def _load_model(ckpt_path: Path, device: torch.device, backbone: str):
    if backbone == "xlsr":
        return _load_xlsr_model(ckpt_path, device)
    if backbone == "wavlm":
        return _load_wavlm_model(ckpt_path, device)
    raise ValueError(f"Unsupported Nes2Net backbone: {backbone}")


def _run_eval(args, compute_eer: bool) -> None:
    if args.config:
        print("[nes2net] --config is ignored; using architecture settings from the selected backbone.")

    ckpt_path = Path(args.ckpt or os.environ.get("NES2NET_CKPT", _default_ckpt(args)))
    backbone = _resolve_backbone(ckpt_path, args)
    meta_path, wav_scp_path = _resolve_meta(args)
    max_len = None if args.dataset == "in_the_wild" else 64600
    if "NES2NET_EVAL_MAX_LEN" in os.environ:
        value = os.environ["NES2NET_EVAL_MAX_LEN"].strip().lower()
        max_len = None if value in {"", "none", "full"} else int(value)
    dataset = Nes2NetEvalDataset(meta_path, wav_scp_path, max_len=max_len)
    default_batch_size = 1 if max_len is None else 16
    batch_size = int(os.environ.get("NES2NET_EVAL_BATCH_SIZE", default_batch_size))
    num_workers = int(os.environ.get("NES2NET_EVAL_NUM_WORKERS", 8))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=_collate_eval,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(ckpt_path, device, backbone)
    model.eval()
    print(
        f"[nes2net] Evaluation files: {len(dataset)}, backbone={backbone}, "
        f"max_len={'full' if max_len is None else max_len}, "
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
            batch_out = model(batch_x)
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
        f.write(f"backbone={backbone}\n")
        f.write(f"wavlm_checkpoint={os.environ.get('WAVLM_LARGE_PATH')}\n")
        f.write(f"xlsr2_300m_checkpoint={os.environ.get('XLSR2_300M_PATH')}\n")
        f.write(f"max_len={'full' if max_len is None else max_len}\n")
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
        "Nes2Net training is not wired into the unified CLI. "
        "Use baselines/nes2net/main.py directly if training from scratch is needed."
    )
