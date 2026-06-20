"""Unified CLI adapter for Whisper-MFCC-MesoNet evaluation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from datasets.registry import ensure_eval_meta

REPO_ROOT = Path(__file__).resolve().parents[2]
WHISPER_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = WHISPER_DIR / "configs" / "training" / "whisper_frontend_mesonet_mfcc.yaml"
DEFAULT_CKPT = Path(
    "/home/user14/anhhd/spoof/pretrained_spoof_models/"
    "trained_on_asvspoof2019la/whisper_mfcc_mesonet/whisper_mfcc_mesonet_finetuned.pth"
)


def _output_root() -> Path:
    return REPO_ROOT / "outputs" / "whisper_mfcc_mesonet"


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


class WhisperMFCCMesoNetEvalDataset(Dataset):
    def __init__(self, meta_path: Path, wav_scp_path: Path):
        self._prepare_imports()
        from src.datasets.base_dataset import apply_preprocessing

        self.apply_preprocessing = apply_preprocessing
        df = pd.read_csv(meta_path, sep="\t")
        wav_scp = _load_wav_scp(wav_scp_path)
        rows = []
        for row in df.itertuples(index=False):
            utt_id = str(row[0])
            label = str(row[1]).lower()
            if utt_id in wav_scp:
                rows.append((utt_id, label, wav_scp[utt_id]))
        self.rows = rows

    @staticmethod
    def _prepare_imports() -> None:
        whisper_path = str(WHISPER_DIR)
        if whisper_path not in sys.path:
            sys.path.insert(0, whisper_path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        try:
            import torchaudio
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Whisper-MFCC-MesoNet eval requires torchaudio with sox support. "
                "Install torchaudio and libsox in the nes2net_anhhd environment."
            ) from exc
        utt_id, label, path = self.rows[index]
        waveform, sample_rate = torchaudio.load(path, normalize=True)
        waveform, _ = self.apply_preprocessing(waveform, sample_rate)
        y = 1 if label == "bonafide" else 0
        return waveform, y, utt_id


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


def _load_checkpoint_state_dict(ckpt_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        checkpoint = (
            checkpoint.get("state_dict")
            or checkpoint.get("model_state_dict")
            or checkpoint.get("model")
            or checkpoint
        )
    if not hasattr(checkpoint, "items"):
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")
    state_dict = {}
    for key, value in checkpoint.items():
        for prefix in ("module.", "model.", "net."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        state_dict[key] = value
    return state_dict


def _load_model(args, ckpt_path: Path, device: torch.device):
    whisper_path = str(WHISPER_DIR)
    if whisper_path not in sys.path:
        sys.path.insert(0, whisper_path)
    from src.models import models

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    model = models.get_model(model_cfg["name"], model_cfg["parameters"], device=str(device)).to(device)
    model.load_state_dict(_load_checkpoint_state_dict(ckpt_path, device), strict=True)
    return model, config_path


def _run_eval(args, compute_eer: bool) -> None:
    ckpt_path = Path(
        args.ckpt
        or os.environ.get("WHISPER_MFCC_MESONET_CKPT")
        or os.environ.get("WHISPER_MFCC_MESONET_CKPT_2021DF")
        or DEFAULT_CKPT
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Whisper-MFCC-MesoNet checkpoint not found: {ckpt_path}. "
            "Pass --ckpt or export WHISPER_MFCC_MESONET_CKPT."
        )

    meta_path, wav_scp_path = _resolve_meta(args)
    dataset = WhisperMFCCMesoNetEvalDataset(meta_path, wav_scp_path)
    batch_size = int(os.environ.get("WHISPER_MFCC_MESONET_EVAL_BATCH_SIZE", 8))
    num_workers = int(os.environ.get("WHISPER_MFCC_MESONET_EVAL_NUM_WORKERS", 4))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config_path = _load_model(args, ckpt_path, device)
    model.eval()
    print(
        f"[whisper_mfcc_mesonet] Evaluation files: {len(dataset)}, "
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
            batch_scores = batch_out.detach().cpu().numpy().reshape(-1)
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
        f.write(f"config={config_path}\n")
        f.write(f"whisper_encoder={os.environ.get('WHISPER_MODEL_WEIGHTS_PATH', '')}\n")
        f.write("score_higher=bonafide\n")
        f.write("max_len=480000\n")
        f.write("sample_rate=16000\n")
        f.write("preprocessing=sox_silence_trim_repeat_pad\n")
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
        "Whisper-MFCC-MesoNet training is not wired into the unified CLI. "
        "Use upstream scripts in baselines/whisper_mfcc_mesonet for training."
    )
