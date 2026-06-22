"""Unified CLI adapter for the MoEF ICASSP baseline."""

from __future__ import annotations

import importlib
import os
import sys
import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from baselines.eval_audio import (
    collate_eval_fixed,
    load_wav_scp,
    read_mono_audio,
    repeat_or_trim,
    tensor_or_error,
    write_eer_unavailable,
    write_skipped_audio,
)
from datasets.registry import ensure_eval_meta

REPO_ROOT = Path(__file__).resolve().parents[2]
MOEF_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_2019LA = Path("/home/user14/anhhd/spoof/BaselinesSpoofDetection/outputs/moef/2026_06_21_18_11_31")


def _output_root() -> Path:
    return REPO_ROOT / "outputs" / "moef"


def _import_moef_modules() -> None:
    moef_dir = str(MOEF_DIR)
    if moef_dir not in sys.path:
        sys.path.insert(0, moef_dir)


def _resolve_ckpt(args) -> Path:
    if args.ckpt:
        return Path(args.ckpt).expanduser()

    run_dir = Path(os.environ.get("MOEF_RUN_2019LA", DEFAULT_RUN_2019LA)).expanduser()
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not checkpoints:
        raise FileNotFoundError(f"No MoEF checkpoint found under {checkpoint_dir}")
    return checkpoints[0]


def _run_dir_from_ckpt(ckpt_path: Path) -> Path:
    if ckpt_path.parent.name == "checkpoints":
        return ckpt_path.parent.parent
    return Path(os.environ.get("MOEF_RUN_2019LA", DEFAULT_RUN_2019LA)).expanduser()


def _load_hparams(run_dir: Path, args) -> SimpleNamespace:
    hparams = {
        "seed": 888,
        "gpuid": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        "module_model": "models.moe_research.w2v2_moe_fz24_aasist",
        "tl_model": "models.tl_model_moe",
        "data_module": "utils.loadData.asvspoof_data_DA",
        "dataset": args.dataset,
        "inference": True,
        "batch_size": int(os.environ.get("MOEF_EVAL_BATCH_SIZE", "128")),
        "num_workers": int(os.environ.get("MOEF_EVAL_NUM_WORKERS", "4")),
        "epochs": 1,
        "no_best_epochs": 1,
        "savedir": str(run_dir),
        "trained_model": str(run_dir),
        "testset": "LA19",
        "truncate": int(os.environ.get("MOEF_EVAL_MAX_LEN", "64600")),
        "loss": "WCE",
        "reduce": 0,
        "loss_lr": 0.01,
        "rho": 0.5,
        "eta": "0.0",
        "optim": "adamw",
        "optim_lr": 0.00001,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "scheduler": "cosWarmup",
        "num_warmup_steps": 3,
        "total_step": 1057,
        "step_size": 5,
        "gamma": 0.1,
        "usingDA": False,
        "da_prob": 2,
        "loss_weight": 0.0,
        "moe_topk": 2,
        "moe_experts": 4,
        "moe_exp_hid": 128,
    }
    for filename in ("hparams.yaml", "hyperparameters.yaml"):
        path = run_dir / filename
        if path.exists():
            with open(path) as f:
                loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                hparams.update(loaded)
            break

    hparams.update(
        dataset=args.dataset,
        inference=True,
        savedir=str(run_dir),
        trained_model=str(run_dir),
        batch_size=int(os.environ.get("MOEF_EVAL_BATCH_SIZE", hparams["batch_size"])),
        num_workers=int(os.environ.get("MOEF_EVAL_NUM_WORKERS", hparams["num_workers"])),
        gpuid=os.environ.get("CUDA_VISIBLE_DEVICES", str(hparams["gpuid"])).split(",")[0],
    )
    return SimpleNamespace(**hparams)


def _resolve_meta(args) -> tuple[Path, Path]:
    return ensure_eval_meta(args.dataset, _output_root(), fold=1)


class MoefEvalDataset(Dataset):
    def __init__(self, meta_path: Path, wav_scp_path: Path, max_len: int):
        wav_scp = load_wav_scp(wav_scp_path)
        rows = []
        with open(meta_path, newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for values in reader:
                if len(values) < 2:
                    continue
                utt_id = str(values[0])
                label = str(values[1]).lower()
                if utt_id in wav_scp:
                    rows.append((utt_id, label, wav_scp[utt_id]))
        self.rows = rows
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        utt_id, label, path = self.rows[index]
        return tensor_or_error(
            lambda wav_path: repeat_or_trim(
                read_mono_audio(wav_path, target_sr=16000, resample_context="MoEF eval"),
                self.max_len,
            ),
            utt_id,
            label,
            path,
        )


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


def _load_model(model_args: SimpleNamespace, ckpt_path: Path, device: torch.device):
    _import_moef_modules()
    module = importlib.import_module(model_args.module_model)
    model = module.Model(model_args)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not hasattr(state_dict, "items"):
        raise RuntimeError(f"Unsupported MoEF checkpoint format: {ckpt_path}")
    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model."):]
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state[key] = value
    missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
    print(f"[moef] Loaded checkpoint: {ckpt_path}")
    print(f"[moef] Missing keys: {len(missing)}; unexpected keys: {len(unexpected)}")
    return model.to(device).eval()


def _run_eval(args, compute_eer: bool) -> None:
    ckpt_path = _resolve_ckpt(args)
    run_dir = _run_dir_from_ckpt(ckpt_path)
    model_args = _load_hparams(run_dir, args)
    meta_path, wav_scp_path = _resolve_meta(args)
    dataset = MoefEvalDataset(meta_path, wav_scp_path, max_len=int(model_args.truncate))
    batch_size = int(os.environ.get("MOEF_EVAL_BATCH_SIZE", model_args.batch_size))
    num_workers = int(os.environ.get("MOEF_EVAL_NUM_WORKERS", model_args.num_workers))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_eval_fixed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(model_args, ckpt_path, device)
    print(
        f"[moef] Evaluation files: {len(dataset)}, max_len={model_args.truncate}, "
        f"batch_size={batch_size}, num_workers={num_workers}"
    )

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = _output_root() / "evals" / f"{timestamp}__{run_dir.name}__on__{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / ("eval_output.txt" if compute_eer else "score.txt")

    utt_ids = []
    labels = []
    scores = []
    skipped_rows = []
    with torch.inference_mode():
        for batch_x, batch_y, batch_utt, batch_skipped in tqdm(loader, desc="Evaluation", dynamic_ncols=True):
            skipped_rows.extend(batch_skipped)
            if batch_x is None:
                continue
            batch_x = batch_x.to(device, non_blocking=True)
            output = model(batch_x)
            logits = output[0] if isinstance(output, (tuple, list)) else output
            batch_scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().ravel()
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
        f.write(f"run_dir={run_dir}\n")
        f.write(f"wav2vec2_checkpoint={os.environ.get('MOEF_WAV2VEC2_PATH')}\n")
        f.write("score_higher=bonafide\n")
        f.write(f"batch_size={batch_size}\n")
        f.write(f"num_workers={num_workers}\n")
        f.write(f"skipped_files={len(skipped_rows)}\n")

    write_skipped_audio(output_dir, skipped_rows)
    print(f"Scores written to {score_path}")
    if compute_eer:
        try:
            eer, threshold = _compute_eer(np.asarray(labels), np.asarray(scores))
        except ValueError as exc:
            write_eer_unavailable(output_dir, exc)
        else:
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
        "MoEF training is handled by baselines/moef_icassp/main_loss.py or moe_run.sh."
    )
