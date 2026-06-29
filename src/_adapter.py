"""Adapter wrapping the SEE-MoLEx training script for main.py's CLI.

This mirrors baselines/molex/_adapter.py but adds an ablation dispatcher
(``--ablation``) for the SEE-MoLEx proposal (stuff/proposal/main.tex, Table 4).
The model code lives in src/ (copied from baselines/molex/src and extended with
entropy-guided routing). baselines/molex/ is left untouched as the reference
baseline.

Currently implemented ablations:
  * M0  MoLEx (baseline)     -> top-K routing (identical to molex)
  * M2  + Entropy Routing    -> entropy-guided adaptive expert activation
Other ablations are recognised but report "đang được phát triển" and exit.

train() shells out to `torchrun ... main_molex.py` (CUDA + DDP, like molex).
eval()/score() load a checkpoint in a single process and evaluate directly.
"""

import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import Tensor
from torch.utils.data import Dataset

from baselines.eval_audio import collate_eval_fixed, format_error, write_eer_unavailable, write_skipped_audio
from datasets.registry import ensure_dataset_meta

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent


# --- Ablation registry -------------------------------------------------------
# id -> (display name, supported, routing type). Entropy params come from the
# config's `routing:` block, so only the routing *type* is selected here.
ABLATIONS = {
    "M0": {"name": "MoLEx (baseline)", "supported": True, "routing_type": "topk"},
    "M1": {"name": "+ Shared Expert", "supported": True, "routing_type": "topk", "shared": True},
    "M2": {"name": "+ Entropy Routing", "supported": True, "routing_type": "entropy"},
    "M3": {"name": "+ Shared + Entropy", "supported": True, "routing_type": "entropy", "shared": True},
    "M4": {"name": "SEE-MoLEx (full)", "supported": False},
    "M5": {"name": "M4 w/o warm-up", "supported": False},
    "M6": {"name": "M4, tau=1 (static)", "supported": False},
    "M7": {"name": "M3 (no consistency)", "supported": False},
}
DEFAULT_ABLATION = "M2"


def _resolve_ablation(args):
    """Return (ablation_id, spec), or None if the ablation is unsupported.

    Unsupported but recognised ablations print an "under development" notice and
    return None so the caller can exit cleanly (exit code 0).
    """
    ablation = getattr(args, "ablation", None) or DEFAULT_ABLATION
    spec = ABLATIONS.get(ablation)
    if spec is None:
        known = ", ".join(ABLATIONS)
        raise SystemExit(f"[see_molex] Unknown ablation '{ablation}'. Choose one of: {known}")
    if not spec["supported"]:
        print(f"[see_molex] Ablation study '{ablation} ({spec['name']})' đang được phát triển.")
        return None
    print(f"[see_molex] Running ablation '{ablation} ({spec['name']})'.")
    return ablation, spec


def _apply_routing(cfg: dict, spec: dict) -> dict:
    """Inject the ablation's routing config into model_config (non-mutating)."""
    routing = {"routing_type": spec["routing_type"]}
    if spec["routing_type"] == "entropy":
        routing.update(cfg.get("routing", {}))  # tau_max/tau_min/k_min/k_max/warmup_epochs
    if spec.get("shared"):
        shared_cfg = cfg.get("shared", {})
        routing["shared_expert"] = True
        routing["lambda_s"] = shared_cfg.get("lambda_s", 1.0)
        routing["lambda_r"] = shared_cfg.get("lambda_r", 1.0)
    model_config = dict(cfg["model_config"])
    model_config["routing"] = routing
    cfg = dict(cfg)
    cfg["model_config"] = model_config
    return cfg


# --- Paths / config ----------------------------------------------------------
def _output_root(ablation: str) -> Path:
    return REPO_ROOT / "outputs" / "see_molex" / ablation


def _new_run_dir(output_root: Path) -> Path:
    run_dir = output_root / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    while run_dir.exists():
        time.sleep(1)
        run_dir = output_root / datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return run_dir


def _latest_checkpoint_run_dir(output_root: Path) -> Path | None:
    checkpoints = list(output_root.rglob("weights/epoch_*.pth"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
    return latest_checkpoint.parent.parent


def _best_eer_checkpoint(run_dir: Path) -> Path | None:
    history_path = run_dir / "validation_eer_history.txt"
    if history_path.exists():
        epoch_eers = {}
        with history_path.open() as handle:
            for line in handle:
                if not line.startswith("Epoch"):
                    continue
                try:
                    epoch_text, eer_text = line.split(":", 1)
                    epoch = int(epoch_text.split()[1]) - 1
                    eer = float(eer_text.strip())
                except (IndexError, ValueError):
                    continue
                if math.isfinite(eer):
                    epoch_eers[epoch] = eer

        for epoch, _ in sorted(epoch_eers.items(), key=lambda item: (item[1], item[0])):
            checkpoints = sorted((run_dir / "weights").glob(f"epoch_{epoch}_*.pth"))
            if checkpoints:
                return checkpoints[0]

    checkpoint_eers = []
    for checkpoint in (run_dir / "weights").glob("epoch_*.pth"):
        match = re.fullmatch(r"epoch_(\d+)_([0-9]+(?:\.[0-9]+)?)\.pth", checkpoint.name)
        if match:
            checkpoint_eers.append((float(match.group(2)), int(match.group(1)), checkpoint))
    if not checkpoint_eers:
        return None
    return min(checkpoint_eers, key=lambda item: (item[0], item[1]))[2]


def _latest_run_dir(output_root: Path) -> Path | None:
    if not output_root.exists():
        return None
    run_dirs = [
        path for path in output_root.iterdir()
        if path.is_dir() and (path / "weights").is_dir()
    ]
    if not run_dirs:
        return None
    return sorted(run_dirs, key=lambda path: path.name)[-1]


def _resume_run_dir(output_root: Path, resume_arg: str) -> Path:
    if resume_arg == "latest":
        run_dir = _latest_run_dir(output_root)
        if run_dir is None:
            raise FileNotFoundError(f"No SEE-MoLEx run directory found under {output_root} for --resume.")
        return run_dir

    requested = Path(resume_arg).expanduser()
    if requested.is_absolute():
        run_dir = requested
    elif (REPO_ROOT / requested).is_dir():
        run_dir = REPO_ROOT / requested
    else:
        run_dir = output_root / requested

    if not run_dir.is_dir():
        raise FileNotFoundError(f"SEE-MoLEx resume run directory does not exist: {run_dir}")
    return run_dir


def _load_yaml_config(args) -> dict:
    config_path = Path(args.config) if args.config else REPO_ROOT / "configs" / "see_molex.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_batch_sizes = cfg.get("batch_size_by_dataset", {})
    if args.dataset in dataset_batch_sizes:
        cfg["batch_size"] = int(dataset_batch_sizes[args.dataset])
    return cfg


def _resolve_meta(cfg: dict, args) -> tuple[Path, Path]:
    """Make sure fold*.tsv + wav.scp exist for args.dataset; return (meta_dir, feat_file)."""
    meta_dir = REPO_ROOT / cfg["paths"]["meta_root"] / args.dataset
    fold = cfg["paths"]["fold"]
    return ensure_dataset_meta(
        args.dataset,
        meta_dir=meta_dir,
        config_roots=cfg.get("paths", {}).get("data_root", {}),
        fold=fold,
    )


def _json_config(cfg: dict, num_epochs: int | None) -> dict:
    return {
        "cudnn_deterministic_toggle": str(cfg.get("cudnn_deterministic_toggle", "True")),
        "cudnn_benchmark_toggle": str(cfg.get("cudnn_benchmark_toggle", "False")),

        "batch_size": cfg["batch_size"],
        "num_epochs": num_epochs if num_epochs is not None else cfg["num_epochs"],
        "model_config": cfg["model_config"],
        "optim_config": cfg["optim_config"],
        "runtime": cfg.get("runtime", {}),
    }


def _truncate_meta(meta_dir: Path, tmp_dir: Path, fold: int, n_rows: int) -> Path:
    """Write a tiny copy of fold*.tsv + wav.scp under tmp_dir, for smoke testing."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    kept_ids = set()
    for split in ("train", "validation", "evaluation"):
        df = pd.read_csv(meta_dir / f"fold{fold}_{split}.tsv", sep="\t")
        if len(df) > n_rows:
            id_col, label_col = df.columns[:2]
            half = max(n_rows // 2, 1)
            bonafide = df[df[label_col] == "bonafide"].head(half)
            spoof = df[df[label_col] != "bonafide"].head(n_rows - len(bonafide))
            df = pd.concat([bonafide, spoof], ignore_index=True)
            if len(df) < n_rows:
                used = set(df[id_col].tolist())
                source = pd.read_csv(meta_dir / f"fold{fold}_{split}.tsv", sep="\t")
                filler = source[~source[id_col].isin(used)].head(n_rows - len(df))
                df = pd.concat([df, filler], ignore_index=True)
        df.to_csv(tmp_dir / f"fold{fold}_{split}.tsv", sep="\t", index=False)
        kept_ids.update(df.iloc[:, 0].tolist())

    with open(meta_dir / "wav.scp") as fin, open(tmp_dir / "wav.scp", "w") as fout:
        for line in fin:
            if line.split(maxsplit=1)[0] in kept_ids:
                fout.write(line)
    return tmp_dir


def _run_torchrun(config_path: Path, meta_dir: Path, feat_file: Path, output_dir: Path,
                  fold: int, exp_idx: int, seed: int, num_gpu: int, resume: str | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={num_gpu}",
        str(SRC_DIR / "main_molex.py"),
        "--config", str(config_path),
        "--meta_dir", str(meta_dir),
        "--feat_file", str(feat_file),
        "--output_dir", str(output_dir),
        "--fold", str(fold),
        "--exp_idx", str(exp_idx),
        "--seed", str(seed),
    ]
    if resume:
        cmd.append("--resume")
    subprocess.run(cmd, cwd=str(SRC_DIR), env=env, check=True)


class SeeMolexSafeEvalDataset(Dataset):
    def __init__(self, list_ids, labels, file_paths):
        self.list_ids = list(list_ids)
        self.labels = list(labels)
        self.file_paths = list(file_paths)

    def __len__(self) -> int:
        return len(self.list_ids)

    def __getitem__(self, index: int):
        from data_utils_NEW import load_audio, pad_eval  # noqa: E402

        utt_id = self.list_ids[index]
        y = int(self.labels[index])
        path = self.file_paths[index]
        try:
            audio = pad_eval(load_audio(path, utt_id=utt_id))
        except Exception as exc:
            return None, y, utt_id, path, format_error(exc)
        return Tensor(audio), y, utt_id, path, None


def train(args) -> None:
    resolved = _resolve_ablation(args)
    if resolved is None:
        return
    ablation, spec = resolved

    cfg = _apply_routing(_load_yaml_config(args), spec)
    print(f"[see_molex] Training batch size for {args.dataset}: {cfg['batch_size']}")
    meta_dir, feat_file = _resolve_meta(cfg, args)
    fold = cfg["paths"]["fold"]
    exp_idx = cfg["paths"]["exp_idx"]
    num_gpu = int(os.environ.get("MOLEX_NUM_GPU", cfg["runtime"]["num_gpu"]))
    output_root = _output_root(ablation)
    if args.resume:
        output_dir = _resume_run_dir(output_root, args.resume)
        print(f"[see_molex] Resuming from run directory: {output_dir}")
    else:
        output_dir = _new_run_dir(output_root)
        print(f"[see_molex] Starting a new run directory: {output_dir}")

    num_epochs = None
    with tempfile.TemporaryDirectory(prefix="see_molex_run_") as tmp:
        tmp_dir = Path(tmp)

        if args.dry_run or args.max_steps:
            steps = args.max_steps or 2
            n_rows = max(steps, 2) * cfg["batch_size"]
            meta_dir = _truncate_meta(meta_dir, tmp_dir / "meta", fold, n_rows)
            feat_file = meta_dir / "wav.scp"
            num_epochs = 1

        config_path = tmp_dir / "see_molex_run.json"
        with open(config_path, "w") as f:
            json.dump(_json_config(cfg, num_epochs), f, indent=2)

        _run_torchrun(config_path, meta_dir, feat_file, output_dir, fold, exp_idx, args.seed, num_gpu, args.resume)


def _load_model_and_eval_loader(cfg: dict, args, ablation: str):
    sys.path.insert(0, str(SRC_DIR))
    from data_utils_NEW import gen_cyber_list  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    meta_dir, feat_file = _resolve_meta(cfg, args)
    fold = cfg["paths"]["fold"]

    eval_keys, eval_labs, eval_paths = gen_cyber_list(
        meta_file=meta_dir / f"fold{fold}_evaluation.tsv", feat_file=feat_file)
    eval_set = SeeMolexSafeEvalDataset(list_ids=eval_keys, labels=eval_labs, file_paths=eval_paths)
    runtime_cfg = cfg.get("runtime", {})
    eval_batch_size = int(os.environ.get(
        "MOLEX_EVAL_BATCH_SIZE",
        runtime_cfg.get("eval_batch_size", cfg["batch_size"]),
    ))
    eval_num_workers = int(os.environ.get(
        "MOLEX_EVAL_NUM_WORKERS",
        runtime_cfg.get("eval_num_workers", 4),
    ))
    eval_loader = DataLoader(
        eval_set,
        batch_size=eval_batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=eval_num_workers,
        collate_fn=collate_eval_fixed,
    )
    print(
        f"[see_molex] Evaluation files: {len(eval_set)}, "
        f"batch_size={eval_batch_size}, num_workers={eval_num_workers}"
    )

    model_config = cfg["model_config"]
    model_class = getattr(importlib.import_module("model_MOE"), model_config["model_name"])
    model = model_class(model_config)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        run_dir = _latest_checkpoint_run_dir(_output_root(ablation))
        if run_dir is None:
            raise FileNotFoundError(f"No SEE-MoLEx checkpoint found under {_output_root(ablation)}.")
        ckpt_path = _best_eer_checkpoint(run_dir)
        if ckpt_path is None:
            raise FileNotFoundError(f"No finite dev-EER checkpoint found under {run_dir / 'weights'}.")
    # Checkpoints saved by main_molex.run_train come from a DDP-wrapped model
    # (model = DDP(model, ...)), so keys are prefixed with "module.".
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, eval_loader, device, ckpt_path


def _eval_output_dir(args, ckpt_path: Path, ablation: str) -> Path:
    checkpoint_run = ckpt_path.parent.parent.name if ckpt_path.parent.name == "weights" else ckpt_path.stem
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return _output_root(ablation) / "evals" / f"{timestamp}__{checkpoint_run}__on__{args.dataset}"


def _run_eval(args, compute_eer: bool) -> None:
    resolved = _resolve_ablation(args)
    if resolved is None:
        return
    ablation, spec = resolved

    cfg = _apply_routing(_load_yaml_config(args), spec)
    model, eval_loader, device, ckpt_path = _load_model_and_eval_loader(cfg, args, ablation)
    from main_molex import compute_nist_eer  # noqa: E402
    from tqdm import tqdm  # noqa: E402

    output_dir = _eval_output_dir(args, ckpt_path, ablation)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / ("eval_output.txt" if compute_eer else "score.txt")

    fname_list = []
    lab_list = []
    score_list = []
    skipped_rows = []
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_y, batch_utt, batch_skipped in tqdm(
            eval_loader, desc="Evaluation", dynamic_ncols=True, leave=True
        ):
            skipped_rows.extend(batch_skipped)
            if batch_x is None:
                continue
            batch_x = batch_x.to(device, non_blocking=True)
            batch_out = model(batch_x)
            batch_scores = batch_out[:, 1].detach().cpu().numpy().ravel()
            fname_list.extend(batch_utt)
            lab_list.extend(batch_y.numpy().tolist())
            score_list.extend(batch_scores.tolist())

    with open(score_path, "w") as f:
        for utt_id, label, score in zip(fname_list, lab_list, score_list):
            label_text = "bonafide" if label == 1 else "spoof"
            f.write(f"{utt_id}\t{label_text}\t{score}\n")

    with open(output_dir / "eval_config.txt", "w") as f:
        f.write(f"ablation={ablation}\n")
        f.write(f"dataset={args.dataset}\n")
        f.write(f"checkpoint={ckpt_path}\n")
        f.write(f"skipped_files={len(skipped_rows)}\n")

    write_skipped_audio(output_dir, skipped_rows)
    print(f"Scores written to {score_path}")

    if compute_eer:
        try:
            eer, threshold = compute_nist_eer(sc_file=score_path, output_file=output_dir / "eval_EER.txt")
        except ValueError as exc:
            write_eer_unavailable(output_dir, exc)
        else:
            print(f"EER: {eer:.3f}% (threshold {threshold:.6f})")


def eval(args) -> None:  # noqa: A001 - dispatched by name from main.py
    _run_eval(args, compute_eer=True)


def score(args) -> None:
    _run_eval(args, compute_eer=False)
