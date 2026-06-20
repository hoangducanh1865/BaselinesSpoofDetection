"""Adapter wrapping the upstream MoLEx training script for main.py's CLI.

baselines/molex/src/main_molex.py requires CUDA and DDP (dist.init_process_group("nccl"))
unconditionally and has no in-process single-GPU path or max-steps hook -- only epoch-based
training via torchrun. So train() shells out to `torchrun ... main_molex.py` rather than
reimplementing its training loop.

eval()/score() do NOT go through main_molex.py: its "evaluation only" branch is broken
upstream (get_DDP_loader returns trn_loader=None when no train/dev meta exist, and run_train
then calls len(trn_loader) unconditionally, crashing). Instead they import main_molex's own
helper functions (produce_evaluation_file, compute_nist_eer) and data_utils_NEW's dataset
classes directly, in a single process, to evaluate a given checkpoint.
"""

import importlib
import json
import os
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

from baselines.eval_audio import collate_eval_fixed, format_error, write_skipped_audio
from datasets.registry import ensure_dataset_meta

REPO_ROOT = Path(__file__).resolve().parents[2]
MOLEX_DIR = Path(__file__).resolve().parent
MOLEX_SRC = MOLEX_DIR / "src"


def _output_root() -> Path:
    return REPO_ROOT / "outputs" / "molex"


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
            raise FileNotFoundError(f"No MoLEx run directory found under {output_root} for --resume.")
        return run_dir

    run_dir = output_root / resume_arg
    if not run_dir.is_dir():
        raise FileNotFoundError(f"MoLEx resume run directory does not exist: {run_dir}")
    return run_dir


def _load_yaml_config(args) -> dict:
    config_path = Path(args.config) if args.config else REPO_ROOT / "configs" / "molex.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


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
                filler = df.iloc[0:0]
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
    env["PYTHONPATH"] = str(MOLEX_SRC) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={num_gpu}",
        str(MOLEX_SRC / "main_molex.py"),
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
    subprocess.run(cmd, cwd=str(MOLEX_DIR), env=env, check=True)


class MolexSafeEvalDataset(Dataset):
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
    cfg = _load_yaml_config(args)
    meta_dir, feat_file = _resolve_meta(cfg, args)
    fold = cfg["paths"]["fold"]
    exp_idx = cfg["paths"]["exp_idx"]
    num_gpu = int(os.environ.get("MOLEX_NUM_GPU", cfg["runtime"]["num_gpu"]))
    output_root = _output_root()
    if args.resume:
        output_dir = _resume_run_dir(output_root, args.resume)
        print(f"[molex] Resuming from run directory: {output_dir}")
    else:
        output_dir = _new_run_dir(output_root)
        print(f"[molex] Starting a new run directory: {output_dir}")

    num_epochs = None
    with tempfile.TemporaryDirectory(prefix="molex_run_") as tmp:
        tmp_dir = Path(tmp)

        if args.dry_run or args.max_steps:
            steps = args.max_steps or 2
            n_rows = max(steps, 2) * cfg["batch_size"]
            meta_dir = _truncate_meta(meta_dir, tmp_dir / "meta", fold, n_rows)
            feat_file = meta_dir / "wav.scp"
            num_epochs = 1

        config_path = tmp_dir / "molex_run.json"
        with open(config_path, "w") as f:
            json.dump(_json_config(cfg, num_epochs), f, indent=2)

        _run_torchrun(config_path, meta_dir, feat_file, output_dir, fold, exp_idx, args.seed, num_gpu, args.resume)


def _load_model_and_eval_loader(cfg: dict, args):
    sys.path.insert(0, str(MOLEX_SRC))
    from data_utils_NEW import gen_cyber_list  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    meta_dir, feat_file = _resolve_meta(cfg, args)
    fold = cfg["paths"]["fold"]

    eval_keys, eval_labs, eval_paths = gen_cyber_list(
        meta_file=meta_dir / f"fold{fold}_evaluation.tsv", feat_file=feat_file)
    eval_set = MolexSafeEvalDataset(list_ids=eval_keys, labels=eval_labs, file_paths=eval_paths)
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
        f"[molex] Evaluation files: {len(eval_set)}, "
        f"batch_size={eval_batch_size}, num_workers={eval_num_workers}"
    )

    model_config = cfg["model_config"]
    model_class = getattr(importlib.import_module("model_MOE"), model_config["model_name"])
    model = model_class(model_config)

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        run_dir = _latest_checkpoint_run_dir(_output_root())
        if run_dir is None:
            raise FileNotFoundError(f"No MoLEx checkpoint found under {_output_root()}.")
        ckpt_path = run_dir / "weights" / "averaged_checkpoint.pth"
    # Checkpoints saved by main_molex.run_train come from a DDP-wrapped model
    # (model = DDP(model, ...)), so keys are prefixed with "module.".
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, eval_loader, device, ckpt_path


def _eval_output_dir(args, ckpt_path: Path) -> Path:
    checkpoint_run = ckpt_path.parent.parent.name if ckpt_path.parent.name == "weights" else ckpt_path.stem
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    return _output_root() / "evals" / f"{timestamp}__{checkpoint_run}__on__{args.dataset}"


def _run_eval(args, compute_eer: bool) -> None:
    cfg = _load_yaml_config(args)
    model, eval_loader, device, ckpt_path = _load_model_and_eval_loader(cfg, args)
    from main_molex import compute_nist_eer  # noqa: E402
    from tqdm import tqdm  # noqa: E402

    output_dir = _eval_output_dir(args, ckpt_path)
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
        f.write(f"dataset={args.dataset}\n")
        f.write(f"checkpoint={ckpt_path}\n")
        f.write(f"skipped_files={len(skipped_rows)}\n")

    write_skipped_audio(output_dir, skipped_rows)
    print(f"Scores written to {score_path}")

    if compute_eer:
        eer, threshold = compute_nist_eer(sc_file=score_path, output_file=output_dir / "eval_EER.txt")
        print(f"EER: {eer:.3f}% (threshold {threshold:.6f})")


def eval(args) -> None:  # noqa: A001 - dispatched by name from main.py
    _run_eval(args, compute_eer=True)


def score(args) -> None:
    _run_eval(args, compute_eer=False)
