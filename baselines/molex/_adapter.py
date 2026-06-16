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
from pathlib import Path

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
MOLEX_DIR = Path(__file__).resolve().parent
MOLEX_SRC = MOLEX_DIR / "src"

# dataset CLI name -> (adapter module, track kwarg). See AGENT_TASK.md step 5;
# datasets.asvspoof5 / datasets.asvspoof2019 must expose
# ensure_meta(data_root, meta_dir, fold, track) -> None, writing
# fold{N}_{train,validation,evaluation}.tsv + wav.scp into meta_dir.
DATASET_MODULES = {
    "asvspoof5": ("datasets.asvspoof5", None),
    "asvspoof2019la": ("datasets.asvspoof2019", "LA"),
    "asvspoof2019pa": ("datasets.asvspoof2019", "PA"),
}


def _load_yaml_config(args) -> dict:
    config_path = Path(args.config) if args.config else REPO_ROOT / "configs" / "molex.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_meta(cfg: dict, args) -> tuple[Path, Path]:
    """Make sure fold*.tsv + wav.scp exist for args.dataset; return (meta_dir, feat_file)."""
    module_name, track = DATASET_MODULES[args.dataset]
    mod = importlib.import_module(module_name)

    data_root = Path(os.environ.get("SPOOF_DATA_ROOT") or cfg["paths"]["data_root"][args.dataset])
    meta_dir = REPO_ROOT / cfg["paths"]["meta_root"] / args.dataset
    fold = cfg["paths"]["fold"]

    mod.ensure_meta(data_root=data_root, meta_dir=meta_dir, fold=fold, track=track)
    return meta_dir, meta_dir / "wav.scp"


def _json_config(cfg: dict, num_epochs: int | None) -> dict:
    return {
        "cudnn_deterministic_toggle": str(cfg.get("cudnn_deterministic_toggle", "True")),
        "cudnn_benchmark_toggle": str(cfg.get("cudnn_benchmark_toggle", "False")),

        "batch_size": cfg["batch_size"],
        "num_epochs": num_epochs if num_epochs is not None else cfg["num_epochs"],
        "model_config": cfg["model_config"],
        "optim_config": cfg["optim_config"],
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
                   fold: int, exp_idx: int, seed: int, num_gpu: int) -> None:
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
    subprocess.run(cmd, cwd=str(MOLEX_DIR), env=env, check=True)


def train(args) -> None:
    cfg = _load_yaml_config(args)
    meta_dir, feat_file = _resolve_meta(cfg, args)
    fold = cfg["paths"]["fold"]
    exp_idx = cfg["paths"]["exp_idx"]
    num_gpu = int(os.environ.get("MOLEX_NUM_GPU", cfg["runtime"]["num_gpu"]))
    output_dir = REPO_ROOT / cfg["paths"]["output_dir"]

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

        _run_torchrun(config_path, meta_dir, feat_file, output_dir, fold, exp_idx, args.seed, num_gpu)


def _load_model_and_eval_loader(cfg: dict, args):
    sys.path.insert(0, str(MOLEX_SRC))
    from data_utils_NEW import CyberEvalDataset, gen_cyber_list  # noqa: E402
    from torch.utils.data import DataLoader  # noqa: E402

    meta_dir, feat_file = _resolve_meta(cfg, args)
    fold = cfg["paths"]["fold"]

    eval_keys, eval_labs, eval_paths = gen_cyber_list(
        meta_file=meta_dir / f"fold{fold}_evaluation.tsv", feat_file=feat_file)
    eval_set = CyberEvalDataset(list_ids=eval_keys, labels=eval_labs, file_paths=eval_paths)
    eval_loader = DataLoader(eval_set, batch_size=cfg["batch_size"], shuffle=False,
                              drop_last=False, pin_memory=True, num_workers=4)

    model_config = cfg["model_config"]
    model_class = getattr(importlib.import_module("model_MOE"), model_config["model_name"])
    model = model_class(model_config)

    ckpt_path = Path(args.ckpt) if args.ckpt else (
        REPO_ROOT / cfg["paths"]["output_dir"] / f"Exp_{cfg['paths']['exp_idx']}"
        / "weights" / "averaged_checkpoint.pth"
    )
    # Checkpoints saved by main_molex.run_train come from a DDP-wrapped model
    # (model = DDP(model, ...)), so keys are prefixed with "module.".
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = {(k[len("module."):] if k.startswith("module.") else k): v
                  for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, eval_loader, device


def _run_eval(args, compute_eer: bool) -> None:
    cfg = _load_yaml_config(args)
    model, eval_loader, device = _load_model_and_eval_loader(cfg, args)
    from main_molex import compute_nist_eer, produce_evaluation_file  # noqa: E402

    output_dir = REPO_ROOT / cfg["paths"]["output_dir"] / f"Exp_{cfg['paths']['exp_idx']}"
    output_dir.mkdir(parents=True, exist_ok=True)
    score_path = output_dir / ("eval_output.txt" if compute_eer else "score.txt")

    produce_evaluation_file(eval_loader, model, device, score_path)
    print(f"Scores written to {score_path}")

    if compute_eer:
        eer, threshold = compute_nist_eer(sc_file=score_path, output_file=output_dir / "eval_EER.txt")
        print(f"EER: {eer:.3f}% (threshold {threshold:.6f})")


def eval(args) -> None:  # noqa: A001 - dispatched by name from main.py
    _run_eval(args, compute_eer=True)


def score(args) -> None:
    _run_eval(args, compute_eer=False)
