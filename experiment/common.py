"""Shared utilities for SEE-MoLEx explainability experiments."""

from __future__ import annotations

import json
import math
import re
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baselines.eval_audio import collate_eval_fixed, format_error  # noqa: E402
from datasets.registry import ensure_dataset_meta  # noqa: E402
from data_utils_NEW import load_audio, pad_eval  # noqa: E402
from model_MOE import Model_MoLEx  # noqa: E402


@dataclass(frozen=True)
class ModelSpec:
    source: str
    name: str
    ablation: str
    checkpoint: Path


class ExplainEvalDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        try:
            audio = pad_eval(load_audio(str(row.path), utt_id=str(row.utt_id)))
        except Exception as exc:
            return None, int(row.label), str(row.utt_id), str(row.path), format_error(exc)
        return Tensor(audio), int(row.label), str(row.utt_id), str(row.path), None


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return yaml.safe_load(handle)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_yaml(path)
    if "sources" not in manifest:
        raise ValueError(f"Manifest has no 'sources' block: {path}")
    return manifest


def _history_best_epoch(run_dir: Path) -> int | None:
    history = run_dir / "validation_eer_history.txt"
    if not history.exists():
        return None
    candidates = []
    with history.open() as handle:
        for line in handle:
            if not line.startswith("Epoch"):
                continue
            try:
                epoch_text, eer_text = line.split(":", 1)
                checkpoint_epoch = int(epoch_text.split()[1]) - 1
                eer = float(eer_text.strip())
            except (IndexError, ValueError):
                continue
            if math.isfinite(eer):
                candidates.append((eer, checkpoint_epoch))
    for _, epoch in sorted(candidates):
        if list((run_dir / "weights").glob(f"epoch_{epoch}_*.pth")):
            return epoch
    return None


def resolve_checkpoint(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if path.is_file():
        return path
    run_dir = path.parent.parent if path.name == "weights" else path
    weights_dir = run_dir / "weights"
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint or run directory does not exist: {path}")

    best_epoch = _history_best_epoch(run_dir)
    if best_epoch is not None:
        matches = sorted(weights_dir.glob(f"epoch_{best_epoch}_*.pth"))
        if matches:
            return matches[0]

    candidates = []
    pattern = re.compile(r"epoch_(\d+)_([0-9]+(?:\.[0-9]+)?)\.pth$")
    for checkpoint in weights_dir.glob("epoch_*.pth"):
        match = pattern.fullmatch(checkpoint.name)
        if match:
            candidates.append((float(match.group(2)), int(match.group(1)), checkpoint))
    if not candidates:
        raise FileNotFoundError(f"No numeric epoch checkpoints found in {weights_dir}")
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def manifest_model_specs(
    manifest: dict[str, Any],
    sources: Iterable[str],
    models: Iterable[str],
) -> list[ModelSpec]:
    specs = []
    for source in sources:
        source_cfg = manifest["sources"][source]
        for name in models:
            model_cfg = source_cfg["models"][name]
            specs.append(
                ModelSpec(
                    source=source,
                    name=name,
                    ablation=model_cfg.get("ablation", name),
                    checkpoint=resolve_checkpoint(model_cfg["run"]),
                )
            )
    return specs


def routing_config(base_cfg: dict[str, Any], ablation: str) -> dict[str, Any]:
    if ablation not in {"M0", "M1", "M2", "M3"}:
        raise ValueError(f"Unsupported ablation: {ablation}")
    use_entropy = ablation in {"M2", "M3"}
    use_shared = ablation in {"M1", "M3"}
    routing: dict[str, Any] = {"routing_type": "entropy" if use_entropy else "topk"}
    # Keep entropy hyperparameters available for M0/M1 inference-time
    # counterfactuals. They are inert while routing_type remains "topk".
    routing.update(base_cfg["routing"])
    if use_shared:
        routing.update(
            {
                "shared_expert": True,
                "lambda_s": float(base_cfg["shared"].get("lambda_s", 1.0)),
                "lambda_r": float(base_cfg["shared"].get("lambda_r", 1.0)),
            }
        )
    return routing


def load_model(
    spec: ModelSpec,
    config_path: Path,
    device: torch.device,
) -> Model_MoLEx:
    cfg = load_yaml(config_path)
    model_cfg = dict(cfg["model_config"])
    model_cfg["routing"] = routing_config(cfg, spec.ablation)
    model = Model_MoLEx(model_cfg)
    state = torch.load(spec.checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch for {spec.checkpoint}: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    model.to(device)
    model.eval()
    install_experiment_router(model)
    return model


def iter_routers(model: torch.nn.Module):
    for layer_index, layer in enumerate(model.ssl_model.encoder.layers):
        if hasattr(layer, "smoe"):
            yield layer_index, layer.smoe.router


def iter_smoe(model: torch.nn.Module):
    for layer_index, layer in enumerate(model.ssl_model.encoder.layers):
        if hasattr(layer, "smoe"):
            yield layer_index, layer.smoe


def _dense_router_values(router, x: Tensor) -> tuple[Tensor, Tensor]:
    logits = router.topkroute_linear(x)
    eps = 1e-8 * torch.arange(
        logits.size(-1), device=logits.device, dtype=logits.dtype
    )
    logits = logits + eps
    return logits, F.softmax(logits, dim=-1)


def _entropy_active_mask(router, logits: Tensor, probs: Tensor) -> tuple[Tensor, Tensor]:
    num_experts = logits.size(-1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1) / math.log(num_experts)
    tau = max(float(router.tau_min), 1e-4)
    gamma = 1.0 - 0.5 * torch.pow(entropy, 1.0 / tau)
    max_p = probs.max(dim=-1, keepdim=True).values
    k_max = min(int(router.k_max), num_experts)
    k_min = max(min(int(router.k_min), k_max), 1)
    top_logits, top_idx = logits.topk(k_max, dim=-1)
    top_probs = probs.gather(-1, top_idx)
    keep = top_probs >= gamma.unsqueeze(-1) * max_p
    keep[..., :k_min] = True
    return top_idx, keep


def _sparse_from_mask(
    logits: Tensor,
    indices: Tensor,
    keep: Tensor,
    drop_expert: int | None = None,
) -> tuple[Tensor, Tensor]:
    top_logits = logits.gather(-1, indices).masked_fill(~keep, float("-inf"))
    sparse = torch.full_like(logits, float("-inf")).scatter(-1, indices, top_logits)
    output = F.softmax(sparse, dim=-1)
    if drop_expert is not None and 0 <= drop_expert < output.size(-1):
        output = output.clone()
        output[..., drop_expert] = 0.0
    return output, indices


def _experiment_router_forward(router, x: Tensor):
    policy = router._experiment_policy
    logits, probs = _dense_router_values(router, x)
    mode = policy["mode"]
    drop_expert = policy.get("drop_expert")

    if mode == "fixed":
        k = max(1, min(int(policy["k"]), logits.size(-1)))
        indices = logits.topk(k, dim=-1).indices
        keep = torch.ones_like(indices, dtype=torch.bool)
        return _sparse_from_mask(logits, indices, keep, drop_expert)

    adaptive_idx, adaptive_keep = _entropy_active_mask(router, logits, probs)
    if mode == "entropy":
        return _sparse_from_mask(logits, adaptive_idx, adaptive_keep, drop_expert)

    counts = adaptive_keep.sum(dim=-1).reshape(-1)
    if mode == "shuffle_k":
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(
            int(policy.get("seed", 1234))
            + int(router._experiment_layer) * 100_003
            + int(router._experiment_call)
        )
        router._experiment_call += 1
        permutation = torch.randperm(counts.numel(), device=logits.device, generator=generator)
        counts = counts[permutation].view(adaptive_keep.shape[:-1])
        rank = torch.arange(adaptive_idx.size(-1), device=logits.device)
        keep = rank.view(*([1] * (adaptive_idx.dim() - 1)), -1) < counts.unsqueeze(-1)
        return _sparse_from_mask(logits, adaptive_idx, keep, drop_expert)

    if mode == "random_experts":
        generator = torch.Generator(device=logits.device)
        generator.manual_seed(
            int(policy.get("seed", 1234))
            + int(router._experiment_layer) * 100_003
            + int(router._experiment_call)
        )
        router._experiment_call += 1
        random_rank = torch.rand(
            logits.shape, device=logits.device, dtype=logits.dtype, generator=generator
        ).argsort(dim=-1, descending=True)
        max_k = adaptive_idx.size(-1)
        indices = random_rank[..., :max_k]
        rank = torch.arange(max_k, device=logits.device)
        keep = rank.view(*([1] * (indices.dim() - 1)), -1) < counts.view(
            adaptive_keep.shape[:-1]
        ).unsqueeze(-1)
        return _sparse_from_mask(logits, indices, keep, drop_expert)

    raise ValueError(f"Unknown experiment routing mode: {mode}")


def install_experiment_router(model: torch.nn.Module) -> None:
    for layer_index, router in iter_routers(model):
        if not hasattr(router, "_experiment_policy"):
            router._experiment_policy = {
                "mode": "entropy" if router.routing_type == "entropy" else "fixed",
                "k": router.top_k,
            }
            router._experiment_layer = layer_index
            router._experiment_call = 0
            router.forward = types.MethodType(_experiment_router_forward, router)


def set_policy(model: torch.nn.Module, policy: dict[str, Any]) -> None:
    for _, router in iter_routers(model):
        router._experiment_policy = dict(policy)
        router._experiment_call = 0


def set_shared_scales(
    model: torch.nn.Module,
    lambda_s: float | None = None,
    lambda_r: float | None = None,
) -> None:
    for _, smoe in iter_smoe(model):
        if smoe.shared_expert is None:
            continue
        if lambda_s is not None:
            smoe.lambda_s = float(lambda_s)
        if lambda_r is not None:
            smoe.lambda_r = float(lambda_r)


def _enrich_metadata(
    dataset: str,
    split: str,
    frame: pd.DataFrame,
    config_roots: dict[str, str],
) -> pd.DataFrame:
    frame["attack"] = "unknown"
    frame["codec"] = "unknown"
    try:
        if dataset == "asvspoof2019la":
            tags = {"train": "train.trn", "validation": "dev.trl", "evaluation": "eval.trl"}
            root = Path(config_roots[dataset])
            protocol = (
                root
                / "ASVspoof2019_LA_cm_protocols"
                / f"ASVspoof2019.LA.cm.{tags[split]}.txt"
            )
            raw = pd.read_csv(protocol, sep=r"\s+", header=None, engine="python")
            attack = dict(zip(raw[1].astype(str), raw[3].astype(str)))
            frame["attack"] = frame["utt_id"].map(attack).fillna("unknown")
        elif dataset == "asvspoof5":
            names = {
                "train": "ASVspoof5.train.tsv",
                "validation": "ASVspoof5.dev.track_1.tsv",
                "evaluation": "ASVspoof5.eval.track_1.tsv",
            }
            protocol = Path(config_roots[dataset]) / "protocols" / names[split]
            raw = pd.read_csv(protocol, sep=r"\s+", header=None, engine="python")
            attack = dict(zip(raw[1].astype(str), raw[7].astype(str)))
            frame["attack"] = frame["utt_id"].map(attack).fillna("unknown")
        elif dataset == "vsasv":
            lowered = frame["path"].str.lower()
            frame.loc[lowered.str.contains("voice_conversion|_vc_", regex=True), "attack"] = "vc"
            frame.loc[lowered.str.contains("replay|_ra_", regex=True), "attack"] = "replay"
            frame.loc[lowered.str.contains("adversarial", regex=False), "attack"] = "adversarial"
            frame.loc[frame["label"] == 1, "attack"] = "bonafide"
        elif dataset == "fake_or_real":
            frame["attack"] = np.where(frame["label"] == 1, "bonafide", "unknown_tts")
        elif dataset == "vlsp2025":
            frame["attack"] = np.where(frame["label"] == 1, "bonafide", "spoof")
    except (FileNotFoundError, KeyError, IndexError, pd.errors.ParserError):
        pass
    return frame


def _metadata_frame(
    dataset: str,
    meta_dir: Path,
    fold: int,
    split: str,
    feat_file: Path,
    config_roots: dict[str, str],
) -> pd.DataFrame:
    frame = pd.read_csv(meta_dir / f"fold{fold}_{split}.tsv", sep="\t")
    frame = frame.iloc[:, :2].copy()
    frame.columns = ["utt_id", "label_text"]
    paths = {}
    with feat_file.open() as handle:
        for line in handle:
            parts = line.rstrip().split(maxsplit=1)
            if len(parts) == 2:
                paths[parts[0]] = parts[1].replace('"', "")
    frame["path"] = frame["utt_id"].map(paths)
    frame = frame[frame["path"].notna()].copy()
    frame["label"] = (frame["label_text"].str.lower() == "bonafide").astype(int)
    frame["group"] = frame["label_text"].str.lower()
    return _enrich_metadata(dataset, split, frame, config_roots)


def _stratified_sample(frame: pd.DataFrame, max_items: int, seed: int) -> pd.DataFrame:
    if max_items <= 0 or len(frame) <= max_items:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    groups = []
    labels = sorted(frame["label"].unique())
    per_label = max_items // max(len(labels), 1)
    for label in labels:
        indices = frame.index[frame["label"] == label].to_numpy()
        take = min(per_label, len(indices))
        groups.extend(rng.choice(indices, size=take, replace=False).tolist())
    remaining = max_items - len(groups)
    if remaining:
        pool = np.setdiff1d(frame.index.to_numpy(), np.asarray(groups), assume_unique=False)
        groups.extend(rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist())
    return frame.loc[groups].sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_loader(
    dataset: str,
    config_path: Path,
    split: str,
    max_items: int,
    seed: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, pd.DataFrame]:
    cfg = load_yaml(config_path)
    fold = int(cfg["paths"]["fold"])
    meta_root = REPO_ROOT / cfg["paths"]["meta_root"]
    meta_dir, feat_file = ensure_dataset_meta(
        dataset,
        meta_dir=meta_root / dataset,
        config_roots=cfg.get("paths", {}).get("data_root", {}),
        fold=fold,
    )
    frame = _metadata_frame(
        dataset,
        meta_dir,
        fold,
        split,
        feat_file,
        cfg.get("paths", {}).get("data_root", {}),
    )
    frame = _stratified_sample(frame, max_items=max_items, seed=seed)
    loader = DataLoader(
        ExplainEvalDataset(frame),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=collate_eval_fixed,
    )
    return loader, frame


class RoutingCollector:
    """Collect token routing statistics while retaining utterance-level summaries."""

    def __init__(
        self,
        model: torch.nn.Module,
        source: str,
        model_name: str,
        dataset: str,
        metadata: pd.DataFrame | None = None,
    ):
        self.source = source
        self.model_name = model_name
        self.dataset = dataset
        self.batch_ids: list[str] = []
        self.batch_labels: list[int] = []
        self.records: list[dict[str, Any]] = []
        self.metadata = {}
        if metadata is not None:
            self.metadata = metadata.set_index("utt_id")[["attack", "codec"]].to_dict("index")
        self.utilization: dict[tuple[int, int, int, str, str], np.ndarray] = defaultdict(
            lambda: np.zeros(3, dtype=np.float64)
        )
        self.coactivation: dict[tuple[int, int, int], int] = defaultdict(int)
        self.handles = [
            router.register_forward_hook(self._hook(layer_index))
            for layer_index, router in iter_routers(model)
        ]

    def begin_batch(self, utt_ids: list[str], labels: list[int]) -> None:
        self.batch_ids = list(utt_ids)
        self.batch_labels = [int(label) for label in labels]

    def _hook(self, layer_index: int):
        def collect(router, inputs, output):
            x = inputs[0]
            gating = output[0].detach()
            _, probs = _dense_router_values(router, x.detach())
            probs = probs.detach()
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            entropy = entropy / math.log(probs.size(-1))
            sorted_probs = probs.topk(min(2, probs.size(-1)), dim=-1).values
            margin = sorted_probs[..., 0]
            if sorted_probs.size(-1) > 1:
                margin = margin - sorted_probs[..., 1]
            active = gating > 1e-8
            active_count = active.sum(dim=-1)
            mass = gating.sum(dim=1)
            selected = active.sum(dim=1)
            # CUDA does not implement matrix multiplication for int64. Counts
            # remain exact in float32 at the batch/token sizes used here.
            active_flat = active.reshape(-1, active.size(-1)).to(torch.float32)
            coactive = active_flat.transpose(0, 1) @ active_flat
            for i in range(active.size(-1)):
                for j in range(i, active.size(-1)):
                    self.coactivation[(layer_index, i, j)] += int(coactive[i, j].cpu())

            for batch_index, utt_id in enumerate(self.batch_ids):
                label = self.batch_labels[batch_index]
                metadata = self.metadata.get(utt_id, {})
                attack = str(metadata.get("attack", "unknown"))
                codec = str(metadata.get("codec", "unknown"))
                k_values = active_count[batch_index].float()
                expert_order = mass[batch_index].argsort(descending=True)
                top_experts = ",".join(
                    str(int(idx)) for idx in expert_order[: min(3, len(expert_order))].cpu()
                )
                row = {
                    "source": self.source,
                    "model": self.model_name,
                    "dataset": self.dataset,
                    "utt_id": utt_id,
                    "label": label,
                    "attack": attack,
                    "codec": codec,
                    "layer": layer_index,
                    "entropy_mean": float(entropy[batch_index].mean().cpu()),
                    "entropy_std": float(entropy[batch_index].std(unbiased=False).cpu()),
                    "active_mean": float(k_values.mean().cpu()),
                    "total_active_mean": float(k_values.mean().cpu())
                    + (1.0 if self.model_name in {"M1", "M3"} else 0.0),
                    "active_std": float(k_values.std(unbiased=False).cpu()),
                    "active_p90": float(torch.quantile(k_values, 0.9).cpu()),
                    "pmax_mean": float(probs[batch_index].max(dim=-1).values.mean().cpu()),
                    "margin_mean": float(margin[batch_index].mean().cpu()),
                    "top_experts": top_experts,
                }
                for k in range(1, probs.size(-1) + 1):
                    row[f"k{k}_frac"] = float((active_count[batch_index] == k).float().mean().cpu())
                self.records.append(row)

                token_count = active.shape[1]
                for expert in range(probs.size(-1)):
                    key = (layer_index, expert, label, attack, codec)
                    self.utilization[key] += np.asarray(
                        [
                            float(selected[batch_index, expert].cpu()),
                            float(mass[batch_index, expert].cpu()),
                            float(token_count),
                        ]
                    )
        return collect

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        utterance = pd.DataFrame(self.records)
        util_rows = []
        for (layer, expert, label, attack, codec), values in self.utilization.items():
            selected_tokens, mass, total_tokens = values
            util_rows.append(
                {
                    "source": self.source,
                    "model": self.model_name,
                    "dataset": self.dataset,
                    "layer": layer,
                    "expert": expert,
                    "label": label,
                    "attack": attack,
                    "codec": codec,
                    "selection_rate": selected_tokens / max(total_tokens, 1.0),
                    "mean_gate_mass": mass / max(total_tokens, 1.0),
                    "selected_tokens": int(selected_tokens),
                    "total_tokens": int(total_tokens),
                }
            )
        co_rows = [
            {
                "source": self.source,
                "model": self.model_name,
                "dataset": self.dataset,
                "layer": layer,
                "expert_i": i,
                "expert_j": j,
                "coactive_tokens": count,
            }
            for (layer, i, j), count in self.coactivation.items()
        ]
        return utterance, pd.DataFrame(util_rows), pd.DataFrame(co_rows)


class ActiveCountCollector:
    """Low-memory collector for policy-level entropy and active-count summaries."""

    def __init__(self, model: torch.nn.Module):
        self.sums: dict[int, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float64))
        self.handles = [
            router.register_forward_hook(self._hook(layer_index))
            for layer_index, router in iter_routers(model)
        ]

    def _hook(self, layer_index: int):
        def collect(router, inputs, output):
            x = inputs[0].detach()
            gating = output[0].detach()
            _, probs = _dense_router_values(router, x)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            entropy = entropy / math.log(probs.size(-1))
            active = (gating > 1e-8).sum(dim=-1).float()
            self.sums[layer_index] += np.asarray(
                [
                    float(entropy.sum().cpu()),
                    float(active.sum().cpu()),
                    float(active.numel()),
                    float((active > 4).sum().cpu()),
                ]
            )

        return collect

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def frame(self) -> pd.DataFrame:
        rows = []
        for layer, values in self.sums.items():
            entropy_sum, active_sum, tokens, above_four = values
            rows.append(
                {
                    "layer": layer,
                    "entropy_mean": entropy_sum / max(tokens, 1.0),
                    "active_mean": active_sum / max(tokens, 1.0),
                    "active_gt4_frac": above_four / max(tokens, 1.0),
                    "tokens": int(tokens),
                }
            )
        return pd.DataFrame(rows)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    collector: RoutingCollector | None = None,
    desc: str = "evaluation",
) -> tuple[pd.DataFrame, list[tuple[str, str, str, str]]]:
    rows = []
    skipped = []
    model.eval()
    with torch.inference_mode():
        for batch_x, batch_y, batch_utt, batch_skipped in tqdm(
            loader, desc=desc, dynamic_ncols=True
        ):
            skipped.extend(batch_skipped)
            if batch_x is None:
                continue
            if collector is not None:
                collector.begin_batch(batch_utt, batch_y.tolist())
            logits = model(batch_x.to(device, non_blocking=True))
            scores = logits[:, 1].detach().cpu().numpy()
            for utt_id, label, score in zip(batch_utt, batch_y.tolist(), scores):
                rows.append({"utt_id": utt_id, "label": int(label), "score": float(score)})
    return pd.DataFrame(rows), skipped


def eer_and_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    bona = scores[labels == 1]
    spoof = scores[labels == 0]
    if not len(bona) or not len(spoof):
        return float("nan"), float("nan")
    all_scores = np.concatenate([bona, spoof])
    all_labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(all_scores, kind="mergesort")
    sorted_labels = all_labels[order]
    misses = np.concatenate([[0.0], np.cumsum(sorted_labels) / len(bona)])
    false_accepts = np.concatenate(
        [[1.0], (len(spoof) - (np.arange(1, len(all_scores) + 1) - np.cumsum(sorted_labels))) / len(spoof)]
    )
    thresholds = np.concatenate([[all_scores[order[0]] - 1e-6], all_scores[order]])
    index = int(np.argmin(np.abs(misses - false_accepts)))
    return float(50.0 * (misses[index] + false_accepts[index])), float(thresholds[index])


def auroc_binary(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    negative_scores = np.asarray(negative_scores, dtype=np.float64)
    positive_scores = np.asarray(positive_scores, dtype=np.float64)
    if not len(negative_scores) or not len(positive_scores):
        return float("nan")
    values = np.concatenate([negative_scores, positive_scores])
    ranks = pd.Series(values).rank(method="average").to_numpy()
    positive_rank_sum = ranks[len(negative_scores) :].sum()
    return float(
        (positive_rank_sum - len(positive_scores) * (len(positive_scores) + 1) / 2)
        / (len(negative_scores) * len(positive_scores))
    )


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if not len(values) or np.allclose(values.sum(), 0.0):
        return 0.0
    values = np.sort(np.maximum(values, 0.0))
    n = len(values)
    return float((2 * np.sum((np.arange(n) + 1) * values) / values.sum() - (n + 1)) / n)


def bootstrap_eer_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    merged = left.merge(right, on=["utt_id", "label"], suffixes=("_left", "_right"))
    labels = merged["label"].to_numpy()
    left_scores = merged["score_left"].to_numpy()
    right_scores = merged["score_right"].to_numpy()
    left_eer, _ = eer_and_threshold(labels, left_scores)
    right_eer, _ = eer_and_threshold(labels, right_scores)
    rng = np.random.default_rng(seed)
    bona_indices = np.flatnonzero(labels == 1)
    spoof_indices = np.flatnonzero(labels == 0)
    deltas = []
    for _ in range(repeats):
        indices = np.concatenate(
            [
                rng.choice(bona_indices, size=len(bona_indices), replace=True),
                rng.choice(spoof_indices, size=len(spoof_indices), replace=True),
            ]
        )
        left_boot, _ = eer_and_threshold(labels[indices], left_scores[indices])
        right_boot, _ = eer_and_threshold(labels[indices], right_scores[indices])
        deltas.append(right_boot - left_boot)
    low, high = np.quantile(deltas, [0.025, 0.975])
    return {
        "left_eer": left_eer,
        "right_eer": right_eer,
        "delta_right_minus_left": right_eer - left_eer,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "paired_items": int(len(merged)),
    }


def save_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, compression="gzip" if path.suffix == ".gz" else None)


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
