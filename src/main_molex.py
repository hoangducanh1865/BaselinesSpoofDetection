"""
Distributed training entrypoint for the MoLEx anti-spoofing model.
Official implementation of the paper: 
"MoLEx: Mixture of Low-Rank Experts for Efficient Fine-Tuning of Self-Supervised Audio Models"
Author: Zihan Pan
"""



import argparse
import json
import math
import os
import re
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from data_utils_NEW import (
    CyberDataset,
    CyberEvalDataset,
    gen_cyber_list,
)
from evaluation import compute_nist_eer
from utils import create_optimizer, seed_worker, set_seed

try:
    import wandb
except ImportError:
    wandb = None

warnings.filterwarnings("ignore", category=FutureWarning)
current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def append_run_log(run_log_path, message):
    with open(run_log_path, 'a') as file:
        file.write(message + "\n")


def log_rank0(rank, run_log_path, message):
    if rank == 0:
        print(message, flush=True)
        append_run_log(run_log_path, message)


def init_wandb_run(rank, run_log_path, model_tag, config, args, meta_path, feat_file):
    if rank != 0:
        return None

    if wandb is None:
        log_rank0(rank, run_log_path, "[wandb] wandb is not installed; skipping W&B logging.")
        return None

    if not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE", "online") != "offline":
        log_rank0(rank, run_log_path, "[wandb] WANDB_API_KEY is not set; skipping W&B logging.")
        return None

    run_id_path = model_tag / "wandb_run_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
    else:
        run_id = model_tag.name
        run_id_path.write_text(run_id + "\n")

    project = os.environ.get("WANDB_PROJECT", "BaselinesSpoofDetection-MoLEx")
    entity = os.environ.get("WANDB_ENTITY") or None

    try:
        if os.environ.get("WANDB_API_KEY"):
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=False)
        run = wandb.init(
            project=project,
            entity=entity,
            id=run_id,
            name=f"molex-{model_tag.name}",
            resume="allow",
            dir=str(model_tag),
            config={
                "seed": args.seed,
                "fold": args.fold,
                "exp_idx": args.exp_idx,
                "resume": args.resume,
                "pretrain_checkpoint": args.pretrain_checkpoint,
                "run_dir": str(model_tag),
                "meta_dir": str(meta_path),
                "feat_file": str(feat_file),
                "num_epochs": config["num_epochs"],
                "batch_size": config["batch_size"],
                "model_config": config["model_config"],
                "optim_config": config["optim_config"],
            },
        )
    except Exception as exc:
        log_rank0(rank, run_log_path, f"[wandb] Failed to initialize W&B: {exc}")
        return None

    wandb.save(str(model_tag / "hyperparameters.json"), base_path=str(model_tag), policy="now")
    wandb.save(str(model_tag / "config.conf"), base_path=str(model_tag), policy="now")
    log_rank0(rank, run_log_path, f"[wandb] Logging to project={project}, run_id={run_id}.")
    return run


def cleanup():
    """Tear down the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def _checkpoint_state_dict(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    return {
        (name[len("module."):] if name.startswith("module.") else name): value
        for name, value in checkpoint.items()
    }


def load_expanded_expert_checkpoint(model, checkpoint_path, source_num_experts):
    """Load a source checkpoint while retaining random rows for added experts."""
    source_state = _checkpoint_state_dict(checkpoint_path)
    target_state = model.state_dict()
    loaded = 0
    expanded_router_tensors = 0
    unexpected = []

    for name, source_value in source_state.items():
        if name not in target_state:
            unexpected.append(name)
            continue
        target_value = target_state[name]
        if source_value.shape == target_value.shape:
            target_state[name] = source_value
            loaded += 1
            continue
        is_expanded_router = (
            ".router." in name
            and source_value.ndim == target_value.ndim
            and source_value.shape[0] == source_num_experts
            and target_value.shape[0] > source_value.shape[0]
            and source_value.shape[1:] == target_value.shape[1:]
        )
        if is_expanded_router:
            expanded = target_value.clone()
            expanded[:source_num_experts].copy_(source_value)
            target_state[name] = expanded
            expanded_router_tensors += 1
            continue
        raise ValueError(
            f"Unsupported checkpoint shape mismatch for {name}: "
            f"source={tuple(source_value.shape)}, target={tuple(target_value.shape)}"
        )

    if unexpected:
        raise ValueError(
            f"Checkpoint contains {len(unexpected)} parameters absent from the target model; "
            f"first entries: {unexpected[:5]}"
        )
    model.load_state_dict(target_state, strict=True)
    return loaded, expanded_router_tensors


def configure_domain_expert_adaptation(model, adaptation_config):
    """Freeze the source model and expose only added experts and routers."""
    source_num_experts = int(adaptation_config["source_num_experts"])
    new_experts = int(adaptation_config["new_experts"])
    expected_total = source_num_experts + new_experts
    train_router = bool(adaptation_config.get("train_router", True))
    train_classifier = bool(adaptation_config.get("train_classifier", False))
    train_featfusion = bool(adaptation_config.get("train_featfusion", False))

    model.requires_grad_(False)
    selected_experts = []
    router_params = []
    moe_layers = [layer for layer in model.ssl_model.encoder.layers if hasattr(layer, "smoe")]
    if not moe_layers:
        raise ValueError("Domain-expert adaptation requires at least one MoE layer.")

    for layer_index, layer in enumerate(moe_layers):
        experts = layer.smoe.experts
        if len(experts) != expected_total:
            raise ValueError(
                f"MoE layer {layer_index} has {len(experts)} experts; expected "
                f"{source_num_experts} source + {new_experts} new = {expected_total}."
            )
        for expert in experts[source_num_experts:]:
            expert.requires_grad_(True)
            selected_experts.append(expert)
        if train_router:
            layer.smoe.router.requires_grad_(True)
            router_params.extend(layer.smoe.router.parameters())

    auxiliary_params = list(router_params)
    if train_classifier:
        model.decoder.requires_grad_(True)
        auxiliary_params.extend(model.decoder.parameters())
    if train_featfusion:
        model.featfusion.requires_grad_(True)
        auxiliary_params.extend(model.featfusion.parameters())

    expert_params = [
        parameter
        for expert in selected_experts
        for parameter in expert.parameters()
        if parameter.requires_grad
    ]
    auxiliary_params = [parameter for parameter in auxiliary_params if parameter.requires_grad]
    if not expert_params:
        raise ValueError("No new expert parameters were selected for adaptation.")
    if not auxiliary_params:
        raise ValueError("No router or auxiliary parameters were selected for adaptation.")
    return [expert_params, auxiliary_params], selected_experts


def _parameter_count(parameters):
    return sum(parameter.numel() for parameter in parameters)


def run_train(args):
    """
    Main function.
    Trains, validates, and evaluates the model.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for distributed training.")

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")

    # create model and move it to GPU with id rank
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No CUDA devices detected.")
    device_id = rank % num_devices
    device = torch.device(f"cuda:{device_id}")

    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]

    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    fold_id = args.fold
    meta_path = Path(args.meta_dir)
    feat_file = Path(args.feat_file)

    trn_list_path = (meta_path / f"fold{fold_id}_train.tsv")
    dev_trial_path = (meta_path / f"fold{fold_id}_validation.tsv")
    eval_trial_path = (meta_path / f"fold{fold_id}_evaluation.tsv")

    model_tag = output_dir
    model_save_path = model_tag / "weights"
    run_log_path = model_tag / "run.log"
    
    os.makedirs(model_save_path, exist_ok=True)
    if rank == 0:
        copy(args.config, model_tag / "config.conf")
        copy(args.config, model_tag / "hyperparameters.json")
        append_run_log(run_log_path, "=" * 80)
        append_run_log(run_log_path, f"Run started at {current_time}")
        append_run_log(run_log_path, f"Run directory: {model_tag}")
        append_run_log(run_log_path, f"Config: {args.config}")
        append_run_log(run_log_path, f"Meta dir: {meta_path}")
        append_run_log(run_log_path, f"Feature file: {feat_file}")
        append_run_log(run_log_path, f"Resume requested: {args.resume}")
        append_run_log(run_log_path, f"Initialization checkpoint: {args.pretrain_checkpoint}")
        append_run_log(run_log_path, "=" * 80)

    wandb_run = init_wandb_run(rank, run_log_path, model_tag, config, args, meta_path, feat_file)

    import importlib
    model_name = model_config['model_name']
    model_class = getattr(importlib.import_module('model_MOE'), model_name)
    log_rank0(rank, run_log_path, f"[setup] Run the model: {model_class}")
    log_rank0(rank, run_log_path, "[setup] Initializing model and loading WavLM checkpoint...")

    model = model_class(model_config)
    log_rank0(rank, run_log_path, "[setup] Model initialized.")

    adaptation_config = config.get("adaptation", {})
    adaptation_enabled = bool(adaptation_config.get("enabled", False))
    orth_experts = None
    if adaptation_enabled:
        if not args.resume:
            if not args.pretrain_checkpoint:
                raise ValueError("Adaptation requires --pretrain_checkpoint for a new run.")
            source_num_experts = int(adaptation_config["source_num_experts"])
            loaded, expanded = load_expanded_expert_checkpoint(
                model, args.pretrain_checkpoint, source_num_experts
            )
            log_rank0(
                rank,
                run_log_path,
                f"[adaptation] Initialized from {args.pretrain_checkpoint}; "
                f"loaded_tensors={loaded}, expanded_router_tensors={expanded}.",
            )
        params_backend, orth_experts = configure_domain_expert_adaptation(
            model, adaptation_config
        )
        log_rank0(
            rank,
            run_log_path,
            "[adaptation] Trainable scope: newly added experts + routers; "
            f"expert_params={_parameter_count(params_backend[0]):,}, "
            f"router_aux_params={_parameter_count(params_backend[1]):,}, "
            f"total={_parameter_count(params_backend[0] + params_backend[1]):,}.",
        )
    else:
        class_head_param = list(model.decoder.parameters()) + (
            list(model.featfusion.parameters()) if hasattr(model, "featfusion") else []
        )
        lora_adapt_param = (
            model.get_MOE_param_list() if hasattr(model, "num_MOE_layer") else []
        )
        params_backend = [lora_adapt_param, class_head_param]
    log_rank0(rank, run_log_path, "[setup] Parameter groups prepared.")


    model = model.to(device)
    model = DDP(model, device_ids=[device_id],find_unused_parameters=True)
    log_rank0(rank, run_log_path, f"[setup] Model moved to {device} and wrapped with DDP.")


    # define dataloaders
    log_rank0(rank, run_log_path, "[data] Building train/validation/evaluation dataloaders...")
    trn_loader, dev_loader, eval_loader, train_sampler = get_DDP_loader(args, feat_file, trn_list_path,
                                                     dev_trial_path, eval_trial_path,
                                                     args.seed, config)
    log_rank0(
        rank,
        run_log_path,
        f"[data] Dataloaders ready: train_batches={len(trn_loader)}, "
        f"valid_batches={len(dev_loader)}, eval_batches={len(eval_loader)}."
    )
    if rank == 0 and wandb_run is not None:
        wandb_run.config.update(
            {
                "train_batches": len(trn_loader),
                "valid_batches": len(dev_loader),
                "eval_batches": len(eval_loader),
                "steps_per_epoch": len(trn_loader),
            },
            allow_val_change=True,
        )


    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)

    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    log_rank0(rank, run_log_path, "[setup] Creating optimizer and scheduler...")
    optimizer, scheduler= create_optimizer(params_backend, optim_config)    
    log_rank0(rank, run_log_path, "[setup] Optimizer and scheduler created.")
    wandb_log_interval = max(
        int(os.environ.get("WANDB_LOG_INTERVAL", config.get("runtime", {}).get("wandb_log_interval", 100))),
        1,
    )
    moe_layers = [layer for layer in model.module.ssl_model.encoder.layers if hasattr(layer, 'smoe')]

    if args.resume:
        log_rank0(rank, run_log_path, "[resume] Searching for the latest resume checkpoint...")
        resume_checkpoint_path, start_epoch = find_latest_resume_checkpoint(model_save_path)
        if resume_checkpoint_path is None:
            resume_checkpoint_path, start_epoch = find_latest_epoch_checkpoint(model_save_path)
            if resume_checkpoint_path is not None:
                log_rank0(
                    rank,
                    run_log_path,
                    "[resume] latest_checkpoint file not found; falling back to the latest epoch checkpoint."
                )
        best_dev_eer = read_best_dev_eer(model_tag)
        if resume_checkpoint_path is None:
            raise FileNotFoundError(
                f"--resume was requested, but no model checkpoint was found in {model_save_path}"
            )
        checkpoint = torch.load(resume_checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        model.load_state_dict(checkpoint)
        training_state_path = get_training_state_path(model_save_path, start_epoch - 1)
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=device)
            optimizer.load_state_dict(training_state["optimizer"])
            if scheduler is not None and training_state.get("scheduler") is not None:
                scheduler.load_state_dict(training_state["scheduler"])
            best_dev_eer = training_state.get("best_dev_eer", best_dev_eer)
            log_rank0(rank, run_log_path, f"[resume] Loaded optimizer/scheduler state from {training_state_path}.")
        else:
            log_rank0(rank, run_log_path, f"[resume] No optimizer/scheduler state found at {training_state_path}; resuming model weights only.")
        log_rank0(rank, run_log_path, f"[resume] Resumed model from {resume_checkpoint_path}; starting at epoch {start_epoch:03d}.")
    else:
        start_epoch = 0
        best_dev_eer = float("inf")
        log_rank0(rank, run_log_path, "[resume] Resume disabled; training from scratch.")

    if start_epoch >= config["num_epochs"]:
        log_rank0(
            rank,
            run_log_path,
            f"Latest checkpoint is epoch {start_epoch - 1:03d}; "
            f"num_epochs={config['num_epochs']}, so no additional training epochs are required."
        )

    for epoch in range(start_epoch, config["num_epochs"]):
        log_rank0(rank, run_log_path, f"Start training epoch{epoch:03d}")

        running_loss = 0
        running_ortho_loss = 0
        num_total = 0.0
        model.train()

        train_sampler.set_epoch(epoch)

        # Entropy routing (M2): expose epoch progress to each router for
        # temperature annealing + warm-up. No-op for the baseline top-K router.
        for layer in moe_layers:
            layer.smoe.router.current_epoch = epoch
            layer.smoe.router.total_epochs = config["num_epochs"]

        # set objective (Loss) functions
        weight = torch.FloatTensor([0.1, 0.9]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight).to(device)

        train_iter = tqdm(
            trn_loader,
            desc=f"Epoch {epoch:03d} train",
            disable=(rank != 0),
            dynamic_ncols=True,
            leave=True,
        )

        for batch_idx, (batch_x, batch_y, utt_id) in enumerate(train_iter):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.view(-1).type(torch.int64).to(device, non_blocking=True)

            optimizer.zero_grad()

            batch_out = model(batch_x) 
            batch_loss = criterion(batch_out, batch_y)

            # add orthogonal loss
            if orth_experts is not None:
                orth_loss = lora_orthogonality_loss(orth_experts)
            else:
                orth_loss = sum(
                    lora_orthogonality_loss(layer.smoe.experts)
                    for layer in moe_layers
                ) if moe_layers else 0
            orth_loss_value = float(orth_loss.detach().item()) if torch.is_tensor(orth_loss) else float(orth_loss)
            batch_loss = batch_loss + orth_loss*0.01


            running_loss += batch_loss.item() * batch_size    
            running_ortho_loss += orth_loss_value * batch_size

            batch_loss.backward()

            optimizer.step()   

            if rank == 0:
                train_iter.set_postfix(
                    loss=f"{batch_loss.item():.4f}",
                    orth=f"{orth_loss_value:.4f}",
                )
                if wandb_run is not None and batch_idx % wandb_log_interval == 0:
                    global_step = epoch * len(trn_loader) + batch_idx
                    wandb_run.log(
                        {
                            "train/batch_loss": batch_loss.item(),
                            "train/orth_loss": orth_loss_value,
                            "train/epoch": epoch,
                            "train/lr": scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )

        running_loss /= num_total       
        running_ortho_loss /= num_total 
      
        valid_loss = produce_evaluation_file(dev_loader, model, device,
                                metric_path / "dev_score.txt",
                                desc=f"Epoch {epoch:03d} valid",
                                disable=(rank != 0))
            

        dev_eer, dev_th = compute_nist_eer(sc_file=metric_path / "dev_score.txt",
                                        output_file=metric_path / "dev_EER_{}epo.txt".format(epoch), printout=False)
        
        scheduler.step()

        if rank == 0:
            save_latest_resume_checkpoint(model_save_path, epoch, model)
            epoch_checkpoint_path = save_epoch_checkpoint(
                model_save_path, epoch, dev_eer, model
            )

            saved_best_checkpoint = False
            if math.isfinite(dev_eer) and dev_eer <= best_dev_eer:
                best_dev_eer = dev_eer
                saved_best_checkpoint = True

            save_training_state(model_save_path, epoch, optimizer, scheduler, best_dev_eer)

            save_logs(epoch, scheduler.get_last_lr(), model_tag, running_loss, dev_eer, valid_loss, running_ortho_loss)
            append_run_log(
                run_log_path,
                f"Epoch {epoch:03d}: train_loss={running_loss:.6f}, "
                f"orth_loss={float(running_ortho_loss):.6f}, valid_loss={valid_loss:.6f}, "
                f"dev_eer={dev_eer:.6f}, lr={scheduler.get_last_lr()[0]:.8f}, "
                f"checkpoint={epoch_checkpoint_path}"
            )
            if wandb_run is not None:
                epoch_step = (epoch + 1) * len(trn_loader)
                wandb_run.log(
                    {
                        "epoch": epoch,
                        "train/loss": running_loss,
                        "train/epoch_orth_loss": running_ortho_loss,
                        "valid/loss": valid_loss,
                        "valid/eer": dev_eer,
                        "valid/threshold": dev_th,
                        "optim/lr": scheduler.get_last_lr()[0],
                        "checkpoint/saved_best": int(saved_best_checkpoint),
                        "checkpoint/best_dev_eer": best_dev_eer,
                    },
                    step=epoch_step,
                )


    # Evaluation with the best model
    if rank == 0:
        delete_latest_resume_checkpoints(model_save_path)
        append_run_log(run_log_path, "Deleted latest resume checkpoint after completed training.")

        best_checkpoint_path = find_best_eer_checkpoint(model_tag, model_save_path)
        if best_checkpoint_path is None:
            best_checkpoint_path, _ = find_latest_epoch_checkpoint(model_save_path)
            append_run_log(
                run_log_path,
                "No finite dev EER checkpoint was found; using the latest epoch checkpoint."
            )
        if best_checkpoint_path is None:
            raise FileNotFoundError(f"No epoch checkpoints found in {model_save_path}.")
        append_run_log(run_log_path, f"Selected best dev-EER checkpoint: {best_checkpoint_path}")
        
        model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))

        eval_score_path = model_tag / 'eval_output.txt'
        eval_loss = produce_evaluation_file(eval_loader, model, device,
                                eval_score_path,
                                desc="Final eval",
                                disable=False)
        eval_eer, _ = compute_nist_eer(sc_file=eval_score_path,
                                    output_file=metric_path / "eval_best.txt")
        append_run_log(run_log_path, f"Final eval: eval_loss={eval_loss:.6f}, eval_eer={eval_eer:.6f}")
        append_run_log(run_log_path, f"Run finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if wandb_run is not None:
            final_step = config["num_epochs"] * len(trn_loader)
            wandb_run.log(
                {
                    "eval/loss": eval_loss,
                    "eval/eer": eval_eer,
                    "eval/checkpoint": str(best_checkpoint_path),
                },
                step=final_step,
            )
            for artifact_path in (
                run_log_path,
                model_tag / "loss_history.txt",
                model_tag / "Orthogonal_loss_history.txt",
                model_tag / "validation_eer_history.txt",
                model_tag / "learning_rate.txt",
                model_tag / "valid_loss.txt",
                metric_path / "eval_best.txt",
            ):
                if artifact_path.exists():
                    wandb.save(str(artifact_path), base_path=str(model_tag), policy="now")
            wandb_run.finish()


    cleanup()



def lora_orthogonality_loss(adapters):
    loss = 0
    for adapter in adapters:
        up_weight = adapter.lora_fc.lora_A
        down_weight = adapter.lora_fc.lora_B
        combined_matrix = torch.matmul(up_weight, down_weight)
        # Compute Gram matrix
        gram_matrix = torch.matmul(combined_matrix, combined_matrix.T)
        identity_matrix = torch.eye(gram_matrix.size(0), device=gram_matrix.device)
        # Penalize deviation from orthogonality
        loss += F.mse_loss(gram_matrix, identity_matrix)
    return loss


def checkpoint_epoch(checkpoint_path):
    match = re.match(r"epoch_(\d+)_.*\.pth$", checkpoint_path.name)
    if match is None:
        return None
    return int(match.group(1))


def latest_resume_checkpoint_epoch(checkpoint_path):
    match = re.match(r"latest_checkpoint_epoch_(\d+)\.pth$", checkpoint_path.name)
    if match is None:
        return None
    return int(match.group(1))


def find_latest_epoch_checkpoint(model_save_path):
    epoch_checkpoints = []
    for checkpoint_path in model_save_path.glob("epoch_*.pth"):
        epoch = checkpoint_epoch(checkpoint_path)
        if epoch is not None:
            epoch_checkpoints.append((epoch, checkpoint_path))

    if not epoch_checkpoints:
        return None, 0

    latest_epoch, latest_checkpoint_path = max(epoch_checkpoints, key=lambda item: item[0])
    return latest_checkpoint_path, latest_epoch + 1


def find_latest_resume_checkpoint(model_save_path):
    resume_checkpoints = []
    for checkpoint_path in model_save_path.glob("latest_checkpoint_epoch_*.pth"):
        epoch = latest_resume_checkpoint_epoch(checkpoint_path)
        if epoch is not None:
            resume_checkpoints.append((epoch, checkpoint_path))

    if not resume_checkpoints:
        return None, 0

    latest_epoch, latest_checkpoint_path = max(resume_checkpoints, key=lambda item: item[0])
    return latest_checkpoint_path, latest_epoch + 1


def get_training_state_path(model_save_path, epoch):
    return model_save_path / f"training_state_epoch_{epoch}.pth"


def delete_latest_resume_checkpoints(model_save_path):
    for checkpoint_path in model_save_path.glob("latest_checkpoint_epoch_*.pth"):
        checkpoint_path.unlink()


def save_latest_resume_checkpoint(model_save_path, epoch, model):
    delete_latest_resume_checkpoints(model_save_path)
    torch.save(model.state_dict(), model_save_path / f"latest_checkpoint_epoch_{epoch}.pth")


def save_epoch_checkpoint(model_save_path, epoch, dev_eer, model):
    """Save model weights for every epoch, independently of metric improvement."""
    for checkpoint_path in model_save_path.glob(f"epoch_{epoch}_*.pth"):
        checkpoint_path.unlink()

    eer_label = f"{dev_eer:.3f}" if math.isfinite(dev_eer) else "nan"
    checkpoint_path = model_save_path / f"epoch_{epoch}_{eer_label}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    return checkpoint_path


def save_training_state(model_save_path, epoch, optimizer, scheduler, best_dev_eer):
    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "best_dev_eer": best_dev_eer,
        },
        get_training_state_path(model_save_path, epoch),
    )


def find_best_eer_checkpoint(model_tag, model_save_path):
    eer_history_path = model_tag / "validation_eer_history.txt"
    if not eer_history_path.exists():
        return None

    epoch_eers = []
    with open(eer_history_path, "r") as file:
        for line in file:
            if not line.startswith("Epoch"):
                continue
            try:
                epoch_text, eer_text = line.split(":", 1)
                epoch = int(epoch_text.split()[1]) - 1
                dev_eer = float(eer_text.strip())
            except (IndexError, ValueError):
                continue
            if math.isfinite(dev_eer):
                epoch_eers.append((dev_eer, epoch))

    for _, epoch in sorted(epoch_eers):
        checkpoints = sorted(model_save_path.glob(f"epoch_{epoch}_*.pth"))
        if checkpoints:
            return checkpoints[0]
    return None


def read_best_dev_eer(model_tag):
    txt_path = model_tag/'validation_eer_history.txt'
    best_dev_eer = float("inf")
    if not txt_path.exists():
        return best_dev_eer

    with open(txt_path, 'r') as file:
        for line in file:
            if line.startswith('Epoch'):
                try:
                    dev_eer = float(line.split(':', 1)[1].strip())
                except (IndexError, ValueError):
                    continue
                if math.isfinite(dev_eer):
                    best_dev_eer = min(best_dev_eer, dev_eer)
    return best_dev_eer



def save_logs(epoch, current_lr,model_tag,running_loss,dev_eer,valid_loss,running_ortho_loss):

    print("Finished epoch{:03d}".format(epoch))

    with open(model_tag/'loss_history.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {running_loss}\n')
    with open(model_tag/'Orthogonal_loss_history.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {running_ortho_loss}\n')        
    with open(model_tag/'validation_eer_history.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {dev_eer}\n')
    # with open(model_tag/'evaluation_eer_history.txt', 'a') as file:
    #     file.write(f'Epoch {epoch + 1}: {eval_eer}\n')
    with open(model_tag/'learning_rate.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {current_lr[0]}\n')
    with open(model_tag/'valid_loss.txt', 'a') as file:
        file.write(f'Epoch {epoch + 1}: {valid_loss}\n')
    # with open(model_tag/'eval_loss.txt', 'a') as file:
    #     file.write(f'Epoch {epoch + 1}: {eval_loss}\n')  




def get_DDP_loader(
        args,
        feat_file: str,
        trn_list_path: str,
        dev_trial_path: str,
        eval_trial_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    if os.path.exists(trn_list_path):
        trn_keys, trn_labs, trn_paths = gen_cyber_list(meta_file=trn_list_path,
                                                       feat_file=feat_file)
        print("no. training files:", len(trn_keys))

        train_set = CyberDataset(list_ids=trn_keys,
                                 labels=trn_labs,
                                 file_paths=trn_paths)
        # gen = torch.Generator()
        train_sampler = DistributedSampler(train_set)
        trn_loader = DataLoader(train_set,
                                batch_size=config["batch_size"],
                                drop_last=True,
                                pin_memory=True,
                                worker_init_fn=seed_worker,
                                num_workers=16, sampler=train_sampler)
    else:
        print('[WARNING] no training file list, it is possible only for evaluation case.')
        trn_loader = None

    if os.path.exists(dev_trial_path):
        dev_keys, dev_labs, dev_paths = gen_cyber_list(meta_file=dev_trial_path,
                                                       feat_file=feat_file)
        print("no. validation files:", len(dev_keys))

        dev_set = CyberEvalDataset(list_ids=dev_keys,
                                   labels=dev_labs,
                                   file_paths=dev_paths)
        # dev_sampler = DistributedSampler(dev_set)
        dev_loader = DataLoader(dev_set,
                                batch_size=config["batch_size"],
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,num_workers=16)
    else:
        print('[WARNING] no dev file list, it is possible only for evaluation case.')
        dev_loader = None

    eval_keys, eval_labs, eval_paths = gen_cyber_list(meta_file=eval_trial_path,
                                                      feat_file=feat_file)
    print("no. evaluation files:", len(eval_keys))
    eval_set = CyberEvalDataset(list_ids=eval_keys,
                                labels=eval_labs,
                                file_paths=eval_paths)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True,num_workers=16)

    return trn_loader, dev_loader, eval_loader, train_sampler

def produce_evaluation_file(
        data_loader: DataLoader,
        model,
        device: torch.device,
        save_path: str,
        desc: str = "Evaluation",
        disable: bool = False) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    lab_list = []

        # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    valid_loss = 0.0
    num_total = 0.0

    eval_iter = tqdm(data_loader, desc=desc, disable=disable, dynamic_ncols=True, leave=True)
    for i, (batch_x, batch_y, utt_id) in enumerate(eval_iter):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        batch_size = batch_x.size(0)
        num_total += batch_size
        with torch.inference_mode():
            # _, batch_out = model(batch_x) # for AASIST
            batch_out = model(batch_x) 
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel() # 1 - detect bona, 0 - detect spoof

            batch_loss = criterion(batch_out, batch_y)
            valid_loss = valid_loss + batch_loss.item()*batch_size
            eval_iter.set_postfix(loss=f"{batch_loss.item():.4f}")
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        lab_list.extend(batch_y)
        #print(i, utt_id)

    with open(save_path, "w") as fh:
        for fn, lab, sco in zip(fname_list, lab_list, score_list):
            lab = "bonafide" if lab == 1 else "spoof"
            fh.write(f"{fn}\t{lab}\t{sco}\n")
    # print("Scores saved to {}".format(save_path))

    valid_loss /= num_total

    return valid_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audio deepfake detection system")


    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )  
    parser.add_argument(
        "--meta_dir",
        dest="meta_dir",
        type=str,
        help="processed meta files following cyber_cookies format",
        default="./data/meta/",
    ) 
    parser.add_argument(
        "--fold",
        dest="fold",
        type=int,
        help="fold number",
        default=1,
    )
    parser.add_argument(
        "--feat_file",
        dest="feat_file",
        type=str,
        help="file with all features, follows cyber_cookies format (wav.scp)",
        default="./data/meta/wav.scp",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument("--SSL_num",
                        type=int,
                        default=12,
                        help="number of the layers in SSL model")    
    parser.add_argument("--pretrain_checkpoint",
                        type=str,
                        default=None,
                        help="the checkpoint path")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_path",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument(
        "--num_gpu",
        action="store_true",
        help="when this flag is given, continue train the model from pre-trained checkpoint")
    parser.add_argument("--exp_idx",
                        type=int,
                        default=0,
                        help="index of running experiment")    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the latest epoch checkpoint in the output directory")
    

    run_train(parser.parse_args())
 
