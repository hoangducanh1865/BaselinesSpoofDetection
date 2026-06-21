import os,yaml,shutil,json,time
from datetime import datetime
from pathlib import Path
from utils.arg_parse import f_args_parsed,set_random_seed
args = f_args_parsed()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
import importlib
import torch

try:
    import lightning as L
    from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch import loggers as pl_loggers
except Exception:
    import pytorch_lightning as L
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning import loggers as pl_loggers
# arguments initialization


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _resolve_run_dir(savedir):
    path = Path(savedir).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_run_config(run_dir, args):
    config = vars(args).copy()
    config["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(run_dir / "hyperparameters.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=True)
    with open(run_dir / "hyperparameters.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def _wandb_logger(run_dir, args):
    wandb_mode = os.environ.get("WANDB_MODE", "online").lower()
    if wandb_mode == "disabled":
        return None
    if not os.environ.get("WANDB_API_KEY") and wandb_mode != "offline":
        print("[wandb] WANDB_API_KEY is not set; WandB logging disabled.")
        return None
    try:
        import wandb
    except ImportError:
        print("[wandb] wandb package is not installed; WandB logging disabled.")
        return None

    run_id_path = run_dir / "wandb_run_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text().strip()
    else:
        run_id = wandb.util.generate_id()
        run_id_path.write_text(run_id + "\n")

    run_name = args.wandb_name or f"moef-{run_dir.name}"
    (run_dir / "wandb").mkdir(parents=True, exist_ok=True)
    kwargs = {
        "project": args.wandb_project,
        "name": run_name,
        "id": run_id,
        "resume": "allow",
        "save_dir": str(run_dir / "wandb"),
        "log_model": False,
    }
    if wandb_mode == "offline":
        kwargs["offline"] = True
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    logger = pl_loggers.WandbLogger(**kwargs)
    logger.experiment.config.update(vars(args), allow_val_change=True)
    print(f"[wandb] Logging to project={args.wandb_project}, run={run_name}, id={run_id}")
    return logger


def _metric_value(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _format_metrics(metrics):
    parts = []
    for key in ("loss", "loss_epoch", "dev_eer", "dev_tdcf", "lr-AdamW", "lr-Adam"):
        if key not in metrics:
            continue
        value = _metric_value(metrics[key])
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else "no tracked metrics yet"


def _lightning_major_version():
    try:
        return int(str(L.__version__).split(".")[0])
    except Exception:
        return 2


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _trainer_device_kwargs(args):
    if args.gpuid.strip() and torch.cuda.is_available():
        device_count = max(torch.cuda.device_count(), 1)
        kwargs = {
            "accelerator": "gpu",
            "devices": device_count,
        }
        if device_count > 1:
            kwargs["strategy"] = "ddp_find_unused_parameters_true" if _lightning_major_version() >= 2 else "ddp"
        return kwargs
    return {}


def _latest_checkpoint_epoch(path):
    stem = Path(path).stem
    try:
        return int(stem.split("latest_checkpoint_epoch_")[1])
    except Exception:
        return -1


def _find_latest_resume_checkpoint(run_dir):
    candidates = list(run_dir.rglob("latest_checkpoint_epoch_*.ckpt"))
    if not candidates:
        return None
    return str(sorted(candidates, key=_latest_checkpoint_epoch)[-1])


def _resolve_resume_checkpoint(run_dir, resume_arg):
    if not resume_arg:
        return None
    resume_path = Path(resume_arg).expanduser()
    if resume_path.is_file():
        return str(resume_path)
    if resume_path.is_dir():
        checkpoint = _find_latest_resume_checkpoint(resume_path)
        if checkpoint:
            return checkpoint
        raise FileNotFoundError(f"No latest_checkpoint_epoch_*.ckpt found in {resume_path}")
    checkpoint = _find_latest_resume_checkpoint(run_dir)
    if checkpoint:
        return checkpoint
    raise FileNotFoundError(f"No latest checkpoint found for resume in {run_dir}")


class LatestResumeCheckpoint(Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.latest_paths = []

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        for old_path in self.latest_paths:
            if old_path.exists():
                old_path.unlink()
        epoch = int(trainer.current_epoch)
        checkpoint_path = self.checkpoint_dir / f"latest_checkpoint_epoch_{epoch}.ckpt"
        trainer.save_checkpoint(str(checkpoint_path), weights_only=False)
        self.latest_paths = [checkpoint_path]
        print(f"[checkpoint] Latest resume checkpoint saved: {checkpoint_path}")

    def on_fit_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        if not getattr(trainer, "interrupted", False):
            for path in self.checkpoint_dir.glob("latest_checkpoint_epoch_*.ckpt"):
                path.unlink()
            print("[checkpoint] Removed latest resume checkpoint after completed training.")


class EpochProgressLogger(Callback):
    def __init__(self, interval):
        self.interval = max(int(interval), 0)
        self.epoch_started_at = None

    def on_fit_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            print(
                f"[progress] Training for max_epochs={trainer.max_epochs}; "
                f"checkpoints={checkpoint_dir}; log_every_n_steps={trainer.log_every_n_steps}",
                flush=True,
            )

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_started_at = time.time()
        if trainer.is_global_zero:
            print(f"[epoch {trainer.current_epoch:03d}] train start", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.interval <= 0 or not trainer.is_global_zero:
            return
        step = batch_idx + 1
        if step % self.interval != 0:
            return
        total = trainer.num_training_batches
        print(
            f"[epoch {trainer.current_epoch:03d}] train batch {step}/{total}: "
            f"{_format_metrics(trainer.callback_metrics)}",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return
        elapsed = time.time() - self.epoch_started_at if self.epoch_started_at else 0.0
        print(
            f"[epoch {trainer.current_epoch:03d}] train end after {elapsed:.1f}s: "
            f"{_format_metrics(trainer.callback_metrics)}",
            flush=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or not trainer.is_global_zero:
            return
        print(
            f"[epoch {trainer.current_epoch:03d}] validation end: "
            f"{_format_metrics(trainer.callback_metrics)}",
            flush=True,
        )


run_dir = _resolve_run_dir(args.savedir)
args.savedir = str(run_dir)
try:
    from dotenv import load_dotenv
    load_dotenv(_repo_root() / ".env")
except ImportError:
    pass
_write_run_config(run_dir, args)
checkpoint_dir = run_dir / "checkpoints"
resume_ckpt = _resolve_resume_checkpoint(run_dir, args.resume)
if resume_ckpt:
    print(f"[resume] Resuming MoEF training from: {resume_ckpt}")
else:
    print("[resume] Resume disabled; training from scratch.")

### temporal config
# 
# args.stage = 1
# 
# ###


# config gpu

# random seed initialization and gpu seed 
set_random_seed(args.seed, args)

# config the base model containing train eval test and inference funtion
tl_model = importlib.import_module(args.tl_model)

# config the data module containing the train set, dev set and test set
dm_module = importlib.import_module(args.data_module)
asvspoof_dm = dm_module.asvspoof_dataModule(args=args)

if True:
    # ⭐train 
    if not args.inference:
        # import model.py
        prj_model = importlib.import_module(args.module_model)
        
        # model 
        model = prj_model.Model(args)

        # init model, including loss func and optim 
        customed_model_wrapper = tl_model.base_model(
            model=model,
            args=args
            )

        tb_logger = pl_loggers.TensorBoardLogger(str(run_dir), name="tensorboard")
        loggers = [tb_logger]
        wb_logger = _wandb_logger(run_dir, args)
        if wb_logger is not None:
            loggers.append(wb_logger)
        
        callbacks = [
                EpochProgressLogger(args.progress_log_interval),
                # dev loss does not decrease => early stop
                EarlyStopping('loss',patience=args.no_best_epochs,mode="min",verbose=True,log_rank_zero_only=True),
                # Save best model by lowest training loss, while dev_eer stays in filename if available.
                ModelCheckpoint(monitor='loss',
                                dirpath=str(checkpoint_dir),
                                save_top_k=1,
                                save_weights_only=True,mode="min",filename='best_model-{epoch:02d}-{dev_eer:.4f}-{loss:.4f}'),
                LatestResumeCheckpoint(checkpoint_dir),
                LearningRateMonitor(logging_interval='epoch'),
                ]

        trainer_kwargs = {
            "max_epochs": args.epochs,
            "log_every_n_steps": 1,
            "callbacks": callbacks,
            "check_val_every_n_epoch": 1,
            "logger": loggers,
            "enable_progress_bar": not args.disable_progress_bar,
        }
        trainer_kwargs.update(_trainer_device_kwargs(args))

        # model initialization
        trainer = L.Trainer(
            **trainer_kwargs
            )
        trainer.fit(
            model=customed_model_wrapper, 
            datamodule=asvspoof_dm,
            ckpt_path=resume_ckpt,
            )
        
        # # test 19 default
        # trainer.test(
        #     model=customed_model_wrapper,
        #     datamodule=asvspoof_dm
        #     )
    
