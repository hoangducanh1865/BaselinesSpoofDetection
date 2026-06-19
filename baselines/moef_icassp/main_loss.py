import os,yaml,shutil,json
from datetime import datetime
from pathlib import Path
from utils.arg_parse import f_args_parsed,set_random_seed
args = f_args_parsed()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid
import lightning as L
import importlib
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint,LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
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
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        return None
    if not os.environ.get("WANDB_API_KEY"):
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
    if args.wandb_entity:
        kwargs["entity"] = args.wandb_entity
    logger = pl_loggers.WandbLogger(**kwargs)
    logger.experiment.config.update(vars(args), allow_val_change=True)
    print(f"[wandb] Logging to project={args.wandb_project}, run={run_name}, id={run_id}")
    return logger


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
        
        # model initialization
        trainer = L.Trainer(
            max_epochs=args.epochs,
            strategy='ddp_find_unused_parameters_true',
            log_every_n_steps = 1,
            callbacks=[
                # dev损失无下降就提前停止
                EarlyStopping('loss',patience=args.no_best_epochs,mode="min",verbose=True,log_rank_zero_only=True),
                # 模型按照最低val_loss来保存
                ModelCheckpoint(monitor='loss',
                                dirpath=str(checkpoint_dir),
                                save_top_k=1,
                                save_weights_only=True,mode="min",filename='best_model-{epoch:02d}-{dev_eer:.4f}-{loss:.4f}'),
                LatestResumeCheckpoint(checkpoint_dir),
                LearningRateMonitor(logging_interval='epoch',log_weight_decay=True),
                ],
            check_val_every_n_epoch=1,
            logger=loggers,
            enable_progress_bar=False
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
    
