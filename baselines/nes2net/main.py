"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import glob
import os
import sys
import random
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils import (TrainDataset,TestDataset, genSpoof_list)
from eval.calculate_metrics import calculate_minDCF_EER_CLLR, calculate_aDCF_tdcf_tEER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)
from tqdm import tqdm


def pad_audio(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def find_audio_files(root_dir):
    audio_files = {}
    root = Path(root_dir)
    for flac_folder in sorted(glob.glob(str(root / "flac_E_*"))):
        eval_dir = Path(flac_folder) / "flac_E_eval"
        if not eval_dir.exists():
            continue
        for audio_file in sorted(eval_dir.glob("*.flac")):
            audio_files[audio_file.stem] = str(audio_file)
    return audio_files


def load_protocol(protocol_path, return_entries=False):
    labels = {}
    entries = []
    if not Path(protocol_path).exists():
        return entries if return_entries else labels
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                labels[parts[1]] = parts[8]
                entries.append((parts[0], parts[1], parts[8]))
    return entries if return_entries else labels


def run_single_inference(model, audio_path, device, audio_len):
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio = pad_audio(audio, audio_len)
    audio = np.asarray(audio, dtype=np.float32)
    x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        batch_out = model(x)
        return batch_out[:, 1].item()


def eval_pretrained_only(args, config, model_config, model_tag, eval_score_path, device):
    audio_len = model_config.get("nb_samp", 64600)
    eval_root = Path(config.get("database_path", "."))
    if not eval_root.exists():
        eval_root = Path.cwd()

    audio_files = find_audio_files(eval_root)
    if not audio_files:
        audio_files = find_audio_files(Path.cwd())
    if not audio_files:
        raise FileNotFoundError("No flac_E_* / flac_E_eval/*.flac files were found.")

    protocol_candidates = [
        eval_root / "ASVspoof5_protocols" / "ASVspoof5.eval.track_1.tsv",
        Path.cwd() / "ASVspoof5_protocols" / "ASVspoof5.eval.track_1.tsv",
    ]
    protocol_path = next((p for p in protocol_candidates if p.exists()), None)
    if protocol_path is None:
        raise FileNotFoundError("ASVspoof5.eval.track_1.tsv was not found.")

    protocol_entries = load_protocol(protocol_path, return_entries=True)
    labeled_eval = [
        (spk_id, utt_id, audio_files[utt_id], label)
        for spk_id, utt_id, label in protocol_entries
        if utt_id in audio_files
    ]
    if not labeled_eval:
        raise FileNotFoundError("No eval audio files matched ASVspoof5.eval.track_1.tsv.")

    missing_count = len(protocol_entries) - len(labeled_eval)
    if missing_count:
        print(f"WARNING: {missing_count}/{len(protocol_entries)} eval protocol utterances were not found locally.")

    total_eval_files = len(labeled_eval)
    if args.eval_subset_size is not None and args.eval_subset_size > 0:
        subset_size = min(args.eval_subset_size, total_eval_files)
        rng = random.Random(args.eval_subset_seed)
        labeled_eval = rng.sample(labeled_eval, subset_size)
        print(f"Using random subset of {subset_size}/{total_eval_files} protocol-matched eval files.")

    model = get_model(model_config, device)
    if args.eval_model_weights:
        eval_weight_path = Path(args.eval_model_weights)
        if eval_weight_path.is_dir():
            eval_weight_path = eval_weight_path / "best_model.pth"
    else:
        eval_weight_path = model_tag / "weights" / "best_model.pth"

    model.load_state_dict(torch.load(eval_weight_path, map_location=device), strict=True)
    print("Model loaded : {}".format(eval_weight_path))
    model.eval()
    print("Start evaluation...")

    text_list = []
    for spk_id, utt_id, audio_path, key in tqdm(labeled_eval, desc="Scoring"):
        score = run_single_inference(model, audio_path, device, audio_len)
        text_list.append(f"{spk_id} {utt_id} {score} {key}")

    with open(eval_score_path, "w") as fh:
        fh.write("\n".join(text_list) + "\n")
    print("Scores saved to {}".format(eval_score_path))

    eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
        cm_scores_file=eval_score_path,
        output_file=model_tag / "loaded_model_result.txt")
    print("DONE. eval_eer: {:.3f}, eval_dcf:{:.5f} , eval_cllr:{:.5f}".format(eval_eer, eval_dcf, eval_cllr))
    return

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    config_path = Path(args.config)
    if not config_path.exists():
        alt_config_path = Path("config") / args.config
        if alt_config_path.exists():
            config_path = alt_config_path

    # load experiment configurations
    with open(config_path, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    if "freq_aug" not in config:
        config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof5.dev.track_1.tsv")
    # define model related paths
    model_tag = config["model_tag"]
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(str(config_path), model_tag / "config.conf")

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    if args.train is False:
        eval_pretrained_only(args, config, model_config, model_tag, eval_score_path, device)
        sys.exit(0)

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, dev_loader, eval_loader = get_loader(
        database_path, args.seed, config)


    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 100.
    best_dev_dcf = 1.
    best_dev_cllr = 1.
    no_improve = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        if args.train is False:
            break
        print("training epoch{:03d}".format(epoch))
        
        running_loss = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        if epoch < args.start_val_epoch:
            print("DONE.\nLoss:{:.5f}. Skip validation step".format(running_loss))
            continue
        
        produce_evaluation_file(dev_loader, model, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer, dev_dcf, dev_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=metric_path/"dev_score.txt",
            output_file=metric_path/"dev_DCF_EER_{}epo.txt".format(epoch),
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}, dev_dcf:{:.5f} , dev_cllr:{:.5f}".format(
            running_loss, dev_eer, dev_dcf, dev_cllr))
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
        writer.add_scalar("dev_dcf", dev_dcf, epoch)
        writer.add_scalar("dev_cllr", dev_cllr, epoch)

        best_dev_dcf = min(dev_dcf, best_dev_dcf)
        best_dev_cllr = min(dev_cllr, best_dev_cllr)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                model_save_path / "epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))
            if os.path.islink(os.path.join(model_save_path, 'best_model.pth')):
                os.unlink(os.path.join(model_save_path, 'best_model.pth'))
            os.symlink("epoch_{}_{:03.3f}.pth".format(epoch, dev_eer),
                    os.path.join(model_save_path, 'best_model.pth'))
            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            no_improve = 0
        else:
            no_improve += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_dcf, epoch)
        writer.add_scalar("best_dev_cllr", best_dev_cllr, epoch)
        if no_improve >= config["early_stop_epochs"]:
            break
    
    # evaluates pretrained model 
    # NOTE: Currently it is evaluated on the development set instead of the evaluation set
    if args.eval:
        eval_trial_path = (database_path / "ASVspoof5.eval.track_1.tsv")
        if args.eval_model_weights:
            eval_weight_path = Path(args.eval_model_weights)
            if eval_weight_path.is_dir():
                eval_weight_path = eval_weight_path / "best_model.pth"
        else:
            eval_weight_path = model_save_path / 'best_model.pth'

        model.load_state_dict(
            torch.load(eval_weight_path, map_location=device))
        print("Model loaded : {}".format(eval_weight_path))
        print("Start evaluation...")
        produce_evaluation_file(eval_loader, model, device,
                                eval_score_path, eval_trial_path)

        eval_dcf, eval_eer, eval_cllr = calculate_minDCF_EER_CLLR(
            cm_scores_file=eval_score_path,
            output_file=model_tag/"loaded_model_result.txt")
        print("DONE. eval_eer: {:.3f}, eval_dcf:{:.5f} , eval_cllr:{:.5f}".format(eval_eer, eval_dcf, eval_cllr))
        sys.exit(0)

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config, device=device).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement"""

    trn_database_path = database_path / "flac_T/"
    dev_database_path = database_path / "flac_D/"
    eval_database_path = database_path / "eval_full/flac_E_eval/"

    trn_list_path = (database_path /
                     "ASVspoof5.train.tsv")
    dev_trial_path = (database_path /
                      "ASVspoof5.dev.track_1.tsv")
    eval_trial_path = (database_path /
                      "ASVspoof5.eval.track_1.tsv")
    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path)
    print("no. training files:", len(file_train))

    train_set = TrainDataset(list_IDs=file_train,
                                           labels=d_label_trn,
                                           base_dir=trn_database_path,
                                           add_noise=str_to_bool(config["add_noise"]))
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    _, file_dev = genSpoof_list(dir_meta=dev_trial_path)
    print("no. dev files:", len(file_dev))

    dev_set = TestDataset(list_IDs=file_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    _, file_eval = genSpoof_list(dir_meta=eval_trial_path)
    print("no. validation files:", len(file_eval))

    eval_set = TestDataset(list_IDs=file_eval,
                                            base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    return trn_loader, dev_loader, eval_loader

def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    #assert len(trial_lines) == len(fname_list) == len(score_list)
    text_list = []
    for fn, sco, trl in zip(fname_list, score_list, trial_lines):
        spk_id, utt_id, _, _, _, _, _, src, key, _ = trl.strip().split(' ')
        assert fn == utt_id
        text_list.append("{} {} {} {}".format(spk_id, utt_id, sco, key))
    
    with open(save_path, "w") as fh:
        fh.write("\n".join(text_list) + '\n')
    del text_list
    fh.close()
    print("Scores saved to {}".format(save_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
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
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument("--start_val_epoch",
                        type=int,
                        default=10)
    parser.add_argument(
        "--train",
        action="store_true",
        default=False)
    parser.add_argument(
        "--eval",
        default=True)
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="path to the model weight file for evaluation")
    parser.add_argument("--eval_subset_size",
                        type=int,
                        default=10000,
                        help="number of random eval utterances to score in eval-only mode; use 0 or None-like behavior by passing a large value if you want full eval")
    parser.add_argument("--eval_subset_seed",
                        type=int,
                        default=1234,
                        help="random seed used when sampling eval utterances")
    main(parser.parse_args())
