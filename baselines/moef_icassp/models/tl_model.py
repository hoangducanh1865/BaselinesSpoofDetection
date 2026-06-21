from typing import Any
try:
    import lightning as L
except Exception:
    import pytorch_lightning as L
import torch
import logging,os
import numpy as np
from pathlib import Path
from utils.wrapper import loss_wrapper, optim_wrapper,schedule_wrapper   
from utils.tools import cul_eer 


def _simple_eer(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    bona = labels == 1
    spoof = labels == 0
    if not bona.any() or not spoof.any():
        return 0.0
    order = np.argsort(scores)
    sorted_labels = labels[order]
    n_bona = bona.sum()
    n_spoof = spoof.sum()
    false_reject = np.cumsum(sorted_labels == 1) / n_bona
    false_accept = (n_spoof - np.cumsum(sorted_labels == 0)) / n_spoof
    idx = np.argmin(np.abs(false_reject - false_accept))
    return float((false_reject[idx] + false_accept[idx]) / 2.0 * 100.0)


def _asvspoof5_dev_eer(score_path):
    root = Path(
        os.environ.get("MOEF_ASVSPOOF5_ROOT", "/home/user14/anhhd/spoof/datasets/asvspoof5")
    ).expanduser()
    candidates = [
        root / "protocols" / "ASVspoof5.dev.track_1.tsv",
        root / "ASVspoof5_protocols" / "ASVspoof5.dev.track_1.tsv",
        root / "ASVspoof5.dev.track_1.tsv",
    ]
    protocol = next((path for path in candidates if path.exists()), None)
    if protocol is None:
        return 0.0
    labels = {}
    with open(protocol, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                labels[parts[1]] = 1 if parts[8] == "bonafide" else 0
    y_true = []
    y_score = []
    with open(score_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] in labels:
                y_true.append(labels[parts[0]])
                y_score.append(float(parts[1]))
    return _simple_eer(y_true, y_score) if y_true else 0.0



class base_model(L.LightningModule):
    def __init__(self, 
                 model,
                 args,
                 ) -> None:
        super().__init__()
        self.args = args
        self.model = model

        self.save_hyperparameters(self.args)
        
        self.model_optimizer = optim_wrapper.optimizer_wrap(self.args, self.model).get_optim()
        self.LRScheduler = schedule_wrapper.scheduler_wrap(self.model_optimizer,self.args).get_scheduler()
        # for loss
        self.args.model = model
        self.args.samloss_optim = self.model_optimizer
        self.loss_criterion,self.loss_optimizer,self.minimizor = loss_wrapper.loss_wrap(self.args).get_loss()
        
        
        self.logging_test = None
        self.logging_predict = None

    def _log_dir(self):
        logger = self.logger
        log_dir = None
        if hasattr(logger, "log_dir"):
            log_dir = logger.log_dir
        elif hasattr(logger, "__iter__"):
            for sub_logger in logger:
                if hasattr(sub_logger, "log_dir"):
                    log_dir = sub_logger.log_dir
                    break
        if log_dir is None:
            log_dir = self.trainer.log_dir or self.args.savedir
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        return str(log_dir)
        
    def forward(self,x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        
        # batch[0] -- tensor
        # batch[1] -- label
        # batch[2] -- filename
        
                
        # model output, better return 2 elements, prediction and any other thing
        output = self.forward(batch[0])
        batch_loss = self.loss_criterion(output[0], batch[1])
        
        batch_loss = batch_loss.mean()
        self.log_dict({
            "loss": batch_loss,
            },on_step=True, 
                on_epoch=True,prog_bar=True, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        return batch_loss
        
    def validation_step(self, batch, batch_idx):
        # batch[0] -- tensor
        # batch[1] -- label
        # batch[2] -- filename
        
        # model output
        output = self.forward(batch[0])
        
        softmax_pred = torch.nn.functional.softmax(output[0],dim=1)
        
        # log the prediction for cul eer
        with open(os.path.join(self._log_dir(),"dev.log"), 'a') as file:
            for i in range(len(softmax_pred)):
                file.write(f"{batch[2][i]} {str(softmax_pred.cpu().numpy()[i][1])}\n")
        
        # batch_loss = self.loss_criterion(data_predict, data_label).mean()
        # # Logging to TensorBoard (if installed) by default
        # self.log("val_loss", batch_loss, batch_size=len(data_in),sync_dist=True)
    
    def on_validation_epoch_end(self) -> None:
        # culculate the dev eer
        dev_eer = 0.
        dev_tdcf = 0.
        dev_log_path = os.path.join(self._log_dir(),"dev.log")
        with open(dev_log_path, 'r') as file:
            lines = file.readlines()

        if getattr(self.args, "dataset", "asvspoof2019la") == "asvspoof5":
            dev_eer = _asvspoof5_dev_eer(dev_log_path)
            with open(dev_log_path, 'w') as file:
                pass
            self.log_dict({
                "dev_eer": dev_eer,
                "dev_tdcf": dev_tdcf,
                },on_step=False,
                    on_epoch=True,prog_bar=False, logger=True,
                    sync_dist=True,
                    )
            return

        asv2019_root = Path(
            os.environ.get(
                "MOEF_ASVSPOOF2019_LA_ROOT",
                "/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA",
            )
        ).expanduser()
        dev_protocol = Path(
            os.environ.get(
                "MOEF_ASVSPOOF2019_DEV_PROTOCOL",
                asv2019_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt",
            )
        ).expanduser()
        dev_asv_scores = Path(
            os.environ.get(
                "MOEF_ASVSPOOF2019_DEV_ASV_SCORES",
                asv2019_root / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt",
            )
        ).expanduser()

        if len(lines) > 10000 and dev_protocol.exists() and dev_asv_scores.exists():
            dev_eer, dev_tdcf = cul_eer.eerandtdcf(
                dev_log_path,
                str(dev_protocol),
                str(dev_asv_scores),
            )
        with open(dev_log_path, 'w') as file:
            pass
        self.log_dict({
            "dev_eer": (dev_eer),
            "dev_tdcf": dev_tdcf,
            },on_step=False, 
                on_epoch=True,prog_bar=False, logger=True,
                # prevent from saving wrong ckp based on the eval_loss from different gpus
                sync_dist=True, 
                )
        
    def on_test_start(self):
        # logging.basicConfig(filename=os.path.join(self.logger.log_dir,f"infer_test.log"),level=logging.INFO,format="")
        self.logging_test = logging.getLogger("logging_test")
        self.logging_test.setLevel(logging.INFO)
        hdl=logging.FileHandler(os.path.join(self._log_dir(),f"infer_19.log"))
        hdl.setFormatter("")
        self.logging_test.addHandler(hdl)        
        
    def test_step(self, batch,) -> Any:
        # batch[0] -- tensor
        # batch[1] -- filename
        
        # model output
        output = self.forward(batch[0])
        
        data_predict = torch.nn.functional.softmax(output[0],dim=1)
        
        for i in range(len(batch[1])):
            self.logging_test.info(f"{batch[1][i]} {str(data_predict.cpu().numpy()[i][0])} {str(data_predict.cpu().numpy()[i][1])}")
        # return data_info[0],data_predict.cpu().numpy()
        return {'loss': 0, 'y_pred': data_predict}
    
    def on_predict_start(self):
        # logging.basicConfig(filename=os.path.join(self.args.savedir,f"infer_predict.log"),level=logging.INFO,format="")
        self.logging_predict = logging.getLogger(f"logging_predict_{self.args.testset}")
        self.logging_predict.setLevel(logging.INFO)
        hdlx = logging.FileHandler(os.path.join(self._log_dir(),f"infer_{self.args.testset}.log"))
        hdlx.setFormatter("")
        self.logging_predict.addHandler(hdlx)
    
    def predict_step(self, batch, batch_idx):
        # batch[0] -- tensor
        # batch[1] -- filename
        
        # model output
        output = self.forward(batch[0])
        
        data_predict = torch.nn.functional.softmax(output[0],dim=1)
         
        # self.logging_predict.info(f"{data_info[0]} {str(data_predict.cpu().numpy()[0][1])} {str(data_predict.cpu().numpy()[0][0])}")
        for i in range(len(batch[1])):
            self.logging_predict.info(f"{batch[1][i]} {str(data_predict.cpu().numpy()[i][1])}")
        # return data_info[0],data_predict.cpu().numpy()
        return 

    def configure_optimizers(self):
        configure = None
        if self.LRScheduler is not None:
            configure = {
                "optimizer":self.model_optimizer,
                'lr_scheduler': self.LRScheduler, 
                'monitor': 'dev_eer'
                }
        else:
            configure = {
                "optimizer":self.model_optimizer,
                }
            
        return configure
