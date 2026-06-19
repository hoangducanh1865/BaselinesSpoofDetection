import numpy as np
import soundfile as sf
import torch,os
from pathlib import Path
from torch import Tensor
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from .RawBoost import process_Rawboost_feature
import lightning as L


DEFAULT_ASV2019_LA_ROOT = "/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA"
DEFAULT_ASV5_ROOT = "/home/user14/anhhd/spoof/datasets/asvspoof5"
DEFAULT_ITW_ROOT = "/home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild"


def _path_with_slash(path):
        return str(Path(path).expanduser()) + "/"


def _first_existing(candidates, description):
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    raise FileNotFoundError(
        f"{description} not found. Checked: "
        + ", ".join(str(Path(c).expanduser()) for c in candidates)
    )


def _parse_asvspoof5_protocol(protocol_path, need_label=True):
    entries = []
    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 9:
                raise ValueError(f"Unexpected ASVspoof5 protocol line: {line.strip()}")
            key = parts[1]
            label = parts[8] if need_label else None
            entries.append((key, label))
    return entries


def _find_audio_path(audio_roots, key):
    for root in audio_roots:
        root = Path(root).expanduser()
        for candidate in (
            root / f"{key}.flac",
            root / "flac" / f"{key}.flac",
            root / "flac_T" / f"{key}.flac",
            root / "flac_D" / f"{key}.flac",
        ):
            if candidate.exists():
                return candidate
    return None


def _asvspoof5_items(protocol_path, audio_roots, need_label=True):
    entries = _parse_asvspoof5_protocol(protocol_path, need_label=need_label)
    items = []
    missing = []
    for key, label in entries:
        audio_path = _find_audio_path(audio_roots, key)
        if audio_path is None:
            missing.append(key)
            continue
        label_id = None if label is None else (1 if label == "bonafide" else 0)
        items.append((key, str(audio_path), label_id))
    if missing:
        print(f"[data] ASVspoof5 missing audio: {len(missing)}/{len(entries)} files; first={missing[0]}")
    return items
        
class asvspoof_dataModule(L.LightningDataModule):
        def __init__(self,args):
                super().__init__()
                self.args = args
                self.dataset = getattr(args, "dataset", os.environ.get("MOEF_DATASET", "asvspoof2019la")).lower()
                
                # ASVspoof2019 LA. Override on a new machine with
                # MOEF_ASVSPOOF2019_LA_ROOT=/path/to/.../LA/LA.
                asv2019_root = Path(
                    os.environ.get("MOEF_ASVSPOOF2019_LA_ROOT", DEFAULT_ASV2019_LA_ROOT)
                ).expanduser()
                self.protocols_path = _path_with_slash(asv2019_root / "ASVspoof2019_LA_cm_protocols")
                self.train_protocols_file = str(asv2019_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt")
                self.dev_protocols_file = str(asv2019_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt")
                self.dataset_base_path = _path_with_slash(asv2019_root)
                self.train_set = _path_with_slash(asv2019_root / "ASVspoof2019_LA_train")
                self.dev_set = _path_with_slash(asv2019_root / "ASVspoof2019_LA_dev")
                # test set 
                self.eval_protocols_file_19 = str(asv2019_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt")
                self.eval_set_19 = _path_with_slash(asv2019_root / "ASVspoof2019_LA_eval")
                self.eval_protocols_file_21 = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/datasets/ASVspoof2021_LA_eval/eval_file/ASVspoof2021.LA.cm.eval.trl.txt"
                self.eval_set_21 = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/datasets/ASVspoof2021_LA_eval/"

                
                self.LA21 = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/ASVspoof2021_LA_eval/eval_file/ASVspoof2021.LA.cm.eval.trl.txt"
                self.LA21FLAC = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/ASVspoof2021_LA_eval/"
                self.LA21TRIAL = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/ASVspoof2021_LA_eval/eval_file/CM_trial_metadata.txt"

                self.DF21 = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/ASVspoof2021_DF_eval/ASVspoof2021.DF.cm.eval.trl.txt"
                self.DF21FLAC = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/ASVspoof2021_DF_eval/"
                self.DF21TRIAL = "/data8/wangzhiyong/project/fakeAudioDetection/investigating_partial_pre-trained_model_for_fake_audio_detection/reference/fad/aasist/datasets/ASVspoof2021_DF_eval/trial_metadata.txt"

                itw_root = Path(
                    os.environ.get("MOEF_IN_THE_WILD_ROOT", DEFAULT_ITW_ROOT)
                ).expanduser()
                self.ITWTXT = str(Path(os.environ.get("MOEF_IN_THE_WILD_LABEL", itw_root / "label.txt")).expanduser())
                self.ITWDIR = str(Path(os.environ.get("MOEF_IN_THE_WILD_WAV_DIR", itw_root / "wav")).expanduser())

                self.asv5_train_protocol = None
                self.asv5_dev_protocol = None
                self.asv5_eval_protocol = None
                self.asv5_train_roots = []
                self.asv5_dev_roots = []
                self.asv5_eval_roots = []
                if self.dataset == "asvspoof5":
                    asv5_root = Path(os.environ.get("MOEF_ASVSPOOF5_ROOT", DEFAULT_ASV5_ROOT)).expanduser()
                    asv5_protocol_roots = [
                        asv5_root / "protocols",
                        asv5_root / "ASVspoof5_protocols",
                        asv5_root,
                    ]
                    self.asv5_train_protocol = _first_existing(
                        [root / "ASVspoof5.train.tsv" for root in asv5_protocol_roots],
                        "ASVspoof5 train protocol",
                    )
                    self.asv5_dev_protocol = _first_existing(
                        [root / "ASVspoof5.dev.track_1.tsv" for root in asv5_protocol_roots],
                        "ASVspoof5 dev protocol",
                    )
                    self.asv5_train_roots = [asv5_root / "flac_T", asv5_root]
                    self.asv5_dev_roots = [asv5_root / "flac_D", asv5_root]
                    self.asv5_eval_protocol = _first_existing(
                        [root / "ASVspoof5.eval.track_1.tsv" for root in asv5_protocol_roots],
                        "ASVspoof5 eval protocol",
                    )
                    self.asv5_eval_roots = [asv5_root / "flac_E_eval", asv5_root]

                
                
                self.truncate = args.truncate
                self.predict = args.testset # LA21, DF21, ITW

        def setup(self, stage: str):
            # Assign train/val datasets for use in dataloaders
            if stage == "fit":
                if self.dataset == "asvspoof5":
                    train_items = _asvspoof5_items(
                        self.asv5_train_protocol,
                        self.asv5_train_roots,
                        need_label=True,
                    )
                    dev_items = _asvspoof5_items(
                        self.asv5_dev_protocol,
                        self.asv5_dev_roots,
                        need_label=True,
                    )
                    print(f"[data] ASVspoof5 train files: {len(train_items)}")
                    print(f"[data] ASVspoof5 dev files: {len(dev_items)}")
                    self.asvspoof19_trn_set = Dataset_AudioPath_train(
                        items=train_items,
                        cut=self.truncate,
                        args=self.args,
                    )
                    self.asvspoof19_val_set = Dataset_AudioPath_devNeval(
                        items=dev_items,
                        cut=self.truncate,
                        args=self.args,
                    )
                else:
                    d_label_trn,file_train = genSpoof_list(
                        dir_meta=self.train_protocols_file,
                        is_train=True,
                        is_eval=False
                        )
                    
                    self.asvspoof19_trn_set = Dataset_ASVspoof2019_train(
                        list_IDs=file_train,
                        labels=d_label_trn,
                        base_dir=self.train_set,
                        cut=self.truncate,
                        args= self.args
                        )
    
                    _, file_dev = genSpoof_list(
                        dir_meta=self.dev_protocols_file,
                        is_train=False,
                        is_eval=False)
                    
                    self.asvspoof19_val_set = Dataset_ASVspoof2019_devNeval(
                        list_IDs=file_dev,
                        base_dir=self.dev_set,
                        args= self.args,
                        cut=self.truncate
                        )
   
            # Assign test dataset for use in dataloader(s)
            if stage == "test":
                if self.dataset == "asvspoof5":
                    eval_items = _asvspoof5_items(
                        self.asv5_eval_protocol,
                        self.asv5_eval_roots,
                        need_label=True,
                    )
                    self.asvspoof19_test_set = Dataset_AudioPath_evaltest(
                        items=eval_items,
                        cut=self.truncate,
                    )
                else:
                    file_eval = genSpoof_list(
                        dir_meta=self.eval_protocols_file_19,
                        is_train=False,
                        is_eval=True
                        )
                    self.asvspoof19_test_set = Dataset_ASVspoof2019_evaltest(
                        list_IDs=file_eval,
                        base_dir=self.eval_set_19,
                        cut=self.truncate
                        )

            if stage == "predict":
                if self.predict == "LA21":
                    file_list=[]
                    with open(self.LA21, 'r') as f:
                        l_meta = f.readlines()
                    for line in l_meta:
                        key= line.strip()
                        file_list.append(key)
                    print(f"no.{(len(file_list))} of eval  trials")
                    self.predict_set = Dataset_ASVspoof2019_evaltest(
                        list_IDs=file_list,
                        base_dir=self.LA21FLAC,
                        cut=self.truncate)
 
                elif self.predict == "DF21":
                    file_list=[]
                    with open(self.DF21, 'r') as f:
                        l_meta = f.readlines()
                    for line in l_meta:
                        key= line.strip()
                        file_list.append(key)
                    print(f"no.{(len(file_list))} of eval  trials")
                    self.predict_set = Dataset_ASVspoof2019_evaltest(
                        list_IDs=file_list,
                        base_dir=self.DF21FLAC,
                        cut=self.truncate)
 
                elif self.predict == "ITW":
                    file_list=[]
                    # 打开文件
                    with open(self.ITWTXT, 'r') as file:
                        lines = file.readlines()
                        for line in lines:
                            columns = line.split()
                            file_list.append(columns[1])
                    self.predict_set = dataset_itw(
                        list_IDs=file_list,
                        base_dir=self.ITWDIR,
                        cut=self.truncate)

                    
                    
                

        def train_dataloader(self):
            return DataLoader(self.asvspoof19_trn_set, batch_size=self.args.batch_size, shuffle=True,drop_last = True,num_workers=self.args.num_workers)

        def val_dataloader(self):
            return DataLoader(self.asvspoof19_val_set, batch_size=self.args.batch_size, shuffle=False,drop_last = False,num_workers=self.args.num_workers)            

        def test_dataloader(self):                
            datald =  DataLoader(
                self.asvspoof19_test_set,batch_size=self.args.batch_size,
                shuffle=False,num_workers=self.args.num_workers
                )
            if "," in self.args.gpuid:
                datald =  DataLoader(
                    self.asvspoof19_test_set,batch_size=self.args.batch_size,
                    shuffle=False,num_workers=self.args.num_workers,
                    sampler=DistributedSampler(self.asvspoof19_test_set)
                    )
            return datald

        def predict_dataloader(self):
            predict_loader = DataLoader(
                self.predict_set,
                batch_size= self.args.batch_size,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
                num_workers=self.args.num_workers)
            if "," in self.args.gpuid:
                predict_loader = DataLoader(
                    self.predict_set,
                    batch_size= self.args.batch_size,
                    shuffle=False,
                    drop_last=False,
                    pin_memory=True,
                    sampler=DistributedSampler(self.predict_set),
                    num_workers=self.args.num_workers
                    )
            return predict_loader
 
      
      
      

class dataset_itw(Dataset):
    def __init__(self, list_IDs, base_dir,cut = 64600):
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(os.path.join(self.base_dir,f"{key}.wav"))
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
    


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]
    if x_len == max_len:
        return x

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def _read_audio_mono(path):
    x, fs = sf.read(path)
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    return x, fs


class Dataset_AudioPath_train(Dataset):
    def __init__(self, items, args, cut=64600):
        self.items = items
        self.args = args
        self.cut = cut

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, audio_path, label = self.items[index]
        x, fs = _read_audio_mono(audio_path)
        if self.args.usingDA and (np.random.rand() < self.args.da_prob):
            x = process_Rawboost_feature(x, fs, self.args, self.args.algo)
        x_pad = x if self.cut == 0 else pad_random(x, self.cut)
        return Tensor(x_pad), label, key


class Dataset_AudioPath_devNeval(Dataset):
    def __init__(self, items, args=None, cut=64600):
        self.items = items
        self.args = args
        self.cut = cut

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, audio_path, _ = self.items[index]
        x, fs = _read_audio_mono(audio_path)
        if self.args and self.args.usingDA:
            x = process_Rawboost_feature(x, fs, self.args, self.args.algo)
        x_pad = x if self.cut == 0 else pad(x, self.cut)
        return Tensor(x_pad), key, key


class Dataset_AudioPath_evaltest(Dataset):
    def __init__(self, items, cut=64600):
        self.items = items
        self.cut = cut

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        key, audio_path, _ = self.items[index]
        x, _ = _read_audio_mono(audio_path)
        x_pad = x if self.cut == 0 else pad(x, self.cut)
        return Tensor(x_pad), key


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir,args,cut = 64600):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.args = args
        self.cut = cut  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs  = sf.read(os.path.join(self.base_dir , f"flac/{key}.flac"))
        if self.args.usingDA and (np.random.rand() < self.args.da_prob):
            X=process_Rawboost_feature(X,fs,self.args,self.args.algo)
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        # 1. tensor 2.label 3.filename
        return x_inp, y, key


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir,args=None,cut = 64600):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir,f"flac/{key}.flac"))
        if self.args.usingDA and ("ASVspoof2019_LA_dev" in self.base_dir):
            X=process_Rawboost_feature(X,fs,self.args,self.args.algo)
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        # 1.tensor 2.filename
        return x_inp, key ,key


class Dataset_ASVspoof2019_evaltest(Dataset):
    def __init__(self, list_IDs, base_dir,args=None,cut = 64600):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = cut  # take ~4 sec audio (64600 samples)
        self.args = args

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, fs = sf.read(os.path.join(self.base_dir,f"flac/{key}.flac"))
        if self.cut == 0:
            X_pad = X
        else:
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key 
      
      
      
      
      
      
      
      
