# Nes2Net

Nes2Net nam o `baselines/nes2net`.

Theo paper `stuff/reference_papers/nes2net/nes2net.pdf`, setup khong dung mot SSL front-end duy nhat cho moi thi nghiem:

- ASVspoof2019/ASVspoof2021/In-The-Wild: dung `wav2vec 2.0`/XLS-R 300M + Nes2Net-X.
- ASVspoof5: dung WavLM-Large + Nes2Net-X theo guideline ASVspoof5.

Code trong repo gom 2 phan tu repo chinh thuc:

- Branch `asvspoof5`: cac file WavLM o `config/`, `models/`, `main.py`.
- Branch `main`: wrapper `model_scripts/wav2vec2_Nes2Net_X.py` cho checkpoint ASVspoof2019/2021/In-The-Wild.

Thu muc `.git` cua repo goc khong duoc vendor vao codebase nay.

Adapter hien tai tu dong chon backbone theo checkpoint path. Co the ep thu cong bang:

```bash
export NES2NET_BACKBONE=xlsr   # hoac wavlm
```

Score cua model la logit lop `bonafide`; score cang cao nghia la audio cang giong that.

## Pretrained SSL models tren server

Da co du 2 SSL checkpoint can thiet:

```text
/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large/WavLM-Large.pt
/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt
```

Adapter mac dinh tro toi 2 duong dan nay. Neu can override:

```bash
export WAVLM_LARGE_PATH=/path/to/WavLM-Large.pt
export XLSR2_300M_PATH=/path/to/xlsr2_300m.pt
```

Khong can tai them SSL model neu 2 file tren ton tai.

## Pretrained Nes2Net checkpoints tren server

Checkpoint train tren ASVspoof2019 LA:

```text
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/nes2net/nes2net_asvspoof2019la.pth
```

Checkpoint train tren ASVspoof5:

```text
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof5/nes2net/nes2net_asvspoof5.pth
```

## Cai dat tren server

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate molex_anhhd
```

Dependency toi thieu:

```bash
pip install soundfile tqdm pandas
```

Voi checkpoint ASVspoof2019LA, backend XLS-R can `fairseq`. Neu env chua import duoc `fairseq`, cai dung snapshot fairseq ma tac gia khuyen dung:

```bash
cd /home/user14/anhhd/spoof
if [ ! -d fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 ]; then
  git clone https://github.com/pytorch/fairseq.git fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
fi
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
git checkout a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
```

Kiem tra nhanh:

```bash
python - <<'PY'
import fairseq, torch
print("fairseq ok")
for p in [
    "/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt",
    "/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large/WavLM-Large.pt",
]:
    print(p, torch.load(p, map_location="cpu").keys())
PY
```

## Eval checkpoint train tren ASVspoof2019 LA

Checkpoint nay dung XLS-R 300M + Nes2Net-X.

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate molex_anhhd

export NES2NET_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/nes2net/nes2net_asvspoof2019la.pth
export XLSR2_300M_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt
export NES2NET_EVAL_BATCH_SIZE=16
export NES2NET_EVAL_NUM_WORKERS=8

python main.py --baseline nes2net --mode eval --dataset asvspoof2019la --ckpt "$NES2NET_CKPT_2019"
python main.py --baseline nes2net --mode eval --dataset asvspoof5 --ckpt "$NES2NET_CKPT_2019"
```

Voi In-The-Wild, paper eval tren full utterance, nen adapter mac dinh de batch size 1 neu khong set batch size:

```bash
unset NES2NET_EVAL_BATCH_SIZE
python main.py --baseline nes2net --mode eval --dataset in_the_wild --ckpt "$NES2NET_CKPT_2019"
```

Neu muon chay nhanh hon va chap nhan padding trong batch:

```bash
export NES2NET_EVAL_BATCH_SIZE=4
python main.py --baseline nes2net --mode eval --dataset in_the_wild --ckpt "$NES2NET_CKPT_2019"
```

## Eval checkpoint train tren ASVspoof5

Checkpoint nay dung WavLM-Large + Nes2Net-X.

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate molex_anhhd

export NES2NET_CKPT_ASV5=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof5/nes2net/nes2net_asvspoof5.pth
export WAVLM_LARGE_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large/WavLM-Large.pt
export NES2NET_EVAL_BATCH_SIZE=16
export NES2NET_EVAL_NUM_WORKERS=8

python main.py --baseline nes2net --mode eval --dataset asvspoof5 --ckpt "$NES2NET_CKPT_ASV5"
python main.py --baseline nes2net --mode eval --dataset asvspoof2019la --ckpt "$NES2NET_CKPT_ASV5"
```

Cross-dataset In-The-Wild:

```bash
unset NES2NET_EVAL_BATCH_SIZE
python main.py --baseline nes2net --mode eval --dataset in_the_wild --ckpt "$NES2NET_CKPT_ASV5"
```

## Score-only

Neu chi muon sinh score, khong tinh EER:

```bash
python main.py --baseline nes2net --mode score --dataset asvspoof2019la --ckpt "$NES2NET_CKPT_2019"
```

## Noi luu ket qua

Ket qua eval luu tai:

```text
outputs/nes2net/evals/YYYY_MM_DD_HH_MM_SS__<checkpoint_name>__on__<dataset>/
  eval_output.txt
  eval_EER.txt
  eval_config.txt
```

`eval_output.txt` co format:

```text
utt_id<TAB>label<TAB>score
```

`eval_config.txt` ghi dataset, checkpoint, backbone, SSL checkpoint, max_len, batch size va num workers.

## Dataset dang ho tro

CLI chung hien ho tro:

```text
asvspoof2019la
asvspoof2019pa
asvspoof5
in_the_wild
```

Duong dan dataset mac dinh:

```text
asvspoof2019la: /home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
asvspoof2019pa: /home/user14/anhhd/spoof/datasets/asvspoof2019/PA/PA
asvspoof5:      /home/user14/anhhd/spoof/datasets/asvspoof5
in_the_wild:    /home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild
```

Neu can override root cua dataset dang chay:

```bash
export SPOOF_DATA_ROOT=/path/to/dataset/root
```

## Ghi chu theo paper

Setup training/eval quan trong:

- ASVspoof2019 train: crop/concat audio thanh doan 64,600 samples ca train va test; train 100 epochs; chon checkpoint tot nhat tren validation.
- In-The-Wild: paper eval full duration cua moi utterance de tranh mat doan bi spoof mot phan.
- ASVspoof5: train/valid/test theo partition chinh thuc, dung WavLM front-end, dung MUSAN/RIR augmentation, early stop neu dev khong cai thien 5 epoch.

Ket qua tham khao trong paper:

```text
ASVspoof2021 LA/DF, train ASVspoof2019:
  Nes2Net-X: 1.73% / 1.65% EER single checkpoint
  Nes2Net-X: 1.88% / 1.49% EER voi average 5 checkpoints

In-The-Wild:
  Nes2Net-X: 5.52% EER best, 6.60% mean

ASVspoof5:
  Nes2Net-X: 5.92% EER
```
