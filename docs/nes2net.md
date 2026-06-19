# Nes2Net

Nes2Net nam o `baselines/nes2net`. Code trong repo hien tai dung WavLM-Large front-end va Nes2Net-X backend (`models/wavlm_nes2net.py`).

Score cua model la logit lop `bonafide`; score cang cao nghia la audio cang giong that.

## Pretrained checkpoints tren server

Checkpoint train tren ASVspoof2019 LA:

```text
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/nes2net/nes2net_asvspoof2019la.pth
```

Thu muc:

```text
meta.yaml
nes2net_asvspoof2019la.pth
README.md
```

Checkpoint train tren ASVspoof5:

```text
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof5/nes2net/nes2net_asvspoof5.pth
```

Ngoai checkpoint Nes2Net, code can WavLM-Large checkpoint:

```text
/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large/WavLM-Large.pt
```

Adapter mac dinh da tro toi duong dan nay. Neu can override:

```bash
export WAVLM_LARGE_PATH=/path/to/WavLM-Large.pt
```

## Cai dat tren server

Dung moi truong project hien tai:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate molex_anhhd
```

Neu thieu dependency:

```bash
pip install soundfile tqdm pandas
```

## Chay eval checkpoint train tren ASVspoof2019 LA

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate molex_anhhd

export NES2NET_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/nes2net/nes2net_asvspoof2019la.pth
export NES2NET_EVAL_BATCH_SIZE=16
export NES2NET_EVAL_NUM_WORKERS=8

python main.py --baseline nes2net --mode eval --dataset asvspoof2019la --ckpt "$NES2NET_CKPT_2019"
python main.py --baseline nes2net --mode eval --dataset asvspoof5 --ckpt "$NES2NET_CKPT_2019"
python main.py --baseline nes2net --mode eval --dataset in_the_wild --ckpt "$NES2NET_CKPT_2019"
```

## Chay eval checkpoint train tren ASVspoof5

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate molex_anhhd

export NES2NET_CKPT_ASV5=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof5/nes2net/nes2net_asvspoof5.pth
export NES2NET_EVAL_BATCH_SIZE=16
export NES2NET_EVAL_NUM_WORKERS=8

python main.py --baseline nes2net --mode eval --dataset asvspoof5 --ckpt "$NES2NET_CKPT_ASV5"
python main.py --baseline nes2net --mode eval --dataset asvspoof2019la --ckpt "$NES2NET_CKPT_ASV5"
python main.py --baseline nes2net --mode eval --dataset in_the_wild --ckpt "$NES2NET_CKPT_ASV5"
```

Neu A100 con nhieu VRAM, co the tang batch:

```bash
export NES2NET_EVAL_BATCH_SIZE=32
```

Neu bi OOM thi giam:

```bash
export NES2NET_EVAL_BATCH_SIZE=8
```

## Chay score-only

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

`eval_config.txt` ghi checkpoint Nes2Net, checkpoint WavLM, dataset, batch size va num workers.

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

## Sanity check checkpoint WavLM

```bash
python - <<'PY'
import torch
p="/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large/WavLM-Large.pt"
ckpt=torch.load(p, map_location="cpu")
print(ckpt.keys())
print(ckpt["cfg"]["encoder_layers"], ckpt["cfg"]["encoder_embed_dim"], ckpt["cfg"]["encoder_attention_heads"])
PY
```

Expected:

```text
dict_keys(['cfg', 'model'])
24 1024 16
```

## Luu y ve checkpoint

Adapter hien tai khoi tao kien truc WavLM-Large + Nes2Net-X tu `baselines/nes2net/models/wavlm_nes2net.py`.
Neu checkpoint bao loi load state dict khong khop key/shape, kha nang checkpoint do thuoc wrapper XLS-R 300M khac voi source hien tai. Khi do can bo sung dung file `_net.py` / wrapper tu checkpoint repo vao codebase truoc khi eval.

## Ket qua tham khao tu HuggingFace

Model card bao cao checkpoint Nes2Net train tren ASVspoof2019 LA dat:

```text
ASVspoof2019 LA: 0.13% EER
ASVspoof2021 LA: 6.14% EER
ASVspoof2021 DF: 3.61% EER
In-The-Wild:     8.48% EER
ASVspoof5:       22.25% EER
```

Ket qua local co the chenh lech do backbone checkpoint, audio loader, version thu vien, batch size hoac cach cat/pad waveform.
