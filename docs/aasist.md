# AASIST

AASIST nam o `baselines/aasist`. Baseline nay dung raw waveform truc tiep va checkpoint pretrained tren ASVspoof2019 LA.

## Pretrained checkpoint tren server

Checkpoint da clone tu HuggingFace duoc dat tai:

```text
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/aasist/aasist_asvspoof2019la.pth
```

Thu muc checkpoint:

```text
aasist_asvspoof2019la.pth
LICENSE
meta.yaml
README.md
```

Score cua AASIST la logit lop `bonafide`; gia tri cang cao nghia la cang giong audio that.

## Cai dat tren server

Co the chay trong moi truong da dung cho project neu da co PyTorch, pandas, soundfile va tqdm:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate molex_anhhd
```

Neu moi truong thieu dependency:

```bash
pip install soundfile tqdm pandas
```

## Chay evaluation bang CLI chung

Dat checkpoint:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate molex_anhhd

export AASIST_CKPT=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/aasist/aasist_asvspoof2019la.pth
export AASIST_EVAL_BATCH_SIZE=128
export AASIST_EVAL_NUM_WORKERS=16
```

Chay in-domain tren ASVspoof2019 LA:

```bash
python main.py --baseline aasist --mode eval --dataset asvspoof2019la --ckpt "$AASIST_CKPT"
```

Chay cross-domain tren ASVspoof5:

```bash
python main.py --baseline aasist --mode eval --dataset asvspoof5 --ckpt "$AASIST_CKPT"
```

Chay cross-domain tren In-The-Wild:

```bash
python main.py --baseline aasist --mode eval --dataset in_the_wild --ckpt "$AASIST_CKPT"
```

Neu chi muon sinh score, khong tinh EER:

```bash
python main.py --baseline aasist --mode score --dataset asvspoof2019la --ckpt "$AASIST_CKPT"
```

## Noi luu ket qua

Ket qua duoc luu duoi:

```text
outputs/aasist/evals/YYYY_MM_DD_HH_MM_SS__<checkpoint_name>__on__<dataset>/
  eval_output.txt
  eval_EER.txt
  eval_config.txt
```

Trong do `eval_output.txt` co format:

```text
utt_id<TAB>label<TAB>score
```

`eval_EER.txt` ghi EER va threshold.

## Dataset dang ho tro

CLI chung hien ho tro:

```text
asvspoof2019la
asvspoof2019pa
asvspoof5
in_the_wild
```

Duong dan dataset mac dinh dang duoc hard-code trong adapter:

```text
asvspoof2019la: /home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
asvspoof2019pa: /home/user14/anhhd/spoof/datasets/asvspoof2019/PA/PA
asvspoof5:      /home/user14/anhhd/spoof/datasets/asvspoof5
in_the_wild:    /home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild
```

Neu can doi root dataset cho mot lan chay, co the dung:

```bash
export SPOOF_DATA_ROOT=/path/to/dataset/root
```

Luu y: `SPOOF_DATA_ROOT` override root cho dataset dang chay, nen chi dung khi test mot dataset tai mot thoi diem.

## Luu y ve ket qua HuggingFace

Model card bao cao checkpoint AASIST pretrained tren ASVspoof2019 LA dat:

```text
ASVspoof2019 LA: 0.83% EER
ASVspoof2021 LA: 12.35% EER
ASVspoof2021 DF: 17.04% EER
In-The-Wild:     43.01% EER
```

Ket qua local co the chenh nhe do version PyTorch, audio loader, batch size, hoac cach cat/pad waveform. Adapter hien tai dung deterministic first-window 64600 samples giong evaluation cua upstream.
