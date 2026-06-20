# AASIST / AASIST-L

AASIST va AASIST-L nam o `baselines/aasist`. Hai bien the dung cung file kien truc `models/AASIST.py`; AASIST-L la ban nhe hon, dung `config/AASIST-L.conf` voi residual stack va graph dimensions nho hon.

## Pretrained checkpoint tren server

Checkpoint AASIST full da clone tu HuggingFace duoc dat tai:

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

Checkpoint AASIST-L da clone tu HuggingFace duoc dat tai:

```text
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/aasist_l/aasist_l_asvspoof2019la.pth
```

Thu muc checkpoint:

```text
aasist_l_asvspoof2019la.pth
aasist_l.py
_net.py
LICENSE
meta.yaml
README.md
```

Score cua AASIST/AASIST-L la logit lop `bonafide`; gia tri cang cao nghia la cang giong audio that.

## Cai dat tren server

Co the chay trong moi truong chung cho cac baseline pretrained/eval:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate nes2net_anhhd
```

Neu moi truong thieu dependency:

```bash
pip install soundfile tqdm pandas pyarrow
```

`pyarrow` chi can cho `dfadd_test` vi dataset nay luu audio trong file parquet. Cac baseline khac van co the chay neu chua cai `pyarrow`.

## Chay evaluation bang CLI chung

Dat checkpoint AASIST full:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate nes2net_anhhd

export AASIST_CKPT=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/aasist/aasist_asvspoof2019la.pth
export AASIST_EVAL_BATCH_SIZE=128
export AASIST_EVAL_NUM_WORKERS=16
```

Chay AASIST full theo thu tu tu nhanh den cham, bo qua `asvspoof2019la` va `in_the_wild` neu da co ket qua:

```bash
python main.py --baseline aasist --mode eval --dataset dfadd_test --ckpt "$AASIST_CKPT"
python main.py --baseline aasist --mode eval --dataset fake_or_real --ckpt "$AASIST_CKPT"
python main.py --baseline aasist --mode eval --dataset vlsp2025 --ckpt "$AASIST_CKPT"
python main.py --baseline aasist --mode eval --dataset vsasv --ckpt "$AASIST_CKPT"
python main.py --baseline aasist --mode eval --dataset asvspoof2021la --ckpt "$AASIST_CKPT"
python main.py --baseline aasist --mode eval --dataset asvspoof2021df --ckpt "$AASIST_CKPT"
python main.py --baseline aasist --mode eval --dataset asvspoof5 --ckpt "$AASIST_CKPT"
python main.py --baseline aasist --mode eval --dataset asvspoof2021pa --ckpt "$AASIST_CKPT"
```

Neu can chay lai cac tap da chay:

```bash
python main.py --baseline aasist --mode eval --dataset asvspoof2019la --ckpt "$AASIST_CKPT"
python main.py --baseline aasist --mode eval --dataset in_the_wild --ckpt "$AASIST_CKPT"
```

Dat checkpoint AASIST-L:

```bash
export AASIST_L_CKPT=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/aasist_l/aasist_l_asvspoof2019la.pth
```

Chay AASIST-L:

```bash
python main.py --baseline aasist_l --mode eval --dataset dfadd_test --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset fake_or_real --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset vlsp2025 --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset vsasv --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset asvspoof2019la --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset in_the_wild --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset asvspoof2021la --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset asvspoof2021df --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset asvspoof5 --ckpt "$AASIST_L_CKPT"
python main.py --baseline aasist_l --mode eval --dataset asvspoof2021pa --ckpt "$AASIST_L_CKPT"
```

Neu chi muon sinh score, khong tinh EER:

```bash
python main.py --baseline aasist --mode score --dataset asvspoof2019la --ckpt "$AASIST_CKPT"
python main.py --baseline aasist_l --mode score --dataset asvspoof2019la --ckpt "$AASIST_L_CKPT"
```

## Noi luu ket qua

Ket qua duoc luu duoi:

```text
outputs/aasist/evals/YYYY_MM_DD_HH_MM_SS__<checkpoint_name>__on__<dataset>/
  eval_output.txt
  eval_EER.txt
  eval_config.txt
```

AASIST-L luu rieng tai:

```text
outputs/aasist_l/evals/YYYY_MM_DD_HH_MM_SS__<checkpoint_name>__on__<dataset>/
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
asvspoof2021la
asvspoof2021df
asvspoof2021pa
asvspoof5
dfadd_test
fake_or_real
fake_or_real_2sec
fake_or_real_norm
fake_or_real_original
fake_or_real_rerec
in_the_wild
vlsp2025
vsasv
```

Duong dan dataset mac dinh dang duoc hard-code trong adapter:

```text
asvspoof2019la:       /home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
asvspoof2019pa:       /home/user14/anhhd/spoof/datasets/asvspoof2019/PA/PA
asvspoof2021*:        /home/user14/anhhd/spoof/datasets/asvspoof2021
asvspoof5:            /home/user14/anhhd/spoof/datasets/asvspoof5
dfadd_test:           /home/user14/anhhd/spoof/datasets/dfadd_test
fake_or_real*:        /home/user14/anhhd/spoof/datasets/fake_or_real
in_the_wild:          /home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild
vlsp2025:             /home/user14/anhhd/spoof/datasets/vlsp2025
vsasv:                /home/user14/anhhd/spoof/datasets/vsasv
```

Ghi chu dataset:

- `fake_or_real` mac dinh dung `fake_or_real_norm`, vi ban nay da xu ly 16 kHz/mono. Neu muon chay cac bien the khac, dung `fake_or_real_2sec`, `fake_or_real_original`, `fake_or_real_rerec`.
- `vlsp2025` dung file labeled dev `data-train/split/original+adversarial+tts+vc/dev.txt`; public/private trial files la SASV/ASV pair trial, khong dung truc tiep de tinh CM EER o adapter nay.
- `vsasv` dung `result_reproduce/cm/bonafide_replay_adversarial_vc.txt` va suy ra label tu path: bonafide la that, replay/voice_conversion/adversarial la spoof.
- `dfadd_test` extract audio tu parquet vao `outputs/aasist/meta/dfadd_test/audio/` o lan chay dau.

Neu can doi root dataset cho mot lan chay, co the dung:

```bash
export SPOOF_DATA_ROOT=/path/to/dataset/root
```

Luu y: `SPOOF_DATA_ROOT` override root cho dataset dang chay, nen chi dung khi test mot dataset tai mot thoi diem.

## Luu y ve ket qua HuggingFace

Model card bao cao checkpoint AASIST full pretrained tren ASVspoof2019 LA dat:

```text
ASVspoof2019 LA: 0.83% EER
ASVspoof2021 LA: 12.35% EER
ASVspoof2021 DF: 17.04% EER
In-The-Wild:     43.01% EER
```

AASIST-L pretrained tren ASVspoof2019 LA dat:

```text
ASVspoof2019 LA: 0.99% EER
ASVspoof2021 LA: 13.15% EER
ASVspoof2021 DF: 15.96% EER
In-The-Wild:     44.45% EER
ASVspoof5:       37.53% EER
```

Ket qua local co the chenh nhe do version PyTorch, audio loader, batch size, hoac cach cat/pad waveform. Adapter hien tai dung deterministic first-window 64600 samples giong evaluation cua upstream.
