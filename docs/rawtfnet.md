# RawTFNet

RawTFNet nam o `baselines/rawtfnet`. Baseline nay la CNN raw-waveform nhe, train tren ASVspoof2019 LA theo paper/model card.

Score la logit lop `bonafide` o index 1; score cang cao nghia la audio cang giong that.

## Checkpoint

Repo hien co checkpoint nho tu source clone:

```text
baselines/rawtfnet/ckpts/Best_RawTFNet_32.pth
```

Co the override bang:

```bash
export RAWTFNET_CKPT=/path/to/Best_RawTFNet_32.pth
```

## Moi truong

Dung env chung cho cac baseline khong phai MoLEx:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate nes2net_anhhd
pip install soundfile pandas tqdm librosa
```

## Chay evaluation

RawTFNet dung deterministic first-window 64,600 samples tai 16 kHz, dung theo model card.

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate nes2net_anhhd

export RAWTFNET_EVAL_BATCH_SIZE=128
export RAWTFNET_EVAL_NUM_WORKERS=8

python main.py --baseline rawtfnet --mode eval --dataset dfadd_test
python main.py --baseline rawtfnet --mode eval --dataset fake_or_real_norm
python main.py --baseline rawtfnet --mode eval --dataset vlsp2025
python main.py --baseline rawtfnet --mode eval --dataset vsasv
python main.py --baseline rawtfnet --mode eval --dataset in_the_wild
python main.py --baseline rawtfnet --mode eval --dataset asvspoof2019la
python main.py --baseline rawtfnet --mode eval --dataset asvspoof2021la
python main.py --baseline rawtfnet --mode eval --dataset asvspoof2021df
python main.py --baseline rawtfnet --mode eval --dataset asvspoof5
```

## Output

```text
outputs/rawtfnet/evals/YYYY_MM_DD_HH_MM_SS__<checkpoint>__on__<dataset>/
  eval_output.txt
  eval_EER.txt
  eval_config.txt
```
