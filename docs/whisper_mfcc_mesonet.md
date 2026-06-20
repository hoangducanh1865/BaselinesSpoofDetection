# Whisper-MFCC-MesoNet

Whisper-MFCC-MesoNet nam o `baselines/whisper_mfcc_mesonet`. Baseline nay dung Whisper tiny.en encoder ket hop MFCC + delta + double-delta, roi dua vao MesoInception4.

Score la raw logit, score cang cao nghia la audio cang giong `bonafide`.

## Checkpoint

Model card dung checkpoint:

```text
whisper_mfcc_mesonet_finetuned.pth
```

Adapter mac dinh tim tai:

```text
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2021df/whisper_mfcc_mesonet/whisper_mfcc_mesonet_finetuned.pth
```

Neu file nam cho khac, export:

```bash
export WHISPER_MFCC_MESONET_CKPT=/path/to/whisper_mfcc_mesonet_finetuned.pth
```

Whisper encoder tiny.en can nam tai:

```text
baselines/whisper_mfcc_mesonet/src/models/assets/tiny_enc.en.pt
```

Hoac override:

```bash
export WHISPER_MODEL_WEIGHTS_PATH=/path/to/tiny_enc.en.pt
```

## Moi truong

Dung env chung cho cac baseline khong phai MoLEx:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate nes2net_anhhd
conda install -c conda-forge sox -y
pip install soundfile pandas tqdm pyyaml torchaudio
```

Baseline nay can libsox vi preprocessing dung `torchaudio.sox_effects` de silence-trim.

## Chay evaluation

Preprocessing dung theo upstream/model card: resample 16 kHz, sox silence-trim, repeat-pad/truncate thanh 480,000 samples (30 s).

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate nes2net_anhhd

export WHISPER_MFCC_MESONET_CKPT=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2021df/whisper_mfcc_mesonet/whisper_mfcc_mesonet_finetuned.pth
export WHISPER_MFCC_MESONET_EVAL_BATCH_SIZE=8
export WHISPER_MFCC_MESONET_EVAL_NUM_WORKERS=4

python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset dfadd_test
python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset fake_or_real_norm
python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset vlsp2025
python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset vsasv
python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset in_the_wild
python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset asvspoof2019la
python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset asvspoof2021la
python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset asvspoof2021df
python main.py --baseline whisper_mfcc_mesonet --mode eval --dataset asvspoof5
```

## Output

```text
outputs/whisper_mfcc_mesonet/evals/YYYY_MM_DD_HH_MM_SS__<checkpoint>__on__<dataset>/
  eval_output.txt
  eval_EER.txt
  eval_config.txt
```
