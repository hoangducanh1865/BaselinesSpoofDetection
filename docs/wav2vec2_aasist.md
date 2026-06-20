# Wav2Vec2-AASIST

Baseline Wav2Vec2-AASIST dung XLS-R 300M lam SSL front-end va AASIST lam back-end. Adapter da duoc noi vao CLI chung voi 2 ten baseline:

- `wav2vec2_aasist`
- `w2v2_aasist`

Checkpoint pretrained tren ASVspoof2019 LA dang duoc dat tren server tai:

```bash
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/w2v2_aasist/w2v2_aasist_asvspoof2019la.pth
```

Model SSL can dung:

```bash
/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt
```

Theo paper va model card, evaluation dung raw waveform mono 16 kHz, crop/pad co dinh ve `64600` samples; score la logit lop `bonafide` o index 1, score cao hon nghia la audio nghieng ve that hon.

## Moi truong

Baseline nay can `torch`, `fairseq`, `soundfile`, `pandas`, `tqdm`. Neu dataset co file khac 16 kHz thi can them `librosa` de resample giong code goc. Co the dung env `nes2net_anhhd` neu env do da cai `fairseq` dung snapshot.

Kiem tra nhanh:

```bash
conda activate nes2net_anhhd
python - <<'PY'
import torch, fairseq, soundfile, pandas, librosa
print("torch", torch.__version__)
print("fairseq ok")
PY
```

Neu can tao/cai env rieng:

```bash
conda create -n w2v2_aasist_anhhd python=3.10 -y
conda activate w2v2_aasist_anhhd
pip install "pip<24.1"
pip install torch==2.2.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<1.24" soundfile tqdm pandas scikit-learn tensorboardX librosa

cd /home/user14/anhhd/spoof
if [ ! -d fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 ]; then
  git clone https://github.com/pytorch/fairseq.git fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
fi
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
git checkout a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
```

## Chay evaluation

Chay tu root repo tren server:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate nes2net_anhhd

export W2V2_AASIST_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/w2v2_aasist/w2v2_aasist_asvspoof2019la.pth
export XLSR2_300M_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt
export W2V2_AASIST_EVAL_BATCH_SIZE=8
export W2V2_AASIST_EVAL_NUM_WORKERS=8

python main.py --baseline wav2vec2_aasist --mode eval --dataset asvspoof2019la --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset asvspoof5 --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset in_the_wild --ckpt "$W2V2_AASIST_CKPT_2019"
```

Dung alias ngan:

```bash
python main.py --baseline w2v2_aasist --mode eval --dataset asvspoof2019la --ckpt "$W2V2_AASIST_CKPT_2019"
```

## Output

Ket qua duoc luu trong:

```bash
outputs/wav2vec2_aasist/evals/<YYYY_MM_DD_HH_MM_SS>__<checkpoint>__on__<dataset>/
```

Moi folder co:

- `eval_output.txt`: score tung utterance, score cao hon nghia la nghieng ve `bonafide`.
- `eval_config.txt`: checkpoint, SSL model, batch size, num workers.
- `eval_EER.txt`: EER va threshold neu dataset co label bonafide/spoof.

## Dataset mac dinh

Adapter dung dataset registry chung trong `datasets/registry.py`. Cac dataset eval dang ho tro gom:

```bash
dfadd_test fake_or_real_norm vlsp2025 vsasv in_the_wild asvspoof2019la asvspoof2021la asvspoof2021df asvspoof5
```

Lenh chay theo thu tu tuong doi be -> lon:

```bash
python main.py --baseline wav2vec2_aasist --mode eval --dataset dfadd_test --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset fake_or_real_norm --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset vlsp2025 --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset vsasv --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset in_the_wild --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset asvspoof2019la --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset asvspoof2021la --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset asvspoof2021df --ckpt "$W2V2_AASIST_CKPT_2019"
python main.py --baseline wav2vec2_aasist --mode eval --dataset asvspoof5 --ckpt "$W2V2_AASIST_CKPT_2019"
```

Co the override root dataset tam thoi bang `SPOOF_DATA_ROOT` khi can test mot dataset khac cung format.
