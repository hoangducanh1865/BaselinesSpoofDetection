# XLSR-SLS

Baseline XLSR-SLS dung XLS-R 300M lam SSL front-end va bo phan loai Sensitive Layer Selection. Adapter da duoc noi vao CLI chung voi 2 ten baseline:

- `xlsr_sls`
- `sls`

Checkpoint pretrained tren ASVspoof2019 LA dang duoc dat tren server tai:

```bash
/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/xlsr_sls/xlsr_sls_asvspoof2019la.pth
```

Model SSL can dung:

```bash
/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt
```

## Moi truong

Baseline nay can `fairseq`. Co the dung env `nes2net_anhhd` neu env do da cai `torch`, `fairseq` va `numpy<1.24`.

Kiem tra nhanh:

```bash
conda activate nes2net_anhhd
python - <<'PY'
import torch, fairseq, numpy
print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("fairseq ok")
PY
```

Neu can tao env rieng:

```bash
conda create -n xlsr_sls_anhhd python=3.10 -y
conda activate xlsr_sls_anhhd
pip install "pip<24.1"
pip install torch==2.2.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<1.24" soundfile tqdm pandas scikit-learn

cd /home/user14/anhhd/spoof
if [ ! -d fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1 ]; then
  git clone https://github.com/pytorch/fairseq.git fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
fi
cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
git checkout a54021305d6b3c4c5959ac9395135f63202db8f1
pip install --editable ./
```

Khuyen nghi khong cai/downgrade `numpy<1.24` vao env `molex_anhhd` neu env do dang dung de train MoLEx.

## Chay evaluation

Chay tu root repo tren server:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate nes2net_anhhd

export XLSR_SLS_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/xlsr_sls/xlsr_sls_asvspoof2019la.pth
export XLSR2_300M_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt
export XLSR_SLS_EVAL_BATCH_SIZE=8
export XLSR_SLS_EVAL_NUM_WORKERS=8

python main.py --baseline xlsr_sls --mode eval --dataset asvspoof2019la --ckpt "$XLSR_SLS_CKPT_2019"
python main.py --baseline xlsr_sls --mode eval --dataset asvspoof5 --ckpt "$XLSR_SLS_CKPT_2019"
python main.py --baseline xlsr_sls --mode eval --dataset in_the_wild --ckpt "$XLSR_SLS_CKPT_2019"
```

Neu muon dung alias ngan:

```bash
python main.py --baseline sls --mode eval --dataset asvspoof2019la --ckpt "$XLSR_SLS_CKPT_2019"
```

## Output

Ket qua duoc luu trong:

```bash
outputs/xlsr_sls/evals/<YYYY_MM_DD_HH_MM_SS>__<checkpoint>__on__<dataset>/
```

Moi folder co:

- `eval_output.txt`: score tung utterance, score cao hon nghia la nghieng ve `bonafide`.
- `eval_config.txt`: checkpoint, SSL model, batch size, num workers.
- `eval_EER.txt`: EER va threshold neu dataset co label bonafide/spoof.

## Dataset mac dinh

Adapter dang dung cac duong dan mac dinh:

```bash
asvspoof5      -> /home/user14/anhhd/spoof/datasets/asvspoof5
asvspoof2019la -> /home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
asvspoof2019pa -> /home/user14/anhhd/spoof/datasets/asvspoof2019/PA/PA
in_the_wild    -> /home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild
```

Co the override root dataset tam thoi bang `SPOOF_DATA_ROOT` khi can test mot dataset khac cung format.
