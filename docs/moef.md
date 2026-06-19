# MoEF

MoEF nam o `baselines/moef_icassp`.

Ket luan sau khi doi chieu voi paper `stuff/reference_papers/moef/moef.pdf`:

- Nen dung `baselines/moef_icassp`, khong phai `baselines/moef`.
- `baselines/moef_icassp` co dung file MoE paper can: `models/moe_research/w2v2_moe_fz24_aasist.py`, `models/tl_model_moe.py`, `main_loss.py`, va script `moe_run.sh`.
- SSL front-end duoc dong bang; chi train MoE fusion va classifier.

## Setup khop paper

Paper `Mixture of Experts Fusion for Fake Audio Detection Using Frozen wav2vec 2.0` dung:

- SSL front-end: frozen wav2vec 2.0 / SSL hidden features.
- Feature fusion: 24 hidden features, gating bang last hidden state.
- Classifier: AASIST.
- Audio length: `64600` samples.
- Data augmentation: RawBoost algorithm 3.
- Optimizer: AdamW, `lr=1e-5`.
- Scheduler: cosine warmup, `num_warmup_steps=3`.
- Batch size: `4`.
- Epochs: `50`.
- Early stopping patience: `3`.
- MoE config: `topk=2`, `experts_per_layer=4`, `expert_hidden=128`.

## Pretrained SSL model

Theo setup server hien tai, MoEF dung local HuggingFace folder:

```text
/home/user14/anhhd/spoof/pretrained_ssl_models/wav2vec2_large_lv60
```

Folder nay can co:

```text
config.json
preprocessor_config.json
pytorch_model.bin
README.md
```

Export neu can override:

```bash
export MOEF_WAV2VEC2_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/wav2vec2_large_lv60
```

## Dataset tren server

Mac dinh:

```text
ASVspoof2019 LA: /home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
ASVspoof5:       /home/user14/anhhd/spoof/datasets/asvspoof5
```

Override neu can:

```bash
export MOEF_ASVSPOOF2019_LA_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
export MOEF_ASVSPOOF5_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof5
```

## WandB va checkpoint

MoEF da co co che tuong tu MoLEx:

- Moi run moi luu vao `outputs/moef/YYYY_MM_DD_HH_MM_SS`.
- Trong run dir co `hyperparameters.yaml`, `hyperparameters.json`, `wandb_run_id.txt`.
- Checkpoint best nam trong `outputs/moef/<run>/checkpoints/`.
- Trong khi train co them `latest_checkpoint_epoch_<N>.ckpt` de resume day du optimizer/scheduler.
- Khi train ket thuc binh thuong, file latest checkpoint se duoc xoa de tranh lan voi best checkpoint.
- Resume cung folder se dung lai `wandb_run_id.txt`, nen WandB tiep tuc log vao cung mot run.

Can co `.env` hoac bien moi truong:

```bash
export WANDB_API_KEY="..."
export WANDB_PROJECT=spoof-detection
```

Tat WandB:

```bash
export WANDB_MODE=disabled
```

## Tao moi truong

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection

conda create -n moef_anhhd python=3.8 -y
conda activate moef_anhhd

pip install -r baselines/moef_icassp/requirement.txt
pip install wandb python-dotenv
```

Kiem tra nhanh:

```bash
python - <<'PY'
import torch
import lightning
import transformers
import wandb
print("torch", torch.__version__)
print("lightning", lightning.__version__)
print("transformers", transformers.__version__)
print("wandb", wandb.__version__)
PY
```

## Train truc tiep tren server

Train ASVspoof2019 LA:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate moef_anhhd

export MOEF_WAV2VEC2_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/wav2vec2_large_lv60
export MOEF_ASVSPOOF2019_LA_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
export TOKENIZERS_PARALLELISM=false

cd baselines/moef_icassp
DATASET=asvspoof2019la bash moe_run.sh 0
```

Train ASVspoof5:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate moef_anhhd

export MOEF_WAV2VEC2_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/wav2vec2_large_lv60
export MOEF_ASVSPOOF5_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof5
export TOKENIZERS_PARALLELISM=false

cd baselines/moef_icassp
DATASET=asvspoof5 bash moe_run.sh 0
```

## Resume truc tiep

Resume folder cu the:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection/baselines/moef_icassp
conda activate moef_anhhd

DATASET=asvspoof5 RESUME=2026_06_19_22_10_00 bash moe_run.sh 0
```

Resume folder moi nhat:

```bash
DATASET=asvspoof5 RESUME=latest bash moe_run.sh 0
```

## Submit bang SLURM

Train ASVspoof2019 LA:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
mkdir -p logs/moef

DATASET=asvspoof2019la \
CONDA_ENV=moef_anhhd \
MOEF_WAV2VEC2_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/wav2vec2_large_lv60 \
MOEF_ASVSPOOF2019_LA_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA \
sbatch -w dgx01 bash/moef/submit-train-job.sh
```

Train ASVspoof5:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
mkdir -p logs/moef

DATASET=asvspoof5 \
CONDA_ENV=moef_anhhd \
MOEF_WAV2VEC2_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/wav2vec2_large_lv60 \
MOEF_ASVSPOOF5_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof5 \
sbatch -w dgx01 bash/moef/submit-train-job.sh
```

Resume SLURM:

```bash
DATASET=asvspoof5 \
RESUME=2026_06_19_22_10_00 \
CONDA_ENV=moef_anhhd \
sbatch -w dgx01 bash/moef/submit-train-job.sh
```

Resume folder moi nhat:

```bash
DATASET=asvspoof5 \
RESUME=latest \
CONDA_ENV=moef_anhhd \
sbatch -w dgx01 bash/moef/submit-train-job.sh
```

Theo doi:

```bash
squeue -u "$USER"
tail -f logs/moef/moef_train_<JOBID>.out
tail -f logs/moef/moef_train_<JOBID>.err
```

## Noi luu ket qua

```text
outputs/moef/<YYYY_MM_DD_HH_MM_SS>/
  hyperparameters.yaml
  hyperparameters.json
  wandb_run_id.txt
  checkpoints/
    best_model-epoch=...ckpt
```
