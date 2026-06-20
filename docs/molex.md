# MoLEx

MoLEx nam o `baselines/molex`. Baseline nay fine-tune WavLM-Large bang Mixture-of-LoRA-Experts cho bai toan audio deepfake / spoof detection.

## Moi truong tren server

Moi truong dang dung on dinh:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate molex_anhhd
```

Checkpoint SSL can co:

```text
/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large__mrdragonfox__llase_g1/pytorch_model.bin
```

Duong dan nay da duoc cau hinh trong `configs/molex.yaml`:

```yaml
model_config:
  wavlm_checkpoint: /home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large__mrdragonfox__llase_g1/pytorch_model.bin
```

## Dataset dang ho tro

`configs/molex.yaml` dang tro toi cac dataset train chinh:

```text
asvspoof5:      /home/user14/anhhd/spoof/datasets/asvspoof5
asvspoof2019la: /home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
asvspoof2019pa: /home/user14/anhhd/spoof/datasets/asvspoof2019/PA/PA
in_the_wild:    /home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild
```

Khi eval cac dataset ngoai YAML, adapter fallback sang `datasets/registry.py`, hien gom:

```text
asvspoof2021la asvspoof2021df asvspoof2021pa dfadd_test fake_or_real_norm vlsp2025 vsasv
```

Adapter se tu tao metadata trung gian tai:

```text
outputs/molex/meta/<dataset>/
  fold1_train.tsv
  fold1_validation.tsv
  fold1_evaluation.tsv
  wav.scp
```

Voi `in_the_wild`, chi co evaluation split.

## Train full tren ASVspoof5

Chay truc tiep:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate molex_anhhd

python main.py \
  --baseline molex \
  --dataset asvspoof5 \
  --mode train \
  --config configs/molex.yaml
```

Chay qua SLURM script:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
mkdir -p logs/molex
DATASET=asvspoof5 CONFIG=configs/molex.yaml sbatch -w dgx01 bash/molex/submit-job.sh
```

## Train tren ASVspoof2019 LA

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate molex_anhhd

python main.py \
  --baseline molex \
  --dataset asvspoof2019la \
  --mode train \
  --config configs/molex.yaml
```

Hoac qua SLURM:

```bash
DATASET=asvspoof2019la CONFIG=configs/molex.yaml sbatch -w dgx01 bash/molex/submit-job.sh
```

## Resume training

Moi run duoc luu theo timestamp:

```text
outputs/molex/YYYY_MM_DD_HH_MM_SS/
```

Resume tu run moi nhat:

```bash
python main.py \
  --baseline molex \
  --dataset asvspoof5 \
  --mode train \
  --config configs/molex.yaml \
  --resume
```

Resume tu mot folder cu the:

```bash
python main.py \
  --baseline molex \
  --dataset asvspoof5 \
  --mode train \
  --config configs/molex.yaml \
  --resume 2026_06_17_13_46_58
```

Qua SLURM:

```bash
DATASET=asvspoof5 RESUME=2026_06_17_13_46_58 sbatch -w dgx01 bash/molex/submit-job.sh
```

Co che resume se doc checkpoint moi nhat trong run folder va train tiep tu epoch sau checkpoint do.

## Smoke test / debug nhanh

Chay it step de kiem tra pipeline:

```bash
python main.py \
  --baseline molex \
  --dataset asvspoof5 \
  --mode train \
  --config configs/molex.yaml \
  --max-steps 32
```

## Evaluation

Dat checkpoint:

```bash
CKPT_ASV5=outputs/molex/2026_06_17_13_46_58/weights/epoch_1_1.708.pth
CKPT_2019=outputs/molex/2026_06_17_22_55_26/weights/epoch_8_0.071.pth
```

Eval checkpoint train tren ASVspoof5:

```bash
python main.py --baseline molex --mode eval --dataset asvspoof5      --config configs/molex.yaml --ckpt "$CKPT_ASV5"
python main.py --baseline molex --mode eval --dataset asvspoof2019la --config configs/molex.yaml --ckpt "$CKPT_ASV5"
python main.py --baseline molex --mode eval --dataset in_the_wild    --config configs/molex.yaml --ckpt "$CKPT_ASV5"
```

Eval checkpoint train tren ASVspoof2019 LA:

```bash
python main.py --baseline molex --mode eval --dataset asvspoof2019la --config configs/molex.yaml --ckpt "$CKPT_2019"
python main.py --baseline molex --mode eval --dataset asvspoof5      --config configs/molex.yaml --ckpt "$CKPT_2019"
python main.py --baseline molex --mode eval --dataset in_the_wild    --config configs/molex.yaml --ckpt "$CKPT_2019"
```

Eval them cac dataset moi:

```bash
python main.py --baseline molex --mode eval --dataset dfadd_test       --config configs/molex.yaml --ckpt "$CKPT_2019"
python main.py --baseline molex --mode eval --dataset fake_or_real_norm --config configs/molex.yaml --ckpt "$CKPT_2019"
python main.py --baseline molex --mode eval --dataset vlsp2025         --config configs/molex.yaml --ckpt "$CKPT_2019"
python main.py --baseline molex --mode eval --dataset vsasv            --config configs/molex.yaml --ckpt "$CKPT_2019"
python main.py --baseline molex --mode eval --dataset asvspoof2021la   --config configs/molex.yaml --ckpt "$CKPT_2019"
python main.py --baseline molex --mode eval --dataset asvspoof2021df   --config configs/molex.yaml --ckpt "$CKPT_2019"
```

Neu eval ASVspoof5 qua lau, co the tang batch eval neu con VRAM:

```bash
export MOLEX_EVAL_BATCH_SIZE=128
export MOLEX_EVAL_NUM_WORKERS=16
```

## Noi luu ket qua

Training:

```text
outputs/molex/YYYY_MM_DD_HH_MM_SS/
  hyperparameters.json
  training_log.txt
  weights/
  metrics/
```

Evaluation:

```text
outputs/molex/evals/YYYY_MM_DD_HH_MM_SS__<run_folder>__on__<dataset>/
  eval_output.txt
  eval_EER.txt
  eval_config.txt
```

Checkpoint trong `weights/` co dang:

```text
epoch_<epoch>_<dev_eer>.pth
```

Vi du:

```text
epoch_1_1.708.pth
```

nghia la checkpoint tai epoch 1 voi dev EER = 1.708%.

## W&B

Neu can log len Weights & Biases, tao file `.env` tai repo root:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
printf 'WANDB_API_KEY="%s"\n' '<YOUR_WANDB_API_KEY>' > .env
```

Training resume se tiep tuc log vao cung run W&B neu run folder da co thong tin W&B tu truoc.
