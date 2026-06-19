# MoEF

MoEF nam o `baselines/moef_icassp`.

Ket luan sau khi doi chieu voi paper `stuff/reference_papers/moef/moef.pdf`:

- Nen dung `baselines/moef_icassp`, khong phai `baselines/moef`.
- `baselines/moef` la repo/framework goc va README cua no con ghi can checkout branch `icassp`; mot so path/import van la placeholder.
- `baselines/moef_icassp` moi co dung cac file MoE paper can: `models/moe_research/w2v2_moe_fz24_aasist.py`, `models/tl_model_moe.py`, `main_loss.py`, checkpoint/log ICASSP mau va script `moe_run.sh`.

## Setup khop paper

Paper `Mixture of Experts Fusion for Fake Audio Detection Using Frozen wav2vec 2.0` dung:

- Train set: ASVspoof2019 LA train.
- Validation: ASVspoof2019 LA dev.
- SSL front-end: wav2vec 2.0 / XLS-R 300M, frozen.
- Feature fusion: 24 hidden features, gating bang last hidden state.
- Classifier: AASIST.
- Audio length: `64600` samples.
- Data augmentation: RawBoost algorithm 3.
- Optimizer: AdamW, `lr=1e-5` khi frozen SSL.
- Scheduler: cosine warmup, `num_warmup_steps=3`.
- Batch size: `4`.
- Epochs: `50`.
- Early stopping patience: `3`.
- MoE config: `topk=2`, `experts_per_layer=4`, `expert_hidden=128`.

Script da setup theo cac gia tri tren:

```text
baselines/moef_icassp/moe_run.sh
```

Model chuan paper la:

```text
models.moe_research.w2v2_moe_fz24_aasist
```

## Pretrained SSL model

MoEF dung `transformers.Wav2Vec2Model.from_pretrained`, vi vay can mot thu muc HuggingFace snapshot cua:

```text
facebook/wav2vec2-xls-r-300m
```

Luu y: file fairseq `/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m/xlsr2_300m.pt` khong dung truc tiep cho baseline nay.

Neu server chua co HF snapshot, tai ve nhu sau:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
conda activate moef_anhhd

python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/wav2vec2-xls-r-300m",
    local_dir="/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m_hf",
    local_dir_use_symlinks=False,
)
PY
```

Sau do export:

```bash
export MOEF_XLSR_HF_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m_hf
```

## Dataset tren server

Mac dinh code da tro toi:

```text
/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
```

Co the override bang:

```bash
export MOEF_ASVSPOOF2019_LA_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
```

Can co cac file:

```text
ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt
ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt
ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt
ASVspoof2019_LA_train/flac/*.flac
ASVspoof2019_LA_dev/flac/*.flac
ASVspoof2019_LA_eval/flac/*.flac
```

## Tao moi truong

Khuyen nghi tao env rieng de tranh va cham voi `molex_anhhd` / `nes2net_anhhd`:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection

conda create -n moef_anhhd python=3.8 -y
conda activate moef_anhhd

pip install -r baselines/moef_icassp/requirement.txt
```

Kiem tra nhanh:

```bash
python - <<'PY'
import torch
import lightning
import transformers
import soundfile
print("torch", torch.__version__)
print("lightning", lightning.__version__)
print("transformers", transformers.__version__)
PY
```

## Train truc tiep tren server

Chay trong `tmux` / `screen` neu khong submit job:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
conda activate moef_anhhd

export MOEF_ASVSPOOF2019_LA_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
export MOEF_XLSR_HF_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m_hf
export TOKENIZERS_PARALLELISM=false

cd baselines/moef_icassp
bash moe_run.sh 0 w2v2_moe_fz24_aasist a_log/w2v2_moe_fz24_aasist/2_4_128
```

Trong do:

- `0`: GPU id nhin thay trong process.
- `w2v2_moe_fz24_aasist`: model chuan paper, frozen XLS-R, 24 hidden features.
- `a_log/w2v2_moe_fz24_aasist/2_4_128`: folder log/checkpoint.

## Submit bang SLURM

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
git pull
mkdir -p logs/moef

CONDA_ENV=moef_anhhd \
MOEF_ASVSPOOF2019_LA_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA \
MOEF_XLSR_HF_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m_hf \
sbatch -w dgx01 bash/moef/submit-train-job.sh
```

Theo doi:

```bash
squeue -u "$USER"
ls -lah logs/moef
tail -f logs/moef/moef_train_<JOBID>.out
tail -f logs/moef/moef_train_<JOBID>.err
```

## Noi luu checkpoint

Voi lenh mac dinh, Lightning se luu vao:

```text
baselines/moef_icassp/a_log/w2v2_moe_fz24_aasist/2_4_128/version_*/
  hparams.yaml
  events.out.tfevents.*
  checkpoints/
    best_model-epoch=...ckpt
```

Checkpoint duoc monitor theo `loss` vi script `main_loss.py` cua branch ICASSP dung training loss de early-stop/save best, phu hop voi mo ta trong paper: model co training loss thap nhat duoc chon de evaluation.

## Eval sau khi train

Chay inference ASVspoof2019 LA eval:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection/baselines/moef_icassp
conda activate moef_anhhd

export MOEF_ASVSPOOF2019_LA_ROOT=/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA
export MOEF_XLSR_HF_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m_hf

bash infer.sh a_log/w2v2_moe_fz24_aasist/2_4_128/version_0 0
bash z_cul_eer.sh a_log/w2v2_moe_fz24_aasist/2_4_128/version_0
```

Ket qua EER se nam trong:

```text
baselines/moef_icassp/a_log/w2v2_moe_fz24_aasist/2_4_128/version_0/eer_19
```
