#!/bin/bash -l
#SBATCH --job-name=see_molex_eval
#SBATCH --partition=defq
#SBATCH --output=logs/see_molex_eval/%x_%j.out
#SBATCH --error=logs/see_molex_eval/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

REPO_DIR="/home/user14/anhhd/spoof/BaselinesSpoofDetection"
CONDA_ENV="${CONDA_ENV:-molex_anhhd}"
ABLATION="${ABLATION:-M2}"
DATASET="${DATASET:-asvspoof2019la}"
CONFIG="${CONFIG:-configs/see_molex.yaml}"
# Optional; when empty the adapter picks the latest checkpoint under
# outputs/see_molex/<ABLATION>/.../weights/averaged_checkpoint.pth.
CKPT="${CKPT:-}"

cd "${REPO_DIR}"
mkdir -p logs/see_molex_eval

# Some conda activation hooks reference unset MKL variables.
set +u
source /home/user14/miniconda3/bin/activate "${CONDA_ENV}"
set -u

export MOLEX_EVAL_BATCH_SIZE="${MOLEX_EVAL_BATCH_SIZE:-128}"
export MOLEX_EVAL_NUM_WORKERS="${MOLEX_EVAL_NUM_WORKERS:-16}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Ablation: ${ABLATION}"
echo "Dataset: ${DATASET}"
echo "Config: ${CONFIG}"
echo "Checkpoint: ${CKPT:-<latest>}"
echo "Started at: $(date)"

cmd=(
  python main.py
  --baseline see_molex
  --ablation "${ABLATION}"
  --mode eval
  --dataset "${DATASET}"
  --config "${CONFIG}"
)
if [[ -n "${CKPT}" ]]; then
  cmd+=(--ckpt "${CKPT}")
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo "Finished at: $(date)"
