#!/bin/bash -l
#SBATCH --job-name=molex_train
#SBATCH --partition=defq
#SBATCH --output=logs/molex/%x_%j.out
#SBATCH --error=logs/molex/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

REPO_DIR="/home/user14/anhhd/spoof/BaselinesSpoofDetection"
CONDA_ENV="${CONDA_ENV:-molex_anhhd}"
DATASET="${DATASET:-asvspoof2019la}"
CONFIG="${CONFIG:-configs/molex.yaml}"
RESUME="${RESUME:-}"

cd "${REPO_DIR}"
mkdir -p logs/molex

# Some conda activation hooks reference unset MKL variables; keep strict mode
# for the job itself, but allow conda's shell hooks to initialize normally.
set +u
source /home/user14/miniconda3/bin/activate "${CONDA_ENV}"
set -u

export MOLEX_NUM_GPU="${MOLEX_NUM_GPU:-1}"
export WANDB_LOG_INTERVAL="${WANDB_LOG_INTERVAL:-100}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Dataset: ${DATASET}"
echo "Config: ${CONFIG}"
echo "Resume: ${RESUME:-disabled}"
echo "Started at: $(date)"

cmd=(
  python main.py
  --baseline molex
  --dataset "${DATASET}"
  --mode train
  --config "${CONFIG}"
)

if [[ -n "${RESUME}" ]]; then
  if [[ "${RESUME}" == "1" || "${RESUME}" == "true" || "${RESUME}" == "latest" ]]; then
    cmd+=(--resume)
  else
    cmd+=(--resume "${RESUME}")
  fi
fi

echo "Command: ${cmd[*]}"
"${cmd[@]}"

echo "Finished at: $(date)"
