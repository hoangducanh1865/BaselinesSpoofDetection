#!/bin/bash -l
#SBATCH --job-name=moef_train
#SBATCH --partition=defq
#SBATCH --output=logs/moef/%x_%j.out
#SBATCH --error=logs/moef/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

REPO_DIR="/home/user14/anhhd/spoof/BaselinesSpoofDetection"
CONDA_ENV="${CONDA_ENV:-moef_cu113}"
GPU_ID="${GPU_ID:-0}"
MODULE_MODEL="${MODULE_MODEL:-w2v2_moe_fz24_aasist}"
DATASET="${DATASET:-asvspoof2019la}"
SAVEDIR="${SAVEDIR:-}"
RESUME="${RESUME:-}"

export MOEF_ASVSPOOF2019_LA_ROOT="${MOEF_ASVSPOOF2019_LA_ROOT:-/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA}"
export MOEF_ASVSPOOF5_ROOT="${MOEF_ASVSPOOF5_ROOT:-/home/user14/anhhd/spoof/datasets/asvspoof5}"
export MOEF_WAV2VEC2_PATH="${MOEF_WAV2VEC2_PATH:-/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr_300m}"
export MOEF_OUTPUT_ROOT="${MOEF_OUTPUT_ROOT:-${REPO_DIR}/outputs/moef}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${REPO_DIR}"
mkdir -p logs/moef

set +u
source /home/user14/miniconda3/bin/activate "${CONDA_ENV}"
set -u

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Conda env: ${CONDA_ENV}"
echo "MoEF module: ${MODULE_MODEL}"
echo "Dataset: ${DATASET}"
echo "Savedir: ${SAVEDIR:-auto}"
echo "Resume: ${RESUME:-disabled}"
echo "ASVspoof2019 LA root: ${MOEF_ASVSPOOF2019_LA_ROOT}"
echo "ASVspoof5 root: ${MOEF_ASVSPOOF5_ROOT}"
echo "wav2vec2 path: ${MOEF_WAV2VEC2_PATH}"
echo "Started at: $(date)"

cd "${REPO_DIR}/baselines/moef_icassp"
DATASET="${DATASET}" RESUME="${RESUME}" bash moe_run.sh "${GPU_ID}" "${MODULE_MODEL}" "${SAVEDIR}"

echo "Finished at: $(date)"
