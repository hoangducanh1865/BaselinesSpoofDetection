#!/bin/bash -l
#SBATCH --job-name=molex_eval
#SBATCH --partition=defq
#SBATCH --output=logs/molex_eval/%x_%j.out
#SBATCH --error=logs/molex_eval/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1

set -euo pipefail

REPO_DIR="/home/user14/anhhd/spoof/BaselinesSpoofDetection"
CONDA_ENV="${CONDA_ENV:-molex_anhhd}"
CONFIG="${CONFIG:-configs/molex.yaml}"
SCENARIO="${SCENARIO:-all}"
CKPT_ASV5="${CKPT_ASV5:-outputs/molex/2026_06_17_13_46_58/weights/epoch_1_1.708.pth}"
CKPT_2019="${CKPT_2019:-outputs/molex/2026_06_17_22_55_26/weights/epoch_8_0.071.pth}"

cd "${REPO_DIR}"
mkdir -p logs/molex_eval

# Some conda activation hooks reference unset MKL variables.
set +u
source /home/user14/miniconda3/bin/activate "${CONDA_ENV}"
set -u

PY="/home/user14/miniconda3/envs/${CONDA_ENV}/bin/python"
export MOLEX_EVAL_BATCH_SIZE="${MOLEX_EVAL_BATCH_SIZE:-128}"
export MOLEX_EVAL_NUM_WORKERS="${MOLEX_EVAL_NUM_WORKERS:-16}"

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Scenario: ${SCENARIO}"
echo "Config: ${CONFIG}"
echo "Eval batch size: ${MOLEX_EVAL_BATCH_SIZE}"
echo "Eval num workers: ${MOLEX_EVAL_NUM_WORKERS}"
echo "Started at: $(date)"

run_eval() {
  local label="$1"
  local dataset="$2"
  local ckpt="$3"

  echo "================================================================"
  echo "[${label}] dataset=${dataset}"
  echo "[${label}] checkpoint=${ckpt}"
  echo "================================================================"
  "${PY}" main.py \
    --baseline molex \
    --mode eval \
    --dataset "${dataset}" \
    --config "${CONFIG}" \
    --ckpt "${ckpt}" 2>&1 | tee "logs/molex_eval/${SLURM_JOB_ID}_${label}.log"
}

case "${SCENARIO}" in
  all)
    run_eval asv5_on_asvspoof5 asvspoof5 "${CKPT_ASV5}"
    run_eval asv5_on_asvspoof2019la asvspoof2019la "${CKPT_ASV5}"
    run_eval asv5_on_in_the_wild in_the_wild "${CKPT_ASV5}"
    run_eval asv2019_on_asvspoof2019la asvspoof2019la "${CKPT_2019}"
    run_eval asv2019_on_asvspoof5 asvspoof5 "${CKPT_2019}"
    run_eval asv2019_on_in_the_wild in_the_wild "${CKPT_2019}"
    ;;
  asv5_on_asvspoof5)
    run_eval "${SCENARIO}" asvspoof5 "${CKPT_ASV5}"
    ;;
  asv5_on_asvspoof2019la)
    run_eval "${SCENARIO}" asvspoof2019la "${CKPT_ASV5}"
    ;;
  asv5_on_in_the_wild)
    run_eval "${SCENARIO}" in_the_wild "${CKPT_ASV5}"
    ;;
  asv2019_on_asvspoof2019la)
    run_eval "${SCENARIO}" asvspoof2019la "${CKPT_2019}"
    ;;
  asv2019_on_asvspoof5)
    run_eval "${SCENARIO}" asvspoof5 "${CKPT_2019}"
    ;;
  asv2019_on_in_the_wild)
    run_eval "${SCENARIO}" in_the_wild "${CKPT_2019}"
    ;;
  *)
    echo "Unknown SCENARIO=${SCENARIO}" >&2
    exit 2
    ;;
esac

echo "Finished at: $(date)"
