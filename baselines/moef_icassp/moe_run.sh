#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "${script_dir}/../.." && pwd)"
cd "${script_dir}"

gpu="${1:-0}"
module_model="${2:-w2v2_moe_fz24_aasist}"
savedir="${3:-}"
dataset="${DATASET:-${4:-asvspoof2019la}}"
output_root="${MOEF_OUTPUT_ROOT:-${repo_dir}/outputs/moef}"
resume="${RESUME:-}"
resume_args=()

if [[ -n "${resume}" ]]; then
  if [[ "${resume}" == "1" || "${resume}" == "true" || "${resume}" == "latest" ]]; then
    savedir="$(find "${output_root}" -maxdepth 1 -type d -name '20??_??_??_??_??_??' 2>/dev/null | sort | tail -1 || true)"
    if [[ -z "${savedir}" ]]; then
      echo "[moef] No previous run directory found under ${output_root}" >&2
      exit 1
    fi
    resume_args+=(--resume)
  elif [[ -d "${output_root}/${resume}" ]]; then
    savedir="${output_root}/${resume}"
    resume_args+=(--resume)
  elif [[ -d "${resume}" ]]; then
    savedir="${resume}"
    resume_args+=(--resume)
  else
    resume_args+=(--resume "${resume}")
  fi
fi

if [[ -z "${savedir}" ]]; then
  savedir="${output_root}/$(date +%Y_%m_%d_%H_%M_%S)"
fi

mkdir -p "$(dirname "${savedir}")" b_gpu_log

echo "[moef] Dataset: ${dataset}"
echo "[moef] Run directory: ${savedir}"
echo "[moef] Resume: ${resume:-disabled}"

python main_loss.py \
  --seed 888 \
  --dataset "${dataset}" \
  --module_model "models.moe_research.${module_model}" \
  --tl_model models.tl_model_moe \
  --data_module utils.loadData.asvspoof_data_DA \
  --savedir "${savedir}" \
  --optim_lr 0.00001 \
  --gpuid "${gpu}" \
  --batch_size 4 \
  --epochs 50 \
  --no_best_epochs 3 \
  --optim adamw \
  --weight_decay 0.0001 \
  --loss WCE \
  --scheduler cosWarmup \
  --num_warmup_steps 3 \
  --truncate 64600 \
  --moe_topk 2 \
  --moe_experts 4 \
  --moe_exp_hid 128 \
  --loss_weight 0 \
  --usingDA \
  "${resume_args[@]}"
