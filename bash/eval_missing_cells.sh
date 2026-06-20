#!/usr/bin/env bash
# Run missing evaluation cells for the Phase 1 comparison table.
#
# Run on the server:
#   cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
#   bash bash/eval_missing_cells.sh
#
# The order is baseline-priority first, then lighter datasets before larger ones.
# ASVspoof2021 datasets are intentionally omitted.

set -u

REPO_ROOT=${REPO_ROOT:-/home/user14/anhhd/spoof/BaselinesSpoofDetection}
LOG_ROOT=${LOG_ROOT:-"$REPO_ROOT/outputs/eval_missing_logs/$(date +%Y_%m_%d_%H_%M_%S)"}
mkdir -p "$LOG_ROOT"

cd "$REPO_ROOT"

init_conda() {
  if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "/home/user14/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "/home/user14/miniconda3/etc/profile.d/conda.sh"
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "[FATAL] conda is not available." >&2
    exit 1
  fi
}

init_conda

export PYTHONUNBUFFERED=1

# Shared SSL checkpoints. Set explicitly to avoid empty env fallback.
export XLSR2_300M_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m__s3prl__converted_ckpts/pytorch_model.bin
export WAVLM_LARGE_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large__mrdragonfox__llase_g1/pytorch_model.bin

# MoLEx checkpoints.
export MOLEX_CKPT_ASV5_BEST=/home/user14/anhhd/spoof/BaselinesSpoofDetection/outputs/molex/2026_06_17_13_46_58/weights/epoch_18_0.728.pth
export MOLEX_CKPT_ASV5_AVG=/home/user14/anhhd/spoof/BaselinesSpoofDetection/outputs/molex/2026_06_17_13_46_58/weights/averaged_checkpoint.pth
# Kept consistent with the existing report row values.
export MOLEX_CKPT_2019LA_BEST=/home/user14/anhhd/spoof/BaselinesSpoofDetection/outputs/molex/2026_06_17_22_55_26/weights/epoch_8_0.071.pth

# Pretrained checkpoints.
export AASIST_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/aasist/aasist_asvspoof2019la.pth
export AASIST_L_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/aasist_l/aasist_l_asvspoof2019la.pth
export XLSR_SLS_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/xlsr_sls/xlsr_sls_asvspoof2019la.pth
export W2V2_AASIST_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/w2v2_aasist/w2v2_aasist_asvspoof2019la.pth
export NES2NET_CKPT_2019=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/nes2net/nes2net_asvspoof2019la.pth
export NES2NET_CKPT_ASV5=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof5/nes2net/nes2net_asvspoof5.pth
export RAWTFNET_CKPT=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof2019la/rawtfnet/Best_RawTFNet_32.pth

# Optional row in the table. If this checkpoint is absent, tasks will be skipped.
export AASIST_CKPT_ASV5=/home/user14/anhhd/spoof/pretrained_spoof_models/trained_on_asvspoof5/aasist/aasist_asvspoof5.pth

# Conservative defaults. Override before running if needed.
export MOLEX_EVAL_BATCH_SIZE=${MOLEX_EVAL_BATCH_SIZE:-128}
export MOLEX_EVAL_NUM_WORKERS=${MOLEX_EVAL_NUM_WORKERS:-16}
export AASIST_EVAL_BATCH_SIZE=${AASIST_EVAL_BATCH_SIZE:-128}
export AASIST_EVAL_NUM_WORKERS=${AASIST_EVAL_NUM_WORKERS:-16}
export XLSR_SLS_EVAL_BATCH_SIZE=${XLSR_SLS_EVAL_BATCH_SIZE:-8}
export XLSR_SLS_EVAL_NUM_WORKERS=${XLSR_SLS_EVAL_NUM_WORKERS:-8}
export W2V2_AASIST_EVAL_BATCH_SIZE=${W2V2_AASIST_EVAL_BATCH_SIZE:-8}
export W2V2_AASIST_EVAL_NUM_WORKERS=${W2V2_AASIST_EVAL_NUM_WORKERS:-8}
export NES2NET_EVAL_BATCH_SIZE=${NES2NET_EVAL_BATCH_SIZE:-16}
export NES2NET_EVAL_NUM_WORKERS=${NES2NET_EVAL_NUM_WORKERS:-8}
export RAWTFNET_EVAL_BATCH_SIZE=${RAWTFNET_EVAL_BATCH_SIZE:-128}
export RAWTFNET_EVAL_NUM_WORKERS=${RAWTFNET_EVAL_NUM_WORKERS:-16}

activate_env() {
  local env_name=$1
  if ! conda activate "$env_name"; then
    echo "[FATAL] Could not activate conda env: $env_name" >&2
    exit 1
  fi
}

run_eval() {
  local env_name=$1
  local baseline=$2
  local dataset=$3
  local ckpt=$4
  local tag=$5
  local log_file="$LOG_ROOT/${tag}__${baseline}__${dataset}.log"

  if [ -z "$ckpt" ]; then
    echo "[SKIP] $tag $baseline $dataset: empty checkpoint path" | tee -a "$LOG_ROOT/skipped.log"
    return 0
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[SKIP] $tag $baseline $dataset: missing checkpoint $ckpt" | tee -a "$LOG_ROOT/skipped.log"
    return 0
  fi

  activate_env "$env_name"

  echo
  echo "================================================================================"
  echo "[RUN] env=$env_name baseline=$baseline dataset=$dataset tag=$tag"
  echo "[CKPT] $ckpt"
  echo "[LOG] $log_file"
  echo "================================================================================"

  EVAL_BASELINE="$baseline" EVAL_DATASET="$dataset" EVAL_CKPT="$ckpt" \
  python -c "
import os
import runpy
import sys
import torch

torch.backends.cudnn.enabled = False

sys.argv = [
    'main.py',
    '--baseline', os.environ['EVAL_BASELINE'],
    '--mode', 'eval',
    '--dataset', os.environ['EVAL_DATASET'],
    '--ckpt', os.environ['EVAL_CKPT'],
]
print('sys.argv =', sys.argv, flush=True)
runpy.run_path('main.py', run_name='__main__')
" 2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[FAIL] $tag $baseline $dataset status=$status" | tee -a "$LOG_ROOT/failed.log"
    return "$status"
  fi
  echo "[DONE] $tag $baseline $dataset" | tee -a "$LOG_ROOT/done.log"
}

run_or_continue() {
  run_eval "$@" || true
}

echo "[INFO] Logs: $LOG_ROOT"
echo "[INFO] XLSR2_300M_PATH=$XLSR2_300M_PATH"
echo "[INFO] WAVLM_LARGE_PATH=$WAVLM_LARGE_PATH"

# =============================================================================
# 1) MoLEx: main baseline. Missing cells only.
# =============================================================================

# Train ASVspoof5 -- best: missing 2019LA, VSASV, ASV5.
run_or_continue molex_anhhd molex asvspoof2019la "$MOLEX_CKPT_ASV5_BEST" molex_asv5_best
run_or_continue molex_anhhd molex vsasv          "$MOLEX_CKPT_ASV5_BEST" molex_asv5_best
run_or_continue molex_anhhd molex asvspoof5      "$MOLEX_CKPT_ASV5_BEST" molex_asv5_best

# Train ASVspoof5 -- averaged: missing ITW, 2019LA, VSASV, ASV5.
run_or_continue molex_anhhd molex in_the_wild    "$MOLEX_CKPT_ASV5_AVG" molex_asv5_avg
run_or_continue molex_anhhd molex asvspoof2019la "$MOLEX_CKPT_ASV5_AVG" molex_asv5_avg
run_or_continue molex_anhhd molex vsasv          "$MOLEX_CKPT_ASV5_AVG" molex_asv5_avg
run_or_continue molex_anhhd molex asvspoof5      "$MOLEX_CKPT_ASV5_AVG" molex_asv5_avg

# MoLEx train 2019LA is intentionally paused for now.
# Uncomment when those missing cells are needed:
# run_or_continue molex_anhhd molex vlsp2025       "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex dfadd_test     "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex fake_or_real   "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex vsasv          "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best

# =============================================================================
# 2) Strong SSL baselines: XLSR-SLS, W2V2-AASIST, Nes2Net.
# =============================================================================

# XLSR-SLS pretrained 2019LA: missing VLSP, VSASV, ASV5.
run_or_continue nes2net_anhhd xlsr_sls vlsp2025  "$XLSR_SLS_CKPT_2019" xlsr_sls_2019la
run_or_continue nes2net_anhhd xlsr_sls vsasv     "$XLSR_SLS_CKPT_2019" xlsr_sls_2019la
run_or_continue nes2net_anhhd xlsr_sls asvspoof5 "$XLSR_SLS_CKPT_2019" xlsr_sls_2019la

# W2V2-AASIST pretrained 2019LA: missing VLSP, VSASV, ASV5.
run_or_continue nes2net_anhhd wav2vec2_aasist vlsp2025  "$W2V2_AASIST_CKPT_2019" w2v2_aasist_2019la
run_or_continue nes2net_anhhd wav2vec2_aasist vsasv     "$W2V2_AASIST_CKPT_2019" w2v2_aasist_2019la
run_or_continue nes2net_anhhd wav2vec2_aasist asvspoof5 "$W2V2_AASIST_CKPT_2019" w2v2_aasist_2019la

# Nes2Net pretrained 2019LA: missing VLSP, VSASV, ASV5.
run_or_continue nes2net_anhhd nes2net vlsp2025  "$NES2NET_CKPT_2019" nes2net_2019la
run_or_continue nes2net_anhhd nes2net vsasv     "$NES2NET_CKPT_2019" nes2net_2019la
run_or_continue nes2net_anhhd nes2net asvspoof5 "$NES2NET_CKPT_2019" nes2net_2019la

# Nes2Net pretrained ASV5: missing VLSP, VSASV, ASV5.
run_or_continue nes2net_anhhd nes2net vlsp2025  "$NES2NET_CKPT_ASV5" nes2net_asv5
run_or_continue nes2net_anhhd nes2net vsasv     "$NES2NET_CKPT_ASV5" nes2net_asv5
run_or_continue nes2net_anhhd nes2net asvspoof5 "$NES2NET_CKPT_ASV5" nes2net_asv5

# =============================================================================
# 3) AASIST family.
# =============================================================================

# AASIST pretrained 2019LA: missing VSASV, ASV5.
run_or_continue nes2net_anhhd aasist vsasv     "$AASIST_CKPT_2019" aasist_2019la
run_or_continue nes2net_anhhd aasist asvspoof5 "$AASIST_CKPT_2019" aasist_2019la

# AASIST pretrained ASV5: all table cells are missing. This skips if checkpoint absent.
run_or_continue nes2net_anhhd aasist vlsp2025       "$AASIST_CKPT_ASV5" aasist_asv5
run_or_continue nes2net_anhhd aasist dfadd_test     "$AASIST_CKPT_ASV5" aasist_asv5
run_or_continue nes2net_anhhd aasist fake_or_real   "$AASIST_CKPT_ASV5" aasist_asv5
run_or_continue nes2net_anhhd aasist in_the_wild    "$AASIST_CKPT_ASV5" aasist_asv5
run_or_continue nes2net_anhhd aasist asvspoof2019la "$AASIST_CKPT_ASV5" aasist_asv5
run_or_continue nes2net_anhhd aasist vsasv          "$AASIST_CKPT_ASV5" aasist_asv5
run_or_continue nes2net_anhhd aasist asvspoof5      "$AASIST_CKPT_ASV5" aasist_asv5

# AASIST-L pretrained 2019LA: missing VSASV, ASV5.
run_or_continue nes2net_anhhd aasist_l vsasv     "$AASIST_L_CKPT_2019" aasist_l_2019la
run_or_continue nes2net_anhhd aasist_l asvspoof5 "$AASIST_L_CKPT_2019" aasist_l_2019la

# =============================================================================
# 4) RawTFNet.
# =============================================================================

# RawTFNet pretrained 2019LA: missing VLSP, 2019LA, VSASV, ASV5.
run_or_continue nes2net_anhhd rawtfnet vlsp2025       "$RAWTFNET_CKPT" rawtfnet_2019la
run_or_continue nes2net_anhhd rawtfnet asvspoof2019la "$RAWTFNET_CKPT" rawtfnet_2019la
run_or_continue nes2net_anhhd rawtfnet vsasv          "$RAWTFNET_CKPT" rawtfnet_2019la
run_or_continue nes2net_anhhd rawtfnet asvspoof5      "$RAWTFNET_CKPT" rawtfnet_2019la

echo
echo "[INFO] Done. Logs: $LOG_ROOT"
echo "[INFO] Failed jobs, if any: $LOG_ROOT/failed.log"
echo "[INFO] Skipped jobs, if any: $LOG_ROOT/skipped.log"
