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
export MOLEX_CKPT_2019LA_BEST=/home/user14/anhhd/spoof/BaselinesSpoofDetection/outputs/molex/2026_06_17_22_55_26/weights/epoch_22_0.040.pth
export MOLEX_CKPT_2019LA_AVG=/home/user14/anhhd/spoof/BaselinesSpoofDetection/outputs/molex/2026_06_17_22_55_26/weights/averaged_checkpoint.pth

# MoEF checkpoints. The MoEF inference script expects the run directory because it
# reads hyperparameters and the checkpoint file from there.
export MOEF_RUN_2019LA=${MOEF_RUN_2019LA:-/home/user14/anhhd/spoof/BaselinesSpoofDetection/outputs/moef/2026_06_21_18_11_31}
export MOEF_CKPT_2019LA=${MOEF_CKPT_2019LA:-"$MOEF_RUN_2019LA/checkpoints/best_model-epoch=31-dev_eer=6.0449-loss=0.0083.ckpt"}

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
export MOEF_EVAL_BATCH_SIZE=${MOEF_EVAL_BATCH_SIZE:-128}
export MOEF_EVAL_NUM_WORKERS=${MOEF_EVAL_NUM_WORKERS:-4}

reset_eval_env() {
  export PYTHONUNBUFFERED=1

  export XLSR2_300M_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr2_300m__s3prl__converted_ckpts/pytorch_model.bin
  export WAVLM_LARGE_PATH=/home/user14/anhhd/spoof/pretrained_ssl_models/wavlm_large__mrdragonfox__llase_g1/pytorch_model.bin
  export MOEF_WAV2VEC2_PATH=${MOEF_WAV2VEC2_PATH:-/home/user14/anhhd/spoof/pretrained_ssl_models/xlsr_300m}
  export MOEF_ASVSPOOF2019_LA_ROOT=${MOEF_ASVSPOOF2019_LA_ROOT:-/home/user14/anhhd/spoof/datasets/asvspoof2019/LA/LA}
  export MOEF_ASVSPOOF5_ROOT=${MOEF_ASVSPOOF5_ROOT:-/home/user14/anhhd/spoof/datasets/asvspoof5}
  export MOEF_IN_THE_WILD_ROOT=${MOEF_IN_THE_WILD_ROOT:-/home/user14/anhhd/spoof/datasets/in_the_wild/release_in_the_wild}
  export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

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
  export MOEF_EVAL_BATCH_SIZE=${MOEF_EVAL_BATCH_SIZE:-128}
  export MOEF_EVAL_NUM_WORKERS=${MOEF_EVAL_NUM_WORKERS:-4}
}

activate_env() {
  local env_name=$1
  set +u
  conda activate "$env_name"
  local status=$?
  set -u
  if [ "$status" -ne 0 ]; then
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
  reset_eval_env

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

run_moef_eval() {
  local env_name=$1
  local run_dir=$2
  local ckpt=$3
  local dataset=$4
  local tag=$5
  local log_file="$LOG_ROOT/${tag}__moef__${dataset}.log"
  local score_file
  local eer_file="$LOG_ROOT/${tag}__moef__${dataset}_eer.txt"

  if [ ! -d "$run_dir" ]; then
    echo "[SKIP] $tag moef $dataset: missing run directory $run_dir" | tee -a "$LOG_ROOT/skipped.log"
    return 0
  fi
  if [ ! -f "$ckpt" ]; then
    echo "[SKIP] $tag moef $dataset: missing checkpoint $ckpt" | tee -a "$LOG_ROOT/skipped.log"
    return 0
  fi

  activate_env "$env_name"
  reset_eval_env

  if [ ! -f "$run_dir/hparams.yaml" ] && [ -f "$run_dir/hyperparameters.yaml" ]; then
    cp "$run_dir/hyperparameters.yaml" "$run_dir/hparams.yaml"
  fi

  echo
  echo "================================================================================"
  echo "[RUN] env=$env_name baseline=moef dataset=$dataset tag=$tag"
  echo "[RUN_DIR] $run_dir"
  echo "[CKPT] $ckpt"
  echo "[LOG] $log_file"
  echo "================================================================================"

  (
    cd "$REPO_ROOT/baselines/moef_icassp"
    python main.py \
      --inference \
      --trained_model "$run_dir" \
      --dataset "$dataset" \
      --module_model models.moe_research.w2v2_moe_fz24_aasist \
      --tl_model models.tl_model_moe \
      --data_module utils.loadData.asvspoof_data_DA \
      --batch_size "$MOEF_EVAL_BATCH_SIZE" \
      --num_workers "$MOEF_EVAL_NUM_WORKERS" \
      --gpuid "${CUDA_VISIBLE_DEVICES:-0}" \
      --truncate 64600 \
      --moe_topk 2 \
      --moe_experts 4 \
      --moe_exp_hid 128
  ) 2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[FAIL] $tag moef $dataset status=$status" | tee -a "$LOG_ROOT/failed.log"
    return "$status"
  fi

  score_file=$(find "$run_dir" -type f -name infer_19.log -exec ls -t {} + 2>/dev/null | head -1 || true)
  if [ -z "$score_file" ]; then
    echo "[FAIL] $tag moef $dataset: infer_19.log not found under $run_dir" | tee -a "$LOG_ROOT/failed.log"
    return 1
  fi

  if [ "$dataset" = "asvspoof5" ]; then
    python - "$score_file" <<'PY' 2>&1 | tee "$eer_file"
import os
import sys
from pathlib import Path

import numpy as np

score_file = Path(sys.argv[1])
root = Path(os.environ.get("MOEF_ASVSPOOF5_ROOT", "/home/user14/anhhd/spoof/datasets/asvspoof5")).expanduser()
protocol_candidates = [
    root / "protocols" / "ASVspoof5.eval.track_1.tsv",
    root / "ASVspoof5_protocols" / "ASVspoof5.eval.track_1.tsv",
    root / "ASVspoof5.eval.track_1.tsv",
]
protocol = next((path for path in protocol_candidates if path.exists()), None)
if protocol is None:
    raise FileNotFoundError("ASVspoof5 eval protocol not found")

labels = {}
with open(protocol, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 9:
            labels[parts[1]] = 1 if parts[8] == "bonafide" else 0

y_true = []
y_score = []
with open(score_file, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0] in labels:
            y_true.append(labels[parts[0]])
            y_score.append(float(parts[2]))

labels_np = np.asarray(y_true)
scores_np = np.asarray(y_score)
bona = labels_np == 1
spoof = labels_np == 0
if not bona.any() or not spoof.any():
    raise RuntimeError("ASVspoof5 eval EER needs both bonafide and spoof scores")

order = np.argsort(scores_np)
sorted_labels = labels_np[order]
n_bona = bona.sum()
n_spoof = spoof.sum()
false_reject = np.cumsum(sorted_labels == 1) / n_bona
false_accept = (n_spoof - np.cumsum(sorted_labels == 0)) / n_spoof
idx = np.argmin(np.abs(false_reject - false_accept))
eer = float((false_reject[idx] + false_accept[idx]) / 2.0 * 100.0)
print(f"ASVspoof5 eval trials matched: {len(y_true)}")
print(f"CM SYSTEM")
print(f"   EER            = {eer:8.5f} % (Equal error rate for countermeasure)")
PY
  else
    (
      cd "$REPO_ROOT/baselines/moef_icassp"
      python utils/tools/cul_eer.py --pos 2 --scoreFile "$score_file"
    ) 2>&1 | tee "$eer_file"
  fi

  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "[FAIL] $tag moef $dataset eer status=$status" | tee -a "$LOG_ROOT/failed.log"
    return "$status"
  fi
  echo "[DONE] $tag moef $dataset score=$score_file eer=$eer_file" | tee -a "$LOG_ROOT/done.log"
}

run_moef_or_continue() {
  run_moef_eval "$@" || true
}

echo "[INFO] Logs: $LOG_ROOT"
echo "[INFO] XLSR2_300M_PATH=$XLSR2_300M_PATH"
echo "[INFO] WAVLM_LARGE_PATH=$WAVLM_LARGE_PATH"

# =============================================================================
# Active jobs for this run.
#
# Requested:
#   - MoEF Pretrained 2019LA on VLSP, DFADD, FoR, ITW, VSASV, ASV5.
#   - MoLEx Train ASVspoof5 best/averaged on ASVspoof2019LA.
#
# Order: smaller/faster datasets first; ASVspoof5 is kept last.
# =============================================================================

run_or_continue moef_cu113 moef vlsp2025       "$MOEF_CKPT_2019LA" moef_2019la_pretrained
run_or_continue moef_cu113 moef dfadd_test     "$MOEF_CKPT_2019LA" moef_2019la_pretrained
run_or_continue moef_cu113 moef fake_or_real   "$MOEF_CKPT_2019LA" moef_2019la_pretrained
run_or_continue moef_cu113 moef in_the_wild    "$MOEF_CKPT_2019LA" moef_2019la_pretrained

run_or_continue molex_anhhd molex asvspoof2019la "$MOLEX_CKPT_ASV5_BEST" molex_asv5_best
run_or_continue molex_anhhd molex asvspoof2019la "$MOLEX_CKPT_ASV5_AVG" molex_asv5_avg

run_or_continue moef_cu113 moef vsasv          "$MOEF_CKPT_2019LA" moef_2019la_pretrained
run_or_continue moef_cu113 moef asvspoof5      "$MOEF_CKPT_2019LA" moef_2019la_pretrained

# =============================================================================
# Commented jobs from previous runs. Keep the code for reproducibility, but do
# not execute any of these in the current run.
# =============================================================================

# MoEF trained on ASVspoof2019LA, already completed:
# run_moef_or_continue moef_cu113 "$MOEF_RUN_2019LA" "$MOEF_CKPT_2019LA" asvspoof2019la moef_2019la_best

# MoLEx Train 2019LA -- best, already completed:
# run_or_continue molex_anhhd molex vlsp2025       "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex dfadd_test     "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex fake_or_real   "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex in_the_wild    "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex asvspoof2019la "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex vsasv          "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best
# run_or_continue molex_anhhd molex asvspoof5      "$MOLEX_CKPT_2019LA_BEST" molex_2019la_best

# MoLEx Train 2019LA -- averaged, already completed:
# run_or_continue molex_anhhd molex vlsp2025       "$MOLEX_CKPT_2019LA_AVG" molex_2019la_avg
# run_or_continue molex_anhhd molex dfadd_test     "$MOLEX_CKPT_2019LA_AVG" molex_2019la_avg
# run_or_continue molex_anhhd molex fake_or_real   "$MOLEX_CKPT_2019LA_AVG" molex_2019la_avg
# run_or_continue molex_anhhd molex in_the_wild    "$MOLEX_CKPT_2019LA_AVG" molex_2019la_avg
# run_or_continue molex_anhhd molex asvspoof2019la "$MOLEX_CKPT_2019LA_AVG" molex_2019la_avg
# run_or_continue molex_anhhd molex vsasv          "$MOLEX_CKPT_2019LA_AVG" molex_2019la_avg
# run_or_continue molex_anhhd molex asvspoof5      "$MOLEX_CKPT_2019LA_AVG" molex_2019la_avg

# MoLEx Train ASVspoof5 rows not requested for this run:
# run_or_continue molex_anhhd molex vsasv     "$MOLEX_CKPT_ASV5_BEST" molex_asv5_best
# run_or_continue molex_anhhd molex asvspoof5 "$MOLEX_CKPT_ASV5_BEST" molex_asv5_best
# run_or_continue molex_anhhd molex in_the_wild "$MOLEX_CKPT_ASV5_AVG" molex_asv5_avg
# run_or_continue molex_anhhd molex vsasv       "$MOLEX_CKPT_ASV5_AVG" molex_asv5_avg
# run_or_continue molex_anhhd molex asvspoof5   "$MOLEX_CKPT_ASV5_AVG" molex_asv5_avg

# Strong SSL baselines:
# run_or_continue nes2net_anhhd xlsr_sls vlsp2025  "$XLSR_SLS_CKPT_2019" xlsr_sls_2019la
# run_or_continue nes2net_anhhd xlsr_sls vsasv     "$XLSR_SLS_CKPT_2019" xlsr_sls_2019la
# run_or_continue nes2net_anhhd xlsr_sls asvspoof5 "$XLSR_SLS_CKPT_2019" xlsr_sls_2019la
# run_or_continue nes2net_anhhd wav2vec2_aasist vlsp2025  "$W2V2_AASIST_CKPT_2019" w2v2_aasist_2019la
# run_or_continue nes2net_anhhd wav2vec2_aasist vsasv     "$W2V2_AASIST_CKPT_2019" w2v2_aasist_2019la
# run_or_continue nes2net_anhhd wav2vec2_aasist asvspoof5 "$W2V2_AASIST_CKPT_2019" w2v2_aasist_2019la
# run_or_continue nes2net_anhhd nes2net vlsp2025  "$NES2NET_CKPT_2019" nes2net_2019la
# run_or_continue nes2net_anhhd nes2net vsasv     "$NES2NET_CKPT_2019" nes2net_2019la
# run_or_continue nes2net_anhhd nes2net asvspoof5 "$NES2NET_CKPT_2019" nes2net_2019la
# run_or_continue nes2net_anhhd nes2net vlsp2025  "$NES2NET_CKPT_ASV5" nes2net_asv5
# run_or_continue nes2net_anhhd nes2net vsasv     "$NES2NET_CKPT_ASV5" nes2net_asv5
# run_or_continue nes2net_anhhd nes2net asvspoof5 "$NES2NET_CKPT_ASV5" nes2net_asv5

# AASIST family:
# run_or_continue nes2net_anhhd aasist vsasv     "$AASIST_CKPT_2019" aasist_2019la
# run_or_continue nes2net_anhhd aasist asvspoof5 "$AASIST_CKPT_2019" aasist_2019la
# run_or_continue nes2net_anhhd aasist vlsp2025       "$AASIST_CKPT_ASV5" aasist_asv5
# run_or_continue nes2net_anhhd aasist dfadd_test     "$AASIST_CKPT_ASV5" aasist_asv5
# run_or_continue nes2net_anhhd aasist fake_or_real   "$AASIST_CKPT_ASV5" aasist_asv5
# run_or_continue nes2net_anhhd aasist in_the_wild    "$AASIST_CKPT_ASV5" aasist_asv5
# run_or_continue nes2net_anhhd aasist asvspoof2019la "$AASIST_CKPT_ASV5" aasist_asv5
# run_or_continue nes2net_anhhd aasist vsasv          "$AASIST_CKPT_ASV5" aasist_asv5
# run_or_continue nes2net_anhhd aasist asvspoof5      "$AASIST_CKPT_ASV5" aasist_asv5
# run_or_continue nes2net_anhhd aasist_l vsasv     "$AASIST_L_CKPT_2019" aasist_l_2019la
# run_or_continue nes2net_anhhd aasist_l asvspoof5 "$AASIST_L_CKPT_2019" aasist_l_2019la

# RawTFNet:
# run_or_continue nes2net_anhhd rawtfnet vlsp2025       "$RAWTFNET_CKPT" rawtfnet_2019la
# run_or_continue nes2net_anhhd rawtfnet asvspoof2019la "$RAWTFNET_CKPT" rawtfnet_2019la
# run_or_continue nes2net_anhhd rawtfnet vsasv          "$RAWTFNET_CKPT" rawtfnet_2019la
# run_or_continue nes2net_anhhd rawtfnet asvspoof5      "$RAWTFNET_CKPT" rawtfnet_2019la

echo
echo "[INFO] Done. Logs: $LOG_ROOT"
echo "[INFO] Failed jobs, if any: $LOG_ROOT/failed.log"
echo "[INFO] Skipped jobs, if any: $LOG_ROOT/skipped.log"
