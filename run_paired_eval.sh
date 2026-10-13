#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PPS_ROOT="${PPS_ROOT:-$PWD/work}"
export HF_HOME="${HF_HOME:-$PPS_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/scripts:$PWD/pps${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
checkpoint="${1:-$PPS_ROOT/checkpoints/teacher_bootstrap_v1/final}"
scope="${2:-full}"
case "$scope" in
  quick) inputs=(math500_quick128.jsonl gsm8k_quick128.jsonl aime25_test.jsonl) ;;
  full) inputs=(math500_test.jsonl gsm8k_test.jsonl aime25_test.jsonl) ;;
  *) echo "Usage: bash run_paired_eval.sh [checkpoint] [quick|full]" >&2; exit 2 ;;
esac
for input in "${inputs[@]}"; do
  name="${input%.jsonl}"
  target="$PPS_ROOT/results/paired/$name"
  python pps/evaluate_local.py --checkpoint "$checkpoint" \
    --input "$PPS_ROOT/data/$input" --output "$target/s2t_local" \
    --generation-adapter-mode native
  python pps/evaluate_local.py --checkpoint "$checkpoint" \
    --input "$PPS_ROOT/data/$input" --output "$target/pps_local" \
    --generation-adapter-mode base_proposals
  python pps/compare_local.py --baseline "$target/s2t_local" \
    --method "$target/pps_local" --output "$target/comparison.json"
done
