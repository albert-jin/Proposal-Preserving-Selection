#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export AUTO3_ROOT="${AUTO3_ROOT:-$PWD/work}"
export HF_HOME="${HF_HOME:-$AUTO3_ROOT/cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONPATH="$PWD/scripts:$PWD/autores${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
checkpoint="${1:-$AUTO3_ROOT/checkpoints/teacher_bootstrap_v1/final}"
scope="${2:-quick}"
case "$scope" in
  quick) inputs=(math500_quick128.jsonl gsm8k_quick128.jsonl aime25_test.jsonl) ;;
  full) inputs=(math500_test.jsonl gsm8k_test.jsonl aime25_test.jsonl) ;;
  *) echo "Usage: bash run_paired_eval.sh [checkpoint] [quick|full]" >&2; exit 2 ;;
esac
for input in "${inputs[@]}"; do
  name="${input%.jsonl}"
  target="$AUTO3_ROOT/results/paired/$name"
  python autores/evaluate_local.py --checkpoint "$checkpoint" \
    --input "$AUTO3_ROOT/data/$input" --output "$target/s2t_local" \
    --generation-adapter-mode native
  python autores/evaluate_local.py --checkpoint "$checkpoint" \
    --input "$AUTO3_ROOT/data/$input" --output "$target/pps_local" \
    --generation-adapter-mode base_proposals
  python autores/compare_local.py --baseline "$target/s2t_local" \
    --method "$target/pps_local" --output "$target/comparison.json"
done
