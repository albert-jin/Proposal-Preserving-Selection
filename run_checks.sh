#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/scripts:$PWD/autores${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=""
mkdir -p work/checks
python autores/check_split_adapter.py --output work/checks/split_adapter.json
python autores/check_memory_objective.py --output work/checks/memory_objective.json
python autores/check_training_integration.py --output work/checks/training_integration.json
python autores/check_training_integration.py --tied-embeddings --output work/checks/training_integration_tied.json
