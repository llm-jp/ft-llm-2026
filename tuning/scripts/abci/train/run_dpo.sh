#!/bin/bash

set -euxo pipefail

if [ $# -ne 4 ]; then
    >&2 echo "Usage: $0 <param-name> <num-nodes> <model-path> <seed>"
    >&2 echo "Example: $0 llmjp4_8b 2 /path/to/model 42"
    exit 1
fi

param_name=$1; shift
num_nodes=$1; shift
model_path=$1; shift
seed=$1; shift

qsub -l select=${num_nodes} \
  -v RTYPE=rt_HF,PARAM_NAME=${param_name},MODEL_PATH="${model_path}",SEED=${seed} \
  -o /dev/null -e /dev/null \
  -m n \
  scripts/abci/train/qsub_dpo.sh
