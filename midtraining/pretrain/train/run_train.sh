#!/bin/bash

set -eu -o pipefail

if [ $# -ne 3 ]; then
    >&2 echo "Usage: $0 <ENV_DIR> <TASK_DIR> <NUM_NODES>"
    >&2 echo "Example: $0 /path/to/env_dir /path/to/task_dir 1"
    exit 1
fi

ENV_DIR=$1; shift
TASK_DIR=$1; shift
NUM_NODES=$1; shift

ENV_DIR=$(realpath "${ENV_DIR}")
TASK_DIR=$(realpath "${TASK_DIR}")

echo ENV_DIR=${ENV_DIR}
echo TASK_DIR=${TASK_DIR}

# この時間を超えるとジョブが強制終了されます．ポイントを使いすぎないための保険です．より長く学習するときなどは適宜増やしてください．
WALLTIME=24:00:00 # 24 hour

qsub \
    -l select=${NUM_NODES},walltime=${WALLTIME} \
    -v RTYPE=rt_HF,ENV_DIR=${ENV_DIR},TASK_DIR=${TASK_DIR} \
    -o /dev/null \
    -e /dev/null \
    -m n \
    train/qsub_train.sh

