#!/bin/bash

set -eu -o pipefail

if [ $# -ne 3 ]; then
    >&2 echo "Usage: $0 <env-dir> <task-dir> <iter>"
    >&2 echo "Example: $0 /path/to/env_dir /path/to/task_dir iter"
    exit 1
fi

env_dir=$1; shift
task_dir=$1; shift
iter=$1; shift

env_dir=$(realpath ${env_dir})
task_dir=$(realpath ${task_dir})

qsub \
  -v ENV_DIR=${env_dir},TASK_DIR=${task_dir},ITER=${iter},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  ckpt/qsub_mcore_to_hf.sh

