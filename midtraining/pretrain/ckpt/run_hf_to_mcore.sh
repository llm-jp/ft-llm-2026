#!/bin/bash

set -eu -o pipefail

if [ $# -ne 3 ]; then
    >&2 echo "Usage: $0 <env-dir> <hf-ckpt-dir> <mcore-ckpt-dir>"
    >&2 echo "Example: $0 /path/to/env_dir /path/to/hf_ckpt_dir /path/to/mcore_ckpt_dir"
    exit 1
fi

env_dir=$1; shift
hf_ckpt_dir=$1; shift
mcore_ckpt_dir=$1; shift

qsub \
  -v ENV_DIR=${env_dir},HF_CHECKPOINT_DIR=${hf_ckpt_dir},MEGATRON_CHECKPOINT_DIR=${mcore_ckpt_dir},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  ckpt/qsub_hf_to_mcore.sh

