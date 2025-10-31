#!/bin/bash

set -eu -o pipefail

if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 <env-dir> <data-root-dir>"
    >&2 echo "Example: $0 /path/to/env_dir /path/to/data_root_dir"
    exit 1
fi

env_dir=$1; shift
data_root_dir=$1; shift

qsub \
  -v ENV_DIR=${env_dir},DATA_ROOT_DIR=${data_root_dir},RTYPE=rt_HC \
  -o /dev/null -e /dev/null \
  -m n \
  qsub_download.sh

