#!/bin/bash

set -eu -o pipefail

if [ $# -ne 1 ]; then
    >&2 echo "Usage: $0 <env-dir>"
    >&2 echo "Example: $0 /path/to/env_dir"
    exit 1
fi

env_dir=$1; shift

qsub \
  -v ENV_DIR=${env_dir},RTYPE=rt_HG \
  -o /dev/null -e /dev/null \
  -m n \
  qsub_setup.sh
