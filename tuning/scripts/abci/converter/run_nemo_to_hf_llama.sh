#!/bin/bash

set -eu -o pipefail

if [ $# -ne 2 ]; then
    >&2 echo "Usage: $0 <input-nemo-path> <output-hf-path>"
    >&2 echo "Example: $0 /path/to/input_nemo /path/to/output_hf"
    exit 1
fi

input_nemo_path=$1; shift
output_hf_path=$1; shift

qsub \
  -v INPUT_NEMO_PATH=${input_nemo_path},OUTPUT_HF_PATH=${output_hf_path},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  scripts/abci/converter/qsub_nemo_to_hf_llama.sh
