#!/bin/bash

set -eu -o pipefail

if [ $# -ne 3 ]; then
    >&2 echo "Usage: $0 <input-hf-path> <output-nemo-path> <hparams-file>"
    >&2 echo "Example: $0 /path/to/input_hf /path/to/output_nemo /path/to/hparams_file"
    exit 1
fi

input_hf_path=$1; shift
output_nemo_path=$1; shift
hparams_file=$1; shift

qsub \
  -v INPUT_HF_PATH=${input_hf_path},OUTPUT_NEMO_PATH=${output_nemo_path},HPARAMS_FILE=${hparams_file},RTYPE=rt_HF \
  -o /dev/null -e /dev/null \
  -m n \
  scripts/abci/converter/qsub_hf_to_nemo_llama.sh
