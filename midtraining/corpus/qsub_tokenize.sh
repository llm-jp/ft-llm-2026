#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HC
#PBS -N tokenize
#PBS -v RTYPE=rt_HC
#PBS -l select=1
#PBS -l walltime=10:00:00
#PBS -o /dev/null
#PBS -e /dev/null

cd $PBS_O_WORKDIR

TIMESTAMP=$(date +%Y%m%d%H%M%S)
JOBID=${PBS_JOBID%%.*}
mkdir -p logs
LOGFILE=logs/tokenize-$JOBID.out
ERRFILE=logs/tokenize-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

# Set dataset path
SOURCE_DIR="${DATA_ROOT_DIR}/raw"
OUTPUT_ROOT="${DATA_ROOT_DIR}/tokenized/v4_alpha1.0"
OUTPUT_DIR="${OUTPUT_ROOT}/data"
OUTPUT_INFO="${OUTPUT_ROOT}/token_info.csv"

source ${ENV_DIR}/venv/bin/activate

# Clone Megatron-LM for tokenize data (branch: llmjp0-mdx)
TOKENIZER_DIRNAME=Megatron-LM-tokenizer
cd ${ENV_DIR}/src
if [ ! -d $TOKENIZER_DIRNAME ]; then
  git clone https://github.com/llm-jp/Megatron-LM.git -b llmjp0-mdx $TOKENIZER_DIRNAME
fi
cd $TOKENIZER_DIRNAME

# Tokenize settings
MODEL_PATH="${ENV_DIR}/src/llm-jp-tokenizer/models/ver4.0_alpha1.0/llm-jp-tokenizer_ver4.0_alpha1.0.model"
TOKENIZER_TYPE="Llama2Tokenizer"
WORKERS=16

# Tokenize
echo "Tokenizer: $MODEL_PATH"
mkdir -p $OUTPUT_DIR

find $SOURCE_DIR -name "*.jsonl" | while read -r file; do
  relative_path="${file#$SOURCE_DIR/}"
  output_path="$OUTPUT_DIR/$relative_path"
  mkdir -p $(dirname "$output_path")

  python tools/preprocess_data.py \
    --input "$file" \
    --output-result-total-token-info $OUTPUT_INFO \
    --output-prefix "${output_path%.jsonl}" \
    --tokenizer-model $MODEL_PATH \
    --tokenizer-type $TOKENIZER_TYPE \
    --workers $WORKERS \
    --append-eod
done

echo "Tokenization done"
