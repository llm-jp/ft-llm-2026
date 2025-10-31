#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HG
#PBS -N convert
#PBS -l select=1:ncpus=96:ngpus=1
#PBS -l walltime=5:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

# Setup logs
cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/convert-$JOBID.out
ERRFILE=./logs/convert-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

echo "Converting NeMo checkpoint to HF format"

# Setup environment
source scripts/abci/common/setup.sh

CHAT_TEMPLATE="{% for message in messages %}{% if message['role'] == 'user' %}{{ '\\n\\n### 指示:\\n' + message['content'] }}{% elif message['role'] == 'system' %}{{ '以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。' }}{% elif message['role'] == 'assistant' %}{{ '\\n\\n### 応答:\\n' + message['content'] + eos_token }}{% endif %}{% if loop.last and add_generation_prompt %}{{ '\\n\\n### 応答:\\n' }}{% endif %}{% endfor %}"

MODEL_ID=$(basename ${INPUT_NEMO_PATH}) # The lowest directory name in INPUT_NEMO_PATH is used as the model id.

if [ -d "${INPUT_NEMO_PATH}/checkpoints" ]; then
  INPUT_NEMO_PATH="${INPUT_NEMO_PATH}/checkpoints"
fi

if [ ! -f "${INPUT_NEMO_PATH}/model_config.yaml" ]; then
  echo "model_config.yaml not found in ${INPUT_NEMO_PATH}"
  exit 1
fi

if [ ! -d "${INPUT_NEMO_PATH}/model_weights" ]; then
  ln -s $(ls -d ${INPUT_NEMO_PATH}/step=*-last) ${INPUT_NEMO_PATH}/model_weights
fi


python scripts/ckpt/convert_llama_nemo_to_hf.py \
  --input-name-or-path ${INPUT_NEMO_PATH} \
  --chat-template "${CHAT_TEMPLATE}" \
  --output-hf-path ${OUTPUT_HF_PATH} \
  --precision bf16 \
  --cpu-only \
  --n-jobs 96
