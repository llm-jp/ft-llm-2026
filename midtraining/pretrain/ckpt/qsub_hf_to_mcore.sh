#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N convert_hf_to_mcore
#PBS -l select=1:ncpus=8:ngpus=8
#PBS -v RTYPE=rt_HF
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/convert-${JOBID}.out
ERRFILE=./logs/convert-${JOBID}.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

MEGATRON_PATH=${ENV_DIR}/src/Megatron-LM
export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH:-}

# distributed settings
TARGET_TP_SIZE=1
TARGET_PP_SIZE=1

mkdir -p ${MEGATRON_CHECKPOINT_DIR}

# convert
python ${MEGATRON_PATH}/tools/checkpoint/convert.py \
  --model-type GPT \
  --loader llama_mistral \
  --saver mcore \
  --target-tensor-parallel-size ${TARGET_TP_SIZE} \
  --target-pipeline-parallel-size ${TARGET_PP_SIZE} \
  --load-dir ${HF_CHECKPOINT_DIR} \
  --save-dir ${MEGATRON_CHECKPOINT_DIR} \
  --tokenizer-model ${HF_CHECKPOINT_DIR} \
  --bf16 \
  --model-size llama2-7B \
  --checkpoint-type hf \
  --saver-transformer-impl "transformer_engine"