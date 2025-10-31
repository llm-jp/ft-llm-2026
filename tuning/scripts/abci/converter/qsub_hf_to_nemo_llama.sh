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

echo "Converting HF checkpoint to NeMo format"

# Setup environment
source scripts/abci/common/setup.sh

python scripts/ckpt/convert_llama_hf_to_nemo.py \
  --input-name-or-path ${INPUT_HF_PATH} \
  --output-path ${OUTPUT_NEMO_PATH} \
  --hparams-file ${HPARAMS_FILE} \
  --cpu-only \
  --n-jobs 96

echo "Extracting the Nemo checkpoint to ${OUTPUT_NEMO_PATH}"
mkdir -p "${OUTPUT_NEMO_PATH}"
tar -xvf "${OUTPUT_NEMO_PATH}.nemo" -C "${OUTPUT_NEMO_PATH}"

if [ -f "${OUTPUT_NEMO_PATH}/model_config.yaml" ] && [ -d "${OUTPUT_NEMO_PATH}/model_weights" ]; then
  echo "Successfully converted the checkpoint to Nemo format. Removing the nemo file."
  rm "${OUTPUT_NEMO_PATH}.nemo"
  echo "Done conversion."
fi
