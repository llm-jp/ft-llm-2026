#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HF
#PBS -N train
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -m n

cd ${PBS_O_WORKDIR}

JOB_ID=${PBS_JOBID%%.*}
mkdir -p ${TASK_DIR}/logs
LOGFILE=${TASK_DIR}/logs/pretrain-${JOB_ID}.out
ERRFILE=${TASK_DIR}/logs/pretrain-${JOB_ID}.err
exec > ${LOGFILE} 2> ${ERRFILE}

set -eu -o pipefail

EXPERIMENT_NAME=pretrain_${JOB_ID}

# Load common environment variables
source ${ENV_DIR}/scripts/environment.sh

# Load modules
source /etc/profile.d/modules.sh
module load cuda/${PRETRAIN_CUDA_VERSION}/${PRETRAIN_CUDA_VERSION}.${PRETRAIN_CUDA_VERSION_PATCH}
module load cudnn/${PRETRAIN_CUDNN_VERSION}/${PRETRAIN_CUDNN_VERSION_WITH_PATCH}
module load hpcx/${PRETRAIN_HPCX_VERSION}
module load nccl/${PRETRAIN_NCCL_VERSION}/${PRETRAIN_NCCL_VERSION_WITH_PATCH}
# For logging
module list

# Load Python venv
source ${ENV_DIR}/venv/bin/activate

## Debug/logging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1

# Set up environment variables for distributed training
export MASTER_ADDR=$(head -n 1 ${PBS_NODEFILE} | hostname -f)
export MASTER_PORT=$((10000 + RANDOM % 1000))
echo "hostname: ${MASTER_ADDR}"

NUM_NODES=$(wc -l < ${PBS_NODEFILE})
NUM_GPUS_PER_NODE=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "nnodes: ${NUM_NODES}; ngpus: ${NUM_GPUS}"
echo NUM_NODES=${NUM_NODES}
echo NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}
echo NUM_GPUS=${NUM_GPUS}

cat ${PBS_NODEFILE}

# Training data: TRAIN_DATA_PATH
source ${TASK_DIR}/train_data.sh

# Synthesize all model params: ALL_PARAMS
# Requires and TRAIN_DATA_PATH
source ${TASK_DIR}/params.sh

# Add logging params
WANDB_ENTITY="xxx"
WANDB_PROJECT="midtraining"

# Load WandB API key from file
if [ -f ~/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat ~/.wandb_api_key)
fi

ALL_PARAMS+=(
    --log-interval 1
    --log-throughput
    --wandb-entity ${WANDB_ENTITY}
    --wandb-project ${WANDB_PROJECT}
    --wandb-exp-name ${EXPERIMENT_NAME}
)

# Add Checkpointing params
BASE_CHECKPOINT_DIR=${TASK_DIR}/base_checkpoints
TASK_CHECKPOINT_DIR=${TASK_DIR}/checkpoints

if [ -e ${TASK_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]; then
    echo "Resume from the last checkpoint in this task"
    LOAD_DIR=${TASK_CHECKPOINT_DIR}
elif [ -e ${BASE_CHECKPOINT_DIR}/latest_checkpointed_iteration.txt ]; then
    echo "Start from the base checkpoint"
    LOAD_DIR=${BASE_CHECKPOINT_DIR}
    ALL_PARAMS+=(
      --finetune
    )
else
    echo "Start from scratch"
    LOAD_DIR=${TASK_CHECKPOINT_DIR}
fi

ALL_PARAMS+=(
    --load ${LOAD_DIR}
    --save ${TASK_CHECKPOINT_DIR}
    --save-interval 1000
)

echo "ALL_PARAMS: ${ALL_PARAMS[@]}"

mpirun \
    --display-allocation \
    --report-bindings \
    --oversubscribe \
    -np ${NUM_GPUS} \
    --npernode ${NUM_GPUS_PER_NODE} \
    -bind-to none \
    -map-by slot \
    python \
        ${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py \
        ${ALL_PARAMS[@]}
