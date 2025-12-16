# Script for setup environment

ENV_DIR="/path/to/your/environment" # FIXME: update this path ex (ENV_DIR="/groups/gch51701/Team014/xxx/tuning/env")

# Setup Python environment
source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

# Set TMPDIR
export TMPDIR=${HOME}/tmp

# Determine master address:port
export MASTER_ADDR=$(head -n 1 $PBS_NODEFILE | hostname -f)
export MASTER_PORT=$((10000 + RANDOM % 1000))
echo "MASTER_ADDR=${MASTER_ADDR}; MASTER_PORT=${MASTER_PORT}"

# Determine amount of employed devices
NUM_NODES=$(wc -l < $PBS_NODEFILE)
NUM_GPUS_PER_NODE=8
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPUS_PER_NODE}))
echo "NUM_NODES=${NUM_NODES}"
echo "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE}"
echo "NUM_GPUS=${NUM_GPUS}"

cat $PBS_NODEFILE

# Set NVIDIA_PYTORCH_VERSION
export NVIDIA_PYTORCH_VERSION=""

# Wandb settings
export WANDB_API_KEY=$(cat ${HOME}/.wandb_api_key 2>/dev/null || echo "")
if [ -z "${WANDB_API_KEY}" ]; then
  export WANDB_MODE=offline
fi

# Debug/logging flags
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1
