#!/bin/bash
#PBS -P gch51701
#PBS -q FIXME
#PBS -N FIXME
#PBS -l select=8:ncpus=192
#PBS -l walltime=168:00:00
#PBS -m n

# Setup logs
cd $PBS_O_WORKDIR

JOBID=${PBS_JOBID%%.*}
mkdir -p ./logs
LOGFILE=./logs/train-$JOBID.out
ERRFILE=./logs/train-$JOBID.err
exec > $LOGFILE 2> $ERRFILE
echo "JOBID=${JOBID}"

set -euxo pipefail

# Setup environment
source scripts/abci/common/setup.sh
echo "ENV_DIR=${ENV_DIR}"

# Name model randomly
NAME="dpo-$(od -vAn -N4 -tu4 < /dev/urandom | tr -d ' ')"
echo "NAME=${NAME}"

# Load MODEL_PARAMS
source scripts/abci/train/params/dpo/${PARAM_NAME}.sh
echo "MODEL_PARAMS: ${MODEL_PARAMS[@]}"

mpirun \
  --display-allocation \
  --report-bindings \
  --oversubscribe \
  -np $NUM_GPUS \
  --npernode $NUM_GPUS_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  -bind-to none \
  -map-by slot \
  -x PATH \
  python train_dpo.py \
    trainer.num_nodes=${NUM_NODES} \
    use_mpi=True \
    use_slurm=True \
    name=${NAME} \
    seed=${SEED} \
    model.restore_from_path=${MODEL_PATH} \
    "${MODEL_PARAMS[@]}"
