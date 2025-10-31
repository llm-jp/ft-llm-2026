#!/bin/bash
#PBS -P gch51701
#PBS -q rt_HC
#PBS -N download
#PBS -v RTYPE=rt_HC
#PBS -l select=1
#PBS -l walltime=10:00:00
#PBS -o /dev/null
#PBS -e /dev/null

cd $PBS_O_WORKDIR

TIMESTAMP=$(date +%Y%m%d%H%M%S)
JOBID=${PBS_JOBID%%.*}
mkdir -p logs
LOGFILE=logs/download-$JOBID.out
ERRFILE=logs/download-$JOBID.err
exec > $LOGFILE 2> $ERRFILE

set -eu -o pipefail

source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/venv/bin/activate

echo "DATA_ROOT_DIR=${DATA_ROOT_DIR}"

python corpus/download_openwebmath.py --raw-data-dir ${DATA_ROOT_DIR}/raw

echo "Download done"