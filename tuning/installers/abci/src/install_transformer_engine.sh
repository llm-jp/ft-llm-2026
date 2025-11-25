# Installs Transformer Engine.

echo "Installing Transformer Engine with commit ${TUNING_TRANSFORMER_ENGINE_COMMIT}"
source ${ENV_DIR}/venv/bin/activate

export CUDA_PATH=/apps/cuda/${TUNING_CUDA_VERSION}.${TUNING_CUDA_VERSION_PATCH}
export CUDNN_PATH=/apps/cudnn/${TUNING_CUDNN_VERSION_WITH_PATCH}/cuda${TUNING_CUDA_VERSION}
echo "CUDA_PATH: ${CUDA_PATH}"
echo "CUDNN_PATH: ${CUDNN_PATH}"

NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@c81733f1032a56a817b594c8971a738108ded7d0 --no-cache-dir

deactivate
