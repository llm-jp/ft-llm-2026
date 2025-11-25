# Install pytorch and torchvision

echo "Installing torch ${TUNING_TORCH_VERSION}+cu${TUNING_CUDA_VERSION_SHORT} and torchvision ${TUNING_TORCHVISION_VERSION}+cu${TUNING_CUDA_VERSION_SHORT}"

source ${ENV_DIR}/venv/bin/activate

python -m pip install \
    --no-cache-dir \
    torch==${TUNING_TORCH_VERSION} \
    torchvision==${TUNING_TORCHVISION_VERSION} \
    --index-url https://download.pytorch.org/whl/cu${TUNING_CUDA_VERSION_SHORT}

deactivate
