# Installs flash attention.

echo "Installing Flash Attention ${TUNING_FLASH_ATTENTION_VERSION}"
source ${ENV_DIR}/venv/bin/activate
pushd ${ENV_DIR}/src

git clone https://github.com/Dao-AILab/flash-attention -b v${TUNING_FLASH_ATTENTION_VERSION}
pushd flash-attention
python -m pip install -e .
popd

deactivate
