# Installs flash attention.

echo "Installing Flash Attention ${PRETRAIN_FLASH_ATTENTION_VERSION}"
source ${ENV_DIR}/venv/bin/activate
pushd ${ENV_DIR}/src

export MAX_JOBS=8
git clone https://github.com/Dao-AILab/flash-attention -b v${PRETRAIN_FLASH_ATTENTION_VERSION}
pushd flash-attention
python -m pip install --no-build-isolation -e .
popd

deactivate
