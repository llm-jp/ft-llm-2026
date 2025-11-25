# Install nemo

echo "Installing nemo with commit ${TUNING_NEMO_COMMIT}"
source ${ENV_DIR}/venv/bin/activate
pushd ${ENV_DIR}/src

git clone https://github.com/NVIDIA/NeMo.git
pushd NeMo

# Checkout the specific commit
git checkout ${TUNING_NEMO_COMMIT}
# Apply the patch
git apply ${SCRIPT_DIR}/src/NeMo_v2.1.0rc0.patch

python -m pip install -e .
popd

popd  # ${ENV_DIR}/src
deactivate
