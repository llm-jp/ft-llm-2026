# Install

echo "Installing apex with commit ${TUNING_APEX_COMMIT}"
source ${TARGET_DIR}/venv/bin/activate
pushd ${TARGET_DIR}/src

git clone git@github.com:NVIDIA/apex.git
pushd apex

# Checkout the specific commit
git checkout ${TUNING_APEX_COMMIT}

python -m pip install . -v --no-build-isolation --disable-pip-version-check --no-cache-dir --config-settings "--build-option=--cpp_ext --cuda_ext --fast_layer_norm --distributed_adam --deprecated_fused_adam --group_norm"
popd

popd  # ${TARGET_DIR}/src
deactivate
