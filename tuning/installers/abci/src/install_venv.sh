# Script to install Python to ENV_DIR
#
# This script will make the following directories:
#   * ${ENV_DIR}/venv ... venv directory inherited from the above Python binary

echo "Setup venv"
pushd ${ENV_DIR}

python/bin/python3 -m venv venv

source venv/bin/activate
pip install --upgrade pip wheel
pip install setuptools==69.5.1
pip install --upgrade cython
pip install packaging
deactivate

popd  # ${ENV_DIR}
