# Script to install Python to TARGET_DIR
#
# This script will make the following directories:
#   * ${TARGET_DIR}/venv ... venv directory inherited from the above Python binary

echo "Setup venv"
pushd ${TARGET_DIR}

python/bin/python3 -m venv venv

source venv/bin/activate
pip install --upgrade pip wheel cython
pip install setuptools==69.5.1
pip install packaging
deactivate

popd  # ${TARGET_DIR}
