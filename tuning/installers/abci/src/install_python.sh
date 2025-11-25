# Script to install Python to ENV_DIR
#
# This script will make the following directories:
#   * ${ENV_DIR}/src/cpython ... Source of Python
#   * ${ENV_DIR}/python ... installed Python binary

echo "Installing Python ${TUNING_PYTHON_VERSION}"
pushd ${ENV_DIR}/src

git clone https://github.com/python/cpython -b v${TUNING_PYTHON_VERSION}
pushd cpython
./configure --prefix="${ENV_DIR}/python" --enable-optimizations
make -j 64
make install
popd  # cpython

popd  # ${ENV_DIR}/src
