# Installs prerequisite packages

echo "Installing requirements"

source ${ENV_DIR}/venv/bin/activate

python -m pip install --no-cache-dir fasttext==0.9.3
python -m pip install --no-cache-dir --no-build-isolation -U -r ${SCRIPT_DIR}/src/requirements.txt

deactivate
