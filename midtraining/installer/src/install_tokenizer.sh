# Install LLM-jp Tokenizer.

echo "Installing LLM-jp Tokenizer ${PRETRAIN_TOKENIZER_TAG}"
pushd ${ENV_DIR}/src

# download our tokeniser
# Tokenizer
git clone https://github.com/llm-jp/llm-jp-tokenizer -b ${PRETRAIN_TOKENIZER_TAG}

popd  # ${ENV_DIR}/src
