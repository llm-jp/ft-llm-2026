EXAMPLE_MODEL := llm-jp/llm-jp-3.1-1.8b-instruct4

MODEL_NAMES := $(EXAMPLE_MODEL)
MODEL_DIRS := $(MODEL_NAMES:%=models/%)

UV_CACHE_DIR := $(shell uv cache dir)

SINGULARITY_BUILD_ARGS = --fakeroot --force
ifneq ($(strip $(UV_CACHE_DIR)),)
SINGULARITY_BUILD_ARGS += --bind $(UV_CACHE_DIR):/root/.cache/uv
endif
ifneq ($(strip $(MODEL_NAMES)),)
SINGULARITY_BUILD_ARGS += --build-arg MODEL_NAMES="$(MODEL_NAMES)"
endif

models/%:
	@if [ ! -d "$@" ]; then \
		uv run python download_model.py --model_name $(patsubst models/%,%,$@); \
	else \
		echo "Model directory $@ already exists, skipping download."; \
	fi

dist/submission.sif: main.py submission.def uv.lock $(MODEL_DIRS)
	singularity build $(SINGULARITY_BUILD_ARGS) \
	    dist/submission.sif submission.def

.PHONY: run
run: dist/submission.sif $(MODEL_DIRS)
	singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
	    --net --network none \
	    dist/submission.sif \
	    --model_path models/$(EXAMPLE_MODEL) \
	    --input_path sample_problems.jsonl \
	    --output_path $(pwd)/output.jsonl
