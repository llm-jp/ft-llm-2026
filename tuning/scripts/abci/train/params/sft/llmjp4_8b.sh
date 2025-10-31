MODEL_PARAMS=()

MODEL_PARAMS+=(
  model="llmjp4-8b"
  model.tensor_model_parallel_size=4
  model.pipeline_model_parallel_size=2
  mbs=4
  lr=0.00002
  min_lr=0.000002
)
