MODEL_PARAMS=()

MODEL_PARAMS+=(
  model="llmjp4-8b"
  model.tensor_model_parallel_size=4
  model.pipeline_model_parallel_size=2
  mbs=2
  gbs=256
  lr=0.0000009
  min_lr=0.0000005
  trainer.dpo.max_epochs=2
  dpo.ref_policy_kl_penalty=0.5
)
