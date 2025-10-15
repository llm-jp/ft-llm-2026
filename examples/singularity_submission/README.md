```
$ singularity build --fakeroot dist/submission.sif submission.def

$ singularity run --nv --writable-tmpfs --env CUDA_VISIBLE_DEVICES=0 \
    dist/submission.sif \
    --model_path models/llm-jp_llm-jp-3.1-1.8b-instruct4 \
    --input_path sample_problems.jsonl \
    --output_path $(pwd)/output.jsonl
```
