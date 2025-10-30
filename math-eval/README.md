# Scripts for the FT-LLM 2026 Workshop

This repository contains scripts and tools developed for the FT-LLM 2026 Workshop.

## math-eval

The `math-eval` script evaluates predictions on mathematical reasoning tasks.
It compares model predictions against gold answers and computes accuracy.
This tool relies on [Math-Verify](https://github.com/huggingface/Math-Verify), a mathematical expression evaluation system developed by Hugging Face.

### Requirements

Install `uv` to run the script following the official guidelines:
- [Installing uv](https://docs.astral.sh/uv/getting-started/installation/)

### Usage

To run the evaluation, use the following command:

```bash
uvx --from "git+https://github.com/llm-jp/ft-llm-2026#subdirectory=math-eval" math-eval <path/to/predictions.jsonl> <path/to/gold.jsonl> [-o <path/to/output.json>]
```

where `<path/to/predictions.jsonl>` is the path to the JSONL file containing model predictions, while `<path/to/gold.jsonl>` is the path to the JSONL file containing gold answers.
Examples can be found in the `examples/` directory.
The optional `-o` argument specifies the output file for saving evaluation results in JSON format.
