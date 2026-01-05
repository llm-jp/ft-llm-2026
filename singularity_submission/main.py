import argparse
import json
from pathlib import Path

from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = """\
以下は数学の問題です。
問題に対する解答のみを出力してください。
推論過程は出力しないでください。

# 問題
{question}

# 解答
"""


def get_default_model_path():
    models_dir = Path("/app/models")
    if not models_dir.exists():
        return None

    for path in models_dir.rglob("config.json"):
        return path.parent


def main():
    parser = argparse.ArgumentParser(description="Singularity Submission Example")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=get_default_model_path(),
        help="Path to the model directory (default: auto-detect from /app/models)",
    )
    parser.add_argument(
        "--input_path", type=Path, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--output_path", type=Path, required=True, help="Path to the output file"
    )

    args = parser.parse_args()

    if args.model_path is None:
        parser.error(
            "Model path not specified and could not auto-detect model in /app/models"
        )

    llm = LLM(model=str(args.model_path.resolve()))

    with open(args.input_path) as f:
        problems = list(map(json.loads, f))

    messages = []
    for problem in problems:
        messages.append(
            [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.format(question=problem["problem"]),
                }
            ]
        )

    outputs = llm.chat(
        messages, sampling_params=SamplingParams(temperature=0.0, max_tokens=64)
    )

    for problem, output in zip(problems, outputs):
        problem["output"] = output.outputs[0].text

    with open(args.output_path, "w") as f:
        for problem in problems:
            f.write(json.dumps(problem, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
