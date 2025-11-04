import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="llm-jp/llm-jp-3.1-1.8b-instruct4",
        help="Model name on Hugging Face Hub",
    )
    args = parser.parse_args()

    model_path = snapshot_download(
        repo_id=args.model_name,
        local_dir=Path("models") / args.model_name,
        local_dir_use_symlinks=False,
    )

    print(f"Model downloaded to: {model_path}")


if __name__ == "__main__":
    main()
