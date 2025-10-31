from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import orjson


def main():
    parser = ArgumentParser()
    parser.add_argument("--raw-data-dir", type=str, required=True)
    args = parser.parse_args()

    raw_data_dir: Path = Path(args.raw_data_dir)
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "allenai/olmo-mix-1124",
        name="open-web-math",
        split="train",
        streaming=True,
    )
    raw_data_path: Path = raw_data_dir / "openwebmath.jsonl"
    with raw_data_path.open("w", encoding="utf-8") as f:
        for ex in tqdm(ds):
            f.write(
                orjson.dumps({
                    "id": ex["id"],
                    "text": ex["text"],
                }, option=orjson.OPT_APPEND_NEWLINE).decode("utf-8")
            )


if __name__ == "__main__":
    main()
