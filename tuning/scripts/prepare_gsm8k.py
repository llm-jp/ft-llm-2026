#!/usr/bin/env python3
"""
GSM8K データセットをダウンロードしてSFT用のフォーマットに変換するスクリプト
"""

import json
import os
from datasets import load_dataset

def convert_to_sft_format(example, idx):
    """GSM8KのサンプルをSFT形式に変換"""
    # answerから計算過程と最終回答を抽出
    answer = example["answer"]

    # <<calculation>>result の形式を読みやすく整形
    # そのまま使用（モデルにこの形式を学習させる）

    return {
        "ID": f"gsm8k_train_{idx:05d}",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that solves math problems step by step."
            },
            {
                "role": "user",
                "content": example["question"]
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
    }

def main():
    # 出力ディレクトリ
    output_dir = "./datasets/tuning_data20251105/tuning/train"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")

    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")

    # 学習データを変換
    output_file = os.path.join(output_dir, "gsm8k.jsonl")
    print(f"Converting and saving to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, example in enumerate(dataset["train"]):
            converted = convert_to_sft_format(example, idx)
            f.write(json.dumps(converted, ensure_ascii=False) + "\n")

    print(f"Done! Saved {len(dataset['train'])} samples to {output_file}")

    # サンプルを表示
    print("\n--- Sample data ---")
    sample = convert_to_sft_format(dataset["train"][0], 0)
    print(json.dumps(sample, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
