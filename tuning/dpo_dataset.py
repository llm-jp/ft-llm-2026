import json
import pickle
from pathlib import Path
from typing import Optional

import torch
from megatron.core import parallel_state
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
)
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingBatchSampler,
    MegatronPretrainingRandomBatchSampler,
)
from nemo.core.classes import Dataset
from nemo.utils import logging
from nemo.utils.get_rank import is_global_rank_zero
from omegaconf import DictConfig
from torch.utils.data import DataLoader


def create_dpo_dataloader(
    dataset: Dataset,
    consumed_samples: int,
    mbs: int,
    gbs: int,
    num_workers: int = 0,
    drop_last: bool = True,
    pad_samples_to_global_batch_size: bool = False,
    load_gbs: bool = True,
    seed: Optional[int] = None,
    use_random_sampler: bool = True,
    collate_fn=None,
) -> DataLoader:
    # Common parameters for batch sampler creation
    common_params = {
        "total_samples": len(dataset),
        "consumed_samples": consumed_samples,
        "micro_batch_size": mbs,
        "data_parallel_rank": parallel_state.get_data_parallel_rank(),
        "data_parallel_size": parallel_state.get_data_parallel_world_size(),
        "drop_last": drop_last,
        "global_batch_size": gbs,
        "pad_samples_to_global_batch_size": pad_samples_to_global_batch_size,
    }

    if use_random_sampler:
        cls = (
            MegatronPretrainingRandomBatchSampler
            if load_gbs
            else MegatronPretrainingRandomSampler
        )
        common_params["seed"] = seed
    else:
        cls = (
            MegatronPretrainingBatchSampler if load_gbs else MegatronPretrainingSampler
        )
    batch_sampler = cls(**common_params)

    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )


def tokenize_dpo_examples(
    orig_dataset_path: Path,
    cached_dataset_path: Path,
    tokenizer: TokenizerSpec,
) -> list[dict]:
    loaded_examples: list[dict] = []
    with orig_dataset_path.open(encoding="utf-8") as f:
        for line in f:
            loaded_examples.append(json.loads(line))

    tokenized_examples: list[dict] = []
    for example_idx, loaded_example in enumerate(loaded_examples):
        conversation: list[dict[str, str]] = loaded_example["messages"]
        assert len(conversation) >= 2
        assert conversation[0]["role"] == "system"

        prompt: str = f"{tokenizer.bos_token}{conversation[0]['content']}"
        for message in conversation[1:]:
            if message["role"] == "user":
                prompt += f"\n\n### 指示:\n{message['content']}"
            elif message["role"] == "assistant":
                prompt += f"\n\n### 応答:\n{message['content']}"
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        prompt += "\n\n### 応答:\n"

        prompt_ids: list[int] = tokenizer.text_to_ids(prompt)
        prompt_len: int = len(prompt_ids)
        chosen_ids: list[int] = tokenizer.text_to_ids(
            prompt + loaded_example["chosen_response"]
        )
        rejected_ids: list[int] = tokenizer.text_to_ids(
            prompt + loaded_example["rejected_response"]
        )

        assert (
            chosen_ids[0:prompt_len] == prompt_ids
        ), "the tokenizer for DPO has merged tokens between prompt and response"
        assert (
            rejected_ids[0:prompt_len] == prompt_ids
        ), "the tokenizer for DPO has merged tokens between prompt and response"

        if is_global_rank_zero() and example_idx < 2:
            logging.info(f"{example_idx = }")
            logging.info(f"{chosen_ids = }")
            logging.info(f"{rejected_ids = }")
        tokenized_examples.append(
            {
                "chosen_ids": chosen_ids,
                "rejected_ids": rejected_ids,
                "prompt_len": prompt_len,
            }
        )

    with cached_dataset_path.open("wb") as f:
        pickle.dump(tokenized_examples, f)

    return tokenized_examples


def load_dpo_datasets(
    cfg: DictConfig, tokenizer: TokenizerSpec
) -> tuple[list[dict[str, list[int]]], list[dict[str, list[int]]]]:
    data_name2num_examples: dict[str, dict[str, int]] = {}
    total_train_examples: list[dict] = []
    total_dev_examples: list[dict] = []
    for data_name, data_info in cfg.datasets.items():
        dataset_dir: Path = Path(f"{cfg.data_dir}/{cfg.data_version}/preference/train")
        cached_dataset_path: Path = dataset_dir / f"{data_name}.pkl"
        orig_dataset_path: Path = dataset_dir / f"{data_name}.jsonl"

        if cached_dataset_path.exists():
            if is_global_rank_zero():
                logging.info(f"Load from cached dataset: {cached_dataset_path}")
            with cached_dataset_path.open("rb") as f:
                tokenized_examples: list[dict[str, list[int]]] = pickle.load(f)
        elif orig_dataset_path.exists():
            if data_info.max_train_samples == 0:
                if is_global_rank_zero():
                    logging.info(
                        f"Skip {orig_dataset_path} because max_train_samples is set to 0."
                    )
                continue

            if is_global_rank_zero():
                logging.info(
                    f"No cached dataset found. Tokenizing {orig_dataset_path}..."
                )
            tokenized_examples = tokenize_dpo_examples(
                orig_dataset_path, cached_dataset_path, tokenizer
            )
        else:
            raise FileNotFoundError(f"{orig_dataset_path} does not exist.")

        if (
            data_info.max_train_samples > len(tokenized_examples)
            and is_global_rank_zero()
        ):
            logging.warning(
                f"{data_name} has only {len(tokenized_examples)} examples, "
                f"but max_train_samples is set to {data_info.max_train_samples}. "
                "Use all examples."
            )

        max_train_samples: int = (
            data_info.max_train_samples
            if data_info.max_train_samples != -1
            else len(tokenized_examples)
        )
        max_dev_samples: int = 0
        if data_info.split_dev:
            max_dev_samples = min(
                cfg.max_dev_samples,
                int(len(tokenized_examples) * cfg.max_dev_ratio),
            )
        train_examples: list[dict[str, list[int]]] = (
            tokenized_examples[max_dev_samples : max_dev_samples + max_train_samples]
            * data_info.upsampling_factor
        )
        dev_examples: list[dict[str, list[int]]] = (
            tokenized_examples[:max_dev_samples] * data_info.upsampling_factor
        )
        total_train_examples.extend(train_examples)
        total_dev_examples.extend(dev_examples)
        data_name2num_examples[data_name] = {
            "train": len(train_examples),
            "dev": len(dev_examples),
            "original": len(tokenized_examples),
            "upsampling_factor": data_info.upsampling_factor,
        }

    if is_global_rank_zero():
        num_total_original_examples: int = 0
        logging.info("------------------------------")
        logging.info("Dataset summary (original -> train/dev)")
        for data_name, num_examples in data_name2num_examples.items():
            num_total_original_examples += num_examples["original"]
            logging.info(
                f"{data_name}: {num_examples['original']} -> {num_examples['train']}/{num_examples['dev']} (upsampling factor: {num_examples['upsampling_factor']})"
            )
        logging.info(
            f"Total: {num_total_original_examples} -> {len(total_train_examples)}/{len(total_dev_examples)}"
        )
        logging.info("------------------------------")

    return total_train_examples, total_dev_examples


class DPOModelDataset(Dataset):
    def __init__(
        self,
        tokenized_examples: list[dict[str, list[int]]],
        tokenizer: TokenizerSpec,
        max_seq_length: int = 4096,
        sequence_parallel: bool = False,
        chosen_reward: float = 1.0,
        rejected_reward: float = 0.0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.sequence_parallel = sequence_parallel
        self.chosen_reward = chosen_reward
        self.rejected_reward = rejected_reward

        self.examples = self._process_examples(tokenized_examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        """Returns a pair of chosen/rejected pairs, their respective lengths, and labels."""
        example: dict[str, list[int]] = self.examples[idx]

        max_curr_seq_len: int = (
            self.max_seq_length
            if self.sequence_parallel
            else max(len(example["chosen_ids"]), len(example["rejected_ids"]))
        )

        chosen_tokens = torch.nn.functional.pad(
            torch.LongTensor(example["chosen_ids"]),
            (0, max_curr_seq_len - len(example["chosen_ids"])),
            mode="constant",
            value=self.tokenizer.eos_id,
        )
        rejected_tokens = torch.nn.functional.pad(
            torch.LongTensor(example["rejected_ids"]),
            (0, max_curr_seq_len - len(example["rejected_ids"])),
            mode="constant",
            value=self.tokenizer.eos_id,
        )
        labels_chosen_tokens = torch.nn.functional.pad(
            torch.LongTensor(example["chosen_labels"]),
            (0, max_curr_seq_len - len(example["chosen_labels"])),
            mode="constant",
            value=-100,
        )
        labels_reject_tokens = torch.nn.functional.pad(
            torch.LongTensor(example["rejected_labels"]),
            (0, max_curr_seq_len - len(example["rejected_labels"])),
            mode="constant",
            value=-100,
        )

        output = {
            "chosen": chosen_tokens,
            "rejected": rejected_tokens,
            "chosen_length": len(example["chosen_ids"]),
            "rejected_length": len(example["rejected_ids"]),
            "chosen_labels": labels_chosen_tokens,
            "rejected_labels": labels_reject_tokens,
            "chosen_reward": self.chosen_reward,
            "rejected_reward": self.rejected_reward,
        }
        return output

    def _process_examples(
        self, tokenized_examples: list[dict]
    ) -> list[dict[str, list[int]]]:
        examples: list[dict[str, list[int]]] = []
        for tokenized_example in tokenized_examples:
            chosen_len: int = len(tokenized_example["chosen_ids"])
            rejected_len: int = len(tokenized_example["rejected_ids"])
            prompt_len: int = tokenized_example["prompt_len"]
            chosen_labels: list[int] = ([-100] * prompt_len) + tokenized_example[
                "chosen_ids"
            ][prompt_len:]
            rejected_labels: list[int] = ([-100] * prompt_len) + tokenized_example[
                "rejected_ids"
            ][prompt_len:]
            if max(chosen_len, rejected_len) > self.max_seq_length:
                continue
            examples.append(
                {
                    "chosen_ids": tokenized_example["chosen_ids"],
                    "rejected_ids": tokenized_example["rejected_ids"],
                    "chosen_labels": chosen_labels,
                    "rejected_labels": rejected_labels,
                }
            )
        return examples
