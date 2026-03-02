"""Text dataset for transformer models."""
import random
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Synthetic text dataset for transformer training.
    Each sample is tokenized text (input_ids, attention_mask) with a label.
    """

    def __init__(
        self,
        num_samples: int = 1024,
        max_length: int = 64,
        vocab_size: int = 1000,
    ) -> None:
        self.num_samples = num_samples
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.samples: list[Tuple[torch.Tensor, torch.Tensor, int]] = []
        for _ in range(num_samples):
            self.samples.append(self._make_example())

    def _make_example(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        seq_len = random.randint(8, self.max_length)
        input_ids = torch.tensor(
            [random.randint(1, self.vocab_size - 1) for _ in range(seq_len)],
            dtype=torch.long,
        )
        attention_mask = torch.ones(seq_len, dtype=torch.long)
        if seq_len < self.max_length:
            pad_len = self.max_length - seq_len
            input_ids = torch.nn.functional.pad(
                input_ids, (0, pad_len), value=self.pad_token_id
            )
            attention_mask = torch.nn.functional.pad(
                attention_mask, (0, pad_len), value=0
            )
        label = int(input_ids.sum().item() % 3)
        return input_ids[: self.max_length], attention_mask[: self.max_length], label

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, attention_mask, label = self.samples[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }
