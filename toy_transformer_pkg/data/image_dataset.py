"""Image dataset for ViT and Stable Diffusion."""
import random
from typing import Tuple

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    """
    Synthetic image dataset for vision models.
    Returns (image_tensor, label) for classification or (image_tensor,) for generation.
    """

    def __init__(
        self,
        num_samples: int = 256,
        image_size: int = 224,
        num_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.samples: list[Tuple[torch.Tensor, int]] = [
            self._make_example() for _ in range(num_samples)
        ]

    def _make_example(self) -> Tuple[torch.Tensor, int]:
        image = torch.randn(
            self.num_channels, self.image_size, self.image_size
        ).float()
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        label = random.randint(0, self.num_classes - 1)
        return image, label

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.samples[idx]
