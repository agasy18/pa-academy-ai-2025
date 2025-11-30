"""Dataset helpers for the face autoencoder example."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from fastai.data.core import DataLoaders
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import Dataset, random_split

from .config import DEFAULTS, RAW_LFW_DIR


@dataclass
class LFWDatasetInfo:
    """Metadata describing the prepared LFW tensors."""

    images: np.ndarray
    image_size: int
    num_samples: int
    num_channels: int
    height: int
    width: int


class LFWAutoencoderDataset(Dataset):
    """Returns (image, image) pairs normalized to [-1, 1] for VAE training."""

    def __init__(self, images: np.ndarray, image_size: int = 64):
        self.images = images.astype(np.float32)
        self.image_size = image_size

    def __len__(self) -> int:  # type: ignore[override]
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        img = torch.from_numpy(self.images[idx]).permute(2, 0, 1)  # (C, H, W)
        img = img.unsqueeze(0)
        img = F.interpolate(
            img,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        img = img.clamp(0.0, 1.0)
        img = img * 2.0 - 1.0  # map to [-1, 1] for tanh decoder outputs
        return img, img


def fetch_lfw_dataset(resize: float = 0.5) -> LFWDatasetInfo:
    """Download (if needed) and load the LFW people dataset."""

    os.environ["SCIKIT_LEARN_DATA"] = str(RAW_LFW_DIR.resolve())
    dataset = fetch_lfw_people(
        color=True,
        resize=resize,
        funneled=True,
        download_if_missing=True,
    )
    images = dataset.images  # (N, H, W, 3) floats in [0, 1]
    num_samples, height, width, channels = images.shape
    return LFWDatasetInfo(
        images=images,
        image_size=int(DEFAULTS["image_size"]),
        num_samples=num_samples,
        num_channels=channels,
        height=height,
        width=width,
    )


def create_dataloaders(
    batch_size: int = DEFAULTS["batch_size"],
    image_size: int = DEFAULTS["image_size"],
    num_workers: int = DEFAULTS["num_workers"],
    seed: int = DEFAULTS["seed"],
    lfw_resize: float = DEFAULTS["lfw_resize"],
) -> Tuple[DataLoaders, LFWDatasetInfo]:
    """Create FastAI dataloaders for VAE training/validation."""

    info = fetch_lfw_dataset(resize=lfw_resize)
    dataset = LFWAutoencoderDataset(info.images, image_size=image_size)
    generator = torch.Generator().manual_seed(seed)
    n_train = int(len(dataset) * 0.9)
    n_valid = len(dataset) - n_train
    train_ds, valid_ds = random_split(dataset, [n_train, n_valid], generator=generator)
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    dls = DataLoaders(train_dl, valid_dl)
    return dls, info


def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo the [-1, 1] normalization and clamp to [0, 1]."""

    return ((t + 1.0) / 2.0).clamp(0.0, 1.0)
