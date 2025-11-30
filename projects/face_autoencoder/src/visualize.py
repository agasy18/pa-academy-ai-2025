"""Utility helpers for rendering reconstructions and training diagnostics inline."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.utils import make_grid

from .data import denormalize
from .model import ConvVAE


def _to_numpy_img(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu()
    array = tensor.permute(1, 2, 0).numpy()
    return np.clip(array, 0.0, 1.0)


def plot_reconstruction_grid(
    model: ConvVAE,
    batch: torch.Tensor,
    n_samples: int = 16,
    title: str | None = None,
) -> plt.Figure:
    """Display originals vs reconstructions."""

    model.eval()
    device = next(model.parameters()).device
    batch = batch[:n_samples].to(device)
    with torch.no_grad():
        output = model(batch)
    originals = denormalize(batch).cpu()
    recon = denormalize(output.reconstruction).cpu()

    grid = make_grid(
        torch.cat([originals, recon], dim=0),
        nrow=n_samples,
    )
    img = grid.permute(1, 2, 0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(n_samples, 4))
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    plt.show()
    return fig


def plot_latent_traversal(
    model: ConvVAE,
    base_latent: np.ndarray,
    component: np.ndarray,
    steps: int = 7,
    travel_range: float = 3.0,
) -> plt.Figure:
    """Display how moving along one PCA component affects generated faces."""

    linspace = np.linspace(-travel_range, travel_range, steps)
    images: list[np.ndarray] = []
    model.eval()
    device = next(model.parameters()).device
    for alpha in linspace:
        z = base_latent + alpha * component
        z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
        with torch.no_grad():
            generated = model.decode(z_tensor)
        img = denormalize(generated.squeeze(0).cpu())
        images.append(_to_numpy_img(img))

    fig, axes = plt.subplots(1, steps, figsize=(steps * 2, 2))
    for ax, img_arr, alpha in zip(axes, images, linspace):
        ax.imshow(img_arr)
        ax.set_title(f"{alpha:.1f}")
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig


def plot_training_curves(
    csv_log: Path,
    smoothing: int = 1,
) -> plt.Figure:
    """Display loss curves from FastAI's CSV logger."""

    if not csv_log.exists():
        raise FileNotFoundError(f"CSV log not found: {csv_log}")

    df = pd.read_csv(csv_log)
    if smoothing > 1:
        df["train_loss"] = df["train_loss"].rolling(smoothing, min_periods=1).mean()
        df["valid_loss"] = df["valid_loss"].rolling(smoothing, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
    ax.plot(df["epoch"], df["valid_loss"], label="Valid Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()
    return fig
