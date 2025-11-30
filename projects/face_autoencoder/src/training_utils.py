"""Shared training helpers for the face autoencoder notebook/app."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F
from fastai.callback.core import Callback
from fastai.callback.schedule import fit_one_cycle as _register_fit_one_cycle  # noqa: F401
from fastai.learner import Learner
from IPython.display import clear_output

from .visualize import plot_reconstruction_grid


def reconstruction_mse(output, target):
    """FastAI metric wrapper for reconstruction error."""

    return F.mse_loss(output.reconstruction, target, reduction="mean")


reconstruction_mse.__name__ = "recon_mse"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_latents(learner: Learner) -> np.ndarray:
    """Run the encoder across the train+valid splits to gather latent means."""

    learner.model.eval()
    device = next(learner.model.parameters()).device
    latents: list[torch.Tensor] = []
    with torch.no_grad():
        for dl in (learner.dls.train, learner.dls.valid):
            for xb, _ in dl:
                xb = xb.to(device)
                output = learner.model(xb)
                latents.append(output.mu.cpu())
    return torch.cat(latents, dim=0).numpy()


class InlineReconstructionCallback(Callback):
    """Render reconstruction grids inline after every epoch."""

    def __init__(self, reference_batch: torch.Tensor, samples: int = 8) -> None:
        self.reference_batch = reference_batch.detach().cpu()
        self.samples = samples

    def after_epoch(self) -> None:
        clear_output(wait=True)
        epoch_idx = getattr(self, "epoch", None)
        title = f"Epoch {epoch_idx}" if epoch_idx is not None else "Epoch"
        plot_reconstruction_grid(
            self.learn.model,
            self.reference_batch,
            n_samples=min(self.samples, self.reference_batch.size(0)),
            title=title,
        )
