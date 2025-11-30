"""Centralized paths and default hyperparameters for the face autoencoder."""

from __future__ import annotations

from pathlib import Path

# Resolve key directories relative to this file so the scripts can be launched
# from anywhere inside the repo.
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_LFW_DIR = DATA_DIR / "lfw"
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
LOG_DIR = PROJECT_DIR / "logs"
ASSET_DIR = PROJECT_DIR / "assets"

# Ensure folders exist so downstream scripts can assume availability.
for path in (DATA_DIR, RAW_LFW_DIR, ARTIFACT_DIR, LOG_DIR, ASSET_DIR):
    path.mkdir(parents=True, exist_ok=True)


DEFAULTS = {
    "image_size": 64,
    "batch_size": 128,
    "latent_dim": 64,
    "lr": 2e-3,
    "epochs": 25,
    "num_workers": 4,
    "pca_components": 8,
    "seed": 42,
    "lfw_resize": 0.5,
}
