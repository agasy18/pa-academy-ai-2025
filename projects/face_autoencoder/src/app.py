"""Gradio app for interacting with PCA-controlled VAE face generations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import gradio as gr
import numpy as np
import torch

from .config import ARTIFACT_DIR
from .data import denormalize
from .model import ConvVAE


def _latest_metadata() -> Path:
    run_dirs = sorted([d for d in ARTIFACT_DIR.iterdir() if d.is_dir()])
    for run_dir in reversed(run_dirs):
        candidate = run_dir / "metadata.json"
        if candidate.exists():
            return candidate
    # Fallback to legacy metadata naming if present
    legacy = sorted(ARTIFACT_DIR.glob("metadata_*.json"))
    if legacy:
        return legacy[-1]
    raise FileNotFoundError(
        "No metadata files found. Train the model using "
        "`projects/face_autoencoder/face_autoencoder_training.ipynb` first."
    )


def load_artifacts(metadata_path: Path | None = None) -> tuple[ConvVAE, dict]:
    metadata_path = metadata_path or _latest_metadata()
    metadata = json.loads(metadata_path.read_text())

    model_path = Path(metadata["model_path"])
    pca_path = Path(metadata["pca_path"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(
        latent_dim=metadata["latent_dim"],
        image_channels=3,
        image_size=metadata.get("image_size", 64),
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device).eval()

    pca_data = np.load(pca_path)
    metadata["pca_components_matrix"] = pca_data["components"]
    metadata["pca_var_ratio"] = pca_data["explained_variance_ratio"]
    metadata["latent_mean"] = pca_data["latent_mean"]

    return model, metadata


def make_generator(
    model: ConvVAE,
    metadata: dict,
) -> callable:
    device = next(model.parameters()).device
    components = metadata["pca_components_matrix"]
    latent_mean = metadata["latent_mean"]

    def generate(*coeffs: float) -> np.ndarray:
        z = latent_mean.copy()
        for coeff, component in zip(coeffs, components):
            z += coeff * component
        z_tensor = torch.from_numpy(z).float().unsqueeze(0).to(device)
        with torch.no_grad():
            decoded = model.decode(z_tensor)
        img = denormalize(decoded.squeeze(0)).permute(1, 2, 0).cpu().numpy()
        return (img * 255).astype(np.uint8)

    return generate


def random_coeffs(num_components: int, scale: float = 2.0) -> List[float]:
    return np.random.uniform(-scale, scale, size=num_components).tolist()


def launch(metadata_path: str | None = None) -> None:
    model, metadata = load_artifacts(Path(metadata_path) if metadata_path else None)
    var_ratio = metadata["pca_var_ratio"]
    generator = make_generator(model, metadata)
    num_components = len(var_ratio)

    with gr.Blocks(title="PCA Face Latent Explorer") as demo:
        gr.Markdown(
            "## Latent Space PCA Sliders\n"
            "Move the sliders to traverse principal directions of the VAE latent "
            "space. These directions roughly correspond to interpretable facial "
            "attributes (lighting, pose, expressions, etc.)."
        )
        slider_rows: list[gr.Slider] = []
        with gr.Row():
            image_output = gr.Image(label="Generated Face", width=256, height=256)
        with gr.Column():
            for idx, ratio in enumerate(var_ratio):
                slider = gr.Slider(
                    minimum=-3.0,
                    maximum=3.0,
                    value=0.0,
                    step=0.1,
                    label=f"PC {idx + 1} ({ratio*100:.1f}% var)",
                )
                slider_rows.append(slider)
        random_btn = gr.Button("Randomize Sliders")

        def update_face(*coeffs: float) -> np.ndarray:
            return generator(*coeffs)

        def randomize_and_generate():
            coeffs = random_coeffs(num_components)
            img = generator(*coeffs)
            return (*coeffs, img)

        for slider in slider_rows:
            slider.change(
                fn=update_face,
                inputs=slider_rows,
                outputs=image_output,
                show_progress=False,
            )
        random_btn.click(
            fn=randomize_and_generate,
            outputs=slider_rows + [image_output],
            show_progress=False,
        )

    demo.launch()


if __name__ == "__main__":
    launch()
