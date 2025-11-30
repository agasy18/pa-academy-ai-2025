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


def _get_available_runs() -> list[tuple[str, Path]]:
    """Get list of available runs with metadata.json files.
    
    Returns:
        List of (run_id, metadata_path) tuples, sorted by run_id descending.
    """
    runs = []
    for run_dir in ARTIFACT_DIR.iterdir():
        if run_dir.is_dir():
            metadata_path = run_dir / "metadata.json"
            if metadata_path.exists():
                run_id = run_dir.name
                runs.append((run_id, metadata_path))
    # Sort by run_id descending (most recent first)
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs


def _latest_metadata() -> Path:
    """Get the most recent metadata file."""
    runs = _get_available_runs()
    if runs:
        return runs[0][1]
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
    available_runs = _get_available_runs()
    if not available_runs:
        raise FileNotFoundError(
            "No metadata files found. Train the model using "
            "`projects/face_autoencoder/face_autoencoder_training.ipynb` first."
        )
    
    # Find maximum number of components across all runs to create enough sliders
    max_components = 0
    for _, path in available_runs:
        try:
            metadata = json.loads(path.read_text())
            pca_path = Path(metadata["pca_path"])
            if pca_path.exists():
                pca_data = np.load(pca_path)
                num_comp = len(pca_data["explained_variance_ratio"])
                max_components = max(max_components, num_comp)
        except Exception:
            continue
    
    # Ensure we have at least some sliders
    max_components = max(max_components, 8)
    
    # Get initial run
    initial_metadata_path = Path(metadata_path) if metadata_path else available_runs[0][1]
    initial_model, initial_metadata = load_artifacts(initial_metadata_path)
    initial_var_ratio = initial_metadata["pca_var_ratio"]
    initial_generator = make_generator(initial_model, initial_metadata)
    initial_num_components = len(initial_var_ratio)
    
    # Find initial run_id
    initial_run_id = None
    for run_id, path in available_runs:
        if path == initial_metadata_path:
            initial_run_id = run_id
            break
    if initial_run_id is None:
        initial_run_id = available_runs[0][0]

    with gr.Blocks(title="PCA Face Latent Explorer") as demo:
        gr.Markdown(
            "## Latent Space PCA Sliders\n"
            "Move the sliders to traverse principal directions of the VAE latent "
            "space. These directions roughly correspond to interpretable facial "
            "attributes (lighting, pose, expressions, etc.)."
        )
        
        # Run selection and info
        with gr.Row():
            run_dropdown = gr.Dropdown(
                choices=[run_id for run_id, _ in available_runs],
                value=initial_run_id,
                label="Select Run",
                interactive=True,
            )
            run_info = gr.Markdown(
                value=_format_run_info(initial_metadata),
                label="Run Information",
            )
        
        slider_rows: list[gr.Slider] = []
        with gr.Row():
            image_output = gr.Image(label="Generated Face", width=256, height=256)
        with gr.Column():
            # Create sliders for maximum components, but only show labels for active ones
            for idx in range(max_components):
                if idx < len(initial_var_ratio):
                    label = f"PC {idx + 1} ({initial_var_ratio[idx]*100:.1f}% var)"
                    visible = True
                else:
                    label = f"PC {idx + 1} (unused)"
                    visible = False
                slider = gr.Slider(
                    minimum=-3.0,
                    maximum=3.0,
                    value=0.0,
                    step=0.1,
                    label=label,
                    visible=visible,
                )
                slider_rows.append(slider)
        random_btn = gr.Button("Randomize Sliders")

        # Store current model and generator in component state
        # Wrap in dicts to prevent Gradio from trying to call them
        model_state = gr.State(value={"model": initial_model})
        generator_state = gr.State(value={"generator": initial_generator})
        metadata_state = gr.State(value=initial_metadata)
        num_components_state = gr.State(value=initial_num_components)

        def update_face(*coeffs: float) -> np.ndarray:
            generator = generator_state.value["generator"]
            num_components = num_components_state.value
            # Only use the first num_components coefficients
            active_coeffs = coeffs[:num_components]
            return generator(*active_coeffs)

        def randomize_and_generate():
            generator = generator_state.value["generator"]
            num_components = num_components_state.value
            coeffs = random_coeffs(num_components)
            img = generator(*coeffs)
            # Pad with zeros for unused sliders
            padded_coeffs = list(coeffs) + [0.0] * (max_components - num_components)
            return (*padded_coeffs, img)

        def switch_run(run_id: str) -> tuple[dict, dict, dict, int, str, np.ndarray, list[dict]]:
            """Switch to a different run and reload artifacts."""
            # Find metadata path for selected run
            metadata_path = None
            for rid, path in available_runs:
                if rid == run_id:
                    metadata_path = path
                    break
            
            if metadata_path is None:
                raise ValueError(f"Run {run_id} not found")
            
            # Load new artifacts
            model, metadata = load_artifacts(metadata_path)
            generator = make_generator(model, metadata)
            var_ratio = metadata["pca_var_ratio"]
            num_components = len(var_ratio)
            
            # Generate initial image with zero coefficients
            initial_img = generator(*([0.0] * num_components))
            
            # Update slider visibility and labels
            slider_updates = []
            for idx in range(max_components):
                if idx < num_components:
                    slider_updates.append(
                        gr.Slider.update(
                            label=f"PC {idx + 1} ({var_ratio[idx]*100:.1f}% var)",
                            value=0.0,
                            visible=True,
                        )
                    )
                else:
                    slider_updates.append(
                        gr.Slider.update(
                            label=f"PC {idx + 1} (unused)",
                            value=0.0,
                            visible=False,
                        )
                    )
            
            return (
                {"model": model},
                {"generator": generator},
                metadata,
                num_components,
                _format_run_info(metadata),
                initial_img,
                slider_updates,
            )

        # Wire up run switching
        run_dropdown.change(
            fn=switch_run,
            inputs=run_dropdown,
            outputs=[model_state, generator_state, metadata_state, num_components_state, run_info, image_output] + slider_rows,
            show_progress=True,
        )

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


def _format_run_info(metadata: dict) -> str:
    """Format run metadata for display."""
    run_id = metadata.get("run_id", "Unknown")
    image_size = metadata.get("image_size", "N/A")
    latent_dim = metadata.get("latent_dim", "N/A")
    loss_type = metadata.get("loss_type", "N/A")
    epochs = metadata.get("epochs", "N/A")
    
    info = f"""
**Current Run:** `{run_id}`

- **Image Size:** {image_size} (read from metadata)
- **Latent Dim:** {latent_dim}
- **Loss Type:** {loss_type}
- **Epochs:** {epochs}
"""
    return info


if __name__ == "__main__":
    launch()
