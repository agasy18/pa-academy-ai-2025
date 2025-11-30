# Face Autoencoder Teaching Demo

This mini-project demonstrates how to train a convolutional variational
autoencoder (VAE) on real faces (LFW dataset), analyze the latent space with
PCA, and expose interactive controls through a Gradio UI so learners can
manipulate high-level facial features with sliders.

## Repository Layout

```
projects/face_autoencoder
├── assets/        # Saved figures (reconstruction grids, latent traversals)
├── artifacts/     # Model checkpoints, PCA metadata, experiment config
├── data/          # Downloaded datasets managed via sklearn helpers
├── logs/          # FastAI CSV logs + console captures
└── src/           # Training + app sources
```

## Usage

1. **Train the VAE + PCA artifacts**: open `projects/face_autoencoder/face_autoencoder_training.ipynb`
   in Jupyter/Lab, tweak the hyperparameters in the configuration cell, and run all
   cells (environment → imports → config → training → analysis). The notebook will
   download LFW as needed, train the model, and export artifacts.

2. **Launch the interactive app** (uses trained weights + PCA sliders):

   ```bash
   python -m projects.face_autoencoder.src.app
   ```

Each training run now creates a dedicated folder under
`projects/face_autoencoder/artifacts/<RUN_ID>/` containing the best model
(`model.pth`), training history (`history.csv`), PCA stats (`pca.npz`), and run
metadata. Visual diagnostics (reconstruction grids, loss curves, latent traversal)
are rendered inline inside the notebook, so only the artifacts required for the
app are persisted. Point the Gradio app to any run by passing the path to its
`metadata.json`, or let it pick the most recent run automatically.

### Notebook Workflow

All training now flows through the notebook (`projects/face_autoencoder/face_autoencoder_training.ipynb`).
It mirrors the previous CLI behavior, exposes hyperparameters at the top, and logs
artifacts to the same per-run folders while showing the diagnostics inline. Run the
cells sequentially (environment setup → imports → config → training → analysis) to
reproduce the full pipeline interactively.

The scripts are heavily commented and designed to be inspected in class or as a
homework extension. Learners can tweak hyper-parameters (including `--image-size`
and `--lfw-resize` to use larger faces), re-run the training, and immediately see
how the PCA sliders alter generated faces.
