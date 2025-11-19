import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ASSETS_DIR = Path(__file__).resolve().parents[1] / "slides" / "assets"


def save_activation(name: str, f, df, xlim=(-5, 5), ylim=None):
    xs = np.linspace(xlim[0], xlim[1], 400)
    ys = f(xs)
    dys = df(xs)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].plot(xs, ys)
    axes[0].set_title(name)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].axvline(0, color="black", linewidth=0.5)
    if ylim is not None:
        axes[0].set_ylim(*ylim)

    axes[1].plot(xs, dys)
    axes[1].set_title(f"{name} derivative")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].axvline(0, color="black", linewidth=0.5)

    fig.tight_layout()
    out_path = ASSETS_DIR / f"{name.lower()}_activation.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    # ReLU
    def relu(x):
        return np.maximum(0.0, x)

    def drelu(x):
        return (x > 0).astype(float)

    save_activation("ReLU", relu, drelu, ylim=(-0.5, 5))

    # Sigmoid
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def dsigmoid(x):
        s = sigmoid(x)
        return s * (1.0 - s)

    save_activation("Sigmoid", sigmoid, dsigmoid, ylim=(-0.1, 1.1))

    # Tanh
    def tanh(x):
        return np.tanh(x)

    def dtanh(x):
        t = tanh(x)
        return 1.0 - t**2

    save_activation("Tanh", tanh, dtanh, ylim=(-1.5, 1.5))

    # LeakyReLU
    alpha = 0.1

    def leaky_relu(x):
        return np.where(x > 0, x, alpha * x)

    def dleaky_relu(x):
        return np.where(x > 0, 1.0, alpha)

    save_activation("LeakyReLU", leaky_relu, dleaky_relu, ylim=(-1, 5))

    # ELU
    elu_alpha = 1.0

    def elu(x):
        return np.where(x > 0, x, elu_alpha * (np.exp(x) - 1.0))

    def delu(x):
        return np.where(x > 0, 1.0, elu_alpha * np.exp(x))

    save_activation("ELU", elu, delu, ylim=(-2, 5))

    # GELU (approximate) — used in transformers
    def gelu(x):
        return 0.5 * x * (
            1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * np.power(x, 3)))
        )

    def dgelu(x):
        eps = 1e-3
        return (gelu(x + eps) - gelu(x - eps)) / (2 * eps)

    save_activation("GELU", gelu, dgelu, ylim=(-1, 5))


if __name__ == "__main__":
    main()
