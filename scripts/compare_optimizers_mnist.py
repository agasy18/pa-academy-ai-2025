import math
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "slides" / "assets"


class MNISTMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(x)


def get_data_loaders(batch_size: int = 128, subset_size: int = 20000) -> DataLoader:
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root=str(ROOT / "notebooks" / "data"), train=True, download=True, transform=transform)

    if subset_size is not None and subset_size < len(train_ds):
        indices = list(range(subset_size))
        train_ds = Subset(train_ds, indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    return train_loader


def train_one_optimizer(
    name: str,
    optimizer_ctor,
    train_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 3,
) -> List[float]:
    torch.manual_seed(0)
    model = MNISTMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optimizer_ctor(model.parameters())

    epoch_losses: List[float] = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        epoch_losses.append(avg_loss)
        print(f"{name} - epoch {epoch+1}/{num_epochs}, loss={avg_loss:.4f}")

    return epoch_losses


def plot_losses(losses: Dict[str, List[float]], out_path: Path) -> None:
    epochs = range(1, len(next(iter(losses.values()))) + 1)
    plt.figure(figsize=(6, 4))
    for name, vals in losses.items():
        plt.plot(epochs, vals, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Optimizer comparison on MNIST (training loss)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = get_data_loaders(batch_size=128, subset_size=20000)

    num_epochs = 30
    losses: Dict[str, List[float]] = {}

    losses["SGD (lr=0.1)"] = train_one_optimizer(
        "SGD",
        lambda params: torch.optim.SGD(params, lr=0.1),
        train_loader,
        device,
        num_epochs=num_epochs,
    )

    losses["SGD+Momentum (lr=0.1, 0.9)"] = train_one_optimizer(
        "SGD+Momentum",
        lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9),
        train_loader,
        device,
        num_epochs=num_epochs,
    )

    losses["Adam (lr=1e-3)"] = train_one_optimizer(
        "Adam",
        lambda params: torch.optim.Adam(params, lr=1e-3),
        train_loader,
        device,
        num_epochs=num_epochs,
    )

    out_path = ASSETS_DIR / "mnist_optimizer_comparison.png"
    plot_losses(losses, out_path)
    print(f"Saved comparison plot to {out_path}")


if __name__ == "__main__":
    main()
