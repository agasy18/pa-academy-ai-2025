"""Convolutional variational autoencoder architecture + loss helpers."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int, negative_slope: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    )


def _deconv_block(in_ch: int, out_ch: int, negative_slope: float = 0.2) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=4,
            stride=2,
            padding=1,
        ),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
    )


class VAEOutput(NamedTuple):
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


class ConvVAE(nn.Module):
    """Convolutional VAE for small color faces.

    Architecture overview:
        * Encoder: strided conv blocks with BatchNorm + LeakyReLU progressively
          downsample the image (64→32→16→8→4). These four blocks learn local
          feature detectors and compress spatial information.
        * Latent heads: two linear layers (`fc_mu`, `fc_logvar`) map the flattened
          encoder features into the mean and log-variance of the latent Gaussian.
        * Decoder: a linear layer expands the latent vector back to the 4×4×C
          tensor, followed by three ConvTranspose blocks that mirror the encoder
          hierarchy (4→8→16→32→64). These layers reconstruct spatial structure.
        * Output layer: final ConvTranspose + Tanh projects to RGB in [-1, 1].
    """

    def __init__(
        self,
        latent_dim: int = 64,
        image_channels: int = 3,
        base_channels: int = 32,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        self.encoder_cnn = nn.Sequential(
            _conv_block(image_channels, base_channels),  # low-level edge/color filters
            _conv_block(base_channels, base_channels * 2),  # mid-level textures
            _conv_block(base_channels * 2, base_channels * 4),  # high-level parts
            _conv_block(base_channels * 4, base_channels * 8),  # compact 4x4 activations
        )

        with torch.no_grad():
            dummy = torch.zeros(1, image_channels, image_size, image_size)
            enc_shape = self.encoder_cnn(dummy)
        self.enc_channels = enc_shape.shape[1]
        self.enc_spatial = enc_shape.shape[2]
        self.flatten_dim = self.enc_channels * self.enc_spatial * self.enc_spatial

        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_cnn = nn.Sequential(
            _deconv_block(self.enc_channels, base_channels * 4),  # 8x8
            _deconv_block(base_channels * 4, base_channels * 2),  # 16x16
            _deconv_block(base_channels * 2, base_channels),  # 32x32
            nn.ConvTranspose2d(
                base_channels,
                image_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # 64x64
            nn.Tanh(),  # outputs in [-1, 1]
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_cnn(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_input(z)
        h = h.view(z.size(0), self.enc_channels, self.enc_spatial, self.enc_spatial)
        return self.decoder_cnn(h)

    def forward(self, x: torch.Tensor) -> VAEOutput:  # type: ignore[override]
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return VAEOutput(recon, mu, logvar)


class VAELoss(nn.Module):
    """Evidence Lower Bound (ELBO) loss with configurable beta.
    
    For images, MSE reconstruction loss can produce blurry results. Consider using
    PerceptualVAELoss for better visual quality, especially for faces.
    
    Args:
        beta: Weight for KL divergence term (higher = more regularization)
        use_log_mse: If True, use log(MSE) instead of MSE for reconstruction loss.
            Log(MSE) can produce sharper results by reweighting gradients and
            reducing the dominance of large errors. This is a valid alternative
            that sometimes performs better than standard MSE.
    """

    def __init__(self, beta: float = 1.0, use_log_mse: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.use_log_mse = use_log_mse

    def forward(self, output: VAEOutput, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        mse = F.mse_loss(output.reconstruction, target, reduction="mean")
        if self.use_log_mse:
            # Log(MSE) with small epsilon for numerical stability
            # This reweights gradients: d/dθ log(MSE) = (1/MSE) * d/dθ MSE
            # Large errors contribute less, small errors contribute more
            recon_loss = torch.log(mse + 1e-8)
        else:
            recon_loss = mse
        
        kl_div = -0.5 * torch.mean(
            1 + output.logvar - output.mu.pow(2) - output.logvar.exp()
        )
        return recon_loss + self.beta * kl_div


class PerceptualVAELoss(nn.Module):
    """Enhanced VAE loss using perceptual (feature-based) reconstruction loss.
    
    This loss uses features from a pre-trained VGG network instead of pixel-wise MSE,
    which typically produces sharper, more realistic reconstructions for images.
    The perceptual loss measures similarity in feature space rather than pixel space.
    
    Architecture:
        * Reconstruction: Perceptual loss (VGG features) + optional MSE weight
        * Regularization: KL divergence (same as standard VAE)
        * Total: recon_loss + beta * kl_div
    
    Args:
        beta: Weight for KL divergence term (higher = more regularization)
        perceptual_weight: Weight for perceptual loss component
        mse_weight: Optional weight for MSE loss (0 = perceptual only)
        feature_layer: Which VGG layer to use for features (default: 'relu3_3')
    """

    def __init__(
        self,
        beta: float = 1.0,
        perceptual_weight: float = 1.0,
        mse_weight: float = 0.0,
        feature_layer: str = "relu3_3",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.perceptual_weight = perceptual_weight
        self.mse_weight = mse_weight

        # Load pre-trained VGG16 and extract feature layers
        try:
            from torchvision.models import vgg16, VGG16_Weights
            import torchvision.transforms as transforms

            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            vgg.eval()  # Freeze VGG weights
            for param in vgg.parameters():
                param.requires_grad = False

            # Map layer names to indices
            layer_map = {
                "relu1_2": 3,
                "relu2_2": 8,
                "relu3_3": 15,
                "relu4_3": 22,
            }
            if feature_layer not in layer_map:
                raise ValueError(
                    f"feature_layer must be one of {list(layer_map.keys())}"
                )

            # Extract layers up to the desired feature layer
            # Register as submodule so it moves to GPU automatically
            self.feature_extractor = nn.Sequential(
                *list(vgg.children())[: layer_map[feature_layer] + 1]
            )
            
            # ImageNet normalization constants (VGG was trained with these)
            self.register_buffer(
                'mean',
                torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            )
            self.register_buffer(
                'std',
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            )
        except ImportError:
            raise ImportError(
                "torchvision is required for PerceptualVAELoss. "
                "Install with: pip install torchvision"
            )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract VGG features from input images.
        
        Args:
            x: Input tensor in range [-1, 1] (from Tanh output)
        """
        # Step 1: Convert from [-1, 1] to [0, 1]
        x_normalized = (x + 1.0) / 2.0
        x_normalized = x_normalized.clamp(0.0, 1.0)
        
        # Step 2: Apply ImageNet normalization (critical for VGG!)
        # VGG was trained with these statistics, so we must match them
        x_imagenet = (x_normalized - self.mean) / self.std
        
        # Step 3: Extract features
        return self.feature_extractor(x_imagenet)

    def forward(self, output: VAEOutput, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Perceptual loss: compare features from VGG
        recon_features = self._extract_features(output.reconstruction)
        target_features = self._extract_features(target)
        perceptual_loss = F.mse_loss(
            recon_features, target_features, reduction="mean"
        )
        
        # Note: Perceptual loss is typically much smaller than pixel MSE
        # The perceptual_weight parameter controls this scaling

        # Optional pixel-wise MSE
        mse_loss = (
            F.mse_loss(output.reconstruction, target, reduction="mean")
            if self.mse_weight > 0
            else torch.tensor(0.0, device=target.device)
        )

        # KL divergence (same as standard VAE)
        kl_div = -0.5 * torch.mean(
            1 + output.logvar - output.mu.pow(2) - output.logvar.exp()
        )

        recon_loss = (
            self.perceptual_weight * perceptual_loss + self.mse_weight * mse_loss
        )
        return recon_loss + self.beta * kl_div


class ReconstructionMSE(nn.Module):
    """Metric: mean squared error between reconstructions and inputs."""

    def forward(self, output: VAEOutput, target: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return F.mse_loss(output.reconstruction, target, reduction="mean")
