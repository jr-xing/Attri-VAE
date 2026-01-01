# -*- coding: utf-8 -*-
"""
2D Convolutional VAE for cardiac MRI slice reconstruction.

This is a 2D adaptation of the 3D ConvVAE from model/model.py.
Instead of processing 80x80x80 volumes, this model processes individual
80x80 slices, which is more appropriate for the limited dataset size
and the fact that the original data is essentially 2D slices.

Architecture:
- Encoder: 5 Conv2d layers with stride-2 downsampling → FC layers → mu/logvar
- Decoder: FC layer → 4 ConvTranspose2d layers with stride-2 upsampling
- MLP branch for classification (optional, for fine-tuning stage)

Input: (batch, 1, 80, 80)
Latent: (batch, latent_size)
Output: (batch, 1, 80, 80)

Author: Claude
Date: 2025-12-31
"""

import torch
from torch.nn import functional as F
import torch.nn as nn


def initialize_weights(m):
    """Initialize network weights using Xavier initialization."""
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


# Bottleneck configuration for 2D
UNFLATTEN_CHANNEL = 2
DIM_START_UP_DECODER = [5, 5]  # 2D spatial dimensions at bottleneck


class Flatten(nn.Module):
    """Flatten tensor to (batch, -1)."""
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten2D(nn.Module):
    """Unflatten tensor to (batch, channels, H, W)."""
    def __init__(self, channels=UNFLATTEN_CHANNEL, spatial_dims=DIM_START_UP_DECODER):
        super().__init__()
        self.channels = channels
        self.h = spatial_dims[0]
        self.w = spatial_dims[1]

    def forward(self, input):
        return input.view(input.size(0), self.channels, self.h, self.w)


class ConvVAE2D(nn.Module):
    """2D Convolutional Variational Autoencoder.

    This model processes 2D cardiac MRI slices (80x80) instead of 3D volumes.
    It has the same general structure as the 3D version but with:
    - Conv2d instead of Conv3d
    - BatchNorm2d instead of BatchNorm3d
    - Smaller bottleneck (2x5x5=50 values vs 2x5x5x5=250)

    Args:
        image_channels: Number of input channels (1 for grayscale)
        h_dim: Hidden dimension before latent space
        latent_size: Size of latent space
        n_filters_ENC: List of encoder filter counts [8, 16, 32, 64, 2]
        n_filters_DEC: List of decoder filter counts [64, 32, 16, 8, 4, 2]
    """

    def __init__(
        self,
        image_channels: int = 1,
        h_dim: int = 64,
        latent_size: int = 16,
        n_filters_ENC: list = None,
        n_filters_DEC: list = None,
    ):
        super(ConvVAE2D, self).__init__()

        # Default filter configurations
        if n_filters_ENC is None:
            n_filters_ENC = [8, 16, 32, 64, 2]
        if n_filters_DEC is None:
            n_filters_DEC = [64, 32, 16, 8, 4, 2]

        self.image_channels = image_channels
        self.h_dim = h_dim
        self.latent_size = latent_size
        self.n_filters_ENC = n_filters_ENC
        self.n_filters_DEC = n_filters_DEC

        # Calculate bottleneck size: 2 channels * 5 * 5 = 50
        self.bottleneck_size = n_filters_ENC[-1] * DIM_START_UP_DECODER[0] * DIM_START_UP_DECODER[1]

        ##############
        ## ENCODER ##
        ##############
        # Input: (1, 80, 80) -> Output: (2, 5, 5)

        # Conv1: (1, 80, 80) -> (8, 40, 40)
        self.conv1_enc = nn.Conv2d(
            in_channels=image_channels,
            out_channels=n_filters_ENC[0],
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn1_enc = nn.BatchNorm2d(n_filters_ENC[0])

        # Conv2: (8, 40, 40) -> (16, 20, 20)
        self.conv2_enc = nn.Conv2d(
            in_channels=n_filters_ENC[0],
            out_channels=n_filters_ENC[1],
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn2_enc = nn.BatchNorm2d(n_filters_ENC[1])

        # Conv3: (16, 20, 20) -> (32, 10, 10)
        self.conv3_enc = nn.Conv2d(
            in_channels=n_filters_ENC[1],
            out_channels=n_filters_ENC[2],
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn3_enc = nn.BatchNorm2d(n_filters_ENC[2])

        # Conv4: (32, 10, 10) -> (64, 5, 5)
        self.conv4_enc = nn.Conv2d(
            in_channels=n_filters_ENC[2],
            out_channels=n_filters_ENC[3],
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.bn4_enc = nn.BatchNorm2d(n_filters_ENC[3])

        # Conv5: (64, 5, 5) -> (2, 5, 5) - channel reduction without spatial downsampling
        self.conv5_enc = nn.Conv2d(
            in_channels=n_filters_ENC[3],
            out_channels=n_filters_ENC[4],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn5_enc = nn.BatchNorm2d(n_filters_ENC[4])

        # Flatten and FC layers
        self.flatten = Flatten()

        # FC1: 50 -> 64
        self.fc1 = nn.Linear(self.bottleneck_size, 64)

        # FC2: 64 -> h_dim
        self.fc2 = nn.Linear(64, h_dim)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Latent space: h_dim -> latent_size
        self.mu = nn.Linear(h_dim, latent_size)
        self.logvar = nn.Linear(h_dim, latent_size)

        # MLP branch for classification (for fine-tuning stage)
        self.mlp1 = nn.Linear(latent_size, max(1, latent_size // 2))
        self.bn1_mlp = nn.BatchNorm1d(max(1, latent_size // 2))
        self.mlp2 = nn.Linear(max(1, latent_size // 2), max(1, latent_size // 4))
        self.bn2_mlp = nn.BatchNorm1d(max(1, latent_size // 4))
        self.mlp3 = nn.Linear(max(1, latent_size // 4), 1)
        self.sigmoid_mlp = nn.Sigmoid()

        ###############
        ### DECODER ###
        ###############
        # Input: latent_size -> Output: (1, 80, 80)

        # FC3: latent_size -> bottleneck_size (50)
        self.fc3 = nn.Linear(latent_size, self.bottleneck_size)

        # Unflatten: 50 -> (2, 5, 5)
        self.unflatten = Unflatten2D(n_filters_ENC[-1], DIM_START_UP_DECODER)

        # Conv1_dec: (2, 5, 5) -> (64, 5, 5) - channel expansion
        self.conv1_dec = nn.Conv2d(
            in_channels=n_filters_ENC[-1],
            out_channels=n_filters_DEC[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1_dec = nn.BatchNorm2d(n_filters_DEC[0])

        # Deconv1: (64, 5, 5) -> (64, 10, 10)
        self.deconv1_dec = nn.ConvTranspose2d(
            in_channels=n_filters_DEC[0],
            out_channels=n_filters_DEC[0],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.bn2_dec = nn.BatchNorm2d(n_filters_DEC[0])

        # Deconv2: (64, 10, 10) -> (32, 20, 20)
        self.deconv2_dec = nn.ConvTranspose2d(
            in_channels=n_filters_DEC[0],
            out_channels=n_filters_DEC[1],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.bn3_dec = nn.BatchNorm2d(n_filters_DEC[1])

        # Deconv3: (32, 20, 20) -> (16, 40, 40)
        self.deconv3_dec = nn.ConvTranspose2d(
            in_channels=n_filters_DEC[1],
            out_channels=n_filters_DEC[2],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.bn4_dec = nn.BatchNorm2d(n_filters_DEC[2])

        # Deconv4: (16, 40, 40) -> (8, 80, 80)
        self.deconv4_dec = nn.ConvTranspose2d(
            in_channels=n_filters_DEC[2],
            out_channels=n_filters_DEC[3],
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )
        self.bn5_dec = nn.BatchNorm2d(n_filters_DEC[3])

        # Conv2_dec: (8, 80, 80) -> (4, 80, 80)
        self.conv2_dec = nn.Conv2d(
            in_channels=n_filters_DEC[3],
            out_channels=n_filters_DEC[4],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn6_dec = nn.BatchNorm2d(n_filters_DEC[4])

        # Conv3_dec: (4, 80, 80) -> (1, 80, 80)
        self.conv3_dec = nn.Conv2d(
            in_channels=n_filters_DEC[4],
            out_channels=image_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor (batch, 1, 80, 80)

        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z_distribution: torch.distributions.Normal object
        """
        h = F.relu(self.bn1_enc(self.conv1_enc(x)))
        h = F.relu(self.bn2_enc(self.conv2_enc(h)))
        h = F.relu(self.bn3_enc(self.conv3_enc(h)))
        h = F.relu(self.bn4_enc(self.conv4_enc(h)))
        h = F.relu(self.bn5_enc(self.conv5_enc(h)))

        h = self.dropout(self.flatten(h))

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        mu, logvar = self.mu(h), self.logvar(h)

        # Define the distribution from mu and logvar
        z_distribution = torch.distributions.Normal(
            loc=mu,
            scale=torch.exp(0.5 * logvar)  # std = exp(0.5 * logvar)
        )
        return mu, logvar, z_distribution

    def decode(self, z):
        """Decode latent vector to reconstructed image.

        Args:
            z: Latent vector (batch, latent_size)

        Returns:
            Reconstructed image (batch, 1, 80, 80)
        """
        z = F.relu(self.fc3(z))
        z = self.unflatten(z)

        z = F.relu(self.bn1_dec(self.conv1_dec(z)))
        z = F.relu(self.bn2_dec(self.deconv1_dec(z)))
        z = F.relu(self.bn3_dec(self.deconv2_dec(z)))
        z = F.relu(self.bn4_dec(self.deconv3_dec(z)))
        z = F.relu(self.bn5_dec(self.deconv4_dec(z)))
        z = F.relu(self.bn6_dec(self.conv2_dec(z)))
        z = self.conv3_dec(z)

        z = self.sigmoid(z)

        return z

    def mlp_predict(self, z):
        """MLP branch for classification.

        Args:
            z: Latent vector (batch, latent_size)

        Returns:
            Classification probability (batch, 1)
        """
        out_mlp = F.relu(self.bn1_mlp(self.mlp1(z)))
        out_mlp = F.relu(self.bn2_mlp(self.mlp2(out_mlp)))
        out_mlp = self.sigmoid_mlp(self.mlp3(out_mlp))
        return out_mlp

    def reparameterize(self, mu, logvar, z_dist):
        """Reparameterization trick for sampling.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            z_dist: torch.distributions.Normal object

        Returns:
            z_tilde: Sampled latent vector (from distribution)
            z_sampled_eq: Sampled latent vector (manual implementation)
            z_prior: Sample from prior
            prior_dist: Prior distribution object
        """
        # Manual reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_sampled_eq = eps.mul(std).add_(mu)

        # Prior distribution (standard normal)
        prior_dist = torch.distributions.Normal(
            loc=torch.zeros_like(z_dist.loc),
            scale=torch.ones_like(z_dist.scale)
        )
        z_prior = prior_dist.sample()

        # Sample from the encoder distribution
        z_tilde = z_dist.rsample()

        return z_tilde, z_sampled_eq, z_prior, prior_dist

    def forward(self, x):
        """Forward pass through the VAE.

        Args:
            x: Input tensor (batch, 1, 80, 80)

        Returns:
            output: Reconstructed image (batch, 1, 80, 80)
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            out_mlp: Classification output
            z_sampled_eq: Sampled latent (manual)
            z_prior: Sample from prior
            prior_dist: Prior distribution
            z_tilde: Sampled latent (from distribution)
            z_dist: Encoder distribution
        """
        mu, logvar, z_dist = self.encode(x)
        z_tilde, z_sampled_eq, z_prior, prior_dist = self.reparameterize(mu, logvar, z_dist)
        out_mlp = self.mlp_predict(z_tilde)
        output = self.decode(z_tilde)

        return output, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist


# Testing code
if __name__ == "__main__":
    # Test the model
    print("Testing ConvVAE2D...")

    # Create model
    model = ConvVAE2D(
        image_channels=1,
        h_dim=64,
        latent_size=16,
        n_filters_ENC=[8, 16, 32, 64, 2],
        n_filters_DEC=[64, 32, 16, 8, 4, 2],
    )

    # Initialize weights
    model.apply(initialize_weights)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")

    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 80, 80)
    print(f"Input shape: {x.shape}")

    output, mu, logvar, out_mlp, z_sampled_eq, z_prior, prior_dist, z_tilde, z_dist = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Latent shape (mu): {mu.shape}")
    print(f"MLP output shape: {out_mlp.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    print("\nModel test passed!")
