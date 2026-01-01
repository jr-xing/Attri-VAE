# -*- coding: utf-8 -*-
"""
Deterministic Autoencoder for debugging VAE reconstruction issues.

This module provides a deterministic autoencoder with the same architecture
as ConvVAE but without the variational components (no mu/logvar split,
no reparameterization, no KL divergence).

Purpose: If VAE produces blurry reconstructions, train this AE first to
isolate whether the problem is:
- Architectural (AE also blurry) -> model capacity or data issue
- VAE-specific (AE is sharp) -> KL weighting, posterior collapse

Author: Claude
Date: 2025-12-30
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(m):
    """Initialize weights using Xavier uniform."""
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


# Fixed unflatten parameters (same as ConvVAE)
unflatten_channel = 2
dim_start_up_decoder = [5, 5, 5]


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def forward(self, input, size=unflatten_channel):
        return input.view(
            input.size(0),
            size,
            dim_start_up_decoder[0],
            dim_start_up_decoder[1],
            dim_start_up_decoder[2],
        )


class ConvAE(nn.Module):
    """
    Deterministic 3D Convolutional Autoencoder.

    Same architecture as ConvVAE but without variational components:
    - No mu/logvar split (single latent vector)
    - No reparameterization trick
    - No KL divergence loss
    - No classification MLP branch

    This provides a baseline to test if the encoder/decoder architecture
    itself can produce sharp reconstructions.
    """

    def __init__(
        self,
        image_channels: int = 1,
        h_dim: int = 96,
        latent_size: int = 64,
        n_filters_ENC: list = None,
        n_filters_DEC: list = None,
    ):
        super(ConvAE, self).__init__()

        if n_filters_ENC is None:
            n_filters_ENC = [8, 16, 32, 64, 2]
        if n_filters_DEC is None:
            n_filters_DEC = [64, 32, 16, 8, 4, 2]

        self.image_channels = image_channels
        self.h_dim = h_dim
        self.latent_size = latent_size
        self.n_filters_ENC = n_filters_ENC
        self.n_filters_DEC = n_filters_DEC

        # ==============
        # ENCODER
        # ==============
        # Conv layers: 80 -> 40 -> 20 -> 10 -> 5 -> 5
        self.conv1_enc = nn.Conv3d(
            in_channels=image_channels,
            out_channels=n_filters_ENC[0],
            kernel_size=[3, 3, 3],
            stride=2,
            padding=1,
        )
        self.bn1_enc = nn.BatchNorm3d(n_filters_ENC[0])

        self.conv2_enc = nn.Conv3d(
            in_channels=n_filters_ENC[0],
            out_channels=n_filters_ENC[1],
            kernel_size=[3, 3, 3],
            stride=2,
            padding=1,
        )
        self.bn2_enc = nn.BatchNorm3d(n_filters_ENC[1])

        self.conv3_enc = nn.Conv3d(
            in_channels=n_filters_ENC[1],
            out_channels=n_filters_ENC[2],
            kernel_size=[3, 3, 3],
            stride=2,
            padding=1,
        )
        self.bn3_enc = nn.BatchNorm3d(n_filters_ENC[2])

        self.conv4_enc = nn.Conv3d(
            in_channels=n_filters_ENC[2],
            out_channels=n_filters_ENC[3],
            kernel_size=[3, 3, 3],
            stride=2,
            padding=1,
        )
        self.bn4_enc = nn.BatchNorm3d(n_filters_ENC[3])

        self.conv5_enc = nn.Conv3d(
            in_channels=n_filters_ENC[3],
            out_channels=n_filters_ENC[4],
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1,
        )
        self.bn5_enc = nn.BatchNorm3d(n_filters_ENC[4])

        self.flatten = Flatten()

        # FC layers to bottleneck
        # After conv5: 2 * 5 * 5 * 5 = 250
        self.fc1 = nn.Linear(250, 128)
        self.fc2 = nn.Linear(128, h_dim)

        # Single latent projection (deterministic, no mu/logvar split)
        self.latent = nn.Linear(h_dim, latent_size)

        # ==============
        # DECODER
        # ==============
        self.fc3 = nn.Linear(latent_size, 250)
        self.unflatten = Unflatten()

        self.conv1_dec = nn.Conv3d(
            in_channels=unflatten_channel,
            out_channels=n_filters_DEC[0],
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1,
        )
        self.bn1_dec = nn.BatchNorm3d(n_filters_DEC[0])

        self.deconv1_dec = nn.ConvTranspose3d(
            in_channels=n_filters_DEC[0],
            out_channels=n_filters_DEC[0],
            kernel_size=[3, 3, 3],
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn2_dec = nn.BatchNorm3d(n_filters_DEC[0])

        self.deconv2_dec = nn.ConvTranspose3d(
            in_channels=n_filters_DEC[0],
            out_channels=n_filters_DEC[1],
            kernel_size=[3, 3, 3],
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn3_dec = nn.BatchNorm3d(n_filters_DEC[1])

        self.deconv3_dec = nn.ConvTranspose3d(
            in_channels=n_filters_DEC[1],
            out_channels=n_filters_DEC[2],
            kernel_size=[3, 3, 3],
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn4_dec = nn.BatchNorm3d(n_filters_DEC[2])

        self.deconv4_dec = nn.ConvTranspose3d(
            in_channels=n_filters_DEC[2],
            out_channels=n_filters_DEC[3],
            kernel_size=[3, 3, 3],
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.bn5_dec = nn.BatchNorm3d(n_filters_DEC[3])

        self.conv2_dec = nn.Conv3d(
            in_channels=n_filters_DEC[3],
            out_channels=n_filters_DEC[4],
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1,
        )
        self.bn6_dec = nn.BatchNorm3d(n_filters_DEC[4])

        self.conv3_dec = nn.Conv3d(
            in_channels=n_filters_DEC[4],
            out_channels=image_channels,
            kernel_size=[3, 3, 3],
            stride=1,
            padding=1,
        )
        self.bn7_dec = nn.BatchNorm3d(image_channels)

        self.sigmoid = nn.Sigmoid()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x: Input tensor of shape (B, 1, 80, 80, 80)

        Returns:
            Latent tensor of shape (B, latent_size)
        """
        h = F.relu(self.bn1_enc(self.conv1_enc(x)))
        h = F.relu(self.bn2_enc(self.conv2_enc(h)))
        h = F.relu(self.bn3_enc(self.conv3_enc(h)))
        h = F.relu(self.bn4_enc(self.conv4_enc(h)))
        h = F.relu(self.bn5_enc(self.conv5_enc(h)))

        h = self.flatten(h)

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))

        z = self.latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstructed image.

        Args:
            z: Latent tensor of shape (B, latent_size)

        Returns:
            Reconstructed tensor of shape (B, 1, 80, 80, 80)
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

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (B, 1, 80, 80, 80)

        Returns:
            Tuple of (reconstructed, latent):
                - reconstructed: shape (B, 1, 80, 80, 80)
                - latent: shape (B, latent_size)
        """
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z
