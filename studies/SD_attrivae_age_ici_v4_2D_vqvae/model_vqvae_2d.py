# VQ-VAE Model for 2D Cardiac MRI
# Adapted from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vq_vae.py
# Modified for grayscale cardiac MRI (80x80)

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Union


class VectorQuantizer(nn.Module):
    """
    Vector Quantizer module for VQ-VAE.

    Maps continuous latent vectors to discrete codebook entries.

    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 beta: float = 0.25):
        """
        Args:
            num_embeddings: Number of codebook entries (K)
            embedding_dim: Dimension of each codebook vector (D)
            beta: Commitment loss weight
        """
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings
        self.D = embedding_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latents: Encoder output [B x D x H x W]

        Returns:
            quantized_latents: Quantized latent codes [B x D x H x W]
            vq_loss: Vector quantization loss (commitment + embedding)
        """
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents (straight-through estimator)
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]

    def get_codebook_indices(self, latents: torch.Tensor) -> torch.Tensor:
        """Get codebook indices for visualization/analysis."""
        latents = latents.permute(0, 2, 3, 1).contiguous()
        flat_latents = latents.view(-1, self.D)

        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())

        encoding_inds = torch.argmin(dist, dim=1)
        return encoding_inds.view(latents.shape[0], latents.shape[1], latents.shape[2])


class ResidualLayer(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.resblock(input)


class VQVAE2D(nn.Module):
    """
    VQ-VAE for 2D cardiac MRI images.

    Architecture:
    - Encoder: Conv layers with residual blocks
    - Vector Quantizer: Maps to discrete codebook
    - Decoder: Transposed conv layers with residual blocks

    For 80x80 input with hidden_dims=[128, 256]:
    - After encoder: 20x20 feature map
    - Codebook: 512 vectors of dimension 64
    """

    def __init__(self,
                 in_channels: int = 1,
                 embedding_dim: int = 64,
                 num_embeddings: int = 512,
                 hidden_dims: List[int] = None,
                 beta: float = 0.25,
                 img_size: int = 80,
                 **kwargs) -> None:
        """
        Args:
            in_channels: Number of input channels (1 for grayscale)
            embedding_dim: Dimension of codebook vectors
            num_embeddings: Number of codebook entries
            hidden_dims: Channel progression for encoder/decoder
            beta: Commitment loss weight
            img_size: Input image size (assumes square)
        """
        super(VQVAE2D, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.in_channels = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [128, 256]

        self.hidden_dims = hidden_dims.copy()

        # Build Encoder
        curr_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(curr_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU())
            )
            curr_channels = h_dim

        modules.append(
            nn.Sequential(
                nn.Conv2d(curr_channels, curr_channels,
                          kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(curr_channels, curr_channels))
        modules.append(nn.LeakyReLU())

        modules.append(
            nn.Sequential(
                nn.Conv2d(curr_channels, embedding_dim,
                          kernel_size=1, stride=1),
                nn.LeakyReLU())
        )

        self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer(num_embeddings,
                                        embedding_dim,
                                        self.beta)

        # Build Decoder
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(embedding_dim,
                          hidden_dims[-1],
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.LeakyReLU())
        )

        for _ in range(6):
            modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        modules.append(nn.LeakyReLU())

        # Reverse hidden_dims for decoder (use a copy to avoid mutating the original)
        hidden_dims_reversed = hidden_dims[::-1]

        for i in range(len(hidden_dims_reversed) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims_reversed[i],
                                       hidden_dims_reversed[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.LeakyReLU())
            )

        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims_reversed[-1],
                                   out_channels=in_channels,
                                   kernel_size=4,
                                   stride=2, padding=1),
                nn.Sigmoid())  # Sigmoid for [0,1] range (grayscale cardiac MRI)
        )

        self.decoder = nn.Sequential(*modules)

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network.

        Args:
            input: Input tensor [N x C x H x W]

        Returns:
            Latent codes before quantization [N x D x H' x W']
        """
        return self.encoder(input)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes onto the image space.

        Args:
            z: Quantized latent codes [B x D x H x W]

        Returns:
            Reconstructed image [B x C x H x W]
        """
        return self.decoder(z)

    def forward(self, input: torch.Tensor, **kwargs) -> List[torch.Tensor]:
        """
        Forward pass through VQ-VAE.

        Args:
            input: Input image [B x C x H x W]

        Returns:
            List containing:
            - recons: Reconstructed image [B x C x H x W]
            - input: Original input [B x C x H x W]
            - vq_loss: Vector quantization loss
        """
        encoding = self.encode(input)
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode(quantized_inputs), input, vq_loss]

    def loss_function(self,
                      recons: torch.Tensor,
                      input: torch.Tensor,
                      vq_loss: torch.Tensor,
                      **kwargs) -> dict:
        """
        Computes the VQ-VAE loss.

        Args:
            recons: Reconstructed image
            input: Original input
            vq_loss: Vector quantization loss from VQ layer

        Returns:
            Dictionary with loss components
        """
        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {
            'loss': loss,
            'Reconstruction_Loss': recons_loss,
            'VQ_Loss': vq_loss
        }

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image.

        Args:
            x: Input image [B x C x H x W]

        Returns:
            Reconstructed image [B x C x H x W]
        """
        return self.forward(x)[0]

    def get_codebook_indices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get codebook indices for an input image.

        Args:
            x: Input image [B x C x H x W]

        Returns:
            Codebook indices [B x H' x W']
        """
        encoding = self.encode(x)
        return self.vq_layer.get_codebook_indices(encoding)


def get_model(config: dict) -> VQVAE2D:
    """
    Factory function to create VQVAE2D model from config.

    Args:
        config: Model configuration dictionary

    Returns:
        VQVAE2D model instance
    """
    model_config = config.get('model', config)

    return VQVAE2D(
        in_channels=model_config.get('image_channels', 1),
        embedding_dim=model_config.get('embedding_dim', 64),
        num_embeddings=model_config.get('num_embeddings', 512),
        hidden_dims=model_config.get('hidden_dims', [128, 256]),
        beta=model_config.get('beta', 0.25),
        img_size=model_config.get('img_size', 80)
    )


if __name__ == '__main__':
    # Test the model
    model = VQVAE2D(
        in_channels=1,
        embedding_dim=64,
        num_embeddings=512,
        hidden_dims=[128, 256],
        beta=0.25,
        img_size=80
    )

    # Test forward pass
    x = torch.randn(4, 1, 80, 80)
    recons, input_img, vq_loss = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recons.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")

    # Test loss function
    loss_dict = model.loss_function(recons, input_img, vq_loss)
    print(f"Total Loss: {loss_dict['loss'].item():.4f}")
    print(f"Recon Loss: {loss_dict['Reconstruction_Loss'].item():.4f}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
