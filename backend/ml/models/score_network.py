"""Score network for learning the score function in diffusion models."""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal position embeddings for diffusion time.

    Maps scalar time t ∈ [0, 1] to a high-dimensional embedding
    using sinusoidal functions at different frequencies.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time embeddings.

        Args:
            t: Time values of shape (batch_size,) in [0, 1]

        Returns:
            Embeddings of shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        # Frequency scale
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / (half_dim - 1)
        )

        # Compute embeddings
        args = t[:, None] * freqs[None, :] * 1000
        embeddings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embeddings


class ConvBlock(nn.Module):
    """Convolutional block with GroupNorm, SiLU, and time conditioning."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU()

        # Residual connection if dimensions match
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time conditioning.

        Args:
            x: Input tensor (batch, channels, height, width)
            t_emb: Time embedding (batch, time_dim)

        Returns:
            Output tensor (batch, out_ch, height, width)
        """
        h = self.act(self.norm1(self.conv1(x)))

        # Add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]

        h = self.act(self.norm2(self.conv2(h)))

        return h + self.residual(x)


class ScoreNetwork(nn.Module):
    """U-Net style score network for lattice systems.

    Predicts the score s(x, t) ≈ ∇log p(x, t) for use in
    reverse diffusion sampling.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        time_embed_dim: int = 64,
        num_blocks: int = 3,
    ):
        """Initialize score network.

        Args:
            in_channels: Number of input channels (1 for Ising)
            base_channels: Base number of channels (doubled at each level)
            time_embed_dim: Dimension of time embedding
            num_blocks: Number of encoder/decoder blocks
        """
        super().__init__()

        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # Encoder
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = base_channels
        encoder_channels = [ch]

        for i in range(num_blocks):
            out_ch = base_channels * (2 ** (i + 1))
            self.encoder.append(ConvBlock(ch, out_ch, time_embed_dim))
            self.downsample.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            ch = out_ch
            encoder_channels.append(ch)

        # Middle
        self.middle = ConvBlock(ch, ch, time_embed_dim)

        # Decoder
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i in range(num_blocks - 1, -1, -1):
            out_ch = base_channels * (2**i) if i > 0 else base_channels
            self.upsample.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            # Input channels = upsampled + skip connection
            self.decoder.append(ConvBlock(ch + encoder_channels[i + 1], out_ch, time_embed_dim))
            ch = out_ch

        # Final convolution
        self.final = nn.Conv2d(base_channels, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict score s(x, t).

        Args:
            x: Noisy input (batch, channels, height, width)
            t: Diffusion time (batch,) in [0, 1]

        Returns:
            Score prediction (batch, channels, height, width)
        """
        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))

        # Initial conv
        h = self.init_conv(x)

        # Encoder (save skip connections)
        skips = [h]
        for block, down in zip(self.encoder, self.downsample):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        # Middle
        h = self.middle(h, t_emb)

        # Decoder (use skip connections)
        for up, block, skip in zip(self.upsample, self.decoder, reversed(skips[1:])):
            h = up(h)
            # Handle size mismatch from downsampling
            if h.shape[-2:] != skip.shape[-2:]:
                h = nn.functional.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        return self.final(h)


if __name__ == "__main__":
    # Test score network
    net = ScoreNetwork(in_channels=1, base_channels=32)

    # Test forward pass
    x = torch.randn(4, 1, 32, 32)
    t = torch.rand(4)
    score = net(x, t)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {score.shape}")
    assert score.shape == x.shape, "Score shape should match input"

    # Test different sizes
    for size in [16, 32, 64]:
        x = torch.randn(2, 1, size, size)
        t = torch.rand(2)
        score = net(x, t)
        print(f"Size {size}: output shape {score.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {n_params:,}")

    print("Score network tests passed!")
