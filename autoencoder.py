import math

import torch
import torch.nn as nn


class SimpleAutoencoder(nn.Module):
    def __init__(self, in_channels=3, latent_channels=4, downsample_factor=4):
        super().__init__()
        if downsample_factor not in (2, 4, 8):
            raise ValueError("downsample_factor must be one of {2, 4, 8}.")
        num_down = int(math.log2(downsample_factor))

        encoder_layers = []
        channels = 64
        encoder_layers.append(nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1))
        encoder_layers.append(nn.GroupNorm(8, channels))
        encoder_layers.append(nn.SiLU())
        for _ in range(num_down):
            encoder_layers.append(nn.Conv2d(channels, channels * 2, kernel_size=4, stride=2, padding=1))
            channels *= 2
            encoder_layers.append(nn.GroupNorm(8, channels))
            encoder_layers.append(nn.SiLU())
        encoder_layers.append(nn.Conv2d(channels, latent_channels, kernel_size=3, stride=1, padding=1))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        decoder_layers.append(nn.Conv2d(latent_channels, channels, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.GroupNorm(8, channels))
        decoder_layers.append(nn.SiLU())
        for _ in range(num_down):
            decoder_layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            decoder_layers.append(nn.Conv2d(channels, channels // 2, kernel_size=3, stride=1, padding=1))
            channels //= 2
            decoder_layers.append(nn.GroupNorm(8, channels))
            decoder_layers.append(nn.SiLU())
        decoder_layers.append(nn.Conv2d(channels, in_channels, kernel_size=3, stride=1, padding=1))
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
