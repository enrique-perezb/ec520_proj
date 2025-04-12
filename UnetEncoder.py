import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UNet3DEncoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32, latent_dim=128, pooling_layers=4):
        super(UNet3DEncoder, self).__init__()
        
        self.pooling_layers = pooling_layers
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_channels = in_channels
        for i in range(pooling_layers):
            self.enc_blocks.append(self._conv_block(prev_channels, base_channels * (2 ** i)))
            self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))
            prev_channels = base_channels * (2 ** i)
        
        self.fc_mu = None
        self.latent_dim = latent_dim

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.pooling_layers):
            x = self.enc_blocks[i](x)
            x = self.pools[i](x)

        x = torch.flatten(x, start_dim=1)

        if self.fc_mu is None:
            self.fc_mu = nn.Linear(x.shape[1], self.latent_dim).to(x.device)
            nn.init.kaiming_normal_(self.fc_mu.weight, nonlinearity='relu')
            if self.fc_mu.bias is not None:
                nn.init.constant_(self.fc_mu.bias, 0)

        latent = self.fc_mu(x)
        return latent


class UNet3DDecoder(nn.Module):
    def __init__(self, latent_dim=128, base_channels=32, output_shape=(45, 54, 45), pooling_layers=4):
        super(UNet3DDecoder, self).__init__()
        
        self.pooling_layers = pooling_layers
        self.fc = nn.Linear(latent_dim, base_channels * (2 ** (pooling_layers - 1)) * 4 * 4 * 4)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(base_channels * (2 ** (pooling_layers - 1)), 4, 4, 4))
        
        self.up_blocks = nn.ModuleList()
        prev_channels = base_channels * (2 ** (pooling_layers - 1))
        for i in range(pooling_layers - 1, -1, -1):
            self.up_blocks.append(self._conv_block(prev_channels, base_channels * (2 ** i)))
            prev_channels = base_channels * (2 ** i)
        
        self.final_conv = nn.Conv3d(base_channels, 1, kernel_size=1)
        self.output_shape = output_shape

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)

        for block in self.up_blocks:
            x = block(x)

        x = F.interpolate(x, size=self.output_shape, mode='trilinear', align_corners=False)
        x = self.final_conv(x)
        return x


class UNet3DAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, pooling_layers=4):
        super(UNet3DAutoencoder, self).__init__()
        self.encoder = UNet3DEncoder(latent_dim=latent_dim, pooling_layers=pooling_layers)
        self.decoder = UNet3DDecoder(latent_dim=latent_dim, output_shape=(45, 54, 45), pooling_layers=pooling_layers)

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon
