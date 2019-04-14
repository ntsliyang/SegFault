"""
    Autoencoder hashing function

    Architecture:
        Input: downsampled grayscale four continuous images with size (1 * 52 * 52)
                Input size: (4 * 52 * 52)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, Linear, BatchNorm2d
import torch.nn.functional as F
from torch.distributions import Uniform


class VAEHash(nn.Module):
    """
        The learned variational autoencoder hashing function.

        Forward propagate through encoder consisting of Conv2d and Linear layers to a vector of size (1024,), and use
            a Linear layer to downsample to the specified size of the hash code
        Downsampled vector will be fed to a sigmoid function. Reparameterization will be applied to the vector
        The hash code is given by taking integer rounding after the noise has been injected. The overall value is then
            fed to the decoder during training to produce reconstructed images, through Linear and TransposedConv2d layers

        - Input size: (4 * 52 * 52)
            - Input should be normalized to values in [0., 1.]
        - Output hash code size: (len_hashcode,)

        - Encoder Architecture:
            - Conv1: kernel size = 5, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (64, 24, 24)
            - Conv2: kernel size = 5, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (64, 10, 10)
            - Conv3: kernel size = 3, stride = 2, dilation = 1, num_kernel = 64 -> Output size = (64, 4, 4)
            - Reshape: Input size = (64, 4, 4) -> Output size = (1024,)
            - Linear1 mean value stream: Input size = (1024,) -> Output size = (k,)
            - Linear1 log variance value stream: Input size = (1024, ) -> Output size = (k,)
            - Sigmoid activation

        - Hashing:
            - Integer rounding to produce hash code
            - Apply uniform noise to activation values: U(-a, a)

        - Decoder Architecture:
            - Linear 2: Input size = (k,) -> Output size = (1024,)
            - Reshape: Input size = (1024,) -> Output size = (64, 4, 4)
            - ConvTransposed1: kernel size = 4, stride = 2, num_kernel = 64 -> Output size = (64, 10, 10)
            - ConvTransposed2: kernel size = 6, stride = 2, num_kernel = 64 -> Output size = (64, 24, 24)
            - ConvTransposed3: kernel size = 6, stride = 2, num_kernel = 64 -> Output size = (64, 52, 52)
            - Conv4: kernel size = 1, stride = 1, dilation = 1, num_kernel = 4 -> Output size = (4, 52, 52)
    """

    def __init__(self, len_hashcode, num_channels=4, noise_scale=0.3, saturating_weight=10., device='cpu'):
        """

        :param len_hashcode: The length of the hashcode
        :param noise_scale: The noise scale a of a uniform noise U(-a, a) to be injected to the sigmoid activations
        :param saturating_weight: Controlling the weight of the saturating term in the loss function
        """
        super(VAEHash, self).__init__()

        self.num_channels = num_channels
        self.k = len_hashcode
        self.a = noise_scale
        self.lam = saturating_weight
        self.device = device

        self.Conv1 = Conv2d(in_channels=self.num_channels, out_channels=64, kernel_size=5, stride=2)
        self.Conv2 = Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.Conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)

        self.Cbn1 = BatchNorm2d(64)
        self.Cbn2 = BatchNorm2d(64)
        self.Cbn3 = BatchNorm2d(64)

        self.FC1_mean = Linear(in_features=1024, out_features=self.k)
        self.FC1_logvar = Linear(in_features=1024, out_features=self.k)

        self.FC2 = Linear(in_features=self.k, out_features=1024)

        self.ConvTrans1 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.ConvTrans2 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=2)
        self.ConvTrans3 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=2)

        self.Ctbn1 = BatchNorm2d(64)
        self.Ctbn2 = BatchNorm2d(64)
        self.Ctbn3 = BatchNorm2d(64)

        self.Conv4 = Conv2d(in_channels=64, out_channels=self.num_channels, kernel_size=1, stride=1)

    def encoder(self, x):
        # Check input shape
        assert x.shape[-3:] == (self.num_channels, 52, 52), "Input must have dimension (None, 4, 52, 52)."

        # Propagate through convolutional layers
        x = self.Conv1(x)
        x = self.Cbn1(x)
        x = F.elu(x)

        x = self.Conv2(x)
        x = self.Cbn2(x)
        x = F.elu(x)

        x = self.Conv3(x)
        x = self.Cbn3(x)
        x = F.elu(x)

        # Reshape
        x = x.view(-1, 1024)

        # Mean and log variance vector
        mean = self.FC1_mean(x)
        logvar = self.FC1_logvar(x)

        # Propagate mean vector through sigmoid activation function
        sig_act = torch.sigmoid(mean)

        return mean, logvar, sig_act

    def decoder(self, x):
        # Upsample to vector of length 1024 through FC layer
        x = self.FC2(x)

        # Reshape
        x = x.view(-1, 64, 4, 4)

        # Propagate through transposed convolutional layers
        x = self.ConvTrans1(x)
        x = self.Ctbn1(x)
        x = F.elu(x)

        x = self.ConvTrans2(x)
        x = self.Ctbn2(x)
        x = F.elu(x)

        x = self.ConvTrans3(x)
        x = self.Ctbn3(x)
        x = F.elu(x)

        x = torch.sigmoid(self.Conv4(x))       # Use sigmoid to normalize output value into range [0., 1.]

        return x

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, base_ten=True):
        # Assume the first dimension of the input is the batch dimension
        assert len(x.shape) == 4, "Input must have 4 dimensions. The first be the batch dimension."

        # Propogate through encoder
        mean, logvar, latent = self.encoder(x)

        # Calculate hash code from latent vector (sigmoid activation)
        # Convert to numpy arrays and obtain code by integer rounding
        binary = (latent > 0.5).type(torch.float32)

        # Convert to base 10 if base_ten is True
        if base_ten:
            code = torch.zeros((latent.shape[0]), dtype=torch.int32)
            for i in range(self.k):
                code += binary[:, -(i + 1)] * (2 ** i)
        else:
            code = binary

        # Reparameterization
        z = self.reparameterize(mean, logvar)         # TODO: Changed!

        # propagate through decoder
        recon = self.decoder(z)

        # return both the reconstructed image and the sigmoid activations as the latent vector
        return mean, logvar, latent, code, recon

    def optimize_model(self, batch, optimizer):
        """
            Optimize the model for one step using the provided batch and optimizer
        :param batch:
        :param optimizer:
        :return:
        """
        # Forward propogate to obtain reconstructed
        mean, logvar, latent, code, recon = self.forward(batch, base_ten=False)

        # Calculate the first loss term: BCE for reconstructing loss
        recon_loss = F.binary_cross_entropy(recon, batch, reduction='mean')

        # Calculate the second loss term: KL divergence
        kld_loss = - 0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - torch.exp(logvar))

        # Calculate the second loss term: MSE for saturating loss
        satur_loss = F.binary_cross_entropy(latent, code, reduction='mean')

        # Final loss is weighted
        loss = recon_loss + kld_loss + self.lam * satur_loss

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Return the loss value
        return loss.item()
