"""
    Autoencoder hashing function

    Architecture:
        Input: downsampled grayscale four continuous images with size (1 * 52 * 52)
                Input size: (4 * 52 * 52)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, Linear
import torch.nn.functional as F
from torch.distributions import Uniform


class AEHash(nn.Module):
    """
        The learned autoencoder hashing function.

        Forward propagate through encoder consisting of Conv2d and Linear layers to a vector of size (1024,), and use
            a Linear layer to downsample to the specified size of the hash code
        Downsampled vector will be fed to a sigmoid function. A uniform noise of given scale will be injected to the
            activations.
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
            - Linear1: Input size = (1024,) -> Output size = (k,)
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

    def __init__(self, len_hashcode, noise_scale=0.3, saturating_weight=10., device='cpu'):
        """

        :param len_hashcode: The length of the hashcode
        :param noise_scale: The noise scale a of a uniform noise U(-a, a) to be injected to the sigmoid activations
        :param saturating_weight: Controlling the weight of the saturating term in the loss function
        """
        super(AEHash, self).__init__()

        self.k = len_hashcode
        self.a = noise_scale
        self.lam = saturating_weight
        self.device = device

        self.Conv1 = Conv2d(in_channels=4, out_channels=64, kernel_size=5, stride=2)
        self.Conv2 = Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2)
        self.Conv3 = Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2)

        self.FC1 = Linear(in_features=1024, out_features=self.k)
        self.FC2 = Linear(in_features=self.k, out_features=1024)

        self.ConvTrans1 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2)
        self.ConvTrans2 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=2)
        self.ConvTrans3 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=6, stride=2)

        self.Conv4 = Conv2d(in_channels=64, out_channels=4, kernel_size=1, stride=1)

    def encoder(self, x):
        # Check input shape
        assert x.shape[-3:] == (4, 52, 52), "Input must have dimension (None, 4, 52, 52)."

        # Propagate through convolutional layers
        x = F.elu(self.Conv1(x))
        x = F.elu(self.Conv2(x))
        x = F.elu(self.Conv3(x))

        # Reshape
        x = x.view(-1, 1024)

        # Downsample using FC layer
        x = self.FC1(x)

        # Propagate through sigmoid activation function
        x = torch.sigmoid(x)

        # Return activations
        return x

    def decoder(self, x):
        # Upsample to vector of length 1024 through FC layer
        x = self.FC2(x)

        # Reshape
        x = x.view(-1, 64, 4, 4)

        # Propagate through transposed convolutional layers
        x = F.elu(self.ConvTrans1(x))
        x = F.elu(self.ConvTrans2(x))
        x = F.elu(self.ConvTrans3(x))
        x = torch.sigmoid(self.Conv4(x))       # Use sigmoid to normalize output value into range [0., 1.]

        return x

    def hash(self, x, base_ten=True):
        """
            Calculate hash code. Return base-10 number if base_ten is True
        :param x:
        :param base_ten:
        :return:
        """
        # Assume the first dimension of the input is the batch dimension
        assert len(x.shape) == 4, "Input must have 4 dimensions. The first be the batch dimension."

        # Propagate through encoder
        x = self.encoder(x)

        # Convert to numpy arrays and obtain code by integer rounding
        binary = np.around(x.cpu().detach().numpy()).astype(np.int0)

        # Convert to base 10 if base_ten is True
        if base_ten:
            code = np.zeros((x.shape[0]), dtype=np.int32)
            for i in range(self.k):
                code += binary[:, -(i + 1)] * (2 ** i)
        else:
            code = binary
        return code, x      # Return both code and latent vector

    def forward(self, x):
        # Propogate through encoder
        latent = self.encoder(x)

        # Apply a random noise sampled from U(-a, a)
        m = Uniform(-self.a, self.a)
        noise = m.sample(latent.shape).to(self.device)
        x = latent + noise

        # Propogate through decoder
        x = self.decoder(x)

        # return both the reconstructed image and the sigmoid activations as the latent vector
        return x, latent

    def optimize_model(self, batch, optimizer):
        """
            Optimize the model for one step using the provided batch and optimizer
        :param batch:
        :param optimizer:
        :return:
        """
        # Forward propogate to obtain reconstructed
        recon, latent = self.forward(batch)

        # Calculate the first loss term: BCE for reconstructing loss
        recon_loss = F.binary_cross_entropy(recon, batch, reduction='mean')

        # Calculate the second loss term: MSE for saturating loss
        target = (latent > 0.5).type(torch.float32)
        satur_loss = F.mse_loss(latent, target, reduction='mean')

        # Final loss is weighted
        loss = recon_loss + self.lam * satur_loss

        # Update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Return the loss value
        return loss.item()
