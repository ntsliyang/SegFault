import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8), constrained_layout=True)
fig.suptitle("Training stats")

def visualize_aehash(input, recon, code, latent):

    # Concatenate input image
    #   input is a numpy array with shape (4, 52, 52)
    display_1 = np.zeros((input.shape[1], input.shape[0] * input.shape[2]), dtype=np.float32)
    for i in range(input.shape[0]):
        display_1[:, i * input.shape[1] : (i+1) * input.shape[1]] = input[i, :, :]
    ax1.cla()
    ax1.imshow(display_1)

    display_2 = np.zeros((recon.shape[1], recon.shape[0] * recon.shape[2]), dtype=np.float32)
    for i in range(recon.shape[0]):
        display_2[:, i * recon.shape[1] : (i+1) * recon.shape[1]] = recon[i, :, :]
    ax2.cla()
    ax2.imshow(display_2)

    ax3.cla()
    ax3.step([(i + 1) for i in range(len(code))], code)

    ax4.cla()
    ax4.step([(i + 1) for i in range(len(latent))], latent)

    fig.show()

