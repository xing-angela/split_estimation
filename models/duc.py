import torch
from torch import nn

class DUC(nn.Module):
    def __init__(self, c):
        """
        Initializes the DUC Layers

        input --> Conv2D --> Pixel Shuffle --> output
        c x h x w --> 4c' x h x w --> c' x 2h x 2w
        """
        super(DUC, self).__init__()

        self.conv1 = nn.Conv2d(c, c * 2, kernel_size=3, padding=1)
        self.regularization = nn.BatchNorm2d(c * 2)
        self.activation = nn.ReLU(inplace=True)
        self.pix_shuff = nn.PixelShuffle(2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.regularization(out)
        out = self.activation(out)
        out = self.pix_shuff(out)
        return out