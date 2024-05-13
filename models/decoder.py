import torch

from torch import nn

class Decoder(nn.Module):
    def __init__(self, num_joints, final_size):
        """
        Initalizes the Decoder Layers

        Parameters
        ----------
        num_joints: int
            the number of output joints
        final_size: (int, int)
            the final size of the output image
        """
        super(Decoder, self).__init__()

        self.final_size = final_size

        # convolution of the low level features from ResNet
        self.low_level_conv = nn.Sequential(nn.Conv2d(256, 48, 1),
                                            nn.BatchNorm2d(48),
                                            nn.ReLU(),
                                            nn.MaxPool2d(3, stride=2, padding=1))

        # convolutional layers after concatenating
        self.final_conv = nn.Sequential(nn.Conv2d(304, 256, 3),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv2d(256, 256, 3),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(256, num_joints, 1),
                                        nn.BatchNorm2d(num_joints),
                                        nn.ReLU())


    def forward(self, low_level, x):
        low_level = self.low_level_conv(low_level)
        x = nn.functional.interpolate(x, size=low_level.size()[2:], mode='bilinear')

        out = torch.cat((x, low_level), dim=1)
        out = self.final_conv(out)
        out = nn.functional.interpolate(out, size=self.final_size, mode='bilinear')

        return out