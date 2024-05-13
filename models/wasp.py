import torch

from torch import nn

class Wasp(nn.Module):
    def __init__(self):
        """
        Initalizes the Wasp Layers

        The paper notes that WASP relies on atrous convolutions, so the 
        convolutional layers have different rates. The different rates 
        correspond to the dilation in the Conv2d pytorch layer
        """
        super(Wasp, self).__init__()

        # atrous layer 1
        self.atrous1 = nn.Sequential(nn.Conv2d(2048, 256, kernel_size=1, dilation=6),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        # atrous layer 2
        self.atrous2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, dilation=12, padding=12),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        # atrous layer 3
        self.atrous3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, dilation=18, padding=18),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        # atrous layer 4
        self.atrous4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, dilation=24, padding=24),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU())

        # waterfall conv layer
        self.conv_layer = nn.Sequential(nn.Conv2d(256, 256, 1), 
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())

        # average pooling layer
        self.pool_layer = nn.Sequential(nn.AvgPool2d(1),
                                        nn.Conv2d(2048, 256, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())

        # final conv layer to ensure shape
        self.final_conv = nn.Sequential(nn.Conv2d(1280, 256, 1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())

        self._init_weight

    def forward(self, x):
        x1 = self.atrous1(x)
        x2 = self.atrous2(x1)
        x3 = self.atrous3(x2)
        x4 = self.atrous4(x3)

        x1 = self.conv_layer(x1)
        x2 = self.conv_layer(x2)
        x3 = self.conv_layer(x3)
        x4 = self.conv_layer(x4)

        x1 = self.conv_layer(x1)
        x2 = self.conv_layer(x2)
        x3 = self.conv_layer(x3)
        x4 = self.conv_layer(x4)

        x5 = self.pool_layer(x)

        out = torch.cat((x1, x2, x3, x4, x5), dim=1)

        out = self.final_conv(out)

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()