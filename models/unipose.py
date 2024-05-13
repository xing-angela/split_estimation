import torch

from torch import nn
from torchvision import models

from models.wasp import Wasp
from models.decoder import Decoder

class ResNet101(nn.Module):
    def __init__(self):
        """
        Initalizes the ResNet model layers such that it will return the low-level
        features and regular output from ResNet model
        """
        super(ResNet101, self).__init__()
        # use the Pytorch pretrained resnet model
        # note: we are looking to extract the low level features, which
        #       means the output should be 256, and the regular features
        resnet101 = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        resnet101.eval()
        resnet_children = list(resnet101.children())
        self.conv1 = nn.Sequential(*resnet_children[:4])
        self.layer1 = resnet_children[4]
        self.layer2 = resnet_children[5]
        self.layer3 = resnet_children[6]
        self.layer4 = resnet_children[7]
    
    def forward(self, x):
        out = self.conv1(x)
        low_level = self.layer1(out)

        out = self.layer2(low_level)
        out = self.layer3(out)
        out = self.layer4(out)

        return low_level, out

class UniPose(nn.Module):
    def __init__(self, image_size):
        """
        Initalizes the model layers

        ResNet101 --> Wasp --> Decoder
        """
        super(UniPose, self).__init__()

        self.resnet101 = ResNet101()
        self.wasp = Wasp()
        self.decoder = Decoder(num_joints=26, final_size=image_size)

    def forward(self, x):
        low_level, out = self.resnet101(x)
        out = self.wasp(out)
        out = self.decoder(low_level, out)

        return out
