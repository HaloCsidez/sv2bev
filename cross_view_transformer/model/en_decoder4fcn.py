import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torchvision.models.resnet import Bottleneck
from typing import List


class En_Decoder(nn.Module):
    def __init__(
            self,
            backbone
    ):
        super().__init__()
        self.backbone = backbone

    def forward(self, batch):
        b, n, _, _, _ = batch['image'].shape

        image = batch['image'].flatten(0, 1)            # b n c h w

        features =  self.backbone(image)

        return features
