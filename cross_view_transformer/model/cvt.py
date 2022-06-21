import torch.nn as nn
from PIL import Image
import numpy as np

class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        dim_last: int = 64,
        outputs: dict = {'bev': [0, 1]}
    ):
        super().__init__()

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.encoder = encoder
        self.decoder = decoder
        self.outputs = outputs

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, dim_max, 1))

    def forward(self, batch):
        x = self.encoder(batch)
        y = self.decoder(x)
        z = self.to_logits(y)

        # Image.fromarray(np.array({k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}['center'].cpu().detach().numpy()).squeeze(1)[0]).convert("RGB").save('/media/wit/HDD_0/zhouhb/cvpr2022/sv2bev/datasets/test.png')
        # 
        # return    list:{
        #               'bev':[],       [4, 1, 200, 200]
        #               'center':[]     [4, 1, 200, 200]
        #           }
        return {k: z[:, start:stop] for k, (start, stop) in self.outputs.items()}