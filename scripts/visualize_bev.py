import cv2
import numpy as np
import torch

COLORS = {
    # static
    'lane':                 (110, 110, 110),
    'road_segment':         (90, 90, 90),

    # dividers
    'road_divider':         (255, 200, 0),
    'lane_divider':         (130, 130, 130),

    # dynamic
    'car':                  (255, 158, 0),
    'truck':                (255, 99, 71),
    'bus':                  (255, 127, 80),
    'trailer':              (255, 140, 0),
    'construction':         (233, 150, 70),
    'pedestrian':           (0, 0, 230),
    'motorcycle':           (255, 61, 99),
    'bicycle':              (220, 20, 60),

    'nothing':              (200, 200, 200)
}
SEMANTICS = []

def get_colors(semantics):
    return np.array([COLORS[s] for s in semantics], dtype=np.uint8)

def visualize_bev(bev):
        """
        (c, h, w) torch [0, 1] float

        returns (h, w, 3) np.uint8
        """
        # if isinstance(bev, torch.Tensor):
        #     bev = bev.cpu().numpy().transpose(1, 2, 0)

        h, w, c = bev.shape

        assert c == len(SEMANTICS)

        # Prioritize higher class labels
        eps = (1e-5 * np.arange(c))[None, None]
        idx = (bev + eps).argmax(axis=-1)
        val = np.take_along_axis(bev, idx[..., None], -1)

        # Spots with no labels are light grey
        empty = np.uint8(COLORS['nothing'])[None, None]

        result = (val * get_colors(SEMANTICS)[idx]) + ((1 - val) * empty)
        result = np.uint8(result)

        return result