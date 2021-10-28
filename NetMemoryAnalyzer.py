import numpy as np
from torch import nn

NUM_BITS_FLOAT32 = 32


class NetMemAnalyzer(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        tot_mbytes = 0
        spat_res = []
        for layer in self.layers:
            h = layer(x)
            mem_h_bytes = np.cumprod(h.shape)[-1] * NUM_BITS_FLOAT32 // 8
            mem_h_mb = mem_h_bytes / 1e6
            print('-' * 30)
            print('New feature map of shape: ', h.shape)
            print('Mem usage: {} MB'.format(mem_h_mb))
            x = h
            if isinstance(layer, nn.Conv2d):
                # keep track of the current spatial width for conv layers
                spat_res.append(h.shape[-1])
            tot_mbytes += mem_h_mb
        print('=' * 30)
        print('Total used memory: {:.2f} MB'.format(tot_mbytes))
        return tot_mbytes, spat_res
