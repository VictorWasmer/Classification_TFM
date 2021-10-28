from torch import nn


class ConvBlock(nn.Module):

    def __init__(self, num_inp_channels, num_out_fmaps,
                 kernel_size, pool_size=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=num_inp_channels, out_channels=num_out_fmaps, kernel_size=kernel_size)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(pool_size)

    def forward(self, x):
        return self.maxpool(self.relu(self.conv(x)))