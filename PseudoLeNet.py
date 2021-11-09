from torch import nn

from ConvBlock import ConvBlock


class PseudoLeNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.pad = nn.ConstantPad2d(padding=2, value=0)
        self.conv1 = ConvBlock(3, 6, 5)
        self.conv2 = ConvBlock(6, 16, 5)
        self.mlp = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.conv2(x)
        bsz, nch, height, width = x.shape
        x = x.view(bsz, -1)
        y = self.mlp(x)
        return y
