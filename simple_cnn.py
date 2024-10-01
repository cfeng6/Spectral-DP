import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer = nn.Linear(32*32,10)
    def forward(self,x):
        x = x.view(-1,32*32)
        x = self.layer(x)
        return x