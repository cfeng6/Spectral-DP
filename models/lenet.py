import torch.nn as nn

from models.bc_linear import *
from models.conv2d import *


class LeNet5(nn.Module):

    def __init__(self, n_classes,num_k):
        super(LeNet5, self).__init__()
        self.k = num_k
        self.feature_extractor = nn.Sequential(            
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            Conv2d_BC_no_padding(1,6,5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            Conv2d_BC_no_padding(6,16,5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=16, out_channels=128, kernel_size=5, stride=1),
            Conv2d_BC_no_padding(16,128,5),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            Linear_BC(128,80,self.k),
            nn.Tanh(),
            Linear_BC(80,n_classes,10)
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        # print(x.size())
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
#         probs = TF.softmax(logits, dim=1)
        return logits