import torch.nn as nn

from models.bc_linear import *

class MLP(nn.Module):
    def __init__(self,num_k):
        super(MLP, self).__init__()
        self.k = num_k
        self.relu = nn.ReLU()
        self.fc1 = Linear_BC(28*28,2048,self.k)
        self.fc2 = Linear_BC(2048,1024,self.k)
        self.fc3 = Linear_BC(1024,160,self.k)
        self.fc4 = Linear_BC(160,10,10)
    def forward(self,x):
        x = x.view((-1,28*28))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)