from torch import nn
from torch.nn import functional as F
import torch.nn.init as init
from torchvision import models
from collections import OrderedDict


class MixFaceMLP(nn.Module):
    def __init__(self,dim=6) -> None:
        super(MixFaceMLP,self).__init__()
        self.model=(nn.Sequential(OrderedDict([('flatten',nn.Flatten(0)),('Linear_1',nn.Linear(dim*3,10)),('relu_1',nn.ReLU())
            ,('Linear_2',nn.Linear(10,3))])))
        self.dim=dim
    def forward(self,x):

        return self.model(F.pad(x,(0,0,0,self.dim-len(x)),"constant", 0))
    