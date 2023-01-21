from torch import nn
from torch.nn import functional as F
import torch
import torch.nn.init as init
from torchvision import models
from collections import OrderedDict

class MixFaceMLP(nn.Module):
    def __init__(self,max_num=6) -> None:
        super(MixFaceMLP,self).__init__()
        self.model=[]
        for i in range(2,max_num+1):
            self.model.append(nn.Sequential(OrderedDict([('Linear_1',nn.Linear(i*3,10)),('relu_1',nn.ReLU())
            ,('Linear_2',nn.Linear(10,3))])))
            
    def forward(self,x):
        i=len(x)
        if i==1:
            return x
        else:
            x=self.model[i-2](x)
            return x