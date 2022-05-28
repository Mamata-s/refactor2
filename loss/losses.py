
import torch
from torch import nn 

class TVRegularizer(nn.Module):
    def __init__(self, tv_weight=1.0):
        super().__init__()
        self.register_buffer('tv_weight', tv_weight)

    def __call__(self, images):
        loss  = self.tv_weight * ( 
            torch.sum(torch.abs(images[:,:,:,:-1]-images[:,:,:,1:]))+
            torch.sum(torch.abs(images[:,:,:-1,:]-images[:,:,1:,:]))
            )
        return loss
