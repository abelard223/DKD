from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class FTLoss(nn.Module):
    def __init__(self):
        super(FTLoss, self).__init__()
        
    def forward(self, factor_s, factor_t):
        loss = F.l1_loss(self.normalize(factor_s), self.normalize(factor_t))
        
        return loss
    
    def normalize(self, factor):
        norm_factor = F.normalize(factor.view(factor.size(0),-1))
        
        return norm_factor