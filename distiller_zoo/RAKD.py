from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class Normalize(nn.Module):
    """normalization layer"""
    def __init__(self, power=2, dim = 1):
        super(Normalize, self).__init__()
        self.power = power
        self.dim = dim
    def forward(self, x):
        norm = x.pow(self.power).sum(self.dim, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out
    
class FTLoss(nn.Module):
    def __init__(self):
        super(FTLoss, self).__init__()
        
    def forward(self, factor_s, factor_t):
        loss = F.l1_loss(self.normalize(factor_s), self.normalize(factor_t))
        
        return loss
    
    def normalize(self, factor):
        norm_factor = F.normalize(factor.view(factor.size(0),-1))
        
        return norm_factor

class MLPEmbed(nn.Module):
    """non-linear embed by MLP"""
    def __init__(self, dim_in=1024, dim_out=128):
        super(MLPEmbed, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_out)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2 * dim_out, dim_out)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(F.relu(x), (1,1))
        x = x.view(x.shape[0], -1)
        #x = self.linear1(x)
        x = self.relu(self.linear1(x))
        #x = self.l2norm(self.linear2(x))
        return x
    
class RAKDLoss(nn.Module):
    def __init__(self, opt):
        super(RAKDLoss, self).__init__()
        self.embed = MLPEmbed(opt.s_dim, opt.feat_dim)
        
    def forward(self, f_s, out_s, out_t):
        f_s = self.embed(f_s)
        loss = FTLoss(f_s, out_t-out_s)
        #loss =  [self.batch_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        return loss