from __future__ import print_function

import torch.nn as nn
import torch.nn.functional as F
import torch


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss

class DoubleDistillKL(nn.Module):
    """Distilling the Knowledge with class information in a Neural Network"""
    def __init__(self, T):
        super(DoubleDistillKL, self).__init__()
        self.T = T
        #self.crit = FTLoss()
        
    def forward(self, y_s, y_t):
        fs_1, ft_1 = self.normalize(y_s), self.normalize(y_t)
        #ps_1, pt_1 = F.log_softmax(fs_1, dim=1), F.softmax(ft_1, dim=1)
        fs_2, ft_2 = self.normalize(self.normalize(y_s).transpose(0,1)), self.normalize(self.normalize(y_t).transpose(0,1))
        #ps_2, pt_2 = F.log_softmax(fs_2, dim=1), F.softmax(ft_2, dim=1)
        #loss_ins = F.mse_loss(fs_1, ft_1) 
        #loss_clu = F.mse_loss(fs_2, ft_2) 
        loss_ins = 2 - 2 * torch.mul(fs_1, ft_1).sum(dim=-1).mean()
        loss_clu = 2 - 2 * torch.mul(fs_2, ft_2).sum(dim=-1).mean()
        return loss_ins + 15 * loss_clu
    
    def normalize(self, factor):
        norm_factor = F.normalize(factor.view(factor.size(0),-1))
        
        return norm_factor