# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:15:28 2019

@author: Mirac
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultilabelAnchorLoss(torch.nn.Module):
    def __init__(self, alpha=0.5, lamda=0.05):
        super(MultilabelAnchorLoss, self).__init__()
        self.alpha = alpha
        self.lamda = lamda
        
    def forward(self, output, target):
        N_LABELS = target.shape[1]
        inv_target = 1 - target     
        probs = torch.sigmoid(output) #32x28
        pos_probs, neg_probs = probs[:, :N_LABELS], probs[:, N_LABELS:] #32x14
        
        thresh = target * pos_probs + inv_target * neg_probs - self.lamda
        
        pos_cost_terms_1 = - (target * torch.log(pos_probs))
        pos_cost_terms_2 = - (inv_target * torch.pow((1 + pos_probs - thresh), self.alpha) * torch.log(1 - pos_probs))
        neg_cost_terms_1 = - (inv_target * torch.log(neg_probs))
        neg_cost_terms_2 = - (target * torch.pow((1 + neg_probs - thresh), self.alpha) * torch.log(1 - neg_probs))
        
        total_loss = torch.mean(pos_cost_terms_1 + pos_cost_terms_2 + neg_cost_terms_1 + neg_cost_terms_2)
        
        return total_loss


class LDAMLoss(torch.nn.Module):
    def __init__(self, pos_neg_sample_nums, C=0.5, *args, **kwargs):
        super(LDAMLoss, self).__init__()

        m_list = 1.0 / np.sqrt(np.sqrt(pos_neg_sample_nums))
        m_list = m_list * (C / np.max(m_list, axis=0))
        self.pos_deltas = torch.cuda.FloatTensor(m_list[0,:])
        self.neg_deltas = torch.cuda.FloatTensor(m_list[1,:])

    def forward(self, output, target):
        N_LABELS = target.shape[1]
        inv_target = 1 - target 
        probs = torch.sigmoid(output)
        pos_score, neg_score = probs[:, :N_LABELS], probs[:, N_LABELS:]
        
        pos_score = output[:,:14];
        neg_score = output[:,14:];
        exp_pos_score = torch.exp(pos_score);
        exp_pos_core_minus_delta = torch.exp(pos_score - self.pos_deltas);
        exp_neg_score = torch.exp(neg_score);
        exp_neg_score_minus_delta = torch.exp(neg_score - self.neg_deltas);
        first_loss_term = -1 * target * torch.log(exp_pos_core_minus_delta/(exp_pos_core_minus_delta + exp_neg_score));
        second_loss_term = -1 * inv_target * torch.log(exp_neg_score_minus_delta/(exp_neg_score_minus_delta + exp_pos_score));
        cost_table = (first_loss_term + second_loss_term);
        
        return torch.mean(cost_table)



class EffectiveNumLoss(torch.nn.Module):
    def __init__(self, alpha, pos_neg_sample_nums, *args, **kwargs):
        super(EffectiveNumLoss, self).__init__()
        
        self.alpha = alpha
        
        pos_denominator_term = 1.0 - np.power(alpha, pos_neg_sample_nums[0]);
        neg_denominator_term = 1.0 - np.power(alpha, pos_neg_sample_nums[1]);
        pos_weights = (1.0 - alpha) / np.array(pos_denominator_term);
        neg_weights = (1.0 - alpha) / np.array(neg_denominator_term);
        
        intra_class_weights = pos_weights / (neg_weights + 1e-12)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.cuda.FloatTensor(intra_class_weights))

    def forward(self, output, target):
        return self.bce_loss(output, target)
        
        

class MultitaskLearningLoss(torch.nn.Module):
    def __init__(self, pos_neg_sample_nums):
        super(MultitaskLearningLoss, self).__init__()
        num_classes = pos_neg_sample_nums.shape[1]
        log_vars = np.zeros(num_classes)
        self.log_vars = torch.cuda.FloatTensor(log_vars)
        self.log_vars.requires_grad_(True)
        intra_class_weights = np.ones(num_classes)
        self.intra_class_weights = torch.cuda.FloatTensor(intra_class_weights)
    
    def forward(self, output, target):
        inter_class_weights = torch.exp(-2 * self.log_vars)
        sum_log_vars = torch.sum(self.log_vars)
        
        cost_terms = F.binary_cross_entropy_with_logits(output, target, reduction='none', pos_weight=self.intra_class_weights)
        weighted_cost_terms = cost_terms * inter_class_weights
        
        total_task_loss = torch.mean(weighted_cost_terms) + sum_log_vars
        
        return total_task_loss