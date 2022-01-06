from lossFunctions import *
import torch.nn as nn

def construct_loss(method, alpha, lamda, pos_neg_sample_nums=None, *kwargs):
    print(f"{method} with alpha {alpha} and lamda {lamda}]")
    
    if method == "anchor":
        criterion = MultilabelAnchorLoss(alpha, lamda)
    elif method == "ldam":
        criterion = LDAMLoss(pos_neg_sample_nums=pos_neg_sample_nums)
    elif method == "effective":
        criterion = EffectiveNumLoss(alpha, pos_neg_sample_nums=pos_neg_sample_nums)
    elif method == "multitask":
        criterion = MultitaskLearningLoss(pos_neg_sample_nums=pos_neg_sample_nums)
    else:
        criterion = nn.BCEWithLogitsLoss() 
        
    return criterion