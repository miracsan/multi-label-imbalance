import torch
import torch.nn.functional as F


def probs_to_one_hot_preds(probs):
    '''
    :param probs: softmax prob tensor of form (Nxd1...dnxC)
    :return: one-hot prediction tensor of form (Nxd1...dnxC)
    '''
    num_classes = probs.shape[-1]
    preds = torch.argmax(probs, dim=-1) #TODO: Functionalise the operation from probs to onehot preds
    one_hot_preds = F.one_hot(preds, num_classes=num_classes)
    return one_hot_preds
    
    
def one_hot_to_dense(one_hot):
    '''
    We don't need this function. A simple torch.argmax is actually enough
    :param one_hot: one-hot tensor of form (Nxd1...dnxC)
    :return: prediction array of form (Nxd1...dn) where the values are in range(C)
    '''
    num_classes = one_hot.shape[-1]
    broadcast_array = torch.arange(num_classes, device=one_hot.device)
    return (one_hot * broadcast_array).sum(dim=-1)
    

