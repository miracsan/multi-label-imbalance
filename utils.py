import torch
import torch.nn.functional as F

import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.colors as colors


def get_configs_from_dataset(dataset):
    # (num_classes, batch_size, task_type)
    config_dic= {"nih": (14, 32, "classification")}

    try:
        configs = config_dic[dataset]
    except:
        raise ValueError("Unknown dataset; aborting")
    return configs


def create_checkpoint(model, optimizer, method, best_acc, epoch, alias, filename):
    state = {'architecture': model.architecture,
             'train_method': method,
             'num_classes': model.num_classes,
             'in_channels': model.in_channels,
             'model_state_dict': model.state_dict(),
             'optim_config': optimizer.optim_config,
             'optimizer_state_dict': optimizer.state_dict(),
             'best_acc': best_acc,
             'epoch' : epoch
            }
    alias_path = os.path.join('results',alias)
    os.makedirs(alias_path, exist_ok=True)
    torch.save(state, os.path.join(alias_path,filename))


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        