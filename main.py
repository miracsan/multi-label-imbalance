#!/usr/bin/env python

from trainCnn import train_cnn

import csv
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--method',
        help='anchor, ce',
        default='ce'
    )
    parser.add_argument(
        '--arch',
        help='densenet121, resnet50',
        default='unet3d'
    )
    parser.add_argument(
        '--alias',
        help='under what folder name the models will be saved',
        default='baseline'
    )
    parser.add_argument(
        '--dataset',
        help='nih',
        default = 'nih'
    )
    parser.add_argument(
        '--use-model',
        help='false, best, last',
        default = 'false'
    )
    parser.add_argument(
        '--pretrain',
        type = int,
        default = 0
    )
    parser.add_argument(
        '--num-epochs',
        help='number of epochs',
        type = int,
        default = 200
    )
    parser.add_argument(
        '--lr',
        help='learning rate',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--weight-decay',
        help='weight decay',
        type = float,
        default = 5e-4
    )
    parser.add_argument(
        '--optimizer',
        help='choose config 1,2...',
        type = int,
        default = 1
    )
    parser.add_argument(
        '--scheduler',
        help='choose config 1,2...',
        type = int,
        default = 1
    )
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        type = float,
        default = 0.1
    )
    parser.add_argument(
        '--lamda',
        help='hyperparameter',
        type = float,
        default = 0.1
    )
    parser.add_argument(
        '--imp',
        help='impatience limit',
        type = int,
        default = 3
    )
    parser.add_argument(
        '--find-new-best',
        type = int,
        default = 0
    )
    
    options = parser.parse_args()
    
    if options.use_model == 'best':
        use_model = os.path.join('results', options.alias, 'best_checkpoint')
    elif options.use_model == 'last':
        use_model = os.path.join('results', options.alias, 'checkpoint')
    elif options.use_model != 'false':
        use_model = os.path.join('results', options.alias, options.use_model)
    else:
        use_model = 0
        
    scheduler_config = f'scheduler_config_{options.scheduler}'
    optimizer_config = f'optimizer_config_{options.optimizer}'
        
    train_cnn(options.method,
              options.arch,
              options.alias,
              options.dataset,
              options.pretrain,
              options.num_epochs,
              options.lr,
              options.weight_decay,
              options.alpha,
              options.lamda,
              options.imp,
              optimizer_config,
              scheduler_config,
              options.find_new_best,
              use_model)
    
