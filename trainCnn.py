# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 21:16:02 2019

@author: Mirac
"""
#from imports import *
from lossFunctions import *
from evalCnn import *
import os
import sys
import csv
from shutil import rmtree
from torch.autograd import Variable
from utils import *
from dataloaders import *
from constructModel import *
from constructLoss import construct_loss
from constructOptimizer import construct_optimizer, load_optimizer, adjust_optimizer_for_new_hyperparams
from constructScheduler import construct_scheduler
import torch
import torch.nn as nn
import numpy as np
import itertools
import time
import copy

from torch.cuda.amp import autocast, GradScaler 

def train_cnn(method, architecture, alias, dataset, pretrain,
              num_epochs, learning_rate, weight_decay, alpha, lamda, impatience,
              optimizer_config, scheduler_config,
              find_new_best, use_model):

    #torch.manual_seed(42)
    num_classes, batch_size, task_type = get_configs_from_dataset(dataset)
    num_pool_ops = get_num_pool_ops(architecture)
        
    if use_model:
        model, starting_epoch, best_loss, prev_method = load_model(use_model)
        if method != prev_method:
            model = adjust_model_for_new_method(model, method, architecture)
        model = put_model_to_device(model) #Model needs to be put into device before constructing the optimizer
        
        #if method == prev_method:
        #    optimizer = load_optimizer(model, use_model)
        #    optimizer = adjust_optimizer_for_new_hyperparams(optimizer)
        #else:
        #    optimizer = construct_optimizer(model, optimizer_config, learning_rate, weight_decay)
            
        #if starting_epoch >= num_epochs:
        #    print("ALREADY TRAINED UP TO NUM_EPOCHS")
        #    return
        
        with open(os.path.join('results', alias, 'log_train'), 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(["LEARNING_RATE : ", learning_rate])

        
    else:
        model = construct_model(architecture, method, num_classes)
        model = put_model_to_device(model)
        
        starting_epoch = 0
        best_loss = 500
        
        rmtree(os.path.join('results',alias), ignore_errors=True)
        os.makedirs(os.path.join('results',alias))
        
        with open(os.path.join('results', alias, 'log_train'), 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(["METHOD : ", method])
            logwriter.writerow(["DATASET : ", dataset])
            logwriter.writerow(["ARCHITECTURE : ", architecture])
            logwriter.writerow(["NUM_EPOCHS : ", num_epochs])
            logwriter.writerow(["LEARNING_RATE : ", learning_rate])
            logwriter.writerow(["WEIGHT_DECAY : ", weight_decay])
            logwriter.writerow(["PRETRAIN : ", pretrain])
            logwriter.writerow(["epoch", "train_loss", "val_loss"])
    
    #Construct datalaoders, loss, optimizer and scheduler
    dataloaders = construct_dataloaders(dataset, batch_size, num_pool_ops, output_filenames=True)
    
    pos_neg_sample_nums = dataloaders['train_set'].pos_neg_sample_nums()
    
    criterion = construct_loss(method, alpha, lamda, pos_neg_sample_nums)
    optimizer = construct_optimizer(model, method, criterion, optimizer_config, learning_rate, weight_decay)
    scheduler = construct_scheduler(scheduler_config, optimizer, starting_epoch)
    
    #if "mixup" in method:
    #    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60,120,160], gamma=0.5)
    #else:
    #    lambda1 = lambda epoch: 0.5 ** (epoch // 25)
    #    #lambda1 = lambda epoch: 1
    #    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1, last_epoch=-1)
    
    #lambda1 = lambda epoch: 0.5 ** (epoch // 25)
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1, last_epoch=-1) #This line is necessary to initialize the scheduler
    #scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda1, last_epoch=starting_epoch)


    # train model
    since = time.time()

    if pretrain:
        model = pretrain_model(architecture, model, method, dataloaders, optimizer, scheduler, alias=alias)
    
    model = train_model(model, method,
                                    dataloaders, criterion, optimizer, scheduler,
                                    starting_epoch, num_epochs, impatience,
                                    alias=alias,
                                    best_loss=best_loss,
                                    find_new_best = find_new_best)
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    


def train_model(model, method, dataloaders, criterion, optimizer, scheduler, starting_epoch, num_epochs, impatience_limit,
                alias,
                best_acc=0.0, best_loss=500, find_new_best=False, autocast=False):
    impatience = 0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scaler = GradScaler()
    
    dataset_sizes = {x: len(dataloaders[x]) * dataloaders[x].batch_size for x in ['train', 'val']}
    
    print(f"Training until the end of epoch {starting_epoch + num_epochs}")
    
    # iterate over epochs
    for epoch in range(starting_epoch + 1, starting_epoch + num_epochs + 1):
        print(f'Epoch {epoch}/{starting_epoch + num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
        #for phase in ['val', 'train']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            lossMeter = AverageMeter()

            # iterate over all data in train/val dataloader:
            for batch_num, (inputs, labels, _) in enumerate(dataloaders[phase]):

                inputs, labels = inputs.to(device), labels.to(device)
                batch_size, num_classes = labels.shape[0], labels.shape[1]
                
                
                inputs, labels = map(Variable, (inputs, labels))
                
                with torch.set_grad_enabled(phase == 'train'):
                    optimizer.zero_grad()
                    
                    if autocast:
                        with autocast():
                            outputs = model(inputs)#['out']
                            loss = criterion(outputs, labels.float())
                          
                            if loss!=loss:
                                import pdb
                                pdb.set_trace()
    
                        if phase == 'train':
                            
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            '''
                            loss.backward()
                            optimizer.step()
                            '''
                            
                    else:
                        outputs = model(inputs)#['out']
                        loss = criterion(outputs, labels.float())
                        
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                        
                    if loss!=loss:
                            import pdb
                            pdb.set_trace()
                        
                    
                lossMeter.update(loss.item(), batch_size)

                sys.stdout.write("\r Progress in the epoch:     %.3f" % ((batch_num+1) * batch_size / dataset_sizes[phase] * 100)) #keep track of the progress
                sys.stdout.flush()

            epoch_loss = lossMeter.avg

            if phase == 'train':
                last_train_loss = epoch_loss


            # checkpoint model if has best val loss yet
            elif phase == 'val':   
                
                #Create checkpoint for the best model
                if np.mean(epoch_loss) < np.mean(best_loss) or find_new_best:
                #if epoch_loss < best_loss or find_new_best:
                    find_new_best = 0
                    impatience = 0
                    best_loss, best_epoch = epoch_loss, epoch
                    create_checkpoint(model, optimizer, method, best_loss, epoch, alias, 'best_checkpoint')

                else:
                    impatience += 1
                    
                #Create checkpoint in each epoch 
                #create_checkpoint(model, optimizer, method, best_loss, epoch, alias, f'checkpoint_{epoch}')
                  
                # log training and validation loss over each epoch
                with open(os.path.join('results',alias,'log_train'), 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])
            
            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(epoch, epoch_loss, dataset_sizes[phase]))
            
            if phase == 'val':
                print(f"impatience : {impatience}")
            
            
            if impatience == impatience_limit:
                return model
            
        scheduler.step()

    return model
    
