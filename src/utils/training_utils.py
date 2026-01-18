import torch
import torch.nn as nn

def initialize_weights(module):
    """Initialize network weights"""
    classname = module.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)

class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.should_stop

class LRSchedulerWrapper:
    """Learning rate scheduler wrapper"""
    def __init__(self, optimizer, halve_epoch=20):
        self.optimizer = optimizer
        self.halve_epoch = halve_epoch
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.current_epoch + 1
        
        self.current_epoch = epoch
        
        if epoch > 0 and epoch % self.halve_epoch == 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f'Learning rate halved at epoch {epoch}')
