import os
import json
from typing import Dict, Any

class Config:
    """Configuration management for experiments"""
    def __init__(self, config_dict: Dict[str, Any] = None):
        if config_dict is None:
            config_dict = {}
        self._config = config_dict
    
    def __getattr__(self, name):
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._config.get(name)
    
    def __setattr__(self, name, value):
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            self._config[name] = value
    
    def update(self, **kwargs):
        self._config.update(kwargs)
    
    def to_dict(self):
        return self._config.copy()
    
    @classmethod
    def from_json(cls, json_path):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def save(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self._config, f, indent=4)

def get_default_config():
    """Get default configuration"""
    return Config({
        # Model
        'model': {
            'in_channels': 2,
            'base_channels': 128,
            'img_size': 32,
            'use_external': False
        },
        
        # Training
        'training': {
            'n_epochs': 100,
            'batch_size': 32,
            'lr': 1e-4,
            'beta1': 0.9,
            'beta2': 0.999,
            'halve_epoch': 20,
            'val_interval': 20,
            'clip_grad': 5.0,
            'seed': 2021
        },
        
        # Data
        'data': {
            'dataset': 'BikeNYC',
            'data_root': './data',
            'scaler_x': 1,
            'scaler_y': 1,
            'val_split': 0.05
        },
        
        # Contrastive Learning
        'contrastive': {
            'embed_dim': 128,
            'margin': 1e-4,
            'loss_type': 'softmax',
            'temperature': 0.1
        },
        
        # Experiment
        'experiment': {
            'name': 'default',
            'save_dir': './experiments',
            'log_interval': 10
        }
    })
