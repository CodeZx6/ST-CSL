import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

def set_random_seed(seed=2021):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed(2021)

class SpatioTemporalDataLoader:
    """Data loader for spatio-temporal flow prediction"""
    def __init__(self, data_root, dataset_name, scaler_x=1, scaler_y=1, 
                 batch_size=32, val_split=0.05, shuffle=True, seed=2021):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.scaler_x = scaler_x
        self.scaler_y = scaler_y
        self.batch_size = batch_size
        self.val_split = val_split
        self.shuffle = shuffle
        self.seed = seed
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    def _load_data_files(self, mode='train'):
        """Load numpy data files"""
        data_path = os.path.join(self.data_root, self.dataset_name, mode)
        
        y = np.load(os.path.join(data_path, 'basis.npy'))
        x_close = np.load(os.path.join(data_path, 'time_correlation.npy'))
        x_period = np.load(os.path.join(data_path, 'period_correlation.npy'))
        x_trend = np.load(os.path.join(data_path, 'trend_correlation.npy'))
        
        # Convert to tensors with scaling
        x_close = self.tensor_type(x_close) / self.scaler_x
        x_period = self.tensor_type(x_period) / self.scaler_x
        x_trend = self.tensor_type(x_trend) / self.scaler_x
        y = self.tensor_type(y) / self.scaler_y
        
        return x_close, x_period, x_trend, y
    
    def get_train_val_loaders(self):
        """Create training and validation data loaders"""
        x_close, x_period, x_trend, y = self._load_data_files('train')
        
        dataset = TensorDataset(x_close, x_period, x_trend, y)
        dataset_size = len(dataset)
        
        indices = list(range(dataset_size))
        split_idx = int(np.floor(self.val_split * dataset_size))
        
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split_idx:], indices[:split_idx]
        
        print(f'Training samples: {len(train_indices)}')
        print(f'Validation samples: {len(val_indices)}')
        
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        
        loader_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'drop_last': False,
            'num_workers': 0
        }
        
        train_loader = DataLoader(dataset, sampler=train_sampler, **loader_params)
        val_loader = DataLoader(dataset, sampler=val_sampler, **loader_params)
        
        return train_loader, val_loader
    
    def get_test_loader(self, batch_size=None):
        """Create test data loader"""
        if batch_size is None:
            batch_size = self.batch_size
        
        x_close, x_period, x_trend, y = self._load_data_files('test')
        
        dataset = TensorDataset(x_close, x_period, x_trend, y)
        print(f'Test samples: {len(dataset)}')
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          drop_last=False, num_workers=0)
        
        return loader

def load_pretrained_weights(model, weight_path, component='encoder'):
    """Load pretrained weights for model components"""
    if not os.path.exists(weight_path):
        print(f'Warning: Weight file not found at {weight_path}')
        return False
    
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
        
        if component == 'encoder':
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)
        
        print(f'Successfully loaded weights from {weight_path}')
        return True
    except Exception as e:
        print(f'Error loading weights: {e}')
        return False
