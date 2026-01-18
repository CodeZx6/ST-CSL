import os
import sys
import argparse
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stcsl import STCSL
from data.loader import SpatioTemporalDataLoader, load_pretrained_weights, set_random_seed
from utils.metrics import compute_rmse, compute_mae, model_parameter_count
from utils.training_utils import initialize_weights, LRSchedulerWrapper
from utils.config import get_default_config, Config

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class STCSLTrainer:
    """Fine-tuning trainer for ST-CSL model"""
    def __init__(self, config, pretrain_path=None):
        self.config = config
        self.pretrain_path = pretrain_path
        set_random_seed(config.training['seed'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_paths()
        self._build_model()
        self._setup_data()
        self._setup_optimization()
    
    def _setup_paths(self):
        """Setup experiment directories"""
        exp_name = self.config.experiment['name']
        dataset = self.config.data['dataset']
        channels = self.config.model['base_channels']
        
        self.save_path = os.path.join(
            self.config.experiment['save_dir'],
            dataset,
            f'STCSL-{channels}-{exp_name}'
        )
        os.makedirs(self.save_path, exist_ok=True)
        self.config.save(os.path.join(self.save_path, 'config.json'))
    
    def _build_model(self):
        """Build ST-CSL model"""
        model_cfg = self.config.model
        self.model = STCSL(
            in_channels=model_cfg['in_channels'],
            out_channels=model_cfg['in_channels'],
            base_channels=model_cfg['base_channels'],
            img_size=model_cfg['img_size'],
            use_external=model_cfg['use_external']
        )
        
        self.model.apply(initialize_weights)
        
        # Load pretrained weights if provided
        if self.pretrain_path and os.path.exists(self.pretrain_path):
            print(f'Loading pretrained weights from {self.pretrain_path}')
            load_pretrained_weights(self.model, self.pretrain_path, 'encoder')
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        model_parameter_count(self.model, 'ST-CSL')
    
    def _setup_data(self):
        """Setup data loaders"""
        data_cfg = self.config.data
        train_cfg = self.config.training
        
        self.data_loader = SpatioTemporalDataLoader(
            data_root=data_cfg['data_root'],
            dataset_name=data_cfg['dataset'],
            scaler_x=data_cfg['scaler_x'],
            scaler_y=data_cfg['scaler_y'],
            batch_size=train_cfg['batch_size'],
            val_split=data_cfg['val_split']
        )
        
        self.train_loader, self.val_loader = self.data_loader.get_train_val_loaders()
    
    def _setup_optimization(self):
        """Setup optimizer and loss"""
        train_cfg = self.config.training
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=train_cfg['lr'],
            betas=(train_cfg['beta1'], train_cfg['beta2'])
        )
        
        self.scheduler = LRSchedulerWrapper(self.optimizer, train_cfg['halve_epoch'])
        self.criterion = nn.MSELoss()
        
        self.best_val_rmse = np.inf
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x_close, x_period, x_trend, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            batch_size = y.shape[0]
            y_pred = self.model(x_close, x_period, x_trend, y, batch_size=batch_size)
            
            loss = self.criterion(y_pred, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                          max_norm=self.config.training['clip_grad'])
            
            self.optimizer.step()
            total_loss += loss.item() * batch_size
            
            if batch_idx % self.config.experiment['log_interval'] == 0:
                print(f'[Epoch {epoch}/{self.config.training["n_epochs"]}] '
                      f'[Batch {batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader.sampler)
        return avg_loss
    
    def validate(self):
        """Validation phase"""
        self.model.eval()
        total_mse = 0
        total_mae = 0
        
        scaler_y = self.config.data['scaler_y']
        
        with torch.no_grad():
            for x_close, x_period, x_trend, y in self.val_loader:
                batch_size = y.shape[0]
                y_pred = self.model(x_close, x_period, x_trend, y, batch_size=batch_size)
                
                preds = y_pred.cpu().numpy() * scaler_y
                targets = y.cpu().numpy() * scaler_y
                
                total_mse += (compute_rmse(preds, targets) ** 2) * batch_size
                total_mae += compute_mae(preds, targets) * batch_size
        
        rmse = np.sqrt(total_mse / len(self.val_loader.sampler))
        mae = total_mae / len(self.val_loader.sampler)
        
        return rmse, mae
    
    def save_checkpoint(self, epoch, rmse, mae, is_best=False):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.save_path, 'final_model.pt')
        torch.save(self.model.state_dict(), checkpoint_path)
        
        result_file = os.path.join(self.save_path, 'training_log.txt')
        with open(result_file, 'a') as f:
            f.write(f'Epoch: {epoch}, RMSE: {rmse:.6f}, MAE: {mae:.6f}\n')
        
        if is_best:
            print(f'✓ Best model saved at epoch {epoch}: RMSE={rmse:.6f}, MAE={mae:.6f}')
    
    def train(self):
        """Main training loop"""
        print(f'\n{"="*60}')
        print(f'Starting ST-CSL Training')
        print(f'Dataset: {self.config.data["dataset"]}')
        print(f'{"="*60}\n')
        
        for epoch in range(1, self.config.training['n_epochs'] + 1):
            epoch_start = datetime.now()
            
            train_loss = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.training['val_interval'] == 0 or epoch == 1:
                rmse, mae = self.validate()
                print(f'\nValidation - RMSE: {rmse:.6f}, MAE: {mae:.6f}')
                
                is_best = rmse < self.best_val_rmse
                if is_best:
                    self.best_val_rmse = rmse
                
                self.save_checkpoint(epoch, rmse, mae, is_best)
            
            self.scheduler.step(epoch)
            
            epoch_time = datetime.now() - epoch_start
            print(f'Epoch {epoch} completed in {epoch_time}\n')
        
        print(f'\nTraining completed! Best RMSE: {self.best_val_rmse:.6f}')

def parse_args():
    parser = argparse.ArgumentParser(description='ST-CSL Fine-tuning')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--pretrain', type=str, default=None,
                       help='Path to pretrained weights')
    parser.add_argument('--dataset', type=str, default='TaxiBJ',
                       help='Dataset name')
    parser.add_argument('--channels', type=int, default=128,
                       help='Base channels')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        config = Config.from_json(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.dataset:
        config.data['dataset'] = args.dataset
    if args.channels:
        config.model['base_channels'] = args.channels
    if args.epochs:
        config.training['n_epochs'] = args.epochs
    if args.batch_size:
        config.training['batch_size'] = args.batch_size
    if args.lr:
        config.training['lr'] = args.lr
    
    # Create trainer and start training
    trainer = STCSLTrainer(config, pretrain_path=args.pretrain)
    trainer.train()

if __name__ == '__main__':
    main()
