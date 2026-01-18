import os
import sys
import argparse
import warnings
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.contrastive import ContrastivePretrainModule
from data.loader import SpatioTemporalDataLoader, set_random_seed
from utils.metrics import ContrastiveLoss, model_parameter_count
from utils.training_utils import initialize_weights, LRSchedulerWrapper
from utils.config import get_default_config

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class ContrastivePretrainer:
    """Contrastive pretraining pipeline"""
    def __init__(self, config):
        self.config = config
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
            f'SSL-trend-{channels}-{exp_name}'
        )
        os.makedirs(self.save_path, exist_ok=True)
        
        # Save config
        self.config.save(os.path.join(self.save_path, 'config.json'))
    
    def _build_model(self):
        """Build contrastive model"""
        self.model = ContrastivePretrainModule(
            in_channels=self.config.model['in_channels'],
            embed_dim=self.config.contrastive['embed_dim'],
            temporal_len=1
        )
        
        self.model.apply(initialize_weights)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                      max_norm=self.config.training['clip_grad'])
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        model_parameter_count(self.model, 'ContrastivePretrainer')
    
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
        
        self.criterion = ContrastiveLoss(
            margin=self.config.contrastive['margin'],
            loss_type=self.config.contrastive['loss_type']
        )
        
        self.best_val_loss = np.inf
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x_close, x_period, x_trend, y) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            
            # Select component based on config
            if self.config.experiment['component'] == 'closeness':
                x_input = x_close
            elif self.config.experiment['component'] == 'period':
                x_input = x_period
            else:  # trend
                x_input = x_trend
            
            embeddings = self.model(x_input)
            loss = self.criterion(embeddings)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * len(y)
            
            if batch_idx % self.config.experiment['log_interval'] == 0:
                print(f'[Epoch {epoch}/{self.config.training["n_epochs"]}] '
                      f'[Batch {batch_idx}/{len(self.train_loader)}] '
                      f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(self.train_loader.sampler)
        return avg_loss
    
    def validate(self):
        """Validation phase"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x_close, x_period, x_trend, y in self.val_loader:
                embeddings = self.model(x_trend)
                loss = self.criterion(embeddings)
                total_loss += loss.item() * len(y)
        
        avg_loss = total_loss / len(self.val_loader.sampler)
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.save_path, 'pretrained_weights.pt')
        torch.save(self.model.projection_head.state_dict(), checkpoint_path)
        
        result_file = os.path.join(self.save_path, 'training_log.txt')
        with open(result_file, 'a') as f:
            f.write(f'Epoch: {epoch}, Val Loss: {val_loss:.6f}\n')
        
        print(f'Checkpoint saved at epoch {epoch} with val loss {val_loss:.6f}')
    
    def train(self):
        """Main training loop"""
        print(f'\n{"="*60}')
        print(f'Starting Contrastive Pretraining')
        print(f'Dataset: {self.config.data["dataset"]}')
        print(f'{"="*60}\n')
        
        for epoch in range(1, self.config.training['n_epochs'] + 1):
            epoch_start = datetime.now()
            
            train_loss = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.training['val_interval'] == 0 and epoch > 0:
                val_loss = self.validate()
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss)
            
            # Learning rate scheduling
            self.scheduler.step(epoch)
            
            epoch_time = datetime.now() - epoch_start
            print(f'Epoch {epoch} completed in {epoch_time}')
        
        print(f'\nTraining completed! Best validation loss: {self.best_val_loss:.6f}')

def parse_args():
    parser = argparse.ArgumentParser(description='Contrastive Pretraining')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--dataset', type=str, default='BikeNYC',
                       help='Dataset name')
    parser.add_argument('--channels', type=int, default=128,
                       help='Base channels')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--component', type=str, default='trend',
                   choices=['closeness', 'period', 'trend'],
                   help='Which temporal component to pretrain')
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
    trainer = ContrastivePretrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()
