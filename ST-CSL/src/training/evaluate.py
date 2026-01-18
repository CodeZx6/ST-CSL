import os
import sys
import argparse
import warnings
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.stcsl import STCSL
from data.loader import SpatioTemporalDataLoader, load_pretrained_weights, set_random_seed
from utils.metrics import compute_rmse, compute_mae, compute_mape, model_parameter_count
from utils.config import get_default_config

warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class STCSLEvaluator:
    """Evaluation pipeline for ST-CSL model"""
    def __init__(self, config, model_path):
        self.config = config
        self.model_path = model_path
        set_random_seed(config.training['seed'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._build_model()
        self._load_weights()
        self._setup_data()
    
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
        
        if torch.cuda.is_available():
            self.model.cuda()
        
        model_parameter_count(self.model, 'ST-CSL')
    
    def _load_weights(self):
        """Load trained model weights"""
        weight_file = os.path.join(self.model_path, 'final_model.pt')
        
        if not os.path.exists(weight_file):
            print(f'Warning: Model weights not found at {weight_file}')
            print('Using randomly initialized weights')
        else:
            load_pretrained_weights(self.model, weight_file)
    
    def _setup_data(self):
        """Setup test data loader"""
        data_cfg = self.config.data
        
        self.data_loader = SpatioTemporalDataLoader(
            data_root=data_cfg['data_root'],
            dataset_name=data_cfg['dataset'],
            scaler_x=data_cfg['scaler_x'],
            scaler_y=data_cfg['scaler_y'],
            batch_size=16
        )
        
        self.test_loader = self.data_loader.get_test_loader()
    
    def evaluate(self):
        """Run evaluation on test set"""
        self.model.eval()
        
        total_mse = 0
        total_mae = 0
        total_mape = 0
        num_samples = 0
        
        print(f'\n{"="*60}')
        print(f'Evaluating ST-CSL Model')
        print(f'Dataset: {self.config.data["dataset"]}')
        print(f'{"="*60}\n')
        
        with torch.no_grad():
            for batch_idx, (x_close, x_period, x_trend, y_true) in enumerate(self.test_loader):
                batch_size = y_true.shape[0]
                
                # Forward pass
                y_pred = self.model(
                    x_close, x_period, x_trend, y_true,
                    batch_size=batch_size
                )
                
                # Convert to numpy and rescale
                scaler_y = self.config.data['scaler_y']
                preds = y_pred.cpu().numpy() * scaler_y
                targets = y_true.cpu().numpy() * scaler_y
                
                # Accumulate metrics
                mse = compute_rmse(preds, targets) ** 2
                mae = compute_mae(preds, targets)
                mape = compute_mape(preds, targets)
                
                total_mse += mse * batch_size
                total_mae += mae * batch_size
                total_mape += mape * batch_size
                num_samples += batch_size
                
                if batch_idx % 10 == 0:
                    print(f'Batch {batch_idx}/{len(self.test_loader)} processed')
        
        # Compute final metrics
        rmse = np.sqrt(total_mse / num_samples)
        mae = total_mae / num_samples
        mape = total_mape / num_samples
        
        print(f'\n{"="*60}')
        print(f'Test Results:')
        print(f'  RMSE: {rmse:.6f}')
        print(f'  MAE:  {mae:.6f}')
        print(f'  MAPE: {mape:.6f}')
        print(f'{"="*60}\n')
        
        # Save results
        result_file = os.path.join(self.model_path, 'test_results.txt')
        with open(result_file, 'w') as f:
            f.write(f'Dataset: {self.config.data["dataset"]}\n')
            f.write(f'RMSE: {rmse:.6f}\n')
            f.write(f'MAE: {mae:.6f}\n')
            f.write(f'MAPE: {mape:.6f}\n')
        
        return {'rmse': rmse, 'mae': mae, 'mape': mape}

def parse_args():
    parser = argparse.ArgumentParser(description='ST-CSL Evaluation')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str, default='BikeNYC',
                       help='Dataset name')
    parser.add_argument('--channels', type=int, default=128,
                       help='Base channels')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load config
    if args.config and os.path.exists(args.config):
        config = get_default_config().from_json(args.config)
    else:
        config = get_default_config()
    
    # Override with command line arguments
    if args.dataset:
        config.data['dataset'] = args.dataset
    if args.channels:
        config.model['base_channels'] = args.channels
    
    # Create evaluator and run evaluation
    evaluator = STCSLEvaluator(config, args.model_path)
    results = evaluator.evaluate()

if __name__ == '__main__':
    main()
