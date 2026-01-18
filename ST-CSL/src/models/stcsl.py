import torch
import torch.nn as nn
from .encoders import ClosenessEncoder, PeriodEncoder, TrendEncoder
from .backbone import conv3x3

class STCSL(nn.Module):
    """Spatio-Temporal Contrastive Self-supervised Learning model"""
    def __init__(self, in_channels=2, out_channels=2, base_channels=64, 
                 img_size=32, use_external=False):
        super(STCSL, self).__init__()
        
        self.img_size = img_size
        self.use_external = use_external
        
        # Temporal aggregation modules
        self.closeness_aggr = nn.Sequential(
            nn.Conv2d(7 * in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        self.period_aggr = nn.Sequential(
            nn.Conv2d(7 * in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        self.trend_aggr = nn.Sequential(
            nn.Conv2d(7 * in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        # Component encoders
        self.closeness_enc = ClosenessEncoder(in_channels, base_channels)
        self.period_enc = PeriodEncoder(in_channels, base_channels)
        self.trend_enc = TrendEncoder(in_channels, base_channels)
        
        # Fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        )
        
        if self.use_external:
            self._init_external_embeddings()
    
    def _init_external_embeddings(self):
        """Initialize external factor embeddings"""
        self.embed_day = nn.Embedding(224, 1)
        self.embed_weekend = nn.Embedding(224, 1)
        self.embed_hour = nn.Embedding(224, 1)
        self.embed_holiday = nn.Embedding(224, 1)
        self.embed_wind = nn.Embedding(224, 1)
        self.embed_weather = nn.Embedding(224, 1)
        self.embed_temperature = nn.Embedding(224, 1)
    
    def _process_external(self, x, ext, batch_size):
        """Process external factors and fuse with flow data"""
        if not self.use_external:
            return x
        
        embeddings = [
            self.embed_day, self.embed_weekend, self.embed_hour,
            self.embed_holiday, self.embed_wind, self.embed_weather,
            self.embed_temperature
        ]
        
        for idx, embed_fn in enumerate(embeddings):
            start_idx = idx * 7
            end_idx = start_idx + 7
            ext_feat = embed_fn(ext[:, start_idx:end_idx].long().view(batch_size, 7, 2, 1))
            ext_feat = ext_feat.view(batch_size, 7, 2, 1).expand(-1, -1, -1, self.img_size, self.img_size)
            x = x + ext_feat
        
        return x
    
    def forward(self, x_close, x_period, x_trend, y_base, 
                ext_close=None, ext_period=None, ext_trend=None, batch_size=None):
        
        if batch_size is None:
            batch_size = x_close.shape[0]
        
        # Process external factors
        if self.use_external:
            x_close = self._process_external(x_close, ext_close, batch_size)
            x_period = self._process_external(x_period, ext_period, batch_size)
            x_trend = self._process_external(x_trend, ext_trend, batch_size)
        
        # Temporal aggregation
        temporal_len = x_close.shape[1]
        channels = x_close.shape[2]
        
        close_feat = x_close.view(batch_size, temporal_len * channels, self.img_size, self.img_size)
        period_feat = x_period.view(batch_size, temporal_len * channels, self.img_size, self.img_size)
        trend_feat = x_trend.view(batch_size, temporal_len * channels, self.img_size, self.img_size)
        
        close_feat = self.closeness_aggr(close_feat)
        period_feat = self.period_aggr(period_feat)
        trend_feat = self.trend_aggr(trend_feat)
        
        # Component encoding
        close_enc = self.closeness_enc(close_feat)
        period_enc = self.period_enc(period_feat)
        trend_enc = self.trend_enc(trend_feat)
        
        # Multi-component fusion
        fused = close_enc + period_enc + trend_enc
        output = self.fusion(fused)
        
        return output
