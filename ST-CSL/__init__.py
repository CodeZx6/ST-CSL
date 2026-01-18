"""
ST-CSL: Spatio-Temporal Contrastive Self-supervised Learning

A deep learning framework for urban flow prediction.
"""

__version__ = "1.0.0"
__author__ = "ST-CSL Contributors"

from src.models.stcsl import STCSL
from src.models.contrastive import ContrastivePretrainModule
from src.data.loader import SpatioTemporalDataLoader
from src.utils.config import get_default_config

__all__ = [
    'STCSL',
    'ContrastivePretrainModule',
    'SpatioTemporalDataLoader',
    'get_default_config'
]
