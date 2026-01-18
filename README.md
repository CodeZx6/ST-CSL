# Spatio-temporal fusion and contrastive learning for urban flow prediction

A deep learning framework for urban flow prediction leveraging contrastive self-supervised pretraining and multi-component spatio-temporal modeling.

## Overview

ST-CSL addresses the challenge of spatio-temporal flow prediction in urban environments through a novel contrastive learning framework that captures temporal closeness, period, and trend dependencies.

### Key Features

- **Multi-component Architecture**: Separate encoders for closeness, period, and trend patterns
- **Contrastive Pretraining**: Self-supervised representation learning through spatial contrastive objectives
- **Residual Architecture**: Deep residual networks for robust feature extraction

### Model Architecture

The ST-CSL framework consists of:

1. **Component Encoders**: Process closeness, period, and trend dependencies independently
2. **Contrastive Module**: Learns spatial representations through contrastive objectives
3. **Fusion Network**: Aggregates multi-component features for final prediction


## Citation

If you use this code in your research, please cite:

```bibtex
@article{ZHANG2023111104,
title = {Spatio-temporal fusion and contrastive learning for urban flow prediction},
journal = {Knowledge-Based Systems},
volume = {282},
pages = {111104},
year = {2023},
issn = {0950-7051},
author = {Xu Zhang and Yongshun Gong and Chengqi Zhang and Xiaoming Wu and Ying Guo and Wenpeng Lu and Long Zhao and Xiangjun Dong}
}
```

## License

This project is licensed under the MIT License.

