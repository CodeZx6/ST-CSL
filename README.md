# Spatio-temporal fusion and contrastive learning for urban flow prediction

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.knosys.2023.111104-blue)](https://doi.org/10.1016/j.knosys.2023.111104)
[![Paper page](https://img.shields.io/badge/paper-page-blue)](https://codezx6.github.io/papers/st-csl.html)

Official code repository for **ST-FCL** (Knowledge-Based Systems 2023): *Spatio-temporal fusion and contrastive learning for urban flow prediction*. The paper calls the method **ST-FCL**; this repository is named ST-CSL. The released code has closeness, period and trend encoders and a distance-threshold contrastive pretraining loss; the paper's temporal-view triplet pretraining and Mix Layers are not included.

ST-FCL predicts grid-level urban inflow and outflow by fusing temporal and spatial views, learned through contrastive pretraining, with an external-factor view. On the full TaxiBJ dataset it reaches RMSE 14.71, against 15.41 for the best baseline, ATFM.

📄 Paper: https://doi.org/10.1016/j.knosys.2023.111104 · 🌐 Paper page with quoted results, FAQ and BibTeX: https://codezx6.github.io/papers/st-csl.html


A deep learning framework for urban flow prediction leveraging contrastive self-supervised pretraining and multi-component spatio-temporal modeling.

## Overview

This code addresses the challenge of spatio-temporal flow prediction in urban environments through a contrastive learning framework that captures temporal closeness, period, and trend dependencies.

### Key Features

- **Multi-component Architecture**: Separate encoders for closeness, period, and trend patterns
- **Contrastive Pretraining**: Self-supervised representation learning through spatial contrastive objectives
- **Residual Architecture**: Deep residual networks for robust feature extraction

### Model Architecture

The code in this repository consists of:

1. **Component Encoders**: Process closeness, period, and trend dependencies independently
2. **Contrastive Module**: Learns spatial representations through contrastive objectives
3. **Fusion Network**: Aggregates multi-component features for final prediction


## Citation

If you use this code in your research, please cite:

```bibtex
@article{zhang2023stcsl,
  title        = {Spatio-temporal fusion and contrastive learning for urban flow prediction},
  author       = {Zhang, Xu and Gong, Yongshun and Zhang, Chengqi and Wu, Xiaoming and Guo, Ying and Lu, Wenpeng and Zhao, Long and Dong, Xiangjun},
  journal      = {Knowledge-Based Systems},
  year         = {2023},
  volume       = {282},
  pages        = {111104},
  doi          = {10.1016/j.knosys.2023.111104},
  issn         = {0950-7051},
  url          = {https://doi.org/10.1016/j.knosys.2023.111104}
}
```

## License

This project is licensed under the MIT License.

