# Spatio-temporal fusion and contrastive learning for urban flow prediction

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.knosys.2023.111104-blue)](https://doi.org/10.1016/j.knosys.2023.111104)
[![Project page](https://img.shields.io/badge/project-page-blue)](https://codezx6.github.io/papers/st-csl.html)

Official implementation of **ST-CSL** — *Spatio-temporal fusion and contrastive learning for urban flow prediction* (Knowledge-Based Systems 2023). ST-CSL fuses temporal-view and spatial-view contrastive learning so an urban flow predictor captures global periodicity and the hidden flow relationships between functionally similar regions.

📄 Paper: https://doi.org/10.1016/j.knosys.2023.111104 · 🌐 Project page with abstract, FAQ and BibTeX: https://codezx6.github.io/papers/st-csl.html · 👤 Author: [Xu Zhang](https://codezx6.github.io)


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

