# Experiments Directory

This directory stores training outputs, checkpoints, and evaluation results.

## Structure

Each experiment creates a subdirectory with the following structure:

```
experiments/
└── [dataset]/
    └── [model_name]-[channels]-[experiment_name]/
        ├── config.json           # Experiment configuration
        ├── pretrained_weights.pt # Model checkpoint
        ├── training_log.txt     # Training history
        └── test_results.txt     # Evaluation metrics
```

