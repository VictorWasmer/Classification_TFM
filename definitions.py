import torch

hparams = {
    "learning_rate": 0.01,
    "momentum": 0.9,
    "num_epochs": 2,
    "batch_size": 16,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "num_classes": 2
}

params_to_track = ['learning_rate', 'num_epochs', 'batch_size']
