import torch

hparams = {
    "learning_rate": 0.01,
    "momentum": 0.9,
    "num_epochs": 20,
    "batch_size": 16,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "num_classes": 1 #Actually two classes but 1 unit enough to binary classification
}

params_to_track = ['learning_rate', 'num_epochs', 'batch_size']
