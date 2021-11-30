import torch

hparams = {
    "learning_rate": 0.01,
    "momentum": 0.9,
    "num_epochs": 20,
    "batch_size": 32,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu',
    "model_outputs": 1 #Binary classification
}

params_to_track = ['learning_rate', 'num_epochs', 'batch_size']
