import torch

hparams = {
    "learning_rate": 0.01,
    "epochs": 2,
    "batch_size": 16,
    "device": 'cuda' if torch.cuda.is_available() else 'cpu'
}

annotation_path = 'data\\annotations\\annotations_xwalk_swalk.csv'
img_path = 'data\\frames'

params_to_track = ['learning_rate', 'epochs', 'batch_size']