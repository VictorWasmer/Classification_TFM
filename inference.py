import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torchvision import models, transforms
from Custom_Dataset import CustomImageDataset
from aux_functions import split_dataset
from definitions import hparams
import paths

transformations = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
# Instantiation of the dataset
my_dataset = CustomImageDataset(annotations_file=paths.annotation_path,
                                img_dir=paths.img_path,
                                transform=transformations)
# Split train/val sets
train_set, val_set = split_dataset(my_dataset, 0.8)


# Dataloader creation
train_loader = DataLoader(
    train_set, batch_size=hparams['batch_size'], shuffle=True)
val_loader = DataLoader(
    val_set, batch_size=hparams['batch_size'], shuffle=True)

model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Sequential(
    nn.Linear(in_features=1280, out_features = 1, bias=True),
    nn.Sigmoid())    

model.load_state_dict(torch.load('models\final_model_20220217-010634.pt', map_location=hparams['device']))

model.eval()

with torch.no_grad():
    for data, target in val_loader:
        data, target = data.float().to(
            hparams['device']), target.float().to(hparams['device'])
        target = target.unsqueeze(-1)
        output = model(data)
        print('Predictions: {}' .format(round(output)))
        print('Labels: {}' .format(target))
