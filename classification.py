import os
import torch
from torch import optim
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import time
from definitions import hparams, params_to_track
import paths
from aux_functions import split_dataset, train_model
import wandb

track_params = {key_track: hparams[key_track] for key_track in params_to_track}
wandb.init(project="Classification_TFM",
           entity="viiiictorr", config=track_params)

transformations = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
# Instantiation of the dataset
my_dataset = CustomImageDataset(annotations_file=paths.annotation_path,
                                img_dir=paths.img_path,
                                transform=transformations)
# Split train/val sets
train_set, val_set = split_dataset(my_dataset, 0.8)

# Create a short subset to make faster tests
#short_trainset = torch.utils.data.Subset(train_set, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#short_valset = torch.utils.data.Subset(val_set, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])

# Dataloader creation
train_loader = DataLoader(
    train_set, batch_size=hparams['batch_size'], shuffle=True)
val_loader = DataLoader(
    val_set, batch_size=hparams['batch_size'], shuffle=True)

# Instantiate the model and modify the last layer to our specific case
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Sequential(
    nn.Linear(in_features=1024,
              out_features=hparams['num_classes'], bias=True),
    nn.Sigmoid())

# Send the model to GPU
model.to(hparams['device'])

# Set all req_grad at False
for param in model.parameters():
    param.requires_grad = False

# We only want to train the classifier part
model.classifier.requires_grad_()

params_to_update = []
print("Params to learn:")

for name, param in model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
        print("\t", name)

# Setup the loss function
criterion = nn.BCELoss()
# Set the optimizer
optimizer = optim.Adam(params_to_update, lr=hparams['learning_rate'])

#Add the loss function and the optimizer to de wandb config file
wandb.config.update({"Loss function": criterion, "Optimizer": optimizer})

train_accuracies, train_losses, val_accuracies, val_losses = train_model(
    model, optimizer, criterion, train_loader, val_loader, hparams, wandb)

model_date = time.strftime("%Y%m%d-%H%M%S")
filename = "model_%s.pt" % model_date
torch.save(model.state_dict(), os.path.join("models", filename))
