import torch
from torch import optim
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

from PseudoLeNet import PseudoLeNet
from definitions import hparams, params_to_track
import paths
from aux_functions import train_model, set_parameter_requires_grad, split_dataset, train_epoch, val_epoch
import wandb

track_params = {key_track: hparams[key_track] for key_track in params_to_track}
wandb.init(project="Classification_TFM", entity="viiiictorr", config=track_params)

# Instantiation of the dataset
transformations = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])

my_dataset = CustomImageDataset(annotations_file=paths.annotation_path,
                                img_dir=paths.img_path,
                                transform=transformations)
# Split train/val sets
train_set, val_set = split_dataset(my_dataset, 0.8)

#Create a short subset to make faster tests
short_trainset = torch.utils.data.Subset(train_set, [5,6,7,8,9,10,11,12,13,14]) 
short_valset = torch.utils.data.Subset(val_set, [4,3,2,1,0,15,16,17,18,19])

# Dataloader creation
train_loader = DataLoader(short_trainset, batch_size=hparams['batch_size'], shuffle=True)
val_loader = DataLoader(short_valset, batch_size=hparams['batch_size'], shuffle=True)

# Instantiate the model and modify the last layer to our specific case
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(in_features=1024, out_features=hparams['num_classes'], bias=True)
# Send the model to GPU
model.to(hparams['device'])

# Set the transfer learning as feature extractor
# feature_extract = True
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

# Setup the loss fxn
criterion = F.nll_loss
# Set the optimizer
optimizer = optim.SGD(params_to_update, lr=hparams['learning_rate'], momentum=hparams['momentum'])
#optimizer = optim.Adam(params_to_update, lr=hparams['learning_rate'])

for epoch in range(1, hparams['num_epochs'] + 1):
    tr_loss, tr_acc = train_epoch(train_loader, model, optimizer, criterion, hparams)
    wandb.log({"Epoch Train Loss": tr_loss,
               "Epoch Train Accuracy": tr_acc})
    val_loss, val_acc = val_epoch(val_loader, model, criterion, hparams)
    wandb.log({"Epoch Val Loss": tr_loss,
               "Epoch Val Accuracy": tr_acc})

