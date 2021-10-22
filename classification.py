from torch import optim
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from definitions import hparams, params_to_track
import paths
from aux_functions import train_model, set_parameter_requires_grad, split_dataset


# Parameters definition
num_epochs = hparams['epochs']
num_classes = 2

# Instantiation of the dataset
my_dataset = CustomImageDataset(annotations_file=paths.annotation_path,
                                img_dir=paths.img_path,
                                transform=transforms.Compose([
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])]))
# Split train/val sets
train_set, val_set = split_dataset(my_dataset, 0.8)

# Dataloader creation
train_loader = DataLoader(train_set, batch_size=hparams['batch_size'], shuffle=True)
val_loader = DataLoader(val_set, batch_size=hparams['batch_size'], shuffle=True)

dataloaders_dict = {'train': train_loader,
                    'val': val_loader
                    }

# Instantiate the model and modify the last layer to our specific case
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

# Send the model to GPU
model = model.to(hparams['device'])


# Set the transfer learning as feature extractor
feature_extract = True
# Set all req_grad at False
set_parameter_requires_grad(model, feature_extract)
# We only want to train the classifier part
model.classifier.requires_grad_()
print("Params to learn:")

if feature_extract:
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Setup the optimizer
optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

track_params = { key_track: hparams[key_track] for key_track in params_to_track }

# Train model
model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs,
                             params_to_track=track_params)
