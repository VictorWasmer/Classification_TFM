import torch
import torch.nn.utils.prune as prune
import torchvision.models as models
import os
import random
import torch
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from aux_functions import  print_size_of_model
import paths
import numpy as np


#! RANDOM SEEDS SETUP
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#! LOAD PRE-TRAINED MODEL (NON-QUANTIZED)
model = models.quantization.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Sequential(
    nn.Linear(in_features=1280, out_features = 1, bias=True),
    nn.Sigmoid())    
device = torch.device('cpu')
model.load_state_dict(torch.load("/mnt/gpid07/imatge/victor.wasmer/TFM/classificationRepo/Classification_TFM/models/quant_model_20220327-161555.pt", map_location=device))
model.to(device)

#! EVENT SETUP
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
warmupIterations = 10

#! DATALOADERS SETUP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
train_set = CustomImageDataset(annotations_file=paths.train_annotation_path,
                              img_dir=paths.train_img_path,
                              transform=transforms.Compose([
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              normalize,
                              ]))
validation_set = CustomImageDataset(annotations_file=paths.validation_annotation_path,
                                img_dir=paths.validation_img_path,
                                transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                normalize,
                                ]))

print("Creating train Dataloader", flush = True)
train_loader = DataLoader(
    train_set, batch_size=64, shuffle=True)
print("Creating validation Dataloader", flush = True)
val_loader = DataLoader(
    validation_set, batch_size=8, shuffle=True)
print("Creating performance Dataloader", flush = True)
performance_dataloader = DataLoader(
   train_set, batch_size=1, shuffle=True)

#! GPU-WARM-UP
print("CPU Warm-up", flush = True)
warmup = 0
for data, target in performance_dataloader:
   data, target = data.float().to(device), target.float().to(device)
   target = target.unsqueeze(-1)
   _ = model(data)
   warmup = warmup + 1
   if warmup == warmupIterations:
      break

#! MEASURE PERFORMANCE OF THE NON PRUNED MODEL
print("Evaluating performance...", flush = True)
with torch.no_grad():
   rep = 0
   for data, target in performance_dataloader:
      data, target = data.float().to(device), target.float().to(device)
      target = target.unsqueeze(-1)
      starter.record()
      output = model(data)
      ender.record()
      # WAIT FOR CPU SYNC
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time
      rep = rep+1
      if rep == repetitions:
         break
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print("Size of model before Pruning")
print_size_of_model(model)
print(f'NON-PRUNED MODEL: Inference time averaged with {repetitions} predictions = {mean_syn}ms with a {std_syn} deviation.') 

parameters_to_prune = (
    (model.features[0][0], 'weight'),
    
    (model.features[1].block[0][0], 'weight'),
    (model.features[1].block[1][0], 'weight'),

    (model.features[2].block[0][0], 'weight'),
    (model.features[2].block[1][0], 'weight'),
    (model.features[2].block[2][0], 'weight'),

    (model.features[3].block[0][0], 'weight'),
    (model.features[3].block[1][0], 'weight'),
    (model.features[3].block[2][0], 'weight'),

    (model.features[4].block[0][0], 'weight'),
    (model.features[4].block[1][0], 'weight'),
    (model.features[4].block[3][0], 'weight'),

    (model.features[5].block[0][0], 'weight'),
    (model.features[5].block[1][0], 'weight'),
    (model.features[5].block[3][0], 'weight'),

    (model.features[6].block[0][0], 'weight'),
    (model.features[6].block[1][0], 'weight'),
    (model.features[6].block[3][0], 'weight'),

    (model.features[7].block[0][0], 'weight'),
    (model.features[7].block[1][0], 'weight'),
    (model.features[7].block[2][0], 'weight'),

    (model.features[8].block[0][0], 'weight'),
    (model.features[8].block[1][0], 'weight'),
    (model.features[8].block[2][0], 'weight'),

    (model.features[9].block[0][0], 'weight'),
    (model.features[9].block[1][0], 'weight'),
    (model.features[9].block[2][0], 'weight'),

    (model.features[10].block[0][0], 'weight'),
    (model.features[10].block[1][0], 'weight'),
    (model.features[10].block[2][0], 'weight'),

    (model.features[11].block[0][0], 'weight'),
    (model.features[11].block[1][0], 'weight'),
    (model.features[11].block[3][0], 'weight'),

    (model.features[12].block[0][0], 'weight'),
    (model.features[12].block[1][0], 'weight'),
    (model.features[12].block[3][0], 'weight'),

    (model.features[13].block[0][0], 'weight'),
    (model.features[13].block[1][0], 'weight'),
    (model.features[13].block[3][0], 'weight'),

    (model.features[14].block[0][0], 'weight'),
    (model.features[14].block[1][0], 'weight'),
    (model.features[14].block[3][0], 'weight'),

    (model.features[15].block[0][0], 'weight'),
    (model.features[15].block[1][0], 'weight'),
    (model.features[15].block[3][0], 'weight'),

    (model.features[16][0], 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.21,
)

#model = prune.remove(model, name = 'weight')

for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.features[0][0].weight == 0)

            + torch.sum(model.features[1].block[0][0].weight == 0)
            + torch.sum(model.features[1].block[1][0].weight == 0)

            + torch.sum(model.features[2].block[0][0].weight == 0)
            + torch.sum(model.features[2].block[1][0].weight == 0)
            + torch.sum(model.features[2].block[2][0].weight == 0)

            + torch.sum(model.features[3].block[0][0].weight == 0)
            + torch.sum(model.features[3].block[1][0].weight == 0)
            + torch.sum(model.features[3].block[2][0].weight == 0)

            + torch.sum(model.features[4].block[0][0].weight == 0)
            + torch.sum(model.features[4].block[1][0].weight == 0)
            + torch.sum(model.features[4].block[3][0].weight == 0)

            + torch.sum(model.features[5].block[0][0].weight == 0)
            + torch.sum(model.features[5].block[1][0].weight == 0)
            + torch.sum(model.features[5].block[3][0].weight == 0)

            + torch.sum(model.features[6].block[0][0].weight == 0)
            + torch.sum(model.features[6].block[1][0].weight == 0)
            + torch.sum(model.features[6].block[3][0].weight == 0)

            + torch.sum(model.features[7].block[0][0].weight == 0)
            + torch.sum(model.features[7].block[1][0].weight == 0)
            + torch.sum(model.features[7].block[2][0].weight == 0)

            + torch.sum(model.features[8].block[0][0].weight == 0)
            + torch.sum(model.features[8].block[1][0].weight == 0)
            + torch.sum(model.features[8].block[2][0].weight == 0)

            + torch.sum(model.features[9].block[0][0].weight == 0)
            + torch.sum(model.features[9].block[1][0].weight == 0)
            + torch.sum(model.features[9].block[2][0].weight == 0)

            + torch.sum(model.features[10].block[0][0].weight == 0)
            + torch.sum(model.features[10].block[1][0].weight == 0)
            + torch.sum(model.features[10].block[2][0].weight == 0)

            + torch.sum(model.features[11].block[0][0].weight == 0)
            + torch.sum(model.features[11].block[1][0].weight == 0)
            + torch.sum(model.features[11].block[3][0].weight == 0)

            + torch.sum(model.features[12].block[0][0].weight == 0)
            + torch.sum(model.features[12].block[1][0].weight == 0)
            + torch.sum(model.features[12].block[3][0].weight == 0)

            + torch.sum(model.features[13].block[0][0].weight == 0)
            + torch.sum(model.features[13].block[1][0].weight == 0)
            + torch.sum(model.features[13].block[3][0].weight == 0)

            + torch.sum(model.features[14].block[0][0].weight == 0)
            + torch.sum(model.features[14].block[1][0].weight == 0)
            + torch.sum(model.features[14].block[3][0].weight == 0)

            + torch.sum(model.features[15].block[0][0].weight == 0)
            + torch.sum(model.features[15].block[1][0].weight == 0)
            + torch.sum(model.features[15].block[3][0].weight == 0)

            + torch.sum(model.features[16][0].weight == 0)
        )
        / float(
            model.features[0][0].weight.nelement()

            + model.features[1].block[0][0].weight.nelement()
            + model.features[1].block[1][0].weight.nelement()

            + model.features[2].block[0][0].weight.nelement()
            + model.features[2].block[1][0].weight.nelement()
            + model.features[2].block[2][0].weight.nelement()

            + model.features[3].block[0][0].weight.nelement()
            + model.features[3].block[1][0].weight.nelement()
            + model.features[3].block[2][0].weight.nelement()

            + model.features[4].block[0][0].weight.nelement()
            + model.features[4].block[1][0].weight.nelement()
            + model.features[4].block[3][0].weight.nelement()

            + model.features[5].block[0][0].weight.nelement()
            + model.features[5].block[1][0].weight.nelement()
            + model.features[5].block[3][0].weight.nelement()

            + model.features[6].block[0][0].weight.nelement()
            + model.features[6].block[1][0].weight.nelement()
            + model.features[6].block[3][0].weight.nelement()

            + model.features[7].block[0][0].weight.nelement()
            + model.features[7].block[1][0].weight.nelement()
            + model.features[7].block[2][0].weight.nelement()

            + model.features[8].block[0][0].weight.nelement()
            + model.features[8].block[1][0].weight.nelement()
            + model.features[8].block[2][0].weight.nelement()

            + model.features[9].block[0][0].weight.nelement()
            + model.features[9].block[1][0].weight.nelement()
            + model.features[9].block[2][0].weight.nelement()

            + model.features[10].block[0][0].weight.nelement()
            + model.features[10].block[1][0].weight.nelement()
            + model.features[10].block[2][0].weight.nelement()

            + model.features[11].block[0][0].weight.nelement()
            + model.features[11].block[1][0].weight.nelement()
            + model.features[11].block[3][0].weight.nelement()

            + model.features[12].block[0][0].weight.nelement()
            + model.features[12].block[1][0].weight.nelement()
            + model.features[12].block[3][0].weight.nelement()

            + model.features[13].block[0][0].weight.nelement()
            + model.features[13].block[1][0].weight.nelement()
            + model.features[13].block[3][0].weight.nelement()

            + model.features[14].block[0][0].weight.nelement()
            + model.features[14].block[1][0].weight.nelement()
            + model.features[14].block[3][0].weight.nelement()

            + model.features[15].block[0][0].weight.nelement()
            + model.features[15].block[1][0].weight.nelement()
            + model.features[15].block[3][0].weight.nelement()

            + model.features[16][0].weight.nelement()
        )
    )
)

#! CPU-WARM-UP
print("CPU Warm-up", flush = True)
warmup = 0
for data, target in performance_dataloader:
   data, target = data.float().to(device), target.float().to(device)
   target = target.unsqueeze(-1)
   _ = model(data)
   warmup = warmup + 1
   if warmup == warmupIterations:
      break

#! MEASURE PERFORMANCE OF THE PRUNED MODEL
print("Evaluating performance...", flush = True)
with torch.no_grad():
   rep = 0
   for data, target in performance_dataloader:
      data, target = data.float().to(device), target.float().to(device)
      target = target.unsqueeze(-1)
      starter.record()
      output = model(data)
      ender.record()
      # WAIT FOR CPU SYNC
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time
      rep = rep+1
      if rep == repetitions:
         break
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print("Size of model after Pruning")
print_size_of_model(model)
print(f'PRUNED MODEL: Inference time averaged with {repetitions} predictions = {mean_syn}ms with a {std_syn} deviation.')