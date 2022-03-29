import os
import random
import torch
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from aux_functions import evaluate_model, print_size_of_model
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

#! MEASURE PERFORMANCE OF THE NON QUANTIZED MODEL
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
print("Size of model before quantization")
print_size_of_model(model)
print(f'NON-QUANTIZED MODEL: Inference time averaged with {repetitions} predictions = {mean_syn}ms with a {std_syn} deviation.')

#! QUANTIZATION START
print("Starting Static Quantization", flush = True)

#! QUANTIZED MODEL SETUP
model.eval()
model.fuse_model()
model.qconfig = torch.quantization.default_qconfig
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)
print('Post Training Quantization Prepare: Inserting Observers')

# Specify quantization configuration
# Start with simple min/max range estimation and per-tensor quantization of weights
model.qconfig = torch.quantization.default_qconfig
print(model.qconfig)
torch.quantization.prepare(model, inplace=True)

# Calibrate with the training set
criterion = nn.BCELoss()
evaluate_model(model,criterion, val_loader)
print('Post Training Quantization: Calibration done')

# Convert to quantized model
torch.quantization.convert(model, inplace=True)
print('Post Training Quantization: Convert done')

print("Size of model after quantization")
print_size_of_model(model)

evaluate_model(model,criterion, val_loader)

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

#! MEASURE PERFORMANCE OF THE QUANTIZED MODEL
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
print_size_of_model(model)
print(f'QUANTIZED MODEL: Inference time averaged with {repetitions} predictions = {mean_syn}ms with a {std_syn} deviation.')