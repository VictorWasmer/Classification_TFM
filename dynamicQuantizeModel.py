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
import time
from definitions import hparams

#! RANDOM SEEDS SETUP
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

criterion = nn.BCELoss()

#! DATALOADERS SETUP
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

validation_set = CustomImageDataset(annotations_file=paths.validation_annotation_path,
                                img_dir=paths.validation_img_path,
                                transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                normalize,
                                ]))

print("Creating validation Dataloader", flush = True)
val_loader = DataLoader(
    validation_set, batch_size=64, shuffle=True)
    
performance_dataloader = DataLoader(
   validation_set, batch_size=1, shuffle=True)

#! LOAD PRE-TRAINED MODEL (NON-QUANTIZED)
model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Sequential(
    nn.Linear(in_features=1280, out_features = 1, bias=True),
    nn.Sigmoid())    
device = torch.device('cpu')
model.load_state_dict(torch.load("/mnt/gpid07/imatge/victor.wasmer/TFM/classificationRepo/Classification_TFM/models/final_model_20220217-010634.pt", map_location=device))
model.to(device)
print(model)
#! EVALUATE NON-QUANTIZED MODEL
evaluate_model(model, criterion, val_loader)

#! EVENT SETUP
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
warmupIterations = 10

#! GPU-WARM-UP
print("GPU Warm-up", flush = True)
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
      # WAIT FOR GPU SYNC
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timings[rep] = curr_time
      rep = rep+1
      if rep == repetitions:
         break
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print_size_of_model(model)
print(f'NON-QUANTIZED MODEL: Inference time averaged with {repetitions} predictions = {mean_syn}ms with a {std_syn} deviation.')


#! QUANTIZATION 
print("Starting Dynamic Quantization", flush = True)
model.to('cpu')
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
quantized_model.to(device)
print(quantized_model)
#! EVALUATE QUANTIZED MODEL

evaluate_model(quantized_model, criterion, val_loader)

#! SAVING QUANTIZED MODEL
model_date = time.strftime("%Y%m%d-%H%M%S")
filename = "dynamic_int8_model_%s.pt" % model_date
print("Saving model...", flush = True)
torch.save(quantized_model.state_dict(), os.path.join("models", filename))
print_size_of_model(quantized_model)
print("Model saved", flush = True)

#! EVENT SETUP
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timingsQuant=np.zeros((repetitions,1))
warmupIterations = 10

#! GPU-WARM-UP
print("GPU Warm-up", flush = True)
warmup = 0
for data, target in performance_dataloader:
   data, target = data.float().to(device), target.float().to(device)
   target = target.unsqueeze(-1)
   _ = quantized_model(data)
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
      output = quantized_model(data)
      ender.record()
      # WAIT FOR GPU SYNC
      torch.cuda.synchronize()
      curr_time = starter.elapsed_time(ender)
      timingsQuant[rep] = curr_time
      rep = rep+1
      if rep == repetitions:
         break
mean_synQuant = np.sum(timingsQuant) / repetitions
std_synQuant = np.std(timingsQuant)

print(f'DYNAMIC QUANTIZED MODEL: Inference time averaged with {repetitions} predictions = {mean_synQuant}ms with a {std_synQuant} deviation.')
