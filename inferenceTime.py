import torch
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import paths
import numpy as np

torch.manual_seed(0)

# Model Setup
model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Sequential(
    nn.Linear(in_features=1280, out_features = 1, bias=True),
    nn.Sigmoid())    
device = torch.device('cuda')
model.load_state_dict(torch.load("/mnt/gpid07/imatge/victor.wasmer/TFM/classificationRepo/Classification_TFM/models/final_model_20220217-010634.pt", map_location=device))
model.to(device)

# Event setup
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
warmupIterations = 10


#Input setup
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
train_set = CustomImageDataset(annotations_file=paths.train_annotation_path,
                              img_dir=paths.train_img_path,
                              transform=transforms.Compose([
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              normalize,
                              ]))
# Dataloader creation
train_loader = DataLoader(
   train_set, batch_size=1, shuffle=True) #, collate_fn = collate_fn

#GPU-WARM-UP
warmup = 0
for data, target in train_loader:
   data, target = data.float().to(device), target.float().to(device)
   target = target.unsqueeze(-1)
   _ = model(data)
   warmup = warmup + 1
   if warmup == warmupIterations:
      break

# MEASURE PERFORMANCE
with torch.no_grad():
   rep = 0
   for data, target in train_loader:
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
print(f'Inference time averaged with {repetitions} predictions = {mean_syn}ms with a {std_syn} deviation.')
