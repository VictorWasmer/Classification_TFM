import torch
from torch import nn
from torchvision import models

model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Sequential(
    nn.Linear(in_features=1280, out_features = 1, bias=True),
    nn.Sigmoid())    

device = torch.device('cuda')

model.load_state_dict(torch.load('models\final_model_20220217-010634.pt', map_location=device))
model.to(device)

dummy_input = torch.randn(64, 3,224,224, dtype=torch.float).to(device)
repetitions=100
total_time = 0
with torch.no_grad():
  for rep in range(repetitions):
     starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
     starter.record()
     _ = model(dummy_input)
     ender.record()
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)/1000
     total_time += curr_time
Throughput = (repetitions*64)/total_time
print(f'Final Throughput: {Throughput}', flush=True)
