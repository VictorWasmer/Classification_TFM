import os
import random
import torch
import wandb
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from aux_functions import print_size_of_model, train_model
from torch import optim
import paths
import numpy as np
import time
from definitions import hparams
import argparse

#! ARGUMENT PARSER SETUP
parser = argparse.ArgumentParser(description='Classification_TFM Training')
parser.add_argument('--epochs', default=hparams['num_epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=hparams['batch_size'], type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=hparams['momentum'], type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--model-outputs', default=hparams['model_outputs'], type=int,
                    metavar='MODEL-OUTS', help='number of outputs of the head classifier')
print("Setting arg parser...", flush = True)
args = parser.parse_args()

#! RANDOM SEEDS SETUP
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cpu')

#! WANDB SETUP
track_params = {'n_epochs': args.epochs, 
                'start_epoch': args.start_epoch, 
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'momentum': args.momentum,
                'weight_decay': args.weight_decay,
                'quantized_model': "Y"}
print(f"Parameters used in run: {time.asctime()}", flush = True)                 
print(f"N_epochs = {args.epochs}, Batch size = {args.batch_size}, Learning rate = {args.lr}, Momentum = {args.momentum}, Weigth_decay = {args.weight_decay}", flush = True)
print("Initializing WandB", flush = True)                
wandb_id = wandb.util.generate_id()
wandb.init(project="Classification_TFM", entity="viiiictorr", config=track_params, resume=True, id  = wandb_id)

#! LOAD PRE-TRAINED MODEL (NON-QUANTIZED)
#model = models.quantization.mobilenet_v3_large(pretrained=True)
#model.classifier[3] = nn.Sequential(
#    nn.Linear(in_features=1280, out_features = 1, bias=True),
#    nn.Sigmoid())    
#model.load_state_dict(torch.load("/mnt/gpid07/imatge/victor.wasmer/TFM/classificationRepo/Classification_TFM/models/final_model_20220217-010634.pt", map_location=device)) #quant_model_20220327-161555.pt

model = models.quantization.resnet50(pretrained=True)
model.fc = nn.Sequential(   
nn.Linear(in_features=2048, out_features=1, bias=True),
nn.Sigmoid()) 
model.load_state_dict(torch.load("/mnt/gpid07/imatge/victor.wasmer/TFM/classificationRepo/Classification_TFM/models/quant_resnet50_model_20220410-215305.pt", map_location=device)) #quant_model_20220327-161555.pt

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
    train_set, batch_size=args.batch_size, shuffle=True)
print("Creating validation Dataloader", flush = True)
val_loader = DataLoader(
    validation_set, batch_size=8, shuffle=True)
print("Creating performance Dataloader", flush = True)    
performance_dataloader = DataLoader(
   train_set, batch_size=1, shuffle=True)

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
wandb.log({"Inference Time non-quant": mean_syn})
print(f'NON-QUANTIZED MODEL: Inference time averaged with {repetitions} predictions = {mean_syn}ms with a {std_syn} deviation.')

#! QUANTIZATION START
print("Starting QAT Quantization", flush = True)

#! QUANTIZED MODEL SETUP
model.fuse_model()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

#! PREPARING QUANTIZED MODEL TRAINING
params_to_update = []
for name, param in model.named_parameters():
    if param.requires_grad:
        params_to_update.append(param)
best_acc1 = 0

criterion = nn.BCELoss()
optimizer = optim.Adam(params_to_update, lr=args.lr)

#! TRAIN QUANTIZED MODEL
train_accuracies, train_losses, val_accuracies, val_losses = train_model(
    model, optimizer, criterion, train_loader, val_loader, hparams, wandb, args, best_acc1)

quantized_model = torch.quantization.convert(model.eval(), inplace=True)
quantized_model.eval()

#! SAVING QUANTIZED MODEL
model_date = time.strftime("%Y%m%d-%H%M%S")
filename = "qat_resnet50_model_%s.pt" % model_date
print(f"Saving model {filename}...", flush = True)
torch.save(quantized_model.state_dict(), os.path.join("models", filename))
print_size_of_model(quantized_model)
print("Model saved", flush = True)

#! CPU-WARM-UP
print("CPU Warm-up", flush = True)
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
      timings[rep] = curr_time
      rep = rep+1
      if rep == repetitions:
         break
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
wandb.log({"Inference Time quant": mean_syn})

print(f'QUANTIZED MODEL: Inference time averaged with {repetitions} predictions = {mean_syn}ms with a {std_syn} deviation.')
