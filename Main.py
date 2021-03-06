import os
import argparse
from random import random
from sentry_sdk import flush
import torch
from torch import optim
#from torch._C import int8
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import time
import random
from definitions import hparams
import paths
from aux_functions import QuantizedMobilenet, print_size_of_model, split_dataset, train_model, collate_fn
import wandb

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


best_acc1 = 0

def main():

    print("Setting arg parser...", flush = True)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #track_params = {key_track: hparams[key_track] for key_track in params_to_track}

    track_params = {'n_epochs': args.epochs, 
                    'start_epoch': args.start_epoch, 
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'momentum': args.momentum,
                    'weight_decay': args.weight_decay,
                    'quantized_model': "N"}
    print(f"Parameters used in run: {time.asctime()}", flush = True)                 
    print(f"N_epochs = {args.epochs}, Batch size = {args.batch_size}, Learning rate = {args.lr}, Momentum = {args.momentum}, Weigth_decay = {args.weight_decay}", flush = True)
    print("Initializing WandB", flush = True)                
    wandb_id = wandb.util.generate_id()
    wandb.init(project="Classification_TFM", entity="viiiictorr", config=track_params, resume=True, id  = wandb_id)

    main_worker(args, wandb)

def main_worker(args, wandb):
    global best_acc1
    print("-----START-----", flush = True)
    print(f"Start time: {time.asctime()}", flush = True)  

    print("Instantiating and setting resnet50", flush = True)
    # Instantiate the model and modify the last layer to our specific case
    #model = models.quantization.mobilenet_v3_large(pretrained=True)

    #model.classifier[3] = nn.Sequential(
    #    nn.Linear(in_features=1280, out_features=args.model_outputs, bias=True),
    #    nn.Sigmoid())

    model = models.quantization.resnet50(pretrained=True)
    model.fc = nn.Sequential(   
    nn.Linear(in_features=2048, out_features=1, bias=True),
    nn.Sigmoid())    

    #model = models.quantization.resnet18(pretrained=True)
    #model.fc = nn.Sequential(   
    #nn.Linear(in_features=512, out_features=1, bias=True),
    #nn.Sigmoid())   

    # Send the model to GPU
    print(f"Sending model to {hparams['device']}", flush = True)
    model.to(hparams['device'])
    train_classifier = False
    if train_classifier:
        print("Setting all req_grad of the model to false", flush = True)
        # Set all req_grad at False
        for param in model.parameters():
            param.requires_grad = False

        # We only want to train the classifier part
        print("Setting classifier req_grad of the model to true", flush = True)

        model.classifier.requires_grad_()

    print("Creating the params to update list", flush = True)
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)

    # Setup the loss function
    criterion = nn.BCELoss()
    # Set the optimizer
    optimizer = optim.Adam(params_to_update, lr=args.lr)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume), flush = True)

            checkpoint = torch.load(args.resume, map_location=hparams['device'])
            
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']), flush = True)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume), flush = True)


    #transformations = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                        std=[0.229, 0.224, 0.225])])
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    # Instantiation of the dataset
    print("Train and val datasets creation", flush = True)
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
    # Split train/val sets
    #train_set, val_set = split_dataset(my_dataset, 0.8) #Split already done in folders

    # Dataloader creation
    print("Creating train Dataloader", flush = True)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True) #, collate_fn = collate_fn
    print("Creating validation Dataloader", flush = True)
    val_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=True) #, collate_fn = collate_fn

    # Add the loss function and the optimizer to de wandb config file
    wandb.config.update({"Loss function": criterion, "Optimizer": optimizer})
    print("Start training...", flush = True)
    train_accuracies, train_losses, val_accuracies, val_losses = train_model(
        model, optimizer, criterion, train_loader, val_loader, hparams, wandb, args, best_acc1)
    #train_accuracies, train_losses, val_accuracies, val_losses = train_model(
        #model, optimizer, criterion, train_loader, val_loader, hparams, args, best_accuracy= best_acc1)
    print("Training end", flush = True)

    model_date = time.strftime("%Y%m%d-%H%M%S")
    filename = "quant_resnet50_model_%s.pt" % model_date
    print("Saving model...", flush = True)
    torch.save(model.state_dict(), os.path.join("models", filename))
    print_size_of_model(model)
    print("Model saved", flush = True)
    print(f"End time: {time.asctime()}", flush = True)  
    print("-----END-----", flush = True)

if __name__ == '__main__':
    print("Calling main function", flush = True)
    main()