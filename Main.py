import os
import argparse
import torch
from torch import optim
#from torch._C import int8
from Custom_Dataset import CustomImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import time
from definitions import hparams
import paths
from aux_functions import split_dataset, train_model, collate_fn
import wandb

parser = argparse.ArgumentParser(description='Classification_TFM Training')
parser.add_argument('--epochs', default=hparams['num_epochs'], type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=hparams['batch_size'], type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=hparams['learning_rate'], type=float,
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

    args = parser.parse_args()

    #track_params = {key_track: hparams[key_track] for key_track in params_to_track}

    track_params = {'n_epochs': args.epochs, 
                    'start_epoch': args.start_epoch, 
                    'batch_size': args.batch_size,
                    'learning_rate': args.lr,
                    'momentum': args.momentum,
                    'weight_decay': args.weight_decay}

    wandb.init(project="Classification_TFM", entity="viiiictorr", config=track_params, resume=True)

    main_worker(args, wandb)


def main_worker(args, wandb):
    global best_acc1
    
    # Instantiate the model and modify the last layer to our specific case
    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Sequential(
        nn.Linear(in_features=1024, out_features=args.model_outputs, bias=True),
        nn.Sigmoid())

    # Send the model to GPU
    model.to(hparams['device'])

    # Set all req_grad at False
    for param in model.parameters():
        param.requires_grad = False

    # We only want to train the classifier part
    model.classifier.requires_grad_()

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
            print("=> loading checkpoint '{}'".format(args.resume))

            checkpoint = torch.load(args.resume, map_location=hparams['device'])
            
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    transformations = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
    # Instantiation of the dataset
    train_set = CustomImageDataset(annotations_file=paths.train_annotation_path,
                                    img_dir=paths.train_img_path,
                                    transform=transformations)
    validation_set = CustomImageDataset(annotations_file=paths.validation_annotation_path,
                                    img_dir=paths.validation_img_path,
                                    transform=transformations)
    # Split train/val sets
    #train_set, val_set = split_dataset(my_dataset, 0.8) #Split already done in folders

    # Dataloader creation
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, collate_fn = collate_fn)

    val_loader = DataLoader(
        validation_set, batch_size=args.batch_size, shuffle=True, collate_fn = collate_fn)

    #Add the loss function and the optimizer to de wandb config file
    wandb.config.update({"Loss function": criterion, "Optimizer": optimizer})

    train_accuracies, train_losses, val_accuracies, val_losses = train_model(
        model, optimizer, criterion, train_loader, val_loader, hparams, wandb, args, best_acc1)

    model_date = time.strftime("%Y%m%d-%H%M%S")
    filename = "final_model_%s.pt" % model_date
    torch.save(model.state_dict(), os.path.join("models", filename))

if __name__ == '__main__':
    main()