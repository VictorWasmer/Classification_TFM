import os
import time
import torch
import copy
from AverageMeter import AverageMeter
import shutil
import torch.nn as nn

def train_model(model, optimizer, loss_fn, train_loader, val_loader, hparams, wandb, args, best_accuracy = None):

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    best_acc1 = best_accuracy

    for epoch in range(args.start_epoch, args.epochs):
        print(f"Start epoch {epoch}", flush = True)
        adjust_learning_rate(optimizer, epoch, args)
        # train
        print("Setting model in train mode...", flush = True)
        model.train()
        train_loss.reset()
        train_accuracy.reset()
        for i, (data, target) in enumerate(train_loader):
            print(f"Start TRAIN Iteration: {i}", flush = True)
            data, target = data.float().to(hparams['device']), target.float().to(hparams['device'])
            target = target.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(data)
            #target = target.unsqueeze(1)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
            train_accuracy.update(acc, n=len(target))
            print(f"End TRAIN Iteration: {i}", flush = True)

        print(f"Epoch {epoch} avg train accuracy = {train_accuracy.avg}.", flush = True)
        print(f"Epoch {epoch} avg train loss = {train_loss.avg}.", flush = True)

        train_losses.append(train_loss.avg)
        train_accuracies.append(train_accuracy.avg)

        print("Setting model in eval mode...", flush = True)
        # validation
        model.eval()
        val_loss.reset()
        val_accuracy.reset()
        with torch.no_grad():
            print("Start VALIDATION...", flush = True)
            for data, target in val_loader:
                data, target = data.float().to(hparams['device']), target.float().to(hparams['device'])
                target = target.unsqueeze(-1)
                output = model(data)
                #target = target.unsqueeze(1)
                loss = loss_fn(output, target)
                val_loss.update(loss.item(), n=len(target))
                pred = output.round()  # get the prediction
                acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
                val_accuracy.update(acc, n=len(target))
            
            print(f"Epoch {epoch} avg val accuracy = {val_accuracy.avg}.", flush = True)
            print(f"Epoch {epoch} avg val loss = {val_loss.avg}.", flush = True)

            print("End VALIDATION...", flush = True)

        is_best = val_accuracy.val > best_acc1
        best_acc1 = max(val_accuracy.val, best_acc1)
        print("Saving Checkpoint...", flush = True)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        
        val_losses.append(val_loss.avg)
        val_accuracies.append(val_accuracy.avg)
        print(time.asctime(), flush = True)
        print("Logging metrics to WandB", flush = True)
        wandb.log({"Epoch Validation Loss": val_loss.avg,
                   "Epoch Validation Accuracy": val_accuracy.avg, 
                   "Epoch Train Loss": train_loss.avg,
                   "Epoch Train Accuracy": train_accuracy.avg}, step = epoch)
        wandb.save('checkpoint.pth.tar')
        
        print(f"End epoch {epoch}", flush = True)
    return train_accuracies, train_losses, val_accuracies, val_losses

def evaluate_model(model,loss_fn, val_loader):
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    model.eval()
    val_loss.reset()
    val_accuracy.reset()
    with torch.no_grad():
        print("Start VALIDATION...", flush = True)
        for data, target in val_loader:
            data, target = data.float().to('cpu'), target.float().to('cpu')
            target = target.unsqueeze(-1)
            output = model(data)
            #target = target.unsqueeze(1)
            loss = loss_fn(output, target)
            val_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
            val_accuracy.update(acc, n=len(target))
        
        print(f"Avg val accuracy = {val_accuracy.avg}.", flush = True)
        print(f"Avg val loss = {val_loss.avg}.", flush = True)

        print("End VALIDATION...", flush = True)
    return val_accuracy.avg, val_loss.avg

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def split_dataset(dataset, train_portion):
    train_size = int(train_portion * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    return train_set, val_set

def collate_fn(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class QuantizedMobilenet(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedMobilenet, self).__init__()
        # QuantStub converts tensors from floating point to quantized.
        # This will only be used for inputs.
        self.quant = torch.quantization.QuantStub()
        # DeQuantStub converts tensors from quantized to floating point.
        # This will only be used for outputs.
        self.dequant = torch.quantization.DeQuantStub()
        # FP32 model
        self.model_fp32 = model_fp32

    def forward(self, x):
        # manually specify where tensors will be converted from floating
        # point to quantized in the quantized model
        x = self.quant(x)
        x = self.model_fp32(x)
        # manually specify where tensors will be converted from quantized
        # to floating point in the quantized model
        x = self.dequant(x)
        return x

#!From here to the end is deprecated code


def correct_predictions(predicted_batch, label_batch):
    # get the index of the max log-probability
    pred = predicted_batch.argmax(dim=1, keepdim=True)
    acum = pred.eq(label_batch.view_as(pred)).sum().item()
    return acum


def train_epoch(train_loader, model, optimizer, criterion, hparams):
    # Activate the train=True flag inside the model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    model.train()
    avg_loss = None
    avg_weight = 0.1
    acc = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.float().to(
            hparams['device']), target.float().to(hparams['device'])
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(-1)
        loss = criterion(output, target)
        loss.backward()
        if avg_loss:
            avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
            acc += correct_predictions(output, target)

        else:
            avg_loss = loss.item()
            acc += correct_predictions(output, target)
        optimizer.step()
    train_acc = 100. * acc / len(train_loader.dataset)

    return avg_loss, train_acc


def val_epoch(val_loader, model, criterion, hparams):
    model.eval()
    device = hparams['device']
    val_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.float().to(
                hparams['device']), target.float().to(hparams['device'])
            output = model(data)
            target = target.unsqueeze(-1)
            # sum up batch loss
            val_loss += criterion(output, target, reduction='sum').item()
            # compute number of correct predictions in the batch
            acc += correct_predictions(output, target)
    # Average acc across all correct predictions batches now
    val_loss /= len(val_loader.dataset)
    val_acc = 100. * acc / len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, acc, len(val_loader.dataset), val_acc,
    ))
    return val_loss, val_acc
