import torch
import copy
from AverageMeter import AverageMeter


def train_model(model, optimizer, loss_fn, train_loader, val_loader, hparams, wandb):

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()

    for epoch in range(hparams['num_epochs']):
        # train
        model.train()
        train_loss.reset()
        train_accuracy.reset()
        for data, target in train_loader:
            data, target = data.float().to(
                hparams['device']), target.float().to(hparams['device'])
            target = target.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
            train_accuracy.update(acc, n=len(target))

        train_losses.append(train_loss.avg)
        train_accuracies.append(train_accuracy.avg)

        # validation
        model.eval()
        val_loss.reset()
        val_accuracy.reset()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.float().to(
                    hparams['device']), target.float().to(hparams['device'])
                target = target.unsqueeze(-1)
                output = model(data)
                loss = loss_fn(output, target)
                val_loss.update(loss.item(), n=len(target))
                pred = output.round()  # get the prediction
                acc = pred.eq(target.view_as(pred)).sum().item()/len(target)
                val_accuracy.update(acc, n=len(target))

        val_losses.append(val_loss.avg)
        val_accuracies.append(val_accuracy.avg)
        wandb.log({"Epoch Validation Loss": val_loss.avg,
                  "Epoch Validation Accuracy": val_accuracy.avg, 
                  "Epoch Train Loss": train_loss.avg,
                  "Epoch Train Accuracy": train_accuracy.avg})

    return train_accuracies, train_losses, val_accuracies, val_losses


def split_dataset(dataset, train_portion):
    train_size = int(train_portion * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size])
    return train_set, val_set

def collate_fn(data):
    print(data)
    img, label = data
    zipped = zip(img, label)
    return list(zipped)

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
