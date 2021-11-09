import torch
import time
import copy
import definitions
import wandb


def correct_predictions(predicted_batch, label_batch):
    pred = predicted_batch.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
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
        data, target = data.to(hparams['device']), target.to(hparams['device'])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        print(loss)
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
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(val_loss)
            val_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            # compute number of correct predictions in the batch
            acc += correct_predictions(output, target)
    # Average acc across all correct predictions batches now
    val_loss /= len(val_loader.dataset)
    val_acc = 100. * acc / len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, acc, len(val_loader.dataset), val_acc,
    ))
    return val_loss, val_acc


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, params_to_track=None):
    wandb.init(project="tfm-classification", entity="viiiictorr")
    wandb.config = params_to_track
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(definitions.hparams['device'])

                labels = labels.to(definitions.hparams['device'])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            wandb.log({"epoch_loss": epoch_loss,
                       "epoch_accuracy": epoch_acc})
            # Optional
            wandb.watch(model)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def split_dataset(dataset, train_portion):
    train_size = int(train_portion * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    return train_set, val_set
