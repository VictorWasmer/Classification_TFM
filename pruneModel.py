import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

import torchvision.models as models
import torch.nn as nn


model = models.mobilenet_v3_large(pretrained=True)
model.classifier[3] = nn.Sequential(
    nn.Linear(in_features=1280, out_features=1, bias=True),
    nn.Sigmoid())

print(model.features[1].block[0])
module = model
#print(list(module.named_parameters()))

#print(list(module.named_buffers()))
parameters_to_prune = (
    (model.features[0][0], 'weight'),
    
    (model.features[1].block[0][0], 'weight'),
    (model.features[1].block[1][0], 'weight'),

    (model.features[2].block[0][0], 'weight'),
    (model.features[2].block[1][0], 'weight'),
    (model.features[2].block[2][0], 'weight'),

    (model.features[3].block[0][0], 'weight'),
    (model.features[3].block[1][0], 'weight'),
    (model.features[3].block[2][0], 'weight'),

    (model.features[4].block[0][0], 'weight'),
    (model.features[4].block[1][0], 'weight'),
    (model.features[4].block[2][0], 'weight'),
    (model.features[4].block[3][0], 'weight'),


    (model.features[5].block[0][0], 'weight'),
    (model.features[5].block[1][0], 'weight'),

    (model.features[6].block[0][0], 'weight'),
    (model.features[6].block[1][0], 'weight'),

    (model.features[7].block[0][0], 'weight'),
    (model.features[7].block[1][0], 'weight'),

    (model.features[8].block[0][0], 'weight'),
    (model.features[8].block[1][0], 'weight'),

    (model.features[9].block[0][0], 'weight'),
    (model.features[9].block[1][0], 'weight'),

    (model.features[10].block[0][0], 'weight'),
    (model.features[10].block[1][0], 'weight'),

    (model.features[11].block[0][0], 'weight'),
    (model.features[11].block[1][0], 'weight'),

    (model.features[12].block[0][0], 'weight'),
    (model.features[12].block[1][0], 'weight'),

    (model.features[13].block[0][0], 'weight'),
    (model.features[13].block[1][0], 'weight'),

    (model.features[14].block[0][0], 'weight'),
    (model.features[14].block[1][0], 'weight'),

    (model.features[15].block[0][0], 'weight'),
    (model.features[15].block[1][0], 'weight'),

    (model.features[16][0], 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)

print(
    "Sparsity in features[0] ConvBNActivation: {:.2f}%".format(
        100. * float(torch.sum(model.features[0][0].weight == 0))
        / float(model.features[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 1 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[1].block[0][0].weight == 0))
        / float(model.features[1].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 1 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[1].block[1][0].weight == 0))
        / float(model.features[1].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 2 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[2].block[0][0].weight == 0))
        / float(model.features[2].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 2 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[2].block[1][0].weight == 0))
        / float(model.features[2].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 2 ConvBNActivation 2: {:.2f}%".format(
        100. * float(torch.sum(model.features[2].block[2][0].weight == 0))
        / float(model.features[2].block[2][0].weight.nelement())
    ))
print(      
    "Sparsity in features Inverted Residual 3 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[3].block[0][0].weight == 0))
        / float(model.features[3].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 3 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[3].block[1][0].weight == 0))
        / float(model.features[3].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 3 ConvBNActivation 2: {:.2f}%".format(
        100. * float(torch.sum(model.features[2].block[2][0].weight == 0))
        / float(model.features[3].block[2][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 4 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[4].block[0][0].weight == 0))
        / float(model.features[4].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 4 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[4].block[1][0].weight == 0))
        / float(model.features[4].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 4 ConvBNActivation 2: {:.2f}%".format(
        100. * float(torch.sum(model.features[4].block[2][0].weight == 0))
        / float(model.features[4].block[2][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 4 ConvBNActivation 3: {:.2f}%".format(
        100. * float(torch.sum(model.features[4].block[3][0].weight == 0))
        / float(model.features[4].block[3][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 5 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[5].block[0][0].weight == 0))
        / float(model.features[5].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 5 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[5].block[1][0].weight == 0))
        / float(model.features[5].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 6 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[6].block[0][0].weight == 0))
        / float(model.features[6].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 6 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[6].block[1][0].weight == 0))
        / float(model.features[6].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 7 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[7].block[0][0].weight == 0))
        / float(model.features[7].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 7 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[7].block[1][0].weight == 0))
        / float(model.features[7].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 8 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[8].block[0][0].weight == 0))
        / float(model.features[8].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 8 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[8].block[1][0].weight == 0))
        / float(model.features[8].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 9 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[9].block[0][0].weight == 0))
        / float(model.features[9].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 9 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[9].block[1][0].weight == 0))
        / float(model.features[9].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 10 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[10].block[0][0].weight == 0))
        / float(model.features[10].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 10 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[10].block[1][0].weight == 0))
        / float(model.features[10].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 11 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[11].block[0][0].weight == 0))
        / float(model.features[11].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 11 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[11].block[1][0].weight == 0))
        / float(model.features[11].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 12 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[12].block[0][0].weight == 0))
        / float(model.features[12].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 12 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[12].block[1][0].weight == 0))
        / float(model.features[12].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 13 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[13].block[0][0].weight == 0))
        / float(model.features[13].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 13 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[13].block[1][0].weight == 0))
        / float(model.features[13].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 14 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[14].block[0][0].weight == 0))
        / float(model.features[14].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 14 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[14].block[1][0].weight == 0))
        / float(model.features[14].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 15 ConvBNActivation 0: {:.2f}%".format(
        100. * float(torch.sum(model.features[15].block[0][0].weight == 0))
        / float(model.features[15].block[0][0].weight.nelement())
    ))
print(
    "Sparsity in features Inverted Residual 15 ConvBNActivation 1: {:.2f}%".format(
        100. * float(torch.sum(model.features[15].block[1][0].weight == 0))
        / float(model.features[15].block[1][0].weight.nelement())
    ))
print(
    "Sparsity in features[16] ConvBNActivation: {:.2f}%".format(
        100. * float(torch.sum(model.features[16][0].weight == 0))
        / float(model.features[16][0].weight.nelement())
    ))

print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.features[0][0].weight == 0)

            + torch.sum(model.features[1].block[0][0].weight == 0)
            + torch.sum(model.features[1].block[1][0].weight == 0)

            + torch.sum(model.features[2].block[0][0].weight == 0)
            + torch.sum(model.features[2].block[1][0].weight == 0)
            + torch.sum(model.features[2].block[2][0].weight == 0)

            + torch.sum(model.features[3].block[0][0].weight == 0)
            + torch.sum(model.features[3].block[1][0].weight == 0)
            + torch.sum(model.features[3].block[2][0].weight == 0)

            + torch.sum(model.features[4].block[0][0].weight == 0)
            + torch.sum(model.features[4].block[1][0].weight == 0)
            + torch.sum(model.features[4].block[2][0].weight == 0)
            + torch.sum(model.features[4].block[3][0].weight == 0)

            + torch.sum(model.features[5].block[0][0].weight == 0)
            + torch.sum(model.features[5].block[0][0].weight == 0)
            + torch.sum(model.features[6].block[0][0].weight == 0)
            + torch.sum(model.features[6].block[0][0].weight == 0)
            + torch.sum(model.features[7].block[0][0].weight == 0)
            + torch.sum(model.features[7].block[0][0].weight == 0)
            + torch.sum(model.features[8].block[0][0].weight == 0)
            + torch.sum(model.features[8].block[0][0].weight == 0)
            + torch.sum(model.features[9].block[0][0].weight == 0)
            + torch.sum(model.features[9].block[0][0].weight == 0)
            + torch.sum(model.features[10].block[0][0].weight == 0)
            + torch.sum(model.features[10].block[0][0].weight == 0)
            + torch.sum(model.features[11].block[0][0].weight == 0)
            + torch.sum(model.features[11].block[0][0].weight == 0)
            + torch.sum(model.features[12].block[0][0].weight == 0)
            + torch.sum(model.features[12].block[0][0].weight == 0)
            + torch.sum(model.features[13].block[0][0].weight == 0)
            + torch.sum(model.features[13].block[0][0].weight == 0)
            + torch.sum(model.features[14].block[0][0].weight == 0)
            + torch.sum(model.features[14].block[0][0].weight == 0)
            + torch.sum(model.features[15].block[0][0].weight == 0)
            + torch.sum(model.features[15].block[0][0].weight == 0)
            + torch.sum(model.features[16][0].weight == 0)
        )
        / float(
            model.features[0][0].weight.nelement()

            + model.features[1].block[0][0].weight.nelement()
            + model.features[1].block[1][0].weight.nelement()

            + model.features[2].block[0][0].weight.nelement()
            + model.features[2].block[1][0].weight.nelement()
            + model.features[2].block[2][0].weight.nelement()

            + model.features[3].block[0][0].weight.nelement()
            + model.features[3].block[1][0].weight.nelement()
            + model.features[3].block[2][0].weight.nelement()

            + model.features[4].block[0][0].weight.nelement()
            + model.features[4].block[1][0].weight.nelement()
            + model.features[4].block[2][0].weight.nelement()
            + model.features[4].block[3][0].weight.nelement()

            + model.features[5].block[0][0].weight.nelement()
            + model.features[5].block[0][0].weight.nelement()
            + model.features[6].block[0][0].weight.nelement()
            + model.features[6].block[0][0].weight.nelement()
            + model.features[7].block[0][0].weight.nelement()
            + model.features[7].block[0][0].weight.nelement()
            + model.features[8].block[0][0].weight.nelement()
            + model.features[8].block[0][0].weight.nelement()
            + model.features[9].block[0][0].weight.nelement()
            + model.features[9].block[0][0].weight.nelement()
            + model.features[10].block[0][0].weight.nelement()
            + model.features[10].block[0][0].weight.nelement()
            + model.features[11].block[0][0].weight.nelement()
            + model.features[11].block[0][0].weight.nelement()
            + model.features[12].block[0][0].weight.nelement()
            + model.features[12].block[0][0].weight.nelement()
            + model.features[13].block[0][0].weight.nelement()
            + model.features[13].block[0][0].weight.nelement()
            + model.features[14].block[0][0].weight.nelement()
            + model.features[14].block[0][0].weight.nelement()
            + model.features[15].block[0][0].weight.nelement()
            + model.features[15].block[0][0].weight.nelement()
            + model.features[16][0].weight.nelement()
        )
    )
)